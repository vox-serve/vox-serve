import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Set

import torch
import zmq

from ..requests import Request
from .base import Scheduler


class DisaggregationScheduler(Scheduler):
    """
    Scheduler optimized for disaggregated inference with parallel LM and detokenization pipelines.

    Extends the base Scheduler but runs two independent async loops:
    - LM loop (GPU 0): Handles prefill and decode operations
    - Detokenizer loop (GPU 1): Handles audio detokenization

    The two loops communicate via async queues, enabling true parallelism where
    GPU 1 can detokenize previous batches while GPU 0 generates new tokens.
    """

    def __init__(self, **kwargs):
        # Check disaggregation availability (requires at least 2 GPUs)
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"DisaggregationScheduler requires at least 2 GPUs, "
                f"but only {torch.cuda.device_count()} GPU(s) available."
            )

        # Force CUDA graph and disaggregation mode
        kwargs["enable_cuda_graph"] = True
        kwargs["enable_disaggregation"] = True
        kwargs["async_scheduling"] = True  # Force async mode for parallel loops

        # Initialize parent scheduler
        super().__init__(**kwargs)

        # Async queues for inter-loop communication
        self.detokenize_queue = asyncio.Queue()  # LM -> Detokenizer: requests ready for detokenization
        self.requests_lock = asyncio.Lock()  # Protect shared request list

        # Track which requests are currently being detokenized to avoid duplicates
        self.detokenizing_request_ids: Set[str] = set()

        # Thread pool executor for running blocking GPU operations
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detokenize")

        self.logger.info(
            "DisaggregationScheduler initialized with parallel LM and detokenization loops"
        )

    def run_forever(self):
        """
        Override base run_forever to use parallel LM and detokenizer loops.
        """
        async def _run():
            # Start both loops concurrently
            await asyncio.gather(
                self._lm_loop(),
                self._detokenizer_loop(),
            )

        asyncio.run(_run())

    async def _lm_loop(self):
        """
        LM processing loop: handles prefill and decode on GPU 0.
        Sends requests with ready tokens to the detokenizer loop via queue.
        """
        self.logger.info("Starting LM loop on GPU 0")
        lm_task = None

        while True:
            # Prepare requests (inherited from base)
            await self._prepare_requests_async()

            # Wait for previous sampling task if any
            if lm_task is not None:
                await lm_task
                lm_task = None

            # Always queue requests that are ready for detokenization.
            await self._queue_detokenize_requests()

            # Select requests for LM processing (inherited from base)
            async with self.requests_lock:
                lm_requests = self._select_lm_requests()

            if not lm_requests:
                await asyncio.sleep(0.001)  # Small sleep if no work
                continue

            # Prepare LM inputs (no detokenize requests needed here)
            lm_inputs = self.model_worker.prepare_lm_inputs(lm_requests, [])

            # Run LM forward pass
            if lm_inputs is not None and lm_inputs["is_prefill"]:
                coro = self.model_worker.run_lm_prefill(lm_requests, lm_inputs)
            else:
                coro = self.model_worker.run_lm_decode(lm_requests, lm_inputs)

            # Create async task for sampling if returned
            if coro:
                lm_task = asyncio.create_task(coro)

            await asyncio.sleep(0)  # Yield control

    async def _detokenizer_loop(self):
        """
        Detokenizer processing loop: handles audio decoding on GPU 1.
        Pulls requests from the detokenize queue and processes them.
        """
        self.logger.info("Starting detokenizer loop on GPU 1")

        while True:
            # Get batch of requests ready for detokenization
            detokenize_requests = await self._get_detokenize_batch()

            if not detokenize_requests:
                await asyncio.sleep(0.001)  # Small sleep if no work
                continue

            # Mark requests as being processed
            for req in detokenize_requests:
                self.detokenizing_request_ids.add(req.request_id)
                # Copy next_audio_decode_idx to audio_decode_idx before detokenization
                # (this is done in prepare_lm_inputs for base scheduler, but we call run_detokenize directly)
                if req.next_audio_decode_idx:
                    req.audio_decode_idx = req.next_audio_decode_idx.copy()
                else:
                    self.logger.warning(
                        f"Request {req.request_id} has no next_audio_decode_idx, skipping detokenization"
                    )
                    continue

            # Run detokenization on GPU 1 in executor to avoid blocking the event loop
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.model_worker.run_detokenize,
                    detokenize_requests
                )
            except Exception as e:
                self.logger.error(f"Error in detokenization: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                continue

            # Send responses back to clients (inherited from base)
            try:
                await self._send_responses_async(detokenize_requests)
            except Exception as e:
                self.logger.error(f"Error sending responses: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")

            # Unmark requests
            for req in detokenize_requests:
                if req.request_id in self.detokenizing_request_ids:
                    self.detokenizing_request_ids.remove(req.request_id)

            await asyncio.sleep(0)  # Yield control

    async def _queue_detokenize_requests(self):
        """
        Find requests that have tokens ready for detokenization and queue them.
        """
        async with self.requests_lock:
            detokenize_interval = self.model_worker.detokenize_interval
            detokenize_overlap = self.model_worker.detokenize_overlap
            step = detokenize_interval - detokenize_overlap

            for req in self.active_requests:
                # Skip if already being processed
                if req.request_id in self.detokenizing_request_ids:
                    continue

                next_decode_idx = req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0

                should_queue = False
                if req.done_lm_generation:
                    # For finished requests, queue if there are tokens left to decode
                    if next_decode_idx < len(req.lm_output_audio_tokens):
                        should_queue = True
                    else:
                        # All tokens have been decoded but done_all wasn't set
                        # (can happen when done_lm_generation is set after the last detokenize)
                        # Queue it so _send_responses can send completion message
                        req.done_all = True
                        should_queue = True
                elif next_decode_idx + detokenize_interval <= len(req.lm_output_audio_tokens):
                    # For ongoing requests, queue if we have enough tokens
                    should_queue = True

                if should_queue:
                    req.next_audio_decode_idx = [next_decode_idx]
                    await self.detokenize_queue.put(req)
                    self.detokenizing_request_ids.add(req.request_id)

    async def _get_detokenize_batch(self) -> List[Request]:
        """
        Get a batch of requests from the detokenize queue.
        """
        detokenize_requests = []

        # Try to get up to max_batch_size requests without blocking
        for _ in range(self.max_batch_size):
            try:
                # Wait a tiny bit for first request if queue is empty
                if not detokenize_requests:
                    req = await asyncio.wait_for(self.detokenize_queue.get(), timeout=0.001)
                else:
                    req = self.detokenize_queue.get_nowait()

                detokenize_requests.append(req)
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break

        return detokenize_requests

    async def _send_responses_async(self, detokenize_requests: List[Request]):
        """
        Override to handle request removal with locking.
        """
        self.logger.debug(f"Sending responses for {len(detokenize_requests)} requests")
        for req in detokenize_requests:
            self.logger.debug(f"Sending responses for request {req.request_id}")
            self.logger.debug(f"Output done all: {req.done_all}")
            self.logger.debug(f"Output queue size: {req.output_audio.qsize()}")
            self.logger.debug(f"Output done lm generation: {req.done_lm_generation}")
            self.logger.debug(f"Output audio decode idx: {req.audio_decode_idx}")
            self.logger.debug(f"Output audio decode idx length: {len(req.audio_decode_idx)}")
            self.logger.debug(f"Output audio tokens length: {len(req.lm_output_audio_tokens)}")
            while not req.output_audio.empty():
                # Send audio chunk
                audio_chunk = req.output_audio.get()

                # Record timestamp and duration for streaming requests
                if req.is_streaming:
                    send_time = time.time()
                    duration = self._calculate_chunk_duration(audio_chunk)
                    req.chunk_send_timestamps.append(send_time)
                    req.chunk_durations.append(duration)

                message = req.request_id.encode("utf-8") + b"|AUDIO|" + audio_chunk
                try:
                    self.logger.debug(
                        f"Sending audio chunk for request {req.request_id}, size: {len(audio_chunk)} bytes"
                    )
                    await self.result_socket.send(message)
                    self.logger.debug(f"Sent audio chunk for request {req.request_id}, size: {len(audio_chunk)} bytes")
                except Exception as e:
                    self.logger.error(f"Error sending audio chunk for request {req.request_id}: {e}")
                    raise

            # Send completion notification for finished requests
            if req.done_all:
                # Need to acquire lock when modifying active_requests
                async with self.requests_lock:
                    self.model_worker.free_kv_cache(req)
                    if req in self.active_requests:
                        self.active_requests.remove(req)

                completion_message = {"status": "completed", "reason": req.finish_reason or "unknown"}
                completion_payload = (
                    req.request_id.encode("utf-8") + b"|COMPLETION|" + json.dumps(completion_message).encode("utf-8")
                )
                self.logger.debug("Sending completion for request %s", req.request_id)
                try:
                    self.logger.debug(
                        f"Sending completion for request {req.request_id}, size: {len(completion_payload)} bytes"
                    )
                    await self.result_socket.send(completion_payload)
                    self.logger.debug("Successfully sent completion for request %s", req.request_id)
                except Exception as e:
                    self.logger.error(f"Error sending completion for request {req.request_id}: {e}")
                    raise

    async def _prepare_requests_async(self):
        """
        Override to add requests with locking.
        """
        while True:
            try:
                message_payload = await self.request_socket.recv(flags=zmq.NOBLOCK)
                new_request = self._handle_request_payload(message_payload)
                if new_request:
                    async with self.requests_lock:
                        self.active_requests.append(new_request)
                        self.logger.debug(f"Added new request {new_request.request_id}")
            except zmq.Again:
                break
            except Exception as e:
                self.logger.error(f"Error receiving requests: {str(e)}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
