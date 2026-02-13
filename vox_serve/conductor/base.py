"""
Conductor base class for distributed inference coordination.

The Conductor orchestrates request lifecycle and coordinates
strategy execution across one or more Pools.
"""

import asyncio
import json
import time
from typing import List, Optional

import torch
import zmq
import zmq.asyncio

from ..pool import CudaGraphPool, Pool
from ..requests import Request
from ..strategy import (
    AudioCodecStrategy,
    EncoderStrategy,
    LLMStrategy,
)
from ..utils import get_logger


class Conductor:
    """
    Coordinates distributed inference across multiple Pools.

    The Conductor is responsible for:
    - Managing request lifecycle (intake, processing, completion)
    - Selecting which requests to process in each step
    - Coordinating strategy execution on Pools
    - Communicating results back to clients via ZMQ
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cuda"),
        max_batch_size: int = 8,
        max_num_pages: int = 1024,
        page_size: int = 128,
        request_socket_path: str = "/tmp/vox_serve_request.ipc",
        result_socket_path: str = "/tmp/vox_serve_result.ipc",
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        repetition_penalty: float = None,
        repetition_window: int = None,
        cfg_scale: float = None,
        greedy: bool = False,
        enable_cuda_graph: bool = True,
        enable_disaggregation: bool = False,
        enable_nvtx: bool = False,
        enable_torch_compile: bool = False,
        async_scheduling: bool = False,
        dp_rank: int = 0,
        dp_size: int = 1,
    ):
        self.device = device
        self.max_batch_size = max_batch_size
        self.async_scheduling = async_scheduling
        self.dp_rank = dp_rank
        self.dp_size = dp_size

        # Create logger with rank prefix for data parallel mode
        base_logger = get_logger(__name__)
        if dp_size > 1:
            import logging

            self.logger = logging.LoggerAdapter(base_logger, {"dp_rank": dp_rank})
            self.logger.process = lambda msg, kwargs: (f"[DP {dp_rank}/{dp_size}] {msg}", kwargs)
        else:
            self.logger = base_logger

        self.logger.info(f"Using {'async' if async_scheduling else 'sync'} scheduling mode")

        # Create Pool
        pool_kwargs = {
            "model_name": model_name_or_path,
            "device": str(device),
            "max_batch_size": max_batch_size,
            "max_num_pages": max_num_pages,
            "page_size": page_size,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "repetition_window": repetition_window,
            "cfg_scale": cfg_scale,
            "greedy": greedy,
            "enable_nvtx": enable_nvtx,
            "enable_torch_compile": enable_torch_compile,
            "dp_rank": dp_rank,
            "dp_size": dp_size,
        }

        if enable_disaggregation:
            pool_kwargs["secondary_device"] = "cuda:1"

        # Choose pool type based on configuration
        if enable_cuda_graph:
            opt_text = " with disaggregation" if enable_disaggregation else " with CUDA graphs"
            self.logger.info(f"Using CudaGraphPool{opt_text}")
            self.pool: Pool = CudaGraphPool(**pool_kwargs)
        else:
            opt_text = " with disaggregation" if enable_disaggregation else ""
            self.logger.info(f"Using Pool{opt_text}")
            self.pool = Pool(**pool_kwargs)

        # Create and register strategies
        self._init_strategies()

        # Initialize CUDA graphs if using CudaGraphPool
        if enable_cuda_graph and isinstance(self.pool, CudaGraphPool):
            self.pool.initialize_cuda_graphs(self.llm_strategy)

        self.active_requests: List[Request] = []

        # Initialize ZMQ contexts based on scheduling mode
        if self.async_scheduling:
            self.context = zmq.asyncio.Context()
            self.request_socket = self.context.socket(zmq.PULL)
            self.request_socket.bind(f"ipc://{request_socket_path}")
            self.result_socket = self.context.socket(zmq.PUSH)
            self.result_socket.connect(f"ipc://{result_socket_path}")
        else:
            self.context = zmq.Context()
            self.request_socket = self.context.socket(zmq.PULL)
            self.request_socket.bind(f"ipc://{request_socket_path}")
            self.result_socket = self.context.socket(zmq.PUSH)
            self.result_socket.connect(f"ipc://{result_socket_path}")

        # Set socket HWMs to reduce blocking under bursty load
        try:
            self.request_socket.setsockopt(zmq.RCVHWM, 256)
            self.result_socket.setsockopt(zmq.SNDHWM, 1024)
            self.request_socket.setsockopt(zmq.LINGER, 0)
            self.result_socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass

        self.available_batch_sizes = self.pool.available_batch_sizes

        # Audio parameters for duration calculation
        self.sample_rate = 24000
        self.bytes_per_sample = 2  # 16-bit
        self.channels = 1  # mono

    def _init_strategies(self):
        """Initialize and register strategies with the pool."""
        model = self.pool.model

        # Encoder strategy (for preprocessing)
        self.encoder_strategy = EncoderStrategy(model)

        # LLM strategy (for generation)
        self.llm_strategy = LLMStrategy(model)
        self.pool.register_strategy(self.llm_strategy)

        # Audio codec strategy (for detokenization)
        self.audio_codec_strategy = AudioCodecStrategy(model)
        self.pool.register_strategy(self.audio_codec_strategy)

        self.logger.info("Strategies initialized: encoder, llm, audio_codec")

    def _step(self):
        """
        Process the next batch of requests (sync version).
        """
        # Intake new requests
        self._prepare_requests()

        # Select requests for detokenization
        detokenize_requests = self._select_detokenize_requests()

        # Select requests for LM processing
        lm_requests = self._select_lm_requests()

        # Prepare LM inputs
        lm_inputs = self._prepare_lm_inputs(lm_requests, detokenize_requests)

        # Run detokenization
        self._run_detokenize(detokenize_requests)

        # Send responses
        self._send_responses(detokenize_requests)

        # Run LM step
        if lm_inputs is not None:
            if lm_inputs["is_prefill"]:
                task = self._run_lm_prefill(lm_requests, lm_inputs)
            else:
                task = self._run_lm_decode(lm_requests, lm_inputs)

            if task is not None:
                asyncio.run(task)

    async def _step_async(self, task, lm_requests, detokenize_requests):
        """
        Process the next batch of requests asynchronously.
        """
        await self._prepare_requests_async()

        lm_inputs = self._prepare_lm_inputs(lm_requests, detokenize_requests)

        async def run_model():
            self._run_detokenize(detokenize_requests)
            await self._send_responses_async(detokenize_requests)

            if lm_inputs is not None and lm_inputs["is_prefill"]:
                coro = self._run_lm_prefill(lm_requests, lm_inputs)
            else:
                coro = self._run_lm_decode(lm_requests, lm_inputs)

            next_task = asyncio.create_task(coro) if coro else None
            return next_task

        async def run_scheduling():
            if task is not None:
                await task

            next_detokenize_requests = self._select_detokenize_requests()
            next_lm_requests = self._select_lm_requests()
            return next_lm_requests, next_detokenize_requests

        model_result, scheduling_result = await asyncio.gather(run_model(), run_scheduling())

        next_task = model_result
        next_lm_requests, next_detokenize_requests = scheduling_result

        return next_task, next_lm_requests, next_detokenize_requests

    async def _run_async_loop(self):
        task, lm_requests, detokenize_requests = None, [], []
        while True:
            task, lm_requests, detokenize_requests = await self._step_async(
                task, lm_requests, detokenize_requests
            )
            await asyncio.sleep(0)

    def run_forever(self):
        """Run the conductor indefinitely."""
        if self.async_scheduling:
            asyncio.run(self._run_async_loop())
        else:
            while True:
                self._step()
                torch.cuda.synchronize()

    def _prepare_lm_inputs(self, lm_requests, detokenize_requests):
        """Prepare inputs for LM processing."""
        # Copy decode indices before processing
        for req in detokenize_requests:
            req.audio_decode_idx = req.next_audio_decode_idx.copy()

        if len(lm_requests) == 0:
            return None

        # Determine phase
        is_prefill = any(not req.done_lm_prefill for req in lm_requests)
        phase = "prefill" if is_prefill else "decode"

        # Get resources
        resources = self.pool.get_resources(self.llm_strategy.name)

        # Prepare inputs using strategy
        prepared = self.llm_strategy.prepare_inputs(lm_requests, resources, phase)
        if prepared is not None:
            prepared["is_prefill"] = is_prefill

        return prepared

    def _run_lm_prefill(self, requests, lm_inputs):
        """Run LM prefill phase."""
        resources = self.pool.get_resources(self.llm_strategy.name)

        async def run():
            return await self.llm_strategy.execute(
                requests, resources, "prefill", lm_inputs
            )

        return run()

    def _run_lm_decode(self, requests, lm_inputs):
        """Run LM decode phase."""
        if lm_inputs is None or len(requests) == 0:
            return None

        resources = self.pool.get_resources(self.llm_strategy.name)

        async def run():
            return await self.llm_strategy.execute(
                requests, resources, "decode", lm_inputs
            )

        return run()

    def _run_detokenize(self, requests):
        """Run detokenization."""
        if len(requests) == 0:
            return

        resources = self.pool.get_resources(self.audio_codec_strategy.name)

        async def run():
            outputs = await self.audio_codec_strategy.execute(
                requests, resources, "decode"
            )
            self.audio_codec_strategy.post_process(requests, outputs, resources)

        asyncio.run(run())

    def _select_lm_requests(self):
        """Select requests that need LM processing."""
        lm_requests = []

        # Get constraints
        if isinstance(self.pool, CudaGraphPool):
            max_prefill_batch_size = self.pool.prefill_graph_batch_size
            max_seq_len = max(self.pool.cuda_graph_seq_len_buckets)
        else:
            max_prefill_batch_size = self.max_batch_size
            max_seq_len = 1024

        # Separate prefill and decode requests
        prefill_requests = []
        decode_requests = []

        for req in self.active_requests:
            if req.done_lm_generation:
                continue

            if not req.done_lm_prefill:
                prefill_requests.append(req)
            else:
                decode_requests.append(req)

        # Allocate prefill requests with constraints
        if prefill_requests:
            current_batch_size = 0
            current_seq_len = 0

            for req in prefill_requests:
                req_seq_len = req.input_length if req.input_length else 0

                if (
                    current_batch_size + 1 <= max_prefill_batch_size
                    and current_seq_len + req_seq_len <= max_seq_len
                ):
                    lm_requests.append(req)
                    current_batch_size += 1
                    current_seq_len += req_seq_len

                    if current_batch_size >= max_prefill_batch_size:
                        break

                # Allow only one prefill request for now
                break

            remaining_slots = max_prefill_batch_size - len(lm_requests)
        else:
            remaining_slots = self.max_batch_size

        # Fill with decode requests
        for i in range(remaining_slots):
            if len(lm_requests) >= self.max_batch_size:
                break
            if i >= len(decode_requests):
                break
            lm_requests.append(decode_requests[i])

        return lm_requests

    def _select_detokenize_requests(self):
        """Select requests that need detokenization."""
        detokenize_requests = []

        detokenize_interval = self.pool.detokenize_interval
        detokenize_overlap = self.pool.detokenize_overlap
        step = detokenize_interval - detokenize_overlap

        for req in self.active_requests:
            if len(detokenize_requests) >= self.max_batch_size:
                break

            next_decode_idx = (
                req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            )

            if req.done_lm_generation:
                if next_decode_idx < len(req.lm_output_audio_tokens):
                    req.next_audio_decode_idx = [next_decode_idx]
                    detokenize_requests.append(req)
            elif next_decode_idx + detokenize_interval <= len(req.lm_output_audio_tokens):
                req.next_audio_decode_idx = [next_decode_idx]
                detokenize_requests.append(req)

        return detokenize_requests

    def _send_responses(self, detokenize_requests):
        """Send responses back to clients (sync version)."""
        for req in detokenize_requests:
            while not req.output_audio.empty():
                audio_chunk = req.output_audio.get()

                if req.is_streaming:
                    send_time = time.time()
                    duration = self._calculate_chunk_duration(audio_chunk)
                    req.chunk_send_timestamps.append(send_time)
                    req.chunk_durations.append(duration)

                message = req.request_id.encode("utf-8") + b"|AUDIO|" + audio_chunk
                self.result_socket.send(message)

            if req.done_all:
                self.pool.free_pages(req)
                completion_message = {
                    "status": "completed",
                    "reason": req.finish_reason or "unknown",
                }
                completion_payload = (
                    req.request_id.encode("utf-8")
                    + b"|COMPLETION|"
                    + json.dumps(completion_message).encode("utf-8")
                )
                self.logger.debug("Sending completion for request %s", req.request_id)
                self.result_socket.send(completion_payload)

    async def _send_responses_async(self, detokenize_requests):
        """Send responses back to clients (async version)."""
        for req in detokenize_requests:
            while not req.output_audio.empty():
                audio_chunk = req.output_audio.get()

                if req.is_streaming:
                    send_time = time.time()
                    duration = self._calculate_chunk_duration(audio_chunk)
                    req.chunk_send_timestamps.append(send_time)
                    req.chunk_durations.append(duration)

                message = req.request_id.encode("utf-8") + b"|AUDIO|" + audio_chunk
                await self.result_socket.send(message)

            if req.done_all:
                self.pool.free_pages(req)
                completion_message = {
                    "status": "completed",
                    "reason": req.finish_reason or "unknown",
                }
                completion_payload = (
                    req.request_id.encode("utf-8")
                    + b"|COMPLETION|"
                    + json.dumps(completion_message).encode("utf-8")
                )
                self.logger.debug("Sending completion for request %s", req.request_id)
                await self.result_socket.send(completion_payload)

    def _calculate_chunk_duration(self, audio_chunk: bytes) -> float:
        """Calculate the duration of an audio chunk in seconds."""
        num_samples = len(audio_chunk) // (self.channels * self.bytes_per_sample)
        return num_samples / self.sample_rate

    def _handle_request_payload(self, message_payload) -> Optional[Request]:
        """Handle request payload parsing and create Request object."""
        delimiter_pos = message_payload.find(b"|")
        if delimiter_pos != -1:
            json_data = message_payload[:delimiter_pos].decode("utf-8")
            request_dict = json.loads(json_data)

            new_request = Request(
                request_id=request_dict["request_id"],
                prompt=request_dict["prompt"],
                audio_path=(
                    request_dict.get("audio_path")
                    if self.pool.supports_audio_input
                    else None
                ),
                is_streaming=request_dict.get("is_streaming", False),
                is_pressing=request_dict.get("is_streaming", False),
            )

            self.logger.debug("new_request=%s", new_request)
            return new_request
        else:
            self.logger.warning(f"Received malformed message: {message_payload[:50]}...")
            return None

    def _prepare_requests(self):
        """Prepare requests for processing (sync version)."""
        while True:
            try:
                message_payload = self.request_socket.recv(flags=zmq.NOBLOCK)
                new_request = self._handle_request_payload(message_payload)
                if new_request:
                    self.active_requests.append(new_request)
            except zmq.Again:
                break
            except Exception as e:
                self.logger.error(f"Error receiving requests: {str(e)}")
                import traceback

                self.logger.error(f"Traceback: {traceback.format_exc()}")

        self.active_requests = [req for req in self.active_requests if not req.done_all]

    async def _prepare_requests_async(self):
        """Prepare requests for processing (async version)."""
        while True:
            try:
                message_payload = await self.request_socket.recv(flags=zmq.DONTWAIT)
                new_request = self._handle_request_payload(message_payload)
                if new_request:
                    self.active_requests.append(new_request)
            except zmq.Again:
                break
            except Exception as e:
                self.logger.error(f"Error receiving requests: {str(e)}")
                import traceback

                self.logger.error(f"Traceback: {traceback.format_exc()}")

        self.active_requests = [req for req in self.active_requests if not req.done_all]
