import asyncio
import json
import queue
import threading
import time
from typing import List

import numpy as np
import torch
import zmq
import zmq.asyncio

from ..requests import Request
from ..utils import get_logger
from ..worker import CudaGraphWorker, ModelWorker


class Scheduler:
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
        unroll_depth_cuda_graph: bool = False,
        async_scheduling: bool = False,
        dp_rank: int = 0,
        dp_size: int = 1,
        detokenize_interval: int = None,
    ):
        self.device = device
        self.max_batch_size = max_batch_size
        self.async_scheduling = async_scheduling
        self.dp_rank = dp_rank
        self.dp_size = dp_size

        # Create logger with rank prefix for data parallel mode
        base_logger = get_logger(__name__)
        if dp_size > 1:
            # Use LoggerAdapter to add rank prefix
            import logging
            self.logger = logging.LoggerAdapter(base_logger, {'dp_rank': dp_rank})
            # Override the process method to add rank prefix
            self.logger.process = lambda msg, kwargs: (f"[DP {dp_rank}/{dp_size}] {msg}", kwargs)
        else:
            self.logger = base_logger

        self.logger.info(f"Using {'async' if async_scheduling else 'sync'} scheduling mode")

        # Choose worker based on user configuration
        worker_kwargs = {
            "model_name": model_name_or_path,
            "max_batch_size": max_batch_size,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "repetition_window": repetition_window,
            "cfg_scale": cfg_scale,
            "greedy": greedy,
            "max_num_pages": max_num_pages,
            "page_size": page_size,
            "enable_nvtx": enable_nvtx,
            "enable_torch_compile": enable_torch_compile,
            "unroll_depth_cuda_graph": unroll_depth_cuda_graph,
            "dp_rank": dp_rank,
            "dp_size": dp_size,
            "detokenize_interval": detokenize_interval,
        }

        # Simplified worker selection logic
        if enable_disaggregation:
            worker_kwargs["detokenizer_device"] = "cuda:1"

        if enable_cuda_graph:
            opt_text = " with disaggregation optimization" if enable_disaggregation else " with CUDA graph optimization"
            self.logger.info(f"Using CudaGraphWorker{opt_text}")
            self.model_worker = CudaGraphWorker(**worker_kwargs)
        else:
            opt_text = (
                " with disaggregation optimization"
                if enable_disaggregation
                else " without CUDA graph optimization"
            )
            self.logger.info(f"Using ModelWorker{opt_text}")
            self.model_worker = ModelWorker(**worker_kwargs)

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
            # Fast shutdown behavior
            self.request_socket.setsockopt(zmq.LINGER, 0)
            self.result_socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass

        # B3: dedicated emitter thread for the SYNC scheduler. run_detokenize_collect snapshots the
        # audio to host on the scheduler thread; this thread does the per-chunk byte conversion and
        # the (ZMQ) socket send, so the scheduler loop returns to the next decode without blocking
        # on conversion/IO. Consequence: in sync mode the result_socket is used ONLY by this thread.
        # The bounded queue applies backpressure (and bounds memory / run-ahead) if the emitter
        # falls behind. The async path keeps its inline _send_responses_async (no emitter thread).
        # A/B TOGGLE: VOX_B3_OFFLOAD=0 disables the emitter thread so the byte-convert + socket send
        # runs INLINE on the scheduler thread (the pre-B3 behavior), isolating "off-thread + bounded
        # queue" as the only changed variable. Default (unset/"1") keeps B3 on. Used to test whether
        # B3 caused the 10rps streaming-viability regression (45.1% -> 16.8%).
        # self._b3_offload_emit = os.environ.get("VOX_B3_OFFLOAD", "1") != "0"
        self._b3_offload_emit = True
        if not self.async_scheduling and self._b3_offload_emit:
            self._emit_queue: "queue.Queue" = queue.Queue(maxsize=16)
            self._emit_thread = threading.Thread(target=self._emit_worker_loop, name="vox-emit", daemon=True)
            self._emit_thread.start()

        self.available_batch_sizes = self.model_worker.available_batch_sizes

        # Audio parameters for duration calculation
        # Assuming 24kHz mono 16-bit audio
        self.sample_rate = 24000
        self.bytes_per_sample = 2  # 16-bit = 2 bytes
        self.channels = 1  # mono

    def _step(self):
        """
        Process the next batch of requests.
        """

        # insert/remove requests to self.active_requests
        self._prepare_requests()

        # Select requests for detokenization
        detokenize_requests = self._select_detokenize_requests()

        # Select requests for LM processing
        lm_requests = self._select_lm_requests()

        # Prepare LM inputs outside the worker and run either prefill or decode
        lm_inputs = self.model_worker.prepare_lm_inputs(lm_requests, detokenize_requests)

        # OVERLAP OPT (Part 2): run the SNAC vocoder concurrently with the LLM decode.
        # The vocoder runs on a dedicated CUDA stream and the decode on the default stream,
        # so they execute in parallel on the GPU. The ordering below is what enables this:
        #   1. launch the vocoder (non-blocking) -> kernels in flight on vocoder_stream
        #   2. enqueue the decode on the default stream -> runs alongside the vocoder
        #   3. collect the vocoder output (CPU blocks only on the vocoder event, by which
        #      point the decode is already running)

        # 1. Launch the vocoder on vocoder_stream (returns immediately; None if nothing to do).
        detokenize_ctx = self.model_worker.run_detokenize_launch(detokenize_requests)

        # 2. Enqueue LLM prefill/decode on the default stream -> overlaps the vocoder.
        if lm_inputs is not None and lm_inputs["is_prefill"]:
            task = self.model_worker.run_lm_prefill(lm_requests, lm_inputs)
        else:
            task = self.model_worker.run_lm_decode(lm_requests, lm_inputs)

        # 3. Collect vocoder audio. Blocks only on the vocoder completion event, so the decode
        #    enqueued in step 2 keeps running on the default stream while the CPU waits. B3: this
        #    now only snapshots audio to host + reads request state; it returns a CollectedAudio.
        collected = self.model_worker.run_detokenize_collect(detokenize_ctx)

        if detokenize_ctx is None:
            # Decode-only step: there is no vocoder collect to provide backpressure, so sync the
            # default stream to stop the CPU running unboundedly ahead and accumulating transient
            # per-step tensors. There is nothing to overlap on a decode-only step, so this sync
            # costs no concurrency. (Replaces the blanket per-step sync removed from run_forever.)
            torch.cuda.synchronize()

        # Free KV pages for finished requests synchronously (cheap: returns page indices to a
        # pool). Kept on the scheduler thread so reclamation is never deferred behind the emitter
        # and the next step's empty_pages.get_nowait() can reuse them immediately. The completion
        # MESSAGE is still sent by the emitter (after the request's audio, preserving ordering).
        for req in detokenize_requests:
            if req.done_all:
                self.model_worker.free_kv_cache(req)

        # 4. B3: hand byte-conversion + socket send to the emitter thread so the scheduler returns
        #    to the next decode immediately. Only enqueue when there is something to emit (audio,
        #    or a done_all completion message) to avoid flooding the queue on empty/idle steps.
        if collected is not None or detokenize_requests:
            if self._b3_offload_emit:
                self._emit_queue.put((collected, detokenize_requests))
            else:
                # B3 OFF (VOX_B3_OFFLOAD=0): emit synchronously on the scheduler thread, same work as
                # the emitter would do, but without the thread or the bounded queue. A/B baseline.
                self._emit_collected(collected, detokenize_requests)
        # self._send_responses(detokenize_requests)  # B3: moved to the emitter thread (_emit_collected)

        # Execute the sampling task right away for synchronous scheduling
        if task is not None:
            asyncio.run(task)


    async def _step_async(self, task, lm_requests, detokenize_requests):
        """
        Process the next batch of requests asynchronously.
        """

        # insert/remove requests to self.active_requests
        await self._prepare_requests_async()

        # Prepare LM inputs outside the worker and run either prefill or decode
        lm_inputs = self.model_worker.prepare_lm_inputs(lm_requests, detokenize_requests)

        async def run_model():
            # OVERLAP OPT (Part 2, async parity): same launch -> decode -> collect ordering as
            # the sync _step, so the SNAC vocoder (on vocoder_stream) overlaps the LLM decode
            # (on the default stream) in async mode too, and both modes emit the same NVTX
            # markers (detokenize_launch_* / detokenize_collect) for apples-to-apples profiling.
            #
            # Old serialized ordering (vocoder ran fully, incl. a device sync, before decode):
            #   self.model_worker.run_detokenize(detokenize_requests)
            #   await self._send_responses_async(detokenize_requests)
            #   if lm_inputs is not None and lm_inputs["is_prefill"]:
            #       coro = self.model_worker.run_lm_prefill(lm_requests, lm_inputs)
            #   else:
            #       coro = self.model_worker.run_lm_decode(lm_requests, lm_inputs)
            #   next_task = asyncio.create_task(coro) if coro else None

            # 1. Launch the vocoder on vocoder_stream (non-blocking; None if nothing to do).
            detokenize_ctx = self.model_worker.run_detokenize_launch(detokenize_requests)

            # 2. Enqueue LLM prefill/decode on the default stream -> overlaps the vocoder.
            if lm_inputs is not None and lm_inputs["is_prefill"]:
                coro = self.model_worker.run_lm_prefill(lm_requests, lm_inputs)
            else:
                coro = self.model_worker.run_lm_decode(lm_requests, lm_inputs)
            next_task = asyncio.create_task(coro) if coro else None

            # 3. Collect vocoder audio. Blocks only on the vocoder completion event, so the
            #    decode enqueued in step 2 keeps running on the default stream meanwhile.
            #    (No ctx-None device sync here, unlike sync _step: in async the previous step's
            #    sampling task is awaited in run_scheduling and its torch.nonzero already bounds
            #    run-ahead, so a blocking sync on the event-loop thread would be wasteful.)
            collected = self.model_worker.run_detokenize_collect(detokenize_ctx)

            # B3 async parity: there is no emitter thread in async mode, so materialize the
            # collected audio into req.output_audio here, then stream it via _send_responses_async
            # exactly as before. The byte conversion is shared with the sync emitter (_chunk_to_bytes).
            if collected is not None:
                for req, i, trim_len in collected.chunks:
                    req.output_audio.put(self._chunk_to_bytes(collected.audio_np[i], trim_len))

            # 4. Return results to clients (after collect, which produced this step's audio).
            await self._send_responses_async(detokenize_requests)

            return next_task

        async def run_scheduling():
            if task is not None:
                await task

            # Select requests for detokenization
            next_detokenize_requests = self._select_detokenize_requests()

            # Select requests for LM processing
            next_lm_requests = self._select_lm_requests()

            return next_lm_requests, next_detokenize_requests

        model_result, scheduling_result = await asyncio.gather(
            run_model(),
            run_scheduling()
        )

        # Unpack the results from the completed tasks
        next_task = model_result
        next_lm_requests, next_detokenize_requests = scheduling_result

        return next_task, next_lm_requests, next_detokenize_requests

    async def _run_async_loop(self):
        task, lm_requests, detokenize_requests = None, [], []
        while True:
            task, lm_requests, detokenize_requests = await self._step_async(task, lm_requests, detokenize_requests)
            await asyncio.sleep(0)

    def run_forever(self):
        """
        Run the scheduler indefinitely.
        """
        if self.async_scheduling:
            asyncio.run(self._run_async_loop())
        else:
            while True:
                self._step()
                # OVERLAP OPT (Part 1): removed the blanket per-step device-wide sync.
                # It was the dominant serializer (forced CPU/GPU lock-step every step and
                # drained the vocoder stream, preventing decode/vocoder overlap). Ordering and
                # backpressure are now handled inside _step: the vocoder completion event in
                # run_detokenize_collect bounds run-ahead on detokenize steps, and a default-
                # stream sync covers decode-only steps.
                # torch.cuda.synchronize()

    def _select_lm_requests(self):
        """
        Select requests that need LM processing.
        For prefill requests, ensure batch size and sequence length don't exceed worker limitations.
        Allocate prefill requests first, then decode requests in remaining slots.
        """
        lm_requests = []

        # Get worker limitations
        if isinstance(self.model_worker, CudaGraphWorker):
            max_prefill_batch_size = self.model_worker.prefill_graph_batch_size
            max_seq_len = max(self.model_worker.cuda_graph_seq_len_buckets)
        else:
            # Fallback for non-CUDA graph worker
            max_prefill_batch_size = self.max_batch_size
            max_seq_len = 1024  # Default assumption

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

        # First, allocate prefill requests with constraints
        if prefill_requests:
            current_batch_size = 0
            current_seq_len = 0

            for req in prefill_requests:
                req_seq_len = req.input_length if req.input_length else 0

                # Check if adding this request would exceed constraints
                if (current_batch_size + 1 <= max_prefill_batch_size and
                    current_seq_len + req_seq_len <= max_seq_len):

                    lm_requests.append(req)
                    current_batch_size += 1
                    current_seq_len += req_seq_len

                    if current_batch_size >= max_prefill_batch_size:
                        break

                # allow only one prefill request for now
                break

            remaining_slots = max_prefill_batch_size - len(lm_requests)

        else:
            remaining_slots = self.max_batch_size

        for i in range(remaining_slots):
            if len(lm_requests) >= self.max_batch_size:
                break

            if i >= len(decode_requests):
                break

            lm_requests.append(decode_requests[i])

        return lm_requests

    def _select_detokenize_requests(self):
        """
        Select requests that need detokenization.
        """
        detokenize_requests = []

        detokenize_interval = self.model_worker.detokenize_interval
        detokenize_overlap = self.model_worker.detokenize_overlap
        step = detokenize_interval - detokenize_overlap

        for req in self.active_requests:
            if len(detokenize_requests) >= self.max_batch_size:
                break

            # req.next_audio_decode_idx[-1] is the last decode index
            next_decode_idx = req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            if req.done_lm_generation:
                # Only schedule if there are tokens left to decode
                if next_decode_idx < len(req.lm_output_audio_tokens):
                    req.next_audio_decode_idx = [next_decode_idx]
                    detokenize_requests.append(req)
                else:
                    # All tokens have been decoded but done_all wasn't set
                    # (can happen when done_lm_generation is set after the last detokenize)
                    # Add to detokenize_requests so _send_responses can send completion
                    req.done_all = True
                    detokenize_requests.append(req)
            elif next_decode_idx + detokenize_interval <= len(req.lm_output_audio_tokens):
                req.next_audio_decode_idx = [next_decode_idx]
                detokenize_requests.append(req)

        return detokenize_requests

    def _send_responses(self, detokenize_requests):
        """
        Send responses back to clients for detokenized requests (sync version).
        """
        for req in detokenize_requests:
            while not req.output_audio.empty():
                # Send audio chunk message: request_id|AUDIO|audio_data
                audio_chunk = req.output_audio.get()

                # Record timestamp and duration for streaming requests
                if req.is_streaming:
                    send_time = time.time()
                    duration = self._calculate_chunk_duration(audio_chunk)
                    req.chunk_send_timestamps.append(send_time)
                    req.chunk_durations.append(duration)

                message = req.request_id.encode("utf-8") + b"|AUDIO|" + audio_chunk
                self.result_socket.send(message)

            # send completion notification for finished requests
            if req.done_all:
                self.model_worker.free_kv_cache(req)
                completion_message = {"status": "completed", "reason": req.finish_reason or "unknown"}
                # Send completion message: request_id|COMPLETION|json_data
                completion_payload = (
                    req.request_id.encode("utf-8") + b"|COMPLETION|" + json.dumps(completion_message).encode("utf-8")
                )
                self.logger.debug("Sending completion for request %s", req.request_id)
                self.result_socket.send(completion_payload)

    async def _send_responses_async(self, detokenize_requests):
        """
        Send responses back to clients for detokenized requests (async version).
        """
        for req in detokenize_requests:
            while not req.output_audio.empty():
                # Send audio chunk message: request_id|AUDIO|audio_data
                audio_chunk = req.output_audio.get()

                # Record timestamp and duration for streaming requests
                if req.is_streaming:
                    send_time = time.time()
                    duration = self._calculate_chunk_duration(audio_chunk)
                    req.chunk_send_timestamps.append(send_time)
                    req.chunk_durations.append(duration)

                message = req.request_id.encode("utf-8") + b"|AUDIO|" + audio_chunk
                await self.result_socket.send(message)

            # send completion notification for finished requests
            if req.done_all:
                self.model_worker.free_kv_cache(req)
                completion_message = {"status": "completed", "reason": req.finish_reason or "unknown"}
                # Send completion message: request_id|COMPLETION|json_data
                completion_payload = (
                    req.request_id.encode("utf-8") + b"|COMPLETION|" + json.dumps(completion_message).encode("utf-8")
                )
                self.logger.debug("Sending completion for request %s", req.request_id)
                await self.result_socket.send(completion_payload)

    # ------------------------------------------------------------------
    # B3: off-thread audio emission (sync scheduler).
    #
    # run_detokenize_collect snapshots audio to an owned host array on the scheduler thread; the
    # emitter thread then byte-converts and sends it, so the scheduler returns to the next decode
    # without blocking on the (Python-bound) conversion loop + socket IO that A2 showed dominate
    # detokenize_collect. The result_socket is therefore used only by the emitter thread in sync
    # mode, and a single FIFO thread preserves per-request chunk ordering (audio before completion).
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_to_bytes(audio_np_row, trim_len):
        """Convert one host audio row [n_channels, samples] (float) to int16 PCM bytes, optionally
        trimming the padded tail (trim_len == -1 means no trim). Pure NumPy/CPU -> thread-safe."""
        audio_int16 = (audio_np_row * 32767).astype(np.int16)
        if trim_len >= 0:
            audio_int16 = audio_int16[:, :trim_len]
        return audio_int16.tobytes()

    def _emit_collected(self, collected, detokenize_requests):
        """Byte-convert + send audio and completion messages. Runs ONLY on the emitter thread."""
        # Non-sync NVTX (respects nvtx_enabled): labels the offloaded work as an "emit" range on
        # the emitter thread's timeline row, so it can be measured there and confirmed gone from
        # detokenize_collect on the scheduler thread. try/finally keeps the range balanced if a
        # send raises. NVTX ranges are per-thread, so this annotates the emitter thread only.
        self.model_worker.nvtx_range_push("emit")
        try:
            if collected is not None:
                for req, i, trim_len in collected.chunks:
                    audio_bytes = self._chunk_to_bytes(collected.audio_np[i], trim_len)
                    if req.is_streaming:
                        req.chunk_send_timestamps.append(time.time())
                        req.chunk_durations.append(self._calculate_chunk_duration(audio_bytes))
                    self.result_socket.send(req.request_id.encode("utf-8") + b"|AUDIO|" + audio_bytes)

            # Completion messages for finished requests (sent after their audio, by FIFO ordering).
            # NOTE: free_kv_cache is intentionally NOT done here — the scheduler thread frees KV
            # pages synchronously in _step so page reclamation is never deferred behind this thread
            # (which could starve the next step's empty_pages.get_nowait() under page pressure).
            for req in detokenize_requests:
                if req.done_all:
                    completion_message = {"status": "completed", "reason": req.finish_reason or "unknown"}
                    completion_payload = (
                        req.request_id.encode("utf-8")
                        + b"|COMPLETION|"
                        + json.dumps(completion_message).encode("utf-8")
                    )
                    self.logger.debug("Sending completion for request %s", req.request_id)
                    self.result_socket.send(completion_payload)
        finally:
            self.model_worker.nvtx_range_pop()  # emit

    def _emit_worker_loop(self):
        """Consume (collected, detokenize_requests) packets and emit them. None is the stop sentinel."""
        while True:
            item = self._emit_queue.get()
            if item is None:
                break
            collected, detokenize_requests = item
            try:
                self._emit_collected(collected, detokenize_requests)
            except Exception:
                self.logger.exception("emitter thread failed while sending responses")

    def _calculate_chunk_duration(self, audio_chunk: bytes) -> float:
        """
        Calculate the duration of an audio chunk in seconds.
        Assumes 24kHz mono 16-bit PCM audio.
        """
        num_samples = len(audio_chunk) // (self.channels * self.bytes_per_sample)
        duration_seconds = num_samples / self.sample_rate
        return duration_seconds

    def _handle_request_payload(self, message_payload):
        """
        Handle request payload parsing and create Request object.
        Returns a Request object if successful, None if malformed.
        """
        delimiter_pos = message_payload.find(b"|")
        if delimiter_pos != -1:
            # Parse JSON request data
            json_data = message_payload[:delimiter_pos].decode("utf-8")
            request_dict = json.loads(json_data)

            # Create Request object from deserialized data
            new_request = Request(
                request_id=request_dict["request_id"],
                prompt=request_dict["prompt"],
                audio_path=request_dict.get("audio_path") if self.model_worker.supports_audio_input else None,
                is_streaming=request_dict.get("is_streaming", False),
                is_pressing=request_dict.get("is_streaming", False), # at first, streaming requests are pressing
                model_kwargs=request_dict.get("model_kwargs", {}),
            )

            self.logger.debug("new_request=%s", new_request)
            return new_request
        else:
            self.logger.warning(f"Received malformed audio message: {message_payload[:50]}...")
            return None

    def _prepare_requests(self):
        """
        Prepare requests for processing (sync version).
        This method gathers new requests from clients.
        """

        # get new requests from ZMQ
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

        # Filter out completed requests
        self.active_requests = [req for req in self.active_requests if not req.done_all]

    async def _prepare_requests_async(self):
        """
        Prepare requests for processing (async version).
        This method gathers new requests from clients.
        """

        # get new requests from ZMQ
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

        # Filter out completed requests
        self.active_requests = [req for req in self.active_requests if not req.done_all]

