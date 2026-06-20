import asyncio
import json
import os
import queue
import threading
import time
from typing import List

import numpy as np
import torch
import zmq
import zmq.asyncio

from ..requests import Request
from ..sampling import SamplingConfig
from ..utils import get_logger
from ..worker import CudaGraphWorker, DeferredStopHandle, ModelWorker


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
        # Deferred-EOS handle carried across steps (sync _step_pipelined and async _step_async).
        self._pending_stop_handle = None
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
            # DEFERRED-COLLECT: the queued CollectedAudio holds a numpy view into a pinned-ring
            # slot whose copy may still be in flight; the slot is reused after ring_depth collects.
            # Keep maxsize small and coupled to the ring depth so a slot is never overwritten while
            # a still-queued item references it (worst-case live views = maxsize + 1 emitting +
            # 1 just-produced). With the scheduler no longer synchronizing each detok step, this
            # bounded queue is also what throttles run-ahead (the role the per-step vocoder sync
            # used to play). maxsize 4 against ring depth 6 satisfies maxsize + 2 <= ring_depth.
            emit_queue_maxsize = 4
            ring_depth = getattr(self.model_worker, "_audio_pinned_ring_depth", emit_queue_maxsize + 2)
            assert emit_queue_maxsize + 2 <= ring_depth, (
                f"emit queue maxsize {emit_queue_maxsize} + 2 must be <= pinned ring depth {ring_depth} "
                "to prevent pinned-slot reuse while a queued CollectedAudio still references it"
            )
            self._emit_queue: "queue.Queue" = queue.Queue(maxsize=emit_queue_maxsize)
            self._emit_thread = threading.Thread(target=self._emit_worker_loop, name="vox-emit", daemon=True)
            self._emit_thread.start()

        self.available_batch_sizes = self.model_worker.available_batch_sizes

        # Carried-batch pipeline (deferred-EOS Phase 3a): when enabled AND the model uses the
        # deferred-EOS path, run_forever uses _step_carried (which enqueues the current decode
        # before selecting + preparing the NEXT batch, so the host work overlaps the in-flight
        # decode) instead of the serial _step_pipelined. Requires the deferred path (handles, not
        # coroutines), i.e. VOX_DEFER_EOS=1, and sync mode. VOX_CARRIED_SYNC=0 forces the serial
        # path for A/B bisection.
        model_defers_eos = getattr(getattr(self.model_worker, "model", None), "_defer_eos", False)
        self._carried_sync = (
            os.environ.get("VOX_CARRIED_SYNC", "1") != "0"
            and not self.async_scheduling
            and model_defers_eos
        )
        # Detokenize frontier safety margin (see _select_detokenize_requests). Only the carried
        # pipeline needs it (=1, the pipeline depth); the serial path keeps 0 to stay byte-identical
        # with the pre-Phase-3a baseline.
        self._overgen_lag = 1 if self._carried_sync else 0
        # Sleep on fully-idle carried steps (no decode, no detok, nothing queued) to avoid a busy
        # spin / unbounded transient allocation when there is genuinely no work.
        self._idle_sleep_s = 0.0005
        if self._carried_sync:
            self.logger.info("Using carried-batch pipelined sync step (_step_carried)")

        # Audio parameters for duration calculation
        # Assuming 24kHz mono 16-bit audio
        self.sample_rate = 24000
        self.bytes_per_sample = 2  # 16-bit = 2 bytes
        self.channels = 1  # mono

    def _step_pipelined(self):
        """
        Process the next batch of requests (deferred-EOS pipeline variant).

        Renamed from _step. EOS is no longer applied inline via asyncio.run(task) at the end of the
        step (which paid the full ~5.9 ms torch.nonzero device sync). Instead the GPU stop-mask
        computed in the PREVIOUS step is applied here at the top via apply_deferred_stops (its CUDA
        event is already complete -> ~us), before request selection; the current step's decode
        returns a DeferredStopHandle carried to the next step. NOTE: this is the light-touch rename
        + async-EOS wiring; a later refactor will adopt the carried-batch (run_model/run_scheduling)
        shape to fully pipeline the sync loop.
        """
        # NVTX (CPU-side): label every host phase of the step so the previously-dark gaps
        # between GPU kernels are attributable in nsys. host_step_begin marks the step boundary
        # (read step cadence from the spacing between these marks); each host_* range below shows
        # which host phase owns the GPU-idle time (typically host_prepare_lm_inputs and
        # host_run_sampling_task). These are pure CPU markers and add no cuda sync.
        w = self.model_worker
        w.nvtx_mark("host_step_begin")

        # Apply the previous step's deferred EOS stop before any selection sees the requests. The
        # CUDA event recorded when the mask was copied is already complete a step later, so this is
        # a ~us wait, not the ~5.9 ms torch.nonzero device sync it replaces.
        with w.nvtx_range("host_apply_deferred_stops"):
            self.model_worker.apply_deferred_stops(self._pending_stop_handle)
            self._pending_stop_handle = None

        # insert/remove requests to self.active_requests
        with w.nvtx_range("host_prepare_requests"):
            self._prepare_requests()

        # Select requests for detokenization
        with w.nvtx_range("host_select_detokenize"):
            detokenize_requests = self._select_detokenize_requests()

        # Select requests for LM processing
        with w.nvtx_range("host_select_lm"):
            lm_requests = self._select_lm_requests()

        # Prepare LM inputs outside the worker and run either prefill or decode
        with w.nvtx_range(f"host_prepare_lm_inputs_bs{len(lm_requests)}"):
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
            # NVTX: this IS a blocking sync, so the host-idle time it owns is genuine GPU-bound
            # waiting (vs the host_* ranges, which are CPU-bound work). Labeling it separates them.
            with w.nvtx_range("host_decode_only_sync"):
                torch.cuda.synchronize()

        # Free KV pages for finished requests synchronously (cheap: returns page indices to a
        # pool). Kept on the scheduler thread so reclamation is never deferred behind the emitter
        # and the next step's empty_pages.get_nowait() can reuse them immediately. The completion
        # MESSAGE is still sent by the emitter (after the request's audio, preserving ordering).
        with w.nvtx_range("host_free_kv"):
            for req in detokenize_requests:
                if req.done_all:
                    self.model_worker.free_kv_cache(req)

        # 4. B3: hand byte-conversion + socket send to the emitter thread so the scheduler returns
        #    to the next decode immediately. Only enqueue when there is something to emit (audio,
        #    or a done_all completion message) to avoid flooding the queue on empty/idle steps.
        with w.nvtx_range("host_emit_enqueue"):
            if collected is not None or detokenize_requests:
                if self._b3_offload_emit:
                    self._emit_queue.put((collected, detokenize_requests))
                else:
                    # B3 OFF (VOX_B3_OFFLOAD=0): emit synchronously on the scheduler thread, same work
                    # as the emitter would do, but without the thread or the bounded queue. A/B baseline.
                    self._emit_collected(collected, detokenize_requests)
        # self._send_responses(detokenize_requests)  # B3: moved to the emitter thread (_emit_collected)

        # Carry the EOS result. Deferred path: run_lm_* returned a DeferredStopHandle -> stash it and
        # apply at the top of the NEXT step (no sync here). Legacy path (VOX_DEFER_EOS=0 or a
        # non-Orpheus model): a coroutine -> run it inline now, preserving the old behavior.
        if isinstance(task, DeferredStopHandle):
            self._pending_stop_handle = task
        elif task is not None:
            with w.nvtx_range("host_run_sampling_task"):
                asyncio.run(task)

    def _step_carried(self, handle, lm_requests, detokenize_requests, lm_inputs):
        """
        Carried-batch pipelined sync step (deferred-EOS Phase 3a).

        Unlike _step_pipelined (which applies the previous stop at the TOP and then selects + preps
        + decodes serially), this enqueues the CURRENT batch's decode FIRST, then — while that
        decode runs on the GPU — applies the previous step's stop, selects the NEXT batch, and runs
        prepare_lm_inputs (the prefill tokenizer + FlashInfer plan()) for it. Nothing synchronizes
        the default stream between the decode enqueue and the next-step host prep, so that host work
        overlaps the in-flight decode. State carried across run_forever iterations:
            handle              -> DeferredStopHandle from the PREVIOUS decode (applied this step)
            lm_requests         -> batch to decode THIS step (selected last step)
            detokenize_requests -> detok batch for THIS step (selected last step)
            lm_inputs           -> prepare_lm_inputs output for lm_requests (computed last step)
        Returns the same tuple shifted by one for the next iteration.

        Correctness rests on two companion changes: apply_deferred_stops truncates the EOS token AND
        any optimistic over-generation token (a stopped request gets one extra decode here before
        its stop is applied), and _select_detokenize_requests holds the streaming frontier back by
        OVERGEN_LAG so an unconfirmed frontier token is never detokenized before truncation.
        """
        w = self.model_worker
        w.nvtx_mark("host_step_begin")

        # --- GPU launch region for the carried batch K (no default-stream sync inside) ---
        # 1. Launch the SNAC vocoder for this step's detok batch (non-blocking; None if nothing).
        detokenize_ctx = w.run_detokenize_launch(detokenize_requests)

        # 2. Enqueue prefill/decode for the carried batch on the default stream -> overlaps the
        #    vocoder AND the host prep below. Returns a DeferredStopHandle (deferred path) or None.
        if lm_inputs is not None and lm_inputs["is_prefill"]:
            new_handle = w.run_lm_prefill(lm_requests, lm_inputs)
        else:
            new_handle = w.run_lm_decode(lm_requests, lm_inputs)
        assert not asyncio.iscoroutine(new_handle), (
            "carried-batch sync requires the deferred-EOS path (VOX_DEFER_EOS=1); got a coroutine"
        )

        # 3. Collect vocoder audio. Blocks ONLY on the vocoder completion event, so the decode from
        #    step 2 keeps running on the default stream while the CPU proceeds below.
        collected = w.run_detokenize_collect(detokenize_ctx)

        # --- CPU region overlapping decode(K) ---
        # 4. Apply the PREVIOUS step's deferred EOS stop. Its event was recorded a full step ago, so
        #    synchronize() is ~us and does NOT wait on decode(K). Truncates EOS + over-gen tokens
        #    before any detok selection below can see them.
        with w.nvtx_range("host_apply_deferred_stops"):
            w.apply_deferred_stops(handle)

        # 5. Intake new requests + drop finished (done_all) ones. No GPU dependency.
        with w.nvtx_range("host_prepare_requests"):
            self._prepare_requests()

        # 6. Free KV pages for requests that finished detokenizing this step (done_all set during
        #    run_detokenize_collect above or by a prior _select_detokenize_requests).
        with w.nvtx_range("host_free_kv"):
            for req in detokenize_requests:
                if req.done_all:
                    w.free_kv_cache(req)

        # 7. Hand byte-conversion + socket send to the emitter thread (B3), or emit inline if off.
        with w.nvtx_range("host_emit_enqueue"):
            if collected is not None or detokenize_requests:
                if self._b3_offload_emit:
                    self._emit_queue.put((collected, detokenize_requests))
                else:
                    self._emit_collected(collected, detokenize_requests)

        # 8-9. Select the NEXT step's batches. This runs BEFORE decode(K)'s stops are known (they
        #      are in new_handle, applied next step) -> a stopped request may be optimistically
        #      selected for one extra decode; it is truncated next step and its unconfirmed frontier
        #      token is held out of the audio by the OVERGEN_LAG margin until then.
        with w.nvtx_range("host_select_detokenize"):
            next_detokenize_requests = self._select_detokenize_requests()
        with w.nvtx_range("host_select_lm"):
            next_lm_requests = self._select_lm_requests()

        # 10. Prepare LM inputs for the NEXT batch (prefill tokenizer is pure CPU; plan() H2D is
        #     enqueued BEHIND decode(K)) -> this is the host work that now overlaps decode(K).
        with w.nvtx_range(f"host_prepare_lm_inputs_bs{len(next_lm_requests)}"):
            next_lm_inputs = w.prepare_lm_inputs(next_lm_requests, next_detokenize_requests)

        # 11. Bound run-ahead. The decode-only torch.cuda.synchronize() is intentionally gone:
        #       - steady decode -> new_handle's event is synchronized next step in
        #         apply_deferred_stops, bounding run-ahead to ~1 step.
        #       - detok-only step -> the vocoder collect event above already bounded it.
        #       - fully idle (no decode, no detok, and nothing queued for next step) -> brief sleep
        #         so the loop does not busy-spin / accumulate transient allocations.
        if (
            new_handle is None
            and detokenize_ctx is None
            and not next_lm_requests
            and not next_detokenize_requests
        ):
            time.sleep(self._idle_sleep_s)

        return new_handle, next_lm_requests, next_detokenize_requests, next_lm_inputs

    async def _step_async(self, task, lm_requests, detokenize_requests):
        """
        Process the next batch of requests asynchronously.
        """

        # NVTX (CPU-side) parity with sync _step: label host phases so async-mode nsys captures
        # show the same host_* breakdown. Here the scheduling host work (select_*) is expected to
        # overlap the model coroutine via asyncio.gather below, so these ranges let you confirm
        # the overlap actually happens (host_select_* should sit under run_model's GPU work).
        w = self.model_worker
        w.nvtx_mark("host_step_begin")

        # insert/remove requests to self.active_requests
        with w.nvtx_range("host_prepare_requests"):
            await self._prepare_requests_async()

        # Prepare LM inputs outside the worker and run either prefill or decode
        with w.nvtx_range(f"host_prepare_lm_inputs_bs{len(lm_requests)}"):
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
            #    Deferred-EOS: run_lm_* now returns a DeferredStopHandle (or a legacy coroutine /
            #    None); it is applied/awaited next step in run_scheduling, not wrapped in a task.
            if lm_inputs is not None and lm_inputs["is_prefill"]:
                next_task = self.model_worker.run_lm_prefill(lm_requests, lm_inputs)
            else:
                next_task = self.model_worker.run_lm_decode(lm_requests, lm_inputs)

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
                # DEFERRED-COLLECT: async has no emitter thread, so the copy_event must be awaited
                # on this (event-loop) thread before reading audio_np. This keeps async correct but
                # unoptimized — the overlap win is sync-only for now (async is not the benchmarked
                # path). No-op when copy_event is None (synchronous fallback).
                if collected.copy_event is not None:
                    collected.copy_event.synchronize()
                for req, i, trim_len in collected.chunks:
                    req.output_audio.put(self._chunk_to_bytes(collected.audio_np[i], trim_len))

            # 4. Return results to clients (after collect, which produced this step's audio).
            await self._send_responses_async(detokenize_requests)

            return next_task

        async def run_scheduling():
            # Apply the previous step's deferred EOS (or run a legacy sampling coroutine) before
            # selecting the next batch. NOTE: async mode is wired for consistency but not yet
            # validated/optimized in this pass.
            if isinstance(task, DeferredStopHandle):
                with w.nvtx_range("host_apply_deferred_stops"):
                    self.model_worker.apply_deferred_stops(task)
            elif task is not None:
                with w.nvtx_range("host_await_sampling_task"):
                    await task

            # Select requests for detokenization
            with w.nvtx_range("host_select_detokenize"):
                next_detokenize_requests = self._select_detokenize_requests()

            # Select requests for LM processing
            with w.nvtx_range("host_select_lm"):
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
        elif self._carried_sync:
            # Carried-batch pipeline (Phase 3a): carry (handle, lm_requests, detokenize_requests,
            # lm_inputs) across iterations so each step enqueues the current decode before doing the
            # next step's host prep, overlapping it with the in-flight decode. Run-ahead is bounded
            # inside _step_carried (next-step apply_deferred_stops event sync / vocoder collect /
            # idle sleep), so no per-iteration device sync here.
            handle, lm_requests, detokenize_requests, lm_inputs = None, [], [], None
            while True:
                handle, lm_requests, detokenize_requests, lm_inputs = self._step_carried(
                    handle, lm_requests, detokenize_requests, lm_inputs
                )
        else:
            while True:
                self._step_pipelined()
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

        # Carried-batch pipeline safety margin: in the pipelined sync loop (_step_carried) a request's
        # EOS is detected one step late, and a stopped request may decode one extra optimistic token
        # before its stop is applied/truncated. Holding the streaming (not-done) detokenize frontier
        # back by OVERGEN_LAG tokens guarantees we never decode an unconfirmed frontier token (a
        # possible EOS / over-gen token) into audio before apply_deferred_stops truncates it; those
        # trailing tokens are only ever handled by the final-flush (done_lm_generation) branch below,
        # which runs after truncation. =0 in the non-pipelined path is also correct, but =1 is
        # harmless there (it just defers a chunk by <1 SNAC frame). detokenize_overlap does NOT
        # provide this margin (it shifts where chunks start, not how close to the frontier they end).
        overgen_lag = self._overgen_lag

        for req in self.active_requests:
            if len(detokenize_requests) >= self.max_batch_size:
                break

            # req.next_audio_decode_idx[-1] is the last decode index
            next_decode_idx = req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            if req.done_lm_generation:
                # Only schedule if there are tokens left to decode (final flush, no margin: the stop
                # is already applied + truncated, so the frontier is confirmed).
                if next_decode_idx < len(req.lm_output_audio_tokens):
                    req.next_audio_decode_idx = [next_decode_idx]
                    detokenize_requests.append(req)
                else:
                    # All tokens have been decoded but done_all wasn't set
                    # (can happen when done_lm_generation is set after the last detokenize)
                    # Add to detokenize_requests so _send_responses can send completion
                    req.done_all = True
                    detokenize_requests.append(req)
            elif next_decode_idx + detokenize_interval <= len(req.lm_output_audio_tokens) - overgen_lag:
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
                # DEFERRED-COLLECT: the GPU->host copy was issued without a CPU wait on the
                # scheduler thread; block on its completion HERE (emitter thread) before reading
                # audio_np's values, so the vocoder wait overlaps the scheduler's next decode
                # instead of stalling it. detok_finalize_wait should sit under the decode bar in
                # nsys. No-op when copy_event is None (synchronous fallback path).
                if collected.copy_event is not None:
                    with self.model_worker.nvtx_range("detok_finalize_wait"):
                        collected.copy_event.synchronize()
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

            cfg_alpha = new_request.model_kwargs.get("cfg_alpha")
            if cfg_alpha is not None:
                new_request.sampling_config = SamplingConfig(cfg_scale=float(cfg_alpha))

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

