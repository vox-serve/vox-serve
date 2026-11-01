"""A CPU stand-in for ProgramWorker and a driver that runs the Python runtime's schedulers against it.

``FakeWorker`` implements the interface the schedulers call (``prepare_lm_inputs``, ``run_lm_prefill`` /
``run_lm_decode``, ``run_detokenize``, ``free_kv_cache`` and the properties they read) with the rules of
``vox_serve.worker.program_worker.ProgramWorker`` that scheduling depends on:

- one prefill per step, admitted only with a free LM slot and, for a stateful codec, a free codec state
  slot; decodes of prefilled requests fill the rest of the batch;
- the LM slot is held from prefill to the last token, the codec state slot until ``free_kv_cache``;
- a step's tokens are applied when its coroutine is collected (``--async-scheduling`` launches the next
  step first, and rows of requests that finished meanwhile are ignored);
- ``done_all`` is set by ``run_detokenize`` once the chunk covering the last frame has been decoded.

Each request generates ``model_kwargs["frames"]`` audio frames and then stops. Its prompt fails, as a
program's ``preprocess`` rejects bad input, with ``model_kwargs["fail"]``, with audio input or with a
voice other than "tara" (``prompt_error``). A decoded chunk emits ``samples_per_frame`` int16 samples per
frame, each equal to the frame index, so a client can check that it received every frame once and in order.
"""

import asyncio
import json
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import zmq


class FakeWorker:
    def __init__(
        self,
        max_batch_size: int = 4,
        detok_slots: Optional[int] = None,
        interval: int = 4,
        overlap: int = 0,
        samples_per_frame: int = 24,
    ):
        self.max_batch_size = max_batch_size
        self.free_slots = list(range(max_batch_size))
        self.n_detok_slots = detok_slots
        self.free_detok_slots = list(range(detok_slots)) if detok_slots else None
        self._interval = interval
        self._overlap = overlap
        self.samples_per_frame = samples_per_frame
        self.decoded: Dict[str, List[int]] = defaultdict(list)  # chunk starts per request, in decode order
        self.errors: List[str] = []

    # ---- what the schedulers read

    @property
    def detokenize_interval(self) -> int:
        return self._interval

    @property
    def detokenize_overlap(self) -> int:
        return self._overlap

    @property
    def supports_audio_input(self) -> bool:
        return True

    @property
    def available_batch_sizes(self) -> List[int]:
        return [1, 2, 4, 8][: max(1, self.max_batch_size.bit_length())]

    @property
    def prefill_graph_batch_size(self) -> int:
        return self.max_batch_size

    @property
    def cuda_graph_seq_len_buckets(self) -> List[int]:
        return [256, 512, 1024, 2048]

    def tokenize_stream_text(self, text: str) -> List[int]:
        return [len(word) for word in text.split()]

    # ---- LM steps

    def prepare_lm_inputs(self, lm_requests, detokenize_requests) -> Optional[dict]:
        for req in detokenize_requests:
            req.audio_decode_idx = req.next_audio_decode_idx.copy()
        reqs = []
        for req in lm_requests:
            if req.done_lm_prefill or req.done_lm_generation:
                continue
            if self._admit(req):
                reqs.append(req)
            break  # one prefill per step; a request without slots waits
        n_prefill = len(reqs)
        for req in lm_requests:
            if not req.done_lm_prefill or req.done_lm_generation or req in reqs:
                continue
            if len(reqs) >= self.max_batch_size:
                break
            reqs.append(req)
        if not reqs:
            return None
        for req in reqs[:n_prefill]:
            req.done_lm_prefill = True
        return {"is_prefill": n_prefill > 0, "requests": reqs, "n_prefill": n_prefill}

    def prompt_error(self, req) -> Optional[str]:
        if req.model_kwargs.get("fail"):
            return "prompt failed"
        if req.audio_path is not None:
            return "this model does not accept audio input"
        if req.model_kwargs.get("voice") not in (None, "tara"):
            return f"voice {req.model_kwargs['voice']} is not available for this model"
        return None

    def _admit(self, req) -> bool:
        error = self.prompt_error(req)
        if error is not None:
            req.done_lm_prefill = req.done_lm_generation = True
            req.finish_reason = f"error: {error}"
            self.free_kv_cache(req)
            return False
        if not self.free_slots or self.free_detok_slots == []:
            return False
        req.input_length = 8
        req.slot = self.free_slots.pop()
        if self.free_detok_slots is not None:
            req.detok_slot = self.free_detok_slots.pop()
        return True

    def run_lm_prefill(self, requests, lm_inputs):
        return self._run(lm_inputs)

    def run_lm_decode(self, requests, lm_inputs):
        return self._run(lm_inputs)

    def _run(self, lm_inputs):
        if lm_inputs is None:
            return None

        async def collect():
            for req in lm_inputs["requests"]:
                if req.done_lm_generation:
                    continue  # launched before its previous step was collected
                if len(req.lm_output_audio_tokens) < int(req.model_kwargs.get("frames", 10)):
                    frame = np.full(2, len(req.lm_output_audio_tokens), dtype=np.int32)
                    req.lm_output_tokens.append(frame)
                    req.lm_output_audio_tokens.append(frame)
                else:
                    req.done_lm_generation, req.finish_reason = True, "stop_id_encountered"
                    self._release_lm(req)

        return collect()

    # ---- detokenization

    def run_detokenize(self, requests) -> None:
        iv, step = self._interval, self._interval - self._overlap
        for req in requests:
            if req.done_all:
                continue
            n = len(req.lm_output_audio_tokens)
            for start in req.audio_decode_idx:
                if start >= n:
                    continue
                last = req.done_lm_generation and start + iv >= n
                if not last and start + iv > n:
                    self.errors.append(f"{req.request_id}: chunk {start} decoded before its {iv} frames exist")
                if start in self.decoded[req.request_id]:
                    self.errors.append(f"{req.request_id}: chunk {start} decoded twice")
                self.decoded[req.request_id].append(start)
                end = n if last else start + step
                samples = np.repeat(np.arange(start, end, dtype=np.int16), self.samples_per_frame)
                req.output_audio.put(samples.tobytes())
        for req in requests:
            n = len(req.lm_output_audio_tokens)
            if req.done_lm_generation and (not req.audio_decode_idx or req.audio_decode_idx[-1] + iv >= n):
                req.done_all = True

    # ---- request resources

    def _release_lm(self, req) -> None:
        if req.slot is not None:
            self.free_slots.append(req.slot)
            req.slot = None

    def free_kv_cache(self, req) -> None:
        self._release_lm(req)
        if req.detok_slot is not None:
            self.free_detok_slots.append(req.detok_slot)
            req.detok_slot = None

    def all_slots_free(self) -> bool:
        lm = len(self.free_slots) == self.max_batch_size
        return lm and (self.free_detok_slots is None or len(self.free_detok_slots) == self.n_detok_slots)


@dataclass
class ClientRequest:
    request_id: str
    streaming: bool
    frames: int
    fail: bool = False
    audio: bytearray = field(default_factory=bytearray)
    accepted: bool = False  # ACCEPTED (the prompt is built) arrived
    completion: Optional[dict] = None
    audio_after_completion: bool = False


class SchedulerHarness:
    """A scheduler wired to a ``FakeWorker`` over real ZMQ IPC sockets, fed with requests in the API
    server's format (``vox_serve.launch``)."""

    def __init__(self, scheduler_cls, worker: FakeWorker, async_scheduling: bool = False, **kwargs):
        self.dir = tempfile.mkdtemp(prefix="vs")  # IPC paths must stay short
        req_path, res_path = f"{self.dir}/req.ipc", f"{self.dir}/res.ipc"
        self.ctx = zmq.Context()
        self.results = self.ctx.socket(zmq.PULL)
        self.results.bind(f"ipc://{res_path}")
        self.worker = worker
        self.scheduler = scheduler_cls(
            model_name_or_path="fake",
            max_batch_size=worker.max_batch_size,
            request_socket_path=req_path,
            result_socket_path=res_path,
            async_scheduling=async_scheduling,
            model_worker=worker,
            **kwargs,
        )
        self.requests = self.ctx.socket(zmq.PUSH)
        self.requests.connect(f"ipc://{req_path}")
        self.clients: Dict[str, ClientRequest] = {}

    def submit(self, request_id: str, streaming: bool, frames: int, fail: bool = False) -> None:
        self.clients[request_id] = ClientRequest(request_id, streaming, frames, fail)
        kwargs = {"frames": frames, "fail": fail}
        payload = {
            "request_id": request_id,
            "prompt": "hello",
            "audio_path": None,
            "is_streaming": streaming,
            "model_kwargs": kwargs,
        }
        self.requests.send(f"{json.dumps(payload)}|audio_data_placeholder".encode())

    def _drain(self) -> None:
        while True:
            try:
                msg = self.results.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            rid, kind, data = msg.split(b"|", 2)
            c = self.clients[rid.decode()]
            if kind == b"ACCEPTED":
                assert not c.accepted and not c.audio and c.completion is None, f"{c.request_id}: late ACCEPTED"
                c.accepted = True
            elif kind == b"AUDIO":
                assert c.accepted, f"{c.request_id}: audio before ACCEPTED"
                c.audio_after_completion |= c.completion is not None
                c.audio += data
            else:
                assert kind == b"COMPLETION", kind
                assert c.completion is None, f"{c.request_id}: second completion"
                c.completion = json.loads(data)

    def done(self) -> bool:
        return all(c.completion is not None for c in self.clients.values())

    def run(self, max_steps: int = 3000, timeout_s: float = 60.0) -> int:
        """Steps the scheduler until every submitted request has completed; returns the step count."""
        deadline = time.monotonic() + timeout_s
        if self.scheduler.async_scheduling:
            return asyncio.run(self._run_async(max_steps, deadline))
        for step in range(max_steps):
            self.scheduler._step()
            self._drain()
            if self.done() or time.monotonic() > deadline:
                return step
        return max_steps

    async def _run_async(self, max_steps: int, deadline: float) -> int:
        task, lm, dt = None, [], []
        for step in range(max_steps):
            task, lm, dt = await self.scheduler._step_async(task, lm, dt)
            await asyncio.sleep(0)
            self._drain()
            if self.done() or time.monotonic() > deadline:
                if task is not None:
                    await task
                return step
        return max_steps

    def stuck(self) -> List[str]:
        return [rid for rid, c in self.clients.items() if c.completion is None]

    def frames_received(self, request_id: str) -> List[int]:
        """Frame index of every frame the client received, in order."""
        samples = np.frombuffer(bytes(self.clients[request_id].audio), dtype=np.int16)
        return samples[:: self.worker.samples_per_frame].tolist()

    def close(self) -> None:
        for s in (self.scheduler.request_socket, self.scheduler.result_socket):
            s.close(linger=0)
        self.scheduler.context.term()
        self.requests.close(linger=0)
        self.results.close(linger=0)
        self.ctx.term()
        for p in Path(self.dir).iterdir():
            p.unlink()
        Path(self.dir).rmdir()
