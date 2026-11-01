"""The Python runtime's schedulers, driven against a CPU stand-in for the worker (no GPU, no model)."""

import json
import time

import pytest
from scheduler_sim import FakeWorker, SchedulerHarness

from vox_serve.requests import Request
from vox_serve.scheduler import OfflineScheduler, OnlineScheduler, Scheduler

POLICIES = {"base": Scheduler, "online": OnlineScheduler, "offline": OfflineScheduler}

# name: (worker options, [(streaming, frames, fail)])
WORKLOADS = {
    # More requests than the batch, streaming and not.
    "mixed": ({}, [(True, 9), (False, 14), (True, 6), (False, 11), (True, 13), (False, 7)]),
    # A stateful codec with fewer state slots than requests: finished requests hold their slot until
    # their audio is decoded (the offline policy used to deadlock here).
    "non_streaming_stateful": ({"detok_slots": 4}, [(False, 9)] * 6),
    "streaming_stateful": ({"detok_slots": 4}, [(True, 10)] * 6),
    # Chunks that overlap by one frame, and lengths around the chunk boundaries.
    "overlap": ({"interval": 4, "overlap": 1}, [(True, n) for n in (3, 4, 5, 7, 10)] + [(False, 8)]),
    # A request whose prompt fails completes without audio; the others are unaffected.
    "failed_prompt": ({}, [(False, 8), (True, 8, True), (True, 5), (False, 0)]),
}


def _run(policy, async_scheduling, workload):
    options, reqs = WORKLOADS[workload]
    worker = FakeWorker(**options)
    h = SchedulerHarness(POLICIES[policy], worker, async_scheduling=async_scheduling)
    try:
        for k, (streaming, frames, *fail) in enumerate(reqs):
            h.submit(f"r{k}", streaming, frames, fail=bool(fail and fail[0]))
        steps = h.run()
        return h, worker, steps
    finally:
        h.close()


@pytest.mark.parametrize("workload", list(WORKLOADS))
@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("policy", list(POLICIES))
def test_every_request_completes_with_all_its_audio(policy, async_scheduling, workload):
    h, worker, steps = _run(policy, async_scheduling, workload)
    assert not h.stuck(), f"requests never completed after {steps} steps: {h.stuck()}"
    assert not worker.errors, worker.errors
    for rid, c in h.clients.items():
        assert not c.audio_after_completion, f"{rid}: audio after its completion message"
        if c.fail:
            assert c.completion["reason"].startswith("error"), c.completion
            assert not c.audio
            continue
        assert c.completion == {"status": "completed", "reason": "stop_id_encountered"}
        # Every frame, in order. (How much audio overlapping chunks emit is codec-specific, so a frame
        # may arrive twice where chunks overlap.)
        frames = h.frames_received(rid)
        assert frames == sorted(frames) and set(frames) == set(range(c.frames)), (rid, frames)
        # Chunks start one detokenize step apart from frame 0, and the last one reaches the final frame.
        step = worker.detokenize_interval - worker.detokenize_overlap
        starts = worker.decoded[rid]
        assert starts == list(range(0, step * len(starts), step)), (rid, starts)
        assert not c.frames or starts[-1] + worker.detokenize_interval >= c.frames, (rid, starts)
    assert worker.all_slots_free()


def _scheduler(cls=OnlineScheduler, **worker_options):
    return SchedulerHarness(cls, FakeWorker(**worker_options))


def test_online_pressing(monkeypatch):
    h = _scheduler()
    try:
        s = h.scheduler
        now = 1000.0
        monkeypatch.setattr(time, "time", lambda: now)
        fresh = Request("fresh", is_streaming=True)

        # Five 1 s chunks; playback starts when the first one is sent. A stream is pressing once its last
        # chunk starts playing within 1 s.
        def stream(rid, sent_ago):
            sent = [now - sent_ago] * 5
            return Request(rid, is_streaming=True, chunk_send_timestamps=sent, chunk_durations=[1.0] * 5)

        buffered, draining = stream("buffered", 0.5), stream("draining", 3.5)
        generating = Request("generating", is_streaming=False)
        generated = Request("generated", is_streaming=False, done_lm_generation=True)
        s.active_requests = [fresh, buffered, draining, generating, generated]
        s._update_pressing_status()
        assert fresh.is_pressing, "a stream without audio yet is pressing"
        assert not buffered.is_pressing, "its last chunk starts playing in 3.5 s"
        assert draining.is_pressing, "its last chunk starts playing in 0.5 s"
        assert not generating.is_pressing
        assert generated.is_pressing, "a non-streaming request is pressing once its tokens are generated"
    finally:
        h.close()


def test_request_payload():
    h = _scheduler(Scheduler)
    try:
        payload = {
            "request_id": "a",
            "prompt": "hi",
            "audio_path": "/x.wav",
            "is_streaming": True,
            "model_kwargs": {"speaker": "x", "cfg_alpha": 1.5},
        }
        r = h.scheduler._handle_request_payload(f"{json.dumps(payload)}|audio_data_placeholder".encode())
        assert (r.request_id, r.prompt, r.audio_path, r.is_streaming) == ("a", "hi", "/x.wav", True)
        assert r.is_pressing, "streaming requests start out pressing"
        assert r.model_kwargs["speaker"] == "x"
        assert r.sampling_config.cfg_scale == 1.5
        assert h.scheduler._handle_request_payload(b"no delimiter") is None
    finally:
        h.close()
