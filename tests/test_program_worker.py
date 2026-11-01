"""ProgramWorker's host-side batching, with the graph launch replaced (no GPU)."""

import numpy as np
import pytest

from vox_serve.requests import Request
from vox_serve.worker import ProgramWorker


def _worker(max_detok_batch: int, stateful: bool):
    w = ProgramWorker.__new__(ProgramWorker)  # no program, no graphs
    w.detok_graphs = [(1, None), (max_detok_batch, None)]
    w.free_detok_slots = [] if stateful else None
    w.nvtx_enabled = False
    w.meta = {"detok_interval": 4}
    w.batches = []
    w._detokenize = lambda chunks: w.batches.append([(req.request_id, start) for req, start in chunks])
    return w


def _request(rid: str, starts):
    req = Request(rid)
    req.lm_output_audio_tokens = [np.zeros(2, dtype=np.int32)] * 16
    req.audio_decode_idx = list(starts)
    return req


@pytest.mark.parametrize("stateful", [True, False], ids=["stateful", "stateless"])
def test_detok_batches(stateful):
    w = _worker(max_detok_batch=2, stateful=stateful)
    w.run_detokenize([_request("a", [0, 4, 8]), _request("b", [0, 4]), _request("c", [0])])
    flat = [chunk for batch in w.batches for chunk in batch]
    assert sorted(flat) == sorted([("a", 0), ("a", 4), ("a", 8), ("b", 0), ("b", 4), ("c", 0)])
    assert all(len(batch) <= 2 for batch in w.batches)
    for rid in "abc":  # a request's chunks are decoded in order
        starts = [s for r, s in flat if r == rid]
        assert starts == sorted(starts)
    if stateful:
        # A stateful codec updates a request's state per chunk: never two chunks of it in one batch.
        assert all(len({r for r, _ in batch}) == len(batch) for batch in w.batches), w.batches
        assert w.batches == [[("a", 0), ("b", 0)], [("c", 0)], [("a", 4), ("b", 4)], [("a", 8)]]
    else:
        assert w.batches == [[("a", 0), ("a", 4)], [("a", 8), ("b", 0)], [("b", 4), ("c", 0)]]
