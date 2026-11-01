"""ProgramWorker's host-side batching, with the graph launch replaced (no GPU)."""

from types import SimpleNamespace

import numpy as np
import pytest

from vox_serve.requests import Request
from vox_serve.rt.prompt import FeaturePool, Prompt, ResourceBusyError
from vox_serve.worker import ProgramWorker
from vox_serve.worker.program_worker import final_chunk


def _worker(max_detok_batch: int, stateful: bool):
    w = ProgramWorker.__new__(ProgramWorker)  # no program, no graphs
    w.detok_graphs = [(1, None), (max_detok_batch, None)]
    w.free_detok_slots = [] if stateful else None
    w.nvtx_enabled = False
    w.meta = {"detok_interval": 4}
    w.detok_holds_back = False
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


def test_admit_retries_busy_prompt():
    """A prompt whose voice slot / feature rows are taken waits for a later step instead of failing or
    blocking: running requests release them on the same (scheduler) thread."""
    busy = [True]

    def preprocess(text, audio_path, model_kwargs):
        if busy[0]:
            raise ResourceBusyError("all voice slots are in use")
        return Prompt([1, 2, 3])

    w = ProgramWorker.__new__(ProgramWorker)
    w.program = SimpleNamespace(preprocess=preprocess)
    w.spec = SimpleNamespace(text_stream=None, max_tokens=16)
    w.c, w.max_prompt_len, w.max_context, w.page_size = 1, 64, 64, 16
    w.free_slots, w.free_detok_slots, w.free_pages = [0], None, list(range(1, 9))
    req = Request("a", prompt="hi")
    assert not w._admit(req)
    assert req.lm_prompt is None and not req.done_lm_generation and req.finish_reason is None
    busy[0] = False
    assert w._admit(req)
    assert req.input_length == 3 and req.slot == 0 and len(req.kv_pages) == 2


def test_feature_pool_busy():
    pool = FeaturePool(10)
    a = pool.alloc(6)
    with pytest.raises(ResourceBusyError):  # does not wait for the rows
        pool.alloc(6)
    with pytest.raises(ValueError):  # can never fit
        pool.alloc(11)
    a.release()
    assert pool.alloc(10).offset == 0


@pytest.mark.parametrize("holds_back", [False, True], ids=["whole_window", "look_ahead"])
def test_done_after_the_last_chunk(holds_back):
    # 16 frames, 4-frame windows 3 apart: the window at 12 reaches the last frame. With look-ahead (the
    # codec takes dt_last), the chunk at 15 still has to emit what the one at 12 held back.
    w = _worker(max_detok_batch=2, stateful=True)
    w.meta["detok_overlap"] = 1
    w.detok_holds_back = holds_back
    req = _request("a", [12])
    req.done_lm_generation = True
    w.run_detokenize([req])
    assert req.done_all is not holds_back
    if holds_back:
        assert w._final_chunk(req, 15)  # (its dt_last)
        req.audio_decode_idx = [15]
        w.run_detokenize([req])
        assert req.done_all


# The programs whose codecs hold look-ahead audio back (they take dt_last), restated from their
# ``detok_step``s, which need a GPU and the weights: which frames a chunk emits, given its start, its
# valid length ``dt_len`` and ``dt_last``. Each returns (frames, interval, step, emit, expected frames)
# for a request of ``n`` generated audio frames.


def _higgs_tts(n):
    # 52 delayed rows per chunk, 25 apart, un-delayed into 12 frames of context, 25 emitted frames and 8 of
    # look-ahead; 25 pre-seeded frames, so the first chunk emits 12. A frame exists once its 8 rows do.
    left, core, cb, preseed = 12, 25, 8, 25

    def emit(start, dt_len, dt_last):
        e0 = max(left, preseed) if start == 0 else left
        e1 = max(min(dt_len - (cb - 1), left + core), e0)
        return list(range(start + e0, start + e1))

    return preseed + n + cb - 1, left + core + 8 + cb - 1, core, emit, list(range(preseed, preseed + n))


def _cosyvoice3(n):
    # 30-token windows, 25 apart: 2 tokens of context (2 pre-seeded frames on the first chunk), 25 core
    # tokens (2 mel frames each) and 3 of look-ahead. The vocoder runs on 40 mel frames of history plus the
    # new ones, emits from frame 32 of that window, and holds the last 8 back unless dt_last.
    left, core, hist, src, hold = 2, 25, 40, 32, 8
    mel_end = 0  # end of the vocoder's history

    def emit(start, dt_len, dt_last):
        nonlocal mel_end
        new = 2 * max(min(dt_len, left + core) - left, 0)
        held = 0 if dt_last else hold
        if start == 0:
            lo, hi = 0, max(new - held, 0)
        else:
            lo, hi = mel_end - hist + src, mel_end - hist + max(hist + new - held, src)
        mel_end += new
        return list(range(lo, hi))

    return n + left, left + core + 3, core, emit, list(range(2 * n))


def _decode(n_frames, interval, step, emit, holds_back=True):
    """The frames a request's chunks emit once its tokens are all generated: chunks ``step`` frames apart
    from 0 while frames are left (as the schedulers select them), until ``run_detokenize`` finishes the
    request. Chunks decoded earlier, while tokens were still being generated, had full windows and
    ``dt_last = 0``, as here."""
    frames, start = [], 0
    while start < n_frames:
        dt_len = min(start + interval, n_frames) - start
        frames += emit(start, dt_len, final_chunk(True, start, n_frames, interval, step, True))
        if final_chunk(True, start, n_frames, interval, step, holds_back):
            break
        start += step
    return frames


@pytest.mark.parametrize("program", [_higgs_tts, _cosyvoice3], ids=["higgs_tts", "cosyvoice3"])
def test_look_ahead_codecs_emit_every_frame(program):
    lossy = 0
    for n in range(1, 400):
        n_frames, interval, step, emit, expected = program(n)
        assert _decode(n_frames, interval, step, emit) == expected, n
        n_frames, interval, step, emit, expected = program(n)
        lossy += _decode(n_frames, interval, step, emit, holds_back=False) != expected
    # Finishing after the first window that reaches the last frame lost the end of the audio (issue #65).
    assert lossy > 0
