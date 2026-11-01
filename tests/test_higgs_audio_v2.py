"""Higgs Audio v2's text normalization and sampling state machine (``decode_tokens`` with the torch sampling
kernels, on synthetic logits; no weights)."""

import pytest
import torch

from vox_serve.model.higgs_audio_v2.program import BOC, EOC, N_CB, NO_EOC, HiggsAudioV2Program, normalize_text


def test_normalize_text():
    assert normalize_text("Hello (world)  it is 70°F，nice") == "Hello world it is 70 degrees Fahrenheit, nice."
    assert normalize_text("  Done!\n\n Next line ") == "Done!\nNext line."
    assert normalize_text("[laugh] ok?") == "<SE>[Laughter]</SE> ok?"


def _program(ras_win: int) -> HiggsAudioV2Program:
    p = HiggsAudioV2Program.__new__(HiggsAudioV2Program)
    p.max_slots, p.ras_win, p.ras_max = 1, ras_win, 2
    p.draw = torch.zeros(1, N_CB, dtype=torch.int32)
    p.alt_draw = torch.zeros(1, N_CB, dtype=torch.int32)
    p.eoc_step = torch.full((2,), NO_EOC, dtype=torch.int32)
    p.trig_step = torch.full((2,), NO_EOC, dtype=torch.int32)
    p.ras_hist = torch.full((2, max(1, ras_win), N_CB), BOC, dtype=torch.int32)
    p.cb_cols = torch.arange(N_CB, dtype=torch.int32)
    p.ones, p.zeros_f, p.zeros_i = torch.ones(1), torch.zeros(1), torch.zeros(1, dtype=torch.int32)
    p.head = lambda x: x  # decode_tokens gets the logits directly
    return p


def _run(p: HiggsAudioV2Program, logits_at, steps: int, temp: float = 0.0) -> torch.Tensor:
    arena = {
        "slot": torch.zeros(1, dtype=torch.int32),
        "active": torch.ones(1, dtype=torch.int32),
        "temp": torch.full((1,), temp),
        "top_k": torch.zeros(1, dtype=torch.int32),
        "top_p": torch.ones(1),
        "min_p": torch.zeros(1),
        "seed": torch.full((1,), 7, dtype=torch.int64),
    }
    out = torch.zeros(1, N_CB, dtype=torch.int32)
    rows = []
    for s in range(steps):
        arena["step"] = torch.tensor([s], dtype=torch.int32)
        arena["first"] = torch.tensor([int(s == 0)], dtype=torch.int32)
        p.decode_tokens(logits_at(s)[None], arena, out, 1)
        rows.append(out[0].clone())
    return torch.stack(rows)


def _peaked(codes, margin: float = 20.0) -> torch.Tensor:
    lg = torch.zeros(N_CB, 1026)
    lg[torch.arange(N_CB), torch.as_tensor(codes)] = margin
    return lg


@pytest.fixture(autouse=True)
def _torch_sampling(monkeypatch):
    monkeypatch.setenv("VOX_TORCH_SAMPLING", "1")


def test_delays_and_end_after_codebook0_eoc():
    e = 12
    rows = _run(_program(0), lambda s: _peaked([EOC if (s == e and c == 0) else 10 + s for c in range(N_CB)]), 24)
    for s in range(8):  # codebooks c > s are still BOC
        assert rows[s].tolist() == [10 + s if c <= s else BOC for c in range(N_CB)]
    assert rows[e, 0] == EOC
    for k in range(1, 7):  # after the EOC, codebooks c < k are forced to EOC; the others finish their frames
        assert rows[e + k].tolist() == [EOC if c < k else 10 + e + k for c in range(N_CB)]
    assert (rows[e + 7 :] == -1).all()  # e frames: rows up to e + 6, then the request has ended


def test_eoc_in_another_codebook_first():
    # The reference starts forcing EOC after the first EOC in any codebook; the end follows codebook 0's EOC.
    t = 10
    rows = _run(_program(0), lambda s: _peaked([EOC if (s == t and c == 3) else 50 for c in range(N_CB)]), 24)
    assert rows[t, 3] == EOC and rows[t, 0] == 50
    assert rows[t + 1, 0] == EOC and rows[t + 1, 1] == 50  # c < 1 forced at step t + 1
    assert rows[t + 2, :2].tolist() == [EOC, EOC]
    assert rows[t + 1 + 6, 0] == EOC and (rows[t + 1 + 7 :] == -1).all()


def test_repetition_aware_sampling():
    # Mildly peaked logits: greedy repeats code 5; RAS redraws it from the untempered distribution once it
    # already occurs twice in the codebook's last 7 rows.
    def lg(s):
        x = torch.zeros(N_CB, 1026)
        x[:, 5] = 2.0
        return x

    plain = _run(_program(0), lg, 12)
    ras = _run(_program(7), lg, 12)
    assert (plain[:, 0] == 5).all()
    assert ras[:2, 0].tolist() == [5, 5]
    assert (ras[2:, 0] != 5).any()
    assert (ras[:, 1:2][:1] == BOC).all()  # delayed codebooks are still held at BOC (never redrawn)
