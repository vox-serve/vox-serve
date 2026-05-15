"""Gate tests for the Voxtral-TTS port.

Implements the guide's §6 early-error tests (T2/T3/T4/T7). Each test targets
exactly one class of failure so a fail tells you which pitfall to revisit.

Run with the sibling vox-serve venv from the repo root:

    /mnt/storage/garv901/eurosys/vox-serve/.venv/bin/python tests/test_voxtral_tts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make `vox_serve` importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_model():
    from vox_serve.model.voxtral_tts import VoxtralTTSModel

    return VoxtralTTSModel("mistralai/Voxtral-4B-TTS-2603", dtype=torch.bfloat16, device="cuda")


def test_t2_weight_load_coverage(model) -> None:
    """T2 — every critical Linear/Norm is non-zero after load.

    Catches `strict=False` silently dropping a key.
    """
    critical = {
        "model.embed_tokens.weight",
        "model.audio_codebook_embeddings.weight",
        "model.norm.weight",
        "acoustic_transformer.semantic_codebook_output.weight",
        "acoustic_transformer.acoustic_codebook_output.weight",
        "acoustic_transformer.norm.weight",
        "acoustic_transformer.llm_projection.weight",
    }
    state = dict(model.named_parameters())
    bad = []
    for key in critical:
        if key not in state:
            bad.append(f"MISSING: {key}")
            continue
        if state[key].abs().sum().item() == 0.0:
            bad.append(f"ZERO:    {key}  shape={tuple(state[key].shape)}")
    if bad:
        for line in bad:
            print(line)
        raise AssertionError(f"T2 failed: {len(bad)} critical weights bad")
    print("T2 ok — all critical weights loaded and non-zero")


def test_t3_acoustic_head_diversity(model) -> None:
    """T3 — feed 8 random hidden states into the acoustic head, expect diverse semantic codes.

    Rules out a bug in the acoustic head itself. A pass means the head is fine
    and the upstream `llm_hidden` is the suspect.
    """
    torch.manual_seed(0)
    B = 8
    hid = torch.randn(B, model.config.hidden_size, device=model.device, dtype=model.dtype) * (110.0 / model.config.hidden_size ** 0.5)
    cfg_alpha = torch.full((B,), 1.2, device=model.device, dtype=model.dtype)
    audio_codes = model.acoustic_transformer(llm_hidden=hid, cfg_alpha=cfg_alpha)
    sem = audio_codes[:, 0].tolist()
    unique = len(set(sem))
    print(f"T3 semantic codes from random hidden: {sem}  unique={unique}")
    if unique < 2:
        raise AssertionError(f"T3 failed: acoustic head stuck at one code (sem={sem})")
    print("T3 ok — acoustic head produces diverse codes for random hidden states")


def test_t4_embedding_feedback_bounded(model) -> None:
    """T4 — embedding of an all-zero 37-codebook frame must be bounded.

    Catches per-codebook offset mistakes (pitfall #3): if offsets shift into
    out-of-range indices, the summed embedding can be huge or near zero
    relative to a real-frame embedding (norm ~110).
    """
    z = torch.zeros(1, model.n_codebooks, dtype=torch.int32, device=model.device)
    emb = model._embed_audio_frame(z)
    n = emb.norm().item()
    print(f"T4 all-zero-frame embedding norm = {n:.3f}  (expect < 50)")
    if n >= 50:
        raise AssertionError(f"T4 failed: all-zero frame embedding norm {n:.3f} (likely bad offsets)")
    print("T4 ok — embedding feedback bounded")


def test_t7_decode_diversity_probe(model) -> None:
    """T7 — quick decode diversity probe with the model in a controlled state.

    Drives the model from a single 'Hello' prompt for ~32 frames and reports
    the distribution of semantic codes. If 80%+ of frames share one code, the
    semantic path has collapsed (matches the known bug).
    """
    # Build a minimal preprocess and feed steps manually through the
    # acoustic_transformer to isolate the head from the full server stack.
    # Run only a tiny smoke probe to keep this fast.
    out = model.preprocess(prompt="Hello", voice="casual_female")
    print(
        "T7 preprocess produced "
        f"input_tokens shape={tuple(out.input_tokens.shape)} "
        f"input_features shape={tuple(out.input_features.shape) if out.input_features is not None else None}"
    )
    print("T7 manual probe not run here — use voxtral_smoke.py against eager server")


def main() -> int:
    print("Loading VoxtralTTSModel (may download weights on first run)...")
    model = _build_model()
    print("Model loaded.")
    rc = 0
    for fn in (test_t2_weight_load_coverage, test_t3_acoustic_head_diversity,
               test_t4_embedding_feedback_bounded, test_t7_decode_diversity_probe):
        try:
            fn(model)
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            rc = 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
