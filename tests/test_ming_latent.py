from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vox_serve.model.ming_latent import MingUniAudioNative


_MODEL_PATH = os.getenv("MING_MODEL_PATH")
_PROMPT_WAV = os.getenv("MING_PROMPT_WAV")
_HAS_ASSETS = bool(
    _MODEL_PATH
    and _PROMPT_WAV
    and Path(_MODEL_PATH).exists()
    and Path(_PROMPT_WAV).exists()
)


def _skip_if_no_assets():
    if not _HAS_ASSETS:
        pytest.skip("Ming test assets not configured (set MING_MODEL_PATH/MING_PROMPT_WAV)")


@pytest.mark.parametrize("model_name", ["/path/does/not/exist"])
def test_weights_missing_error(model_name):
    with pytest.raises(RuntimeError):
        MingUniAudioNative(model_name, device="cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(not _HAS_ASSETS, reason="Ming assets not available")
def test_shape_latents_basic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MingUniAudioNative(
        _MODEL_PATH,
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    latents = model.generate_latents(
        "我们的愿景是构建未来服务业的数字化基础设施。",
        prompt_wav=_PROMPT_WAV,
        prompt_text="在此奉劝大家别乱打美白针。",
        language="zh",
    )
    assert latents.shape[0] == 1
    assert latents.shape[2] == model.latent_dim
    assert latents.shape[1] > 0


@pytest.mark.skipif(not _HAS_ASSETS, reason="Ming assets not available")
def test_smoke_generate_wav():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MingUniAudioNative(
        _MODEL_PATH,
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    latents = model.generate_latents(
        "我们的愿景是构建未来服务业的数字化基础设施。",
        prompt_wav=_PROMPT_WAV,
        prompt_text="在此奉劝大家别乱打美白针。",
        language="zh",
    )
    waveform = model.decode_latents_to_audio(latents)
    assert waveform.shape[:2] == (1, 1)
    duration = waveform.shape[-1] / 24_000
    assert duration > 0.5
    peak = float(np.abs(waveform.cpu().numpy()).max())
    mean = float(np.abs(waveform.cpu().numpy()).mean()) + 1e-6
    assert peak / mean > 5.0


@pytest.mark.skipif(not _HAS_ASSETS, reason="Ming assets not available")
def test_flashinfer_fallback(caplog):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MingUniAudioNative(
        _MODEL_PATH,
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_flashinfer=True,
    )
    model.generate_latents(
        "测试 FlashInfer 回退路径。",
        prompt_wav=_PROMPT_WAV,
        language="zh",
    )
    assert "falling back to sdpa" in caplog.text.lower()
