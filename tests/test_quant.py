"""FP8 linears (``vox_serve.quant``): the GEMM against the dequantized weight, and ``--fuse-lm`` merging."""

import pytest
import torch
from torch import nn

from vox_serve.fusion.projections import MergedLinearMember, merge_projections
from vox_serve.quant import Fp8Linear, fp8_linear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9), reason="needs an FP8 GPU"
)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("m", [1, 7, 64, 300])
def test_fp8_linear_matches_dequantized(m):
    torch.manual_seed(0)
    lin = nn.Linear(256, 384, bias=False, device="cuda", dtype=torch.bfloat16)
    q = Fp8Linear.quantize(lin)
    assert q.weight.dtype == torch.float8_e4m3fn and q.weight_scale.shape == (384,)
    x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ (q.weight.float() * q.weight_scale[:, None]).t()
    y = q(x)
    assert y.shape == (m, 384) and y.dtype == torch.bfloat16
    assert _rel(y, ref) < 0.05  # per-token FP8 activations
    assert _rel(fp8_linear(x.view(1, m, 256), q.weight, q.weight_scale), ref) < 0.05


def test_merge_fp8_projections():
    torch.manual_seed(0)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            for name, n in (("q_proj", 128), ("k_proj", 64), ("v_proj", 64)):
                lin = nn.Linear(256, n, bias=False, device="cuda", dtype=torch.bfloat16)
                setattr(self, name, Fp8Linear.quantize(lin))

    attn = Attn()
    x = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
    before = [getattr(attn, n)(x) for n in ("q_proj", "k_proj", "v_proj")]
    assert merge_projections(attn) == 1
    assert all(isinstance(getattr(attn, n), MergedLinearMember) for n in ("q_proj", "k_proj", "v_proj"))
    after = [getattr(attn, n)(x) for n in ("q_proj", "k_proj", "v_proj")]
    for a, b in zip(after, before, strict=True):
        assert a.shape == b.shape
        assert _rel(a, b) < 1e-2  # same weights; only the activation scale's batch differs (none here)
