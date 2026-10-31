"""Fused Llama / Qwen-style decoder stack (``--fuse-layers``): the same math as the modules' forward with
fewer kernels per layer.

Per layer: residual add + RMSNorm in one kernel (FlashInfer ``fused_add_rmsnorm``), one QKV GEMM, one
Triton kernel for the optional per-head q/k RMSNorm (Qwen3), RoPE and the KV-cache write (k and v), the
attention, the output GEMM, add + RMSNorm, one gate/up GEMM, FlashInfer ``silu_and_mul`` and the down
GEMM: 9-10 kernels instead of 16-17. The merged weights replace the modules' own (the modules keep views
into them), so memory does not grow.

Rounding follows the module path where it is observable: the q/k norm output is rounded to bf16 before
RoPE, as two separate kernels would. The residual stream is bf16 as before; SiLU and the product are
computed in fp32 and rounded once (the modules round after each).
"""

from typing import List, Optional

import flashinfer
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from .projections import MergedLinearMember


@triton.jit
def _qk_rope_kv_kernel(
    qkv,
    qkv_stride,
    pos_ptr,
    cs_ptr,
    qw_ptr,
    kw_ptr,
    eps,
    q_out,
    kv,
    page_ptr,
    off_ptr,
    s_page,
    s_kv,
    s_tok,
    s_head,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    HAS_NORM: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = HD // 2
    i = tl.arange(0, HALF)
    src = qkv + r.to(tl.int64) * qkv_stride + h * HD
    x1 = tl.load(src + i).to(tl.float32)
    x2 = tl.load(src + HALF + i).to(tl.float32)
    if HAS_NORM:
        rs = tl.math.rsqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / HD + eps)
        if h < NQ:
            w1 = tl.load(qw_ptr + i).to(tl.float32)
            w2 = tl.load(qw_ptr + HALF + i).to(tl.float32)
        else:
            w1 = tl.load(kw_ptr + i).to(tl.float32)
            w2 = tl.load(kw_ptr + HALF + i).to(tl.float32)
        x1 = (x1 * rs * w1).to(tl.bfloat16).to(tl.float32)
        x2 = (x2 * rs * w2).to(tl.bfloat16).to(tl.float32)
    p = tl.load(pos_ptr + r).to(tl.int64)
    c = tl.load(cs_ptr + p * HD + i)
    s = tl.load(cs_ptr + p * HD + HALF + i)
    o1 = (x1 * c - x2 * s).to(tl.bfloat16)
    o2 = (x2 * c + x1 * s).to(tl.bfloat16)
    if h < NQ:
        dst = q_out + (r.to(tl.int64) * NQ + h) * HD
        tl.store(dst + i, o1)
        tl.store(dst + HALF + i, o2)
    else:
        kh = h - NQ
        pg = tl.load(page_ptr + r).to(tl.int64)
        of = tl.load(off_ptr + r).to(tl.int64)
        dst = kv + pg * s_page + of * s_tok + kh * s_head
        tl.store(dst + i, o1)
        tl.store(dst + HALF + i, o2)
        j = tl.arange(0, HD)
        v = tl.load(qkv + r.to(tl.int64) * qkv_stride + (NQ + NKV + kh) * HD + j)
        tl.store(dst + s_kv + j, v)


def qk_rope_kv(qkv, pos, cos_sin, q_w, k_w, eps, kv_cache, page, off, nq, nkv, hd):
    """qkv [n, (nq + 2 nkv) hd] -> rotated q [n, nq, hd]; rotated k and v go to kv_cache[page, 0/1, off]
    (kv_cache: [pages, 2, page_size, nkv, hd]). cos_sin [max_pos, hd] fp32 (cos | sin halves, neox)."""
    n = qkv.shape[0]
    q = torch.empty(n, nq, hd, dtype=qkv.dtype, device=qkv.device)
    has_norm = q_w is not None
    _qk_rope_kv_kernel[(n, nq + nkv)](
        qkv,
        qkv.stride(0),
        pos,
        cos_sin,
        q_w if has_norm else qkv,
        k_w if has_norm else qkv,
        eps,
        q,
        kv_cache,
        page,
        off,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        NQ=nq,
        NKV=nkv,
        HD=hd,
        HAS_NORM=has_norm,
        num_warps=1 if hd <= 64 else 2,
    )
    return q


def neox_cos_sin(max_pos: int, head_dim: int, theta: float, device) -> torch.Tensor:
    """fp32 cos / sin table [max_pos, head_dim] for FlashInfer-style (non-interleaved) RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv_freq[None]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device)


def _merged(mod: nn.Module, names) -> tuple:
    """Concatenated weight (and bias) of sibling linears; the linears then hold views into it."""
    lins = [getattr(mod, n) for n in names]
    if isinstance(lins[0], MergedLinearMember):  # already merged by --fuse-lm
        g = lins[0].group
        return g.weight, g.bias
    with torch.no_grad():
        w = torch.cat([lin.weight for lin in lins], 0).contiguous()
        b = None if lins[0].bias is None else torch.cat([lin.bias for lin in lins], 0).contiguous()
    start = 0
    for lin in lins:
        size = lin.out_features
        lin.weight = nn.Parameter(w[start : start + size], requires_grad=False)
        if b is not None:
            lin.bias = nn.Parameter(b[start : start + size], requires_grad=False)
        start += size
    return w, b


class FusedDecoderStack:
    """Runs ``layers`` (each with ``input_layernorm``, ``self_attn.{q,k,v,o}_proj`` [+ ``q_norm`` /
    ``k_norm``], ``post_attention_layernorm``, ``mlp.{gate,up,down}_proj`` with SiLU) and the final
    ``norm``. ``cos_sin``: [max_pos, head_dim] fp32 RoPE table (neox)."""

    def __init__(self, layers: List[nn.Module], norm: nn.Module, n_q: int, n_kv: int, head_dim: int, cos_sin):
        self.nq, self.nkv, self.hd = n_q, n_kv, head_dim
        self.cos_sin = cos_sin
        self.norm_w = norm.weight
        self.eps = float(norm.variance_epsilon)
        self.layers = []
        for layer in layers:
            a, mlp = layer.self_attn, layer.mlp
            qkv_w, qkv_b = _merged(a, ("q_proj", "k_proj", "v_proj"))
            gu_w, _ = _merged(mlp, ("gate_proj", "up_proj"))
            qn, kn = getattr(a, "q_norm", None), getattr(a, "k_norm", None)
            self.layers.append(
                dict(
                    in_w=layer.input_layernorm.weight,
                    qkv_w=qkv_w,
                    qkv_b=qkv_b,
                    q_w=None if qn is None else qn.weight,
                    k_w=None if kn is None else kn.weight,
                    qk_eps=float(qn.variance_epsilon) if qn is not None else 0.0,
                    o_w=a.o_proj.weight,
                    o_b=a.o_proj.bias,
                    post_w=layer.post_attention_layernorm.weight,
                    gu_w=gu_w,
                    down_w=mlp.down_proj.weight,
                )
            )

    def __call__(self, h: torch.Tensor, pos: torch.Tensor, attn, kv_cache, l0: int = 0, l1: Optional[int] = None):
        """Layers [l0, l1) and the final norm when l1 is the last layer; h [n, H] is not modified."""
        l1 = len(self.layers) if l1 is None else l1
        n = h.shape[0]
        page, off = attn.kv_page[:n], attn.kv_off[:n]
        residual = h.clone()
        x = None
        for li in range(l0, l1):
            w = self.layers[li]
            if x is None:
                x = flashinfer.norm.rmsnorm(residual, w["in_w"], self.eps)
            else:
                flashinfer.norm.fused_add_rmsnorm(x, residual, w["in_w"], self.eps)
            qkv = F.linear(x, w["qkv_w"], w["qkv_b"])
            q = qk_rope_kv(
                qkv, pos, self.cos_sin, w["q_w"], w["k_w"], w["qk_eps"], kv_cache[li], page, off,
                self.nq, self.nkv, self.hd,
            )  # fmt: skip
            o = attn.run(q, kv_cache[li]).reshape(n, -1)
            x = F.linear(o, w["o_w"], w["o_b"])
            flashinfer.norm.fused_add_rmsnorm(x, residual, w["post_w"], self.eps)
            x = F.linear(flashinfer.activation.silu_and_mul(F.linear(x, w["gu_w"])), w["down_w"])
        if l1 == len(self.layers):
            flashinfer.norm.fused_add_rmsnorm(x, residual, self.norm_w, self.eps)
            return x
        return residual + x
