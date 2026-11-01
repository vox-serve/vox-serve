"""Fused Triton kernels for chunked DiT decoders and their vocoders (CosyVoice3 flow and HiFT).

The flow decoder runs 10 solver steps x 22 blocks per chunk, so per-op kernels dominate: at batch 1
the step is launch-bound, at large batches the fp32 residual stream makes every unfused elementwise
op a full memory pass. These kernels cover the non-GEMM work of a block:

* ``gconv_mish``: causal grouped Conv1d + Mish (the conv position embedding), one launch instead of
  cuDNN's per-group kernels and layout transposes;
* ``res_ln_mod``: ``h += gate * o`` (optional), then ``LayerNorm(h) * (1 + scale) + shift`` cast to
  the GEMM dtype (adaLN-Zero modulation);
* ``rope_qk``: in-place partial rotary on channels [0, 64) of q and k inside a fused qkv tensor.

and of the HiFT vocoder's residual blocks:

* ``snake_pad``: Snake activation + causal left padding, one pass instead of six; optionally also
  the residual add and the producing convolution's bias (``--fuse-codec``).

``ring_write`` (``--fuse-codec``) stores a chunk's keys and values into the flow's history ring.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gconv_mish_kernel(
    X, W, Bias, Res, Y, T, sxb, sxt, srb, srt, syb, syt,
    K: tl.constexpr, CG: tl.constexpr, BLOCK_T: tl.constexpr, HAS_RES: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0)
    g = tl.program_id(1)
    t = tl.program_id(2) * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.arange(0, CG)
    live = t[:, None] < T
    acc = tl.zeros((BLOCK_T, CG), tl.float32)
    x_base = X + b * sxb + g * CG + c[None, :]
    w_base = W + g * K * CG * CG + c[:, None] * CG + c[None, :]
    for k in range(K):
        x = tl.load(x_base + (t[:, None] + k) * sxt, mask=live, other=0.0)
        w = tl.load(w_base + k * CG * CG)
        acc += tl.dot(x, w)
    acc += tl.load(Bias + g * CG + c).to(tl.float32)[None, :]
    # mish(x) = x * tanh(softplus(x)), rounded to the activation dtype as in the unfused module
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    y = (acc * (1.0 - 2.0 / (tl.exp(2.0 * sp) + 1.0))).to(X.dtype.element_ty)
    if HAS_RES:
        r = tl.load(Res + b * srb + t[:, None] * srt + g * CG + c[None, :], mask=live, other=0.0)
        out = r.to(tl.float32) + y.to(tl.float32)
        tl.store(Y + b * syb + t[:, None] * syt + g * CG + c[None, :], out, mask=live)
    else:
        tl.store(Y + b * syb + t[:, None] * syt + g * CG + c[None, :], y, mask=live)


def gconv_weight(conv: torch.nn.Conv1d) -> torch.Tensor:
    """Conv1d weight [C, C / G, K] -> [G, K, C / G (in), C / G (out)] for ``gconv_mish``."""
    c_out, cg, k = conv.weight.shape
    g = conv.groups
    return conv.weight.detach().view(g, cg, cg, k).permute(0, 3, 2, 1).contiguous()


def gconv_mish(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, res: torch.Tensor = None) -> torch.Tensor:
    """x [B, K - 1 + T, C] (time-major, the K - 1 frames before the output frames first) ->
    mish(conv(x)) [B, T, C] in x's dtype, or ``res + mish(conv(x))`` in fp32 if ``res`` [B, T, C]
    is given. ``w`` from ``gconv_weight``."""
    B, L, C = x.shape
    G, K, CG, _ = w.shape
    T = L - (K - 1)
    assert x.stride(2) == 1 and C == G * CG
    if res is None:
        y = torch.empty(B, T, C, dtype=x.dtype, device=x.device)
        r = y
    else:
        assert res.stride(2) == 1
        y = torch.empty(B, T, C, dtype=torch.float32, device=x.device)
        r = res
    BLOCK_T = 64
    grid = (B, G, triton.cdiv(T, BLOCK_T))
    _gconv_mish_kernel[grid](
        x, w, bias, r, y, T, x.stride(0), x.stride(1), r.stride(0), r.stride(1), y.stride(0), y.stride(1),
        K=K, CG=CG, BLOCK_T=BLOCK_T, HAS_RES=res is not None, num_warps=4,
    )  # fmt: skip
    return y


@triton.jit
def _res_ln_mod_kernel(H, O, Gate, Scale, Shift, N, so, eps, D: tl.constexpr, HAS_RES: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, D)
    h = tl.load(H + row * D + c)
    if HAS_RES:
        o = tl.load(O + row * so + c).to(tl.float32)
        h = h + tl.load(Gate + c) * o
        tl.store(H + row * D + c, h)
    mean = tl.sum(h, 0) / D
    d = h - mean
    var = tl.sum(d * d, 0) / D
    n = d * tl.rsqrt(var + eps) * (1.0 + tl.load(Scale + c)) + tl.load(Shift + c)
    tl.store(N + row * D + c, n.to(N.dtype.element_ty))


def res_ln_mod(
    h: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    out_dtype: torch.dtype,
    o: torch.Tensor = None,
    gate: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """h [.., D] fp32 contiguous. If ``o`` is given, first ``h += gate * o`` (in place, o in any
    dtype, rows contiguous). Returns ``LayerNorm(h) * (1 + scale) + shift`` in ``out_dtype``;
    ``scale`` / ``shift`` / ``gate`` are fp32 [D]."""
    D = h.shape[-1]
    assert h.is_contiguous() and h.dtype == torch.float32
    rows = h.numel() // D
    n = torch.empty(h.shape, dtype=out_dtype, device=h.device)
    if o is not None:
        assert o.stride(-1) == 1
        o2 = o.reshape(rows, D)
        so = o2.stride(0)
    else:
        o2, so, gate = h, 0, scale
    _res_ln_mod_kernel[(rows,)](
        h, o2, gate, scale, shift, n, so, eps, D=D, HAS_RES=o is not None, num_warps=4
    )  # fmt: skip
    return n


@triton.jit
def _rope_qk_kernel(QKV, Cos, Sin, s_row, k_off, ROT: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, ROT)
    pair = c ^ 1
    sign = tl.where(c % 2 == 0, -1.0, 1.0)
    cos = tl.load(Cos + row * ROT + c)
    sin = tl.load(Sin + row * ROT + c)
    for off in tl.static_range(2):
        base = QKV + row * s_row + off * k_off
        x = tl.load(base + c).to(tl.float32)
        xp = tl.load(base + pair).to(tl.float32)
        tl.store(base + c, (x * cos + sign * xp * sin).to(QKV.dtype.element_ty))


def rope_qk(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_off: int) -> None:
    """In place: rotates channels [0, ROT) of q (at channel 0) and k (at ``k_off``) of qkv [.., W]
    (rows contiguous) by interleaved-pair rotary angles cos / sin [.., ROT] (fp32, contiguous)."""
    rot = cos.shape[-1]
    rows = cos.numel() // rot
    q2 = qkv.view(rows, qkv.shape[-1])
    assert cos.is_contiguous() and sin.is_contiguous() and q2.stride(1) == 1
    _rope_qk_kernel[(rows,)](q2, cos, sin, q2.stride(0), k_off, ROT=rot, num_warps=1)


@triton.jit
def _snake_pad_kernel(
    X, Alpha, Y, C, T, pad, R, Bias, S,
    HAS_R: tl.constexpr, HAS_BIAS: tl.constexpr, WRITE_SUM: tl.constexpr, SNAKE: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0)  # b * C + c
    t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)  # output position
    src = t - pad
    m = (src >= 0) & (src < T)
    x = tl.load(X + row * T + src, mask=m, other=0.0)
    if HAS_BIAS:  # (conv + bias) + residual: the module path's order, so the sum is bit-identical
        x += tl.load(Bias + row % C)
    if HAS_R:
        x += tl.load(R + row * T + src, mask=m, other=0.0)
    if WRITE_SUM:
        tl.store(S + row * T + src, x, mask=m)
    if SNAKE:
        a = tl.load(Alpha + row % C)
        s = libdevice.sin(x * a)  # tl.sin is sin.approx, inaccurate for large arguments
        x = x + (1.0 / (a + 1e-9)) * s * s
    tl.store(Y + row * (T + pad) + t, tl.where(src >= 0, x, 0.0), mask=t < T + pad)


def snake_pad(
    x: torch.Tensor,
    alpha: torch.Tensor,
    pad: int = 0,
    res: torch.Tensor = None,
    bias: torch.Tensor = None,
    write_sum: bool = False,
    snake: bool = True,
):
    """Snake activation ``s + sin(a s)^2 / a`` of s = (x [+ bias[c]]) [+ res] for x [B, C, T] (fp32,
    contiguous), left-padded with ``pad`` zeros: [B, C, pad + T]. With ``write_sum`` returns (s, out);
    ``snake=False`` skips the activation (a fused residual / bias add)."""
    B, C, T = x.shape
    assert x.is_contiguous() and x.dtype == torch.float32
    assert res is None or (res.is_contiguous() and res.shape == x.shape)
    y = torch.empty(B, C, pad + T, dtype=x.dtype, device=x.device)
    s = torch.empty_like(x) if write_sum else y
    BLOCK = 1024
    _snake_pad_kernel[(B * C, triton.cdiv(pad + T, BLOCK))](
        x, alpha, y, C, T, pad, x if res is None else res, x if bias is None else bias, s,
        HAS_R=res is not None, HAS_BIAS=bias is not None, WRITE_SUM=write_sum, SNAKE=snake,
        BLOCK=BLOCK, num_warps=4,
    )  # fmt: skip
    return (s, y) if write_sum else y


@triton.jit
def _ring_write_kernel(Dst, Src, Ring, s_kv, s_slot, s_src, s_frame, k_off, v_off, D: tl.constexpr):
    r = tl.program_id(0)
    f = tl.program_id(1)
    d = tl.arange(0, D)
    slot = tl.load(Ring + r).to(tl.int64)
    src = Src + r.to(tl.int64) * s_src + f * s_frame
    dst = Dst + slot * s_slot + f * D
    tl.store(dst + d, tl.load(src + k_off + d))
    tl.store(dst + s_kv + d, tl.load(src + v_off + d))


def ring_write(hist: torch.Tensor, qkv: torch.Tensor, ring: torch.Tensor, dim: int) -> None:
    """hist[0][ring[r]] = k[r], hist[1][ring[r]] = v[r] for qkv [rows, frames, 3 dim] (q | k | v) and
    hist [2, slots, frames * dim]: both ``index_copy_`` calls of the history ring in one launch, 16-byte
    accesses instead of the strided element-wise copy."""
    rows, frames, _ = qkv.shape
    assert hist.stride(2) == 1 and qkv.stride(2) == 1 and hist.shape[2] == frames * dim
    _ring_write_kernel[(rows, frames)](
        hist, qkv, ring, hist.stride(0), hist.stride(1), qkv.stride(0), qkv.stride(1), dim, 2 * dim,
        D=dim, num_warps=4,
    )  # fmt: skip
