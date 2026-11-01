"""Fused DAC decoder forward (Higgs Audio codec, ``--fuse-codec``).

The DAC decoder is convolutions joined by Snake activations ``x + sin(a x)^2 / a`` and residual adds.
Unfused, each Snake is five full memory passes (mul, sin, pow, reciprocal-mul, add) and each residual
add another, which at large batches costs more than the convolutions. Here every Snake is one pass,
the residual add and the producing convolution's bias (which PyTorch's cuDNN path adds in a separate
broadcast kernel) ride along with the Snake that follows them (writing the sum only when a later unit
still needs it), and the output stage (add + Snake + Conv1d(32 -> 1, k=7)) is two light Triton kernels
instead of three passes and cuDNN's slow single-output-channel convolution.

Activations are 4-D channels-last tensors [B, C, 1, T] (memory [B, T, C]) and the convolution weights
get 4-D channels-last copies: cuDNN's tensor-core convolutions are NHWC, so with the default layout every
convolution pays an NCHW -> NHWC transpose of its input and weight and one back for its output. (3-D
tensors cannot be channels-last as far as PyTorch is concerned: it copies them back to NCL.)

Math is unchanged; Snake is computed in fp32 and rounded once (the module path rounds after every op
in the decoder's dtype). cuDNN picks its algorithms by benchmarking (``cudnn.benchmark``, on during the
fused decode only, so on the warm-up runs before graph capture): its heuristics choose a ~1000x slower
kernel for some dilated NHWC shapes (512 channels, dilation 9, batch 64).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from ..utils import get_logger

logger = get_logger(__name__)


CL = torch.channels_last


def _is_cl(x: torch.Tensor) -> bool:
    """x [B, C, 1, T] stored as [B, T, C]."""
    return x.dim() == 4 and x.shape[2] == 1 and x.is_contiguous(memory_format=CL)


def to_cl(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T] or [B, C, 1, T] -> channels-last [B, C, 1, T]."""
    return (x.unsqueeze(2) if x.dim() == 3 else x).contiguous(memory_format=CL)


@triton.jit
def _snake_kernel(
    X, Y, Bias, Alpha, S, Out, R, C,
    HAS_Y: tl.constexpr, HAS_BIAS: tl.constexpr, WRITE_SUM: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):  # fmt: skip
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)  # b * T + t
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    m = (r < R)[:, None]
    off = r.to(tl.int64)[:, None] * C + c[None, :]
    # (conv + bias) + residual, each rounded to the decoder's dtype as in the module path
    x = tl.load(X + off, mask=m, other=0.0)
    if HAS_BIAS:
        x = (x.to(tl.float32) + tl.load(Bias + c).to(tl.float32)[None, :]).to(Out.dtype.element_ty)
    if HAS_Y:
        x = (x.to(tl.float32) + tl.load(Y + off, mask=m, other=0.0).to(tl.float32)).to(Out.dtype.element_ty)
    if WRITE_SUM:
        tl.store(S + off, x, mask=m)
    x = x.to(tl.float32)
    a = tl.load(Alpha + c).to(tl.float32)[None, :]
    s = libdevice.sin(a * x)
    tl.store(Out + off, (x + s * s / (a + 1e-9)).to(Out.dtype.element_ty), mask=m)


def snake(x: torch.Tensor, alpha: torch.Tensor, y: torch.Tensor = None, bias: torch.Tensor = None, write_sum=False):
    """Snake of s = (x [+ bias[c]]) [+ y] for channels-last x [B, C, 1, T]; alpha / bias with C elements
    (C a power of two). Returns snake(s), or (s, snake(s)) with ``write_sum``."""
    B, C, _, T = x.shape
    assert _is_cl(x) and (y is None or (_is_cl(y) and y.shape == x.shape)) and C & (C - 1) == 0
    out = torch.empty_like(x)
    s = torch.empty_like(x) if write_sum else out
    BLOCK_C = min(C, 256)
    BLOCK_R = 4096 // BLOCK_C
    R = B * T
    _snake_kernel[(triton.cdiv(R, BLOCK_R), C // BLOCK_C)](
        x, x if y is None else y, x if bias is None else bias, alpha, s, out, R, C,
        HAS_Y=y is not None, HAS_BIAS=bias is not None, WRITE_SUM=write_sum, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C,
        num_warps=4,
    )  # fmt: skip
    return (s, out) if write_sum else out


@triton.jit
def _conv_out_kernel(
    X, W, s_wc, s_wk, Bias, Out, T, C: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr
):
    b = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    c = tl.arange(0, C)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(K):
        src = t + (k - PAD)
        m = ((src >= 0) & (src < T))[:, None]
        x = tl.load(X + (b * T + src)[:, None] * C + c[None, :], mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + c * s_wc + k * s_wk).to(tl.float32)
        acc += tl.sum(x * w[None, :], 1)
    acc += tl.load(Bias).to(tl.float32)
    tl.store(Out + b * T + t, acc.to(Out.dtype.element_ty), mask=t < T)


def conv_out(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """Conv1d with one output channel, stride 1 and 'same' zero padding of channels-last x [B, C, 1, T]
    -> [B, 1, T]."""
    B, C, _, T = x.shape
    K, pad = conv.kernel_size[0], conv.padding[0]
    assert conv.out_channels == 1 and conv.stride[0] == 1 and conv.dilation[0] == 1 and 2 * pad == K - 1
    assert _is_cl(x) and C & (C - 1) == 0
    out = torch.empty(B, 1, T, dtype=x.dtype, device=x.device)
    w = conv.weight
    BLOCK = 128
    _conv_out_kernel[(B, triton.cdiv(T, BLOCK))](
        x, w, w.stride(1), w.stride(2), conv.bias, out, T, C=C, K=K, PAD=pad, BLOCK=BLOCK, num_warps=4
    )
    return out


def prepare(dec: nn.Module) -> None:
    """Channels-last 4-D copies of the decoder's convolution weights (``_w_cl``); call again after
    moving or casting the decoder. Done on first use otherwise (better before graph capture)."""
    for m in dec.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            m._w_cl = m.weight.detach().unsqueeze(2).contiguous(memory_format=CL)


def _w(m: nn.Module) -> torch.Tensor:
    w = getattr(m, "_w_cl", None)
    if w is None or w.device != m.weight.device or w.dtype != m.weight.dtype:
        m._w_cl = w = m.weight.detach().unsqueeze(2).contiguous(memory_format=CL)
    return w


def _conv(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The convolution without its bias (the bias joins the next fused elementwise kernel), channels-last
    [B, C, 1, T] in and out."""
    s, p, d = m.stride[0], m.padding[0], m.dilation[0]
    if isinstance(m, nn.ConvTranspose1d):
        y = F.conv_transpose2d(x, _w(m), None, (1, s), (0, p), (0, m.output_padding[0]), m.groups, (1, d))
    elif m.kernel_size[0] == 1 and s == 1 and p == 0 and m.groups == 1:  # pointwise: one GEMM over B * T rows
        y = F.linear(x.permute(0, 2, 3, 1), m.weight[:, :, 0]).permute(0, 3, 1, 2)
    else:
        y = F.conv2d(x, _w(m), None, (1, s), (0, p), (1, d), m.groups)
    if not _is_cl(y):
        if not getattr(m, "_warned_layout", False):
            logger.warning("%s returned a non-channels-last output; converting", m)
            m._warned_layout = True
        y = to_cl(y)
    return y


def fused_dac_decode(dec: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """``DacDecoder.forward`` (transformers) with fused Snake / bias / residual / output stages, in
    channels-last layout: z [B, D, T] -> audio [B, 1, T * hop]. ``dec.tanh`` must be an identity (the
    Higgs codec drops it)."""
    assert isinstance(dec.tanh, nn.Identity)
    prev = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        return _decode(dec, z)
    finally:
        torch.backends.cudnn.benchmark = prev


def _decode(dec: nn.Module, z: torch.Tensor) -> torch.Tensor:
    blocks = list(dec.block)
    a = snake(_conv(dec.conv1, to_cl(z)), blocks[0].snake1.alpha, bias=dec.conv1.bias)
    for bi, blk in enumerate(blocks):
        units = [blk.res_unit1, blk.res_unit2, blk.res_unit3]
        h, a = snake(_conv(blk.conv_t1, a), units[0].snake1.alpha, bias=blk.conv_t1.bias, write_sum=True)
        for ui, u in enumerate(units):
            y = _conv(u.conv2, snake(_conv(u.conv1, a), u.snake2.alpha, bias=u.conv1.bias))
            assert y.shape == h.shape  # 'same' convolutions: no residual cropping
            if ui + 1 < len(units):
                h, a = snake(y, units[ui + 1].snake1.alpha, h, u.conv2.bias, write_sum=True)
            elif bi + 1 < len(blocks):
                a = snake(y, blocks[bi + 1].snake1.alpha, h, u.conv2.bias)
            else:
                return conv_out(snake(y, dec.snake1.alpha, h, u.conv2.bias), dec.conv2)
