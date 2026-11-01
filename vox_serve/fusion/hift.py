"""Channels-last HiFT vocoder decode (CosyVoice3 ``--fuse-codec``).

cuDNN's tensor-core (TF32) convolutions are NHWC. With the default NCL layout every one of the vocoder's
~85 convolutions pays an NCHW -> NHWC transpose of its input and weight and one back for its output, as
much memory traffic as the convolution itself at large batches. Here activations are 4-D channels-last
tensors [B, C, 1, T] (memory [B, T, C]) and weights are prepared once in the same layout, so cuDNN runs
NHWC end to end. The residual blocks' Snake activations, causal left padding, convolution biases and
residual adds are one Triton pass each; leaky ReLU + nearest upsampling + padding is another.

Same math as ``CausalHiFT.decode``; the convolutions may pick different cuDNN algorithms (rounding-level
differences, amplified by TF32 like any change to this vocoder).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from ..tokenizer.cosyvoice3 import DILATIONS, N_FFT, UP_KERNELS, UPSAMPLE

CL = torch.channels_last


@triton.jit
def _snake_cl_kernel(
    X, Alpha, Y, R, Bias, S, C, T, pad, rows,
    HAS_R: tl.constexpr, HAS_BIAS: tl.constexpr, WRITE_SUM: tl.constexpr, SNAKE: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):  # fmt: skip
    o = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)  # output row b * (pad + T) + t
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    b = o // (pad + T)
    src = o % (pad + T) - pad
    m = ((o < rows) & (src >= 0))[:, None]
    i = (b * T + src).to(tl.int64)[:, None] * C + c[None, :]
    x = tl.load(X + i, mask=m, other=0.0)
    if HAS_BIAS:  # (conv + bias) + residual, as the module path
        x += tl.load(Bias + c)[None, :]
    if HAS_R:
        x += tl.load(R + i, mask=m, other=0.0)
    if WRITE_SUM:
        tl.store(S + i, x, mask=m)
    if SNAKE:
        a = tl.load(Alpha + c)[None, :]
        s = libdevice.sin(x * a)
        x = x + (1.0 / (a + 1e-9)) * s * s
    tl.store(Y + o.to(tl.int64)[:, None] * C + c[None, :], tl.where(m, x, 0.0), mask=(o < rows)[:, None])


def snake_cl(x, alpha, pad=0, res=None, bias=None, write_sum=False, snake=True):
    """Snake of s = (x [+ bias[c]]) [+ res], left-padded with ``pad`` zeros, for channels-last fp32 x
    [B, C, 1, T] -> [B, C, 1, pad + T] (channels-last); (s, out) with ``write_sum``."""
    B, C, _, T = x.shape
    assert x.is_contiguous(memory_format=CL) and x.dtype == torch.float32 and C & (C - 1) == 0
    assert res is None or (res.is_contiguous(memory_format=CL) and res.shape == x.shape)
    y = torch.empty(B, C, 1, pad + T, dtype=x.dtype, device=x.device, memory_format=CL)
    s = torch.empty_like(x) if write_sum else y
    BLOCK_C = min(C, 128)
    BLOCK_R = 2048 // BLOCK_C
    rows = B * (pad + T)
    _snake_cl_kernel[(triton.cdiv(rows, BLOCK_R), C // BLOCK_C)](
        x, alpha, y, x if res is None else res, x if bias is None else bias, s, C, T, pad, rows,
        HAS_R=res is not None, HAS_BIAS=bias is not None, WRITE_SUM=write_sum, SNAKE=snake,
        BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=4,
    )  # fmt: skip
    return (s, y) if write_sum else y


@triton.jit
def _lrelu_up_pad_kernel(X, Y, C, T, up, pad, rows, slope, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    o = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)  # output row b * (pad + up T) + t
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = pad + up * T
    b = o // n
    t = o % n - pad
    m = ((o < rows) & (t >= 0))[:, None] & (c < C)[None, :]
    x = tl.load(X + (b * T + t // up).to(tl.int64)[:, None] * C + c[None, :], mask=m, other=0.0)
    x = tl.where(x >= 0, x, x * slope)
    tl.store(Y + o.to(tl.int64)[:, None] * C + c[None, :], x, mask=(o < rows)[:, None] & (c < C)[None, :])


def lrelu_up_pad(x, slope, up=1, pad=0):
    """F.pad(F.interpolate(F.leaky_relu(x, slope), scale_factor=up, mode="nearest"), (pad, 0)) for
    channels-last x [B, C, 1, T]."""
    B, C, _, T = x.shape
    assert x.is_contiguous(memory_format=CL)
    y = torch.empty(B, C, 1, pad + up * T, dtype=x.dtype, device=x.device, memory_format=CL)
    BLOCK_C = min(triton.next_power_of_2(C), 128)
    BLOCK_R = 2048 // BLOCK_C
    rows = B * (pad + up * T)
    _lrelu_up_pad_kernel[(triton.cdiv(rows, BLOCK_R), triton.cdiv(C, BLOCK_C))](
        x, y, C, T, up, pad, rows, slope, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=4
    )
    return y


def _w(conv: nn.Conv1d) -> torch.Tensor:
    """Channels-last 4-D weight [Co, Ci, 1, K], cached on the module."""
    w = getattr(conv, "_w_cl", None)
    if w is None or w.device != conv.weight.device:
        w = conv.weight.detach().unsqueeze(2).contiguous(memory_format=CL)
        conv._w_cl = w
    return w


def _conv(conv: nn.Conv1d, x: torch.Tensor, bias: bool = True) -> torch.Tensor:
    return F.conv2d(
        x, _w(conv), conv.bias if bias else None, (1, conv.stride[0]), (0, conv.padding[0]), (1, conv.dilation[0])
    )


def _resblock(rb: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``_ResBlock.forward`` on channels-last x [B, C, 1, T]."""
    k = rb.k
    xt = b2 = None
    for c1, c2, a1, a2, d in zip(rb.convs1, rb.convs2, rb.activations1, rb.activations2, DILATIONS, strict=True):
        if xt is None:
            a = snake_cl(x, a1.alpha, (k - 1) * d)
        else:  # x += conv2 output + bias, then Snake
            x, a = snake_cl(xt, a1.alpha, (k - 1) * d, res=x, bias=b2, write_sum=True)
        a = snake_cl(_conv(c1, a, bias=False), a2.alpha, k - 1, bias=c1.bias)
        xt, b2 = _conv(c2, a, bias=False), c2.bias
    return snake_cl(xt, b2, 0, res=x, bias=b2, snake=False)


def hift_decode_cl(hift: nn.Module, mel: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``CausalHiFT.decode`` (mel [B, 80, T], source [B, T * 480] -> audio [B, T * 480]) in channels-last."""
    s_stft = hift.stft(s)
    x = mel.unsqueeze(2).contiguous(memory_format=CL)
    x = _conv(hift.conv_pre, torch.cat([x, x.new_zeros(*x.shape[:3], 4)], 3))  # F.pad(mel, (0, 4))
    for i, (u, k) in enumerate(zip(UPSAMPLE, UP_KERNELS, strict=True)):
        x = _conv(hift.ups[i], lrelu_up_pad(x, 0.1, u, k - 1))
        if i == len(UPSAMPLE) - 1:  # reflect pad (1, 0)
            y = torch.empty(*x.shape[:3], x.shape[3] + 1, dtype=x.dtype, device=x.device, memory_format=CL)
            y[..., 1:] = x
            y[..., :1] = x[..., 1:2]
            x = y
        down = hift.source_downs[i]  # 18 input channels: stays NCL
        si = down(F.pad(s_stft, (down.stride[0] - 1, 0)) if down.stride[0] > 1 else s_stft)
        x = x + _resblock(hift.source_resblocks[i], si.unsqueeze(2).contiguous(memory_format=CL))
        r = [_resblock(hift.resblocks[i * 3 + j], x) for j in range(3)]
        x = (r[0] + r[1] + r[2]) / 3
    x = _conv(hift.conv_post, lrelu_up_pad(x, 0.01, 1, 6))[:, :, 0]  # [B, 18, T], memory [B, T, 18]
    audio = hift.istft(torch.exp(x[:, : N_FFT // 2 + 1]), torch.sin(x[:, N_FFT // 2 + 1 :]))
    return audio.clamp(-0.99, 0.99)
