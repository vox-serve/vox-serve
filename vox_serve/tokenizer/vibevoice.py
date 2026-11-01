"""VibeVoice acoustic decoder: 64-d latents at 7.5 Hz -> 24 kHz mono audio (3200 samples per frame).

Port of the decoder of ``VibeVoiceAcousticTokenizerModel`` (microsoft/VibeVoice,
``vibevoice/modular/modular_vibevoice_tokenizer.py``, MIT). A causal conv stem, seven stages of ConvNeXt-like
blocks (RMSNorm, depthwise conv of width 7, GELU FFN, layer scales) separated by causal transposed convs
(x8, x5, x5, x4, x2, x2), and a conv head; checkpoint names are kept so the weights load as they are.

Every convolution is causal, so a chunk of frames decoded with each layer's last inputs (``states``, one
tensor per conv, zero for a request's first chunk) is exactly the reference's frame-by-frame streaming
decode (``VibeVoiceTokenizerStreamingCache``): a zero history adds nothing, as an empty one does there.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class _Holder(nn.Module):
    """Attribute container that keeps the reference's module nesting (``conv.conv``, ``convtr.convtr``)."""

    def __init__(self, **children: nn.Module):
        super().__init__()
        for name, m in children.items():
            setattr(self, name, m)


class _Conv(nn.Module):
    """``SConv1d`` (causal, stride 1): the input is prefixed with the previous ``k - 1`` samples."""

    def __init__(self, cin: int, cout: int, k: int, groups: int = 1):
        super().__init__()
        self.conv = _Holder(conv=nn.Conv1d(cin, cout, k, groups=groups))
        self.cin, self.ctx = cin, k - 1

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        full = torch.cat([state, x], dim=2)
        return self.conv.conv(full), full[:, :, -self.ctx :]


class _ConvTr(nn.Module):
    """``SConvTranspose1d`` (causal, kernel ``2 * stride``, right trim ``kernel - stride``): the new
    output of a chunk includes the overlap of the previous ``kernel - 1`` inputs."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        k = 2 * stride
        self.convtr = _Holder(convtr=nn.ConvTranspose1d(cin, cout, k, stride))
        self.cin, self.ctx, self.stride, self.trim = cin, k - 1, stride, k - stride

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        full = torch.cat([state, x], dim=2)
        y = self.convtr.convtr(full)
        y = y[:, :, : y.shape[-1] - self.trim]
        return y[:, :, -x.shape[-1] * self.stride :], full[:, :, -self.ctx :]


class _RMSNorm(nn.Module):
    """``ConvRMSNorm``: over channels of ``[B, C, T]``, fp32 statistics."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight
        return y.transpose(1, 2)


class _FFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class _Block(nn.Module):
    """``Block1D`` with a depthwise causal conv mixer."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.norm = _RMSNorm(dim, eps)
        self.ffn_norm = _RMSNorm(dim, eps)
        self.mixer = _Holder(conv=_Conv(dim, dim, 7, groups=dim))
        self.ffn = _FFN(dim, 4 * dim)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.ffn_gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y, state = self.mixer.conv(self.norm(x), state)
        x = x + y * self.gamma.unsqueeze(-1)
        y = self.ffn(self.ffn_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + y * self.ffn_gamma.unsqueeze(-1), state


class VibeVoiceAcousticDecoder(nn.Module):
    """``TokenizerDecoder`` of the acoustic tokenizer, streamed over explicit per-conv states."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        depths = cfg.get("decoder_depths") or cfg["encoder_depths"]
        depths = [int(d) for d in depths.split("-")] if isinstance(depths, str) else list(depths)
        if not cfg.get("decoder_depths"):
            depths = depths[::-1]
        ratios = list(cfg["decoder_ratios"])
        nf, eps = int(cfg["decoder_n_filters"]), float(cfg["layernorm_eps"])
        supported = cfg.get("causal", True) and cfg.get("disable_last_norm", True)
        if not supported or cfg["mixer_layer"] != "depthwise_conv":
            raise ValueError("unsupported VibeVoice acoustic decoder configuration")
        n = len(depths)
        ch = [nf * 2 ** (n - 1 - i) for i in range(n)]
        self.upsample_layers = nn.ModuleList([nn.ModuleList([_Conv(int(cfg["vae_dim"]), ch[0], 7)])])
        for i, r in enumerate(ratios):
            self.upsample_layers.append(nn.ModuleList([_ConvTr(ch[i], ch[i + 1], r)]))
        self.stages = nn.ModuleList(
            [nn.ModuleList([_Block(ch[i], eps) for _ in range(d)]) for i, d in enumerate(depths)]
        )
        self.head = _Conv(ch[-1], int(cfg["channels"]), 7)
        self.hop = 1
        for r in ratios:
            self.hop *= r

    def convs(self) -> List[nn.Module]:
        """The convolutions in the order their states are passed."""
        out: List[nn.Module] = []
        for up, stage in zip(self.upsample_layers, self.stages, strict=True):
            out.append(up[0])
            out.extend(b.mixer.conv for b in stage)
        out.append(self.head)
        return out

    def state_shapes(self) -> List[Tuple[int, int]]:
        """``(channels, samples)`` of each conv's state."""
        return [(c.cin, c.ctx) for c in self.convs()]

    def forward(self, latents: torch.Tensor, states: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """latents ``[B, vae_dim, T]`` -> audio ``[B, 1, T * hop]`` and the states after this chunk."""
        x, it, new = latents, iter(states), []
        for up, stage in zip(self.upsample_layers, self.stages, strict=True):
            x, s = up[0](x, next(it))
            new.append(s)
            for block in stage:
                x, s = block(x, next(it))
                new.append(s)
        x, s = self.head(x, next(it))
        new.append(s)
        return x, new
