# https://github.com/hubertsiuzdak/snac/tree/main

import json
import math
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.utils import parametrize as _parametrize
from torch.nn.utils.parametrizations import weight_norm

from ..utils import get_logger

logger = get_logger(__name__)

# Inference-time fusion toggles. Hardcode here; flip to False for the upstream
# snac path when debugging or measuring deltas.
#
# Bake weight_norm reparametrization (g/||v||*v) into the underlying Conv weights
# at load time. Saves ~71 small kernels per forward; helps small batches the most.
FUSE_REMOVE_WEIGHT_NORM = True
# Compose each quantizer's (codebook_embed -> out_proj 1x1 conv) into a single
# Embedding lookup table at load. from_codes becomes one Embedding(codebook_size,
# input_dim) per codebook instead of Embedding(codebook_size, codebook_dim) plus a
# 1x1 Conv(codebook_dim -> input_dim).
FUSE_COMPOSE_QUANTIZER_TABLE = True
# Swap depthwise nn.Conv1d (groups == in_channels == out_channels) for a hand-tuned
# Triton kernel. cuDNN's depthwise Conv1d path is launch- and memory-bound and
# accounts for ~42% of decoder time at bs=64; this replacement is the largest
# remaining lever.
FUSE_TRITON_DEPTHWISE_CONV = True


_depthwise_kernel = None


def _get_depthwise_kernel():
    """Lazy-build Triton depthwise Conv1d kernel. Returns the launch function or None."""
    global _depthwise_kernel
    if _depthwise_kernel is not None:
        return _depthwise_kernel if _depthwise_kernel else None
    try:
        import triton
        import triton.language as tl
    except ImportError:
        _depthwise_kernel = False
        return None

    @triton.jit
    def _kernel(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, C, T,
        K: tl.constexpr, DILATION: tl.constexpr, PADDING: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        t_out = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        out_mask = t_out < T

        if HAS_BIAS:
            bias_val = tl.load(b_ptr + pid_c).to(tl.float32)
        else:
            bias_val = 0.0
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32) + bias_val

        x_base = pid_b * C * T + pid_c * T
        w_base = pid_c * K
        # K is small (typically 7); unroll explicitly so each tap is a scalar load.
        for k in tl.static_range(K):
            t_in = t_out + k * DILATION - PADDING
            valid = (t_in >= 0) & (t_in < T) & out_mask
            x_val = tl.load(x_ptr + x_base + t_in, mask=valid, other=0.0).to(tl.float32)
            wk = tl.load(w_ptr + w_base + k).to(tl.float32)
            acc += x_val * wk

        out_offs = pid_b * C * T + pid_c * T + t_out
        tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)

    def launch(x: torch.Tensor, weight: torch.Tensor, bias, dilation: int, padding: int):
        # x: (B, C, T) contiguous. weight: (C, 1, K). bias: (C,) or None.
        assert x.is_contiguous() and weight.is_contiguous()
        B, C, T = x.shape
        K = weight.shape[-1]
        out = torch.empty_like(x)
        BLOCK_T = 128
        grid = (B, C, triton.cdiv(T, BLOCK_T))
        _kernel[grid](
            x, weight.view(C, K), bias if bias is not None else x, out,
            B, C, T,
            K=K, DILATION=dilation, PADDING=padding,
            HAS_BIAS=bias is not None,
            BLOCK_T=BLOCK_T,
            num_warps=4,
        )
        return out

    _depthwise_kernel = launch
    return launch


class _TritonDepthwiseConv1d(nn.Module):
    """Drop-in replacement for nn.Conv1d when groups == in_channels == out_channels."""

    def __init__(self, src: nn.Conv1d):
        super().__init__()
        assert src.groups == src.in_channels == src.out_channels
        assert src.stride == (1,)
        # src.weight shape: (C, 1, K)
        self.register_buffer("weight", src.weight.detach().contiguous())
        if src.bias is not None:
            self.register_buffer("bias", src.bias.detach().contiguous())
        else:
            self.bias = None
        self.dilation = src.dilation[0]
        self.padding = src.padding[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = _get_depthwise_kernel()
        if fn is None:
            # Fall back to nn.functional.conv1d if triton isn't available.
            return F.conv1d(
                x, self.weight, self.bias,
                stride=1, padding=self.padding,
                dilation=self.dilation, groups=self.weight.shape[0],
            )
        return fn(x.contiguous(), self.weight, self.bias, self.dilation, self.padding)


def _swap_depthwise_convs(module: nn.Module) -> int:
    """Replace every depthwise nn.Conv1d in `module` with _TritonDepthwiseConv1d.

    Depthwise here means groups == in_channels == out_channels (i.e. one filter per
    channel). Returns the number of modules swapped.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d) and (
            child.groups == child.in_channels == child.out_channels
            and child.stride == (1,)
        ):
            setattr(module, name, _TritonDepthwiseConv1d(child))
            count += 1
        else:
            count += _swap_depthwise_convs(child)
    return count


def _bake_out_weight_norm(model: nn.Module) -> int:
    """Fold weight_norm reparametrization into the underlying weight tensor.

    weight_norm computes `w = g * v / ||v||` on every forward pass — fine for
    training, wasteful at inference. After calling this the module behaves as a
    plain Conv1d/ConvTranspose1d with a single `weight` tensor.
    Returns the number of modules unwrapped.
    """
    count = 0
    for m in model.modules():
        if _parametrize.is_parametrized(m, "weight"):
            _parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
            count += 1
    return count


class LocalMHA(nn.Module):
    def __init__(self, dim=1024, window_size=32, dim_head=64, use_rotary_pos_emb=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = dim // dim_head
        self.window_size = window_size
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        if use_rotary_pos_emb:
            self.rel_pos = SinusoidalEmbeddings(dim_head, scale_base=window_size // 2)
        else:
            self.rel_pos = None
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        logger.debug("LocalMHA input shape: %s", x.shape)
        B, C, T = x.shape
        residual = x
        x = self.norm(x.transpose(1, 2))
        windows = T // self.window_size
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b (w n) (h d) -> b h w n d", w=windows, h=self.heads), (q, k, v))
        if self.rel_pos is not None:
            pos_emb, scale = self.rel_pos(k)
            q, k = apply_rotary_pos_emb(q, k, pos_emb, scale)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h w n d -> b (w n) (h d)")
        out = self.to_out(out)
        return out.transpose(1, 2) + residual


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim, scale_base=None, use_xpos=False):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # xpos related
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        assert not (use_xpos and scale_base is None), "scale base must be defined if using xpos"
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, x):
        seq_len, device = x.shape[-2], x.device
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)
        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "b ... (r d) -> b ... r d", r=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs, scale=1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    inv_scale = scale**-1
    if scale.ndim == 2:
        scale = scale[-q_len:, :]
    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k


class Encoder(nn.Module):
    def __init__(
        self,
        d_model=64,
        strides=[3, 3, 7, 7],
        depthwise=False,
        attn_window_size=32,
    ):
        super().__init__()
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            layers += [EncoderBlock(output_dim=d_model, stride=stride, groups=groups)]
        if attn_window_size is not None:
            layers += [LocalMHA(dim=d_model, window_size=attn_window_size)]
        groups = d_model if depthwise else 1
        layers += [
            WNConv1d(d_model, d_model, kernel_size=7, padding=3, groups=groups),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        noise=False,
        depthwise=False,
        attn_window_size=32,
        d_out=1,
    ):
        super().__init__()
        if depthwise:
            layers = [
                WNConv1d(input_channel, input_channel, kernel_size=7, padding=3, groups=input_channel),
                WNConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        if attn_window_size is not None:
            layers += [LocalMHA(dim=channels, window_size=attn_window_size)]

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            layers.append(DecoderBlock(input_dim, output_dim, stride, noise, groups=groups))

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1, kernel=7, groups=1):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, output_dim=16, input_dim=None, stride=1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            ResidualUnit(input_dim, dilation=1, groups=groups),
            ResidualUnit(input_dim, dilation=3, groups=groups),
            ResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        noise = torch.randn((B, 1, T), device=x.device, dtype=x.dtype)
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim=16, output_dim=8, stride=1, noise=False, groups=1):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if noise:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                ResidualUnit(output_dim, dilation=1, groups=groups),
                ResidualUnit(output_dim, dilation=3, groups=groups),
                ResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class VectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, stride: int = 1):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.stride = stride

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        if self.stride > 1:
            z = torch.nn.functional.avg_pool1d(z, self.stride, self.stride)

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)
        z_q = z_e + (z_q - z_e).detach()  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        if self.stride > 1:
            z_q = z_q.repeat_interleave(self.stride, dim=-1)

        return z_q, indices

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        vq_strides: List[int] = [1, 1, 1, 1],
    ):
        super().__init__()
        self.n_codebooks = len(vq_strides)
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, codebook_dim, stride) for stride in vq_strides]
        )

    def forward(self, z):
        z_q = 0
        residual = z
        codes = []
        for _, quantizer in enumerate(self.quantizers):
            z_q_i, indices_i = quantizer(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            codes.append(indices_i)

        return z_q, codes

    def from_codes(self, codes: List[torch.Tensor]) -> torch.Tensor:
        if FUSE_COMPOSE_QUANTIZER_TABLE and getattr(self, "_composed", False):
            # composed_embed: (codebook_size, input_dim), bias: (input_dim,)
            z_q = 0.0
            for i in range(self.n_codebooks):
                q = self.quantizers[i]
                # (B, T_i, input_dim) -> (B, input_dim, T_i)
                z_q_i = F.embedding(codes[i], q.composed_weight).transpose(1, 2)
                if q.composed_bias is not None:
                    z_q_i = z_q_i + q.composed_bias.view(1, -1, 1)
                z_q_i = z_q_i.repeat_interleave(q.stride, dim=-1)
                z_q += z_q_i
            return z_q

        z_q = 0.0
        for i in range(self.n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[i])
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q_i = z_q_i.repeat_interleave(self.quantizers[i].stride, dim=-1)
            z_q += z_q_i
        return z_q

    def _compose_quantizer_tables(self):
        """Precompute composed (codebook_size, input_dim) lookup table per quantizer.

        Replaces `codebook(idx) -> transpose -> out_proj_1x1conv -> +bias` with a
        single Embedding lookup. Called once after weights are loaded and any
        weight_norm bake-out has run, so out_proj.weight is the materialized
        Conv1d weight of shape (input_dim, codebook_dim, 1).
        """
        for q in self.quantizers:
            E = q.codebook.weight  # (codebook_size, codebook_dim)
            W = q.out_proj.weight.squeeze(-1)  # (input_dim, codebook_dim)
            composed = E @ W.t()  # (codebook_size, input_dim)
            q.composed_weight = nn.Parameter(composed.contiguous(), requires_grad=False)
            if q.out_proj.bias is not None:
                q.composed_bias = nn.Parameter(q.out_proj.bias.detach().clone(), requires_grad=False)
            else:
                q.composed_bias = None
        self._composed = True


class SNAC(nn.Module):
    def __init__(
        self,
        sampling_rate=44100,
        encoder_dim=64,
        encoder_rates=[3, 3, 7, 7],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[7, 7, 3, 3],
        attn_window_size=32,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[8, 4, 2, 1],
        noise=True,
        depthwise=True,
        enable_torch_compile=False,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )
        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
        )
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            noise,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )

        if enable_torch_compile:
            self.decode = torch.compile(self.decode, mode="max-autotune-no-cudagraphs")

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        lcm = math.lcm(self.vq_strides[0], self.attn_window_size or 1)
        pad_to = self.hop_length * lcm
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def forward(self, audio_data: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data)
        z_q, codes = self.quantizer(z)
        audio_hat = self.decoder(z_q)
        return audio_hat[..., :length], codes

    def encode(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data)
        _, codes = self.quantizer(z)
        return codes

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        z_q = self.quantizer.from_codes(codes)
        audio_hat = self.decoder(z_q)
        return audio_hat

    @classmethod
    def from_config(cls, config_path, enable_torch_compile=False):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["enable_torch_compile"] = enable_torch_compile
        model = cls(**config)
        return model

    @classmethod
    def from_pretrained(cls, repo_id, enable_torch_compile=False, **kwargs):
        from huggingface_hub import hf_hub_download

        if not os.path.isdir(repo_id):
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs)
            model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", **kwargs)
            model = cls.from_config(config_path, enable_torch_compile=enable_torch_compile)
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            model = cls.from_config(os.path.join(repo_id, "config.json"), enable_torch_compile=enable_torch_compile)
            state_dict = torch.load(os.path.join(repo_id, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        if FUSE_REMOVE_WEIGHT_NORM:
            n = _bake_out_weight_norm(model)
            logger.info(f"SNAC: folded weight_norm into {n} module(s)")
        if FUSE_COMPOSE_QUANTIZER_TABLE:
            model.quantizer._compose_quantizer_tables()
            logger.info(f"SNAC: composed quantizer tables for {model.quantizer.n_codebooks} codebooks")
        if FUSE_TRITON_DEPTHWISE_CONV:
            n = _swap_depthwise_convs(model)
            logger.info(f"SNAC: swapped {n} depthwise Conv1d -> Triton kernel")
        return model


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device).to(torch.bfloat16)
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        codes = [
            torch.zeros(bs, 4, device=device, dtype=torch.int32),  # Codebook 1
            torch.zeros(bs, 8, device=device, dtype=torch.int32),  # Codebook 2
            torch.zeros(bs, 16, device=device, dtype=torch.int32),  # Codebook 3
        ]
        for _ in range(5):
            torch.cuda.synchronize()
            tick = time.time()
            audio_hat = model.decode(codes)  # audio_hat: [bs, 1, 8192]
            torch.cuda.synchronize()
            logger.info(f"Batch size {bs} took {(time.time() - tick) * 1000:.3f} ms")
        logger.info("=====")
        # print(f"Output shape: {audio_hat.shape}, Codes: {len(codes)} codebooks")
