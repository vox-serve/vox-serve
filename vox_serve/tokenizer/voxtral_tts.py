"""Audio codec decoder and tokenizer for Voxtral-4B-TTS."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base import DecoderCache

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CODEC_NORM_EPS = 1e-2


@dataclass
class AudioTokenizerArgs:
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36
    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm_eps: float = 1e-6
    qk_norm: bool = True
    use_biases: bool = False
    norm_eps: float = CODEC_NORM_EPS
    layer_scale: bool = True
    layer_scale_init: Optional[float] = 0.01
    decoder_transformer_lengths_str: str = "2,2,2,2"
    decoder_convs_kernels_str: str = "3,4,4,4"
    decoder_convs_strides_str: str = "1,2,2,2"

    @property
    def decoder_transformer_lengths(self):
        return tuple(int(x) for x in self.decoder_transformer_lengths_str.split(","))

    @property
    def decoder_convs_kernels(self):
        return tuple(int(x) for x in self.decoder_convs_kernels_str.split(","))

    @property
    def decoder_convs_strides(self):
        return tuple(int(x) for x in self.decoder_convs_strides_str.split(","))

    @property
    def latent_dim(self):
        return self.semantic_dim + self.acoustic_dim

    @property
    def downsample_factor(self):
        result = self.pretransform_patch_size
        for s in self.decoder_convs_strides:
            result *= s
        return result


# ---------------------------------------------------------------------------
# Decoder cache for accumulated acoustic codes
# ---------------------------------------------------------------------------


@dataclass
class VoxtralDecoderCache(DecoderCache):
    """Stores accumulated acoustic codes from the flow-matching acoustic transformer."""

    acoustic_codes: torch.Tensor  # (batch, max_frames, 36)
    # Use a 1-element tensor so DecoderCache.copy_from and __getitem__ work correctly
    consumed_frames: torch.Tensor = None  # scalar tensor, default 0

    def __post_init__(self):
        if self.consumed_frames is None:
            self.consumed_frames = torch.zeros(1, dtype=torch.long)


# ---------------------------------------------------------------------------
# Quantizer components
# ---------------------------------------------------------------------------


class SemanticCodebook(nn.Module):
    """VQ codebook using Euclidean distance, stored as EMA sums."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.epsilon = 1e-5
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("_embedding", None, persistent=False)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            usage = self.cluster_usage.unsqueeze(1).clamp(min=self.epsilon)
            self._embedding = self.embedding_sum / usage
        return self._embedding

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, 1, T) int -> (B, codebook_dim, T) float"""
        emb = self.embedding  # (codebook_size, codebook_dim)
        # Gather along codebook axis
        flat_codes = codes[:, 0, :]  # (B, T)
        out = F.embedding(flat_codes, emb)  # (B, T, codebook_dim)
        return out.permute(0, 2, 1)  # (B, codebook_dim, T)


class AcousticCodebook(nn.Module):
    """Finite Scalar Quantization for acoustic codebooks."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.dim = codebook_dim
        self.n_levels = codebook_size

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """codes: (B, 36, T) int [0, n_levels-1] -> (B, 36, T) float [-1, 1]"""
        return codes.to(dtype) / (self.n_levels - 1) * 2 - 1


class MistralAudioCodebook(nn.Module):
    """Combined semantic VQ + acoustic FSQ quantizer."""

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.semantic_codebook = SemanticCodebook(args.semantic_codebook_size, args.semantic_dim)
        self.acoustic_codebook = AcousticCodebook(args.acoustic_codebook_size, args.acoustic_dim)
        self.semantic_dim = args.semantic_dim
        self.acoustic_dim = args.acoustic_dim

    @property
    def num_codebooks(self) -> int:
        return 1 + self.acoustic_dim  # 37

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """codes: (B, 37, T) -> (B, 292, T) continuous latent"""
        semantic_codes = codes[:, 0:1, :]  # (B, 1, T)
        acoustic_codes = codes[:, 1:, :]  # (B, 36, T)

        semantic = self.semantic_codebook.decode(semantic_codes).to(dtype)  # (B, 256, T)
        acoustic = self.acoustic_codebook.decode(acoustic_codes, dtype=dtype)  # (B, 36, T)

        return torch.cat([semantic, acoustic], dim=1)  # (B, 292, T)


# ---------------------------------------------------------------------------
# Causal convolution layers
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.pad_mode = pad_mode
        self._stride = stride
        self._effective_kernel_size = (kernel_size - 1) * dilation + 1
        self._padding_total = self._effective_kernel_size - stride

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
        )
        if use_weight_norm:
            self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal padding: all on the left
        padding_left = self._padding_total
        if padding_left > 0:
            x = F.pad(x, (padding_left, 0), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        trim_ratio: float = 1.0,
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.trim_ratio = trim_ratio
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=use_bias,
        )
        if use_weight_norm:
            self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        # Trim right side for causal behavior
        padding_total = self.conv.kernel_size[0] - self.conv.stride[0]
        trim_right = math.ceil(padding_total * self.trim_ratio)
        trim_left = padding_total - trim_right
        if trim_left > 0:
            y = y[:, :, trim_left:]
        if trim_right > 0:
            y = y[:, :, :-trim_right]
        return y


# ---------------------------------------------------------------------------
# Codec transformer components (ALiBi attention)
# ---------------------------------------------------------------------------


def _compute_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for each head."""
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = torch.arange(1, closest_power_of_2 + 1)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != n_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        remaining = n_heads - closest_power_of_2
        extra_powers = torch.arange(1, 2 * remaining + 1, 2)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)])

    return slopes


class CodecRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = CODEC_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(dtype) * self.weight


class CodecFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, use_biases: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=use_biases)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=use_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CodecAttention(nn.Module):
    """Causal sliding-window attention with ALiBi positional bias."""

    def __init__(self, args: AudioTokenizerArgs, layer_id: int):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = args.attn_sliding_window_size

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.use_biases)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

        self.q_norm = CodecRMSNorm(args.n_heads * args.head_dim, eps=args.qk_norm_eps) if args.qk_norm else None
        self.k_norm = CodecRMSNorm(args.n_kv_heads * args.head_dim, eps=args.qk_norm_eps) if args.qk_norm else None

        # ALiBi slopes
        self.register_buffer(
            "alibi_slopes",
            _compute_alibi_slopes(args.n_heads),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        xq = self.wq(x)  # (B, T, n_heads * head_dim)
        xk = self.wk(x)  # (B, T, n_kv_heads * head_dim)
        xv = self.wv(x)

        # Apply QK norm before reshaping to heads
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim)

        # Expand KV heads for GQA
        if self.repeats > 1:
            xk = xk.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).reshape(B, T, self.n_heads, self.head_dim)
            xv = xv.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).reshape(B, T, self.n_heads, self.head_dim)

        # Transpose for attention: (B, n_heads, T, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # ALiBi bias: (n_heads, T, T)
        positions = torch.arange(T, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        alibi = self.alibi_slopes.unsqueeze(1).unsqueeze(2) * rel_pos.unsqueeze(0)  # (n_heads, T, T)

        # Causal mask with sliding window
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        if self.sliding_window is not None and self.sliding_window > 0:
            window_mask = torch.tril(
                torch.full((T, T), float("-inf"), device=x.device),
                diagonal=-(self.sliding_window + 1),
            )
            causal_mask = causal_mask + window_mask

        attn_bias = alibi + causal_mask.unsqueeze(0)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(xq, xk.transpose(-2, -1)) * scale + attn_bias
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
        output = torch.matmul(attn_weights, xv)

        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(output)


class CodecTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AudioTokenizerArgs):
        super().__init__()
        self.attention = CodecAttention(args, layer_id)
        self.feed_forward = CodecFeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = CodecRMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = CodecRMSNorm(args.dim, eps=args.norm_eps)

        self.layer_scale = args.layer_scale
        if args.layer_scale:
            init_val = args.layer_scale_init if args.layer_scale_init is not None else 0.01
            self.attention_scale = nn.Parameter(torch.full((args.dim,), init_val))
            self.ffn_scale = nn.Parameter(torch.full((args.dim,), init_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with optional layer scaling
        h = self.attention(self.attention_norm(x))
        if self.layer_scale:
            h = h * self.attention_scale
        x = x + h

        # FFN with optional layer scaling
        h = self.feed_forward(self.ffn_norm(x))
        if self.layer_scale:
            h = h * self.ffn_scale
        x = x + h
        return x


class CodecTransformer(nn.Module):
    def __init__(self, args: AudioTokenizerArgs, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleDict({str(i): CodecTransformerBlock(i, args) for i in range(n_layers)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, T) -> transpose to (B, T, dim) for transformer -> back
        x = x.transpose(1, 2)  # (B, T, dim)
        for layer in self.layers.values():
            x = layer(x)
        return x.transpose(1, 2)  # (B, dim, T)


# ---------------------------------------------------------------------------
# Codec Decoder
# ---------------------------------------------------------------------------


class VoxtralCodecDecoder(nn.Module):
    """
    Voxtral audio codec decoder.
    Converts quantized codes (37 codebooks) to audio waveform.

    Architecture: alternating CausalConv + Transformer blocks with upsampling.
    """

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.args = args
        self.quantizer = MistralAudioCodebook(args)

        # Build decoder blocks: [Conv, Transformer, ConvTranspose, Transformer, ...] × n_stages
        self.decoder_blocks = nn.ModuleList()
        strides = args.decoder_convs_strides
        kernels = args.decoder_convs_kernels
        lengths = args.decoder_transformer_lengths

        for i in range(len(strides)):
            if i == 0:
                # First conv: project from latent_dim to working dim
                in_ch = args.latent_dim  # 292
            else:
                in_ch = args.dim  # 1024

            if strides[i] == 1:
                # Regular conv (no upsampling)
                self.decoder_blocks.append(
                    CausalConv1d(in_ch, args.dim, kernels[i], stride=1, use_weight_norm=args.conv_weight_norm)
                )
            else:
                # Transposed conv for upsampling
                self.decoder_blocks.append(
                    CausalConvTranspose1d(
                        in_ch, args.dim, kernels[i], stride=strides[i], use_weight_norm=args.conv_weight_norm
                    )
                )

            # Transformer layers after each conv
            self.decoder_blocks.append(CodecTransformer(args, n_layers=lengths[i]))

        # Output projection: dim -> patch_size (channels)
        self.output_proj = CausalConv1d(
            args.dim,
            args.pretransform_patch_size,
            args.patch_proj_kernel_size,
            stride=1,
            use_weight_norm=args.conv_weight_norm,
        )

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Decode audio tokens to waveform.

        Args:
            codes: (B, 37, T) integer codes
        Returns:
            audio: (B, 1, T * downsample_factor) waveform
        """
        # Dequantize codes to continuous latent
        latent = self.quantizer.decode(codes, dtype=dtype)  # (B, 292, T)

        # Cast to model dtype for decoder blocks (which may be bf16)
        model_dtype = next(self.decoder_blocks[0].parameters()).dtype
        x = latent.to(model_dtype)
        for block in self.decoder_blocks:
            x = block(x)

        # Output projection: (B, dim, T_up) -> (B, patch_size, T_up)
        x = self.output_proj(x)

        # Reshape to waveform: (B, patch_size, T_up) -> (B, 1, patch_size * T_up)
        B, C, T_up = x.shape
        x = x.permute(0, 2, 1).reshape(B, 1, T_up * C).to(dtype)

        return x
