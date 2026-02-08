"""
Qwen3 TTS Audio Codec implementation.

This could be based on an existing codec (SoundStream, EnCodec, DAC, etc.)
or a custom codec specific to Qwen3 TTS.
"""

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
except ImportError:
    # Fallback for older transformers versions
    ROPE_INIT_FUNCTIONS = None

from dataclasses import field

from transformers import MimiConfig, MimiModel

from .base import DecoderCache


@dataclass
class Qwen3TTSDecoderCache(DecoderCache):
    """Cache for Qwen3 TTS decoder streaming inference.

    This cache stores:
    1. Attention KV cache for transformer layers (strict sliding window of 72 tokens)
    2. Causal conv activation caches for all convolution layers
    3. Pre-allocated work buffers for CUDA graph compatibility
    """

    # Attention KV cache for transformer layers
    # Shape: [batch, num_layers, num_heads, cache_len, 2 * head_dim]
    # Keys and values are concatenated in the last dimension
    attention_cache: Optional[torch.Tensor] = None

    # Position offset for RoPE (tracks cumulative processed tokens)
    # Stored as a 1-element tensor for in-place update compatibility
    position_offset: Optional[torch.Tensor] = None

    # pre_conv cache: [batch, channels, kernel_size-1]
    pre_conv_cache: Optional[torch.Tensor] = None

    # ConvNeXt caches for upsample blocks (list of 2)
    # Each: [batch, channels, kernel_size-1]
    upsample_conv_caches: Optional[List[torch.Tensor]] = None

    # Decoder conv caches - flat list structure
    # Index 0: initial conv cache
    # Index 1-12: 4 blocks × 3 residual units (only conv1 of each residual unit)
    # Index 13: final conv cache
    decoder_conv_caches: Optional[List[torch.Tensor]] = None

    # Work buffers for CUDA graph compatibility (avoid torch.cat allocations)
    # Each buffer: [batch, channels, padding + input_length]
    pre_conv_work_buffer: Optional[torch.Tensor] = None
    upsample_work_buffers: Optional[List[torch.Tensor]] = None
    decoder_work_buffers: Optional[List[torch.Tensor]] = None

    # Output buffers for CUDA graph compatibility (avoid conv output allocations)
    # Each buffer: [batch, out_channels, output_length]
    pre_conv_output_buffer: Optional[torch.Tensor] = None
    upsample_output_buffers: Optional[List[torch.Tensor]] = None
    decoder_output_buffers: Optional[List[torch.Tensor]] = None

    # TransConvNet caches for streaming - stores last input sample at each stage
    # Each: [batch, in_channels, 1]
    # 4 caches for decoder TransConvNet stages (rates 8, 5, 4, 3)
    transconv_caches: Optional[List[torch.Tensor]] = None

    # TransConvNet work buffers - pre-allocated for context + input
    # Each: [batch, in_channels, 1 + input_length]
    transconv_work_buffers: Optional[List[torch.Tensor]] = None


@dataclass
class Qwen3TTSTokenizerV2DecoderConfig:
    attention_bias: bool = False
    attention_dropout: float = 0.0
    latent_dim: int = 1024
    codebook_dim: int = 512
    codebook_size: int = 2048
    decoder_dim: int = 1536
    hidden_act: str = "silu"
    hidden_size: int = 512
    intermediate_size: int = 1024
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    head_dim: int = 64
    num_attention_heads: int = 16
    num_hidden_layers: int = 8
    num_key_value_heads: int = 16
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: int = 10000
    semantic_codebook_size: int = 4096
    sliding_window: int = 72
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 4, 3])
    upsampling_ratios: List[int] = field(default_factory=lambda: [2, 2])
    vector_quantization_hidden_dimension: int = 512


@dataclass
class Qwen3TTSTokenizerV2EncoderConfig:
    _frame_rate: float = 12.5
    attention_bias: bool = False
    attention_dropout: float = 0.0
    audio_channels: int = 1
    codebook_dim: int = 256
    codebook_size: int = 2048
    compress: int = 2
    dilation_growth_rate: int = 2
    dtype: str = "float32"
    head_dim: int = 64
    hidden_size: int = 512
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    intermediate_size: int = 2048
    kernel_size: int = 7
    last_kernel_size: int = 3
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    norm_eps: float = 1e-5
    normalize: bool = False
    num_attention_heads: int = 8
    num_filters: int = 64
    num_hidden_layers: int = 8
    num_key_value_heads: int = 8
    num_quantizers: int = 32
    num_residual_layers: int = 1
    num_semantic_quantizers: int = 1
    pad_mode: str = "constant"
    residual_kernel_size: int = 3
    rope_theta: float = 10000.0
    sampling_rate: int = 24000
    sliding_window: int = 250
    transformers_version: str = "4.57.0.dev0"
    trim_right_ratio: float = 1.0
    upsample_groups: int = 512
    upsampling_ratios: List[int] = field(default_factory=lambda: [8, 6, 5, 4])
    use_cache: bool = False
    use_causal_conv: bool = True
    use_conv_shortcut: bool = False
    use_streaming: bool = False
    vector_quantization_hidden_dimension: int = 256


@dataclass
class Qwen3TTSTokenizerV2Config:
    architectures: List[str] = field(default_factory=lambda: [
        "Qwen3TTSTokenizerV2Model"
    ])
    model_type: str = "qwen3_tts_tokenizer_12hz"

    encoder_valid_num_quantizers: int = 16
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    decode_upsample_rate: int = 1920
    encode_downsample_rate: int = 1920

    decoder_config: Qwen3TTSTokenizerV2DecoderConfig = field(
        default_factory=Qwen3TTSTokenizerV2DecoderConfig
    )
    encoder_config: Qwen3TTSTokenizerV2EncoderConfig = field(
        default_factory=Qwen3TTSTokenizerV2EncoderConfig
    )

    transformers_version: str = "4.57.3"

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        # Convert nested dicts to config dataclasses (JSON gives plain dicts)
        if "decoder_config" in config_dict and isinstance(
            config_dict["decoder_config"], dict
        ):
            config_dict = dict(config_dict)
            config_dict["decoder_config"] = Qwen3TTSTokenizerV2DecoderConfig(
                **config_dict["decoder_config"]
            )
        if "encoder_config" in config_dict and isinstance(
            config_dict["encoder_config"], dict
        ):
            config_dict = dict(config_dict)
            config_dict["encoder_config"] = Qwen3TTSTokenizerV2EncoderConfig(
                **config_dict["encoder_config"]
            )
        return cls(**config_dict)



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3TTSTokenizerV2CausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, hidden_state):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state, (self.padding, extra_padding), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()

    def forward_chunk(
        self,
        hidden_state: torch.Tensor,
        conv_cache: Optional[torch.Tensor] = None,
        work_buffer: Optional[torch.Tensor] = None,
        output_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with caching for streaming inference.

        Uses in-place operations and pre-allocated buffers for CUDA graph compatibility.

        Args:
            hidden_state: [batch, channels, new_length]
            conv_cache: [batch, channels, padding] - pre-allocated buffer
            work_buffer: [batch, channels, padding + max_input_length] - pre-allocated work buffer
                        If None, will create one (not CUDA graph compatible)
            output_buffer: [batch, out_channels, max_output_length] - pre-allocated output buffer
                          If None, will allocate new tensor (not CUDA graph compatible)

        Returns:
            output: [batch, out_channels, new_length] (same temporal length as input)
            conv_cache: Same buffer, updated in-place
        """
        batch_size = hidden_state.shape[0]
        in_channels = hidden_state.shape[1]
        new_length = hidden_state.shape[2]

        # Initialize cache with zeros if not provided
        if conv_cache is None:
            conv_cache = hidden_state.new_zeros(batch_size, in_channels, self.padding)

        if self.padding > 0:
            total_length = self.padding + new_length

            if work_buffer is not None:
                # Use pre-allocated work buffer (CUDA graph compatible)
                # Copy cache and input into work buffer
                work_buffer[:batch_size, :, :self.padding].copy_(conv_cache)
                work_buffer[:batch_size, :, self.padding:total_length].copy_(hidden_state)
                hidden_with_cache = work_buffer[:batch_size, :, :total_length]
            else:
                # Fallback to torch.cat (not CUDA graph compatible)
                hidden_with_cache = torch.cat([conv_cache, hidden_state], dim=2)

            # Update cache IN-PLACE with the last `padding` elements
            if new_length >= self.padding:
                conv_cache.copy_(hidden_state[:, :, -self.padding:])
            else:
                keep_len = self.padding - new_length
                conv_cache[:, :, :keep_len].copy_(conv_cache[:, :, -keep_len:].clone())
                conv_cache[:, :, keep_len:].copy_(hidden_state)
        else:
            hidden_with_cache = hidden_state

        # Apply convolution
        conv_output = self.conv(hidden_with_cache)

        # Copy to output buffer if provided (CUDA graph compatible)
        if output_buffer is not None:
            output_len = conv_output.shape[2]
            output_buffer[:batch_size, :, :output_len].copy_(conv_output)
            output = output_buffer[:batch_size, :, :output_len]
        else:
            output = conv_output

        # Return the same cache buffer (updated in-place)
        return output, conv_cache


class Qwen3TTSTokenizerV2CausalTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.stride = stride

        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = pad = self.left_pad

    def forward(self, hidden_state):
        """Batch forward - trims both left and right padding."""
        hidden_state = self.conv(hidden_state)
        hidden_state = hidden_state[..., self.left_pad : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()

    def forward_chunk(
        self,
        hidden_state: torch.Tensor,
        cache: torch.Tensor,
        work_buffer: torch.Tensor,
    ) -> torch.Tensor:
        """Streaming forward with input caching for proper boundary blending.

        This method caches the last input sample and prepends it to the next chunk.
        This allows proper blending at the left edge of each chunk. The right edge
        is trimmed as usual (no future context available in streaming).

        Args:
            hidden_state: [batch, in_channels, length] - new input samples
            cache: [batch, in_channels, 1] - last input sample from previous chunk
                   (zeros for first chunk, updated in-place)
            work_buffer: [batch, in_channels, 1 + length] - pre-allocated buffer

        Returns:
            output: [batch, out_channels, length * stride] - exactly stride samples per input
        """
        length = hidden_state.shape[-1]

        # Populate work buffer: [cache, new_input]
        work_buffer[:, :, 0:1].copy_(cache)
        work_buffer[:, :, 1:1 + length].copy_(hidden_state)

        # ConvTranspose1d on combined input (length + 1 samples)
        # Output shape: [batch, out_channels, length * stride + kernel]
        raw_output = self.conv(work_buffer[:, :, :1 + length])

        # Update cache in-place with last input sample (for next chunk)
        cache.copy_(hidden_state[:, :, -1:])

        # Trim: left stride (context-only contribution), right (kernel - stride)
        # Output shape: [batch, out_channels, length * stride]
        output = raw_output[:, :, self.stride : self.stride + length * self.stride]

        return output.contiguous()


class Qwen3TTSTokenizerV2ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Qwen3TTSTokenizerV2CausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states):
        input = hidden_states

        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)

        hidden_states = self.gamma * hidden_states

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = input + hidden_states

        return hidden_states

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        conv_cache: Optional[torch.Tensor] = None,
        work_buffer: Optional[torch.Tensor] = None,
        output_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with caching for streaming inference.

        Args:
            hidden_states: [batch, channels, length]
            conv_cache: [batch, channels, padding] for dwconv or None
            work_buffer: [batch, channels, padding + length] for CUDA graph compatibility
            output_buffer: [batch, channels, length] for dwconv output (CUDA graph compatible)

        Returns:
            output: [batch, channels, length]
            conv_cache: Same buffer, updated in-place
        """
        residual = hidden_states

        hidden_states, _ = self.dwconv.forward_chunk(hidden_states, conv_cache, work_buffer, output_buffer)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)

        hidden_states = self.gamma * hidden_states

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = residual + hidden_states

        return hidden_states, conv_cache


class Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # Use ROPE_INIT_FUNCTIONS from transformers if available, otherwise fallback
        if ROPE_INIT_FUNCTIONS is not None:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        else:
            # Fallback implementation for older transformers
            inv_freq = 1.0 / (
                config.rope_theta
                ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device) / config.head_dim)
            )
            self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3TTSTokenizerV2DecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        # attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            # attn_mask=attention_mask,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with KV caching for streaming inference.

        Uses in-place operations for CUDA graph compatibility and memory efficiency.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) from rotary embedding
            kv_cache: [batch, num_heads, sliding_window, 2*head_dim] - pre-allocated buffer

        Returns:
            attn_output: [batch, seq_len, hidden_size]
            kv_cache: Same buffer, updated in-place
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads

        # Project Q, K, V
        query_states = self.q_norm(self.q_proj(hidden_states))
        query_states = query_states.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_norm(self.k_proj(hidden_states))
        key_states = key_states.view(batch_size, seq_len, num_kv_heads, self.head_dim).transpose(1, 2)

        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(batch_size, seq_len, num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to new Q, K
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            # kv_cache: [batch, num_kv_heads, sliding_window, 2*head_dim]
            # Update cache IN-PLACE for CUDA graph compatibility
            # Shift left by seq_len and copy new values to the end
            if seq_len < self.sliding_window:
                # Shift existing cache left
                kv_cache[:, :, :-seq_len, :].copy_(kv_cache[:, :, seq_len:, :])
                # Copy new keys and values to the end
                kv_cache[:, :, -seq_len:, :self.head_dim].copy_(key_states)
                kv_cache[:, :, -seq_len:, self.head_dim:].copy_(value_states)
            else:
                # seq_len >= sliding_window: just use the last sliding_window tokens
                kv_cache[:, :, :, :self.head_dim].copy_(key_states[:, :, -self.sliding_window:, :])
                kv_cache[:, :, :, self.head_dim:].copy_(value_states[:, :, -self.sliding_window:, :])

            # Read back full K/V from cache for attention
            full_key_states = kv_cache[:, :, :, :self.head_dim]
            full_value_states = kv_cache[:, :, :, self.head_dim:]
        else:
            # No cache - just use current K/V
            full_key_states = key_states
            full_value_states = value_states

        # Expand KV for GQA if needed
        if self.num_key_value_groups > 1:
            full_key_states = full_key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            full_value_states = full_value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention with causal mask
        kv_len = full_key_states.shape[2]
        attn_output = F.scaled_dot_product_attention(
            query_states,
            full_key_states,
            full_value_states,
            is_causal=(kv_cache is None),  # Only use built-in causal for first chunk
            attn_mask=None if kv_cache is None else self._create_chunk_causal_mask(
                seq_len, kv_len, hidden_states.device, hidden_states.dtype
            ),
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        # Return the same cache buffer (updated in-place)
        return attn_output, kv_cache

    def _create_chunk_causal_mask(
        self,
        query_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create causal attention mask for chunked inference.

        Each query position i can attend to KV positions 0 to (kv_len - query_len + i).
        """
        # Query positions: 0 to query_len-1
        # KV positions: 0 to kv_len-1
        # Query position i corresponds to absolute position (kv_len - query_len + i)
        # It can attend to KV positions 0 to (kv_len - query_len + i)
        q_positions = torch.arange(query_len, device=device)
        kv_positions = torch.arange(kv_len, device=device)

        # Absolute position of each query
        abs_q_positions = kv_len - query_len + q_positions

        # Create mask: query i can attend to kv j if j <= abs_q_positions[i]
        mask = kv_positions.unsqueeze(0) <= abs_q_positions.unsqueeze(1)

        # Convert to float mask for scaled_dot_product_attention
        attn_mask = torch.zeros(query_len, kv_len, dtype=dtype, device=device)
        attn_mask.masked_fill_(~mask, float("-inf"))

        return attn_mask


class Qwen3TTSTokenizerV2DecoderMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3TTSTokenizerV2DecoderRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3TTSTokenizerV2DecoderRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3TTSTokenizerV2DecoderLayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://huggingface.co/papers/2103.17239).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.
    """

    def __init__(self, config):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.scale * x


class Qwen3TTSTokenizerV2DecoderTransformerLayer(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSTokenizerV2DecoderAttention(config, layer_idx)
        self.mlp = Qwen3TTSTokenizerV2DecoderMlp(config)
        self.input_layernorm = Qwen3TTSTokenizerV2DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSTokenizerV2DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = Qwen3TTSTokenizerV2DecoderLayerScale(config)
        self.mlp_layer_scale = Qwen3TTSTokenizerV2DecoderLayerScale(config)
        self.attention_type = "sliding_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            # attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return hidden_states

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with KV caching for streaming inference.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) from rotary embedding
            kv_cache: [batch, num_heads, cache_len, 2*head_dim] or None

        Returns:
            output: [batch, seq_len, hidden_size]
            new_kv_cache: [batch, num_heads, new_cache_len, 2*head_dim]
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention with KV caching
        hidden_states, new_kv_cache = self.self_attn.forward_chunk(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        # Fully Connected (no caching needed)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return hidden_states, new_kv_cache


class Qwen3TTSTokenizerV2DecoderTransformerModel(nn.Module):
    _can_record_outputs = {
        "hidden_states": Qwen3TTSTokenizerV2DecoderTransformerLayer,
        "attentions": Qwen3TTSTokenizerV2DecoderAttention,
    }

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Qwen3TTSTokenizerV2DecoderTransformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3TTSTokenizerV2DecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(config=config)
        self.gradient_checkpointing = False
        # self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.window_size = config.sliding_window

        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is not None:
            raise ValueError("input_ids is not expected")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.input_proj(inputs_embeds)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            # # Create the masks
            # causal_mask_mapping = {
            #     "full_attention": create_causal_mask(**mask_kwargs),
            # }
            # # The sliding window alternating layers are not always activated depending on the config
            # if self.has_sliding_layers:
            #     causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                # attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        return hidden_states

    def forward_chunk(
        self,
        inputs_embeds: torch.Tensor,
        position_offset: Union[int, torch.Tensor] = 0,
        attention_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with KV caching for streaming inference.

        Args:
            inputs_embeds: [batch, seq_len, latent_dim]
            position_offset: Starting position index for RoPE (int or 1-element tensor)
            attention_cache: [batch, num_layers, num_heads, cache_len, 2*head_dim] or None

        Returns:
            output: [batch, seq_len, latent_dim]
            new_attention_cache: [batch, num_layers, num_heads, new_cache_len, 2*head_dim]
            new_position_offset: Updated position offset as 1-element tensor
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        num_layers = self.config.num_hidden_layers

        inputs_embeds = self.input_proj(inputs_embeds)

        # Ensure position_offset is a tensor for CUDA graph compatibility
        if not torch.is_tensor(position_offset):
            position_offset = torch.tensor([position_offset], dtype=torch.long, device=inputs_embeds.device)

        # Create position IDs with offset using tensor operations (CUDA graph compatible)
        # Create base range [0, 1, 2, ..., seq_len-1] and add offset
        base_positions = torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long)
        position_ids = (base_positions + position_offset).unsqueeze(0).expand(batch_size, -1)

        hidden_states = inputs_embeds

        # Create position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Process through layers with per-layer KV cache
        # The per-layer caches are views into attention_cache and are updated in-place
        for i, decoder_layer in enumerate(self.layers[:num_layers]):
            # Extract per-layer cache (this is a view, updates happen in-place)
            layer_kv_cache = None
            if attention_cache is not None:
                layer_kv_cache = attention_cache[:, i]

            hidden_states, _ = decoder_layer.forward_chunk(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                kv_cache=layer_kv_cache,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)

        # Update position offset using tensor operations (CUDA graph compatible)
        new_position_offset = position_offset + seq_len

        # Return original attention_cache (updated in-place)
        return hidden_states, attention_cache, new_position_offset


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin,
          Tilman Hartwig, Masahito Ueda: https://huggingface.co/papers/2006.08195
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )

        return hidden_states


class Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()

        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3TTSTokenizerV2CausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3TTSTokenizerV2CausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual

    def forward_chunk(
        self,
        hidden_state: torch.Tensor,
        conv_cache: Optional[torch.Tensor] = None,
        work_buffer: Optional[torch.Tensor] = None,
        output_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with caching for streaming inference.

        Only conv1 needs caching (kernel_size=7 with dilation).
        conv2 has kernel_size=1, so no cache needed.

        Args:
            hidden_state: [batch, channels, length]
            conv_cache: [batch, channels, padding] for conv1 or None
            work_buffer: [batch, channels, padding + length] for CUDA graph compatibility
            output_buffer: [batch, channels, length] for conv1 output (CUDA graph compatible)

        Returns:
            output: [batch, channels, length]
            conv_cache: Same buffer, updated in-place
        """
        residual = hidden_state

        hidden_state = self.act1(hidden_state)
        hidden_state, _ = self.conv1.forward_chunk(hidden_state, conv_cache, work_buffer, output_buffer)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)  # kernel_size=1, no cache needed

        return hidden_state + residual, conv_cache


class Qwen3TTSTokenizerV2DecoderDecoderBlock(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig, layer_idx):
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]

        block = [
            SnakeBeta(in_dim),
            Qwen3TTSTokenizerV2CausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ]

        for dilation in (1, 3, 9):
            block.append(Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(out_dim, dilation))

        self.block = nn.ModuleList(block)

    def forward(self, hidden):
        for block in self.block:
            hidden = block(hidden)
        return hidden

    def forward_chunk(
        self,
        hidden: torch.Tensor,
        conv_caches: Optional[List[torch.Tensor]] = None,
        work_buffers: Optional[List[torch.Tensor]] = None,
        output_buffers: Optional[List[torch.Tensor]] = None,
        transconv_cache: Optional[torch.Tensor] = None,
        transconv_work_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward with caching for streaming inference.

        Args:
            hidden: [batch, channels, length]
            conv_caches: List of 3 conv caches for the 3 ResidualUnits or None
            work_buffers: List of 3 work buffers for CUDA graph compatibility
            output_buffers: List of 3 output buffers for CUDA graph compatibility
            transconv_cache: [batch, in_channels, 1] - TransConvNet input cache
            transconv_work_buffer: [batch, in_channels, 1 + length] - TransConvNet work buffer

        Returns:
            output: [batch, out_channels, upsampled_length]
            conv_caches: Same list of caches, updated in-place
        """
        if conv_caches is None:
            conv_caches = [None, None, None]
        if work_buffers is None:
            work_buffers = [None, None, None]
        if output_buffers is None:
            output_buffers = [None, None, None]

        cache_idx = 0

        for block in self.block:
            if isinstance(block, Qwen3TTSTokenizerV2DecoderDecoderResidualUnit):
                hidden, _ = block.forward_chunk(
                    hidden,
                    conv_caches[cache_idx],
                    work_buffers[cache_idx],
                    output_buffers[cache_idx],
                )
                cache_idx += 1
            elif isinstance(block, Qwen3TTSTokenizerV2CausalTransConvNet):
                # CausalTransConvNet with input caching for proper boundary blending
                hidden = block.forward_chunk(hidden, transconv_cache, transconv_work_buffer)
            else:
                # SnakeBeta - no caching
                hidden = block(hidden)

        return hidden, conv_caches


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon

        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        quantized = F.embedding(codes, embedding)
        return quantized


class VectorQuantization(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim

        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            epsilon=epsilon
        )
        self.codebook_size = codebook_size

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = quantized.transpose(1, 2)
        return quantized


class ResidualVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros([1], device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            assert isinstance(layer, VectorQuantization)
            quantized = quantized + layer.decode(layer_codes)
        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dimension: int = 128,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        n_q: int = 8,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        bins: int = 1024,
        decay: float = 0.99,
        force_projection: bool = False,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        self.decay = decay
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False
            )
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False
            )
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        quantized = self.output_proj(quantized)
        return quantized


class SplitResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with separate projections for the first quantizer and the rest.

    Args:
        n_q (int): Number of residual vector quantizers used.
        n_semantic_q (int): Number of residual vector quantizers used for the semantic quantizer.
        **kwargs: Arguments to the constructor of `ResidualVectorQuantizer` that are shared between both.
    """

    def __init__(
        self,
        *,
        n_q: int = 8,
        n_q_semantic: int = 1,
        **kwargs,
    ):
        super().__init__()
        assert n_q > n_q_semantic, (
            f"Number of quantizers {n_q} must be larger "
            f"than the number of semantic quantizers {n_q_semantic}."
        )
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        q_dropout = kwargs.pop("q_dropout", False)
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks.
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized


class Qwen3TTSTokenizerV2Decoder(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig):
        super().__init__()
        self.config = config
        self.total_upsample = np.prod(config.upsample_rates + config.upsampling_ratios)
        self.pre_transformer = Qwen3TTSTokenizerV2DecoderTransformerModel(config)

        self.quantizer = SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=1,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )

        self.pre_conv = Qwen3TTSTokenizerV2CausalConvNet(
            config.codebook_dim,
            config.latent_dim,
            kernel_size=3,
        )

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3TTSTokenizerV2CausalTransConvNet(config.latent_dim, config.latent_dim, factor, factor),
                        Qwen3TTSTokenizerV2ConvNeXtBlock(config.latent_dim),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3TTSTokenizerV2CausalConvNet(config.latent_dim, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3TTSTokenizerV2DecoderDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3TTSTokenizerV2CausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes):
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}")

        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)

        hidden = self.pre_transformer(inputs_embeds=hidden)
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    @torch.no_grad()
    def init_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        detokenize_interval: int = 10,
    ) -> "Qwen3TTSDecoderCache":
        """Initialize empty cache for streaming inference.

        Args:
            batch_size: Number of items in batch
            device: Device to create tensors on
            dtype: Data type for tensors
            detokenize_interval: Number of tokens processed per chunk (for work buffer sizing)

        Returns:
            Qwen3TTSDecoderCache with all fields initialized to zeros
        """
        # Track sequence length at each stage for buffer allocation
        seq_len = detokenize_interval

        # Pre-conv cache, work buffer, and output buffer
        pre_conv_cache = torch.zeros(
            batch_size, self.config.codebook_dim, self.pre_conv.padding,
            device=device, dtype=dtype
        )
        pre_conv_work_buffer = torch.zeros(
            batch_size, self.config.codebook_dim, self.pre_conv.padding + seq_len,
            device=device, dtype=dtype
        )
        pre_conv_output_buffer = torch.zeros(
            batch_size, self.pre_conv.conv.out_channels, seq_len,
            device=device, dtype=dtype
        )

        # Upsample ConvNeXt caches, work buffers, and output buffers
        upsample_conv_caches = []
        upsample_work_buffers = []
        upsample_output_buffers = []
        for blocks in self.upsample:
            for block in blocks:
                if isinstance(block, Qwen3TTSTokenizerV2CausalTransConvNet):
                    # TransConvNet upsamples the sequence
                    stride = block.conv.stride[0] if isinstance(block.conv.stride, tuple) else block.conv.stride
                    seq_len = seq_len * stride
                elif isinstance(block, Qwen3TTSTokenizerV2ConvNeXtBlock):
                    upsample_conv_caches.append(
                        torch.zeros(
                            batch_size, self.config.latent_dim, block.dwconv.padding,
                            device=device, dtype=dtype
                        )
                    )
                    upsample_work_buffers.append(
                        torch.zeros(
                            batch_size, self.config.latent_dim, block.dwconv.padding + seq_len,
                            device=device, dtype=dtype
                        )
                    )
                    # ConvNeXt output has same shape as input (residual connection)
                    upsample_output_buffers.append(
                        torch.zeros(
                            batch_size, self.config.latent_dim, seq_len,
                            device=device, dtype=dtype
                        )
                    )

        # Decoder conv caches, work buffers, and output buffers
        decoder_conv_caches = []
        decoder_work_buffers = []
        decoder_output_buffers = []

        # TransConvNet caches and work buffers (4 stages for rates 8, 5, 4, 3)
        # Each cache stores the last input sample: [batch, in_channels, 1]
        # Each work buffer holds context + input: [batch, in_channels, 1 + seq_len]
        transconv_caches = []
        transconv_work_buffers = []

        for block in self.decoder:
            if isinstance(block, Qwen3TTSTokenizerV2CausalConvNet):
                # Initial conv or final conv
                in_channels = block.conv.in_channels
                out_channels = block.conv.out_channels
                decoder_conv_caches.append(
                    torch.zeros(batch_size, in_channels, block.padding, device=device, dtype=dtype)
                )
                decoder_work_buffers.append(
                    torch.zeros(batch_size, in_channels, block.padding + seq_len, device=device, dtype=dtype)
                )
                decoder_output_buffers.append(
                    torch.zeros(batch_size, out_channels, seq_len, device=device, dtype=dtype)
                )
            elif isinstance(block, Qwen3TTSTokenizerV2DecoderDecoderBlock):
                # Each DecoderBlock has ResidualUnits and possibly TransConvNet for upsampling
                for res_unit in block.block:
                    if isinstance(res_unit, Qwen3TTSTokenizerV2CausalTransConvNet):
                        # Allocate TransConvNet cache and work buffer BEFORE upsampling
                        in_channels = res_unit.conv.in_channels
                        transconv_caches.append(
                            torch.zeros(batch_size, in_channels, 1, device=device, dtype=dtype)
                        )
                        transconv_work_buffers.append(
                            torch.zeros(batch_size, in_channels, 1 + seq_len, device=device, dtype=dtype)
                        )
                        # Now upsample: output length = input * stride
                        s = res_unit.conv.stride
                        stride = s[0] if isinstance(s, tuple) else s
                        seq_len = seq_len * stride
                    elif isinstance(res_unit, Qwen3TTSTokenizerV2DecoderDecoderResidualUnit):
                        # Only conv1 needs cache (conv2 has kernel_size=1)
                        in_channels = res_unit.conv1.conv.in_channels
                        decoder_conv_caches.append(
                            torch.zeros(batch_size, in_channels, res_unit.conv1.padding, device=device, dtype=dtype)
                        )
                        decoder_work_buffers.append(
                            torch.zeros(
                                batch_size,
                                in_channels,
                                res_unit.conv1.padding + seq_len,
                                device=device,
                                dtype=dtype,
                            )
                        )
                        # ResidualUnit output has same shape as input (residual connection)
                        decoder_output_buffers.append(
                            torch.zeros(batch_size, in_channels, seq_len, device=device, dtype=dtype)
                        )

        # Pre-allocate attention cache with zeros
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_key_value_heads
        head_dim = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads
        )
        sliding_window = self.config.sliding_window

        attention_cache = torch.zeros(
            batch_size, num_layers, num_heads, sliding_window, 2 * head_dim,
            device=device, dtype=dtype
        )

        # Position offset as 1-element tensor for in-place updates
        position_offset = torch.zeros(1, dtype=torch.long, device=device)

        return Qwen3TTSDecoderCache(
            attention_cache=attention_cache,
            position_offset=position_offset,
            pre_conv_cache=pre_conv_cache,
            upsample_conv_caches=upsample_conv_caches,
            decoder_conv_caches=decoder_conv_caches,
            pre_conv_work_buffer=pre_conv_work_buffer,
            upsample_work_buffers=upsample_work_buffers,
            decoder_work_buffers=decoder_work_buffers,
            pre_conv_output_buffer=pre_conv_output_buffer,
            upsample_output_buffers=upsample_output_buffers,
            decoder_output_buffers=decoder_output_buffers,
            transconv_caches=transconv_caches,
            transconv_work_buffers=transconv_work_buffers,
        )

    def forward_chunk(
        self,
        codes: torch.Tensor,
        decoder_cache: Optional["Qwen3TTSDecoderCache"] = None,
    ) -> Tuple[torch.Tensor, "Qwen3TTSDecoderCache"]:
        """Forward with caching for streaming inference.

        Args:
            codes: [batch, num_quantizers, chunk_length]
            decoder_cache: Previous cache or None for first chunk

        Returns:
            wav: [batch, 1, audio_length]
            new_cache: Updated cache for next chunk
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}")

        batch_size = codes.shape[0]
        device = codes.device

        # Initialize cache if not provided
        if decoder_cache is None:
            decoder_cache = self.init_cache(batch_size, device, torch.float32)

        # Quantizer decode (no caching needed)
        hidden = self.quantizer.decode(codes)

        # Pre-conv with caching, work buffer, and output buffer
        hidden, _ = self.pre_conv.forward_chunk(
            hidden,
            decoder_cache.pre_conv_cache,
            decoder_cache.pre_conv_work_buffer,
            decoder_cache.pre_conv_output_buffer,
        )
        hidden = hidden.transpose(1, 2)

        # Transformer with attention caching (in-place updates)
        hidden, _, new_position_offset = self.pre_transformer.forward_chunk(
            inputs_embeds=hidden,
            position_offset=decoder_cache.position_offset,
            attention_cache=decoder_cache.attention_cache,
        )
        # Update position offset in-place
        decoder_cache.position_offset.copy_(new_position_offset)
        hidden = hidden.permute(0, 2, 1)

        # Upsample blocks with caching
        upsample_cache_idx = 0
        for blocks in self.upsample:
            for block in blocks:
                if isinstance(block, Qwen3TTSTokenizerV2ConvNeXtBlock):
                    caches = decoder_cache.upsample_conv_caches
                    cache = caches[upsample_cache_idx] if caches else None
                    work_bufs = decoder_cache.upsample_work_buffers
                    work_buf = work_bufs[upsample_cache_idx] if work_bufs else None
                    out_bufs = decoder_cache.upsample_output_buffers
                    output_buf = out_bufs[upsample_cache_idx] if out_bufs else None
                    hidden, _ = block.forward_chunk(hidden, cache, work_buf, output_buf)
                    upsample_cache_idx += 1
                elif isinstance(block, Qwen3TTSTokenizerV2CausalTransConvNet):
                    # Upsample TransConvNet has kernel_size == stride, so no trimming needed
                    # Use regular forward (no caching required)
                    hidden = block(hidden)
                else:
                    # Other blocks - no caching needed
                    hidden = block(hidden)

        # Decoder blocks with caching
        wav = hidden
        decoder_cache_idx = 0
        transconv_cache_idx = 0

        for block in self.decoder:
            if isinstance(block, Qwen3TTSTokenizerV2CausalConvNet):
                # Initial conv or final conv
                conv_caches = decoder_cache.decoder_conv_caches
                cache = conv_caches[decoder_cache_idx] if conv_caches else None
                work_bufs = decoder_cache.decoder_work_buffers
                work_buf = work_bufs[decoder_cache_idx] if work_bufs else None
                out_bufs = decoder_cache.decoder_output_buffers
                output_buf = out_bufs[decoder_cache_idx] if out_bufs else None
                wav, _ = block.forward_chunk(wav, cache, work_buf, output_buf)
                decoder_cache_idx += 1
            elif isinstance(block, Qwen3TTSTokenizerV2DecoderDecoderBlock):
                # DecoderBlock with 3 ResidualUnit caches + 1 TransConvNet cache
                block_caches = []
                block_work_bufs = []
                block_output_bufs = []
                for _ in range(3):  # 3 ResidualUnits per block
                    if decoder_cache.decoder_conv_caches and decoder_cache_idx < len(decoder_cache.decoder_conv_caches):
                        block_caches.append(decoder_cache.decoder_conv_caches[decoder_cache_idx])
                    else:
                        block_caches.append(None)
                    work_bufs = decoder_cache.decoder_work_buffers
                    if work_bufs and decoder_cache_idx < len(work_bufs):
                        block_work_bufs.append(work_bufs[decoder_cache_idx])
                    else:
                        block_work_bufs.append(None)
                    out_bufs = decoder_cache.decoder_output_buffers
                    if out_bufs and decoder_cache_idx < len(out_bufs):
                        block_output_bufs.append(out_bufs[decoder_cache_idx])
                    else:
                        block_output_bufs.append(None)
                    decoder_cache_idx += 1

                # Get TransConvNet cache and work buffer for this block
                transconv_cache = None
                transconv_work_buf = None
                if decoder_cache.transconv_caches and transconv_cache_idx < len(decoder_cache.transconv_caches):
                    transconv_cache = decoder_cache.transconv_caches[transconv_cache_idx]
                tw_bufs = decoder_cache.transconv_work_buffers
                if tw_bufs and transconv_cache_idx < len(tw_bufs):
                    transconv_work_buf = tw_bufs[transconv_cache_idx]
                transconv_cache_idx += 1

                wav, _ = block.forward_chunk(
                    wav, block_caches, block_work_bufs, block_output_bufs,
                    transconv_cache, transconv_work_buf
                )
            else:
                # SnakeBeta or other non-caching blocks
                wav = block(wav)

        # Return the same cache object (all updates were in-place)
        return wav.clamp(min=-1, max=1), decoder_cache


class Qwen3TTSTokenizerV2Encoder(MimiModel):
    """Encoder based on MimiModel with decoder parts disabled."""

    def __init__(self, config: MimiConfig):
        super().__init__(config)
        self.config = config

        self.upsample = None
        self.decoder_transformer = None
        self.decoder = None


class Qwen3TTSTokenizerV2Model(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerV2Config):
        super().__init__()
        self.config = config

        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers

        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate

        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        # Create MimiConfig from encoder_config for MimiModel-based encoder
        encoder_cfg = config.encoder_config
        mimi_config = MimiConfig(
            audio_channels=encoder_cfg.audio_channels,
            codebook_dim=encoder_cfg.codebook_dim,
            codebook_size=encoder_cfg.codebook_size,
            compress=encoder_cfg.compress,
            dilation_growth_rate=encoder_cfg.dilation_growth_rate,
            head_dim=encoder_cfg.head_dim,
            hidden_size=encoder_cfg.hidden_size,
            intermediate_size=encoder_cfg.intermediate_size,
            kernel_size=encoder_cfg.kernel_size,
            last_kernel_size=encoder_cfg.last_kernel_size,
            num_filters=encoder_cfg.num_filters,
            num_hidden_layers=encoder_cfg.num_hidden_layers,
            num_attention_heads=encoder_cfg.num_attention_heads,
            num_key_value_heads=encoder_cfg.num_key_value_heads,
            num_quantizers=encoder_cfg.num_quantizers,
            num_residual_layers=encoder_cfg.num_residual_layers,
            pad_mode=encoder_cfg.pad_mode,
            residual_kernel_size=encoder_cfg.residual_kernel_size,
            rope_theta=encoder_cfg.rope_theta,
            sampling_rate=encoder_cfg.sampling_rate,
            sliding_window=encoder_cfg.sliding_window,
            trim_right_ratio=encoder_cfg.trim_right_ratio,
            upsample_groups=encoder_cfg.upsample_groups,
            upsampling_ratios=encoder_cfg.upsampling_ratios,
            use_cache=encoder_cfg.use_cache,
            use_conv_shortcut=encoder_cfg.use_conv_shortcut,
            vector_quantization_hidden_dimension=encoder_cfg.vector_quantization_hidden_dimension,
        )
        self.encoder = Qwen3TTSTokenizerV2Encoder(mimi_config)
        self.decoder = Qwen3TTSTokenizerV2Decoder(self.config.decoder_config)

    def get_model_type(self):
        return self.config.model_type

    def get_input_sample_rate(self):
        return self.input_sample_rate

    def get_output_sample_rate(self):
        return self.output_sample_rate

    def get_encode_downsample_rate(self):
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self):
        return self.decode_upsample_rate

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are
                either 1 for *not masked* or 0 for *masked*.
        """
        encoded_frames = self.encoder.encode(input_values=input_values.unsqueeze(1),
                                             return_dict=True)
        audio_codes = encoded_frames.audio_codes[:, :self.encoder_valid_num_quantizers]
        audio_codes = [
            code[..., :-(-mask.sum() // self.encode_downsample_rate)].transpose(0, 1)
            for code, mask in zip(audio_codes, padding_mask, strict=False)
        ]

        return audio_codes

    def decode(
        self,
        audio_codes: torch.Tensor,
    ):
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, codes_length, num_quantizers)`, *optional*):
                Discret code embeddings computed using `model.encode`.
        """
        audio_values = self.decoder.chunked_decode(audio_codes)

        return audio_values


class Qwen3TTSDecoder(nn.Module):
    """
    Audio codec for Qwen3 TTS model.

    TODO: Implement the actual codec architecture based on Qwen3 TTS specifications.
    This should include both encoder (audio -> tokens) and decoder (tokens -> audio).
    """

    def __init__(
        self,
        model_repo: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Qwen3 audio codec.

        Args:
            model_repo: HuggingFace model repository ID
            device: Device to load the model on
            dtype: Data type for model weights (e.g., torch.float32, torch.bfloat16)
        """
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Load config
        config_path = hf_hub_download(
            repo_id=model_repo,
            filename="config.json",
        )
        config = Qwen3TTSTokenizerV2Config.from_json(config_path)

        # Create model from config
        self.model = Qwen3TTSTokenizerV2Model(config)

        # Load pretrained weights using transformers utilities
        from transformers.utils import cached_file

        # Try to load weights (codec model is usually not sharded)
        try:
            # Try safetensors first
            weights_path = cached_file(model_repo, "model.safetensors")
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device=str(device))
        except Exception:
            try:
                # Fallback to pytorch_model.bin
                weights_path = cached_file(model_repo, "pytorch_model.bin")
                state_dict = torch.load(weights_path, map_location=device)
            except Exception as e:
                raise RuntimeError(f"Could not load model weights from {model_repo}: {e}") from e

        # Load state dict into model
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device=device, dtype=dtype)
        self.model.eval()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete tokens to audio waveform.

        Args:
            codes: Tensor of shape (batch_size, num_quantizers, codes_length)
                  e.g., (1, 16, 100) for 100 frames with 16 codebooks

        Returns:
            Audio tensor. Shape: (batch_size, n_channels, audio_length)
        """
        # Call decoder.chunked_decode directly - codes are already in correct shape
        # Expected input: (batch_size, num_quantizers, codes_length)
        audio_values = self.model.decoder.chunked_decode(codes)

        # audio_values shape: (batch_size, 1, audio_length) with channel dimension
        return audio_values

    def init_cache(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        detokenize_interval: int = 10,
    ) -> Qwen3TTSDecoderCache:
        """Initialize empty cache for streaming inference.

        Args:
            batch_size: Number of items in batch
            device: Device to create tensors on (defaults to self.device)
            dtype: Data type for tensors
            detokenize_interval: Number of tokens per chunk (for work buffer sizing)

        Returns:
            Qwen3TTSDecoderCache with all fields initialized
        """
        if device is None:
            device = self.device
        return self.model.decoder.init_cache(batch_size, device, dtype, detokenize_interval)

    def decode_chunk(
        self,
        codes: torch.Tensor,
        decoder_cache: Optional[Qwen3TTSDecoderCache] = None,
    ) -> Tuple[torch.Tensor, Qwen3TTSDecoderCache]:
        """Decode a chunk of tokens with caching for streaming inference.

        Args:
            codes: Tensor of shape (batch_size, num_quantizers, chunk_length)
            decoder_cache: Previous cache or None for first chunk

        Returns:
            audio_values: Audio tensor. Shape: (batch_size, 1, audio_length)
            new_cache: Updated cache for next chunk
        """
        audio_values, new_cache = self.model.decoder.forward_chunk(codes, decoder_cache)
        return audio_values, new_cache

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Encode audio waveform to discrete codes.

        Args:
            input_values: Tensor of shape (batch_size, sequence_length)
                         Float values of the input audio waveform at 24kHz.
            padding_mask: Tensor of shape (batch_size, sequence_length)
                         Indicates which inputs are to be ignored due to padding,
                         where elements are either 1 for *not masked* or 0 for *masked*.

        Returns:
            List of audio codes tensors, each of shape (T, num_quantizers) where T
            is the sequence length after downsampling (varies per sample based on padding).
        """
        return self.model.encode(input_values, padding_mask)
