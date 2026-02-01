"""
Qwen3 TTS Audio Codec implementation.

This could be based on an existing codec (SoundStream, EnCodec, DAC, etc.)
or a custom codec specific to Qwen3 TTS.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union, List

import numpy as np
import torch
import json
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F


from dataclasses import dataclass, field
from typing import List, Optional, Union 


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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


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


class Qwen3TTSTokenizerV2CausalTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)

        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = pad = self.left_pad

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = hidden_state[..., self.left_pad : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


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
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
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
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


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
        attention_mask: Optional[torch.Tensor] = None,
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
            attention_mask=attention_mask,
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


class Qwen3TTSTokenizerV2DecoderTransformerModel(nn.Module):
    _can_record_outputs = {
        "hidden_states": Qwen3TTSTokenizerV2DecoderTransformerLayer,
        "attentions": Qwen3TTSTokenizerV2DecoderAttention,
    }

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3TTSTokenizerV2DecoderTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSTokenizerV2DecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
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
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        return hidden_states


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
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://huggingface.co/papers/2006.08195
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

        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
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


class Qwen3TTSTokenizerV2Encoder(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerV2EncoderConfig):
        super().__init__()
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

        self.encoder = Qwen3TTSTokenizerV2Encoder(self.config.encoder_config)
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
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
        """
        encoded_frames = self.encoder.encode(input_values=input_values.unsqueeze(1),
                                             return_dict=True)
        audio_codes = encoded_frames.audio_codes[:, :self.encoder_valid_num_quantizers]
        audio_codes = [code[..., :-(-mask.sum() // self.encode_downsample_rate)].transpose(0, 1) for code, mask in zip(audio_codes, padding_mask)]

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
        audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(1)

        audio_lengths = (audio_codes[..., 0] > 0).sum(1) * self.decode_upsample_rate
        audio_values = [a[:l] for a, l in zip(audio_values, audio_lengths)]

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
    ):
        """
        Initialize Qwen3 audio codec.

        Args:
            sample_rate: Audio sample rate in Hz
            n_codebooks: Number of codebooks in the codec
            codebook_size: Size of each codebook (vocabulary size per codebook)
            **kwargs: Additional codec-specific parameters
        """
        super().__init__()
        self.device = device
        config_path = hf_hub_download(
            repo_id=model_repo,
            filename="config.json",
        )
        config = Qwen3TTSTokenizerV2Config.from_json(config_path)  
        self.model = Qwen3TTSTokenizerV2Model(config)
        self.model.to(device)
        self.model.eval()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete tokens to audio waveform.

        Args:
            codes: Tensor of shape (batch_size, codes_length, num_quantizers)

        Returns:
            Audio tensor. Shape: (batch_size, audio_length)
        """
        return self.model.decode(codes)
