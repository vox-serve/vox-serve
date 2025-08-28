import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# from hyperpyyaml import load_hyperpyyaml
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import WhisperFeatureExtractor

from ..utils import load_hf_safetensor_state_dict


@dataclass
class GLMEncoderConfig:
    # _name_or_path: str = "THUDM/glm-4-voice-tokenizer"
    activation_dropout: float = 0.0
    # activation_function: str = "gelu"
    apply_spec_augment: bool = False
    # architectures: List[str] = field(default_factory=lambda: ["WhisperVQEncoder"])
    attention_dropout: float = 0.0
    begin_suppress_tokens: List[int] = field(default_factory=lambda: [220, 50257])
    bos_token_id: int = 50257
    classifier_proj_size: int = 256
    d_model: int = 1280
    # decoder_attention_heads: int = 20
    # decoder_ffn_dim: int = 5120
    # decoder_layerdrop: float = 0.0
    # decoder_layers: int = 32
    # decoder_start_token_id: int = 50258
    # dropout: float = 0.0
    encoder_attention_heads: int = 20
    encoder_causal_attention: bool = False
    encoder_causal_convolution: bool = True
    encoder_ffn_dim: int = 5120
    encoder_layerdrop: float = 0.0
    encoder_layers: int = 32
    eos_token_id: int = 50257
    init_std: float = 0.02
    is_encoder_decoder: bool = True
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    mask_feature_prob: float = 0.0
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_time_prob: float = 0.05
    max_length: int = 448
    max_source_positions: int = 1500
    max_target_positions: int = 448
    median_filter_width: int = 7
    model_type: str = "whisper"
    num_hidden_layers: int = 32
    num_mel_bins: int = 128
    pad_token_id: int = 50256
    pooling_kernel_size: int = 4
    pooling_position: int = 16
    pooling_type: str = "avg"
    quantize_causal_block_size: int = 200
    quantize_causal_encoder: bool = False
    quantize_commit_coefficient: float = 0.25
    quantize_ema_decay: float = 0.99
    quantize_encoder_only: bool = True
    quantize_loss_scale: float = 10.0
    quantize_position: int = 16
    quantize_restart_interval: int = 100
    quantize_vocab_size: int = 16384
    scale_embedding: bool = False
    skip_language_detection: bool = True
    torch_dtype: str = "float32"
    transformers_version: str = "4.44.1"
    use_cache: bool = True
    use_weighted_layer_sum: bool = False
    vocab_size: int = 51866

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GLMEncoderConfig":
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        # Filter config_dict to only include known fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs
    ):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inp):
        x = torch.nn.functional.pad(inp.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)


class GLMWhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[GLMEncoderConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class GLMWhisperVQEncoderLayer(nn.Module):
    def __init__(self, config: GLMEncoderConfig, is_causal=False):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = GLMWhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            is_causal=is_causal,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GLMWhisperVQEncoder(nn.Module):
    def __init__(self, config: GLMEncoderConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_source_positions = config.max_source_positions
        self.quantize_vocab_size = config.quantize_vocab_size
        self.num_mel_bins = config.num_mel_bins

        self.embed_positions = nn.Embedding(self.max_source_positions, self.d_model)
        self.embed_positions2 = nn.Embedding(self.max_source_positions // config.pooling_kernel_size, self.d_model)
        self.codebook = nn.Embedding(self.quantize_vocab_size, self.d_model)

        self.conv1 = CausalConv1d(self.num_mel_bins, self.d_model, kernel_size=3, padding=1)
        self.conv2 = CausalConv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList(
            [
                GLMWhisperVQEncoderLayer(
                    config,
                    is_causal=False,
                )
                for layer_id in range(config.quantize_position)
            ]
        )

        self.pooling_layer = nn.AvgPool1d(kernel_size=config.pooling_kernel_size)

        self.register_buffer("ema_count", torch.ones(config.quantize_vocab_size, dtype=torch.float))
        self.register_buffer("ema_weight", self.codebook.weight.data.clone().float())

    def vector_quantize(self, inputs, codebook):
        embedding_size = codebook.size(1)
        inputs_flatten = inputs.reshape(-1, embedding_size)
        codebook_sqr = torch.sum(codebook**2, dim=1)
        inputs_sqr = torch.sum(inputs_flatten**2, dim=1, keepdim=True)
        # Compute the distances to the codebook
        distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

        _, indices_flatten = torch.min(distances, dim=1)
        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)
        return codes, indices_flatten, distances

    def get_block_causal_attention_mask(self, attention_mask, block_size=50):
        dtype = torch.bfloat16
        batch_size, seq_length = attention_mask.shape
        causal_mask = torch.torch.tril(
            torch.ones(1, seq_length, seq_length, dtype=torch.bool, device=attention_mask.device)
        )
        block_square_mask = []
        for start in range(0, seq_length, block_size):
            end = min(start + block_size, seq_length)
            length = end - start
            block_square_mask.append(causal_mask.new_ones((length, length)))
        block_square_mask = torch.block_diag(*block_square_mask)
        block_causal_mask = causal_mask | block_square_mask
        block_causal_mask = block_causal_mask & attention_mask[:, None, :]
        block_causal_mask = block_causal_mask.to(dtype=dtype)  # fp16 compatibility
        block_causal_mask = (1.0 - block_causal_mask) * torch.finfo(dtype).min
        block_causal_mask = block_causal_mask.unsqueeze(1)
        return block_causal_mask

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, feature_size, seq_length = input_features.shape
        seq_length = seq_length // (self.conv1.stride[0] * self.conv2.stride[0])

        attention_mask = attention_mask[:, :: self.conv1.stride[0] * self.conv2.stride[0]]
        extended_attention_mask = self.get_block_causal_attention_mask(
            attention_mask, block_size=self.config.quantize_causal_block_size
        )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos[:seq_length]

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                extended_attention_mask,
            )

            if idx + 1 == self.config.pooling_position and self.config.pooling_kernel_size is not None:
                hidden_states = hidden_states.permute(0, 2, 1)
                if hidden_states.shape[-1] % self.config.pooling_kernel_size != 0:
                    hidden_states = torch.nn.functional.pad(
                        hidden_states,
                        (
                            0,
                            self.config.pooling_kernel_size - hidden_states.shape[-1] % self.config.pooling_kernel_size,
                        ),
                    )
                hidden_states = self.pooling_layer(hidden_states).permute(0, 2, 1)
                attention_mask = attention_mask[:, :: self.config.pooling_kernel_size]
                extended_attention_mask = self.get_block_causal_attention_mask(
                    attention_mask, block_size=self.config.quantize_causal_block_size // self.config.pooling_kernel_size
                )

            if idx + 1 == self.config.quantize_position and self.config.quantize_vocab_size is not None:
                hidden_quantized, indices_flat, distances = self.vector_quantize(hidden_states, self.codebook.weight)
                quantized_token_ids = indices_flat.reshape(batch_size, hidden_quantized.shape[1])
                hidden_states = hidden_quantized
                hidden_states = hidden_states + self.embed_positions2.weight[: hidden_states.shape[1]]

        return quantized_token_ids


class GLMVoiceEncoder:
    def __init__(self, repo_id: str, dtype: torch.dtype, device: str):
        self.repo_id = repo_id
        self.device = device
        self.dtype = dtype

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=None)
        self.config = GLMEncoderConfig.from_dict(json.load(open(config_path)))

        self.encoder = GLMWhisperVQEncoder(self.config)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(repo_id)

        self.encoder.load_state_dict(
            load_hf_safetensor_state_dict(repo_id=repo_id, revision=None, token=None),
            strict=True,
        )
        self.encoder.to(dtype).to(device)

        pooling_kernel_size = self.config.pooling_kernel_size or 1
        self.stride = (
            self.encoder.conv1.stride[0]
            * self.encoder.conv2.stride[0]
            * pooling_kernel_size
            * self.feature_extractor.hop_length
        )

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        features = (
            self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
                device=self.device,
                padding="longest",
                pad_to_multiple_of=self.stride,
            )
            .to(self.device)
            .to(self.dtype)
        )
        speech_tokens = self.encoder(**features)
        return speech_tokens
