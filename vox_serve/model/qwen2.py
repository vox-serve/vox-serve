from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm

_VIBEVOICE_REPO = "microsoft/VibeVoice-1.5B"
_LM_PREFIX = "model.language_model."
_GLOBAL_PARAM_SUFFIXES = (
    "embed_tokens.weight",
    "norm.weight",
)


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    hidden_act: str
    attention_dropout: float
    pad_token_id: int | None
    attention_bias: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_qwen2_config(
        cls,
        config: Qwen2Config,
        *,
        num_hidden_layers: int | None = None,
    ) -> "Qwen2Config":
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=num_hidden_layers or config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            hidden_act=config.hidden_act,
            attention_dropout=getattr(config, "attention_dropout", 0.0),
            pad_token_id=getattr(config, "pad_token_id", None),
            attention_bias=getattr(config, "attention_bias", True),
        )


@dataclass(frozen=True)
class Qwen2SplitMetadata:
    split_idx: int
    total_layers: int
    tts_backbone_num_hidden_layers: int
    lower_layer_range: tuple[int, int]
    upper_layer_range: tuple[int, int]
    upper_layer_remap: dict[int, int]


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm(
            hidden_states=hidden_states,
            weight=self.weight,
            eps=self.variance_epsilon,
        )

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        # HF Qwen2 uses bias=False for o_proj.
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_value_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(query_shape)
        key_states = self.k_proj(hidden_states).view(key_value_shape)
        value_states = self.v_proj(hidden_states).view(key_value_shape)

        query_states, key_states = apply_rope_pos_ids(
            query_states=query_states,
            key_states=key_states,
            position_ids=position_ids,
            rope_theta=self.rope_theta,
            interleave=False,
        )

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )
        return self.norm(hidden_states)

def _format_keys_for_error(keys: set[str], limit: int = 10) -> str:
    if not keys:
        return "[]"
    sorted_keys = sorted(keys)
    if len(sorted_keys) <= limit:
        return str(sorted_keys)
    return str(sorted_keys[:limit] + [f"... ({len(sorted_keys) - limit} more)"])


def validate_vibevoice_qwen2_config(decoder_config: Qwen2Config | Mapping[str, Any]) -> Qwen2Config:
    if isinstance(decoder_config, Mapping):
        decoder_config = Qwen2Config(**dict(decoder_config))
    if not isinstance(decoder_config, Qwen2Config):
        raise TypeError("decoder_config must be Qwen2Config or dict compatible with Qwen2Config.")

    if decoder_config.model_type != "qwen2":
        raise ValueError(f"Unsupported decoder model type: {decoder_config.model_type}")

    expected_pairs = {
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "hidden_size": 1536,
        "vocab_size": 151936,
        "rope_theta": 1_000_000.0,
    }
    for field_name, expected_value in expected_pairs.items():
        actual_value = getattr(decoder_config, field_name)
        if actual_value != expected_value:
            raise ValueError(f"Unsupported decoder_config.{field_name}: {actual_value} != {expected_value}")

    if getattr(decoder_config, "hidden_act", None) != "silu":
        raise ValueError(f"Unsupported hidden_act: {decoder_config.hidden_act}. Expected 'silu'.")
    if getattr(decoder_config, "attention_bias", True) is not True:
        raise ValueError("Unsupported attention_bias=False. VibeVoice-1.5B requires attention_bias=True.")

    if getattr(decoder_config, "sliding_window", None) not in (None, 0):
        raise ValueError(
            "Sliding-window attention is unsupported for VibeVoice-native Qwen2 in this phase."
        )
    if getattr(decoder_config, "use_sliding_window", False):
        raise ValueError(
            "use_sliding_window=True is unsupported for VibeVoice-native Qwen2 in this phase."
        )

    rope_scaling = getattr(decoder_config, "rope_scaling", None)
    if rope_scaling not in (None, {}):
        raise ValueError(
            f"Non-default rope_scaling is unsupported in this phase. Got: {rope_scaling}"
        )

    layer_types = getattr(decoder_config, "layer_types", None)
    if layer_types is not None and any(layer_type != "full_attention" for layer_type in layer_types):
        raise ValueError("Only full_attention layers are supported in this phase.")

    return decoder_config

def _load_decoder_config_from_repo(repo_id: str) -> Qwen2Config:
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=None)
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    decoder_config_dict = config_data.get("decoder_config")
    if decoder_config_dict is None:
        raise ValueError("config.json does not contain decoder_config.")
    return validate_vibevoice_qwen2_config(decoder_config_dict)


def _load_language_model_weight_map(repo_id: str, *, num_hidden_layers: int) -> dict[str, str]:
    index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json", revision=None)
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    full_weight_map = index_data.get("weight_map")
    if not isinstance(full_weight_map, dict):
        raise ValueError("model.safetensors.index.json does not contain a valid weight_map.")

    language_model_weight_map = {
        key: shard for key, shard in full_weight_map.items() if key.startswith(_LM_PREFIX)
    }
    return language_model_weight_map


def _load_language_model_state_dict_from_shards(
    repo_id: str,
    language_model_weight_map: Mapping[str, str],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("safetensors is required to load VibeVoice-1.5B Qwen2 weights.") from exc

    state_dict: dict[str, torch.Tensor] = {}
    keys_per_shard: dict[str, list[str]] = {}
    for key, shard in language_model_weight_map.items():
        keys_per_shard.setdefault(shard, []).append(key)

    for shard in sorted(keys_per_shard):
        shard_path = hf_hub_download(repo_id=repo_id, filename=shard, revision=None)
        shard_state = load_file(shard_path, device=str(device))
        for key in keys_per_shard[shard]:
            if key not in shard_state:
                raise ValueError(f"Missing expected tensor '{key}' in shard '{shard}'.")
            state_dict[key] = shard_state[key]

    missing_after_load = set(language_model_weight_map.keys()) - set(state_dict.keys())
    if missing_after_load:
        raise ValueError(
            "Failed to load all expected language_model tensors. "
            f"Missing: {_format_keys_for_error(missing_after_load)}"
        )
    return state_dict


def _strip_language_model_prefix(language_model_state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for key, value in language_model_state_dict.items():
        if not key.startswith(_LM_PREFIX):
            raise ValueError(f"Unexpected key without language model prefix: '{key}'")
        stripped[key[len(_LM_PREFIX):]] = value
    return stripped

def split_vibevoice_qwen2_state_dict(
    language_model_state_dict: Mapping[str, torch.Tensor],
    *,
    num_hidden_layers: int = 28,
    tts_backbone_num_hidden_layers: int = 20,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], Qwen2SplitMetadata]:

    pseudo_weight_map = {key: "_" for key in language_model_state_dict.keys()}

    split_idx = num_hidden_layers - tts_backbone_num_hidden_layers
    upper_layer_remap = {
        old_idx: old_idx - split_idx for old_idx in range(split_idx, num_hidden_layers)
    }
    metadata = Qwen2SplitMetadata(
        split_idx=split_idx,
        total_layers=num_hidden_layers,
        tts_backbone_num_hidden_layers=tts_backbone_num_hidden_layers,
        lower_layer_range=(0, split_idx - 1),
        upper_layer_range=(split_idx, num_hidden_layers - 1),
        upper_layer_remap=upper_layer_remap,
    )

    lower_state: dict[str, torch.Tensor] = {}
    upper_state: dict[str, torch.Tensor] = {}

    layer_key_regex = re.compile(rf"^{re.escape(_LM_PREFIX)}layers\.(\d+)\.(.+)$")

    for key, value in language_model_state_dict.items():
        suffix = key[len(_LM_PREFIX):]
        if suffix in _GLOBAL_PARAM_SUFFIXES:
            # ["embed_tokens.weight", "norm.weight"] are shared between the lower and upper models
            lower_state[suffix] = value
            upper_state[suffix] = value
            continue
        
        # Regex based mapping for model.language_model.layers.X where
        # X < split_idx is lower model, X >= split_idx is upper model
        m = layer_key_regex.match(key)
        if m is None:
            raise ValueError(f"Unexpected key in language model state dict: '{key}'")
        layer_idx = int(m.group(1))
        layer_suffix = m.group(2)

        if layer_idx < split_idx:
            lower_state[f"layers.{layer_idx}.{layer_suffix}"] = value
        else:
            remapped_layer = layer_idx - split_idx
            upper_state[f"layers.{remapped_layer}.{layer_suffix}"] = value

    return lower_state, upper_state, metadata


def _load_language_model_state(
    repo_id: str,
    *,
    device: torch.device,
) -> tuple[Qwen2Config, dict[str, torch.Tensor]]:
    decoder_config = _load_decoder_config_from_repo(repo_id)
    weight_map = _load_language_model_weight_map(
        repo_id,
        num_hidden_layers=decoder_config.num_hidden_layers,
    )
    prefixed_state_dict = _load_language_model_state_dict_from_shards(
        repo_id=repo_id,
        language_model_weight_map=weight_map,
        device=device,
    )
    return decoder_config, prefixed_state_dict


def load_qwen2_from_hf(
    repo_id: str = _VIBEVOICE_REPO,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen2Model:
    torch_device = torch.device(device)
    decoder_config, prefixed_state_dict = _load_language_model_state(
        repo_id,
        device=torch_device,
    )
    stripped_state_dict = _strip_language_model_prefix(prefixed_state_dict)

    native_config = Qwen2Config.from_qwen2_config(decoder_config)
    model = Qwen2Model(native_config)
    model.load_state_dict(stripped_state_dict, strict=True)
    model = model.to(dtype).to(torch_device)
    model.eval()
    return model


def load_vibevoice_qwen2_split_from_hf(
    repo_id: str = _VIBEVOICE_REPO,
    *,
    tts_backbone_num_hidden_layers: int = 20,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Qwen2Model, Qwen2Model]:
    torch_device = torch.device(device)
    decoder_config, prefixed_state_dict = _load_language_model_state(
        repo_id,
        device=torch_device,
    )

    lower_state, upper_state, metadata = split_vibevoice_qwen2_state_dict(
        prefixed_state_dict,
        num_hidden_layers=decoder_config.num_hidden_layers,
        tts_backbone_num_hidden_layers=tts_backbone_num_hidden_layers,
    )

    lower_config = Qwen2Config.from_qwen2_config(
        decoder_config,
        num_hidden_layers=metadata.split_idx,
    )
    upper_config = Qwen2Config.from_qwen2_config(
        decoder_config,
        num_hidden_layers=metadata.tts_backbone_num_hidden_layers,
    )

    language_backbone = Qwen2Model(lower_config)
    tts_backbone = Qwen2Model(upper_config)

    language_backbone.load_state_dict(lower_state, strict=True)
    tts_backbone.load_state_dict(upper_state, strict=True)

    language_backbone = language_backbone.to(dtype).to(torch_device)
    tts_backbone = tts_backbone.to(dtype).to(torch_device)
    language_backbone.eval()
    tts_backbone.eval()

    # Expose split metadata for auditability without changing return shape.
    language_backbone.split_metadata = metadata
    tts_backbone.split_metadata = metadata

    return language_backbone, tts_backbone


def build_vibevoice_qwen2_split_metadata(
    *,
    num_hidden_layers: int = 28,
    tts_backbone_num_hidden_layers: int = 20,
) -> Qwen2SplitMetadata:
    split_idx = num_hidden_layers - tts_backbone_num_hidden_layers
    if split_idx <= 0:
        raise ValueError(
            "tts_backbone_num_hidden_layers must be smaller than num_hidden_layers."
        )
    upper_layer_remap = {
        old_idx: old_idx - split_idx for old_idx in range(split_idx, num_hidden_layers)
    }
    return Qwen2SplitMetadata(
        split_idx=split_idx,
        total_layers=num_hidden_layers,
        tts_backbone_num_hidden_layers=tts_backbone_num_hidden_layers,
        lower_layer_range=(0, split_idx - 1),
        upper_layer_range=(split_idx, num_hidden_layers - 1),
        upper_layer_remap=upper_layer_remap,
    )


__all__ = [
    "Qwen2Config",
    "Qwen2SplitMetadata",
    "Qwen2RMSNorm",
    "Qwen2MLP",
    "Qwen2Attention",
    "Qwen2DecoderLayer",
    "Qwen2Model",
    "split_vibevoice_qwen2_state_dict",
    "build_vibevoice_qwen2_split_metadata",
    "load_vibevoice_qwen2_split_from_hf",
]
