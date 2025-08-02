from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
import asyncio
import threading
import queue
import wave

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers import LlamaConfig, LlamaPreTrainedModel
from huggingface_hub import hf_hub_download
import safetensors

from ..tokenizer.snac import SNAC
from ..flashinfer_utils import FlashInferWrapper
from ..sampling import top_p_sampling, apply_repetition_penalty
from .base import BaseLM


from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class BackboneConfig:
    d_model: int = 1024
    d_intermediate: int = 0
    attn_mlp_d_intermediate: int = 0
    n_layer: int = 16
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = False
    residual_in_fp32: bool = False
    norm_epsilon: float = 1e-5
    rope_theta: float = 10000  # Default value for RoPE theta
    head_dim: int = 0  # Will be set based on d_model and num_heads in the model config


@dataclass
class PrefixConditionerConfig:
    conditioners: list[dict]
    projection: Literal["none", "linear", "mlp"]


@dataclass
class ZonosConfig:
    backbone: BackboneConfig
    prefix_conditioner: PrefixConditionerConfig
    eos_token_id: int = 1024
    masked_token_id: int = 1025
    pad_vocab_to_multiple_of: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "ZonosConfig":
        d = d.copy()
        backbone_config = BackboneConfig(**d.pop("backbone"))
        prefix_conditioner_config = PrefixConditionerConfig(**d.pop("prefix_conditioner"))
        config = cls(backbone_config, prefix_conditioner_config, **d)
        return config


class ZonosRotaryEmbedding(nn.Module):
    def __init__(self, config: BackboneConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = getattr(config, "max_position_embeddings", 16384)
        self.original_max_seq_len = getattr(config, "max_position_embeddings", 16384)

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.config.rope_theta = 10000 
        self.config.head_dim = config.d_model // config.attn_cfg["num_heads"]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # If position_ids is flat (i.e. 1D [total_seq_len]), unsqueeze it to [1, total_seq_len]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        # Core RoPE block: now position_ids is of shape [batch, seq_len] where batch=1 if it was flattened
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # (batch, num_freq, 1) @ (batch, 1, seq_len) -> (batch, num_freq, seq_len)
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Apply scaling
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        # Squeeze out the batch dimension if it was originally flattened
        return cos.squeeze(0).to(dtype=x.dtype), sin.squeeze(0).to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=0):
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


class ZonosMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.attn_mlp_d_intermediate, bias=False)
        self.fc2 = nn.Linear(config.attn_mlp_d_intermediate, config.d_model, bias=False)

    def forward(self, x):
        y, gate = self.fc1(x).chunk(2, dim=-1)
        y = self.fc2(y * F.silu(gate))
        return y


class ZonosAttention(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.attn_cfg["num_heads"]
        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // self.num_heads
        # self.num_key_value_groups = config.num_heads // config.num_key_value_heads
        # self.scaling = self.head_dim**-0.5
        # self.attention_dropout = config.attention_dropout
        # self.is_causal = True

        total_head_dim = (self.num_heads + 2 * self.num_heads_kv) * self.head_dim
        self.in_proj = nn.Linear(config.d_model, total_head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim
        query_states, key_states, value_states = self.in_proj(hidden_states).split([q_size, kv_size, kv_size], dim=-1)

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_heads_kv, self.head_dim)
        value_states = value_states.view(-1, self.num_heads_kv, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_wrapper.set_kv_cache(kv_cache, key_states.transpose(0, 1), value_states.transpose(0, 1))
        attn_output = attn_wrapper.run(query_states.transpose(0, 1), kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class ZonosDecoderLayer(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.config = config 

        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mixer = ZonosAttention(config=config, layer_idx=layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mlp = ZonosMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        # Self Attention
        hidden_states = self.mixer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class ZonosBackboneModel(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [ZonosDecoderLayer(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.rotary_emb = ZonosRotaryEmbedding(config=config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )

        hidden_states = self.norm_f(hidden_states)

        return hidden_states


class ZonosForCausalLM(nn.Module):

    def __init__(self, config: ZonosConfig):
        super().__init__(config)
        self.backbone = ZonosBackboneModel(config)

        dim = config.backbone.d_model
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

    def embed_tokens(self, input_ids):
        return self.model.embed_tokens(input_ids)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        logits = self.lm_head(outputs) 

        return logits



class ZonosModel(BaseLM):
    def __init__(self, model_name, dtype=torch.bfloat16, device="cuda:0", tokenizer_path="canopylabs/orpheus-3b-0.1-ft"):
        # TODO: Zonos hybrid model is not yet supported
        super().__init__(model_name, device, dtype)
        config_path = hf_hub_download(repo_id=model_name, filename="config.json", revision=None)
        model_path = hf_hub_download(repo_id=model_name, filename="model.safetensors", revision=None)

        self.config = ZonosConfig.from_dict(torch.load(config_path))
        self.model = ZonosForCausalLM(self.config)
        self.model.to(dtype).to(device)

        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                if k in self.model.state_dict():
                    self.model.state_dict()[k].copy_(f.get_tensor(k))

        # Use provided tokenizer path or default to model_name
        self.text_tokenizer = self._load_tokenizer(tokenizer_path)
        # self.audio_tokenizer = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

        dim = self.config.backbone.d_model
        self.eos_token_id = self.config.eos_token_id
        self.masked_token_id = self.config.masked_token_id

        self.autoencoder = DACAutoencoder()
        # self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        self._num_attention_heads = self.model.config.num_attention_heads 
        self._num_key_value_heads = self.model.config.num_key_value_heads
        self._num_hidden_layers = self.model.config.num_hidden_layers
        self._hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        # if config.pad_vocab_to_multiple_of:
        #     self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)


    @property
    def num_attention_heads(self) -> int:
        """Number of attention heads in the model."""
        return self._num_attention_heads
    
    @property
    def num_key_value_heads(self) -> int:
        """Number of key-value heads in the model."""
        return self._num_key_value_heads
    
    @property
    def num_hidden_layers(self) -> int:
        """Number of hidden layers in the model."""
        return self._num_hidden_layers
    
    @property
    def hidden_size(self) -> int:
        """Hidden size of the model."""
        return self._hidden_size

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_path)
    
    def preprocess(self, prompt, voice="tara", model_type="larger"):
        """Prepare the prompt for the model, formatting it according to Orpheus specifications."""
        # self.validate_voice(voice)
        input_ids, prompt_string = self.orpheus_format_prompt(prompt, voice, model_type)
        return input_ids[0].tolist(), prompt_string
    
    def forward(self, input_ids, position_ids, attn_wrapper, kv_cache, repetition_cache):
        """Forward pass through the model."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        logits = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        
        logits = apply_repetition_penalty(logits, repetition_cache, self.repetition_penalty)

        output_ids = top_p_sampling(logits, top_p=self.top_p, temperature=self.temperature)

        if getattr(attn_wrapper, "qo_indptr", None) is not None:
            output_ids = output_ids[attn_wrapper.qo_indptr[:-1] - 1]

        return output_ids
    
    def decode_text_token(self, token_id):
        return self.text_tokenizer.decode(token_id)
    
    def turn_token_into_id(self, output_ids):
        """Modoel's output ids to audio ids"""
        return (output_ids - 128256 - 10) % 4096
    
    def is_stop_id(self, token_id):
        return token_id == self.stop_token_id

    def postprocess(self, tokens_list, next_audio_decode_idx, done_all):
        pass
