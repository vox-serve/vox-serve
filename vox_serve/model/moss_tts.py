import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoTokenizer

from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ..requests import Request
from ..sampling import Sampler, SamplingConfig
from ..tokenizer.moss_audio_tokenizer import MossAudioTokenizer
from ..utils import get_logger
from .base import BaseLMWithDepth, PreprocessOutput

# ---------------------------------------------------------------------------
# Configuration dataclass (mirrors MossTTSDelayConfig from HuggingFace)
# ---------------------------------------------------------------------------


@dataclass
class MossTTSLanguageConfig:
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_hidden_layers: int = 28
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 151936
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960


@dataclass
class MossTTSConfig:
    language_config: MossTTSLanguageConfig = field(default_factory=MossTTSLanguageConfig)
    n_vq: int = 32
    audio_vocab_size: int = 1024
    audio_pad_code: int = 1024
    audio_start_token_id: int = 151652
    audio_end_token_id: int = 151653
    audio_user_slot_token_id: int = 151654
    audio_assistant_gen_slot_token_id: int = 151656
    pad_token_id: int = 151643
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    sampling_rate: int = 24000
    additional_mlp_ffn_hidden_size: int = 2048
    local_ffn_hidden_size: int = 8960
    local_hidden_size: int = 1536
    local_num_layers: int = 4

    @classmethod
    def from_json(cls, path: str) -> "MossTTSConfig":
        with open(path) as f:
            data = json.load(f)
        lang_data = data.get("language_config", {})
        lang_cfg = MossTTSLanguageConfig(
            hidden_size=lang_data.get("hidden_size", 2048),
            intermediate_size=lang_data.get("intermediate_size", 11008),
            num_attention_heads=lang_data.get("num_attention_heads", 16),
            num_key_value_heads=lang_data.get("num_key_value_heads", 8),
            num_hidden_layers=lang_data.get("num_hidden_layers", 28),
            head_dim=lang_data.get("head_dim", 128),
            rms_norm_eps=lang_data.get("rms_norm_eps", 1e-6),
            rope_theta=lang_data.get("rope_theta", 1000000.0),
            vocab_size=lang_data.get("vocab_size", 151936),
            attention_bias=lang_data.get("attention_bias", False),
            mlp_bias=lang_data.get("mlp_bias", False),
            hidden_act=lang_data.get("hidden_act", "silu"),
            max_position_embeddings=lang_data.get("max_position_embeddings", 40960),
        )
        return cls(
            language_config=lang_cfg,
            n_vq=data.get("n_vq", 32),
            audio_vocab_size=data.get("audio_vocab_size", 1024),
            audio_pad_code=data.get("audio_pad_code", 1024),
            audio_start_token_id=data.get("audio_start_token_id", 151652),
            audio_end_token_id=data.get("audio_end_token_id", 151653),
            audio_user_slot_token_id=data.get("audio_user_slot_token_id", 151654),
            audio_assistant_gen_slot_token_id=data.get("audio_assistant_gen_slot_token_id", 151656),
            pad_token_id=data.get("pad_token_id", 151643),
            im_start_token_id=data.get("im_start_token_id", 151644),
            im_end_token_id=data.get("im_end_token_id", 151645),
            sampling_rate=data.get("sampling_rate", 24000),
            additional_mlp_ffn_hidden_size=data.get("additional_mlp_ffn_hidden_size", 2048),
            local_ffn_hidden_size=data.get("local_ffn_hidden_size", 8960),
            local_hidden_size=data.get("local_hidden_size", 1536),
            local_num_layers=data.get("local_num_layers", 4),
        )


# ---------------------------------------------------------------------------
# FlashInfer-based NN modules
# ---------------------------------------------------------------------------


class MossTTSRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(
            hidden_states=hidden_states,
            weight=self.weight,
            eps=self.variance_epsilon,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MossTTSBackboneMLP(nn.Module):
    """Standard SwiGLU MLP for backbone transformer layers."""

    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MossTTSLocalMLP(nn.Module):
    """SwiGLU MLP for local transformer layers (different hidden/intermediate sizes)."""

    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MossTTSProjectionMLP(nn.Module):
    """SwiGLU MLP for projections between backbone and local dimensions.

    Used for speech_embedding_to_local_mlp and local_to_speech_embedding_mlps.
    These have variable input/output sizes and a shared ffn_hidden_size.
    """

    def __init__(self, input_size, ffn_hidden_size, output_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, ffn_hidden_size, bias=bias)
        self.up_proj = nn.Linear(input_size, ffn_hidden_size, bias=bias)
        self.down_proj = nn.Linear(ffn_hidden_size, output_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MossTTSBackboneAttention(nn.Module):
    """FlashInfer Qwen3 attention WITH RoPE + q_norm/k_norm for backbone."""

    def __init__(self, config: MossTTSLanguageConfig, layer_idx: int):
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
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = MossTTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MossTTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_norm(self.q_proj(hidden_states).view(-1, self.head_dim)).view(
            -1, self.num_attention_heads, self.head_dim
        )
        key_states = self.k_norm(self.k_proj(hidden_states).view(-1, self.head_dim)).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        query_states, key_states = apply_rope_pos_ids(
            query_states=query_states,
            key_states=key_states,
            position_ids=position_ids,
            interleave=False,
            rope_theta=self.rope_theta,
        )

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class MossTTSLocalAttention(nn.Module):
    """FlashInfer Qwen3 attention WITHOUT RoPE + q_norm/k_norm for local transformer."""

    def __init__(
        self, hidden_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, attention_bias=False
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.q_norm = MossTTSRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = MossTTSRMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_norm(self.q_proj(hidden_states).view(-1, self.head_dim)).view(
            -1, self.num_attention_heads, self.head_dim
        )
        key_states = self.k_norm(self.k_proj(hidden_states).view(-1, self.head_dim)).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        # NO RoPE for local transformer
        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class MossTTSBackboneDecoderLayer(nn.Module):
    def __init__(self, config: MossTTSLanguageConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MossTTSBackboneAttention(config=config, layer_idx=layer_idx)
        self.mlp = MossTTSBackboneMLP(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.input_layernorm = MossTTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MossTTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_ids, attn_wrapper, kv_cache):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attn_wrapper, kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MossTTSLocalDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        attention_bias=False,
        mlp_bias=False,
    ):
        super().__init__()
        self.self_attn = MossTTSLocalAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            attention_bias,
        )
        self.mlp = MossTTSLocalMLP(hidden_size, intermediate_size, bias=mlp_bias)
        self.input_layernorm = MossTTSRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = MossTTSRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, position_ids, attn_wrapper, kv_cache):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attn_wrapper, kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MossTTSBackboneModel(nn.Module):
    """Backbone model: 28 Qwen3-1.7B decoder layers + embeddings for 33 channels."""

    def __init__(self, config: MossTTSLanguageConfig, n_vq: int, audio_vocab_size: int, audio_pad_code: int):
        super().__init__()
        self.config = config
        channels = 1 + n_vq  # text + audio

        self.embedding_list = nn.ModuleList()
        # Channel 0: text embedding
        self.embedding_list.append(nn.Embedding(config.vocab_size, config.hidden_size))
        # Channels 1-32: audio embeddings (audio_vocab_size + 1 to include pad token)
        for _ in range(n_vq):
            self.embedding_list.append(nn.Embedding(audio_vocab_size + 1, config.hidden_size, audio_pad_code))

        self.layers = nn.ModuleList(
            [MossTTSBackboneDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MossTTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs_embeds, position_ids, attn_wrapper, kv_cache):
        hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, position_ids, attn_wrapper, kv_cache[i])
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MossTTSLocalModel(nn.Module):
    """Local transformer: 4 decoder layers without RoPE, for depth phase via FlashInfer."""

    def __init__(
        self, config: MossTTSLanguageConfig, local_hidden_size: int, local_ffn_hidden_size: int, local_num_layers: int
    ):
        super().__init__()
        # Local transformer uses backbone's head_dim but different hidden_size
        head_dim = config.head_dim
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        rms_norm_eps = config.rms_norm_eps

        self.layers = nn.ModuleList(
            [
                MossTTSLocalDecoderLayer(
                    hidden_size=local_hidden_size,
                    intermediate_size=local_ffn_hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    rms_norm_eps=rms_norm_eps,
                    attention_bias=config.attention_bias,
                    mlp_bias=config.mlp_bias,
                )
                for _ in range(local_num_layers)
            ]
        )
        self.norm = MossTTSRMSNorm(local_hidden_size, eps=rms_norm_eps)

    def forward(self, inputs_embeds, position_ids, attn_wrapper, kv_cache):
        hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, position_ids, attn_wrapper, kv_cache[i])
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MossTTSForCausalLM(nn.Module):
    """Top-level module combining backbone + local transformers + projections + heads."""

    def __init__(self, config: MossTTSConfig):
        super().__init__()
        self.config = config
        lang = config.language_config
        channels = 1 + config.n_vq  # 33

        # Backbone model with embeddings
        self.backbone = MossTTSBackboneModel(
            lang,
            config.n_vq,
            config.audio_vocab_size,
            config.audio_pad_code,
        )

        # Local transformer for depth phase (FlashInfer-based)
        self.local_model = MossTTSLocalModel(
            lang,
            config.local_hidden_size,
            config.local_ffn_hidden_size,
            config.local_num_layers,
        )

        # Projection MLPs
        self.speech_embedding_to_local_mlp = MossTTSProjectionMLP(
            lang.hidden_size,
            config.additional_mlp_ffn_hidden_size,
            config.local_hidden_size,
        )

        self.local_to_speech_embedding_mlps = nn.ModuleList(
            [
                MossTTSProjectionMLP(
                    config.local_hidden_size,
                    config.additional_mlp_ffn_hidden_size,
                    lang.hidden_size,
                )
                for _ in range(channels)
            ]
        )

        # Layer norms before lm heads
        self.layer_norm_before_lm_heads = nn.ModuleList([MossTTSRMSNorm(lang.hidden_size) for _ in range(channels)])

        # LM heads
        self.lm_heads = nn.ModuleList()
        self.lm_heads.append(nn.Linear(lang.hidden_size, lang.vocab_size, bias=False))  # text head
        for _ in range(config.n_vq):
            self.lm_heads.append(nn.Linear(lang.hidden_size, config.audio_vocab_size + 1, bias=False))  # audio heads

        # Pre-stacked buffers for CUDA graph compatibility (audio heads 1-32)
        audio_vocab = config.audio_vocab_size + 1  # 1025
        d_global = lang.hidden_size  # 2048
        d_local = config.local_hidden_size  # 1536
        ffn_hidden = config.additional_mlp_ffn_hidden_size  # 2048

        self.register_buffer("stacked_lm_head_weight", torch.zeros(config.n_vq, audio_vocab, d_global))
        self.register_buffer("stacked_norm_weight", torch.zeros(config.n_vq, d_global))
        # local_to_speech_embedding MLP stacked weights (for audio channels 1-32)
        self.register_buffer("stacked_up2_gate", torch.zeros(config.n_vq, ffn_hidden, d_local))
        self.register_buffer("stacked_up2_up", torch.zeros(config.n_vq, ffn_hidden, d_local))
        self.register_buffer("stacked_up2_down", torch.zeros(config.n_vq, d_global, ffn_hidden))

    def _sync_stacked_buffers(self):
        """Copy per-channel module weights into pre-stacked contiguous buffers."""
        for i in range(self.config.n_vq):
            idx = i + 1  # skip text channel
            self.stacked_lm_head_weight[i].copy_(self.lm_heads[idx].weight)
            self.stacked_norm_weight[i].copy_(self.layer_norm_before_lm_heads[idx].weight)
            self.stacked_up2_gate[i].copy_(self.local_to_speech_embedding_mlps[idx].gate_proj.weight)
            self.stacked_up2_up[i].copy_(self.local_to_speech_embedding_mlps[idx].up_proj.weight)
            self.stacked_up2_down[i].copy_(self.local_to_speech_embedding_mlps[idx].down_proj.weight)

    def forward_backbone(self, inputs_embeds, position_ids, attn_wrapper, kv_cache):
        """Run backbone transformer, return hidden_states."""
        return self.backbone(inputs_embeds, position_ids, attn_wrapper, kv_cache)

    def forward_depth(self, inputs_embeds, position_ids, attn_wrapper, kv_cache):
        """Run local transformer for depth phase.

        Args:
            inputs_embeds: (bs, D_global) - in backbone hidden dimension space
            position_ids: (bs,) - depth iteration index
            attn_wrapper: FlashInfer wrapper for local transformer
            kv_cache: KV cache for local transformer

        Returns:
            logits: (bs, audio_vocab_size+1) = (bs, 1025)
        """
        # Project from D_global to D_local
        local_input = self.speech_embedding_to_local_mlp(inputs_embeds)

        # Run local transformer with FlashInfer
        hidden_states = self.local_model(local_input, position_ids, attn_wrapper, kv_cache)

        # Use position_ids.max() to index into pre-stacked weights (CUDA graph compatible)
        pos = position_ids.max().clamp(min=1, max=self.config.n_vq).sub(1).to(torch.long).view(1)

        # Apply local_to_speech_embedding via stacked gate/up/down
        gate_w = self.stacked_up2_gate.index_select(0, pos).squeeze(0)  # (ffn, d_local)
        up_w = self.stacked_up2_up.index_select(0, pos).squeeze(0)  # (ffn, d_local)
        down_w = self.stacked_up2_down.index_select(0, pos).squeeze(0)  # (d_global, ffn)
        gate_out = F.silu(F.linear(hidden_states, gate_w, None))
        up_out = F.linear(hidden_states, up_w, None)
        projected = F.linear(gate_out * up_out, down_w, None)  # (bs, d_global)

        # Apply layer_norm via stacked norm weights
        norm_w = self.stacked_norm_weight.index_select(0, pos).squeeze(0)  # (d_global,)
        normed = rms_norm(projected, norm_w)

        # Apply lm_head via stacked lm_head weights
        head_w = self.stacked_lm_head_weight.index_select(0, pos).squeeze(0)  # (audio_vocab, d_global)
        logits = F.linear(normed, head_w, None)  # (bs, 1025)

        return logits

    def text_prediction_sdpa(self, backbone_hidden: torch.Tensor) -> torch.Tensor:
        """Predict text token using local transformer with SDPA (no FlashInfer).

        For single-token sequence: attention degenerates to O_proj(expand_gqa(V_proj(x)))
        since softmax of a single element is always 1.0.

        Args:
            backbone_hidden: (bs, D_global) backbone hidden states

        Returns:
            text_logits: (bs, text_vocab_size)
        """
        # Project to local dimension
        x = self.speech_embedding_to_local_mlp(backbone_hidden)  # (bs, D_local)

        # Process through local transformer layers with trivial single-token attention
        for layer in self.local_model.layers:
            residual = x
            h = layer.input_layernorm(x)

            # Trivial attention: softmax([single_score]) = 1.0, so output = O(V(h))
            # We compute V and then O, skipping Q/K since attention is trivially 1.0
            attn = layer.self_attn
            v = attn.v_proj(h)
            # For GQA: expand v from num_kv_heads to num_attention_heads
            v_shaped = v.view(-1, attn.num_key_value_heads, attn.head_dim)
            groups = attn.num_attention_heads // attn.num_key_value_heads
            v_expanded = v_shaped.unsqueeze(2).expand(-1, -1, groups, -1)
            v_expanded = v_expanded.reshape(-1, attn.num_attention_heads * attn.head_dim)
            h = attn.o_proj(v_expanded)
            x = residual + h

            residual = x
            x = layer.post_attention_layernorm(x)
            x = layer.mlp(x)
            x = residual + x

        x = self.local_model.norm(x)

        # Project back to global dimension and apply text head
        x = self.local_to_speech_embedding_mlps[0](x)  # (bs, D_global)
        x = self.layer_norm_before_lm_heads[0](x)
        text_logits = self.lm_heads[0](x)  # (bs, text_vocab_size)

        return text_logits


# ---------------------------------------------------------------------------
# MossTTSModel: BaseLMWithDepth implementation
# ---------------------------------------------------------------------------


class MossTTSModel(BaseLMWithDepth):
    def __init__(
        self,
        model_name,
        dtype=torch.bfloat16,
        device="cuda:0",
        enable_torch_compile=False,
        audio_decoder_device=None,
    ):
        if model_name in ("moss-tts", "moss-tts-local"):
            model_name = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
        super().__init__(model_name, device, dtype, enable_torch_compile, audio_decoder_device)
        self.logger = get_logger(__name__)

        # Load config
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        self.config = MossTTSConfig.from_json(config_path)
        lang = self.config.language_config

        # Build model architecture
        self.model = MossTTSForCausalLM(self.config)

        # Load weights
        self._load_weights(model_name, device)
        self.model.to(dtype).to(device)

        # Sync pre-stacked buffers after weight loading
        self.model._sync_stacked_buffers()

        # Load text tokenizer (Qwen tokenizer from model repo)
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load audio decoder
        with torch.cuda.device(self.audio_decoder_device):
            self.audio_decoder = MossAudioTokenizer(
                device=self.audio_decoder_device,
                dtype=torch.float32,
            )

        # Cache config values
        self._num_attention_heads = lang.num_attention_heads
        self._num_key_value_heads = lang.num_key_value_heads
        self._num_hidden_layers = lang.num_hidden_layers
        self._hidden_size = lang.hidden_size
        self._head_dim = lang.head_dim

        self.audio_end_token_id = self.config.audio_end_token_id

        self.default_sampling_config = SamplingConfig(
            top_k=50,
            top_p=None,
            min_p=None,
            temperature=0.9,
            repetition_penalty=None,
            repetition_window=None,
            cfg_scale=None,
        )

    def _load_weights(self, model_name: str, device: str):
        """Load and map HuggingFace weights to our architecture."""
        from transformers.utils import cached_file

        try:
            index_file = cached_file(model_name, "model.safetensors.index.json")
            resolved_archive_file = None
        except Exception:
            try:
                resolved_archive_file = cached_file(model_name, "model.safetensors")
            except Exception:
                resolved_archive_file = cached_file(model_name, "pytorch_model.bin")

        if resolved_archive_file:
            if resolved_archive_file.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(resolved_archive_file, device=str(device))
            else:
                state_dict = torch.load(resolved_archive_file, map_location=device)
        else:
            import json as json_lib

            with open(index_file) as f:
                index_data = json_lib.load(f)
            state_dict = {}
            for shard_file in set(index_data["weight_map"].values()):
                shard_path = cached_file(model_name, shard_file)
                if shard_file.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    shard_dict = load_file(shard_path, device=str(device))
                else:
                    shard_dict = torch.load(shard_path, map_location=device)
                state_dict.update(shard_dict)

        # Map HF keys to our architecture
        mapped = {}
        for key, value in state_dict.items():
            new_key = self._map_weight_key(key)
            if new_key is not None:
                mapped[new_key] = value

        missing, unexpected = self.model.load_state_dict(mapped, strict=False)
        if missing:
            self.logger.warning("Missing keys during weight loading: %s", missing[:10])
        if unexpected:
            self.logger.warning("Unexpected keys during weight loading: %s", unexpected[:10])

    @staticmethod
    def _map_weight_key(key: str) -> Optional[str]:
        """Map HuggingFace weight key to our architecture key."""
        # model.embedding_list.{i}.* → backbone.embedding_list.{i}.*
        if key.startswith("model.embedding_list."):
            return key.replace("model.embedding_list.", "backbone.embedding_list.", 1)

        # model.language_model.layers.{i}.* → backbone.layers.{i}.*
        if key.startswith("model.language_model.layers."):
            return key.replace("model.language_model.layers.", "backbone.layers.", 1)

        # model.language_model.norm.* → backbone.norm.*
        if key.startswith("model.language_model.norm."):
            return key.replace("model.language_model.norm.", "backbone.norm.", 1)

        # local_transformer.layers.{i}.* → local_model.layers.{i}.*
        if key.startswith("local_transformer.layers."):
            return key.replace("local_transformer.layers.", "local_model.layers.", 1)

        # local_transformer.norm.* → local_model.norm.*
        if key.startswith("local_transformer.norm."):
            return key.replace("local_transformer.norm.", "local_model.norm.", 1)

        # These pass through directly:
        # speech_embedding_to_local_mlp.*
        # local_to_speech_embedding_mlps.*
        # layer_norm_before_lm_heads.*
        # lm_heads.*
        if key.startswith(
            (
                "speech_embedding_to_local_mlp.",
                "local_to_speech_embedding_mlps.",
                "layer_norm_before_lm_heads.",
                "lm_heads.",
            )
        ):
            return key

        return None

    # ---------------------------------------------------------------------------
    # Properties required by BaseLMWithDepth
    # ---------------------------------------------------------------------------

    @property
    def supports_audio_input(self) -> bool:
        return True

    @property
    def n_codebooks(self) -> int:
        return 1 + self.config.n_vq  # 33

    @property
    def depth_n_codebooks(self) -> int:
        return 1 + self.config.n_vq  # 33 (loop runs i=1..32 → 32 audio codebooks)

    @property
    def num_attention_heads(self) -> int:
        return self._num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self._num_key_value_heads

    @property
    def num_hidden_layers(self) -> int:
        return self._num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def depth_num_attention_heads(self) -> int:
        return self.config.language_config.num_attention_heads

    @property
    def depth_num_key_value_heads(self) -> int:
        return self.config.language_config.num_key_value_heads

    @property
    def depth_num_hidden_layers(self) -> int:
        return self.config.local_num_layers

    @property
    def depth_hidden_size(self) -> int:
        return self.config.local_hidden_size

    @property
    def depth_head_dim(self) -> int:
        return self.config.language_config.head_dim

    @property
    def depth_vocab_size(self) -> int:
        return self.config.audio_vocab_size + 1  # 1025

    @property
    def vocab_size(self) -> int:
        return self.config.audio_vocab_size + 1  # 1025 (for dummy backbone logits)

    @property
    def detokenize_interval(self) -> int:
        return 10

    @property
    def detokenize_overlap(self) -> int:
        return 0

    @property
    def n_channels(self) -> int:
        return 1

    @property
    def output_audio_length(self) -> int:
        return 19200  # 10 frames * 1920 samples/frame

    @property
    def max_tokens(self) -> int:
        if self.default_sampling_config.max_tokens is not None:
            return self.default_sampling_config.max_tokens
        return 2048

    # ---------------------------------------------------------------------------
    # Core methods
    # ---------------------------------------------------------------------------

    def is_stop_id(self, token_ids: List[int]) -> bool:
        return token_ids[0] == self.audio_end_token_id

    def preprocess(self, prompt: str = None, audio_path: str = None, **kwargs) -> PreprocessOutput:
        """Build unified_codes from text prompt using MOSS chat template."""
        instruction = kwargs.get("instruct", None)
        language = kwargs.get("language", None)

        # Encode reference audio if provided
        audio_codes = None
        if audio_path is not None:
            audio_codes = self._load_and_encode_reference_audio(audio_path)  # (T_audio, 32)

        # Build user content
        user_content = self._build_user_content(
            prompt=prompt,
            has_reference=(audio_codes is not None),
            instruction=instruction,
            language=language,
        )

        # Apply chat template
        messages = [{"role": "user", "content": user_content}]
        content = self.text_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if audio_codes is not None:
            # Replace <|audio|> placeholder with slot token sequence
            content = self._replace_audio_placeholder(content, len(audio_codes))

        text_token_ids = self.text_tokenizer.encode(content)

        if audio_codes is not None:
            unified_codes = self._build_unified_codes_with_audio(text_token_ids, audio_codes)
        else:
            unified_codes = self._build_unified_codes_text_only(text_token_ids)

        return PreprocessOutput(input_tokens=unified_codes)

    def _build_user_content(
        self,
        prompt: str,
        has_reference: bool = False,
        instruction: str = None,
        tokens: int = None,
        quality: str = None,
        language: str = None,
    ) -> str:
        """Build the user instruction content for MOSS TTS."""
        reference = "[S1]:\n<|audio|>" if has_reference else "None"
        return (
            "<user_inst>\n"
            f"- Reference(s):\n{reference}\n"
            f"- Instruction:\n{instruction if instruction is not None else 'None'}\n"
            f"- Tokens:\n{tokens if tokens is not None else 'None'}\n"
            f"- Quality:\n{quality if quality is not None else 'None'}\n"
            "- Sound Event:\nNone\n"
            "- Ambient Sound:\nNone\n"
            f"- Language:\n{language if language is not None else 'None'}\n"
            f"- Text:\n{prompt}\n"
            "</user_inst>"
        )

    def _load_and_encode_reference_audio(self, audio_path: str) -> torch.Tensor:
        """Load, preprocess, and encode reference audio for voice cloning.

        Returns:
            (T_audio, 32) audio codes tensor on CPU.
        """
        import torchaudio

        wav, sr = torchaudio.load(audio_path)

        # Stereo to mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Resample to 24kHz if needed
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=24000)

        # Loudness normalize (operates on 1D)
        wav = self._loudness_normalize(wav.squeeze(0)).unsqueeze(0)  # (1, audio_length)

        return self.audio_decoder.encode(wav)  # (T_audio, 32)

    @staticmethod
    def _loudness_normalize(
        wav: torch.Tensor,
        target_dbfs: float = -20,
        max_gain_db: float = 3,
    ) -> torch.Tensor:
        """Normalize audio loudness to target dBFS, clamping gain."""
        wav = wav.to(torch.float32)
        if wav.numel() == 0:
            return wav
        current_dbfs = 10.0 * torch.log10(torch.mean(wav**2) + 1e-9)
        gain = float(target_dbfs - current_dbfs)
        gain = max(-max_gain_db, min(gain, max_gain_db))
        factor = 10.0 ** (gain / 20.0)
        return wav * factor

    def _replace_audio_placeholder(self, content: str, audio_length: int) -> str:
        """Replace <|audio|> placeholder with audio slot token sequence."""
        audio_start_token = self.text_tokenizer.convert_ids_to_tokens(self.config.audio_start_token_id)
        audio_end_token = self.text_tokenizer.convert_ids_to_tokens(self.config.audio_end_token_id)
        audio_user_slot_token = self.text_tokenizer.convert_ids_to_tokens(self.config.audio_user_slot_token_id)
        replacement = audio_start_token + audio_user_slot_token * audio_length + audio_end_token
        return content.replace("<|audio|>", replacement, 1)

    def _build_unified_codes_with_audio(self, text_token_ids: list, audio_codes: torch.Tensor) -> torch.Tensor:
        """Build unified codes with reference audio.

        Args:
            text_token_ids: Token IDs from tokenizer.encode()
            audio_codes: (T_audio, 32) audio codes on CPU

        Returns:
            (T+1, 33) unified codes tensor on device
        """
        n_cb = self.n_codebooks  # 33
        text_codes = torch.tensor(text_token_ids, dtype=torch.long, device=self.device)
        seq_len = len(text_codes)

        # Find audio_start and audio_end positions
        audio_start_positions = (text_codes == self.config.audio_start_token_id).nonzero(as_tuple=True)[0]
        audio_end_positions = (text_codes == self.config.audio_end_token_id).nonzero(as_tuple=True)[0]
        audio_start_idx = int(audio_start_positions[0].item())
        audio_end_idx = int(audio_end_positions[0].item())

        # Build audio channels (T, 32) - all pad initially
        audio_channels = torch.full(
            (seq_len, self.config.n_vq),
            self.config.audio_pad_code,
            dtype=torch.long,
            device=self.device,
        )

        # Insert actual audio codes between start and end markers
        audio_codes_device = audio_codes.to(self.device)
        audio_channels[audio_start_idx + 1 : audio_start_idx + 1 + len(audio_codes)] = audio_codes_device

        # Concatenate: text channel (T, 1) + audio channels (T, 32) → (T, 33)
        unified_codes = torch.cat([text_codes.unsqueeze(1), audio_channels], dim=1)

        # Append audio_start_token row for generation trigger
        audio_start_row = torch.full((1, n_cb), self.config.audio_pad_code, dtype=torch.long, device=self.device)
        audio_start_row[0, 0] = self.config.audio_start_token_id
        unified_codes = torch.cat([unified_codes, audio_start_row], dim=0)

        return unified_codes

    def _build_unified_codes_text_only(self, text_token_ids: list) -> torch.Tensor:
        """Build unified codes for text-only mode (no reference audio).

        Returns:
            (T+1, 33) unified codes tensor on device
        """
        seq_len = len(text_token_ids)
        n_cb = self.n_codebooks  # 33

        unified_codes = torch.full(
            (seq_len, n_cb),
            self.config.audio_pad_code,
            dtype=torch.long,
            device=self.device,
        )
        unified_codes[:, 0] = torch.tensor(text_token_ids, dtype=torch.long, device=self.device)

        # Append audio_start_token row
        audio_start_row = torch.full((1, n_cb), self.config.audio_pad_code, dtype=torch.long, device=self.device)
        audio_start_row[0, 0] = self.config.audio_start_token_id
        unified_codes = torch.cat([unified_codes, audio_start_row], dim=0)

        return unified_codes

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the backbone.

        Sums embeddings from all 33 channels, runs backbone, returns dummy logits + hidden states.
        """
        bs = input_ids.shape[0]

        # Sum embeddings from all channels
        inputs_embeds = torch.zeros(bs, self._hidden_size, device=self.device, dtype=self.dtype)
        for i in range(self.n_codebooks):
            inputs_embeds = inputs_embeds + self.model.backbone.embedding_list[i](input_ids[:, i])

        # Run backbone
        hidden_states = self.model.forward_backbone(inputs_embeds, position_ids, attn_wrapper, kv_cache)

        # Return dummy logits (backbone doesn't predict directly; text prediction happens in sampling)
        dummy_logits = torch.zeros(bs, 1, self.depth_vocab_size, device=self.device, dtype=self.dtype)

        return dummy_logits, hidden_states

    def sampling(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        repetition_cache: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample text token via SDPA, then set up depth transformer input."""
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        bs = hidden_states.shape[0]

        # Predict text token using local transformer with SDPA
        text_logits = self.model.text_prediction_sdpa(hidden_states)  # (bs, text_vocab_size)

        # Sample text token
        text_token_ids = Sampler.run_sampling(text_logits, config=sampling_params)  # (bs,)

        # Allocate output_ids for all codebooks
        output_ids = torch.full(
            (bs, self.n_codebooks),
            self.config.audio_pad_code,
            dtype=torch.long,
            device=self.device,
        )
        output_ids[:, 0] = text_token_ids

        # Embed text token and project to backbone embedding space
        text_embed = self.model.backbone.embedding_list[0](text_token_ids)  # (bs, D_global)

        # hidden_for_depth = [hidden_states, text_embed] as prefill (bs, 2, D_global)
        hidden_for_depth = torch.stack([hidden_states, text_embed], dim=1)  # (bs, 2, D_global)

        # Update request state
        for i, req in enumerate(requests):
            req.input_tokens = torch.zeros(1, self.n_codebooks, dtype=torch.long, device=self.device)
            req.input_tokens[0, 0] = text_token_ids[i].item()
            # Fill audio channels with pad
            req.input_tokens[0, 1:] = self.config.audio_pad_code

            req.lm_output_tokens.append(output_ids[i : i + 1])

            # Check stop condition
            if text_token_ids[i].item() == self.audio_end_token_id:
                req.done_lm_generation = True
                req.finish_reason = "stop_id_encountered"
            elif req.next_position_id > self.max_tokens:
                req.done_lm_generation = True
                req.finish_reason = "max_tokens_reached"
            else:
                req.lm_output_audio_tokens.append(output_ids[i : i + 1])

        return output_ids, hidden_for_depth

    def depth_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the depth (local) transformer."""
        return self.model.forward_depth(hidden_states, position_ids, attn_wrapper, kv_cache)

    def depth_sampling(
        self,
        logits: torch.Tensor,
        i_iteration: int,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        cfg_scale: float | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample audio token from depth logits."""
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        # Suppress audio_pad_code (1024) in logits
        logits[:, self.config.audio_pad_code] = -torch.inf

        # Sample audio token
        output_ids = Sampler.run_sampling(logits, config=sampling_params)  # (bs,)

        # Embed via backbone embedding_list[i_iteration] → (bs, D_global)
        ci_embed = self.model.backbone.embedding_list[i_iteration](output_ids)

        # Update request state
        for i, req in enumerate(requests):
            token_id = output_ids[i].item()
            req.input_tokens[0, i_iteration] = token_id
            req.lm_output_tokens[-1][0, i_iteration] = token_id
            if len(req.lm_output_audio_tokens) > 0:
                req.lm_output_audio_tokens[-1][0, i_iteration] = token_id

        return output_ids, ci_embed

    def postprocess(self, token_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convert audio tokens to waveform.

        Args:
            token_ids: (batch, interval, 33)

        Returns:
            (batch, 1, audio_length) waveform tensor
        """
        # Extract audio channels: cols 1-32
        audio_tokens = token_ids[:, :, 1:]  # (batch, interval, 32)
        audio_tokens = audio_tokens.transpose(1, 2)  # (batch, 32, interval)
        audio_tokens = audio_tokens.clamp(0, self.config.audio_vocab_size - 1)  # clamp to [0, 1023]

        audio_tokens = audio_tokens.to(self.audio_decoder_device)
        audio_tensor = self.audio_decoder.decode(audio_tokens)  # (batch, 1, audio_length)
        return audio_tensor
