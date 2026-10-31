"""
Voxtral-TTS model for vox-serve.

Two parts:

1. Mistral-style backbone classes (``VoxtralAttention`` / ``VoxtralDecoderLayer`` /
   ``VoxtralBackboneModel``) mirroring ``cosyvoice2.py``'s Qwen2 backbone, built on
   ``FlashInferWrapper`` / ``apply_rope_pos_ids`` / ``rms_norm``. Voxtral's text
   decoder is structurally Mistral (bias-free attention, RoPE, RMSNorm).

2. ``VoxtralTTSModel(BaseLM)`` -- the inline-audio-head TTS model. The backbone
   never emits real text logits; instead a fake-EOS signal is produced by the
   acoustic flow-matching transformer. Semantic-codebook selection is pure argmax
   (no repetition penalty), and the acoustic codebooks come from an Euler ODE.

The checkpoint is Mistral-format (``consolidated.safetensors`` + ``params.json`` +
``tekken.json``), NOT HF -- ``load_state_dict`` below remaps the flat Mistral key
names onto the module hierarchy.

Reference: vllm-omni ``voxtral_tts_audio_generation.py`` / ``voxtral_tts.py``.
Interface contract: ``voxtral_contract.md`` (FROZEN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
from torch import nn
from torch.cuda import CUDAGraph

if TYPE_CHECKING:
    from ...tokenizer.voxtral_tts import FlowMatchingAudioTransformer

from ...flashinfer_utils import (
    FlashInferWrapper,
    rms_norm,
)
from ...sampling import SamplingConfig
from ...utils import get_logger
from ..base import BaseLM

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pinned config (contract section 1)
# ---------------------------------------------------------------------------

# Audio special tokens predicted by the semantic codebook head. Output audio
# tokens from the quantizer are offset by ``len(_AUDIO_SPECIAL_TOKENS)``.
# Order matters: [EMPTY_AUDIO]=0, [END_AUDIO]=1 (mirrors vllm-omni).
_AUDIO_SPECIAL_TOKENS = ("[EMPTY_AUDIO]", "[END_AUDIO]")
_EMPTY_AUDIO_TOKEN_ID = 0
_END_AUDIO_TOKEN_ID = 1

# Voice presets (contract section 1.6): name -> index. The index is unused at
# runtime (embeddings are looked up by name) but kept for completeness.
VOXTRAL_VOICE_PRESETS = [
    "casual_female",
    "casual_male",
    "cheerful_female",
    "neutral_female",
    "neutral_male",
    "pt_male",
    "pt_female",
    "nl_male",
    "nl_female",
    "it_male",
    "it_female",
    "fr_male",
    "fr_female",
    "es_male",
    "es_female",
    "de_male",
    "de_female",
    "ar_male",
    "hi_male",
    "hi_female",
]


@dataclass
class VoxtralBackboneConfig:
    """Backbone (text decoder) config -- contract section 1.1 / ``params.json`` top level."""

    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_hidden_layers: int = 26
    head_dim: int = 128
    vocab_size: int = 131072
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-05
    max_position_embeddings: int = 128000

    # Audio codebook structure -- contract section 1.3
    n_codebooks: int = 37
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    condition_dropped_token_id: int = 42
    # ``mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight`` is
    # [9088, 3072] in the checkpoint -- a single shared embedding table over all
    # codebooks.
    audio_codebook_embedding_size: int = 9088


# ---------------------------------------------------------------------------
# Part 1 -- Mistral backbone classes (mirror cosyvoice2.py:122-239)
# ---------------------------------------------------------------------------


class VoxtralRMSNorm(nn.Module):
    """RMSNorm wrapper over the FlashInfer ``rms_norm`` kernel (PRE-norm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
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


class VoxtralMLP(nn.Module):
    """Bias-free SwiGLU MLP. Mistral key names: ``w1`` (gate), ``w3`` (up), ``w2`` (down)."""

    def __init__(self, config: VoxtralBackboneConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _voxtral_apply_rope_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    rope_theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32-precision RoPE for Voxtral attention.

    FlashInfer's ``apply_rope_pos_ids`` runs the rotation in the working dtype
    (bf16). Voxtral uses ``rope_theta=1e6``, so the inverse frequencies span
    ``1.0..1e-6`` -- the high-index frequencies quantize hard in bf16 and the
    resulting cos/sin phase drift compounds across 26 layers. vllm-omni
    computes cos/sin in fp32 and casts back; this helper reproduces that.

    Layer-by-layer diff (vox-serve vs vllm-omni) shows the first layer already
    diverges (cos sim 0.97 vs 1.0); each subsequent layer amplifies. Switching
    Voxtral's attention to this fp32-RoPE closes the gap.

    Args:
        q, k: (..., n_heads, head_dim) in working dtype (bf16).
        position_ids: (...,) int. Leading shape matches q/k's leading shape.
        rope_theta: base frequency.

    Returns rotated (q, k) in the original dtype.
    """
    head_dim = q.shape[-1]
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim)
    )
    # (...,) -> (..., 1) * (head_dim/2,) -> (..., head_dim/2)
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
    # standard Mistral pattern: concat [freqs, freqs] -> (..., head_dim)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(-2)  # (..., 1, head_dim) -- broadcast over n_heads
    sin = emb.sin().unsqueeze(-2)

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    q_rot = q_fp32 * cos + _rotate_half(q_fp32) * sin
    k_rot = k_fp32 * cos + _rotate_half(k_fp32) * sin
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


class VoxtralAttention(nn.Module):
    """Bias-free GQA attention with RoPE, mirroring ``CosyVoice2Attention``.

    The only structural difference from CosyVoice2 is that Voxtral is fully
    bias-free (``use_biases=false`` in ``params.json``), so q/k/v projections
    have ``bias=False``.
    """

    def __init__(self, config: VoxtralBackboneConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.rope_theta = config.rope_theta

        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        # bias-free attention (Mistral-style)
        self.q_proj = nn.Linear(config.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states, key_states = _voxtral_apply_rope_fp32(
            q=query_states,
            k=key_states,
            position_ids=position_ids,
            rope_theta=self.rope_theta,
        )

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class VoxtralDecoderLayer(nn.Module):
    """Pre-norm Mistral decoder layer, mirroring ``CosyVoice2DecoderLayer``."""

    def __init__(self, config: VoxtralBackboneConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = VoxtralAttention(config=config, layer_idx=layer_idx)
        self.mlp = VoxtralMLP(config)

        self.input_layernorm = VoxtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = VoxtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class VoxtralBackboneModel(nn.Module):
    """The Mistral-style text decoder stack, mirroring ``CosyVoice2BackboneModel``.

    Holds two embedding tables:
    - ``embed_tokens``: the text token embedding (``mm_audio_embeddings.tok_embeddings``
      in the checkpoint).
    - ``audio_codebook_embeddings``: the shared audio-codebook embedding table
      (``mm_audio_embeddings.audio_codebook_embeddings.embeddings`` in the
      checkpoint), used to embed previously generated audio codes during decode.

    ``lm_head`` is intentionally NOT instantiated: the model never emits real text
    logits (only a fake-EOS signal from the acoustic head).
    """

    def __init__(self, config: VoxtralBackboneConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_codebook_embedding_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [VoxtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = VoxtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        import os
        dump = os.environ.get('VOXTRAL_DUMP_LAYERS') == '1' and not getattr(self, '_dump_done', False)
        captures = []  # list of (idx, last-token tensor)
        hidden_states = inputs_embeds
        if dump:
            captures.append(('input_embeds', hidden_states[..., -1, :].detach().clone().float().cpu()))

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )
            if dump:
                captures.append((f'layer_{i}', hidden_states[..., -1, :].detach().clone().float().cpu()))

        hidden_states = self.norm(hidden_states)
        if dump:
            captures.append(('final_norm', hidden_states[..., -1, :].detach().clone().float().cpu()))
            self._dump_done = True
            torch.save(captures, '/tmp/voxserve_layer_dumps.pt')
        return hidden_states

    @torch.no_grad()
    def load_state_dict_from_mistral(self, mistral_state: dict) -> None:
        """Remap a flat Mistral-format state dict onto this module hierarchy.

        The Voxtral checkpoint (``consolidated.safetensors``) stores the text
        decoder with flat key names::

            mm_audio_embeddings.tok_embeddings.weight                          -> embed_tokens.weight
            mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight     -> audio_codebook_embeddings.weight
            layers.{i}.attention.wq.weight                                     -> layers.{i}.self_attn.q_proj.weight
            layers.{i}.attention.wk.weight                                     -> layers.{i}.self_attn.k_proj.weight
            layers.{i}.attention.wv.weight                                     -> layers.{i}.self_attn.v_proj.weight
            layers.{i}.attention.wo.weight                                     -> layers.{i}.self_attn.o_proj.weight
            layers.{i}.feed_forward.w1.weight                                  -> layers.{i}.mlp.gate_proj.weight
            layers.{i}.feed_forward.w3.weight                                  -> layers.{i}.mlp.up_proj.weight
            layers.{i}.feed_forward.w2.weight                                  -> layers.{i}.mlp.down_proj.weight
            layers.{i}.attention_norm.weight                                   -> layers.{i}.input_layernorm.weight
            layers.{i}.ffn_norm.weight                                         -> layers.{i}.post_attn_layernorm
            norm.weight                                                        -> norm.weight

        All other keys (acoustic_transformer.*, audio_tokenizer.*) belong to P1's
        modules and are ignored here. ``output.weight`` (tied lm_head) is skipped.
        """
        remapped: dict[str, torch.Tensor] = {}

        attn_map = {
            "wq": "self_attn.q_proj",
            "wk": "self_attn.k_proj",
            "wv": "self_attn.v_proj",
            "wo": "self_attn.o_proj",
        }
        ffn_map = {
            "w1": "mlp.gate_proj",
            "w3": "mlp.up_proj",
            "w2": "mlp.down_proj",
        }
        norm_map = {
            "attention_norm": "input_layernorm",
            "ffn_norm": "post_attention_layernorm",
        }

        cfg = self.config
        n_heads_q = cfg.num_attention_heads
        n_heads_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim

        def _permute_for_rope(w: torch.Tensor, n_heads: int) -> torch.Tensor:
            # Mistral consolidated.safetensors stores wq/wk pair-interleaved on the
            # head_dim axis: rows [a0,b0,a1,b1,...]. HF/vLLM-omni's RoPE expects
            # split-half layout: rows [a0,a1,...,b0,b1,...]. Mirrors vllm's
            # `permute()` in vllm/model_executor/models/mistral.py.
            out_dim, in_dim = w.shape
            assert out_dim == n_heads * head_dim, (out_dim, n_heads, head_dim)
            return (
                w.view(n_heads, head_dim // 2, 2, in_dim)
                .transpose(1, 2)
                .reshape(out_dim, in_dim)
            )

        for key, value in mistral_state.items():
            if key == "mm_audio_embeddings.tok_embeddings.weight":
                remapped["embed_tokens.weight"] = value
                continue
            if key == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight":
                remapped["audio_codebook_embeddings.weight"] = value
                continue
            if key == "norm.weight":
                remapped["norm.weight"] = value
                continue
            if key in ("output.weight", "tok_embeddings.weight"):
                # tied lm_head / alt text-embed key: SKIP (no lm_head in this model)
                continue
            if not key.startswith("layers."):
                # acoustic_transformer.* / audio_tokenizer.* belong to other modules
                continue

            parts = key.split(".")
            # parts == ["layers", "{i}", <sub>, ...]
            layer_idx = parts[1]
            sub = parts[2]
            if sub == "attention":
                proj = parts[3]  # wq / wk / wv / wo
                if proj not in attn_map:
                    continue
                if proj == "wq" and parts[4] == "weight":
                    value = _permute_for_rope(value, n_heads_q)
                elif proj == "wk" and parts[4] == "weight":
                    value = _permute_for_rope(value, n_heads_kv)
                remapped[f"layers.{layer_idx}.{attn_map[proj]}.{parts[4]}"] = value
            elif sub == "feed_forward":
                proj = parts[3]  # w1 / w2 / w3
                if proj not in ffn_map:
                    continue
                remapped[f"layers.{layer_idx}.{ffn_map[proj]}.{parts[4]}"] = value
            elif sub in norm_map:
                remapped[f"layers.{layer_idx}.{norm_map[sub]}.{parts[3]}"] = value
            # else: unknown layer-local key -> ignore

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing:
            logger.warning("VoxtralBackboneModel: missing keys after remap: %s", missing)
        if unexpected:
            logger.warning("VoxtralBackboneModel: unexpected keys after remap: %s", unexpected)


# ---------------------------------------------------------------------------
# Part 2 -- VoxtralTTSModel(BaseLM)
# ---------------------------------------------------------------------------


class VoxtralTTSModel(BaseLM):
    """Voxtral-4B-TTS inline-audio-head model.

    Mirrors the structural surface of ``CosyVoice2Model`` but:
    - the backbone produces hidden states, not logits (``has_inline_audio_head``);
    - sampling runs its own semantic argmax + acoustic ODE (no FlashInfer sampler);
    - audio output codes are shaped ``(B, 37)`` per step, postprocessed through a
      37-codebook audio tokenizer with a model-private 25-frame left-context ring
      buffer.
    """

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
        enable_torch_compile: bool = False,
        audio_decoder_device: str | None = None,
    ):
        if model_name == "voxtral-tts":
            model_name = "mistralai/Voxtral-4B-TTS-2603"
        super().__init__(model_name, device, dtype, enable_torch_compile, audio_decoder_device)
        self.logger = logger

        self.config = VoxtralBackboneConfig()

        # ----- pinned audio constants (contract sections 1.3 / 1.5) -----
        self.audio_token_id = self.config.audio_token_id
        self.begin_audio_token_id = self.config.begin_audio_token_id
        self.downsample_factor = 1920  # sampling_rate / frame_rate = 24000 / 12.5

        # ----- default sampling config (contract section 4) -----
        self.default_sampling_config = SamplingConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            repetition_penalty=None,  # gap fix #2/#11 -- pure argmax semantic selection
            cfg_scale=1.2,
            greedy=True,
        )

        # ----- backbone -----
        self.model = VoxtralBackboneModel(self.config)
        self._load_backbone_weights(model_name)
        self.model.to(dtype).to(device)

        # ----- acoustic transformer + audio tokenizer (P1's modules) -----
        # Imported here (function-local) so this module can still be imported and
        # smoke-tested before P1's file merges. Resolves at merge time (Wave 2).
        from ...tokenizer.voxtral_tts import (  # noqa: PLC0415
            FlowMatchingAudioTransformer,
            VoxtralTTSAudioTokenizer,
        )

        self.acoustic_transformer, self.audio_tokenizer = self._load_audio_modules(
            model_name, FlowMatchingAudioTransformer, VoxtralTTSAudioTokenizer
        )

        # Per-codebook base offsets into the shared ``audio_codebook_embeddings``
        # table, used to embed a previously-generated 37-codebook audio frame
        # during decode. Mirrors vllm-omni's ``MultiVocabEmbeddings.offsets``
        # (``cumsum([0] + codebook_sizes[:-1])`` over the special-token-inclusive,
        # unpadded codebook sizes). The audio tokenizer's ``audio_token_embedding``
        # already computed these from the same ``audio_model_args``; reuse them so
        # the offsets cannot drift from the embedding table layout.
        self._audio_codebook_offsets = (
            self.audio_tokenizer.audio_token_embedding.offsets.to(self.device).long()
        )  # (n_codebooks,)

        # ----- tokenizer (mistral-common, tekken.json) -----
        self.text_tokenizer = self._load_tokenizer(model_name)

        # ----- voice registry (contract section 1.6) -----
        self.voice_to_embedding: dict[str, torch.Tensor] = {}
        self._load_voice_registry(model_name)

        # ----- backbone shape metadata -----
        self._num_attention_heads = self.config.num_attention_heads
        self._num_key_value_heads = self.config.num_key_value_heads
        self._num_hidden_layers = self.config.num_hidden_layers
        self._hidden_size = self.config.hidden_size

        # ----- Phase-4 acoustic CUDA graph hook (set by enable_acoustic_graph) -----
        self._acoustic_graph = None

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _load_backbone_weights(self, model_name: str) -> None:
        """Load the Mistral-format text decoder weights into ``self.model``."""
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415

        ckpt_path = hf_hub_download(repo_id=model_name, filename="consolidated.safetensors")
        full_state = load_file(ckpt_path, device="cpu")
        self.model.load_state_dict_from_mistral(full_state)

    def _load_audio_modules(self, model_name: str, flow_cls, tokenizer_cls):
        """Construct + weight-load the acoustic transformer and audio tokenizer.

        P1 owns ``FlowMatchingAudioTransformer`` / ``VoxtralTTSAudioTokenizer`` and
        their exact constructor signatures. We pass the relevant sub-dicts from
        ``params.json``. If P1's constructors expect a different shape this is the
        single place to adapt -- the rest of this model only depends on the
        contract's public methods (``forward`` / ``decode``).
        """
        import json  # noqa: PLC0415

        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415

        params_path = hf_hub_download(repo_id=model_name, filename="params.json")
        with open(params_path) as f:
            params = json.load(f)
        audio_model_args = params["multimodal"]["audio_model_args"]
        audio_tokenizer_args = params["multimodal"]["audio_tokenizer_args"]

        # n_decoding_steps is not in params.json; pin to 7 (contract section 1.2).
        acoustic_args = dict(audio_model_args)
        acoustic_args.setdefault("acoustic_transformer_args", {})
        acoustic_args["acoustic_transformer_args"] = dict(acoustic_args["acoustic_transformer_args"])
        acoustic_args["acoustic_transformer_args"].setdefault("n_decoding_steps", 7)

        acoustic_transformer = flow_cls(acoustic_args)
        # VoxtralTTSAudioTokenizer expects a config dict with ``codec_args`` +
        # ``audio_model_args`` sub-dicts and the backbone hidden size.
        audio_tokenizer = tokenizer_cls(
            {
                "codec_args": audio_tokenizer_args,
                "audio_model_args": acoustic_args,
            },
            text_hidden_size=self.config.hidden_size,
        )

        # Weight load: split the flat checkpoint by prefix into per-module state
        # dicts, then load each module from its subset.
        ckpt_path = hf_hub_download(repo_id=model_name, filename="consolidated.safetensors")
        full_state = load_file(ckpt_path, device="cpu")
        acoustic_state: dict[str, torch.Tensor] = {}
        tokenizer_state: dict[str, torch.Tensor] = {}
        for name, weight in full_state.items():
            if name.startswith("acoustic_transformer."):
                acoustic_state[name[len("acoustic_transformer.") :]] = weight
            elif name.startswith("audio_tokenizer."):
                tokenizer_state[name[len("audio_tokenizer.") :]] = weight

        a_missing, a_unexpected = acoustic_transformer.load_state_dict(acoustic_state, strict=False)
        if a_missing:
            self.logger.warning("acoustic_transformer missing keys: %s", a_missing)
        if a_unexpected:
            self.logger.warning("acoustic_transformer unexpected keys: %s", a_unexpected)
        audio_tokenizer.load_weights(tokenizer_state, strict=False)

        acoustic_transformer = acoustic_transformer.to(self.dtype).to(self.device)
        audio_tokenizer = audio_tokenizer.to(self.dtype).to(self.audio_decoder_device)
        return acoustic_transformer, audio_tokenizer

    def _load_tokenizer(self, model_name: str):
        """Load the Mistral tokenizer (tekken.json) via mistral-common.

        mistral-common >= 1.11 parses the Voxtral-TTS ``tekken.json`` natively
        (including the ``voice_num_audio_tokens`` audio-block field) and exposes
        ``encode_speech_request``, which produces the authoritative TTS prompt
        token layout used by ``preprocess``.
        """
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # noqa: PLC0415

        tekken_path = hf_hub_download(repo_id=model_name, filename="tekken.json")
        return MistralTokenizer.from_file(tekken_path)

    def _load_voice_registry(self, model_name: str) -> None:
        """Download the 20 voice-embedding presets into ``self.voice_to_embedding``."""
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        for name in VOXTRAL_VOICE_PRESETS:
            try:
                path = hf_hub_download(repo_id=model_name, filename=f"voice_embedding/{name}.pt")
                self.voice_to_embedding[name] = torch.load(path, map_location="cpu")
            except Exception:  # noqa: BLE001
                self.logger.warning("Voxtral voice embedding not found: %s", name)
        self.logger.info("Voxtral available voice embeddings: %s", list(self.voice_to_embedding.keys()))

    # ------------------------------------------------------------------
    # BaseLM properties
    # ------------------------------------------------------------------

    @property
    def n_codebooks(self) -> int:
        return 37

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
        # NOTE: Voxtral's attention uses an explicit head_dim of 128, so the
        # per-head qkv dimension (num_attention_heads * head_dim = 32 * 128 =
        # 4096) does NOT equal the residual-stream width (config.hidden_size =
        # 3072). This property reports the attention qkv width (4096) used to
        # size the FlashInfer wrappers; ``embedding_hidden_size`` reports the
        # residual-stream width (3072) used to size the input_features /
        # backbone_hidden_states CUDA-graph buffers. All model-internal modules
        # use ``self.config.hidden_size`` (3072) directly.
        return self._num_attention_heads * self.config.head_dim

    @property
    def head_dim(self) -> int:
        # Explicit head_dim from params.json (128), not hidden_size // n_heads.
        return self.config.head_dim

    @property
    def detokenize_interval(self) -> int:
        return 25

    @property
    def first_chunk_frames(self) -> int:
        """Emit the first audio chunk after 5 real frames instead of 25 (TTFA gain).

        Mirrors vllm-omni's ``codec_chunk_frames_at_begin=5``
        (``vllm_omni/deploy/voxtral_tts.yaml``). The model pre-seeds 20 zero-coded
        silence frames in ``sampling`` before the first real frame is appended;
        the worker trims those samples from the resulting PCM. See
        ``BaseLM.first_chunk_frames`` for the contract.
        """
        return 5

    @property
    def detokenize_overlap(self) -> int:
        # left-context is a model-private ring buffer, NOT scheduler overlap
        return 0

    @property
    def n_channels(self) -> int:
        return 1

    @property
    def output_audio_length(self) -> int:
        # 25 frames * 1920 samples/frame
        return 48000

    @property
    def max_tokens(self) -> int:
        if self.default_sampling_config.max_tokens is not None:
            return self.default_sampling_config.max_tokens
        return 4096

    @property
    def vocab_size(self) -> int:
        return 131072

    # ------------------------------------------------------------------
    # BaseLM methods
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase-4 acoustic CUDA graph hook
    # ------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Acoustic-head CUDA graph (was vox_serve/model/voxtral_tts_acoustic_graph.py)
# ---------------------------------------------------------------------------


class AcousticHeadCudaGraph:
    """
    CUDA Graph wrapper for the acoustic flow-matching transformer.

    Ported from vllm-omni's ``CUDAGraphAcousticTransformerWrapper``. Captures the
    acoustic transformer forward pass (semantic logit + n-step Euler ODE with CFG)
    into CUDA graphs for fixed batch sizes, eliminating kernel launch overhead on
    every decode step. Replaces the eager sampler path (Python-level branching and
    dynamic tensor allocation) with a CUDA-graph-compatible path using
    ``torch.argmax`` (equivalent to ``top_k=1`` greedy sampling) and pre-allocated
    static buffers.

    Deviations from the vllm-omni original (per the Voxtral-TTS interface contract):
    - ``capture_sizes`` is a constructor argument (gap fix #1), not a hardcoded
      ``DEFAULT_CAPTURE_SIZES`` class attribute.
    - ``graph_pool`` is an injected constructor argument (gap fix), replacing
      ``vllm.platforms.current_platform.get_global_graph_pool()``.
    - ``static_noise`` is filled via ``static_noise.normal_(generator=...)`` with a
      seeded ``torch.Generator`` before each replay (gap fix #8), so RNG state is
      not baked into the captured graph.

    Args:
        flow_transformer: the acoustic flow-matching transformer
            (``FlowMatchingAudioTransformer``). Provides ``model_args``,
            ``acoustic_transformer_args``, ``acoustic_embeddings_levels``,
            ``semantic_codebook_output``, ``time_embedding`` and
            ``_predict_velocity``.
        graph_pool: the CUDA graph memory pool handle to capture into.
        capture_sizes: batch sizes to capture graphs for (gap fix #1).
        device: capture device.
        dtype: capture dtype.
        hidden_dim: backbone hidden size fed into the acoustic transformer.
        seed: seed for the ``torch.Generator`` driving ``static_noise`` refills.
    """

    def __init__(
        self,
        flow_transformer: "FlowMatchingAudioTransformer",
        graph_pool,
        capture_sizes: list[int],
        device: torch.device,
        dtype: torch.dtype,
        hidden_dim: int,
        seed: int = 42,
    ):
        self.acoustic_transformer = flow_transformer
        self.graph_pool = graph_pool

        self.capture_sizes = sorted(capture_sizes)
        self.device = torch.device(device)
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.seed = seed

        # Pre-compute constants from the acoustic transformer. The audio special
        # tokens ([EMPTY_AUDIO]=0, [END_AUDIO]=1) are shared with the model via the
        # module-level ``_AUDIO_SPECIAL_TOKENS`` constants.
        self.empty_audio_token_id = _EMPTY_AUDIO_TOKEN_ID
        self.end_audio_token_id = _END_AUDIO_TOKEN_ID
        self.semantic_mask_start = (
            len(_AUDIO_SPECIAL_TOKENS) + self.acoustic_transformer.model_args.semantic_codebook_size
        )
        self.n_acoustic_codebook = self.acoustic_transformer.model_args.n_acoustic_codebook
        self.acoustic_embeddings_levels = self.acoustic_transformer.acoustic_embeddings_levels

        self.n_steps = self.acoustic_transformer.acoustic_transformer_args.n_decoding_steps

        # Seeded RNG for refilling the noise buffer before each replay (gap fix #8).
        # Kept off the captured graph so random state is not baked in.
        self._noise_generator = torch.Generator(device=self.device)
        self._noise_generator.manual_seed(self.seed)

        # Graph storage
        self.graphs: dict[int, CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_noise: dict[int, torch.Tensor] = {}
        self.static_cfg_alpha: dict[int, torch.Tensor] = {}
        self.static_fake_eos: dict[int, torch.Tensor] = {}
        self.static_audio_codes: dict[int, torch.Tensor] = {}

        self.enabled = False
        self._warmed_up = False

        # Capture eagerly at construction time.
        self._warmup_and_capture(self.device, self.dtype, self.hidden_dim)

    def _warmup_and_capture(self, device: torch.device, dtype: torch.dtype, hidden_dim: int):
        """Perform eager warmup and CUDA graph capture for all bucket sizes."""
        if self._warmed_up:
            logger.warning("AcousticHeadCudaGraph already warmed up, skipping")
            return

        logger.info(
            "AcousticHeadCudaGraph: starting warmup and capture for sizes %s",
            self.capture_sizes,
        )

        # Pre-create persistent buffers
        self.timesteps = torch.linspace(0, 1, self.n_steps + 1, device=device, dtype=dtype)
        self.fake_eos_one = torch.tensor(1.0, dtype=dtype, device=device)
        self.fake_eos_zero = torch.tensor(0.0, dtype=dtype, device=device)

        # Phase 1: Eager warmup for ALL capture sizes
        for size in self.capture_sizes:
            dummy = torch.zeros(size, hidden_dim, device=device, dtype=dtype)
            dummy_cfg_alpha = torch.full((size, 1), 1.2, device=device, dtype=dtype)
            dummy_noise = torch.randn(size, self.n_acoustic_codebook, device=device, dtype=dtype)
            with torch.no_grad():
                self._forward_cudagraph_compatible(dummy, cfg_alpha=dummy_cfg_alpha, noise=dummy_noise)

        torch.accelerator.synchronize(device)

        # Phase 2: Capture graphs
        for size in self.capture_sizes:
            try:
                self._capture_graph_for_size(size, device, dtype, hidden_dim)
                logger.info("  Captured CUDA Graph for batch_size=%d", size)
            except Exception:
                logger.warning(
                    "  Failed to capture CUDA Graph for batch_size=%d",
                    size,
                    exc_info=True,
                )

        self.enabled = True
        self._warmed_up = True
        logger.info(
            "AcousticHeadCudaGraph warmup complete. Captured %d/%d graphs.",
            len(self.graphs),
            len(self.capture_sizes),
        )

    def _forward_cudagraph_compatible(
        self,
        hidden_states: torch.Tensor,
        cfg_alpha: torch.Tensor,
        noise: torch.Tensor,
    ):
        """
        The actual computation captured by the CUDA graph.

        This replaces the full ``compute_mm_logits -> acoustic_transformer.forward()``
        path with a graph-compatible version:
        - Uses argmax instead of an eager Sampler (equivalent for top_k=1)
        - Uses pre-created timesteps buffer instead of torch.linspace
        - Uses pre-created scalar tensors for torch.where
        - Calls ``_predict_velocity`` directly
        - Uses a pre-allocated noise buffer to avoid baking random state
          into the CUDA graph
        - Uses a pre-allocated cfg_alpha buffer for per-request CFG strength
        """
        at = self.acoustic_transformer
        B = hidden_states.shape[0]

        # --- Semantic logits via linear projection ---
        semantic_logit = at.semantic_codebook_output(hidden_states).float()
        semantic_logit[:, self.empty_audio_token_id] = -float("inf")
        semantic_logit[:, self.semantic_mask_start :] = -float("inf")

        # argmax == top_k=1 greedy sampling
        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)  # (B, 1)

        # --- Flow matching: Euler ODE ---
        should_decode = semantic_code.squeeze(1) != self.end_audio_token_id

        x = noise

        # Pre-compute zero hidden states for unconditional CFG branch
        hidden_states_zero = torch.zeros_like(hidden_states)

        timesteps = self.timesteps
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            # Batch conditional + unconditional velocity in a single forward pass
            t_emb = at.time_embedding(t.view(-1, 1).repeat(B, 1)).to(hidden_states.dtype)
            x_batched = torch.cat([x, x], dim=0)  # (2B, C)
            llm_batched = torch.cat([hidden_states, hidden_states_zero], dim=0)  # (2B, D)
            t_emb_batched = t_emb.repeat(2, 1)  # (2B, D)

            v_all = at._predict_velocity(x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched)
            v_t, uncond_v_t = v_all[:B], v_all[B:]

            # CFG combination (cfg_alpha is (B, 1), v_t is (B, C))
            v_t = cfg_alpha * v_t + (1 - cfg_alpha) * uncond_v_t

            x = x + v_t * dt

        # --- Quantize ---
        sampled = torch.clamp(x, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = self.empty_audio_token_id
        acoustic_codes = output_codes + len(_AUDIO_SPECIAL_TOKENS)

        # --- Combine semantic + acoustic ---
        audio_codes = torch.cat([semantic_code, acoustic_codes], dim=1)  # (B, 1 + n_acoustic)

        # --- Compute fake_eos ---
        fake_eos = torch.where(
            audio_codes[:, 0] == self.end_audio_token_id,
            self.fake_eos_one,
            self.fake_eos_zero,
        )

        return fake_eos, audio_codes

    def _capture_graph_for_size(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
        hidden_dim: int,
    ):
        """Capture a CUDA graph for a specific batch size."""
        static_input = torch.zeros(size, hidden_dim, device=device, dtype=dtype)
        static_noise = torch.randn(size, self.n_acoustic_codebook, device=device, dtype=dtype)
        static_cfg_alpha = torch.full((size, 1), 1.2, device=device, dtype=dtype)

        # Stabilizing eager run
        with torch.no_grad():
            _ = self._forward_cudagraph_compatible(static_input, cfg_alpha=static_cfg_alpha, noise=static_noise)

        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=self.graph_pool):
                static_fake_eos, static_audio_codes = self._forward_cudagraph_compatible(
                    static_input, cfg_alpha=static_cfg_alpha, noise=static_noise
                )

        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_noise[size] = static_noise
        self.static_cfg_alpha[size] = static_cfg_alpha
        self.static_fake_eos[size] = static_fake_eos
        self.static_audio_codes[size] = static_audio_codes

    def _get_padded_size(self, actual_size: int) -> int | None:
        """Round up to the nearest captured bucket size."""
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def __call__(
        self,
        backbone_hidden_states: torch.Tensor,
        cfg_alpha_per_req: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the acoustic head via a captured CUDA graph.

        Args:
            backbone_hidden_states: ``(B, hidden_dim)`` backbone hidden states.
            cfg_alpha_per_req: ``(B,)`` per-request CFG strength.

        Returns:
            ``(audio_codes, fake_eos)`` where ``audio_codes`` is ``(B, 37)`` int
            and ``fake_eos`` is ``(B,)`` bool.

        Raises:
            RuntimeError: if no captured graph covers the requested batch size,
            or if the wrapper has not been warmed up.
        """
        actual_size = backbone_hidden_states.shape[0]

        if not self.enabled or not self._warmed_up:
            raise RuntimeError("AcousticHeadCudaGraph is not warmed up; no captured graphs available.")

        padded_size = self._get_padded_size(actual_size)
        if padded_size is None or padded_size not in self.graphs:
            raise RuntimeError(
                f"AcousticHeadCudaGraph has no captured graph for batch size {actual_size} "
                f"(capture sizes: {self.capture_sizes})."
            )

        # Zero static input, then copy actual data
        self.static_inputs[padded_size].zero_()
        self.static_inputs[padded_size][:actual_size] = backbone_hidden_states

        # Copy per-request cfg_alpha into static buffer (pad with 1.2 default).
        # The wrapper writes per-request cfg_alpha into static_cfg_alpha[:B, 0].
        self.static_cfg_alpha[padded_size].fill_(1.2)
        self.static_cfg_alpha[padded_size][:actual_size, 0] = cfg_alpha_per_req

        # Fill noise buffer with fresh random values before replay so the
        # flow-matching ODE starts from different initial noise each time.
        # Uses a seeded torch.Generator so RNG is not baked into the graph
        # (gap fix #8).
        self.static_noise[padded_size].normal_(generator=self._noise_generator)

        # Replay captured graph
        self.graphs[padded_size].replay()

        # Clone and slice outputs for actual batch size
        fake_eos = self.static_fake_eos[padded_size][:actual_size].clone().bool()
        audio_codes = self.static_audio_codes[padded_size][:actual_size].clone()

        return audio_codes, fake_eos


# ---------------------------------------------------------------------------
# Gap fix #12 -- GPU-less backbone shape smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """CPU-only shape smoke test for the Mistral backbone classes.

    Constructs ``VoxtralAttention`` / ``VoxtralDecoderLayer`` / ``VoxtralBackboneModel``
    with a tiny config and asserts output shapes. Does NOT require P1's
    ``vox_serve.tokenizer.voxtral_tts`` -- only the backbone classes are exercised
    (those are the part this file fully owns). FlashInfer kernels are CUDA-only, so
    the attention/full-backbone forward is NOT run here; we instead verify module
    construction and parameter shapes, plus a forward through the bias-free MLP
    (which is pure PyTorch).
    """
    cfg = VoxtralBackboneConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=8,
        vocab_size=128,
        audio_codebook_embedding_size=96,
    )

    # --- MLP: pure-PyTorch, runnable on CPU ---
    mlp = VoxtralMLP(cfg)
    x = torch.randn(5, cfg.hidden_size)
    out = mlp(x)
    assert out.shape == (5, cfg.hidden_size), f"MLP out {tuple(out.shape)}"

    # --- RMSNorm weight shape ---
    norm = VoxtralRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    assert norm.weight.shape == (cfg.hidden_size,)

    # --- Attention: verify bias-free projection shapes (forward needs CUDA/FlashInfer) ---
    attn = VoxtralAttention(cfg, layer_idx=0)
    assert attn.q_proj.bias is None and attn.k_proj.bias is None, "attention must be bias-free"
    assert attn.v_proj.bias is None and attn.o_proj.bias is None, "attention must be bias-free"
    assert attn.q_proj.weight.shape == (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    assert attn.k_proj.weight.shape == (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    assert attn.v_proj.weight.shape == (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    assert attn.o_proj.weight.shape == (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)

    # --- DecoderLayer: verify sub-module wiring ---
    layer = VoxtralDecoderLayer(cfg, layer_idx=0)
    assert isinstance(layer.self_attn, VoxtralAttention)
    assert isinstance(layer.mlp, VoxtralMLP)
    assert isinstance(layer.input_layernorm, VoxtralRMSNorm)
    assert isinstance(layer.post_attention_layernorm, VoxtralRMSNorm)

    # --- BackboneModel: verify embedding tables + layer count ---
    backbone = VoxtralBackboneModel(cfg)
    assert backbone.embed_tokens.weight.shape == (cfg.vocab_size, cfg.hidden_size)
    assert backbone.audio_codebook_embeddings.weight.shape == (
        cfg.audio_codebook_embedding_size,
        cfg.hidden_size,
    )
    assert len(backbone.layers) == cfg.num_hidden_layers
    assert backbone.norm.weight.shape == (cfg.hidden_size,)

    # --- Mistral-format weight remapper: build a fake flat state dict and remap ---
    fake_state: dict[str, torch.Tensor] = {
        "mm_audio_embeddings.tok_embeddings.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight": torch.randn(
            cfg.audio_codebook_embedding_size, cfg.hidden_size
        ),
        "norm.weight": torch.randn(cfg.hidden_size),
        "output.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),  # tied lm_head -> skipped
    }
    for i in range(cfg.num_hidden_layers):
        fake_state[f"layers.{i}.attention.wq.weight"] = torch.randn(
            cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size
        )
        fake_state[f"layers.{i}.attention.wk.weight"] = torch.randn(
            cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size
        )
        fake_state[f"layers.{i}.attention.wv.weight"] = torch.randn(
            cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size
        )
        fake_state[f"layers.{i}.attention.wo.weight"] = torch.randn(
            cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim
        )
        fake_state[f"layers.{i}.feed_forward.w1.weight"] = torch.randn(
            cfg.intermediate_size, cfg.hidden_size
        )
        fake_state[f"layers.{i}.feed_forward.w3.weight"] = torch.randn(
            cfg.intermediate_size, cfg.hidden_size
        )
        fake_state[f"layers.{i}.feed_forward.w2.weight"] = torch.randn(
            cfg.hidden_size, cfg.intermediate_size
        )
        fake_state[f"layers.{i}.attention_norm.weight"] = torch.randn(cfg.hidden_size)
        fake_state[f"layers.{i}.ffn_norm.weight"] = torch.randn(cfg.hidden_size)
    # also include an out-of-scope key to confirm it's ignored
    fake_state["acoustic_transformer.norm.weight"] = torch.randn(cfg.hidden_size)

    backbone2 = VoxtralBackboneModel(cfg)
    backbone2.load_state_dict_from_mistral(fake_state)
    # verify a couple of remapped tensors landed correctly
    assert torch.equal(
        backbone2.embed_tokens.weight, fake_state["mm_audio_embeddings.tok_embeddings.weight"]
    )
    assert torch.equal(
        backbone2.layers[0].self_attn.q_proj.weight, fake_state["layers.0.attention.wq.weight"]
    )
    assert torch.equal(
        backbone2.layers[1].mlp.down_proj.weight, fake_state["layers.1.feed_forward.w2.weight"]
    )
    assert torch.equal(
        backbone2.layers[0].input_layernorm.weight, fake_state["layers.0.attention_norm.weight"]
    )

    print("VoxtralTTS backbone smoke test passed (CPU-only: construction + shapes + Mistral remap).")
    print("  NOTE: attention/backbone forward needs CUDA+FlashInfer and is not exercised here.")


if __name__ == "__main__":
    _smoke_test()
