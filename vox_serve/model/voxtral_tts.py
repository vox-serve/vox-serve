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
from typing import Any, List, Tuple

import torch
from torch import nn

from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ..requests import Request
from ..sampling import SamplingConfig
from ..utils import get_logger
from .base import BaseLM, PreprocessOutput

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

        query_states, key_states = apply_rope_pos_ids(
            query_states=query_states,
            key_states=key_states,
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
        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )

        hidden_states = self.norm(hidden_states)
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
        from ..tokenizer.voxtral_tts import (  # noqa: PLC0415
            FlowMatchingAudioTransformer,
            VoxtralTTSAudioTokenizer,
        )

        self.acoustic_transformer, self.audio_tokenizer = self._load_audio_modules(
            model_name, FlowMatchingAudioTransformer, VoxtralTTSAudioTokenizer
        )

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

        The Voxtral-TTS ``tekken.json`` carries a ``voice_num_audio_tokens`` key
        inside its ``audio`` block that ``mistral_common==1.8.6``'s ``AudioConfig``
        does not understand. We capture that mapping for ``preprocess`` (it tells
        how many ``audio_token_id`` placeholders to insert per voice), strip it,
        and load the tokenizer from a sanitized copy.
        """
        import json  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # noqa: PLC0415

        tekken_path = hf_hub_download(repo_id=model_name, filename="tekken.json")
        with open(tekken_path) as f:
            tekken = json.load(f)

        self.voice_num_audio_tokens: dict[str, int] = {}
        audio_block = tekken.get("audio")
        if isinstance(audio_block, dict) and "voice_num_audio_tokens" in audio_block:
            self.voice_num_audio_tokens = dict(audio_block.pop("voice_num_audio_tokens"))

        # ``MistralTokenizer.from_file`` sniffs the *filename* for "tekken", so the
        # sanitized copy must keep that name.
        sanitized_dir = tempfile.mkdtemp(prefix="voxtral_tekken_")
        sanitized_path = f"{sanitized_dir}/tekken.json"
        with open(sanitized_path, "w") as tf:
            json.dump(tekken, tf)
        return MistralTokenizer.from_file(sanitized_path)

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
        # 3072). The worker derives the FlashInfer head_dim and the KV-cache
        # head_dim from this property as ``hidden_size // num_attention_heads``,
        # so it must report the attention qkv dimension (4096) here, not 3072.
        # All model-internal modules use ``self.config.hidden_size`` (3072)
        # directly and are unaffected.
        return self._num_attention_heads * self.config.head_dim

    @property
    def head_dim(self) -> int:
        # Explicit head_dim from params.json (128), not hidden_size // n_heads.
        return self.config.head_dim

    @property
    def has_inline_audio_head(self) -> bool:
        """Voxtral's audio codes come from the inline acoustic head, not the backbone."""
        return True

    @property
    def use_repetition_penalty(self) -> bool:
        """Voxtral semantic selection is pure argmax -- never apply repetition penalty.

        Gap fix #2/#11: vllm-omni's voxtral module never wires a repetition penalty,
        and the default ``SamplingConfig`` has ``repetition_penalty=None``.
        """
        return False

    @property
    def supports_audio_input(self) -> bool:
        return True

    @property
    def needs_input_features(self) -> bool:
        return True

    @property
    def needs_input_masks(self) -> bool:
        return True

    @property
    def detokenize_interval(self) -> int:
        return 25

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

    def is_stop_id(self, token_ids: List[int]) -> bool:
        """Stop is signalled by the fake-EOS path, not by a token id in the stream.

        The semantic codebook (codebook 0) emitting ``[END_AUDIO]`` is the stop
        condition; that is handled in ``sampling``'s coroutine. Here we just check
        codebook 0 for the end-audio marker for completeness.
        """
        return len(token_ids) > 0 and token_ids[0] == _END_AUDIO_TOKEN_ID

    def _resolve_voice(self, kwargs: dict) -> tuple[str, torch.Tensor]:
        """Resolve the voice name + embedding from request kwargs, with a logged fallback.

        Returns ``(voice_name, embedding)`` where ``embedding`` is the per-position
        speaker reference sequence shaped ``(num_audio_tokens, hidden_size)``.
        """
        voice = kwargs.get("voice", None)
        if isinstance(voice, list) and voice:
            voice = voice[0]
        if voice is None or voice not in self.voice_to_embedding:
            if not self.voice_to_embedding:
                raise RuntimeError("VoxtralTTSModel: no voice embeddings loaded")
            fallback = next(iter(self.voice_to_embedding.keys()))
            if voice is not None:
                self.logger.warning(
                    "Voxtral voice %r not in registry; falling back to %r", voice, fallback
                )
            else:
                self.logger.warning("Voxtral no voice specified; falling back to %r", fallback)
            voice = fallback
        return voice, self.voice_to_embedding[voice].to(self.device).clone().detach()

    @torch.no_grad()
    def preprocess(
        self,
        prompt: str | None = None,
        audio_path: str | None = None,
        **kwargs: Any,
    ) -> PreprocessOutput:
        """Tokenize the prompt and scatter the voice embedding onto audio-token slots.

        Mirrors ``cosyvoice2.py:924-1006``. ``input_features`` carries the voice
        embedding at every position where ``input_ids == audio_token_id`` (24); the
        backbone embeds text tokens normally everywhere else and overwrites with
        ``input_features`` where ``input_masks`` is true.
        """
        assert audio_path is None, "audio_path is not supported yet for this model"
        assert prompt is not None, "prompt is required for VoxtralTTSModel"

        # Resolve the voice: name + per-position speaker reference sequence
        # shaped (num_audio_tokens, hidden_size).
        voice_name, voice_emb = self._resolve_voice(kwargs)
        voice_emb = voice_emb.to(self.device, dtype=self.dtype)
        if voice_emb.dim() == 1:
            voice_emb = voice_emb.unsqueeze(0)  # (1, hidden)
        num_audio_tokens = voice_emb.shape[0]

        # mistral-common tokenization -> flat list of text token ids (no BOS;
        # we prepend <s> ourselves as part of the TTS prompt layout).
        text_ids = list(
            self.text_tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=False, eos=False)
        )

        # Voxtral-TTS prompt layout (mirrors vllm-omni's dummy builder):
        #   <s> [BEGIN_AUDIO] [AUDIO]*N [REPEAT_AUDIO_TEXT] <text> [NEXT_AUDIO_TEXT] [BEGIN_AUDIO]
        # The run of N [AUDIO] placeholders carries the voice reference; the
        # trailing [BEGIN_AUDIO] triggers audio generation.
        bos_token_id = 1
        repeat_audio_text_id = 35
        next_audio_text_id = 36
        token_ids = (
            [bos_token_id, self.begin_audio_token_id]
            + [self.audio_token_id] * num_audio_tokens
            + [repeat_audio_text_id]
            + text_ids
            + [next_audio_text_id, self.begin_audio_token_id]
        )

        input_ids = torch.tensor(token_ids, dtype=torch.int32, device=self.device).view(-1, 1)
        seq_len = input_ids.shape[0]

        # input_features: voice reference scattered onto [AUDIO]-token positions
        # (one row per placeholder, in order).
        input_features = torch.zeros(seq_len, self.config.hidden_size, device=self.device, dtype=self.dtype)
        audio_mask_1d = input_ids[:, 0] == self.audio_token_id  # (seq_len,)
        if audio_mask_1d.any():
            input_features[audio_mask_1d] = voice_emb[: int(audio_mask_1d.sum())]

        # input_masks: 1 where the backbone should use input_features (audio slots)
        input_masks = torch.zeros(seq_len, self.n_codebooks, device=self.device, dtype=torch.bool)
        input_masks[audio_mask_1d, :] = True

        # No repetition penalty for Voxtral (gap fix #2/#11).
        repetition_cache = None

        # model-private left-context ring buffer (contract section 3.1)
        from ..tokenizer.voxtral_tts import VoxtralTTSDecoderCache  # noqa: PLC0415

        decoder_cache = VoxtralTTSDecoderCache(
            left_ctx=torch.zeros(25, self.n_codebooks, device=self.device, dtype=torch.int64)
        )

        return PreprocessOutput(
            input_tokens=input_ids,
            repetition_cache=repetition_cache,
            input_masks=input_masks,
            input_features=input_features,
            decoder_cache=decoder_cache,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        input_masks: torch.Tensor,
        input_features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Embed, scatter voice features on audio-token slots, run backbone.

        Mirrors ``cosyvoice2.py:1008-1033``. Returns hidden states with a codebook
        dimension added (``[:, None, :]``) so the worker's logits-buffer plumbing
        stays uniform; the actual audio decoding happens in ``sampling`` via the
        inline acoustic head.
        """
        # codebook 0 holds the (text or audio) token id; embed via the text table
        inputs_embeds = self.model.embed_tokens(
            input_ids[:, 0].clamp(0, self.model.embed_tokens.weight.shape[0] - 1)
        )

        inputs_embeds = torch.where(input_masks[:, :1], input_features, inputs_embeds)

        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        return hidden_states[:, None, :]  # add codebook dimension

    def _cfg_alpha_per_req(self, requests: List[Request], batch_size: int, device, dtype) -> torch.Tensor:
        """Build a ``(B,)`` per-request cfg_alpha tensor (default 1.2)."""
        default = self.default_sampling_config.cfg_scale or 1.2
        values = []
        for req in requests:
            cfg = None
            if req.sampling_config is not None:
                cfg = req.sampling_config.cfg_scale
            values.append(cfg if cfg is not None else default)
        return torch.tensor(values[:batch_size], device=device, dtype=dtype)

    def sampling(
        self,
        logits: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        repetition_cache: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """Inline-audio-head sampling: semantic argmax + acoustic ODE.

        Same return contract as ``cosyvoice2.py:1035-1091`` (gap fix #4):
        ``(output_ids, task_coroutine)``. There is NO FlashInfer sampler call -- the
        eager path runs its own semantic argmax + acoustic Euler ODE through
        ``self.acoustic_transformer`` (gap fix #3). If a captured acoustic CUDA
        graph is available for this batch size it is dispatched through instead.

        ``backbone_hidden_states`` is read from ``kwargs`` -- the worker passes it
        for inline-audio-head models.
        """
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        # The worker passes the backbone hidden states for inline-audio-head models.
        backbone_hidden_states = kwargs.get("backbone_hidden_states", None)
        if backbone_hidden_states is None:
            # ``logits`` here is actually the hidden state tensor from forward()
            # shaped (B, 1, hidden_size); fall back to it.
            backbone_hidden_states = logits.reshape(logits.shape[0], -1)
        else:
            backbone_hidden_states = backbone_hidden_states.reshape(backbone_hidden_states.shape[0], -1)

        batch_size = backbone_hidden_states.shape[0]
        device = backbone_hidden_states.device
        dtype = backbone_hidden_states.dtype

        cfg_alpha = self._cfg_alpha_per_req(requests, batch_size, device, dtype)

        # ---- dispatch: captured acoustic CUDA graph, else eager ----
        use_graph = (
            self._acoustic_graph is not None
            and getattr(self._acoustic_graph, "_get_padded_size", None) is not None
            and self._acoustic_graph._get_padded_size(batch_size) is not None
        )
        if use_graph:
            audio_codes, fake_eos = self._acoustic_graph(backbone_hidden_states, cfg_alpha)
        else:
            audio_codes, fake_eos = self._eager_acoustic_step(backbone_hidden_states, cfg_alpha)

        # output_ids shaped (B, 37)
        output_ids = audio_codes.to(torch.int64)

        # Mirror cosyvoice2: stage next-step inputs back onto each request.
        for i, req in enumerate(requests):
            req.input_tokens = output_ids[i : i + 1]
            if req.input_features is not None:
                req.input_features = req.input_features[:1].zero_()
            if req.input_masks is not None:
                req.input_masks = req.input_masks[:1].zero_()

        async def update_req_states():
            for i, req in enumerate(requests):
                req.lm_output_tokens.append(output_ids[i : i + 1])
                if bool(fake_eos[i]):
                    req.done_lm_generation = True
                    req.finish_reason = "stop_id_encountered"
                else:
                    # gap fix #10: lm_output_audio_tokens entries are (1, n_codebooks)
                    req.lm_output_audio_tokens.append(output_ids[i : i + 1])
                if req.next_position_id is not None and req.next_position_id > self.max_tokens:
                    req.done_lm_generation = True
                    req.finish_reason = "max_tokens_reached"

        return output_ids, update_req_states()

    @torch.no_grad()
    def _eager_acoustic_step(
        self,
        backbone_hidden_states: torch.Tensor,
        cfg_alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager semantic argmax + acoustic ODE.

        Delegates to ``self.acoustic_transformer.forward`` (P1's module) per the
        contract: it consumes ``(B, hidden_size)`` hidden states + ``cfg_alpha`` and
        returns ``audio_codes: (B, n_codebooks)`` int. ``fake_eos`` is derived from
        codebook 0 (``[END_AUDIO]``).
        """
        audio_codes = self.acoustic_transformer(
            llm_hidden=backbone_hidden_states,
            cfg_alpha=cfg_alpha,
        )
        fake_eos = audio_codes[:, 0] == _END_AUDIO_TOKEN_ID
        return audio_codes, fake_eos.bool()

    @torch.no_grad()
    def postprocess(self, token_ids: torch.Tensor, decoder_cache=None) -> torch.Tensor:
        """Decode a chunk of audio codes to a waveform.

        Mirrors ``cosyvoice2.py:1093-1117``. ``token_ids`` is ``(B, 25, 37)``; it is
        prepended with the 25-frame left-context ring buffer to give ``(B, 50, 37)``,
        decoded through the audio tokenizer, and the leading ``25 * 1920 = 48000``
        context samples are trimmed. The ring buffer is updated in place
        (graph-safe) from the last 25 frames of this chunk.
        """
        from ..tokenizer.voxtral_tts import VoxtralTTSDecoderCache  # noqa: PLC0415

        # The eager ``ModelWorker.run_detokenize`` path calls ``postprocess`` without
        # threading the per-request ``decoder_cache``. Fall back to a model-side
        # ring buffer keyed by batch size (correct for the batch-1 eager path).
        if decoder_cache is None:
            batch = token_ids.shape[0]
            if getattr(self, "_fallback_decoder_cache", None) is None or (
                self._fallback_decoder_cache.left_ctx.shape[0] != batch
            ):
                self._fallback_decoder_cache = VoxtralTTSDecoderCache(
                    left_ctx=torch.zeros(
                        batch, 25, self.n_codebooks,
                        device=self.audio_decoder_device, dtype=torch.int64,
                    )
                )
            decoder_cache = self._fallback_decoder_cache

        token_ids = token_ids.to(decoder_cache.left_ctx.device, dtype=decoder_cache.left_ctx.dtype)

        # (B, 50, 37): [left_ctx | new chunk]
        full = torch.cat([decoder_cache.left_ctx, token_ids], dim=1)

        # audio tokenizer wants (B, n_codebooks, T)
        audio = self.audio_tokenizer.decode(full.transpose(1, 2), dtype=self.dtype)

        # trim leading context samples
        trim = 25 * self.downsample_factor  # 48000
        audio = audio[..., trim:]

        # update the ring buffer in place from the last 25 frames (graph-safe)
        new_cache = VoxtralTTSDecoderCache(left_ctx=full[:, -25:, :].contiguous())
        decoder_cache.copy_from(new_cache)

        # audio is (B, 1, T) from the mono tokenizer; ensure channel dim present
        if audio.dim() == 2:
            audio = audio[:, None, :]
        # decode runs in bf16 to match conv weights; the worker consumes the
        # waveform via .numpy(), which requires fp32.
        return audio.float()

    # ------------------------------------------------------------------
    # Phase-4 acoustic CUDA graph hook
    # ------------------------------------------------------------------

    def enable_acoustic_graph(self, graph_pool, capture_sizes) -> None:
        """Construct + capture the acoustic-head CUDA graph (Phase-4 hook).

        Lazily imports ``AcousticHeadCudaGraph`` so there is no hard import-time
        dependency on ``voxtral_tts_acoustic_graph`` before it merges.
        """
        from .voxtral_tts_acoustic_graph import AcousticHeadCudaGraph  # noqa: PLC0415

        self._acoustic_graph = AcousticHeadCudaGraph(
            flow_transformer=self.acoustic_transformer,
            graph_pool=graph_pool,
            capture_sizes=list(capture_sizes),
            device=torch.device(self.device),
            dtype=self.dtype,
            hidden_dim=self.config.hidden_size,
        )


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
