"""Voxtral-TTS acoustic flow-matching transformer + waveform-decoder audio tokenizer.

Ported from `vllm-omni` (`vllm_omni/model_executor/models/voxtral_tts/`):
  - `voxtral_tts_audio_generation.py` -> `FlowMatchingAudioTransformer` + helpers.
  - `voxtral_tts_audio_tokenizer.py`  -> `VoxtralTTSAudioTokenizer` (DECODER PATH ONLY).

All vLLM-isms (``VllmConfig`` / ``default_weight_loader`` / ``vllm.logger`` /
``vllm_omni.platforms``) are dropped in favour of plain ``nn.Module`` semantics and
``vox_serve`` utilities. Encoder code is intentionally not ported.
"""

import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Union, get_args, get_origin

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import RMSNorm

from ..utils import get_logger
from .base import DecoderCache

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover - portability fallback
    flash_attn_func = None
    HAS_FLASH_ATTN = False

logger = get_logger(__name__)

if not HAS_FLASH_ATTN:
    logger.warning(
        "flash_attn is not installed. Falling back to PyTorch SDPA for "
        "Voxtral-TTS audio tokenizer attention. Install flash-attn for better performance."
    )

# weight_norm parametrization used by the causal convs in the waveform decoder.
weight_norm = torch.nn.utils.parametrizations.weight_norm


# ---------------------------------------------------------------------------
# Shared dataclasses / helpers (Source A)
# ---------------------------------------------------------------------------


class AudioSpecialTokens(str, Enum):
    """Special tokens predicted by audio codebook heads.

    Output audio tokens from the quantizer are offset by ``len(all_special_tokens())``
    to avoid colliding with these.
    """

    empty_audio = "[EMPTY_AUDIO]"
    end_audio = "[END_AUDIO]"

    @staticmethod
    def all_special_tokens() -> list["AudioSpecialTokens"]:
        return list(AudioSpecialTokens)

    @staticmethod
    def id(token: "AudioSpecialTokens") -> int:
        return AudioSpecialTokens.all_special_tokens().index(token)


@dataclass
class AcousticTransformerArgs:
    input_dim: int
    dim: int = 768
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 2048
    n_heads: int = 6
    n_kv_heads: int = 2
    use_biases: bool = False
    norm_eps: float = 1e-5
    sigma: float = 1e-5
    n_decoding_steps: int | None = None  # Number of Euler ODE steps for flow matching


@dataclass
class MultimodalAudioModelArgs:
    # The first token in a codebook is always reserved to indicate absence;
    # the codebook size is inclusive of this.
    semantic_codebook_size: int
    acoustic_codebook_size: int
    n_acoustic_codebook: int
    acoustic_transformer_args: AcousticTransformerArgs

    @property
    def codebook_sizes(self) -> list[int]:
        return [
            self.semantic_codebook_size,
            *[self.acoustic_codebook_size for _ in range(self.n_acoustic_codebook)],
        ]

    def get_codebook_sizes(
        self, pad_to_multiple: int | None = 128, include_special_tokens: bool = True
    ) -> list[int]:
        def _round_up_to_multiple_of_number(n: int, multiple: int) -> int:
            return multiple * ((n + multiple - 1) // multiple)

        result_codebook_sizes = []
        for cb_size in self.codebook_sizes:
            if include_special_tokens:
                cb_size += len(AudioSpecialTokens.all_special_tokens())
            if pad_to_multiple is not None:
                cb_size = _round_up_to_multiple_of_number(cb_size, pad_to_multiple)
            result_codebook_sizes.append(cb_size)
        return result_codebook_sizes


def from_nested_dict(cls, d):
    """Recursively instantiate dataclasses from nested dicts."""
    if not is_dataclass(cls):
        return d

    kwargs = {}
    for f in fields(cls):
        value = d.get(f.name, getattr(cls, f.name, None))
        field_type = f.type

        # Unwrap Optional / Union types.
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            non_none_types = [a for a in args if a is not type(None)]
            if len(non_none_types) == 1:
                field_type = non_none_types[0]

        # Recurse if nested dataclass.
        if is_dataclass(field_type) and isinstance(value, dict):
            value = from_nested_dict(field_type, value)

        kwargs[f.name] = value

    return cls(**kwargs)


def _repeat_interleave(t: torch.Tensor, repeats: int) -> torch.Tensor:
    return t.unsqueeze(3).expand([-1, -1, -1, repeats, -1]).flatten(2, 3)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
    if repeats > 1:
        keys = _repeat_interleave(keys, repeats=repeats)
        values = _repeat_interleave(values, repeats=repeats)
    return keys, values


# ---------------------------------------------------------------------------
# Flow-matching acoustic transformer (Source A)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, use_biases: bool) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BidirectionalAttention(nn.Module):
    """Attention layer (without any RoPE embeddings)."""

    def __init__(self, args: AcousticTransformerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args

        self.n_local_heads: int = args.n_heads
        self.n_local_kv_heads: int = args.n_kv_heads
        self.layer_id = layer_id

        self.head_dim = args.head_dim

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.use_biases)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

        self.softmax_scale: float = self.args.head_dim**-0.5
        self.repeats = self.n_local_heads // self.n_local_kv_heads

    def _native_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = attn @ value
        return attn.transpose(1, 2).contiguous()

    def _forward_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        key, value = repeat_kv(key, value, repeats=self.repeats)
        bsz, seqlen, _, _ = query.shape
        output = self._native_attention(query, key, value)
        return output.view(bsz, seqlen, -1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            bsz, (seqlen, _) = 1, x.shape
        else:
            bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        output = self._forward_attention(query=xq, key=xk, value=xv, **kwargs)
        output = output.view(bsz, seqlen, self.n_local_heads * self.head_dim)
        return self.wo(output).squeeze(0)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AcousticTransformerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = BidirectionalAttention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding for encoding the flow-matching time step."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = torch.exp(-math.log(theta) * torch.arange(dim // 2).float() / (dim // 2))
        # Mistral codebase saves/loads this buffer, hence persistent=True.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("bi, j -> bj", t, self.inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class FlowMatchingAudioTransformer(nn.Module):
    """Acoustic flow-matching transformer.

    ``forward`` consumes backbone hidden states ``(B, input_dim)`` plus a per-request
    ``cfg_alpha`` tensor ``(B,)``, runs an ``n_decoding_steps``-step Euler ODE, and
    returns ``audio_codes`` of shape ``(B, n_codebooks)`` (int).
    """

    def __init__(self, audio_model_args: dict) -> None:
        super().__init__()
        audio_model_args = dict(audio_model_args)
        if "codebook_sizes" in audio_model_args:
            codebook_sizes = [int(c) for c in audio_model_args.pop("codebook_sizes").split(",")]
            audio_model_args.update(
                {
                    "semantic_codebook_size": codebook_sizes[0],
                    "acoustic_codebook_size": codebook_sizes[1],
                    "n_acoustic_codebook": len(codebook_sizes) - 1,
                }
            )
        self.model_args: MultimodalAudioModelArgs = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        assert isinstance(self.model_args, MultimodalAudioModelArgs)
        args = self.model_args.acoustic_transformer_args
        self.acoustic_transformer_args = args
        assert isinstance(self.acoustic_transformer_args, AcousticTransformerArgs)

        # Currently assuming always 1 semantic codebook + N acoustic codebooks.
        self.num_non_acoustic_embeddings = 1
        self.num_acoustic_codebooks = len(self.model_args.get_codebook_sizes()) - self.num_non_acoustic_embeddings

        # Flow matching utils.
        self.sigma = args.sigma

        acoustic_codebook_sizes = self.model_args.get_codebook_sizes(
            pad_to_multiple=None, include_special_tokens=False
        )[1:]
        assert len(set(acoustic_codebook_sizes)) == 1, "only 1 size for acoustic codebooks supported"
        self.acoustic_embeddings_levels = acoustic_codebook_sizes[0]
        self.acoustic_embeddings_dim = len(acoustic_codebook_sizes)

        self._init_audio_embeddings_layer()
        self._init_output_layer()
        self._init_layers()

        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self._empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)

        # Flow matching constants.
        self._n_steps = args.n_decoding_steps
        self._noise_scale = 1.0
        self.register_buffer(
            "_timesteps",
            torch.linspace(0, 1, self._n_steps + 1),
            persistent=False,
        )

    def _init_audio_embeddings_layer(self) -> None:
        self.time_embedding = TimeEmbedding(self.acoustic_transformer_args.dim)
        input_dim = self.acoustic_embeddings_dim

        self.input_projection = nn.Linear(input_dim, self.acoustic_transformer_args.dim, bias=False)
        self.time_projection = nn.Linear(
            self.acoustic_transformer_args.dim,
            self.acoustic_transformer_args.dim,
            bias=False,
        )
        self.llm_projection = nn.Linear(
            self.acoustic_transformer_args.input_dim,
            self.acoustic_transformer_args.dim,
            bias=False,
        )

    def _init_output_layer(self) -> None:
        padded_codebook_sizes = self.model_args.get_codebook_sizes(pad_to_multiple=128)
        self.semantic_codebook_output = nn.Linear(
            self.acoustic_transformer_args.dim,
            padded_codebook_sizes[0],
            self.acoustic_transformer_args.use_biases,
        )
        self.acoustic_codebook_output = nn.Linear(
            in_features=self.acoustic_transformer_args.dim,
            out_features=self.model_args.n_acoustic_codebook,
            bias=False,
        )

    def _init_layers(self) -> None:
        self.layers_ids: list[int] = list(range(self.acoustic_transformer_args.n_layers))
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            block = AcousticTransformerBlock(layer_id=layer_id, args=self.acoustic_transformer_args)
            self.layers[str(layer_id)] = block

        self.norm = RMSNorm(self.acoustic_transformer_args.dim, self.acoustic_transformer_args.norm_eps)

    def forward_attention_layers(self, h: torch.Tensor) -> torch.Tensor:
        for layer_id in self.layers_ids:
            layer = self.layers[str(layer_id)]
            h = layer(h)
        return h

    def decode_one_frame(
        self,
        semantic_code: torch.Tensor,
        llm_hidden: torch.Tensor,
        cfg_alpha: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = semantic_code.shape[0]

        # Skip decoding if codebook 0 is the [END_AUDIO] token.
        should_decode = semantic_code != self._end_audio_token_id

        # acoustic_codes starts from x_0. When ``noise`` is provided the caller
        # supplies the initial Euler-ODE noise draw verbatim (enabling bitwise
        # eager-vs-CUDA-graph parity tests); otherwise preserve the original
        # bare ``torch.randn`` behavior exactly.
        if noise is None:
            x_0 = torch.randn(B, self.model_args.n_acoustic_codebook).to(
                dtype=llm_hidden.dtype, device=llm_hidden.device
            )
        else:
            x_0 = noise.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
        x_0 = self._noise_scale * x_0

        timesteps = self._timesteps.to(dtype=llm_hidden.dtype)
        llm_hidden_zero = torch.zeros_like(llm_hidden)

        # Reshape cfg_alpha for broadcasting: (B,) -> (B, 1).
        cfg_alpha = cfg_alpha.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
        cfg_alpha = cfg_alpha.unsqueeze(1)

        # Euler integration with batched conditional + unconditional velocity.
        sampled = x_0
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            t_emb = self.time_embedding(t.view(-1, 1).repeat(B, 1)).to(llm_hidden.dtype)

            # Batch cond + uncond into a single forward pass (2B batch).
            x_batched = torch.cat([sampled, sampled], dim=0)
            llm_batched = torch.cat([llm_hidden, llm_hidden_zero], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)

            v_all = self._predict_velocity(
                x_t=x_batched,
                llm_output=llm_batched,
                t_emb=t_emb_batched,
            )
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            v_t = cfg_alpha * v_t + (1 - cfg_alpha) * uncond_v_t

            sampled = sampled + v_t * dt

        # Quantize & mask end-of-audio.
        sampled = torch.clamp(sampled, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = self._empty_audio_token_id
        return output_codes + len(AudioSpecialTokens)

    def _predict_velocity(
        self,
        x_t: torch.Tensor,  # BxC
        llm_output: torch.Tensor,  # BxD
        t_emb: torch.Tensor,  # BxD
    ) -> torch.Tensor:
        x_t = x_t.to(llm_output.dtype)

        t_emb = self.time_projection(t_emb)
        llm_output = self.llm_projection(llm_output)

        acoustic_and_semantic_embeddings = [
            self.input_projection(x_t.unsqueeze(1)),  # Bx1xD
            t_emb.unsqueeze(1),
            llm_output.unsqueeze(1),
        ]
        acoustic_transformer_inputs = torch.concatenate(acoustic_and_semantic_embeddings, dim=1)

        attn_output = self.forward_attention_layers(acoustic_transformer_inputs)
        final_hidden = self.norm(attn_output)
        final_hidden = final_hidden.view(-1, acoustic_transformer_inputs.shape[1], final_hidden.shape[-1])
        v_t = self.acoustic_codebook_output(final_hidden[:, 0, :])

        return v_t

    def forward(
        self,
        llm_hidden: torch.Tensor,
        cfg_alpha: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # llm_hidden: BxD
        # ``noise`` (optional, ``(B, n_acoustic_codebook)``) lets a caller inject
        # the initial Euler-ODE noise draw verbatim; when ``None`` the per-frame
        # helper falls back to the original bare ``torch.randn`` behavior.
        semantic_logit = self.semantic_codebook_output(llm_hidden).float()
        semantic_logit[:, self._empty_audio_token_id] = -float("inf")  # eoa is allowed
        semantic_logit[:, (len(AudioSpecialTokens) + self.model_args.semantic_codebook_size) :] = -float("inf")

        # semantic_logit: Bx1
        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)

        acoustic_codes = self.decode_one_frame(
            semantic_code.squeeze(1),
            llm_hidden,
            cfg_alpha=cfg_alpha,
            noise=noise,
        )

        audio_codes = torch.concatenate([semantic_code, acoustic_codes], dim=1)
        return audio_codes


# ---------------------------------------------------------------------------
# Audio tokenizer — DECODER PATH ONLY (Source B)
# ---------------------------------------------------------------------------

CODEC_NORM_EPS = 1e-2


@dataclass
class AudioTokenizerArgs:
    # audio setting
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7

    # quantizer setting
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36

    # architecture (general)
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
    norm_eps: float = 1e-2
    layer_scale: bool = True
    layer_scale_init: float | None = None

    # architecture (encoder) — kept for config-parity only; encoder is not ported.
    encoder_transformer_lengths_str: str = "2,2,2,2"
    encoder_convs_kernels_str: str = "4,4,4,3"
    encoder_convs_strides_str: str = "2,2,2,1"

    # architecture (decoder)
    decoder_transformer_lengths_str: str = "2,2,2,2"
    decoder_convs_kernels_str: str = "3,4,4,4"
    decoder_convs_strides_str: str = "1,2,2,2"

    def __post_init__(self) -> None:
        assert (
            len(self.encoder_transformer_lengths) == len(self.encoder_convs_kernels) == len(self.encoder_convs_strides)
        )
        assert (
            len(self.decoder_transformer_lengths) == len(self.decoder_convs_kernels) == len(self.decoder_convs_strides)
        )

    def __str2list__(self, input_str: str) -> tuple[int, ...]:
        return tuple(int(i) for i in input_str.split(","))

    @property
    def encoder_transformer_lengths(self) -> tuple[int, ...]:
        return self.__str2list__(self.encoder_transformer_lengths_str)

    @property
    def encoder_convs_kernels(self) -> tuple[int, ...]:
        return self.__str2list__(self.encoder_convs_kernels_str)

    @property
    def encoder_convs_strides(self) -> tuple[int, ...]:
        return self.__str2list__(self.encoder_convs_strides_str)

    @property
    def decoder_transformer_lengths(self) -> tuple[int, ...]:
        return self.__str2list__(self.decoder_transformer_lengths_str)

    @property
    def decoder_convs_kernels(self) -> tuple[int, ...]:
        return self.__str2list__(self.decoder_convs_kernels_str)

    @property
    def decoder_convs_strides(self) -> tuple[int, ...]:
        return self.__str2list__(self.decoder_convs_strides_str)

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / (self.pretransform_patch_size * math.prod(self.encoder_convs_strides))


class SemanticCodebook(nn.Module):
    """Euclidean distance-based codebook for semantic quantization (decode path only)."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.epsilon: float = 1e-5
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        embedding = torch.zeros(codebook_size, codebook_dim)
        self.register_buffer("embedding_sum", embedding)
        self.register_buffer("_embedding", None, persistent=False)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            self.register_buffer("_embedding", embedding, persistent=False)
            return embedding
        return self._embedding

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        assert not codes.dtype.is_floating_point, f"Codes should be integers, got {codes.dtype}"
        assert codes.shape[1] == self.num_codebooks == 1  # only 1 semantic codebook for now
        codes = codes.squeeze(1)  # BxT
        embedding = self.embedding.to(codes.device)
        quantized = F.embedding(codes, embedding)
        quantized = quantized.transpose(1, 2)  # b t d -> b d t
        return quantized

    @property
    def num_codebooks(self) -> int:
        return 1

    @property
    def codebook_sizes(self) -> list[int]:
        return [self.codebook_size]


class AcousticCodebook(nn.Module):
    """Finite Scalar Quantization for acoustic codebooks (decode path only)."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.dim = codebook_dim
        self.n_levels = codebook_size
        self.num_codebooks = codebook_dim

    def _rescale(self, x: torch.Tensor, levels: int | torch.Tensor) -> torch.Tensor:
        return (x * 2 / (levels - 1)) - 1

    def _quantized_from_codes(self, codes: torch.Tensor, levels: int) -> torch.Tensor:
        return self._rescale(codes, levels)

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        assert not codes.dtype.is_floating_point, f"Codes should be integers, got {codes.dtype}"
        quantized = self._quantized_from_codes(codes, self.n_levels).to(dtype)
        return quantized


class MistralAudioCodebook(nn.Module):
    """Combines semantic + acoustic quantization (decode path only)."""

    def __init__(self, audio_tokenizer_args: AudioTokenizerArgs) -> None:
        super().__init__()

        self.semantic_codebook = SemanticCodebook(
            codebook_size=audio_tokenizer_args.semantic_codebook_size,
            codebook_dim=audio_tokenizer_args.semantic_dim,
        )
        self.acoustic_codebook = AcousticCodebook(
            codebook_size=audio_tokenizer_args.acoustic_codebook_size,
            codebook_dim=audio_tokenizer_args.acoustic_dim,
        )

        self.semantic_dim = audio_tokenizer_args.semantic_dim
        self.acoustic_dim = audio_tokenizer_args.acoustic_dim
        self.total_dim = self.semantic_dim + self.acoustic_dim

    @property
    def num_codebooks(self) -> int:
        return self.semantic_codebook.num_codebooks + self.acoustic_codebook.num_codebooks

    @property
    def codebook_sizes(self) -> list[int]:
        return self.semantic_codebook.codebook_sizes + [self.acoustic_codebook.n_levels]

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode discrete codes ``[B, K, T]`` back into continuous ``[B, D, T]``."""
        assert not codes.dtype.is_floating_point, f"Codes should be integers, got {codes.dtype}"

        semantic_codes = codes[:, : self.semantic_codebook.num_codebooks, :]
        acoustic_codes = codes[:, self.semantic_codebook.num_codebooks :, :]

        semantic_emb = self.semantic_codebook.decode(semantic_codes).to(dtype)  # [B, semantic_dim, T]
        acoustic_emb = self.acoustic_codebook.decode(acoustic_codes).to(dtype)  # [B, acoustic_dim, T]

        emb = torch.cat([semantic_emb, acoustic_emb], dim=1)
        return emb


def pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """``F.pad`` wrapper that supports reflect padding on small inputs."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


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
    ) -> None:
        super().__init__()
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.use_weight_norm = use_weight_norm
        self.pad_mode = pad_mode
        self._stride = self.conv.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.conv.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride
        self.stride = self.conv.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (x.shape[-1] - self._effective_kernel_size + self._padding_total) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (self._effective_kernel_size - self._padding_total)
        extra_padding = target_length - x.shape[-1]
        x = pad1d(x, (self._padding_total, extra_padding), mode=self.pad_mode)
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
    ) -> None:
        super().__init__()
        conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.use_weight_norm = use_weight_norm
        self.trim_ratio = trim_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        total_padding = kernel_size - stride
        out = self.conv(x)
        right_padding = math.ceil(total_padding * self.trim_ratio)
        left_padding = total_padding - right_padding
        return out[..., left_padding : out.shape[-1] - right_padding]


class MultiVocabEmbeddings(nn.Module):
    def __init__(self, audio_model_args: dict, embedding_dim: int) -> None:
        super().__init__()
        self.model_args = from_nested_dict(MultimodalAudioModelArgs, dict(audio_model_args))
        self.codebook_sizes = list(self.model_args.get_codebook_sizes(pad_to_multiple=None))
        self.offsets = torch.from_numpy(np.cumsum([0] + self.codebook_sizes[:-1]))
        self.total_vocab_size = sum(self.codebook_sizes)
        padded_size = 128 * ((self.total_vocab_size + 127) // 128)
        self.embeddings = nn.Embedding(padded_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: BxCxL
        self.offsets = self.offsets.to(input_ids.device)
        input_ids = input_ids + self.offsets[torch.newaxis, :, torch.newaxis]
        return self.embeddings(input_ids)


class Attention(nn.Module):
    """ALiBi + causal + sliding-window attention.

    Uses ``flash_attn_func`` when available, otherwise an SDPA fallback that
    builds the equivalent additive bias.
    """

    def __init__(self, args: AudioTokenizerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args

        self.n_local_heads: int = args.n_heads
        self.n_local_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_local_heads // self.n_local_kv_heads
        self.layer_id = layer_id
        self.sliding_window = args.attn_sliding_window_size

        def get_alibi_slopes(n_heads: int) -> torch.Tensor:
            def slopes_power_of_2(n: int) -> torch.Tensor:
                r = 2.0 ** (-8.0 / n)
                return torch.tensor([r**i for i in range(n)], dtype=torch.float32)

            if math.log2(n_heads).is_integer():
                slopes = slopes_power_of_2(n_heads)
            else:
                m = 2 ** math.floor(math.log2(n_heads))
                slopes = torch.cat(
                    [
                        slopes_power_of_2(m),
                        slopes_power_of_2(2 * m)[::2][: n_heads - m],
                    ]
                )
            return slopes

        self.register_buffer(
            "alibi_slopes",
            get_alibi_slopes(self.n_local_heads),
            persistent=False,
        )

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

        if args.qk_norm:
            self.q_norm = RMSNorm(args.n_heads * args.head_dim, eps=args.qk_norm_eps)
            self.k_norm = RMSNorm(args.n_kv_heads * args.head_dim, eps=args.qk_norm_eps)

    def _native_attention(
        self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor
    ) -> torch.Tensor:
        """SDPA fallback. Builds alibi + causal + sliding-window bias explicitly."""
        B, S, H, D = xq.shape
        Hkv = xk.shape[2]

        # (B, S, H, D) -> (B, H, S, D)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # Expand KV heads for GQA.
        if H != Hkv:
            repeats = H // Hkv
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Build attention bias: alibi + causal + sliding window.
        positions = torch.arange(S, device=xq.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (S, S), rel_pos[i, j] = j - i

        alibi_slopes = self.alibi_slopes.to(dtype=xq.dtype, device=xq.device)
        attn_bias = alibi_slopes.view(H, 1, 1) * rel_pos.unsqueeze(0).to(xq.dtype)

        if self.args.causal:
            attn_bias = attn_bias.masked_fill(rel_pos.unsqueeze(0) > 0, float("-inf"))

        window_left = self.sliding_window
        window_right = 0 if self.args.causal else self.sliding_window
        outside_window = (rel_pos < -window_left) | (rel_pos > window_right)
        attn_bias = attn_bias.masked_fill(outside_window.unsqueeze(0), float("-inf"))

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.unsqueeze(0))
        # (B, H, S, D) -> (B, S, H, D)
        return output.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz, (seqlen, _) = 1, x.shape
        else:
            bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.args.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)

        if HAS_FLASH_ATTN:
            alibi_slopes = self.alibi_slopes.to(torch.float32)
            output = flash_attn_func(
                xq,
                xk,
                xv,
                causal=self.args.causal,
                window_size=(
                    self.sliding_window,
                    0 if self.args.causal else self.sliding_window,
                ),
                alibi_slopes=alibi_slopes,
            )
        else:
            output = self._native_attention(xq, xk, xv)

        output = output.view(bsz, seqlen, self.n_local_heads * self.args.head_dim)
        return self.wo(output).squeeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AudioTokenizerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args, layer_id=layer_id)

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            use_biases=args.use_biases,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_norm: nn.Module | None = None
        self.post_ffn_norm: nn.Module | None = None
        self.args = args

        self.layer_scale = args.layer_scale
        if self.layer_scale:
            if args.layer_scale_init is None:
                if layer_id < 18:
                    init_scale = 0.1
                elif layer_id <= 24:
                    init_scale = 1e-5
                else:
                    init_scale = 1e-6
            else:
                init_scale = args.layer_scale_init
            self.attention_scale = nn.Parameter(torch.full((args.dim,), init_scale, requires_grad=True))
            self.ffn_scale = nn.Parameter(torch.full((args.dim,), init_scale, requires_grad=True))

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        if self.post_attention_norm is not None:
            r = self.post_attention_norm(r)
        if self.layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        if self.post_ffn_norm is not None:
            r = self.post_ffn_norm(r)
        if self.layer_scale:
            r = self.ffn_scale * r
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(self, args: AudioTokenizerArgs, n_layers: int) -> None:
        super().__init__()
        self.args = args
        self.n_layers = n_layers
        self.layers_ids: list[int] = list(range(n_layers))

        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            block = TransformerBlock(layer_id=layer_id, args=args)
            self.layers[str(layer_id)] = block

        assert len(self.layers) == len(self.layers_ids), (len(self.layers), len(self.layers_ids))

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        h = input_ids
        for layer_id in self.layers_ids:
            layer = self.layers[str(layer_id)]
            assert layer.layer_id == layer_id, (layer.layer_id, layer_id)
            h = layer(h)
        return h


def _remap_weight_norm_keys(state_dict: dict) -> dict:
    """Normalize weight-norm key naming in a Mistral-format checkpoint.

    PyTorch's ``parametrizations.weight_norm`` stores parameters under
    ``<conv>.parametrizations.weight.original0`` / ``original1``. Older / Mistral
    checkpoints may instead ship ``<conv>.weight_g`` / ``<conv>.weight_v`` (legacy
    ``torch.nn.utils.weight_norm`` naming). This remaps the legacy form to the
    parametrize form so ``load_state_dict`` succeeds.
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            remapped[f"{base}.parametrizations.weight.original0"] = value
        elif key.endswith(".weight_v"):
            base = key[: -len(".weight_v")]
            remapped[f"{base}.parametrizations.weight.original1"] = value
        else:
            remapped[key] = value
    return remapped


class VoxtralTTSAudioTokenizer(nn.Module):
    """Voxtral-TTS waveform decoder (DECODER PATH ONLY).

    Public method:
        ``decode(codes: (B, n_codebooks, T)) -> waveform (B, 1, T * downsample_factor)``.

    The encoder (``_forward_encoder`` / ``encode_*``) is intentionally not ported.
    """

    def __init__(self, audio_config: dict, text_hidden_size: int) -> None:
        super().__init__()
        args = from_nested_dict(AudioTokenizerArgs, dict(audio_config["codec_args"]))
        self.args = args

        if not args.causal:
            raise NotImplementedError("only causal audio tokenizer is supported")

        self.patch_size = args.pretransform_patch_size
        self.latent_dim = args.semantic_dim + args.acoustic_dim

        # ``input_proj`` and ``encoder_blocks`` belong to the encoder; not constructed.
        cur_window_size = args.attn_sliding_window_size

        # Track the encoder's downsampling effect on the sliding-window so the
        # decoder windows match the original construction order.
        for _idx, stride in enumerate(args.encoder_convs_strides):
            if args.half_attn_window_upon_downsampling and stride > 1:
                assert stride == 2, "only supporting 2x downsampling"
                cur_window_size = cur_window_size // 2
                assert cur_window_size >= 2

        ### Audio token lookup table for the LLM (kept; used by P2's forward()).
        self.audio_token_embedding = MultiVocabEmbeddings(
            audio_model_args=audio_config["audio_model_args"],
            embedding_dim=text_hidden_size,
        )

        ### Decoder
        decoder_blocks: list[nn.Module] = []
        decoder_convs_kernels = args.decoder_convs_kernels
        decoder_convs_strides = args.decoder_convs_strides
        decoder_transformer_lengths = args.decoder_transformer_lengths

        # First projection layer is necessary.
        decoder_blocks.append(
            CausalConv1d(
                self.latent_dim,
                args.dim,
                kernel_size=decoder_convs_kernels[0],
                stride=decoder_convs_strides[0],
                pad_mode="replicate",
                use_weight_norm=args.conv_weight_norm,
                use_bias=False,
            )
        )
        if args.half_attn_window_upon_downsampling and (decoder_convs_strides[0] > 1):
            assert decoder_convs_strides[0] == 2, "only supporting 2x upsampling"
            cur_window_size = cur_window_size * 2

        for idx, n_layers in enumerate(decoder_transformer_lengths):
            from copy import deepcopy

            layer_args = deepcopy(args)
            layer_args.attn_sliding_window_size = cur_window_size
            decoder_transformer = Transformer(args=layer_args, n_layers=n_layers)
            decoder_blocks.append(decoder_transformer)

            if (idx + 1 != len(decoder_transformer_lengths)) and (
                (decoder_convs_kernels[idx + 1] != 1) or (decoder_convs_strides[idx + 1] != 1)
            ):
                decoder_blocks.append(
                    CausalConvTranspose1d(
                        args.dim,
                        args.dim,
                        kernel_size=decoder_convs_kernels[idx + 1],
                        stride=decoder_convs_strides[idx + 1],
                        use_weight_norm=args.conv_weight_norm,
                        use_bias=False,
                    )
                )
                if args.half_attn_window_upon_downsampling and (decoder_convs_strides[idx + 1] > 1):
                    assert decoder_convs_strides[idx + 1] == 2, "only supporting 2x upsampling"
                    cur_window_size = cur_window_size * 2

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.quantizer = MistralAudioCodebook(args)

        self.output_proj = CausalConv1d(
            args.dim,
            args.pretransform_patch_size,
            kernel_size=args.patch_proj_kernel_size,
            use_weight_norm=args.conv_weight_norm,
            use_bias=False,
        )

        scale_factor = math.prod(decoder_convs_strides)
        assert scale_factor == math.prod(args.encoder_convs_strides)
        self._frame_rate = args.sampling_rate / (self.patch_size * scale_factor)
        self._sampling_rate = args.sampling_rate
        self._channels = args.channels
        if self._channels != 1:
            raise NotImplementedError("only mono audio is supported")

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def downsample_factor(self) -> int:
        assert self._sampling_rate % self._frame_rate == 0
        return int(self._sampling_rate / self._frame_rate)

    @property
    def num_codebooks(self) -> int:
        return self.quantizer.num_codebooks

    @property
    def codebook_sizes(self) -> list[int]:
        return self.quantizer.codebook_sizes

    def load_weights(self, state_dict: dict, strict: bool = False) -> None:
        """Load a Mistral-format checkpoint (decoder subset) and flatten weight-norm.

        Only keys present in this module are consumed; encoder keys in the
        checkpoint are ignored. After load, ``weight_norm`` parametrizations on
        every causal conv are removed so the flattened ``weight`` is
        CUDA-graph-capture-safe (gap fix #9).
        """
        state_dict = _remap_weight_norm_keys(state_dict)
        own_keys = set(self.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in own_keys}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        if strict and missing:
            raise RuntimeError(f"Missing keys when loading VoxtralTTSAudioTokenizer: {missing}")
        if unexpected:
            logger.warning("Unexpected keys when loading VoxtralTTSAudioTokenizer: %s", unexpected)
        self.remove_weight_norm_parametrizations()

    def remove_weight_norm_parametrizations(self) -> None:
        """Flatten every ``weight_norm`` parametrized causal conv to a plain ``weight``."""
        for module in self.modules():
            if isinstance(module, (CausalConv1d, CausalConvTranspose1d)) and module.use_weight_norm:
                conv = module.conv
                if torch.nn.utils.parametrize.is_parametrized(conv, "weight"):
                    torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)

    def _forward_decoder(self, emb: torch.Tensor) -> torch.Tensor:
        emb = emb.transpose(1, 2).contiguous()  # b d t -> b t d

        for block in self.decoder_blocks:
            if isinstance(block, (CausalConvTranspose1d, CausalConv1d)):
                emb = emb.transpose(1, 2)  # b t d -> b d t
                emb = block(emb)
                emb = emb.transpose(1, 2)  # b d t -> b t d
            else:
                # Transformer block — supports batched attention.
                emb = block(emb)  # (b, t, d)

        emb = emb.transpose(1, 2)  # b t d -> b d t
        emb = self.output_proj(emb)

        # b (c h) t -> b c (t h)
        b, ch, t = emb.shape
        h = self.patch_size
        c = ch // h
        out = emb.view(b, c, h, t).permute(0, 1, 3, 2).reshape(b, c, t * h)
        return out

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode codes ``[B, K, T]`` -> reconstructed waveform ``[B, C, T * downsample_factor]``."""
        emb = self.quantizer.decode(codes, dtype)  # (b, k, t)
        return self._forward_decoder(emb)


# ---------------------------------------------------------------------------
# Decoder cache
# ---------------------------------------------------------------------------


@dataclass
class VoxtralTTSDecoderCache(DecoderCache):
    """Model-private left-context ring buffer for the Voxtral-TTS waveform decoder.

    ``left_ctx`` is shaped ``(B, 25, n_codebooks)``. Inherits ``cat`` / ``copy_from`` /
    ``__getitem__`` / ``to`` from :class:`DecoderCache` — no new framework code here.
    """

    left_ctx: torch.Tensor


# ---------------------------------------------------------------------------
# GPU-less shape smoke test (gap fix #12)
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Construct the modules on CPU with tiny configs and assert output shapes.

    No weights, no CUDA. Run via ``python vox_serve/tokenizer/voxtral_tts.py``.
    """
    torch.manual_seed(0)

    # --- FlowMatchingAudioTransformer ---------------------------------------
    tiny_hidden = 64
    audio_model_args = {
        "semantic_codebook_size": 32,
        "acoustic_codebook_size": 21,
        "n_acoustic_codebook": 4,  # tiny: n_codebooks = 1 + 4 = 5
        "acoustic_transformer_args": {
            "input_dim": tiny_hidden,
            "dim": tiny_hidden,
            "n_layers": 2,
            "head_dim": 16,
            "hidden_dim": 128,
            "n_heads": 4,
            "n_kv_heads": 2,
            "use_biases": False,
            "norm_eps": 1e-5,
            "sigma": 1e-5,
            "n_decoding_steps": 7,
        },
    }
    flow = FlowMatchingAudioTransformer(audio_model_args).eval()
    n_codebooks = 1 + audio_model_args["n_acoustic_codebook"]

    B = 3
    llm_hidden = torch.randn(B, tiny_hidden)
    cfg_alpha = torch.full((B,), 1.2)
    with torch.no_grad():
        audio_codes = flow(llm_hidden, cfg_alpha)
    assert audio_codes.shape == (B, n_codebooks), audio_codes.shape
    assert not audio_codes.dtype.is_floating_point, audio_codes.dtype
    print(f"[smoke] FlowMatchingAudioTransformer: audio_codes {tuple(audio_codes.shape)} OK")

    # --- VoxtralTTSAudioTokenizer (decoder) ---------------------------------
    tiny_dim = 32
    text_hidden_size = 48
    audio_config = {
        "codec_args": {
            "channels": 1,
            "sampling_rate": 24000,
            "pretransform_patch_size": 4,
            "patch_proj_kernel_size": 3,
            "semantic_codebook_size": 32,
            "semantic_dim": 8,
            "acoustic_codebook_size": 21,
            "acoustic_dim": 4,  # n_codebooks = 1 + 4 = 5
            "conv_weight_norm": True,
            "causal": True,
            "attn_sliding_window_size": 16,
            "half_attn_window_upon_downsampling": True,
            "dim": tiny_dim,
            "hidden_dim": 64,
            "head_dim": 8,
            "n_heads": 4,
            "n_kv_heads": 4,
            "qk_norm_eps": 1e-6,
            "qk_norm": True,
            "use_biases": False,
            "norm_eps": 1e-2,
            "layer_scale": True,
            "layer_scale_init": 0.01,
            "encoder_transformer_lengths_str": "1,1,1,1",
            "encoder_convs_kernels_str": "4,4,4,3",
            "encoder_convs_strides_str": "2,2,2,1",
            "decoder_transformer_lengths_str": "1,1,1,1",
            "decoder_convs_kernels_str": "3,4,4,4",
            "decoder_convs_strides_str": "1,2,2,2",
        },
        "audio_model_args": audio_model_args,
    }
    tokenizer = VoxtralTTSAudioTokenizer(audio_config, text_hidden_size=text_hidden_size).eval()

    # downsample_factor = patch_size(4) * prod(decoder strides 1*2*2*2 = 8) = 32
    expected_downsample = 4 * (1 * 2 * 2 * 2)
    assert tokenizer.downsample_factor == expected_downsample, tokenizer.downsample_factor
    assert tokenizer.num_codebooks == n_codebooks, tokenizer.num_codebooks

    # Exercise weight-norm flattening with the current (random) state dict.
    tokenizer.load_weights(dict(tokenizer.state_dict()), strict=False)

    T = 6
    semantic = torch.randint(0, 32, (B, 1, T))
    acoustic = torch.randint(0, 21, (B, n_codebooks - 1, T))
    codes = torch.cat([semantic, acoustic], dim=1)
    with torch.no_grad():
        waveform = tokenizer.decode(codes, dtype=torch.float32)
    assert waveform.shape == (B, 1, T * expected_downsample), waveform.shape
    print(f"[smoke] VoxtralTTSAudioTokenizer.decode: waveform {tuple(waveform.shape)} OK")

    # --- VoxtralTTSDecoderCache --------------------------------------------
    left_ctx = torch.zeros(B, 25, n_codebooks, dtype=torch.long)
    cache = VoxtralTTSDecoderCache(left_ctx=left_ctx)
    cache2 = VoxtralTTSDecoderCache(left_ctx=torch.ones(B, 25, n_codebooks, dtype=torch.long))
    cache.copy_from(cache2)
    assert torch.all(cache.left_ctx == 1), "copy_from failed"
    merged = VoxtralTTSDecoderCache.cat([cache, cache2])
    assert merged.left_ctx.shape == (2 * B, 25, n_codebooks), merged.left_ctx.shape
    sliced = merged[:B]
    assert sliced.left_ctx.shape == (B, 25, n_codebooks), sliced.left_ctx.shape
    moved = cache.to("cpu")
    assert moved.left_ctx.device.type == "cpu"
    print("[smoke] VoxtralTTSDecoderCache: cat/copy_from/__getitem__/to OK")

    print("[smoke] all Voxtral-TTS shape smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
