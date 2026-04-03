"""Voxtral-4B-TTS model implementation for VoxServe."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn

from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ..requests import Request
from ..sampling import SamplingConfig
from ..tokenizer.voxtral_tts import (
    AudioTokenizerArgs,
    VoxtralCodecDecoder,
    VoxtralDecoderCache,
)
from ..utils import get_logger
from .base import BaseLM, PreprocessOutput

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VoxtralTTSConfig:
    dim: int = 3072
    n_layers: int = 26
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 131072
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    # Audio-specific
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    sampling_rate: int = 24000
    frame_rate: float = 12.5
    # Acoustic transformer
    acoustic_n_layers: int = 3
    acoustic_dim: int = 3072
    acoustic_head_dim: int = 128
    acoustic_hidden_dim: int = 9216
    acoustic_n_heads: int = 32
    acoustic_n_kv_heads: int = 8
    acoustic_rope_theta: float = 10000.0
    acoustic_sigma: float = 1e-5
    # Token IDs
    bos_token_id: int = 1
    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    # Semantic output head size (padded to multiple of 128)
    semantic_output_size: int = 8320


# ---------------------------------------------------------------------------
# LLM Backbone layers (Mistral architecture with FlashInfer)
# ---------------------------------------------------------------------------


class VoxtralTTSRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class VoxtralTTSMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class VoxtralTTSAttention(nn.Module):
    def __init__(self, config: VoxtralTTSConfig, layer_id: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.layer_id = layer_id

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]

        q = self.wq(x).view(B, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, self.n_kv_heads, self.head_dim)

        q, k = apply_rope_pos_ids(q, k, position_ids, rope_theta=self.rope_theta)

        attn_wrapper.set_kv_cache(kv_cache[self.layer_id], k, v)
        output = attn_wrapper.run(q, kv_cache[self.layer_id])

        output = output.view(B, -1)
        return self.wo(output)


class VoxtralTTSDecoderLayer(nn.Module):
    def __init__(self, config: VoxtralTTSConfig, layer_id: int):
        super().__init__()
        self.attention = VoxtralTTSAttention(config, layer_id)
        self.feed_forward = VoxtralTTSMLP(config.dim, config.hidden_dim)
        self.attention_norm = VoxtralTTSRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = VoxtralTTSRMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), position_ids, attn_wrapper, kv_cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class VoxtralTTSBackbone(nn.Module):
    """26-layer Mistral transformer backbone."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.layers = nn.ModuleList([VoxtralTTSDecoderLayer(config, i) for i in range(config.n_layers)])
        self.norm = VoxtralTTSRMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, position_ids, attn_wrapper, kv_cache)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Acoustic Transformer (Flow-Matching)
# ---------------------------------------------------------------------------


class VoxtralTTSTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding for flow-matching."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        half_dim = dim // 2
        inv_freq = torch.exp(-math.log(theta) * torch.arange(half_dim).float() / half_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float [0, 1] -> (B, dim) embedding"""
        freqs = t.unsqueeze(1) * self.inv_freq.unsqueeze(0)  # (B, half_dim)
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # (B, dim)


class VoxtralTTSAcousticAttention(nn.Module):
    """Bidirectional attention for acoustic transformer (no RoPE, no causal mask)."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, use_biases: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.repeats = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=use_biases)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=use_biases)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=use_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, dim) -> (B, dim)"""
        B = x.shape[0]

        # Unsqueeze to (B, 1, dim) for attention
        x_3d = x.unsqueeze(1)

        q = self.wq(x_3d).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x_3d).view(B, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x_3d).view(B, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA expansion
        if self.repeats > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).reshape(B, self.n_heads, 1, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).reshape(B, self.n_heads, 1, self.head_dim)

        # Simple attention (seq_len=1, so trivial)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(B, -1)
        return self.wo(output)


class VoxtralTTSAcousticTransformerBlock(nn.Module):
    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.attention = VoxtralTTSAcousticAttention(
            config.acoustic_dim,
            config.acoustic_n_heads,
            config.acoustic_n_kv_heads,
            config.acoustic_head_dim,
        )
        self.feed_forward = VoxtralTTSMLP(config.acoustic_dim, config.acoustic_hidden_dim)
        self.attention_norm = VoxtralTTSRMSNorm(config.acoustic_dim, config.norm_eps)
        self.ffn_norm = VoxtralTTSRMSNorm(config.acoustic_dim, config.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class VoxtralTTSFlowMatchingAcousticTransformer(nn.Module):
    """
    Flow-matching acoustic transformer.
    Converts LLM hidden states to semantic + acoustic codes per audio frame.
    """

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.config = config
        self.n_acoustic_codebook = config.n_acoustic_codebook
        self.acoustic_codebook_size = config.acoustic_codebook_size
        self.sigma = config.acoustic_sigma

        # Projections
        self.input_projection = nn.Linear(config.n_acoustic_codebook, config.acoustic_dim, bias=False)
        self.llm_projection = nn.Linear(config.acoustic_dim, config.acoustic_dim, bias=False)
        self.time_projection = nn.Linear(config.acoustic_dim, config.acoustic_dim, bias=False)

        # Time embedding
        self.time_embedding = VoxtralTTSTimeEmbedding(config.acoustic_dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [VoxtralTTSAcousticTransformerBlock(config) for _ in range(config.acoustic_n_layers)]
        )
        self.norm = VoxtralTTSRMSNorm(config.acoustic_dim, config.norm_eps)

        # Output heads
        self.semantic_codebook_output = nn.Linear(config.acoustic_dim, config.semantic_output_size, bias=False)
        self.acoustic_codebook_output = nn.Linear(config.acoustic_dim, config.n_acoustic_codebook, bias=False)

        # Flow matching config
        self._acoustic_decode_iters = 8
        self._cfg_alpha = 1.2
        self._noise_scale = 1.0
        self.register_buffer(
            "_timesteps",
            torch.linspace(0, 1, self._acoustic_decode_iters + 1),
            persistent=False,
        )

    def _predict_velocity(
        self,
        x_t: torch.Tensor,  # (B, n_acoustic)
        llm_output: torch.Tensor,  # (B, dim)
        t_emb: torch.Tensor,  # (B, dim)
    ) -> torch.Tensor:
        """Predict velocity field for flow matching."""
        h = self.input_projection(x_t)  # (B, dim)
        h = h + llm_output + t_emb

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        return self.acoustic_codebook_output(h)  # (B, n_acoustic)

    @torch.no_grad()
    def forward(self, llm_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate semantic and acoustic codes from LLM hidden state.

        Args:
            llm_hidden: (B, dim) LLM last hidden state

        Returns:
            semantic_codes: (B,) int semantic code indices
            acoustic_codes: (B, 36) int acoustic code indices [0, 20]
        """
        B = llm_hidden.shape[0]
        device = llm_hidden.device

        # Project LLM hidden
        llm_output = self.llm_projection(llm_hidden)  # (B, dim)

        # Semantic code: argmax on semantic head
        semantic_logits = self.semantic_codebook_output(llm_output)  # (B, semantic_output_size)
        semantic_codes = semantic_logits[:, : self.config.semantic_codebook_size].argmax(dim=-1)  # (B,)

        # Acoustic codes via flow matching (Euler integration)
        x_t = self._noise_scale * torch.randn(B, self.n_acoustic_codebook, device=device, dtype=llm_hidden.dtype)

        timesteps = self._timesteps.to(device)
        for i in range(self._acoustic_decode_iters):
            t = timesteps[i].expand(B)
            t_emb = self.time_projection(self.time_embedding(t))  # (B, dim)
            dt = timesteps[i + 1] - timesteps[i]

            # Classifier-free guidance
            # Conditional prediction
            v_cond = self._predict_velocity(x_t, llm_output, t_emb)
            # Unconditional prediction (zero LLM context)
            v_uncond = self._predict_velocity(x_t, torch.zeros_like(llm_output), t_emb)

            # CFG combination
            v_t = self._cfg_alpha * v_cond + (1 - self._cfg_alpha) * v_uncond
            x_t = x_t + v_t * dt

        # Quantize to FSQ codes
        x_t = x_t.clamp(-1, 1)
        acoustic_codes = ((x_t + 1) / 2 * (self.acoustic_codebook_size - 1)).round().long()
        acoustic_codes = acoustic_codes.clamp(0, self.acoustic_codebook_size - 1)

        return semantic_codes, acoustic_codes

    @torch.no_grad()
    def forward_from_projected(
        self, llm_output: torch.Tensor, semantic_codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate acoustic codes from already-projected LLM hidden state.
        Used when llm_projection was already applied in forward().

        Args:
            llm_output: (B, dim) pre-projected LLM hidden state
            semantic_codes: (B,) already-sampled semantic codes

        Returns:
            semantic_codes: (B,) passed through unchanged
            acoustic_codes: (B, 36) int acoustic code indices [0, 20]
        """
        B = llm_output.shape[0]
        device = llm_output.device

        # Acoustic codes via flow matching (Euler integration)
        x_t = self._noise_scale * torch.randn(B, self.n_acoustic_codebook, device=device, dtype=llm_output.dtype)

        timesteps = self._timesteps.to(device)
        for i in range(self._acoustic_decode_iters):
            t = timesteps[i].expand(B)
            t_emb = self.time_projection(self.time_embedding(t))  # (B, dim)
            dt = timesteps[i + 1] - timesteps[i]

            # Classifier-free guidance
            v_cond = self._predict_velocity(x_t, llm_output, t_emb)
            v_uncond = self._predict_velocity(x_t, torch.zeros_like(llm_output), t_emb)

            v_t = self._cfg_alpha * v_cond + (1 - self._cfg_alpha) * v_uncond
            x_t = x_t + v_t * dt

        # Quantize to FSQ codes
        x_t = x_t.clamp(-1, 1)
        acoustic_codes = ((x_t + 1) / 2 * (self.acoustic_codebook_size - 1)).round().long()
        acoustic_codes = acoustic_codes.clamp(0, self.acoustic_codebook_size - 1)

        return semantic_codes, acoustic_codes


# ---------------------------------------------------------------------------
# Audio codebook embedding (for computing next-step input)
# ---------------------------------------------------------------------------


class VoxtralTTSAudioCodebookEmbedding(nn.Module):
    """
    Multi-vocabulary embedding that sums embeddings from all 37 codebooks.
    Loaded from mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight.
    """

    def __init__(self, total_vocab_size: int, dim: int, codebook_sizes: List[int]):
        super().__init__()
        self.embeddings = nn.Embedding(total_vocab_size, dim)
        self.codebook_sizes = codebook_sizes
        # Compute offsets for each codebook
        offsets = [0]
        for s in codebook_sizes[:-1]:
            offsets.append(offsets[-1] + s)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Embed and sum codes from all codebooks.

        Args:
            codes: (B, n_codebooks) integer codes

        Returns:
            (B, dim) summed embeddings
        """
        B = codes.shape[0]
        n_cb = codes.shape[1]
        result = torch.zeros(B, self.embeddings.embedding_dim, device=codes.device, dtype=self.embeddings.weight.dtype)

        for i in range(n_cb):
            idx = codes[:, i] + self.offsets[i]
            result = result + self.embeddings(idx)

        return result


# ---------------------------------------------------------------------------
# Tekken tokenizer wrapper
# ---------------------------------------------------------------------------


class VoxtralTTSTekkenTokenizer:
    """Wrapper for Mistral's Tekken tokenizer using mistral_common."""

    def __init__(self, model_path: str):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        self.tokenizer = MistralTokenizer.from_file(model_path)

    def encode_speech(self, text: str, voice: str = "neutral_female") -> List[int]:
        """Encode a TTS request to token IDs using the proper speech format.

        Returns token IDs in the format:
            [BOS, BEGIN_AUDIO, AUDIO×N, text_tokens..., BEGIN_AUDIO]
        where AUDIO positions (token_id=24) should have their embeddings
        replaced with the voice embedding.
        """
        from mistral_common.protocol.speech.request import SpeechRequest

        request = SpeechRequest(input=text, voice=voice)
        tokenized = self.tokenizer.encode_speech_request(request)
        return tokenized.tokens


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class VoxtralTTSModel(BaseLM):
    """Voxtral-4B-TTS model for VoxServe."""

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
        enable_torch_compile: bool = False,
        audio_decoder_device: str = None,
    ):
        if model_name in ("voxtral", "voxtral-tts"):
            model_name = "mistralai/Voxtral-4B-TTS-2603"
        super().__init__(model_name, device, dtype, enable_torch_compile, audio_decoder_device)
        self.logger = get_logger(__name__)

        # Load params.json
        self.config = self._load_config(model_name)

        # Build LLM backbone
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.backbone = VoxtralTTSBackbone(self.config)

        # Build acoustic transformer
        self.acoustic_transformer = VoxtralTTSFlowMatchingAcousticTransformer(self.config)

        # Build audio codebook embedding for next-step input computation
        # Codebook sizes include 2 special tokens (EMPTY_AUDIO, END_AUDIO) per codebook.
        # These must match the offsets used during training, NOT the logit head size.
        n_special = 2
        semantic_cb_size = self.config.semantic_codebook_size + n_special  # 8194
        acoustic_cb_size = self.config.acoustic_codebook_size + n_special  # 23
        codebook_sizes = [semantic_cb_size] + [acoustic_cb_size] * self.config.n_acoustic_codebook
        self._codebook_sizes = codebook_sizes

        # Build codec decoder
        codec_args = AudioTokenizerArgs(
            sampling_rate=self.config.sampling_rate,
            semantic_codebook_size=self.config.semantic_codebook_size,
            acoustic_codebook_size=self.config.acoustic_codebook_size,
            acoustic_dim=self.config.n_acoustic_codebook,
        )
        self.codec_decoder = VoxtralCodecDecoder(codec_args)

        # Load weights
        self._load_weights(model_name)

        # Move to device
        self.tok_embeddings.to(dtype).to(device)
        self.backbone.to(dtype).to(device)
        self.acoustic_transformer.to(dtype).to(device)
        self.codec_decoder.to(dtype).to(self.audio_decoder_device)
        if hasattr(self, "audio_codebook_embedding"):
            self.audio_codebook_embedding.to(dtype).to(device)

        # Load voice embeddings
        self.voice_embeddings = self._load_voice_embeddings(model_name)

        # Tokenizer
        self._init_tokenizer(model_name)

        # Hidden state buffer for passing from forward() to sampling()
        self._last_hidden: Optional[torch.Tensor] = None

        # Sampling config
        self.default_sampling_config = SamplingConfig(
            top_k=50,
            top_p=None,
            min_p=None,
            temperature=0.7,
            repetition_penalty=None,
            repetition_window=None,
            cfg_scale=None,
        )

        self._num_attention_heads = self.config.n_heads
        self._num_key_value_heads = self.config.n_kv_heads
        self._num_hidden_layers = self.config.n_layers
        self._hidden_size = self.config.dim

        # Special token IDs in the semantic head output space
        self._empty_audio_id = 0  # masked during sampling
        self._end_audio_id = 1  # stop signal
        self._semantic_offset = 2  # actual VQ codes start at index 2

    def _load_config(self, model_name: str) -> VoxtralTTSConfig:
        """Load model config from params.json."""
        is_local = Path(model_name).is_dir()
        if is_local:
            params_path = Path(model_name) / "params.json"
        else:
            params_path = hf_hub_download(repo_id=model_name, filename="params.json")

        with open(params_path) as f:
            params = json.load(f)

        multimodal = params.get("multimodal", {})
        audio_model = multimodal.get("audio_model_args", {})
        acoustic_args = audio_model.get("acoustic_transformer_args", {})
        tokenizer_args = multimodal.get("audio_tokenizer_args", {})
        audio_encoding = audio_model.get("audio_encoding_args", {})

        return VoxtralTTSConfig(
            dim=params.get("dim", 3072),
            n_layers=params.get("n_layers", 26),
            head_dim=params.get("head_dim", 128),
            hidden_dim=params.get("hidden_dim", 9216),
            n_heads=params.get("n_heads", 32),
            n_kv_heads=params.get("n_kv_heads", 8),
            vocab_size=params.get("vocab_size", 131072),
            rope_theta=params.get("rope_theta", 1000000.0),
            norm_eps=params.get("norm_eps", 1e-5),
            semantic_codebook_size=audio_model.get("semantic_codebook_size", 8192),
            acoustic_codebook_size=audio_model.get("acoustic_codebook_size", 21),
            n_acoustic_codebook=audio_model.get("n_acoustic_codebook", 36),
            sampling_rate=audio_encoding.get("sampling_rate", tokenizer_args.get("sampling_rate", 24000)),
            frame_rate=audio_encoding.get("frame_rate", 12.5),
            acoustic_n_layers=acoustic_args.get("n_layers", 3),
            acoustic_dim=acoustic_args.get("dim", 3072),
            acoustic_head_dim=acoustic_args.get("head_dim", 128),
            acoustic_hidden_dim=acoustic_args.get("hidden_dim", 9216),
            acoustic_n_heads=acoustic_args.get("n_heads", 32),
            acoustic_n_kv_heads=acoustic_args.get("n_kv_heads", 8),
            acoustic_rope_theta=acoustic_args.get("rope_theta", 10000.0),
            acoustic_sigma=acoustic_args.get("sigma", 1e-5),
            bos_token_id=multimodal.get("bos_token_id", 1),
            audio_token_id=audio_model.get("audio_token_id", 24),
            begin_audio_token_id=audio_model.get("begin_audio_token_id", 25),
        )

    def _load_weights(self, model_name: str):
        """Load weights from consolidated.safetensors."""
        is_local = Path(model_name).is_dir()
        if is_local:
            weights_path = Path(model_name) / "consolidated.safetensors"
        else:
            weights_path = hf_hub_download(repo_id=model_name, filename="consolidated.safetensors")

        self.logger.info("Loading weights from %s", weights_path)

        with safe_open(weights_path, framework="pt", device="cpu") as f:
            weight_keys = list(f.keys())

            for key in weight_keys:
                tensor = f.get_tensor(key)
                self._assign_weight(key, tensor)

        self.logger.info("Weights loaded successfully")

    def _assign_weight(self, key: str, tensor: torch.Tensor):
        """Map a weight key from consolidated.safetensors to our model structure."""

        # LLM backbone layers
        if key.startswith("layers."):
            # layers.{i}.attention.{wq,wk,wv,wo}.weight
            # layers.{i}.feed_forward.{w1,w2,w3}.weight
            # layers.{i}.{attention_norm,ffn_norm}.weight
            parts = key.split(".")
            layer_idx = int(parts[1])
            layer = self.backbone.layers[layer_idx]

            if parts[2] == "attention":
                attn = layer.attention
                if parts[3] == "wq":
                    attn.wq.weight.data.copy_(tensor)
                elif parts[3] == "wk":
                    attn.wk.weight.data.copy_(tensor)
                elif parts[3] == "wv":
                    attn.wv.weight.data.copy_(tensor)
                elif parts[3] == "wo":
                    attn.wo.weight.data.copy_(tensor)
            elif parts[2] == "feed_forward":
                mlp = layer.feed_forward
                if parts[3] == "w1":
                    mlp.w1.weight.data.copy_(tensor)
                elif parts[3] == "w2":
                    mlp.w2.weight.data.copy_(tensor)
                elif parts[3] == "w3":
                    mlp.w3.weight.data.copy_(tensor)
            elif parts[2] == "attention_norm":
                layer.attention_norm.weight.data.copy_(tensor)
            elif parts[2] == "ffn_norm":
                layer.ffn_norm.weight.data.copy_(tensor)

        # Final norm
        elif key == "norm.weight":
            self.backbone.norm.weight.data.copy_(tensor)

        # Text embeddings
        elif key == "mm_audio_embeddings.tok_embeddings.weight":
            self.tok_embeddings.weight.data.copy_(tensor)

        # Audio codebook embeddings
        elif key == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight":
            total_size = tensor.shape[0]
            dim = tensor.shape[1]
            self.audio_codebook_embedding = VoxtralTTSAudioCodebookEmbedding(
                total_size,
                dim,
                self._codebook_sizes,
            )
            self.audio_codebook_embedding.embeddings.weight.data.copy_(tensor)

        # Acoustic transformer
        elif key.startswith("acoustic_transformer."):
            remainder = key[len("acoustic_transformer.") :]
            self._assign_acoustic_weight(remainder, tensor)

        # Audio tokenizer / codec decoder
        elif key.startswith("audio_tokenizer."):
            remainder = key[len("audio_tokenizer.") :]
            self._assign_codec_weight(remainder, tensor)

    def _assign_acoustic_weight(self, key: str, tensor: torch.Tensor):
        """Assign acoustic transformer weights."""
        at = self.acoustic_transformer
        parts = key.split(".")

        if parts[0] == "layers":
            layer_idx = int(parts[1])
            layer = at.layers[layer_idx]

            if parts[2] == "attention":
                attn = layer.attention
                if parts[3] == "wq":
                    attn.wq.weight.data.copy_(tensor)
                elif parts[3] == "wk":
                    attn.wk.weight.data.copy_(tensor)
                elif parts[3] == "wv":
                    attn.wv.weight.data.copy_(tensor)
                elif parts[3] == "wo":
                    attn.wo.weight.data.copy_(tensor)
            elif parts[2] == "feed_forward":
                mlp = layer.feed_forward
                if parts[3] == "w1":
                    mlp.w1.weight.data.copy_(tensor)
                elif parts[3] == "w2":
                    mlp.w2.weight.data.copy_(tensor)
                elif parts[3] == "w3":
                    mlp.w3.weight.data.copy_(tensor)
            elif parts[2] == "attention_norm":
                layer.attention_norm.weight.data.copy_(tensor)
            elif parts[2] == "ffn_norm":
                layer.ffn_norm.weight.data.copy_(tensor)

        elif parts[0] == "norm":
            at.norm.weight.data.copy_(tensor)
        elif parts[0] == "input_projection":
            at.input_projection.weight.data.copy_(tensor)
        elif parts[0] == "llm_projection":
            at.llm_projection.weight.data.copy_(tensor)
        elif parts[0] == "time_projection":
            at.time_projection.weight.data.copy_(tensor)
        elif parts[0] == "semantic_codebook_output":
            at.semantic_codebook_output.weight.data.copy_(tensor)
        elif parts[0] == "acoustic_codebook_output":
            at.acoustic_codebook_output.weight.data.copy_(tensor)

    def _assign_codec_weight(self, key: str, tensor: torch.Tensor):
        """Assign codec decoder weights."""
        cd = self.codec_decoder

        if key.startswith("decoder_blocks."):
            parts = key.split(".", 2)
            block_idx = int(parts[1])
            remainder = parts[2]

            block = cd.decoder_blocks[block_idx]

            # Handle weight-normed conv layers
            if remainder.startswith("conv.parametrizations.weight."):
                param_name = remainder.split(".")[-1]
                if param_name == "original0":
                    block.conv.parametrizations.weight.original0.data.copy_(tensor)
                elif param_name == "original1":
                    block.conv.parametrizations.weight.original1.data.copy_(tensor)
            elif remainder.startswith("layers."):
                # Transformer layers within the block
                layer_parts = remainder.split(".", 2)
                layer_idx = int(layer_parts[1])
                layer_key = layer_parts[2]

                # Navigate to the right transformer block
                # CodecTransformer.layers is a ModuleDict
                transformer = block
                layer = transformer.layers[str(layer_idx)]

                self._assign_codec_transformer_weight(layer, layer_key, tensor)

        elif key.startswith("output_proj.conv.parametrizations.weight."):
            param_name = key.split(".")[-1]
            if param_name == "original0":
                cd.output_proj.conv.parametrizations.weight.original0.data.copy_(tensor)
            elif param_name == "original1":
                cd.output_proj.conv.parametrizations.weight.original1.data.copy_(tensor)

        elif key.startswith("quantizer."):
            q_key = key[len("quantizer.") :]
            if q_key.startswith("semantic_codebook."):
                cb_key = q_key[len("semantic_codebook.") :]
                if cb_key == "cluster_usage":
                    cd.quantizer.semantic_codebook.cluster_usage.data.copy_(tensor)
                elif cb_key == "embedding_sum":
                    cd.quantizer.semantic_codebook.embedding_sum.data.copy_(tensor)
                    # Invalidate cached embedding
                    cd.quantizer.semantic_codebook._embedding = None

    def _assign_codec_transformer_weight(self, layer, key: str, tensor: torch.Tensor):
        """Assign weight to a codec transformer block."""
        if key.startswith("attention."):
            attn_key = key[len("attention.") :]
            attn = layer.attention
            if attn_key == "wq.weight":
                attn.wq.weight.data.copy_(tensor)
            elif attn_key == "wk.weight":
                attn.wk.weight.data.copy_(tensor)
            elif attn_key == "wv.weight":
                attn.wv.weight.data.copy_(tensor)
            elif attn_key == "wo.weight":
                attn.wo.weight.data.copy_(tensor)
            elif attn_key == "q_norm.weight":
                if attn.q_norm is not None:
                    attn.q_norm.weight.data.copy_(tensor)
            elif attn_key == "k_norm.weight":
                if attn.k_norm is not None:
                    attn.k_norm.weight.data.copy_(tensor)
        elif key.startswith("feed_forward."):
            ff_key = key[len("feed_forward.") :]
            ff = layer.feed_forward
            if ff_key == "w1.weight":
                ff.w1.weight.data.copy_(tensor)
            elif ff_key == "w2.weight":
                ff.w2.weight.data.copy_(tensor)
            elif ff_key == "w3.weight":
                ff.w3.weight.data.copy_(tensor)
        elif key == "attention_norm.weight":
            layer.attention_norm.weight.data.copy_(tensor)
        elif key == "ffn_norm.weight":
            layer.ffn_norm.weight.data.copy_(tensor)
        elif key == "attention_scale":
            if layer.layer_scale:
                layer.attention_scale.data.copy_(tensor)
        elif key == "ffn_scale":
            if layer.layer_scale:
                layer.ffn_scale.data.copy_(tensor)

    def _load_voice_embeddings(self, model_name: str) -> dict:
        """Load preset voice embeddings."""
        voices = {}
        voice_names = [
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

        is_local = Path(model_name).is_dir()

        for name in voice_names:
            try:
                if is_local:
                    path = Path(model_name) / "voice_embedding" / f"{name}.pt"
                else:
                    path = hf_hub_download(
                        repo_id=model_name,
                        filename=f"voice_embedding/{name}.pt",
                    )
                voices[name] = torch.load(path, map_location="cpu", weights_only=True)
                self.logger.info("Loaded voice embedding: %s", name)
            except Exception as e:
                self.logger.warning("Could not load voice %s: %s", name, e)

        return voices

    def _init_tokenizer(self, model_name: str):
        """Initialize the Tekken text tokenizer."""
        is_local = Path(model_name).is_dir()

        try:
            if is_local:
                tekken_path = str(Path(model_name) / "tekken.json")
            else:
                tekken_path = hf_hub_download(repo_id=model_name, filename="tekken.json")

            self.text_tokenizer = VoxtralTTSTekkenTokenizer(tekken_path)
            self.logger.info("Tekken tokenizer loaded")
        except Exception as e:
            self.logger.warning("Could not load Tekken tokenizer: %s", e)
            self.text_tokenizer = None

    # ---------------------------------------------------------------------------
    # BaseLM properties
    # ---------------------------------------------------------------------------

    @property
    def n_codebooks(self) -> int:
        return 1

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
        return self.config.head_dim

    @property
    def needs_input_features(self) -> bool:
        return True

    @property
    def needs_input_masks(self) -> bool:
        return True

    @property
    def detokenize_interval(self) -> int:
        return 12  # ~1 second at 12.5 Hz

    @property
    def detokenize_overlap(self) -> int:
        return 0

    @property
    def max_tokens(self) -> int:
        return 4096

    @property
    def vocab_size(self) -> int:
        return self.config.semantic_output_size  # 8320

    @property
    def n_channels(self) -> int:
        return 1

    @property
    def output_audio_length(self) -> int:
        # 12 frames × 1920 samples/frame = 23040
        samples_per_frame = int(self.config.sampling_rate / self.config.frame_rate)
        return self.detokenize_interval * samples_per_frame

    def is_stop_id(self, token_ids: List[int]) -> bool:
        # END_AUDIO token (index 1) signals end of audio generation
        return token_ids[0] == self._end_audio_id

    def audio_decoder_initial_cache(self, batch_size: int) -> Optional[VoxtralDecoderCache]:
        """Create initial decoder cache for accumulating acoustic codes."""
        max_frames = self.max_tokens
        acoustic_codes = torch.zeros(
            batch_size,
            max_frames,
            self.config.n_acoustic_codebook,
            dtype=torch.long,
            device=self.device,
        )
        return VoxtralDecoderCache(
            acoustic_codes=acoustic_codes,
            consumed_frames=torch.zeros(1, dtype=torch.long),
        )

    # ---------------------------------------------------------------------------
    # Core methods
    # ---------------------------------------------------------------------------

    @torch.no_grad()
    def preprocess(
        self,
        prompt: str | None = None,
        audio_path: str | None = None,
        voice: str = "neutral_female",
        **kwargs,
    ) -> PreprocessOutput:
        """Tokenize text and prepare input with voice embedding."""
        assert prompt is not None, "Text prompt is required"

        # Get voice embedding
        voice_name = kwargs.get("voice", voice)
        if voice_name not in self.voice_embeddings:
            self.logger.warning("Voice '%s' not found, using 'neutral_female'", voice_name)
            voice_name = "neutral_female"
        voice_emb = self.voice_embeddings[voice_name].to(self.device).to(self.dtype)

        # Tokenize using the proper speech request format.
        # This produces: [BOS=1, BEGIN_AUDIO=25, AUDIO×N, text_tokens..., BEGIN_AUDIO=25]
        # where AUDIO (token_id=24) positions should use voice embeddings.
        token_ids = self.text_tokenizer.encode_speech(prompt, voice=voice_name)
        input_ids = torch.tensor(token_ids, dtype=torch.int32, device=self.device)
        input_ids = input_ids.view(-1, 1)  # (seq_len, 1) - add codebook dim

        # Build input features: embed each token, replacing audio positions
        # with voice embedding.
        token_ids_long = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        text_embeds = self.tok_embeddings(token_ids_long).to(self.dtype)  # (seq_len, dim)

        # Replace audio_token_id positions with voice embedding
        audio_mask = token_ids_long == self.config.audio_token_id  # (seq_len,)
        audio_positions = audio_mask.nonzero(as_tuple=True)[0]
        n_audio = audio_positions.shape[0]

        if n_audio > 0 and voice_emb.shape[0] == n_audio:
            text_embeds[audio_positions] = voice_emb.to(self.dtype)
        elif n_audio > 0:
            # Truncate or pad voice embedding to match audio positions
            n_use = min(n_audio, voice_emb.shape[0])
            text_embeds[audio_positions[:n_use]] = voice_emb[:n_use].to(self.dtype)

        input_features = text_embeds  # (seq_len, dim)

        # Input masks: True everywhere (always use input_features during prefill)
        input_masks = torch.ones(
            input_ids.shape[0],
            self.n_codebooks,
            dtype=torch.bool,
            device=self.device,
        )

        # Initial decoder cache
        decoder_cache = self.audio_decoder_initial_cache(batch_size=1)

        return PreprocessOutput(
            input_tokens=input_ids,
            input_features=input_features,
            input_masks=input_masks,
            decoder_cache=decoder_cache,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        input_masks: torch.Tensor = None,
        input_features: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass through the LLM backbone.

        During prefill: uses pre-computed input_features for all positions.
        During decode: uses input_features (combined codebook embedding from previous frame).
        """
        # Always use input_features as embedding
        if input_features is not None:
            inputs_embeds = input_features
        else:
            # Fallback: embed from input_ids (shouldn't normally happen)
            inputs_embeds = self.tok_embeddings(input_ids[:, 0].long())

        # Run through backbone
        hidden = self.backbone(inputs_embeds, position_ids, attn_wrapper, kv_cache)

        # Store raw backbone output for the acoustic transformer in sampling().
        # The llm_projection is applied in sampling() for flow-matching only.
        # Note: during prefill, B = total_tokens; the worker filters logits to
        # last-token-per-request before passing to sampling(). We store all and
        # slice in sampling().
        self._last_backbone_hidden = hidden

        # Compute semantic logits directly from backbone output (no llm_projection).
        # The semantic_codebook_output head was trained on raw backbone hidden states.
        logits = self.acoustic_transformer.semantic_codebook_output(hidden)  # (B, 8320)

        return logits[:, None, :]  # (B, 1, vocab_size) - add codebook dim

    def sampling(
        self,
        logits: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        repetition_cache: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        # Sample semantic code from logits using greedy (argmax), matching vllm-omni.
        # Mask special tokens: EMPTY_AUDIO (id=0) and anything beyond valid range.
        B = logits.shape[0]
        semantic_logits = logits[:, 0, :].float()  # (B, vocab_size)
        semantic_logits[:, 0] = float("-inf")  # mask EMPTY_AUDIO
        n_special = 2
        valid_end = n_special + self.config.semantic_codebook_size  # 2 + 8192 = 8194
        semantic_logits[:, valid_end:] = float("-inf")  # mask padding
        semantic_codes = semantic_logits.argmax(dim=-1)  # (B,) greedy

        # Get the raw backbone hidden states matching the filtered logits.
        # During prefill, _last_backbone_hidden has all tokens but logits has
        # been filtered to last-token-per-request by the worker. We take the
        # last B entries since prefill processes requests sequentially.
        backbone_hidden = self._last_backbone_hidden  # (N, dim) where N >= B
        if backbone_hidden.shape[0] != B:
            backbone_hidden = backbone_hidden[-B:]

        # Project through llm_projection for flow-matching conditioning
        llm_output = self.acoustic_transformer.llm_projection(backbone_hidden)

        # Run acoustic transformer flow-matching to get acoustic codes
        _, acoustic_codes = self.acoustic_transformer.forward_from_projected(llm_output, semantic_codes)  # (B, 36) int

        # Compute next step input embedding (sum of all 37 codebook embeddings)
        all_codes = torch.cat(
            [
                semantic_codes.unsqueeze(1),  # (B, 1)
                acoustic_codes,  # (B, 36)
            ],
            dim=1,
        )  # (B, 37)

        next_embedding = self.audio_codebook_embedding(all_codes)  # (B, dim)

        output_ids = semantic_codes.unsqueeze(1)  # (B, 1) - n_codebooks dim

        # Update request state
        for i, req in enumerate(requests):
            req.input_tokens = output_ids[i : i + 1]  # (1, 1)
            req.input_features = next_embedding[i : i + 1]  # (1, dim)
            req.input_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

            # Store acoustic codes in decoder cache
            if req.decoder_cache is not None:
                frame_idx = len(req.lm_output_audio_tokens)
                if frame_idx < req.decoder_cache.acoustic_codes.shape[1]:
                    req.decoder_cache.acoustic_codes[0, frame_idx] = acoustic_codes[i]

        async def update_req_states():
            for i, req in enumerate(requests):
                req.lm_output_tokens.append(output_ids[i : i + 1])
                req.lm_output_audio_tokens.append(output_ids[i : i + 1])

                # Check stop condition: END_AUDIO token (index 1)
                if semantic_codes[i].item() == self._end_audio_id:
                    req.done_lm_generation = True
                    req.finish_reason = "stop_id_encountered"
                if req.next_position_id > self.max_tokens:
                    req.done_lm_generation = True
                    req.finish_reason = "max_tokens_reached"

        task = update_req_states()
        return output_ids, task

    def postprocess(
        self,
        token_ids: torch.Tensor,
        decoder_cache: VoxtralDecoderCache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert semantic + acoustic codes to audio waveform.

        Args:
            token_ids: (B, interval, 1) semantic codes
            decoder_cache: contains accumulated acoustic codes
        Returns:
            audio: (B, 1, output_audio_length)
        """
        B, T, _ = token_ids.shape
        device = self.audio_decoder_device

        # Semantic codes from the model include a +2 offset for special tokens
        # (0=EMPTY_AUDIO, 1=END_AUDIO). The codec expects raw VQ codes [0, 8191].
        semantic_codes = (token_ids[:, :, 0] - self._semantic_offset).clamp(min=0).to(device)  # (B, T)

        # Retrieve acoustic codes from decoder cache
        if decoder_cache is not None:
            start_frame = int(decoder_cache.consumed_frames.item())
            acoustic = decoder_cache.acoustic_codes[:B, start_frame : start_frame + T, :].to(device)  # (B, T, 36)
            decoder_cache.consumed_frames.fill_(start_frame + T)
        else:
            # Fallback: zeros
            acoustic = torch.zeros(B, T, self.config.n_acoustic_codebook, dtype=torch.long, device=device)

        # Combine: (B, 37, T) for codec
        all_codes = torch.cat(
            [
                semantic_codes.unsqueeze(1),  # (B, 1, T)
                acoustic.permute(0, 2, 1),  # (B, 36, T)
            ],
            dim=1,
        )  # (B, 37, T)

        # Decode to audio (use float32 for numpy compatibility)
        audio = self.codec_decoder.decode(all_codes, dtype=torch.float32)  # (B, 1, samples)

        # Ensure output length matches expected
        expected = self.output_audio_length
        if audio.shape[2] > expected:
            audio = audio[:, :, :expected]
        elif audio.shape[2] < expected:
            audio = F.pad(audio, (0, expected - audio.shape[2]))

        return audio
