from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Coroutine, List, Optional, Tuple

import torch

from ..flashinfer_utils import FlashInferWrapper
from ..requests import Request
from ..sampling import SamplingConfig
from ..tokenizer.base import DecoderCache


@dataclass
class PreprocessOutput:
    """
    Output data structure for the preprocess method of language models.

    This replaces the dictionary return format to provide better type safety
    and clearer interface for preprocessing results.
    """

    input_tokens: List[List[int]]
    repetition_cache: Optional[torch.Tensor] = None
    input_masks: Optional[torch.Tensor] = None
    input_features: Optional[torch.Tensor] = None
    decoder_cache: Optional[DecoderCache] = None


class BaseLM(ABC):
    """
    Base class for language models used in vox-serve.

    This class defines the common interface that all models must implement
    to work with the ModelWorker and scheduler system.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_torch_compile: bool = False,
        audio_decoder_device: str = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_torch_compile = enable_torch_compile
        # Audio decoder device defaults to main device if not specified
        self.audio_decoder_device = audio_decoder_device or device

    @property
    @abstractmethod
    def n_codebooks(self) -> int:
        """Number of codebooks in the model."""
        pass

    @property
    @abstractmethod
    def num_attention_heads(self) -> int:
        """Number of attention heads in the model."""
        pass

    @property
    @abstractmethod
    def num_key_value_heads(self) -> int:
        """Number of key-value heads in the model."""
        pass

    @property
    @abstractmethod
    def num_hidden_layers(self) -> int:
        """Number of hidden layers in the model."""
        pass

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Hidden size of the model.

        This is the attention working width (``num_attention_heads * head_dim``),
        used to size the FlashInfer attention wrappers. For most models it equals
        the residual-stream width; when they differ (e.g. an explicit ``head_dim``
        that makes ``n_heads * head_dim != residual_width``), override
        ``embedding_hidden_size`` to report the residual width separately.
        """
        pass

    @property
    def embedding_hidden_size(self) -> int:
        """Residual-stream / embedding width of the model.

        Used to size the ``input_features`` and ``backbone_hidden_states`` CUDA
        graph buffers. Defaults to ``hidden_size``; override when the attention
        working width differs from the residual-stream width.
        """
        return self.hidden_size

    @property
    def head_dim(self) -> int:
        """Head dimension of the model."""
        return self.hidden_size // self.num_attention_heads

    @property
    def has_depth_transformer(self) -> bool:
        """Indicates if the model has a depth transformer."""
        return False

    @property
    def has_inline_audio_head(self) -> bool:
        """Indicates if the model has an inline audio head (sampling produces audio codes directly)."""
        return False

    @property
    def first_decode_position_offset(self) -> int:
        """Offset added to ``len(input_tokens)`` to get the first decode-step position id.

        Default ``1`` preserves the vox-serve historical convention (CosyVoice2 / CSM /
        Zonos / Orpheus / Qwen3-TTS all depend on it). Models trained against a standard
        transformer pipeline (e.g. Voxtral-TTS, which mirrors vllm-omni) override to ``0``
        so the first decode position equals the prefill length rather than length+1.
        """
        return 1

    @property
    def first_chunk_frames(self) -> Optional[int]:
        """Optional small-first-chunk size for TTFA-optimized streaming.

        When ``None`` (default), the detokenizer always emits chunks of
        ``detokenize_interval`` frames. When set (e.g. ``5`` for Voxtral-TTS,
        mirroring vllm-omni's ``codec_chunk_frames_at_begin``), the model
        pre-seeds ``detokenize_interval - first_chunk_frames`` zero-coded
        silence frames into ``req.lm_output_audio_tokens`` before the first
        real frame is appended. The scheduler then dispatches the first
        detokenize chunk as soon as ``first_chunk_frames`` real frames are
        available (instead of waiting for the full ``detokenize_interval``);
        the worker trims the leading silence-frame samples from the resulting
        PCM. Subsequent chunks behave normally.

        Models that override must also rely on the codec mapping ``code 0``
        to silence (Voxtral does, via the ``(x - 2).clamp(min=0)`` shift in
        ``VoxtralTTSModel.postprocess``).
        """
        return None

    @property
    def supports_audio_input(self) -> bool:
        """Indicates if the model accepts audio input."""
        return False

    @property
    def needs_watermarking(self) -> bool:
        """Indicates if the model requires watermarking."""
        return False

    @property
    def watermarker_type(self) -> str:
        """Indicates the watermarker type to use."""
        return None

    @property
    def needs_input_features(self) -> bool:
        """Indicates if the model requires input_features."""
        return False

    @property
    def needs_input_masks(self) -> bool:
        """Indicates if the model requires input_masks."""
        return False

    @property
    def use_repetition_penalty(self) -> bool:
        """Indicates if the model has repetition penalty enabled in default sampling config."""
        return (
            hasattr(self, "default_sampling_config")
            and self.default_sampling_config.repetition_penalty is not None
            and self.default_sampling_config.repetition_penalty != 1.0
        )

    @property
    def supports_input_streaming(self) -> bool:
        """Indicates if the model supports input streaming mode."""
        return False

    def audio_decoder_initial_cache(self, batch_size: int) -> Optional[DecoderCache]:
        """
        Optional initial cache for audio decoders used during postprocessing.

        Args:
            batch_size: Desired batch size for the cache instance.

        Returns:
            None by default; models with audio decoders may override to return
            a DecoderCache sized for `batch_size`.
        """
        return None

    @property
    @abstractmethod
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        pass

    @property
    @abstractmethod
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """
        Maximum number of tokens the model generates in a single request.
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        pass

    @abstractmethod
    def is_stop_id(self, token_ids: List[int]) -> int:
        """
        Check if the given token ID is a stop token.

        Args:
            token_ids: List of token IDs to check. Shape: (n_codebooks,)

        Returns:
            True if the token ID is a stop token, False otherwise
        """
        pass

    @abstractmethod
    def preprocess(self, prompt: str = None, audio_path: str = None, **kwargs) -> PreprocessOutput:
        """
        Preprocess the input prompt for the model.

        Args:
            prompt: Input text prompt (optional if audio_path provided)
            audio_path: Path to input audio file (optional)
            **kwargs: Additional model-specific parameters

        Returns:
            PreprocessOutput containing input tokens and additional model-specific data
        """
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs. Shape: (batch_size, n_codebooks)
            position_ids: Position IDs for the tokens. Shape: (batch_size)
            attn_wrapper: FlashInfer attention wrapper
            kv_cache: KV cache tensor
            **kwargs: Additional model-specific parameters

        Returns:
            Output logits tensor. Shape: (batch_size, n_codebooks, vocab_size)
        """
        pass

    @abstractmethod
    def sampling(
        self,
        logits: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None,
        repetition_cache: torch.Tensor | None,
        cfg_scale: float | None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Coroutine]:
        """
        Sampling and other model-specific logics for generating output tokens.
        `requests` will be updated with the sampled tokens.

        Args:
            logits: Output logits from the model. Shape: (batch_size, n_codebooks, vocab_size)
            requests: List of Request objects containing sampling configurations etc.
            sampling_params: Optional common sampling configurations
            repetition_cache: Optional tensor for repetition penalty.
                Shape: (batch_size, window_size, n_codebooks, vocab_size)
            cfg_scale: Optional common classifier-free guidance scale
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple containing:
                - Output token IDs from sampling. Shape: (batch_size, n_codebooks)
                - Coroutine for request state update, for asynchronous scheduling
        """
        pass

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        pass

    @property
    @abstractmethod
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        pass

    @abstractmethod
    def postprocess(self, token_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert model output tokens to audio bytes. This should include model-specific logic
        on when to do detokenization for each request.

        Args:
            token_ids: token IDs generated by the model. Shape: (batch_size, interval, n_codebooks)
            kwargs: Additional model-specific parameters

        Returns:
            Tensor of audio data. Shape: (batch_size, n_channels, audio_length)
        """
        pass


class BaseLMWithDepth(BaseLM):
    """
    Base class for language models with depth transformer used in vox-serve.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_torch_compile: bool = False,
        audio_decoder_device: str = None,
    ):
        super().__init__(model_name, device, dtype, enable_torch_compile, audio_decoder_device)

    @property
    def has_depth_transformer(self) -> bool:
        """Indicates if the model has a depth transformer."""
        return True

    @property
    @abstractmethod
    def depth_n_codebooks(self) -> int:
        """Number of codebooks in the depth transformer."""
        pass

    @property
    @abstractmethod
    def depth_num_attention_heads(self) -> int:
        """Number of attention heads in the depth transformer."""
        pass

    @property
    @abstractmethod
    def depth_num_key_value_heads(self) -> int:
        """Number of key-value heads in the depth transformer."""
        pass

    @property
    @abstractmethod
    def depth_num_hidden_layers(self) -> int:
        """Number of hidden layers in the depth transformer."""
        pass

    @property
    @abstractmethod
    def depth_hidden_size(self) -> int:
        """Hidden size of the depth transformer."""
        pass

    @property
    @abstractmethod
    def depth_head_dim(self) -> int:
        """Head dimension in the depth transformer."""
        pass

    @property
    @abstractmethod
    def depth_vocab_size(self) -> int:
        """Vocabulary size of the depth transformer output."""
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the backbone model.

        Args:
            input_ids: Input token IDs. Shape: (batch_size, n_codebooks)
            position_ids: Position IDs for the tokens. Shape: (batch_size)
            attn_wrapper: FlashInfer attention wrapper
            kv_cache: KV cache tensor
            **kwargs: Additional model-specific parameters

        Returns:
            logits tensor. Shape: (batch_size, n_codebooks, vocab_size)
            input feature for depth transformer. Shape: (batch_size, hidden_size)
        """
        pass

    @abstractmethod
    def sampling(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None,
        cfg_scale: float | None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sampling and other model-specific logics for generating output tokens.
        `requests` will be updated with the sampled tokens.

        Args:
            logits: Output logits from the model. Shape: (batch_size, n_codebooks, vocab_size)
            hidden_states: Output hidden states from the backbone model. Shape: (batch_size, hidden_size)
            requests: List of Request objects containing sampling configurations etc.
            sampling_params: Optional common sampling configurations
            cfg_scale: Optional common classifier-free guidance scale
            **kwargs: Additional model-specific parameters

        Returns:
            output token IDs from sampling. Shape: (batch_size, n_codebooks)
            input feature for depth transformer. Shape: (batch_size, hidden_size)
        """
        pass

    @abstractmethod
    def depth_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the depth transformer for some models.

        Args:
            hidden_states: Output hidden states from the previous iteration or backbone model.
                Shape: (batch_size, hidden_size)
            position_ids: Position IDs for the tokens. Shape: (batch_size)
            attn_wrapper: FlashInfer attention wrapper
            kv_cache: KV cache tensor
            **kwargs: Additional model-specific parameters

        Returns:
            Output logits tensor. Shape: (batch_size, vocab_size)
        """
        assert self.has_depth_transformer, "This model does not support depth transformer."
        pass

    def depth_forward_unrolled(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_pos: int,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the depth transformer using SDPA (for unrolled CUDA graph).

        Uses torch.nn.functional.scaled_dot_product_attention with a dense KV cache
        instead of FlashInfer, avoiding workspace buffer conflicts in multi-step graphs.

        Args:
            hidden_states: (bs, seq_len, hidden_size) batched input embeddings
            position_ids: (bs, seq_len) position IDs
            kv_cache: (n_layers, bs, 2, max_seq_len, n_kv_heads, head_dim) dense KV cache
            cache_pos: write position in the cache

        Returns:
            Output logits tensor. Shape: (bs, seq_len, vocab_size)
        """
        raise NotImplementedError("depth_forward_unrolled not implemented for this model")

    @abstractmethod
    def depth_sampling(
        self,
        logits: torch.Tensor,
        i_iteration: int,
        requests: List[Request],
        sampling_params: SamplingConfig | None,
        cfg_scale: float | None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sampling for generating output tokens from the depth transformer model.

        Args:
            logits: Output logits from the model. Shape: (batch_size, n_codebooks, vocab_size)
            i_iteration: Current iteration number for the request.
            hidden_states: Output hidden states from the backbone model. Shape: (batch_size, hidden_size)
            requests: List of Request objects containing sampling configurations etc.
            sampling_params: Optional common sampling configurations
            cfg_scale: Optional common classifier-free guidance scale
            **kwargs: Additional model-specific parameters

        Returns:
            output token IDs from sampling. Shape: (batch_size, n_codebooks)
            input feature for the next iteration. Shape: (batch_size, hidden_size)
        """
        pass

    @abstractmethod
    def depth_sampling_gpu(
        self,
        logits: torch.Tensor,
        i_iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-only depth sampling for CUDA graph capture.
        Performs sampling and embedding lookup without CPU-side request updates.

        Args:
            logits: Output logits from depth transformer. Shape: (batch_size, depth_vocab_size)
            i_iteration: Current codebook iteration (1 to depth_n_codebooks-1)

        Returns:
            sampled token IDs. Shape: (batch_size,)
            embeddings for next iteration. Shape: (batch_size, hidden_size)
        """
        pass

    def depth_update_requests(
        self,
        all_output_ids: torch.Tensor,
        requests: List[Request],
        embed_accum: Optional[torch.Tensor] = None,
    ) -> None:
        """
        CPU-only request state update after all depth iterations complete.
        Called after unrolled depth CUDA graph replay.

        Args:
            all_output_ids: All sampled token IDs. Shape: (batch_size, depth_n_codebooks)
            requests: List of Request objects to update
            embed_accum: Accumulated embeddings across depth iterations.
                Shape: (batch_size, hidden_size). Only used by models that accumulate
                depth embeddings (e.g. qwen3-tts).
        """
        for i in range(1, self.depth_n_codebooks):
            for j, req in enumerate(requests):
                token_id = int(all_output_ids[j, i].item())
                req.lm_output_tokens[-1][0, i] = token_id
                if not req.done_lm_generation:
                    req.lm_output_audio_tokens[-1][0, i] = token_id
