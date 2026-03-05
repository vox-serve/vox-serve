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


@dataclass
class VibeVoiceForwardOutput:
    """
    Output data structure for the unified forward pass of VibeVoice models.

    This dataclass captures the outputs from both the LM backbone and TTS LM
    in a single coherent structure, supporting CUDA graph compatibility.

    Attributes:
        lm_hidden_state: Hidden states from the LM backbone.
            Shape: [batch_size, seq_len, hidden_size] for text phase,
            or None for speech phase (when LM is not invoked).
        tts_hidden_state: Final hidden states from the TTS backbone.
            Shape: [batch_size, 1, hidden_size]
        eos_logits: Binary classification logits for EOS prediction.
            Shape: [batch_size, 1]
    """

    lm_hidden_state: Optional[torch.FloatTensor]
    tts_hidden_state: torch.FloatTensor
    eos_logits: torch.FloatTensor


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
        """Hidden size of the model."""
        pass

    @property
    def head_dim(self) -> int:
        """Head dimension of the model."""
        return self.hidden_size // self.num_attention_heads

    @property
    def has_depth_transformer(self) -> bool:
        """Indicates if the model has a depth transformer."""
        return False

    @property
    def has_continuous_speech(self) -> bool:
        """Indicates if the model generates continuous speech latents."""
        return False

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

class BaseLMWithContinuousSpeech(BaseLM):
    """
    Base class for language models that generate continuous speech representations.

    These models output continuous latent vectors that are processed by a diffusion head
    and then decoded to audio by an acoustic tokenizer decoder.
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
    def has_continuous_speech(self) -> bool:
        """Indicates if the model generates continuous speech latents."""
        return True

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension of the continuous speech latents."""
        pass

    @abstractmethod
    def forward_lm(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.Tensor]:
        """
        Single pass of the base text LM.

        - Builds embeddings if `inputs_embeds` not provided.
        - Uses (and returns) `past_key_values` when `use_cache=True`.
        - No loss / no lm_head / no speech logic.

        Args:
            input_ids: (B, S) token ids.
            attention_mask: (B, S) mask.
            past_key_values: cache from previous steps.
            cache_position: positions for cached tokens.
            labels: unsupported (will raise).

        Returns:
            Originally returned BaseModelOutputWithPast with `last_hidden_state` and `past_key_values`.
            
            - last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
              Sequence of hidden-states at the output of the last layer of the model.
              If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
            - past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
              It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
              Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        """
        pass

    @abstractmethod
    def forward_tts_lm(
        self,
        input_ids: torch.Tensor,           # (batch_size, sequence_length)
        position_ids: torch.Tensor,         # (batch_size,)
        attn_wrapper: FlashInferWrapper,    # FlashInfer wrapper
        kv_cache: torch.Tensor,             # KV cache for TTS backbone
        lm_last_hidden_state: torch.FloatTensor,  # (batch_size, 1, hidden_size)
        tts_text_masks: torch.Tensor,       # (batch_size, 1) - 1=text, 0=speech
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.Tensor]:
        """
        Single pass of the TTS LM.

        - Overwrites tail embeddings with `lm_last_hidden_state`.
        - Adds type embedding via `tts_text_masks` (1=text, 0=speech).
        - Predicts EOS from last hidden state (binary classifier).
        - No loss / no full acoustic decoding here.

        Args:
            input_ids: (batch_size, n_codebooks) token ids.
            position_ids: (batch_size,) position ids for tokens.
            attn_wrapper: FlashInfer attention wrapper.
            kv_cache: KV cache tensor for TTS backbone.
            lm_last_hidden_state: (batch_size, 1, hidden_size) hidden states from LM backbone.
            tts_text_masks: (batch_size, 1) mask marking current position as text(1)/speech(0).

        Returns:
            Tuple containing:
                - tts_eos_logits: (batch_size, 1) - EOS classification logits
                - last_hidden_state: (batch_size, 1, hidden_size) - final hidden state
                - kv_cache: Updated KV cache
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