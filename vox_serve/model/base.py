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
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_torch_compile = enable_torch_compile

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
    def has_depth_transformer(self) -> bool:
        """Indicates if the model has a depth transformer."""
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
        return (hasattr(self, 'default_sampling_config') and
                self.default_sampling_config.repetition_penalty is not None and
                self.default_sampling_config.repetition_penalty != 1.0)

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
    def postprocess(
        self,
        token_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
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
    ):
        super().__init__(model_name, device, dtype, enable_torch_compile)

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
