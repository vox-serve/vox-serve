from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict
import torch

from ..flashinfer_utils import FlashInferWrapper
from ..sampling import SamplingConfig
from ..requests import Request


class BaseLM(ABC):
    """
    Base class for language models used in vox-serve.
    
    This class defines the common interface that all models must implement
    to work with the ModelWorker and scheduler system.
    """
    
    def __init__(self, model_name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
    
    @property
    def has_depth_transformer(self) -> bool:
        """Indicates if the model has a depth transformer."""
        return False
    
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
    def preprocess(self, prompt: str, **kwargs) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Preprocess the input prompt for the model.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional model-specific parameters
            
        Returns:
            Tuple of (input_token_ids, formatted_prompt_string)
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
            Output logits tensor
        """
        pass

    def depth_forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: torch.Tensor, 
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the depth transformer for some models.
        
        Args:
            input_ids: Input token IDs. Shape: (batch_size, n_codebooks)
            position_ids: Position IDs for the tokens. Shape: (batch_size)
            attn_wrapper: FlashInfer attention wrapper
            kv_cache: KV cache tensor
            **kwargs: Additional model-specific parameters
            
        Returns:
            Output logits tensor
        """
        assert self.has_depth_transformer, "This model does not support depth transformer."
        pass

    @abstractmethod
    def sampling(
        self, 
        logits: torch.Tensor, 
        requests: List[Request],
        sampling_params: SamplingConfig | None,
        cfg_scale: float | None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            logits: Output logits from the model. Shape: (batch_size, n_codebooks, vocab_size)
            requests: List of Request objects containing sampling configurations etc.
            sampling_params: Optional common sampling configurations
            cfg_scale: Optional common classifier-free guidance scale
            **kwargs: Additional model-specific parameters
            
        Returns:
            Output token IDs from sampling
        """
        pass
    
    @abstractmethod
    def postprocess(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert model output tokens to audio bytes. This should include model-specific logic 
        on when to do detokenization for each request.
        
        Args:
            token_ids: token IDs generated by the model. Shape: (batch_size, interval, n_codebooks)
            
        Returns:
            Tensor of audio data. Shape: (batch_size, audio_length)
        """
        pass
    