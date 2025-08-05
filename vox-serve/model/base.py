from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict
import torch

from ..flashinfer_utils import FlashInferWrapper
from ..sampling import SamplingConfig


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
            input_ids: Input token IDs
            position_ids: Position IDs for the tokens
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
            input_ids: Input token IDs. Shape: (batch_size, n_codebook)
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
        sampling_params: List[SamplingConfig] | None,
        repetition_cache: torch.Tensor | None, 
        cfg_scale: float | None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            logits: Output logits from the model. Shape: (batch_size, vocab_size, hidden_size)
            sampling_params: Optional list of sampling configurations
            repetition_cache: Optional repetition cache tensor
            cfg_scale: Optional classifier-free guidance scale
            **kwargs: Additional model-specific parameters
            
        Returns:
            Output token IDs from sampling
        """
        pass
    
    @abstractmethod
    def postprocess(self, token_ids: List[List[int]]) -> Optional[bytes]:
        """
        Convert model output tokens to audio bytes.
        
        Args:
            token_ids: 2D list of token IDs
            
        Returns:
            Audio bytes if conversion successful, None otherwise
        """
        pass
    