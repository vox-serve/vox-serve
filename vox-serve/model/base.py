from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
import torch

from ..flashinfer_utils import FlashInferWrapper


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
    def preprocess(self, prompt: str, voice: Optional[str] = None, **kwargs) -> Tuple[List[int], str]:
        """
        Preprocess the input prompt for the model.
        
        Args:
            prompt: Input text prompt
            voice: Optional voice identifier
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
        repetition_cache: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            position_ids: Position IDs for the tokens
            attn_wrapper: FlashInfer attention wrapper
            kv_cache: KV cache tensor
            
        Returns:
            Output token IDs from sampling
        """
        pass
    
    @abstractmethod
    def postprocess(self, token_sequence: List[int]) -> Optional[bytes]:
        """
        Convert model output tokens to audio bytes.
        
        Args:
            token_sequence: Sequence of output tokens (typically 28 tokens)
            
        Returns:
            Audio bytes if conversion successful, None otherwise
        """
        pass
    