from typing import Dict, Type, Any
import torch

from .base import BaseLM, BaseLMWithDepth
from .orpheus import OrpheusModel
from .csm import CSMModel
from .zonos import ZonosModel

# Registry mapping model name patterns to model classes
MODEL_REGISTRY: Dict[str, Type[BaseLM]] = {
    "orpheus": OrpheusModel,
    "canopylabs/orpheus-3b-0.1-ft": OrpheusModel,
    "csm": CSMModel,
    "sesame/csm-1b": CSMModel,
    "zonos": ZonosModel, 
    "Zyphra/Zonos-v0.1-transformer": ZonosModel,
}


def get_model_class(model_name: str) -> Type[BaseLM]:
    """
    Get the appropriate model class for a given model name.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        Model class that can handle this model
        
    Raises:
        ValueError: If no suitable model class is found
    """
    model_name_lower = model_name.lower()
    
    # Check for exact matches first
    if model_name_lower in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_lower]
    
    # Check for partial matches (e.g., "canopylabs/orpheus-3b-0.1-ft" matches "canopylabs/orpheus")
    for pattern, model_class in MODEL_REGISTRY.items():
        if pattern in model_name_lower:
            return model_class
    
    # Default fallback - if no match found, raise an error
    available_models = list(MODEL_REGISTRY.keys())
    raise ValueError(
        f"No model class found for '{model_name}'. "
        f"Available model patterns: {available_models}"
    )


def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    **kwargs: Any
) -> BaseLM | BaseLMWithDepth:
    """
    Load a model instance based on the model name.
    
    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on
        dtype: Data type for the model
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Loaded model instance
        
    Raises:
        ValueError: If no suitable model class is found
    """
    model_class = get_model_class(model_name)
    return model_class(model_name=model_name, device=device, dtype=dtype, **kwargs)


def register_model(pattern: str, model_class: Type[BaseLM]) -> None:
    """
    Register a new model class for a given pattern.
    
    Args:
        pattern: Pattern to match model names against
        model_class: Model class to use for matching names
    """
    MODEL_REGISTRY[pattern.lower()] = model_class


def list_supported_models() -> Dict[str, Type[BaseLM]]:
    """
    Get a dictionary of all supported model patterns and their classes.
    
    Returns:
        Dictionary mapping patterns to model classes
    """
    return MODEL_REGISTRY.copy()