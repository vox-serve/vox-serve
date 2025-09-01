from typing import Any, Dict, Type

import torch

from ..sampling import SamplingConfig
from .base import BaseLM, BaseLMWithDepth
from .csm import CSMModel
from .glm_voice import GLMVoiceModel
from .orpheus import OrpheusModel
from .zonos import ZonosModel

# Registry mapping model name patterns to model classes
MODEL_REGISTRY: Dict[str, Type[BaseLM]] = {
    "orpheus": OrpheusModel,
    "canopylabs/orpheus-3b-0.1-ft": OrpheusModel,
    "csm": CSMModel,
    "sesame/csm-1b": CSMModel,
    "zonos": ZonosModel,
    "Zyphra/Zonos-v0.1-transformer": ZonosModel,
    "glm": GLMVoiceModel,
    "zai-org/glm-4-voice-9b": GLMVoiceModel,
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
    raise ValueError(f"No model class found for '{model_name}'. Available model patterns: {available_models}")


def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    top_p: float = None,
    top_k: int = None,
    min_p: float = None,
    temperature: float = None,
    repetition_penalty: float = None,
    repetition_window: int = None,
    cfg_scale: float = None,
    greedy: bool = False,
    **kwargs: Any
) -> BaseLM | BaseLMWithDepth:
    """
    Load a model instance based on the model name.

    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on
        dtype: Data type for the model
        top_p: Top-p sampling parameter (overrides model default if provided)
        top_k: Top-k sampling parameter (overrides model default if provided)
        min_p: Min-p sampling parameter (overrides model default if provided)
        temperature: Temperature for sampling (overrides model default if provided)
        repetition_penalty: Repetition penalty (overrides model default if provided)
        repetition_window: Repetition window size (overrides model default if provided)
        cfg_scale: CFG scale for guidance (overrides model default if provided)
        greedy: Enable greedy sampling (ignores other sampling parameters if True)
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        Loaded model instance

    Raises:
        ValueError: If no suitable model class is found
    """
    model_class = get_model_class(model_name)
    model = model_class(model_name=model_name, device=device, dtype=dtype, **kwargs)

    # Override default sampling config if CLI parameters are provided
    if any(param is not None for param in [
        top_p, top_k, min_p, temperature, repetition_penalty, repetition_window, cfg_scale
    ]) or greedy:
        # Get current default config
        current_config = model.default_sampling_config

        # Create new config with overrides
        model.default_sampling_config = SamplingConfig(
            top_p=top_p if top_p is not None else current_config.top_p,
            top_k=top_k if top_k is not None else current_config.top_k,
            min_p=min_p if min_p is not None else current_config.min_p,
            temperature=temperature if temperature is not None else current_config.temperature,
            repetition_penalty=(
                repetition_penalty if repetition_penalty is not None
                else current_config.repetition_penalty
            ),
            repetition_window=(
                repetition_window if repetition_window is not None
                else current_config.repetition_window
            ),
            cfg_scale=cfg_scale if cfg_scale is not None else current_config.cfg_scale,
            greedy=greedy,
        )

    return model


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
