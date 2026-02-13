"""
Generation strategies for different model modules.

Strategies define inference logic while remaining agnostic about resource
allocation. Pools allocate resources based on strategy specifications.
"""

from .audio_codec import AudioCodecStrategy
from .base import (
    AllocatedResources,
    CacheSpec,
    CacheType,
    GenerationStrategy,
    ResourceSpec,
    StrategyPhase,
    StrategyType,
)
from .depth_transformer import DepthTransformerStrategy
from .encoder import EncoderStrategy
from .llm import LLMStrategy
from .vision_diffusion import VisionDiffusionStrategy

__all__ = [
    # Base classes and types
    "GenerationStrategy",
    "StrategyType",
    "StrategyPhase",
    "CacheType",
    "CacheSpec",
    "ResourceSpec",
    "AllocatedResources",
    # Concrete strategies
    "EncoderStrategy",
    "LLMStrategy",
    "DepthTransformerStrategy",
    "VisionDiffusionStrategy",
    "AudioCodecStrategy",
]
