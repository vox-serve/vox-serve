"""
Vision Diffusion Strategy for diffusion-based audio synthesis.

This is a placeholder for future support of diffusion-based models
like flow matching vocoders used in some TTS systems.
"""

from typing import Any, Dict, List, Optional

from ..model.base import BaseLM
from ..requests import Request
from .base import (
    AllocatedResources,
    GenerationStrategy,
    ResourceSpec,
    StrategyPhase,
    StrategyType,
)


class VisionDiffusionStrategy(GenerationStrategy):
    """
    Strategy for vision/audio diffusion models.

    This is a stateless strategy for diffusion-based synthesis, such as
    flow matching vocoders or image diffusion models. Unlike autoregressive
    models, diffusion models don't maintain KV cache state.

    Note: This is a placeholder implementation. Actual diffusion model
    support will require model-specific implementations.
    """

    def __init__(self, model: BaseLM, num_steps: int = 50, **kwargs):
        super().__init__(model, **kwargs)
        self._num_steps = num_steps
        self._phases = [
            StrategyPhase(
                name="diffusion",
                is_stateful=False,  # Diffusion is stateless per generation
                requires_cache=False,
                batch_size_limits=None,
            ),
        ]

    @property
    def name(self) -> str:
        return "vision_diffusion"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.VISION_DIFFUSION

    @property
    def phases(self) -> List[StrategyPhase]:
        return self._phases

    @property
    def num_steps(self) -> int:
        """Number of diffusion steps."""
        return self._num_steps

    def resource_spec(
        self,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
    ) -> ResourceSpec:
        """
        Diffusion models don't require caches.
        May benefit from CUDA graphs for fixed step counts.
        """
        return ResourceSpec(
            cache_specs=[],  # No cache needed
            requires_cuda_graph=True,  # Can use CUDA graphs for fixed steps
            requires_flashinfer=False,  # No paged attention
        )

    def prepare_inputs(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        phase: str,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for diffusion.

        Placeholder implementation.
        """
        if len(requests) == 0:
            return None

        return {
            "requests": requests,
            "num_steps": self._num_steps,
        }

    async def execute(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        phase: str,
        prepared_inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute diffusion sampling.

        Placeholder implementation - actual diffusion models will
        override this with model-specific logic.
        """
        raise NotImplementedError(
            "VisionDiffusionStrategy is a placeholder. "
            "Subclass with model-specific implementation."
        )

    def requires_cuda_graph_for_phase(self, phase: str) -> bool:
        """Diffusion can use CUDA graphs for fixed step counts."""
        return True
