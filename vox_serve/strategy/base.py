"""
Base classes for generation strategies.

Strategies define inference logic for different module types while remaining
agnostic about resource allocation. Pools allocate resources based on
strategy specifications and execute strategies asynchronously.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..model.base import BaseLM


class StrategyType(Enum):
    """Types of generation strategies corresponding to model modules."""

    ENCODER = "encoder"  # Stateless, no cache
    LLM = "llm"  # Stateful, KV cache
    DEPTH_TRANSFORMER = "depth"  # Stateful, separate KV cache
    VISION_DIFFUSION = "vision"  # Stateless
    AUDIO_CODEC = "audio_codec"  # Stateful, decoder cache


class CacheType(Enum):
    """Types of caches that strategies can request from pools."""

    KV_CACHE = "kv_cache"  # Standard KV cache for LLM
    DEPTH_KV_CACHE = "depth_kv"  # Separate KV cache for depth transformer
    DECODER_CACHE = "decoder"  # Audio decoder state cache


@dataclass
class StrategyPhase:
    """
    Defines a phase of strategy execution.

    Strategies can have multiple phases (e.g., prefill and decode for LLM).
    Each phase may have different resource requirements and batch constraints.
    """

    name: str  # "prefill", "decode", "step", etc.
    is_stateful: bool  # Whether this phase modifies cache state
    requires_cache: bool  # Whether this phase needs cache access
    batch_size_limits: Optional[Tuple[int, int]] = None  # (min, max) batch size


@dataclass
class CacheSpec:
    """Specification for a single cache type."""

    cache_type: CacheType
    shape: Tuple[int, ...]  # Shape of the cache tensor
    dtype: torch.dtype = torch.bfloat16
    device: Optional[str] = None  # None means use pool's default device


@dataclass
class ResourceSpec:
    """
    Specification of resources a strategy needs from a Pool.

    Strategies return this from resource_spec() to tell the Pool what
    resources to allocate. The Pool is agnostic about what these resources
    are used for.
    """

    cache_specs: List[CacheSpec] = field(default_factory=list)
    cuda_graph_batch_sizes: Optional[List[int]] = None
    requires_cuda_graph: bool = True
    requires_flashinfer: bool = True
    flashinfer_config: Optional[Dict[str, Any]] = None  # n_qo_head, n_kv_head, etc.
    extra_buffers: Optional[Dict[str, Tuple[Tuple[int, ...], torch.dtype]]] = None


@dataclass
class AllocatedResources:
    """
    Resources allocated by Pool for strategy execution.

    This is passed to strategy.execute() containing all the buffers,
    caches, and wrappers the strategy needs.
    """

    # Cache tensors keyed by CacheType
    caches: Dict[CacheType, torch.Tensor] = field(default_factory=dict)

    # FlashInfer attention wrappers
    prefill_wrapper: Optional[Any] = None  # FlashInferPrefillWrapper
    decode_wrapper: Optional[Any] = None  # FlashInferDecodeWrapper

    # CUDA graph related
    cuda_graph_buffers: Optional[Dict[str, torch.Tensor]] = None
    cuda_graphs: Optional[Dict[str, "torch.cuda.CUDAGraph"]] = None

    # Page management (for KV cache)
    page_size: int = 128
    max_num_pages: int = 2048

    # Extra buffers requested by strategy
    extra_buffers: Optional[Dict[str, torch.Tensor]] = None

    # Device info
    device: str = "cuda:0"


class GenerationStrategy(ABC):
    """
    Abstract base class for generation strategies.

    Strategies define the inference logic for a specific module type
    (e.g., LLM, AudioCodec) while remaining agnostic about resource
    allocation. They specify what resources they need, and the Pool
    allocates and manages them.

    Key responsibilities:
    - Define resource requirements via resource_spec()
    - Prepare inputs for execution via prepare_inputs()
    - Execute inference via execute()
    - Handle post-processing via post_process()
    """

    def __init__(self, model: BaseLM, **kwargs):
        """
        Initialize the strategy with a model.

        Args:
            model: The model this strategy wraps
            **kwargs: Additional strategy-specific configuration
        """
        self.model = model
        self._config = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this strategy instance."""
        pass

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Type of this strategy."""
        pass

    @property
    @abstractmethod
    def phases(self) -> List[StrategyPhase]:
        """List of phases this strategy supports."""
        pass

    @abstractmethod
    def resource_spec(
        self,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
    ) -> ResourceSpec:
        """
        Specification of resources this strategy needs.

        Called by Pool during strategy registration to allocate resources.

        Args:
            max_batch_size: Maximum batch size the pool will support
            max_num_pages: Maximum number of KV cache pages
            page_size: Size of each KV cache page

        Returns:
            ResourceSpec describing required resources
        """
        pass

    @abstractmethod
    def prepare_inputs(
        self,
        requests: List[Any],  # List[Request]
        resources: AllocatedResources,
        phase: str,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for execution.

        Transforms request data into the format needed for model execution.

        Args:
            requests: List of Request objects to process
            resources: Allocated resources from the pool
            phase: Current execution phase ("prefill", "decode", etc.)

        Returns:
            Dictionary of prepared inputs for execute()
        """
        pass

    @abstractmethod
    async def execute(
        self,
        requests: List[Any],  # List[Request]
        resources: AllocatedResources,
        phase: str,
        prepared_inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute the strategy phase.

        Args:
            requests: List of Request objects to process
            resources: Allocated resources from the pool
            phase: Current execution phase
            prepared_inputs: Optional pre-prepared inputs from prepare_inputs()

        Returns:
            Strategy-specific output (e.g., output tokens, audio)
        """
        pass

    def post_process(
        self,
        requests: List[Any],  # List[Request]
        outputs: Any,
        resources: AllocatedResources,
    ) -> None:
        """
        Handle outputs after execution.

        Updates request state based on execution outputs.
        Default implementation does nothing.

        Args:
            requests: List of Request objects that were processed
            outputs: Outputs from execute()
            resources: Allocated resources from the pool
        """
        pass

    def supports_phase(self, phase: str) -> bool:
        """Check if this strategy supports a given phase."""
        return any(p.name == phase for p in self.phases)

    def get_phase(self, phase_name: str) -> Optional[StrategyPhase]:
        """Get a phase by name."""
        for phase in self.phases:
            if phase.name == phase_name:
                return phase
        return None

    def requires_cuda_graph_for_phase(self, phase: str) -> bool:
        """
        Check if CUDA graph should be used for a given phase.

        Default implementation returns True for decode phases.
        Subclasses can override for more specific behavior.
        """
        return phase == "decode"
