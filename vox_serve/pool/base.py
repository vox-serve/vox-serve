"""
Pool base class for hardware resource management.

Pools manage compute resources (GPU memory, CUDA graphs, etc.) and execute
strategies asynchronously. They are agnostic about inference logic - they
just allocate buffers and run strategies based on resource specifications.
"""

import queue
from typing import Any, Dict, List, Optional

import torch

from ..flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper
from ..model import load_model
from ..model.base import BaseLM
from ..requests import Request
from ..strategy.base import (
    AllocatedResources,
    CacheSpec,
    GenerationStrategy,
    ResourceSpec,
)
from ..utils import get_logger


class Pool:
    """
    Hardware resource manager for strategy execution.

    Pools are responsible for:
    - Allocating caches and buffers based on strategy specifications
    - Managing KV cache pages
    - Creating FlashInfer attention wrappers
    - Executing strategies with allocated resources

    Pools are agnostic about what the strategies do - they just provide
    resources and run the strategies.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        max_batch_size: int = 8,
        max_num_pages: int = 2048,
        page_size: int = 128,
        # Sampling parameters (passed to model)
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        repetition_penalty: float = None,
        repetition_window: int = None,
        cfg_scale: float = None,
        greedy: bool = False,
        # Optimization flags
        enable_nvtx: bool = False,
        enable_torch_compile: bool = False,
        # Multi-device support
        secondary_device: Optional[str] = None,
        # Data parallel info
        dp_rank: int = 0,
        dp_size: int = 1,
    ):
        """
        Initialize the pool.

        Args:
            model_name: Name or path of the model to load
            device: Primary device for this pool
            max_batch_size: Maximum batch size for inference
            max_num_pages: Maximum number of KV cache pages
            page_size: Size of each KV cache page
            secondary_device: Optional secondary device (for disaggregation)
            dp_rank: Data parallel rank
            dp_size: Data parallel world size
        """
        self.device = device
        self.secondary_device = secondary_device or device
        self.max_batch_size = max_batch_size
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.nvtx_enabled = enable_nvtx

        # Set up logging
        base_logger = get_logger(__name__)
        if dp_size > 1:
            import logging

            self.logger = logging.LoggerAdapter(base_logger, {"dp_rank": dp_rank})
            self.logger.process = lambda msg, kwargs: (f"[DP {dp_rank}/{dp_size}] {msg}", kwargs)
        else:
            self.logger = base_logger

        # Load model
        self.model: BaseLM = load_model(
            model_name,
            device=device,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            cfg_scale=cfg_scale,
            greedy=greedy,
            enable_torch_compile=enable_torch_compile,
            audio_decoder_device=secondary_device,
        )

        # Initialize page management
        self.empty_pages: queue.Queue = queue.Queue()
        for i in range(max_num_pages):
            self.empty_pages.put(i)

        # Strategy registry
        self._strategies: Dict[str, GenerationStrategy] = {}
        self._strategy_resources: Dict[str, AllocatedResources] = {}

        # Initialize shared resources
        self._init_shared_resources()

    def _init_shared_resources(self):
        """Initialize resources shared across strategies."""
        # FlashInfer buffer (shared)
        self.flashinfer_buffer = torch.empty(
            256 * 1024 * 1024,
            dtype=torch.uint8,
            device=self.device,
        )

    @property
    def strategies(self) -> Dict[str, GenerationStrategy]:
        """Registered strategies."""
        return self._strategies

    @property
    def available_batch_sizes(self) -> Optional[List[int]]:
        """
        Available batch sizes for this pool.
        Base pool has no restriction.
        """
        return None

    def register_strategy(self, strategy: GenerationStrategy) -> AllocatedResources:
        """
        Register a strategy and allocate its required resources.

        Args:
            strategy: The strategy to register

        Returns:
            AllocatedResources containing all allocated resources
        """
        if strategy.name in self._strategies:
            self.logger.warning(f"Strategy '{strategy.name}' already registered, replacing")

        # Get resource specification
        spec = strategy.resource_spec(
            max_batch_size=self.max_batch_size,
            max_num_pages=self.max_num_pages,
            page_size=self.page_size,
        )

        # Allocate resources
        resources = self._allocate_resources(spec, strategy)

        # Store strategy and resources
        self._strategies[strategy.name] = strategy
        self._strategy_resources[strategy.name] = resources

        self.logger.info(f"Registered strategy '{strategy.name}' ({strategy.strategy_type.value})")

        return resources

    def _allocate_resources(
        self,
        spec: ResourceSpec,
        strategy: GenerationStrategy,
    ) -> AllocatedResources:
        """
        Allocate resources based on specification.

        Args:
            spec: Resource specification from strategy
            strategy: The strategy requesting resources

        Returns:
            AllocatedResources with allocated caches and wrappers
        """
        resources = AllocatedResources(
            page_size=self.page_size,
            max_num_pages=self.max_num_pages,
            device=self.device,
        )

        # Allocate caches
        for cache_spec in spec.cache_specs:
            cache = self._allocate_cache(cache_spec)
            resources.caches[cache_spec.cache_type] = cache

        # Create FlashInfer wrappers if needed
        if spec.requires_flashinfer and spec.flashinfer_config:
            resources.prefill_wrapper = FlashInferPrefillWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=spec.flashinfer_config["n_qo_head"],
                n_kv_head=spec.flashinfer_config["n_kv_head"],
                n_state=spec.flashinfer_config["n_state"],
                page_size=spec.flashinfer_config["page_size"],
                use_cuda_graph=False,  # Base pool doesn't use CUDA graphs
            )
            resources.decode_wrapper = FlashInferDecodeWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=spec.flashinfer_config["n_qo_head"],
                n_kv_head=spec.flashinfer_config["n_kv_head"],
                n_state=spec.flashinfer_config["n_state"],
                page_size=spec.flashinfer_config["page_size"],
                use_cuda_graph=False,
            )

        # Allocate extra buffers if specified
        if spec.extra_buffers:
            resources.extra_buffers = {}
            for name, (shape, dtype) in spec.extra_buffers.items():
                resources.extra_buffers[name] = torch.zeros(
                    shape,
                    dtype=dtype,
                    device=self.device,
                )

        return resources

    def _allocate_cache(self, cache_spec: CacheSpec) -> Optional[torch.Tensor]:
        """
        Allocate a cache tensor based on specification.

        Args:
            cache_spec: Cache specification

        Returns:
            Allocated cache tensor, or None if shape is None
        """
        if cache_spec.shape is None:
            # Dynamic cache - will be managed per-request
            return None

        device = cache_spec.device or self.device
        cache = torch.zeros(
            cache_spec.shape,
            dtype=cache_spec.dtype,
            device=device,
        )

        cache_size = cache.numel() * cache.element_size()
        self.logger.info(
            f"Allocated {cache_spec.cache_type.value} cache: {cache_size / 1024 / 1024:.2f} MB"
        )

        return cache

    def get_resources(self, strategy_name: str) -> Optional[AllocatedResources]:
        """Get allocated resources for a strategy."""
        return self._strategy_resources.get(strategy_name)

    async def execute(
        self,
        strategy: GenerationStrategy,
        requests: List[Request],
        phase: str,
        prepared_inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a strategy phase with allocated resources.

        Args:
            strategy: Strategy to execute
            requests: Requests to process
            phase: Execution phase
            prepared_inputs: Optional pre-prepared inputs

        Returns:
            Strategy output
        """
        resources = self._strategy_resources.get(strategy.name)
        if resources is None:
            raise ValueError(f"Strategy '{strategy.name}' not registered")

        return await strategy.execute(requests, resources, phase, prepared_inputs)

    def allocate_pages(self, request: Request, num_pages: int) -> List[int]:
        """
        Allocate KV cache pages for a request.

        Args:
            request: Request to allocate pages for
            num_pages: Number of pages to allocate

        Returns:
            List of allocated page indices
        """
        pages = []
        for _ in range(num_pages):
            try:
                page = self.empty_pages.get_nowait()
                pages.append(page)
            except queue.Empty as e:
                # Return already allocated pages
                for p in pages:
                    self.empty_pages.put(p)
                raise RuntimeError("No available KV cache pages") from e

        request.kv_pages = pages
        return pages

    def allocate_additional_page(self, request: Request) -> int:
        """
        Allocate one additional page for a request.

        Args:
            request: Request to allocate page for

        Returns:
            Allocated page index
        """
        try:
            page = self.empty_pages.get_nowait()
            request.kv_pages.append(page)
            return page
        except queue.Empty as e:
            raise RuntimeError("No available KV cache pages") from e

    def free_pages(self, request: Request):
        """
        Free KV cache pages used by a request.

        Args:
            request: Request whose pages to free
        """
        if hasattr(request, "kv_pages") and request.kv_pages:
            for page_idx in request.kv_pages:
                self.empty_pages.put(page_idx)
            request.kv_pages = []
            request.kv_token_len = 0
            request.kv_last_page_len = 0

    def nvtx_range_push(self, name: str):
        """Push an NVTX range if profiling is enabled."""
        if self.nvtx_enabled:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(name)

    def nvtx_range_pop(self):
        """Pop an NVTX range if profiling is enabled."""
        if self.nvtx_enabled:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

    # Properties from model (for backward compatibility)
    @property
    def detokenize_interval(self) -> int:
        return self.model.detokenize_interval

    @property
    def detokenize_overlap(self) -> int:
        return self.model.detokenize_overlap

    @property
    def supports_audio_input(self) -> bool:
        return self.model.supports_audio_input
