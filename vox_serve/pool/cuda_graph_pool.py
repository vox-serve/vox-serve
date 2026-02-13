"""
CUDA Graph Pool for optimized inference.

Extends the base Pool with CUDA graph capture and replay for reduced
kernel launch overhead during inference.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper
from ..strategy.base import (
    AllocatedResources,
    CacheType,
    GenerationStrategy,
    ResourceSpec,
)
from .base import Pool


class CudaGraphPool(Pool):
    """
    Pool with CUDA graph optimization for improved inference performance.

    CUDA graphs capture and replay computation graphs to eliminate Python
    overhead during inference, providing significant speedup.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # CUDA graph storage
        self.cuda_graphs_lm_decode: Dict[int, torch.cuda.CUDAGraph] = {}
        self.cuda_graphs_lm_prefill: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.cuda_graphs_detokenization: Dict[int, torch.cuda.CUDAGraph] = {}
        self.cuda_graphs_depth_prefill: Dict[int, torch.cuda.CUDAGraph] = {}
        self.cuda_graphs_depth_decode: Dict[int, torch.cuda.CUDAGraph] = {}
        self.cuda_graph_buffers: Dict[str, torch.Tensor] = {}

        # Batch size configuration
        self.cuda_graph_batch_sizes = [2**i for i in range(int(np.log2(self.max_batch_size)) + 1)][::-1]
        self.cuda_graph_seq_len_buckets = [1024][::-1]
        self.prefill_graph_batch_size = 8

        # CUDA graph memory pool
        self.cuda_graph_pool = torch.cuda.graph_pool_handle()

        # Separate pool for detokenizer if on different device
        if self.secondary_device != self.device:
            with torch.cuda.device(self.secondary_device):
                self.detokenizer_cuda_graph_pool = torch.cuda.graph_pool_handle()
        else:
            self.detokenizer_cuda_graph_pool = None

        # Index buffers for FlashInfer
        self._init_index_buffers()

    @property
    def available_batch_sizes(self) -> List[int]:
        """Return supported batch sizes for CUDA graphs."""
        return self.cuda_graph_batch_sizes

    def _init_shared_resources(self):
        """Initialize shared resources including FlashInfer buffer."""
        super()._init_shared_resources()

    def _init_index_buffers(self):
        """Initialize index buffers for FlashInfer attention planning."""
        self.qo_indptr_buffer = torch.zeros(
            self.max_batch_size + 1, dtype=torch.int32, device=self.device
        )
        self.paged_kv_indptr_buffer = torch.zeros(
            self.max_batch_size + 1, dtype=torch.int32, device=self.device
        )
        self.paged_kv_indices_buffer = torch.zeros(
            self.max_num_pages, dtype=torch.int32, device=self.device
        )
        self.paged_kv_last_page_len_buffer = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

    def _allocate_resources(
        self,
        spec: ResourceSpec,
        strategy: GenerationStrategy,
    ) -> AllocatedResources:
        """
        Allocate resources with CUDA graph support.

        Creates wrappers configured for CUDA graph usage.
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

        # Create FlashInfer wrappers with CUDA graph support
        if spec.requires_flashinfer and spec.flashinfer_config:
            # Non-CUDA graph wrapper for fallback
            resources.prefill_wrapper = FlashInferPrefillWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=spec.flashinfer_config["n_qo_head"],
                n_kv_head=spec.flashinfer_config["n_kv_head"],
                n_state=spec.flashinfer_config["n_state"],
                page_size=spec.flashinfer_config["page_size"],
                use_cuda_graph=False,
            )

            # Create CUDA graph-enabled wrappers for each batch size
            resources.decode_wrapper = self._create_decode_wrappers(spec.flashinfer_config)

            # Store prefill wrappers dict on resources for CUDA graph use
            resources.extra_buffers = resources.extra_buffers or {}
            resources.extra_buffers["prefill_wrappers"] = self._create_prefill_wrappers(
                spec.flashinfer_config
            )
            resources.extra_buffers["decode_wrappers"] = resources.decode_wrapper

        # Initialize CUDA graph buffers based on strategy type
        resources.cuda_graph_buffers = self._init_cuda_graph_buffers(strategy)
        resources.cuda_graphs = {}

        return resources

    def _create_decode_wrappers(
        self,
        flashinfer_config: Dict[str, Any],
    ) -> Dict[int, FlashInferDecodeWrapper]:
        """Create decode wrappers for each batch size."""
        wrappers = {}
        for batch_size in self.cuda_graph_batch_sizes:
            wrappers[batch_size] = FlashInferDecodeWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=flashinfer_config["n_qo_head"],
                n_kv_head=flashinfer_config["n_kv_head"],
                n_state=flashinfer_config["n_state"],
                page_size=flashinfer_config["page_size"],
                batch_size=batch_size,
                paged_kv_indptr_buffer=self.paged_kv_indptr_buffer[: batch_size + 1],
                paged_kv_indices_buffer=self.paged_kv_indices_buffer,
                paged_kv_last_page_len_buffer=self.paged_kv_last_page_len_buffer[:batch_size],
                use_cuda_graph=True,
            )
        return wrappers

    def _create_prefill_wrappers(
        self,
        flashinfer_config: Dict[str, Any],
    ) -> Dict[Tuple[int, int], FlashInferPrefillWrapper]:
        """Create prefill wrappers for CUDA graph capture."""
        wrappers = {}
        for seq_len in self.cuda_graph_seq_len_buckets:
            batch_size = self.prefill_graph_batch_size
            key = (batch_size, seq_len)
            wrappers[key] = FlashInferPrefillWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=flashinfer_config["n_qo_head"],
                n_kv_head=flashinfer_config["n_kv_head"],
                n_state=flashinfer_config["n_state"],
                page_size=flashinfer_config["page_size"],
                batch_size=batch_size,
                max_seq_len=seq_len,
                qo_indptr_buffer=self.qo_indptr_buffer[: batch_size + 1],
                paged_kv_indptr_buffer=self.paged_kv_indptr_buffer[: batch_size + 1],
                paged_kv_indices_buffer=self.paged_kv_indices_buffer,
                paged_kv_last_page_len_buffer=self.paged_kv_last_page_len_buffer[:batch_size],
                use_cuda_graph=True,
            )
        return wrappers

    def _init_cuda_graph_buffers(
        self,
        strategy: GenerationStrategy,
    ) -> Dict[str, torch.Tensor]:
        """Initialize CUDA graph input/output buffers for a strategy."""
        buffers = {}

        # Common buffers based on model properties
        n_codebooks = self.model.n_codebooks
        hidden_size = self.model.hidden_size
        vocab_size = self.model.vocab_size
        max_seq_len = max(self.cuda_graph_seq_len_buckets)

        # Decode phase buffers
        buffers["input_ids"] = torch.zeros(
            self.max_batch_size, n_codebooks, dtype=torch.int32, device=self.device
        )
        buffers["position_ids"] = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )
        buffers["input_features"] = torch.zeros(
            self.max_batch_size, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        buffers["input_masks"] = torch.zeros(
            self.max_batch_size, n_codebooks, dtype=torch.bool, device=self.device
        )

        has_depth = self.model.has_depth_transformer
        logits_n_codebooks = 1 if has_depth else n_codebooks
        buffers["logits"] = torch.zeros(
            self.max_batch_size, logits_n_codebooks, vocab_size,
            dtype=torch.bfloat16, device=self.device
        )
        buffers["backbone_hidden_states"] = torch.zeros(
            self.max_batch_size, hidden_size, dtype=torch.bfloat16, device=self.device
        )

        # Prefill phase buffers
        buffers["prefill_input_ids"] = torch.zeros(
            max_seq_len, n_codebooks, dtype=torch.int32, device=self.device
        )
        buffers["prefill_position_ids"] = torch.zeros(
            max_seq_len, dtype=torch.int32, device=self.device
        )
        buffers["prefill_input_features"] = torch.zeros(
            max_seq_len, hidden_size, dtype=torch.bfloat16, device=self.device
        )
        buffers["prefill_input_masks"] = torch.zeros(
            max_seq_len, n_codebooks, dtype=torch.bool, device=self.device
        )
        buffers["prefill_logits"] = torch.zeros(
            max_seq_len, logits_n_codebooks, vocab_size,
            dtype=torch.bfloat16, device=self.device
        )
        buffers["prefill_backbone_hidden_states"] = torch.zeros(
            max_seq_len, hidden_size, dtype=torch.bfloat16, device=self.device
        )

        # Detokenization buffers (on secondary device)
        with torch.cuda.device(self.secondary_device):
            buffers["detokenize_input"] = torch.zeros(
                self.max_batch_size,
                self.model.detokenize_interval,
                n_codebooks,
                dtype=torch.int32,
                device=self.secondary_device,
            )
            buffers["detokenize_output"] = torch.zeros(
                self.max_batch_size,
                self.model.n_channels,
                self.model.output_audio_length,
                dtype=torch.float32,
                device=self.secondary_device,
            )
            buffers["detokenize_cache"] = self.model.audio_decoder_initial_cache(
                self.max_batch_size
            )

        # Depth transformer buffers if needed
        if has_depth:
            buffers["depth_hidden_states"] = torch.zeros(
                2 * self.max_batch_size, hidden_size,
                dtype=torch.bfloat16, device=self.device
            )
            buffers["depth_position_ids"] = torch.zeros(
                2 * self.max_batch_size, dtype=torch.int32, device=self.device
            )
            buffers["depth_logits"] = torch.zeros(
                2 * self.max_batch_size, self.model.depth_vocab_size,
                dtype=torch.bfloat16, device=self.device
            )

        return buffers

    def initialize_cuda_graphs(self, strategy: GenerationStrategy):
        """
        Initialize CUDA graphs for a registered strategy.

        This should be called after register_strategy() to capture
        the CUDA graphs for optimized inference.
        """
        resources = self._strategy_resources.get(strategy.name)
        if resources is None:
            raise ValueError(f"Strategy '{strategy.name}' not registered")

        self.logger.info(f"Initializing CUDA graphs for strategy '{strategy.name}'...")

        original_nvtx = self.nvtx_enabled
        self.nvtx_enabled = False  # Disable during capture

        try:
            self._initialize_decode_cuda_graphs(strategy, resources)
            self._initialize_prefill_cuda_graphs(strategy, resources)
            self._initialize_detokenization_cuda_graphs(strategy, resources)

            if self.model.has_depth_transformer:
                self._initialize_depth_cuda_graphs(strategy, resources)

            self.logger.info(
                f"CUDA graphs initialized for batch sizes: {list(self.cuda_graphs_lm_decode.keys())}"
            )
        finally:
            self.nvtx_enabled = original_nvtx

        # Zero caches after initialization
        kv_cache = resources.caches.get(CacheType.KV_CACHE)
        if kv_cache is not None:
            kv_cache.zero_()

        depth_kv_cache = resources.caches.get(CacheType.DEPTH_KV_CACHE)
        if depth_kv_cache is not None:
            depth_kv_cache.zero_()

    def _initialize_decode_cuda_graphs(
        self,
        strategy: GenerationStrategy,
        resources: AllocatedResources,
    ):
        """Initialize CUDA graphs for decode phase."""
        self.logger.info("Initializing decode CUDA graphs...")
        kv_cache = resources.caches.get(CacheType.KV_CACHE)
        decode_wrappers = resources.extra_buffers.get("decode_wrappers", {})
        buffers = resources.cuda_graph_buffers

        for batch_size in self.cuda_graph_batch_sizes:
            if batch_size > self.max_batch_size:
                continue

            self.logger.info(f"Capturing decode CUDA graph for batch size {batch_size}")

            # Plan wrapper
            page_per_request = self.max_num_pages // self.max_batch_size
            paged_kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * page_per_request
            paged_kv_indices = torch.arange(batch_size * page_per_request, dtype=torch.int32)
            paged_kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32)

            decode_wrappers[batch_size].plan(
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                torch.bfloat16,
            )
            torch.cuda.synchronize()

            # Warmup
            for _ in range(5):
                self.model.forward(
                    input_ids=buffers["input_ids"][:batch_size],
                    position_ids=buffers["position_ids"][:batch_size],
                    attn_wrapper=decode_wrappers[batch_size],
                    kv_cache=kv_cache,
                    input_features=buffers["input_features"][:batch_size],
                    input_masks=buffers["input_masks"][:batch_size],
                )
            torch.cuda.synchronize()

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.cuda_graph_pool):
                if self.model.has_depth_transformer:
                    logits, hidden = self.model.forward(
                        input_ids=buffers["input_ids"][:batch_size],
                        position_ids=buffers["position_ids"][:batch_size],
                        attn_wrapper=decode_wrappers[batch_size],
                        kv_cache=kv_cache,
                        input_features=buffers["input_features"][:batch_size],
                        input_masks=buffers["input_masks"][:batch_size],
                    )
                    buffers["logits"][:batch_size].copy_(logits)
                    buffers["backbone_hidden_states"][:batch_size].copy_(hidden)
                else:
                    logits = self.model.forward(
                        input_ids=buffers["input_ids"][:batch_size],
                        position_ids=buffers["position_ids"][:batch_size],
                        attn_wrapper=decode_wrappers[batch_size],
                        kv_cache=kv_cache,
                        input_features=buffers["input_features"][:batch_size],
                        input_masks=buffers["input_masks"][:batch_size],
                    )
                    buffers["logits"][:batch_size].copy_(logits)

            self.cuda_graphs_lm_decode[batch_size] = graph
            resources.cuda_graphs[f"decode_{batch_size}"] = graph

            # Benchmark
            self._benchmark_cuda_graph(graph, f"decode (batch={batch_size})")

    def _initialize_prefill_cuda_graphs(
        self,
        strategy: GenerationStrategy,
        resources: AllocatedResources,
    ):
        """Initialize CUDA graphs for prefill phase."""
        self.logger.info("Initializing prefill CUDA graphs...")
        kv_cache = resources.caches.get(CacheType.KV_CACHE)
        prefill_wrappers = resources.extra_buffers.get("prefill_wrappers", {})
        buffers = resources.cuda_graph_buffers

        for seq_len in self.cuda_graph_seq_len_buckets:
            batch_size = self.prefill_graph_batch_size
            key = (batch_size, seq_len)

            self.logger.info(f"Capturing prefill CUDA graph for batch_size={batch_size}, seq_len={seq_len}")

            seq_len_per_batch = seq_len // batch_size
            qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * seq_len_per_batch
            pages_per_batch = max(1, seq_len_per_batch // self.page_size)
            paged_kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * pages_per_batch
            paged_kv_indices = torch.arange(self.max_num_pages, dtype=torch.int32)
            paged_kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32)

            prefill_wrappers[key].plan(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                torch.bfloat16,
            )
            torch.cuda.synchronize()

            # Warmup
            for _ in range(5):
                self.model.forward(
                    input_ids=buffers["prefill_input_ids"][:seq_len],
                    position_ids=buffers["prefill_position_ids"][:seq_len],
                    attn_wrapper=prefill_wrappers[key],
                    kv_cache=kv_cache,
                    input_features=buffers["prefill_input_features"][:seq_len],
                    input_masks=buffers["prefill_input_masks"][:seq_len],
                )
            torch.cuda.synchronize()

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.cuda_graph_pool):
                if self.model.has_depth_transformer:
                    logits, hidden = self.model.forward(
                        input_ids=buffers["prefill_input_ids"][:seq_len],
                        position_ids=buffers["prefill_position_ids"][:seq_len],
                        attn_wrapper=prefill_wrappers[key],
                        kv_cache=kv_cache,
                        input_features=buffers["prefill_input_features"][:seq_len],
                        input_masks=buffers["prefill_input_masks"][:seq_len],
                    )
                    buffers["prefill_logits"][:seq_len].copy_(logits)
                    buffers["prefill_backbone_hidden_states"][:seq_len].copy_(hidden)
                else:
                    logits = self.model.forward(
                        input_ids=buffers["prefill_input_ids"][:seq_len],
                        position_ids=buffers["prefill_position_ids"][:seq_len],
                        attn_wrapper=prefill_wrappers[key],
                        kv_cache=kv_cache,
                        input_features=buffers["prefill_input_features"][:seq_len],
                        input_masks=buffers["prefill_input_masks"][:seq_len],
                    )
                    buffers["prefill_logits"][:seq_len].copy_(logits)

            self.cuda_graphs_lm_prefill[key] = graph
            resources.cuda_graphs[f"prefill_{batch_size}_{seq_len}"] = graph

            self._benchmark_cuda_graph(graph, f"prefill (batch={batch_size}, seq_len={seq_len})")

    def _initialize_detokenization_cuda_graphs(
        self,
        strategy: GenerationStrategy,
        resources: AllocatedResources,
    ):
        """Initialize CUDA graphs for detokenization phase."""
        self.logger.info("Initializing detokenization CUDA graphs...")
        buffers = resources.cuda_graph_buffers

        for batch_size in self.cuda_graph_batch_sizes:
            if batch_size > self.max_batch_size:
                continue

            self.logger.info(f"Capturing detokenization CUDA graph for batch size {batch_size}")

            with torch.cuda.device(self.secondary_device):
                s = torch.cuda.Stream(device=self.secondary_device)
                s.wait_stream(torch.cuda.current_stream(self.secondary_device))

                # Warmup
                with torch.cuda.stream(s):
                    for _ in range(5):
                        if buffers["detokenize_cache"] is not None:
                            self.model.postprocess(
                                buffers["detokenize_input"][:batch_size],
                                decoder_cache=buffers["detokenize_cache"][:batch_size],
                            )
                        else:
                            self.model.postprocess(buffers["detokenize_input"][:batch_size])

                torch.cuda.current_stream(self.secondary_device).wait_stream(s)
                torch.cuda.synchronize(self.secondary_device)

                # Capture graph
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    if buffers["detokenize_cache"] is not None:
                        audio = self.model.postprocess(
                            buffers["detokenize_input"][:batch_size],
                            decoder_cache=buffers["detokenize_cache"][:batch_size],
                        )
                    else:
                        audio = self.model.postprocess(buffers["detokenize_input"][:batch_size])
                    buffers["detokenize_output"][:batch_size].copy_(audio)

                self.cuda_graphs_detokenization[batch_size] = graph
                resources.cuda_graphs[f"detokenize_{batch_size}"] = graph

                self._benchmark_cuda_graph(
                    graph, f"detokenize (batch={batch_size})", device=self.secondary_device
                )

    def _initialize_depth_cuda_graphs(
        self,
        strategy: GenerationStrategy,
        resources: AllocatedResources,
    ):
        """Initialize CUDA graphs for depth transformer."""
        self.logger.info("Initializing depth transformer CUDA graphs...")
        depth_kv_cache = resources.caches.get(CacheType.DEPTH_KV_CACHE)
        buffers = resources.cuda_graph_buffers

        # Create depth transformer wrappers
        depth_prefill_wrappers = {}
        depth_decode_wrappers = {}
        depth_state_size = self.model.depth_num_attention_heads * self.model.depth_head_dim

        # Index buffers for depth
        depth_qo_indptr_buffer = torch.zeros(
            self.max_batch_size + 1, dtype=torch.int32, device=self.device
        )
        depth_kv_indptr_buffer = torch.zeros(
            self.max_batch_size + 1, dtype=torch.int32, device=self.device
        )
        depth_kv_indices_buffer = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )
        depth_kv_last_page_len_buffer = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

        for batch_size in self.cuda_graph_batch_sizes:
            depth_prefill_wrappers[batch_size] = FlashInferPrefillWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=self.model.depth_num_attention_heads,
                n_kv_head=self.model.depth_num_key_value_heads,
                n_state=depth_state_size,
                page_size=self.model.depth_n_codebooks,
                batch_size=batch_size,
                max_seq_len=2 * batch_size,
                qo_indptr_buffer=depth_qo_indptr_buffer[: batch_size + 1],
                paged_kv_indptr_buffer=depth_kv_indptr_buffer[: batch_size + 1],
                paged_kv_indices_buffer=depth_kv_indices_buffer[:batch_size],
                paged_kv_last_page_len_buffer=depth_kv_last_page_len_buffer[:batch_size],
                use_cuda_graph=True,
            )

            depth_decode_wrappers[batch_size] = FlashInferDecodeWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=self.model.depth_num_attention_heads,
                n_kv_head=self.model.depth_num_key_value_heads,
                n_state=depth_state_size,
                page_size=self.model.depth_n_codebooks,
                batch_size=batch_size,
                paged_kv_indptr_buffer=depth_kv_indptr_buffer[: batch_size + 1],
                paged_kv_indices_buffer=depth_kv_indices_buffer[:batch_size],
                paged_kv_last_page_len_buffer=depth_kv_last_page_len_buffer[:batch_size],
                use_cuda_graph=True,
            )

        resources.extra_buffers["depth_prefill_wrappers"] = depth_prefill_wrappers
        resources.extra_buffers["depth_decode_wrappers"] = depth_decode_wrappers

        for batch_size in self.cuda_graph_batch_sizes:
            if batch_size > self.max_batch_size:
                continue

            self.logger.info(f"Capturing depth CUDA graph for batch size {batch_size}")

            # Setup indices
            depth_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * 2
            depth_kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32)
            depth_kv_indices = torch.arange(batch_size, dtype=torch.int32)
            depth_kv_last_page_len = torch.ones(batch_size, dtype=torch.int32) * 2

            # Prefill graph
            depth_prefill_wrappers[batch_size].plan(
                depth_qo_indptr,
                depth_kv_indptr,
                depth_kv_indices,
                depth_kv_last_page_len,
                torch.bfloat16,
            )
            torch.cuda.synchronize()

            for _ in range(5):
                self.model.depth_forward(
                    hidden_states=buffers["depth_hidden_states"][: 2 * batch_size],
                    position_ids=buffers["depth_position_ids"][: 2 * batch_size],
                    attn_wrapper=depth_prefill_wrappers[batch_size],
                    kv_cache=depth_kv_cache,
                )
            torch.cuda.synchronize()

            prefill_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(prefill_graph, pool=self.cuda_graph_pool):
                logits = self.model.depth_forward(
                    hidden_states=buffers["depth_hidden_states"][: 2 * batch_size],
                    position_ids=buffers["depth_position_ids"][: 2 * batch_size],
                    attn_wrapper=depth_prefill_wrappers[batch_size],
                    kv_cache=depth_kv_cache,
                )
                buffers["depth_logits"][: 2 * batch_size].copy_(logits)

            self.cuda_graphs_depth_prefill[batch_size] = prefill_graph
            resources.cuda_graphs[f"depth_prefill_{batch_size}"] = prefill_graph

            # Decode graph
            depth_decode_wrappers[batch_size].plan(
                depth_kv_indptr,
                depth_kv_indices,
                depth_kv_last_page_len,
                torch.bfloat16,
            )
            torch.cuda.synchronize()

            for _ in range(5):
                self.model.depth_forward(
                    hidden_states=buffers["depth_hidden_states"][:batch_size],
                    position_ids=buffers["depth_position_ids"][:batch_size],
                    attn_wrapper=depth_decode_wrappers[batch_size],
                    kv_cache=depth_kv_cache,
                )
            torch.cuda.synchronize()

            decode_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(decode_graph, pool=self.cuda_graph_pool):
                logits = self.model.depth_forward(
                    hidden_states=buffers["depth_hidden_states"][:batch_size],
                    position_ids=buffers["depth_position_ids"][:batch_size],
                    attn_wrapper=depth_decode_wrappers[batch_size],
                    kv_cache=depth_kv_cache,
                )
                buffers["depth_logits"][:batch_size].copy_(logits)

            self.cuda_graphs_depth_decode[batch_size] = decode_graph
            resources.cuda_graphs[f"depth_decode_{batch_size}"] = decode_graph

    def _benchmark_cuda_graph(
        self,
        graph: torch.cuda.CUDAGraph,
        name: str,
        device: Optional[str] = None,
    ):
        """Benchmark a CUDA graph replay latency."""
        device = device or self.device

        # Warmup
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)

        # Measure
        times = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))

        self.logger.debug(f"CUDA graph {name} avg replay: {sum(times)/len(times):.3f}ms")

    def get_cuda_graph_batch_size(self, actual_batch_size: int) -> int:
        """Find the next valid CUDA graph batch size for padding."""
        for batch_size in sorted(self.cuda_graph_batch_sizes):
            if batch_size >= actual_batch_size:
                return batch_size
        return max(self.cuda_graph_batch_sizes)

    def get_cuda_graph_seq_len(self, actual_seq_len: int) -> Optional[int]:
        """Find the next valid CUDA graph sequence length bucket."""
        for seq_len in sorted(self.cuda_graph_seq_len_buckets):
            if seq_len >= actual_seq_len:
                return seq_len
        return None
