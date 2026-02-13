"""
Depth Transformer Strategy for multi-codebook generation.

This strategy handles the depth transformer used in models like CSM
for generating multiple codebook tokens from backbone hidden states.
"""

from typing import Any, Dict, List, Optional

import torch

from ..model.base import BaseLMWithDepth
from ..requests import Request
from .base import (
    AllocatedResources,
    CacheSpec,
    CacheType,
    GenerationStrategy,
    ResourceSpec,
    StrategyPhase,
    StrategyType,
)


class DepthTransformerStrategy(GenerationStrategy):
    """
    Strategy for depth transformer inference.

    The depth transformer generates multiple codebook tokens from backbone
    hidden states. It has its own KV cache separate from the main LLM.

    This strategy handles:
    - Initial prefill with 2 positions (start token + first codebook)
    - Iterative decode for remaining codebooks
    """

    def __init__(self, model: BaseLMWithDepth, **kwargs):
        if not model.has_depth_transformer:
            raise ValueError("Model does not have a depth transformer")
        super().__init__(model, **kwargs)
        self._phases = [
            StrategyPhase(
                name="depth_prefill",
                is_stateful=True,
                requires_cache=True,
                batch_size_limits=None,
            ),
            StrategyPhase(
                name="depth_decode",
                is_stateful=True,
                requires_cache=True,
                batch_size_limits=None,
            ),
        ]

    @property
    def name(self) -> str:
        return "depth_transformer"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.DEPTH_TRANSFORMER

    @property
    def phases(self) -> List[StrategyPhase]:
        return self._phases

    @property
    def depth_model(self) -> BaseLMWithDepth:
        """Type-safe access to model with depth transformer."""
        return self.model

    def resource_spec(
        self,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
    ) -> ResourceSpec:
        """Specify depth KV cache and FlashInfer requirements."""
        depth_kv_cache_spec = CacheSpec(
            cache_type=CacheType.DEPTH_KV_CACHE,
            shape=(
                self.depth_model.depth_num_hidden_layers,
                max_num_pages,
                2,  # K and V
                self.depth_model.depth_n_codebooks,  # Sequence dimension for depth
                self.depth_model.depth_num_key_value_heads,
                self.depth_model.depth_head_dim,
            ),
            dtype=torch.bfloat16,
        )

        depth_state_size = self.depth_model.depth_num_attention_heads * self.depth_model.depth_head_dim

        return ResourceSpec(
            cache_specs=[depth_kv_cache_spec],
            requires_cuda_graph=True,
            requires_flashinfer=True,
            flashinfer_config={
                "n_qo_head": self.depth_model.depth_num_attention_heads,
                "n_kv_head": self.depth_model.depth_num_key_value_heads,
                "n_state": depth_state_size,
                "page_size": page_size,
            },
        )

    def prepare_inputs(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        phase: str,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for depth transformer execution.

        Args:
            requests: List of requests
            resources: Allocated resources
            phase: "depth_prefill" or "depth_decode"

        Returns:
            Dictionary with depth transformer inputs
        """
        if len(requests) == 0:
            return None

        batch_size = len(requests)
        device = resources.device

        if phase == "depth_prefill":
            # Initial depth prefill: sequence length of 2
            depth_position_ids = torch.tensor(
                [0, 1] * batch_size,
                device=device,
                dtype=torch.int32,
            )
            depth_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * 2
            depth_kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32)
            depth_kv_indices = torch.arange(batch_size, dtype=torch.int32)
            depth_kv_last_page_len = torch.tensor([2] * batch_size, dtype=torch.int32)
        else:
            # Decode: single position
            # Position will be set based on iteration
            depth_position_ids = None  # Set in execute
            depth_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32)
            depth_kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32)
            depth_kv_indices = torch.arange(batch_size, dtype=torch.int32)
            depth_kv_last_page_len = None  # Updated per iteration

        return {
            "batch_size": batch_size,
            "depth_position_ids": depth_position_ids,
            "depth_qo_indptr": depth_qo_indptr,
            "depth_kv_indptr": depth_kv_indptr,
            "depth_kv_indices": depth_kv_indices,
            "depth_kv_last_page_len": depth_kv_last_page_len,
        }

    async def execute(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        phase: str,
        prepared_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Execute depth transformer for all codebooks.

        This runs the full depth transformer loop, generating tokens
        for all codebooks iteratively.

        Args:
            requests: List of requests with backbone outputs
            resources: Allocated resources including depth KV cache
            phase: Not used - depth runs full loop internally
            prepared_inputs: Must include "output_ids" and "hidden_for_depth"
                from backbone sampling

        Returns:
            output_ids tensor with all codebook tokens filled in
        """
        if len(requests) == 0 or prepared_inputs is None:
            return None

        output_ids = prepared_inputs["output_ids"]
        hidden_for_depth = prepared_inputs["hidden_for_depth"]

        depth_kv_cache = resources.caches.get(CacheType.DEPTH_KV_CACHE)
        depth_attn_wrapper = resources.prefill_wrapper  # Depth uses prefill-style wrapper
        device = resources.device

        batch_size = output_ids.shape[0]

        # Initialize depth transformer indices
        depth_position_ids = torch.tensor([0, 1] * batch_size, device=device, dtype=torch.int32)
        depth_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32) * 2
        depth_kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32)
        depth_kv_indices = torch.arange(batch_size, dtype=torch.int32)
        depth_kv_last_page_len = torch.tensor([2] * batch_size, dtype=torch.int32)

        # Zero the depth KV cache for this batch
        depth_kv_cache.zero_()

        # Iterate over codebooks
        for i in range(1, self.depth_model.depth_n_codebooks):
            depth_attn_wrapper.plan(
                qo_indptr=depth_qo_indptr,
                paged_kv_indptr=depth_kv_indptr,
                paged_kv_indices=depth_kv_indices,
                paged_kv_last_page_len=depth_kv_last_page_len,
                dtype=torch.bfloat16,
            )
            torch.cuda.synchronize()

            if i == 1:
                # First iteration: reshape hidden states for prefill
                hidden_for_depth = hidden_for_depth.view(batch_size * 2, -1)
                depth_position_ids = depth_position_ids.view(batch_size * 2)

                depth_logits = self.depth_model.depth_forward(
                    hidden_states=hidden_for_depth,
                    position_ids=depth_position_ids,
                    attn_wrapper=depth_attn_wrapper,
                    kv_cache=depth_kv_cache,
                )

                # Select last position for each request
                actual_qo_indptr = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * 2
                depth_logits = depth_logits[actual_qo_indptr[1:] - 1]

                output_ids[:, i], hidden_for_depth = self.depth_model.depth_sampling(
                    logits=depth_logits,
                    i_iteration=i,
                    requests=requests,
                )
            else:
                # Subsequent iterations: single token decode
                depth_logits = self.depth_model.depth_forward(
                    hidden_states=hidden_for_depth,
                    position_ids=depth_position_ids,
                    attn_wrapper=depth_attn_wrapper,
                    kv_cache=depth_kv_cache,
                )

                output_ids[:, i], hidden_for_depth = self.depth_model.depth_sampling(
                    logits=depth_logits,
                    i_iteration=i,
                    requests=requests,
                )

            # Update for next iteration
            depth_position_ids = torch.tensor([i + 1] * batch_size, device=device, dtype=torch.int32)
            depth_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32)
            depth_kv_last_page_len += 1

        return output_ids

    def requires_cuda_graph_for_phase(self, phase: str) -> bool:
        """Depth transformer can use CUDA graphs for decode iterations."""
        return phase == "depth_decode"
