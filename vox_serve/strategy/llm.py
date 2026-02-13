"""
LLM Strategy for language model backbone inference.

This strategy handles prefill and decode phases for autoregressive
language model generation with KV cache management.
"""

from typing import Any, Coroutine, Dict, List, Optional, Tuple

import torch

from ..model.base import BaseLM
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


class LLMStrategy(GenerationStrategy):
    """
    Strategy for LLM backbone inference with KV cache.

    Handles:
    - Prefill phase: Process entire prompt, initialize KV cache
    - Decode phase: Generate tokens one at a time, update KV cache

    The strategy is stateful, requiring KV cache management at the Pool level.
    """

    def __init__(self, model: BaseLM, **kwargs):
        super().__init__(model, **kwargs)
        self._phases = [
            StrategyPhase(
                name="prefill",
                is_stateful=True,
                requires_cache=True,
                batch_size_limits=(1, 8),  # Prefill is more memory-intensive
            ),
            StrategyPhase(
                name="decode",
                is_stateful=True,
                requires_cache=True,
                batch_size_limits=None,  # Can use full batch size
            ),
        ]

    @property
    def name(self) -> str:
        return "llm"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.LLM

    @property
    def phases(self) -> List[StrategyPhase]:
        return self._phases

    def resource_spec(
        self,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
    ) -> ResourceSpec:
        """Specify KV cache and FlashInfer requirements."""
        head_dim = self.model.hidden_size // self.model.num_attention_heads

        kv_cache_spec = CacheSpec(
            cache_type=CacheType.KV_CACHE,
            shape=(
                self.model.num_hidden_layers,
                max_num_pages,
                2,  # K and V
                page_size,
                self.model.num_key_value_heads,
                head_dim,
            ),
            dtype=torch.bfloat16,
        )

        return ResourceSpec(
            cache_specs=[kv_cache_spec],
            requires_cuda_graph=True,
            requires_flashinfer=True,
            flashinfer_config={
                "n_qo_head": self.model.num_attention_heads,
                "n_kv_head": self.model.num_key_value_heads,
                "n_state": self.model.hidden_size,
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
        Prepare inputs for LM execution.

        For prefill: Runs model.preprocess() and prepares full sequences
        For decode: Prepares single-token inputs from previous outputs
        """
        if len(requests) == 0:
            return None

        # FlashInfer inputs
        qo_indptr = [0]
        paged_kv_indptr = [0]
        paged_kv_indices = []
        paged_kv_last_page_len = []

        # Model inputs
        input_ids_list = []
        position_ids_list = []
        input_features_list = []
        input_masks_list = []
        repetition_cache_list = []

        is_prefill = phase == "prefill"

        for req in requests:
            if is_prefill and not req.done_lm_prefill:
                # Prefill request - run preprocessing
                preprocess_output = self.model.preprocess(
                    prompt=req.prompt,
                    audio_path=req.audio_path,
                )
                req.input_tokens = preprocess_output.input_tokens
                if req.input_tokens is not None:
                    req.input_length = req.input_tokens.shape[0]

                if preprocess_output.input_features is not None:
                    req.input_features = preprocess_output.input_features
                if preprocess_output.input_masks is not None:
                    req.input_masks = preprocess_output.input_masks
                if preprocess_output.repetition_cache is not None:
                    req.repetition_cache = preprocess_output.repetition_cache
                if getattr(preprocess_output, "decoder_cache", None) is not None:
                    req.decoder_cache = preprocess_output.decoder_cache

                input_ids_list.append(req.input_tokens.to(resources.device, non_blocking=True))
                position_ids_list.extend([i for i in range(len(req.input_tokens))])
                input_features_list.append(req.input_features)
                input_masks_list.append(req.input_masks)
                repetition_cache_list.append(req.repetition_cache)

                # KV cache page allocation will be handled by Pool
                # Here we just track what we need
                n_pages_to_allocate = (len(req.input_tokens) + resources.page_size - 1) // resources.page_size
                req.kv_token_len = len(req.input_tokens)
                req.kv_last_page_len = len(req.input_tokens) % resources.page_size
                if req.kv_last_page_len == 0:
                    req.kv_last_page_len = resources.page_size

                qo_indptr.append(qo_indptr[-1] + len(req.input_tokens))
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(req.kv_pages))
                paged_kv_indices.extend(req.kv_pages)
                paged_kv_last_page_len.append(req.kv_last_page_len)

                req.next_position_id = len(req.input_tokens) + 1
                req.done_lm_prefill = True

            else:
                # Decode request - use previous output as input
                input_ids_list.append(req.input_tokens)
                input_features_list.append(req.input_features)
                input_masks_list.append(req.input_masks)
                repetition_cache_list.append(req.repetition_cache)

                req.kv_token_len += 1
                req.kv_last_page_len += 1
                # Page allocation for new pages handled by Pool

                qo_indptr.append(qo_indptr[-1] + 1)
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(req.kv_pages))
                paged_kv_indices.extend(req.kv_pages)
                paged_kv_last_page_len.append(req.kv_last_page_len)

                position_ids_list.append(req.next_position_id)
                req.next_position_id += 1

        # Build tensors
        input_ids = torch.cat(input_ids_list, dim=0)
        position_ids = torch.tensor(position_ids_list, device=resources.device, dtype=torch.int32)

        # Prepare optional inputs
        input_masks = None
        if self.model.needs_input_masks and input_masks_list:
            masks = [m for m in input_masks_list if m is not None]
            if masks:
                input_masks = torch.cat(masks, dim=0)

        input_features = None
        if self.model.needs_input_features and input_features_list:
            features = [f for f in input_features_list if f is not None]
            if features:
                input_features = torch.cat(features, dim=0)

        repetition_cache = None
        if self.model.use_repetition_penalty and repetition_cache_list:
            caches = [c for c in repetition_cache_list if c is not None]
            if caches:
                repetition_cache = torch.stack(caches, dim=0)

        return {
            "qo_indptr": qo_indptr,
            "paged_kv_indptr": paged_kv_indptr,
            "paged_kv_indices": paged_kv_indices,
            "paged_kv_last_page_len": paged_kv_last_page_len,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "input_features": input_features,
            "input_masks": input_masks,
            "repetition_cache": repetition_cache,
            "is_prefill": is_prefill,
        }

    async def execute(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        phase: str,
        prepared_inputs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Coroutine]]:
        """
        Execute LM prefill or decode.

        Returns:
            Tuple of (output_ids, optional async task for request updates)
        """
        if len(requests) == 0:
            return None, None

        if prepared_inputs is None:
            prepared_inputs = self.prepare_inputs(requests, resources, phase)

        if prepared_inputs is None:
            return None, None

        kv_cache = resources.caches.get(CacheType.KV_CACHE)

        # Build FlashInfer index tensors
        qo_indptr_tensor = torch.tensor(prepared_inputs["qo_indptr"], dtype=torch.int32)
        paged_kv_indptr_tensor = torch.tensor(prepared_inputs["paged_kv_indptr"], dtype=torch.int32)
        paged_kv_indices_tensor = torch.tensor(prepared_inputs["paged_kv_indices"], dtype=torch.int32)
        paged_kv_last_page_len_tensor = torch.tensor(prepared_inputs["paged_kv_last_page_len"], dtype=torch.int32)

        input_ids = prepared_inputs["input_ids"]
        position_ids = prepared_inputs["position_ids"]
        input_features = prepared_inputs["input_features"]
        input_masks = prepared_inputs["input_masks"]
        repetition_cache = prepared_inputs["repetition_cache"]

        # Ensure dtype consistency
        model_dtype = getattr(self.model, "dtype", torch.bfloat16)
        if input_features is not None and input_features.is_floating_point() and input_features.dtype != model_dtype:
            input_features = input_features.to(model_dtype)

        if phase == "prefill":
            return await self._execute_prefill(
                requests=requests,
                resources=resources,
                kv_cache=kv_cache,
                qo_indptr_tensor=qo_indptr_tensor,
                paged_kv_indptr_tensor=paged_kv_indptr_tensor,
                paged_kv_indices_tensor=paged_kv_indices_tensor,
                paged_kv_last_page_len_tensor=paged_kv_last_page_len_tensor,
                input_ids=input_ids,
                position_ids=position_ids,
                input_features=input_features,
                input_masks=input_masks,
                repetition_cache=repetition_cache,
            )
        else:
            return await self._execute_decode(
                requests=requests,
                resources=resources,
                kv_cache=kv_cache,
                paged_kv_indptr_tensor=paged_kv_indptr_tensor,
                paged_kv_indices_tensor=paged_kv_indices_tensor,
                paged_kv_last_page_len_tensor=paged_kv_last_page_len_tensor,
                input_ids=input_ids,
                position_ids=position_ids,
                input_features=input_features,
                input_masks=input_masks,
                repetition_cache=repetition_cache,
            )

    async def _execute_prefill(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        kv_cache: torch.Tensor,
        qo_indptr_tensor: torch.Tensor,
        paged_kv_indptr_tensor: torch.Tensor,
        paged_kv_indices_tensor: torch.Tensor,
        paged_kv_last_page_len_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_features: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
        repetition_cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Coroutine]]:
        """Execute prefill phase."""
        prefill_wrapper = resources.prefill_wrapper

        prefill_wrapper.plan(
            qo_indptr_tensor,
            paged_kv_indptr_tensor,
            paged_kv_indices_tensor,
            paged_kv_last_page_len_tensor,
            torch.bfloat16,
        )
        torch.cuda.synchronize()

        # Forward pass
        logits = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_wrapper=prefill_wrapper,
            kv_cache=kv_cache,
            input_features=input_features,
            input_masks=input_masks,
        )

        # Select last token for each request
        if getattr(prefill_wrapper, "qo_indptr", None) is not None:
            logits = logits[prefill_wrapper.qo_indptr[:-1] - 1]

        # Sampling
        output_ids, task = self.model.sampling(
            logits=logits,
            requests=requests,
            repetition_cache=repetition_cache,
        )

        return output_ids, task

    async def _execute_decode(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        kv_cache: torch.Tensor,
        paged_kv_indptr_tensor: torch.Tensor,
        paged_kv_indices_tensor: torch.Tensor,
        paged_kv_last_page_len_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_features: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
        repetition_cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Coroutine]]:
        """Execute decode phase."""
        decode_wrapper = resources.decode_wrapper

        decode_wrapper.plan(
            paged_kv_indptr_tensor,
            paged_kv_indices_tensor,
            paged_kv_last_page_len_tensor,
            torch.bfloat16,
        )
        torch.cuda.synchronize()

        # Forward pass
        logits = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_wrapper=decode_wrapper,
            kv_cache=kv_cache,
            input_features=input_features,
            input_masks=input_masks,
        )

        # Sampling
        output_ids, task = self.model.sampling(
            logits=logits,
            requests=requests,
            repetition_cache=repetition_cache,
        )

        return output_ids, task
