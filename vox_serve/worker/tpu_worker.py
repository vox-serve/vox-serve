"""TPU worker using Tokamax attention wrappers and XLA compilation.

Extends ModelWorker to run on Google Cloud TPU v6e devices.  Uses separate
K/V cache tensors (required by Tokamax) wrapped in ``TPUKVCache``, and
replaces FlashInfer attention with Tokamax prefill / PyTorch decode wrappers.
"""

import queue
from typing import Coroutine, List, Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from ..model import load_model
from ..requests import LMInputs, Request
from ..tokamax_utils import (
    TokmaxDecodeWrapper,
    TokmaxPrefillWrapper,
    TPUKVCache,
)
from ..utils import get_logger


class TPUWorker:
    """Model worker for TPU execution.

    Mirrors the ``ModelWorker`` API (prepare_lm_inputs, run_lm_prefill,
    run_lm_decode, run_lm_depth, run_detokenize, free_kv_cache, etc.)
    but uses Tokamax wrappers and XLA compilation instead of FlashInfer
    and CUDA graphs.
    """

    def __init__(
        self,
        model_name: str,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        repetition_penalty: float = None,
        repetition_window: int = None,
        cfg_scale: float = None,
        greedy: bool = False,
        enable_nvtx: bool = False,
        enable_torch_compile: bool = False,
        detokenizer_device: Optional[str] = None,
        dp_rank: int = 0,
        dp_size: int = 1,
        detokenize_interval: int = None,
    ):
        self.xla_device = xm.xla_device()
        self.device = self.xla_device
        # Audio decoding runs on CPU (no TPU audio codec support yet)
        self.detokenizer_device = detokenizer_device or "cpu"
        self.max_batch_size = max_batch_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size

        # Logger
        base_logger = get_logger(__name__)
        if dp_size > 1:
            import logging

            self.logger = logging.LoggerAdapter(base_logger, {"dp_rank": dp_rank})
            self.logger.process = lambda msg, kwargs: (f"[DP {dp_rank}/{dp_size}] {msg}", kwargs)
        else:
            self.logger = base_logger

        self.logger.info(f"Initializing TPUWorker on {self.xla_device}")

        # No NVTX on TPU
        self.nvtx_enabled = False

        # Store sampling parameters
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.repetition_window = repetition_window
        self.cfg_scale = cfg_scale

        self.max_num_pages = max_num_pages
        self.page_size = page_size

        # Load model — load on CPU first, then move to XLA device
        self.logger.info(f"Loading model {model_name} ...")
        self.model = load_model(
            model_name,
            device="cpu",
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            cfg_scale=cfg_scale,
            greedy=greedy,
            enable_torch_compile=False,
            audio_decoder_device=self.detokenizer_device,
            detokenize_interval=detokenize_interval,
        )
        # Move model parameters to TPU (audio decoder stays on its device)
        self.model.model.to(self.xla_device)
        # Update the model's device attribute so sampling creates tensors on XLA
        self.model.device = str(self.xla_device)
        self.logger.info("Model loaded and moved to TPU")

        # Offsets tensor for client tracking
        self.offsets = torch.zeros(self.max_batch_size, dtype=torch.int32, device=self.device)

        # Page pool
        self.empty_pages: queue.Queue = queue.Queue()
        for i in range(self.max_num_pages):
            self.empty_pages.put(i)

        # Watermarking (not supported on TPU for now)
        self.needs_watermarking = False

        # Batch size buckets for XLA compilation (powers of 2)
        self._batch_size_buckets = sorted(
            [2**i for i in range(int(np.log2(max(max_batch_size, 1))) + 1)],
            reverse=True,
        )

        self._prepare_attention_wrappers()

    # ------------------------------------------------------------------
    # Properties (mirror ModelWorker)
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_pow2(lst: list) -> list:
        """Pad a list with zeros to the next power-of-2 length for stable XLA shapes."""
        n = len(lst)
        if n == 0:
            return [0]
        bucketed = 1 << (n - 1).bit_length()
        if bucketed > n:
            return lst + [0] * (bucketed - n)
        return lst

    @property
    def detokenize_interval(self) -> int:
        return self.model.detokenize_interval

    @property
    def detokenize_overlap(self) -> int:
        return self.model.detokenize_overlap

    @property
    def supports_audio_input(self) -> bool:
        return self.model.supports_audio_input

    @property
    def available_batch_sizes(self) -> Optional[List[int]]:
        return self._batch_size_buckets

    # ------------------------------------------------------------------
    # Attention wrappers & KV cache
    # ------------------------------------------------------------------

    def _prepare_attention_wrappers(self):
        wrapper_kwargs = dict(
            n_qo_head=self.model.num_attention_heads,
            n_kv_head=self.model.num_key_value_heads,
            n_state=self.model.hidden_size,
            page_size=self.page_size,
            device=self.xla_device,
        )

        self.prefill_wrapper = TokmaxPrefillWrapper(**wrapper_kwargs)
        self.decode_wrapper = TokmaxDecodeWrapper(**wrapper_kwargs)

        # Separate K/V caches for Tokamax
        cache_shape = (
            self.model.num_hidden_layers,
            self.max_num_pages,
            self.page_size,
            self.model.num_key_value_heads,
            self.model.head_dim,
        )
        self.kv_cache = TPUKVCache(
            k_cache=torch.zeros(*cache_shape, dtype=torch.bfloat16, device=self.xla_device),
            v_cache=torch.zeros(*cache_shape, dtype=torch.bfloat16, device=self.xla_device),
        )

        kv_cache_size = self.kv_cache.k_cache.numel() * 2 * self.kv_cache.k_cache.element_size()
        self.logger.info(f"TPU KV cache size: {kv_cache_size / 1024 / 1024:.2f} MB")

        # Depth transformer
        self.has_depth_transformer = self.model.has_depth_transformer
        if self.has_depth_transformer:
            depth_state_size = self.model.depth_num_attention_heads * self.model.depth_head_dim
            # Depth cache "page_size" = depth_n_codebooks (not backbone page_size)
            self._depth_page_size = self.model.depth_n_codebooks
            depth_wrapper_kwargs = dict(
                n_qo_head=self.model.depth_num_attention_heads,
                n_kv_head=self.model.depth_num_key_value_heads,
                n_state=depth_state_size,
                page_size=self._depth_page_size,
                device=self.xla_device,
            )
            self.depth_attn_wrapper = TokmaxPrefillWrapper(**depth_wrapper_kwargs)

            depth_cache_shape = (
                self.model.depth_num_hidden_layers,
                self.max_num_pages,
                self.model.depth_n_codebooks,
                self.model.depth_num_key_value_heads,
                self.model.depth_head_dim,
            )
            self.depth_kv_cache = TPUKVCache(
                k_cache=torch.zeros(*depth_cache_shape, dtype=torch.bfloat16, device=self.xla_device),
                v_cache=torch.zeros(*depth_cache_shape, dtype=torch.bfloat16, device=self.xla_device),
            )
        else:
            self.depth_attn_wrapper = None
            self.depth_kv_cache = None

        self._warmup_xla_compilation()

    def _warmup_xla_compilation(self):
        """Trigger XLA compilation for all graph shapes used during inference.

        XLA compiles lazily on first execution of each unique graph shape.
        Pre-compiling during startup avoids latency spikes on the first request.
        Warms up: decode backbone, depth transformer (both first-iter and rest),
        and prefill with a short dummy sequence.
        """
        import time

        self.logger.info("Starting XLA compilation warmup...")
        t_total = time.time()
        n_codebooks = self.model.n_codebooks

        # Helper: create tensors on CPU then transfer (avoids XLA constants)
        def _t(data, dtype=torch.int32):
            return torch.tensor(data, dtype=dtype).to(self.xla_device)

        # Common forward kwargs
        def _fwd_kwargs(input_ids, position_ids, wrapper, cache):
            kw = dict(input_ids=input_ids, position_ids=position_ids, attn_wrapper=wrapper, kv_cache=cache)
            if self.model.needs_input_features:
                kw["input_features"] = torch.zeros(
                    input_ids.shape[0], self.model.hidden_size, dtype=torch.bfloat16, device=self.xla_device,
                )
            if self.model.needs_input_masks:
                kw["input_masks"] = torch.zeros(
                    input_ids.shape[0], n_codebooks, dtype=torch.bool, device=self.xla_device,
                )
            return kw

        with torch.no_grad():
            # ---- 1. Prefill (short dummy sequence) ----
            seq_len = 16
            dummy_ids = torch.zeros(seq_len, n_codebooks, dtype=torch.long, device=self.xla_device)
            dummy_pos = torch.arange(seq_len, dtype=torch.int32).to(self.xla_device)

            self.prefill_wrapper.plan(
                _t([0, seq_len]), _t([0, 1]), _t([0]), _t([seq_len]), torch.bfloat16,
            )
            xm.mark_step()

            t0 = time.time()
            self.model.forward(**_fwd_kwargs(dummy_ids, dummy_pos, self.prefill_wrapper, self.kv_cache))
            xm.mark_step()
            self.logger.info(f"  Prefill graph compiled in {time.time() - t0:.1f}s")

            # ---- 2. Decode for each batch size bucket ----
            for bs in self._batch_size_buckets:
                dec_ids = torch.zeros(bs, n_codebooks, dtype=torch.long, device=self.xla_device)
                dec_pos = _t([seq_len + 1] * bs)
                indptr = _t(list(range(bs + 1)))
                indices = _t(list(range(bs)))
                lpl = _t([seq_len + 1] * bs)

                self.decode_wrapper.plan(indptr, indices, lpl, torch.bfloat16)
                xm.mark_step()

                t0 = time.time()
                self.model.forward(**_fwd_kwargs(dec_ids, dec_pos, self.decode_wrapper, self.kv_cache))
                xm.mark_step()
                self.logger.info(f"  Decode bs={bs} compiled in {time.time() - t0:.1f}s")

            # ---- 3. Depth transformer (if present) ----
            if self.has_depth_transformer:
                nc = self.model.depth_n_codebooks
                depth_kv_budget = nc
                depth_dim = self.model.depth_num_attention_heads * self.model.depth_head_dim

                for bs in self._batch_size_buckets:
                    # 3a. Depth first iteration (seq_len=2 per request)
                    d_qo = torch.arange(bs + 1, dtype=torch.int32).to(self.xla_device) * 2
                    d_kvi = torch.arange(bs + 1, dtype=torch.int32).to(self.xla_device)
                    d_kvx = torch.arange(bs, dtype=torch.int32).to(self.xla_device)
                    d_lpl = _t([2] * bs)

                    self.depth_attn_wrapper.plan(
                        d_qo, d_kvi, d_kvx, d_lpl, torch.bfloat16, kv_len_budget=depth_kv_budget,
                    )
                    xm.mark_step()

                    t0 = time.time()
                    self.model.depth_forward(
                        hidden_states=torch.zeros(bs * 2, depth_dim, dtype=torch.bfloat16, device=self.xla_device),
                        position_ids=_t([0, 1] * bs),
                        attn_wrapper=self.depth_attn_wrapper, kv_cache=self.depth_kv_cache,
                    )
                    xm.mark_step()
                    self.logger.info(f"  Depth (i=1, bs={bs}) compiled in {time.time() - t0:.1f}s")

                    # 3b. Depth subsequent iterations (seq_len=1 per request)
                    d_qo2 = torch.arange(bs + 1, dtype=torch.int32).to(self.xla_device)
                    d_lpl2 = _t([3] * bs)
                    self.depth_attn_wrapper.plan(
                        d_qo2, d_kvi, d_kvx, d_lpl2, torch.bfloat16, kv_len_budget=depth_kv_budget,
                    )
                    xm.mark_step()

                    t0 = time.time()
                    self.model.depth_forward(
                        hidden_states=torch.zeros(bs, depth_dim, dtype=torch.bfloat16, device=self.xla_device),
                        position_ids=_t([3] * bs),
                        attn_wrapper=self.depth_attn_wrapper, kv_cache=self.depth_kv_cache,
                    )
                    xm.mark_step()
                    self.logger.info(f"  Depth (i>1, bs={bs}) compiled in {time.time() - t0:.1f}s")

                self.depth_kv_cache.k_cache.zero_()
                self.depth_kv_cache.v_cache.zero_()

            self.kv_cache.k_cache.zero_()
            self.kv_cache.v_cache.zero_()
            xm.mark_step()

        self.logger.info(f"XLA warmup complete in {time.time() - t_total:.1f}s")

    # ------------------------------------------------------------------
    # prepare_lm_inputs  (identical to ModelWorker — operates on CPU lists)
    # ------------------------------------------------------------------

    def prepare_lm_inputs(
        self,
        lm_requests: List[Request],
        detokenize_requests: List[Request],
    ) -> Optional[LMInputs]:
        """Prepare inputs for the LM step (same logic as ModelWorker)."""
        for req in detokenize_requests:
            req.audio_decode_idx = req.next_audio_decode_idx.copy()

        if len(lm_requests) == 0:
            return None

        qo_indptr = [0]
        paged_kv_indptr = [0]
        paged_kv_indices = []
        paged_kv_last_page_len = []

        input_ids_list = []
        position_ids_list = []
        input_features_list = []
        input_masks_list = []
        repetition_cache_list = []

        is_prefill = any(not req.done_lm_prefill for req in lm_requests)

        for req in lm_requests:
            if not req.done_lm_prefill:
                if req.is_input_streaming and not self.model.supports_input_streaming:
                    raise ValueError(
                        f"Input streaming is not supported by model {self.model.model_name}. "
                        f"Only Qwen3-TTS models support input streaming mode."
                    )

                model_kwargs = req.model_kwargs.copy()
                if req.is_input_streaming:
                    model_kwargs["is_input_streaming"] = True
                preprocess_output = self.model.preprocess(
                    prompt=req.prompt,
                    audio_path=req.audio_path,
                    **model_kwargs,
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

                input_ids_list.append(req.input_tokens.to(self.device))
                position_ids_list.extend(list(range(len(req.input_tokens))))
                input_features_list.append(req.input_features)
                input_masks_list.append(req.input_masks)
                repetition_cache_list.append(req.repetition_cache)

                n_pages_to_allocate = (len(req.input_tokens) + self.page_size - 1) // self.page_size
                req.kv_token_len = len(req.input_tokens)
                req.kv_pages = [self.empty_pages.get_nowait() for _ in range(n_pages_to_allocate)]
                req.kv_last_page_len = len(req.input_tokens) % self.page_size
                if req.kv_last_page_len == 0:
                    req.kv_last_page_len = self.page_size

                qo_indptr.append(qo_indptr[-1] + len(req.input_tokens))
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(req.kv_pages))
                paged_kv_indices.extend(req.kv_pages)
                paged_kv_last_page_len.append(req.kv_last_page_len)

                req.next_position_id = len(req.input_tokens) + 1
                req.done_lm_prefill = True
            else:
                if req.is_input_streaming:
                    self._inject_streaming_text_token(req)

                input_ids_list.append(req.input_tokens)
                input_features_list.append(req.input_features)
                input_masks_list.append(req.input_masks)
                repetition_cache_list.append(req.repetition_cache)

                req.kv_token_len += 1
                req.kv_last_page_len += 1
                if req.kv_last_page_len > self.page_size:
                    req.kv_pages.append(self.empty_pages.get_nowait())
                    req.kv_last_page_len = 1

                qo_indptr.append(qo_indptr[-1] + 1)
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(req.kv_pages))
                paged_kv_indices.extend(req.kv_pages)
                paged_kv_last_page_len.append(req.kv_last_page_len)

                position_ids_list.append(req.next_position_id)
                req.next_position_id += 1

        input_ids = torch.cat(input_ids_list, dim=0)
        position_ids = torch.tensor(position_ids_list, device=self.device, dtype=torch.int32)

        if self.model.needs_input_masks and input_masks_list:
            input_masks = torch.cat([m for m in input_masks_list if m is not None], dim=0).to(self.device)
        else:
            input_masks = None

        if self.model.needs_input_features and input_features_list:
            input_features = torch.cat([f for f in input_features_list if f is not None], dim=0).to(self.device)
        else:
            input_features = None

        if self.model.use_repetition_penalty and repetition_cache_list:
            repetition_cache = torch.stack([c for c in repetition_cache_list if c is not None], dim=0).to(self.device)
        else:
            repetition_cache = None

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

    def _inject_streaming_text_token(self, req: Request) -> None:
        """Inject the next text token for input streaming requests (same as ModelWorker)."""
        try:
            next_text_token = req.pending_text_tokens.get_nowait()
            req.text_token_cursor += 1
            req.input_tokens[0, -1] = next_text_token
        except Exception:
            if req.text_complete and not req.eos_injected:
                if hasattr(self.model, "config") and hasattr(self.model.config, "tts_eos_token_id"):
                    req.input_tokens[0, -1] = self.model.config.tts_eos_token_id
                    req.eos_injected = True
                elif hasattr(self.model, "config") and hasattr(self.model.config, "tts_pad_token_id"):
                    req.input_tokens[0, -1] = self.model.config.tts_pad_token_id
            elif hasattr(self.model, "config") and hasattr(self.model.config, "tts_pad_token_id"):
                req.input_tokens[0, -1] = self.model.config.tts_pad_token_id

    # ------------------------------------------------------------------
    # LM prefill
    # ------------------------------------------------------------------

    def _to_xla(self, data, dtype=torch.int32):
        """Create tensor on CPU, transfer to XLA device as runtime input (not a compile-time constant)."""
        return torch.tensor(data, dtype=dtype).to(self.xla_device)

    def run_lm_prefill(self, requests: List[Request], lm_inputs: LMInputs) -> Optional[Coroutine]:
        if len(requests) == 0:
            return None

        qo_indptr = lm_inputs["qo_indptr"]
        paged_kv_indptr = lm_inputs["paged_kv_indptr"]
        paged_kv_indices = lm_inputs["paged_kv_indices"]
        paged_kv_last_page_len = lm_inputs["paged_kv_last_page_len"]
        input_ids = lm_inputs["input_ids"]
        position_ids = lm_inputs["position_ids"]
        input_features = lm_inputs["input_features"]
        input_masks = lm_inputs["input_masks"]
        repetition_cache = lm_inputs["repetition_cache"]

        model_dtype = getattr(self.model, "dtype", torch.bfloat16)
        if input_features is not None and input_features.is_floating_point() and input_features.dtype != model_dtype:
            input_features = input_features.to(model_dtype)

        qo_indptr_tensor = self._to_xla(qo_indptr)
        paged_kv_indptr_tensor = self._to_xla(paged_kv_indptr)
        paged_kv_indices_tensor = self._to_xla(self._pad_to_pow2(paged_kv_indices))
        paged_kv_last_page_len_tensor = self._to_xla(paged_kv_last_page_len)

        self.prefill_wrapper.plan(
            qo_indptr_tensor,
            paged_kv_indptr_tensor,
            paged_kv_indices_tensor,
            paged_kv_last_page_len_tensor,
            torch.bfloat16,
        )
        xm.mark_step()

        with torch.no_grad():
            if self.has_depth_transformer:
                logits, backbone_hidden_states = self.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_wrapper=self.prefill_wrapper,
                    kv_cache=self.kv_cache,
                    input_features=input_features,
                    input_masks=input_masks,
                )
                xm.mark_step()

                if getattr(self.prefill_wrapper, "qo_indptr", None) is not None:
                    logits = logits[self.prefill_wrapper.qo_indptr[:-1] - 1]
                    backbone_hidden_states = backbone_hidden_states[self.prefill_wrapper.qo_indptr[:-1] - 1]

                output_ids, hidden_for_depth = self.model.sampling(
                    logits=logits,
                    hidden_states=backbone_hidden_states,
                    requests=requests,
                    repetition_cache=repetition_cache,
                )

                output_ids = self.run_lm_depth(output_ids, hidden_for_depth, requests)
            else:
                logits = self.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_wrapper=self.prefill_wrapper,
                    kv_cache=self.kv_cache,
                    input_features=input_features,
                    input_masks=input_masks,
                )

                if getattr(self.prefill_wrapper, "qo_indptr", None) is not None:
                    logits = logits[self.prefill_wrapper.qo_indptr[:-1] - 1]

                output_ids, task = self.model.sampling(
                    logits=logits,
                    requests=requests,
                    repetition_cache=repetition_cache,
                )
                return task

    # ------------------------------------------------------------------
    # LM decode
    # ------------------------------------------------------------------

    def run_lm_decode(self, requests: List[Request], lm_inputs: LMInputs) -> Optional[Coroutine]:
        if len(requests) == 0:
            return None

        paged_kv_indptr = lm_inputs["paged_kv_indptr"]
        paged_kv_indices = lm_inputs["paged_kv_indices"]
        paged_kv_last_page_len = lm_inputs["paged_kv_last_page_len"]
        input_ids = lm_inputs["input_ids"]
        position_ids = lm_inputs["position_ids"]
        input_features = lm_inputs["input_features"]
        input_masks = lm_inputs["input_masks"]
        repetition_cache = lm_inputs["repetition_cache"]

        model_dtype = getattr(self.model, "dtype", torch.bfloat16)
        if input_features is not None and input_features.is_floating_point() and input_features.dtype != model_dtype:
            input_features = input_features.to(model_dtype)

        paged_kv_indptr_tensor = self._to_xla(paged_kv_indptr)
        paged_kv_indices_tensor = self._to_xla(self._pad_to_pow2(paged_kv_indices))
        paged_kv_last_page_len_tensor = self._to_xla(paged_kv_last_page_len)

        self.decode_wrapper.plan(
            paged_kv_indptr_tensor,
            paged_kv_indices_tensor,
            paged_kv_last_page_len_tensor,
            torch.bfloat16,
        )
        xm.mark_step()

        with torch.no_grad():
            if self.has_depth_transformer:
                logits, backbone_hidden_states = self.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_wrapper=self.decode_wrapper,
                    kv_cache=self.kv_cache,
                    input_features=input_features,
                    input_masks=input_masks,
                )

                output_ids, hidden_for_depth = self.model.sampling(
                    logits=logits,
                    hidden_states=backbone_hidden_states,
                    requests=requests,
                    repetition_cache=repetition_cache,
                )

                output_ids = self.run_lm_depth(output_ids, hidden_for_depth, requests)
            else:
                logits = self.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attn_wrapper=self.decode_wrapper,
                    kv_cache=self.kv_cache,
                    input_features=input_features,
                    input_masks=input_masks,
                )

                output_ids, task = self.model.sampling(
                    logits=logits,
                    requests=requests,
                    repetition_cache=repetition_cache,
                )
                return task

    # ------------------------------------------------------------------
    # Depth transformer
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_lm_depth(
        self,
        output_ids: torch.Tensor,
        hidden_for_depth: torch.Tensor,
        requests: List[Request],
    ) -> torch.Tensor:
        # Flush pending XLA ops to free HBM before depth transformer iterations
        xm.mark_step()

        bs = output_ids.shape[0]
        n_codebooks = self.model.depth_n_codebooks

        depth_position_ids = self._to_xla([0, 1] * bs)
        depth_qo_indptr = torch.arange(bs + 1, dtype=torch.int32).to(self.xla_device) * 2
        depth_kv_indptr = torch.arange(bs + 1, dtype=torch.int32).to(self.xla_device)
        depth_kv_indices = torch.arange(bs, dtype=torch.int32).to(self.xla_device)
        depth_kv_last_page_len = self._to_xla([2] * bs)
        self.depth_kv_cache.k_cache.zero_()
        self.depth_kv_cache.v_cache.zero_()

        # Fixed KV length budget = depth_n_codebooks so all iterations share
        # the same bucketed attention shape → XLA reuses cached HLO programs.
        depth_kv_budget = n_codebooks

        for i in range(1, n_codebooks):
            self.depth_attn_wrapper.plan(
                qo_indptr=depth_qo_indptr,
                paged_kv_indptr=depth_kv_indptr,
                paged_kv_indices=depth_kv_indices,
                paged_kv_last_page_len=depth_kv_last_page_len,
                dtype=torch.bfloat16,
                kv_len_budget=depth_kv_budget,
            )
            xm.mark_step()

            if i == 1:
                hidden_for_depth = hidden_for_depth.view(bs * 2, -1)
                depth_position_ids = depth_position_ids.view(bs * 2)

                depth_logits = self.model.depth_forward(
                    hidden_states=hidden_for_depth,
                    position_ids=depth_position_ids,
                    attn_wrapper=self.depth_attn_wrapper,
                    kv_cache=self.depth_kv_cache,
                )

                actual_qo_indptr = torch.arange(bs + 1, dtype=torch.int32).to(self.device) * 2
                depth_logits = depth_logits[actual_qo_indptr[1:] - 1]

                output_ids[:, i], hidden_for_depth = self.model.depth_sampling(
                    logits=depth_logits,
                    i_iteration=i,
                    requests=requests,
                )
            else:
                depth_logits = self.model.depth_forward(
                    hidden_states=hidden_for_depth,
                    position_ids=depth_position_ids,
                    attn_wrapper=self.depth_attn_wrapper,
                    kv_cache=self.depth_kv_cache,
                )

                output_ids[:, i], hidden_for_depth = self.model.depth_sampling(
                    logits=depth_logits,
                    i_iteration=i,
                    requests=requests,
                )

            depth_position_ids = self._to_xla([i + 1] * bs)
            depth_qo_indptr = torch.arange(bs + 1, dtype=torch.int32).to(self.xla_device)
            depth_kv_last_page_len += 1

        return output_ids

    # ------------------------------------------------------------------
    # Detokenization  (audio decoding on CPU)
    # ------------------------------------------------------------------

    def run_detokenize(self, requests: List[Request]):
        if len(requests) == 0:
            return

        token_ids = []
        decoder_caches = []
        request_chunk_mapping = []

        for req_idx, req in enumerate(requests):
            for chunk_idx in range(len(req.audio_decode_idx)):
                decode_idx = req.audio_decode_idx[chunk_idx]
                new_tokens = req.lm_output_audio_tokens[decode_idx : decode_idx + self.detokenize_interval]

                if len(new_tokens) < self.detokenize_interval:
                    new_tokens.extend([new_tokens[-1]] * (self.detokenize_interval - len(new_tokens)))

                token_ids.append(torch.cat(new_tokens, dim=0))
                if req.decoder_cache is not None:
                    decoder_caches.append(req.decoder_cache)
                request_chunk_mapping.append((req_idx, chunk_idx))

        if not token_ids:
            return

        token_ids = torch.stack(token_ids, dim=0)

        # Transfer to CPU for audio decoding (ensure integer dtype for codec)
        token_ids = token_ids.to(device="cpu", dtype=torch.long)

        # Use streaming decode_chunk() when decoder_cache is available (much faster
        # than the non-streaming chunked_decode() fallback, especially for small intervals)
        from vox_serve.tokenizer.base import DecoderCache

        if decoder_caches:
            batched_cache = DecoderCache.cat(decoder_caches)
            audio_tensors = self.model.postprocess(token_ids, decoder_cache=batched_cache)
            # Copy updated cache state back to each request
            for i, (req_idx, _) in enumerate(request_chunk_mapping):
                requests[req_idx].decoder_cache.copy_from(batched_cache[i : i + 1])
        else:
            audio_tensors = self.model.postprocess(token_ids)
        self.logger.debug("Audio tensors: %s", audio_tensors)

        for i, (req_idx, chunk_idx) in enumerate(request_chunk_mapping):
            req = requests[req_idx]
            decode_idx = req.audio_decode_idx[chunk_idx]

            audio = audio_tensors[i].detach().cpu().float().numpy()
            audio_int16 = (audio * 32767).astype(np.int16)

            last_chunk_len = len(req.lm_output_audio_tokens[decode_idx : decode_idx + self.detokenize_interval])
            if last_chunk_len < self.detokenize_interval:
                trim_len = int(audio_int16.shape[1] * (last_chunk_len - 0.5) / self.detokenize_interval)
                audio_int16 = audio_int16[:, :trim_len]

            audio_bytes = audio_int16.tobytes()
            req.output_audio.put(audio_bytes)

        for req in requests:
            if req.done_lm_generation and req.audio_decode_idx and (
                req.audio_decode_idx[-1] + self.detokenize_interval >= len(req.lm_output_audio_tokens)
            ):
                req.done_all = True

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def free_kv_cache(self, request: Request):
        if hasattr(request, "kv_pages") and request.kv_pages:
            for page_idx in request.kv_pages:
                self.empty_pages.put(page_idx)
            request.kv_pages = []
            request.kv_token_len = 0
            request.kv_last_page_len = 0

    # ------------------------------------------------------------------
    # NVTX stubs (no-op on TPU)
    # ------------------------------------------------------------------

    def nvtx_range_push(self, name: str):
        pass

    def nvtx_range_pop(self):
        pass
