"""JAX TPU worker for Qwen3-TTS inference.

Uses the vLLM ragged paged attention (RPA) Pallas kernel for backbone
attention. The RPA kernel fuses KV cache write + attention, supporting
both prefill and decode in a single call.

CPU-side logic (preprocessing, audio decoding, request management) stays
in PyTorch via the existing Qwen3TTSModel.
"""

from __future__ import annotations

import queue
from functools import partial
from typing import Coroutine, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ..model import load_model
from ..model.jax.attention import allocate_kv_cache
from ..model.jax.params import load_params
from ..model.jax.qwen3_tts import (
    backbone_forward_rpa,
    decode_step,
    depth_forward_fused,
    embed_inputs,
)
from ..model.jax.sampling import greedy_sample, top_k_sample, top_p_sample
from ..requests import LMInputs, Request
from ..utils import get_logger


class JaxTPUWorker:
    """Model worker using pure JAX on TPU with vLLM RPA kernel."""

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
        self.logger = get_logger(__name__)
        self.max_batch_size = max_batch_size
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.detokenizer_device = detokenizer_device or "cpu"
        self.greedy = greedy
        self._temperature = temperature or 1.0

        # Max pages per sequence (for RPA page_indices flat array)
        self._pages_per_seq = (max_num_pages + max_batch_size - 1) // max(max_batch_size, 1)

        # ---- Load PyTorch model on CPU ----
        self.logger.info(f"Loading PyTorch model {model_name} on CPU...")
        self.model = load_model(
            model_name, device="cpu",
            top_p=top_p, top_k=top_k, min_p=min_p,
            temperature=temperature, max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            cfg_scale=cfg_scale, greedy=greedy,
            enable_torch_compile=False,
            audio_decoder_device=self.detokenizer_device,
            detokenize_interval=detokenize_interval,
        )
        self.device = "cpu"

        # ---- Load JAX params ----
        self.logger.info("Loading JAX params...")
        hf_name = model_name
        if model_name == "qwen3-tts":
            hf_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        elif model_name == "qwen3-tts-base":
            hf_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        self.jax_params, self.jax_config = load_params(hf_name)
        cfg = self.jax_config
        self.logger.info("JAX params loaded")

        # ---- Allocate paged KV cache (RPA merged format, per-layer) ----
        self.kv_caches = allocate_kv_cache(
            cfg.backbone_layers, max_num_pages, page_size,
            cfg.backbone_kv_heads, cfg.head_dim,
        )
        kv_mb = sum(c.nbytes for c in self.kv_caches) / (1024 * 1024)
        self.logger.info(f"Backbone KV cache ({len(self.kv_caches)} layers): {kv_mb:.1f} MB")

        # Page pool
        self.empty_pages: queue.Queue = queue.Queue()
        for i in range(max_num_pages):
            self.empty_pages.put(i)

        # ---- Properties ----
        self.has_depth_transformer = self.model.has_depth_transformer
        self.needs_watermarking = False
        self.nvtx_enabled = False
        self._batch_size_buckets = [max(max_batch_size, 1)]

        # ---- RNG state ----
        self._rng_key = jax.random.PRNGKey(42)

        # ---- Sampling function ----
        if greedy:
            self._sample_fn = greedy_sample
        elif top_k is not None:
            self._sample_fn = partial(top_k_sample, k=top_k, temperature=self._temperature)
        elif top_p is not None:
            self._sample_fn = partial(top_p_sample, p=top_p, temperature=self._temperature)
        else:
            self._sample_fn = greedy_sample

        # ---- JIT compile and warmup ----
        if self.has_depth_transformer:
            self._build_jit_decode()
        self._warmup_all()
        self.logger.info("JaxTPUWorker initialization complete")

    # ------------------------------------------------------------------
    # JIT compilation
    # ------------------------------------------------------------------

    def _build_jit_decode(self):
        """Build JIT-compiled decode: backbone(RPA) + sample + depth."""
        cfg = self.jax_config
        sample_fn = self._sample_fn

        def _decode_fn(
            params, kv_caches, input_ids, position_ids, input_masks,
            input_features, kv_lens, page_indices, cu_q_lens,
            distribution, rng_key,
        ):
            return decode_step(
                params, kv_caches, input_ids, position_ids,
                input_masks, input_features, kv_lens, page_indices,
                cu_q_lens, distribution, rng_key, cfg,
                sample_fn=sample_fn,
            )

        self._jit_decode = jax.jit(
            _decode_fn, static_argnames=(), donate_argnums=(1,)
        )
        self.logger.info("JIT decode function built (traces on first call)")

    def _warmup_all(self):
        """Pre-compile all JIT functions with dummy data.

        Exercises prefill backbone, prefill depth, decode (backbone+depth fused)
        so that no compilation happens during actual serving.
        """
        import time

        cfg = self.jax_config
        params = self.jax_params
        bs = max(self.max_batch_size, 1)
        rng = jax.random.PRNGKey(0)

        self.logger.info(f"Starting warmup (bs={bs})...")
        t_total = time.time()

        max_num_seqs = bs
        pages_per_seq = self._pages_per_seq

        # ---- 1. Prefill backbone (JIT, multiple bucket sizes) ----
        if not hasattr(self, "_jit_prefill_backbone"):
            @jax.jit
            def _jit_prefill(bp, emb, pos, kvc, kvl, pi, cql, dist):
                return backbone_forward_rpa(bp, emb, pos, kvc, kvl, pi, cql, dist, cfg)
            self._jit_prefill_backbone = _jit_prefill

        # Warm up common prefill bucket sizes
        for bucket_len in [16, 32]:
            t0 = time.time()
            dummy_embeds = jnp.zeros((bucket_len, cfg.backbone_hidden), dtype=jnp.bfloat16)
            dummy_pos = jnp.arange(bucket_len, dtype=jnp.int32)
            dummy_kv_lens = jnp.array(
                [bucket_len] + [0] * (max_num_seqs - 1), dtype=jnp.int32
            )
            dummy_pi = jnp.zeros(max_num_seqs * pages_per_seq, dtype=jnp.int32)
            dummy_cql = jnp.array(
                [0, bucket_len] + [bucket_len] * (max_num_seqs - 1), dtype=jnp.int32
            )
            dummy_dist = jnp.array([0, 0, 1], dtype=jnp.int32)

            logits, hidden, self.kv_caches = self._jit_prefill_backbone(
                params["backbone"], dummy_embeds, dummy_pos,
                self.kv_caches, dummy_kv_lens, dummy_pi, dummy_cql, dummy_dist,
            )
            logits.block_until_ready()
            self.logger.info(f"  Prefill backbone (len={bucket_len}): {time.time()-t0:.1f}s")

        # ---- 2. Prefill depth (JIT) ----
        t0 = time.time()
        dummy_c0 = jnp.zeros(1, dtype=jnp.int32)
        dummy_hidden = jnp.zeros((1, cfg.backbone_hidden), dtype=jnp.bfloat16)

        if not hasattr(self, "_jit_depth"):
            sample_fn = self._sample_fn

            @jax.jit
            def _jit_depth_fn(dp, ec, bh, c0, dkk, dkv, rng):
                return depth_forward_fused(dp, ec, bh, c0, dkk, dkv, cfg, rng, sample_fn=sample_fn)
            self._jit_depth = _jit_depth_fn

        depth_shape = (1, cfg.depth_layers, 17, cfg.depth_kv_heads, cfg.head_dim)
        dkk = jnp.zeros(depth_shape, dtype=jnp.bfloat16)
        dkv = jnp.zeros(depth_shape, dtype=jnp.bfloat16)
        _, _, _, _, rng = self._jit_depth(
            params["depth"], params["backbone"]["embed_codec"],
            dummy_hidden, dummy_c0, dkk, dkv, rng,
        )
        self.logger.info(f"  Prefill depth: {time.time()-t0:.1f}s")

        # ---- 3. Decode step (JIT, backbone+depth fused) ----
        if hasattr(self, "_jit_decode"):
            t0 = time.time()
            dummy_ids = jnp.zeros((bs, 17), dtype=jnp.int32)
            dummy_dpos = jnp.full(bs, 17, dtype=jnp.int32)
            dummy_masks = jnp.ones((bs, 17), dtype=jnp.bool_)
            dummy_feat = jnp.zeros((bs, cfg.backbone_hidden), dtype=jnp.bfloat16)
            dec_kv_lens = jnp.array(
                [18] * bs + [0] * (max_num_seqs - bs), dtype=jnp.int32
            )
            dec_pi = jnp.zeros(max_num_seqs * pages_per_seq, dtype=jnp.int32)
            dec_cql = jnp.zeros(max_num_seqs + 1, dtype=jnp.int32)
            dec_cql = dec_cql.at[1:bs + 1].set(jnp.arange(1, bs + 1))
            dec_cql = dec_cql.at[bs + 1:].set(bs)
            dec_dist = jnp.array([bs, bs, bs], dtype=jnp.int32)  # all decode

            _, _, self.kv_caches, rng = self._jit_decode(
                params, self.kv_caches,
                dummy_ids, dummy_dpos, dummy_masks, dummy_feat,
                dec_kv_lens, dec_pi, dec_cql, dec_dist, rng,
            )
            # Run twice to ensure stable
            _, _, self.kv_caches, rng = self._jit_decode(
                params, self.kv_caches,
                dummy_ids, dummy_dpos, dummy_masks, dummy_feat,
                dec_kv_lens, dec_pi, dec_cql, dec_dist, rng,
            )
            jax.block_until_ready(self.kv_caches)
            self.logger.info(f"  Decode step (fused): {time.time()-t0:.1f}s")

        # Reset KV caches
        self.kv_caches = allocate_kv_cache(
            cfg.backbone_layers, self.max_num_pages, self.page_size,
            cfg.backbone_kv_heads, cfg.head_dim,
        )
        self.logger.info(f"Warmup complete in {time.time()-t_total:.1f}s")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_lm_inputs(
        self, lm_requests: List[Request], detokenize_requests: List[Request],
    ) -> Optional[LMInputs]:
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
                model_kwargs = req.model_kwargs.copy()
                if req.is_input_streaming and not self.model.supports_input_streaming:
                    raise ValueError("Input streaming not supported.")
                if req.is_input_streaming:
                    model_kwargs["is_input_streaming"] = True

                preprocess_output = self.model.preprocess(
                    prompt=req.prompt, audio_path=req.audio_path, **model_kwargs,
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

                input_ids_list.append(req.input_tokens)
                position_ids_list.extend(list(range(len(req.input_tokens))))
                input_features_list.append(req.input_features)
                input_masks_list.append(req.input_masks)
                repetition_cache_list.append(req.repetition_cache)

                n_pages = (len(req.input_tokens) + self.page_size - 1) // self.page_size
                req.kv_token_len = len(req.input_tokens)
                req.kv_pages = [self.empty_pages.get_nowait() for _ in range(n_pages)]
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
        position_ids = torch.tensor(position_ids_list, dtype=torch.int32)

        input_masks = None
        if self.model.needs_input_masks and input_masks_list:
            input_masks = torch.cat([m for m in input_masks_list if m is not None], dim=0)
        input_features = None
        if self.model.needs_input_features and input_features_list:
            input_features = torch.cat([f for f in input_features_list if f is not None], dim=0)
        repetition_cache = None
        if self.model.use_repetition_penalty and repetition_cache_list:
            repetition_cache = torch.stack([c for c in repetition_cache_list if c is not None], dim=0)

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
    # RPA metadata computation
    # ------------------------------------------------------------------

    def _compute_rpa_metadata(self, lm_inputs: LMInputs, n_req: int, is_prefill: bool):
        """Compute RPA metadata tensors from page info.

        Returns numpy arrays for: kv_lens, page_indices, cu_q_lens, distribution.
        """
        qo_indptr = np.array(lm_inputs["qo_indptr"], dtype=np.int32)
        paged_kv_indptr = np.array(lm_inputs["paged_kv_indptr"], dtype=np.int32)
        paged_kv_indices = np.array(lm_inputs["paged_kv_indices"], dtype=np.int32)
        paged_kv_last_page_len = np.array(lm_inputs["paged_kv_last_page_len"], dtype=np.int32)

        max_num_seqs = self.max_batch_size
        pages_per_seq = self._pages_per_seq

        # kv_lens: total KV length per sequence
        num_pages_per_req = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        kv_lens_valid = (num_pages_per_req - 1) * self.page_size + paged_kv_last_page_len
        kv_lens = np.zeros(max_num_seqs, dtype=np.int32)
        kv_lens[:n_req] = kv_lens_valid

        # page_indices: flat (max_num_seqs * pages_per_seq,)
        page_indices = np.zeros(max_num_seqs * pages_per_seq, dtype=np.int32)
        for i in range(n_req):
            n_p = num_pages_per_req[i]
            src_start = paged_kv_indptr[i]
            dst_start = i * pages_per_seq
            page_indices[dst_start:dst_start + n_p] = paged_kv_indices[src_start:src_start + n_p]

        # cu_q_lens: cumulative query lengths
        cu_q_lens = np.zeros(max_num_seqs + 1, dtype=np.int32)
        q_lens = qo_indptr[1:] - qo_indptr[:-1]
        cu_q_lens[1:n_req + 1] = np.cumsum(q_lens)
        # Pad remaining entries with the total
        if n_req < max_num_seqs:
            cu_q_lens[n_req + 1:] = cu_q_lens[n_req]

        # distribution: [num_decode, end_prefill, end_mixed]
        if is_prefill:
            # All requests are prefill (mixed mode since q_len varies)
            distribution = np.array([0, 0, n_req], dtype=np.int32)
        else:
            # All requests are decode (q_len = 1)
            distribution = np.array([n_req, n_req, n_req], dtype=np.int32)

        return {
            "kv_lens": kv_lens,
            "page_indices": page_indices,
            "cu_q_lens": cu_q_lens,
            "distribution": distribution,
        }

    # ------------------------------------------------------------------
    # PyTorch ↔ JAX conversion
    # ------------------------------------------------------------------

    def _torch_to_jax(self, t: torch.Tensor, dtype=None) -> jnp.ndarray:
        t_cpu = t.detach().cpu()
        if t_cpu.dtype == torch.bfloat16:
            np_arr = t_cpu.float().numpy()
        else:
            np_arr = t_cpu.numpy()
        if dtype:
            return jnp.array(np_arr, dtype=dtype)
        return jnp.array(np_arr)

    # ------------------------------------------------------------------
    # LM prefill
    # ------------------------------------------------------------------

    def run_lm_prefill(
        self, requests: List[Request], lm_inputs: LMInputs,
    ) -> Optional[Coroutine]:
        if len(requests) == 0:
            return None

        import time as _time

        cfg = self.jax_config
        params = self.jax_params
        n_req = len(requests)

        t0 = _time.time()

        # Convert inputs
        input_ids = self._torch_to_jax(lm_inputs["input_ids"], dtype=jnp.int32)
        position_ids = self._torch_to_jax(lm_inputs["position_ids"], dtype=jnp.int32)
        if lm_inputs["input_features"] is not None:
            input_features = self._torch_to_jax(lm_inputs["input_features"], dtype=jnp.bfloat16)
        else:
            input_features = jnp.zeros(
                (input_ids.shape[0], cfg.backbone_hidden), dtype=jnp.bfloat16
            )
        if lm_inputs["input_masks"] is not None:
            input_masks = self._torch_to_jax(lm_inputs["input_masks"], dtype=jnp.bool_)
        else:
            input_masks = jnp.ones((input_ids.shape[0], 17), dtype=jnp.bool_)

        # Embed
        embeds = embed_inputs(params["backbone"], input_ids, input_masks, input_features)

        # Compute RPA metadata
        meta = self._compute_rpa_metadata(lm_inputs, n_req, is_prefill=True)

        t1 = _time.time()

        # JIT the prefill backbone (lazy build)
        if not hasattr(self, "_jit_prefill_backbone"):
            @jax.jit
            def _jit_prefill(bp, emb, pos, kvc, kvl, pi, cql, dist):
                return backbone_forward_rpa(bp, emb, pos, kvc, kvl, pi, cql, dist, cfg)
            self._jit_prefill_backbone = _jit_prefill

        # Pad embeds/positions to bucket size to avoid recompilation per prompt length
        actual_len = embeds.shape[0]
        bucket_len = 1 << max((actual_len - 1).bit_length(), 4)  # min 16
        if embeds.shape[0] < bucket_len:
            embeds = jnp.pad(embeds, ((0, bucket_len - actual_len), (0, 0)))
            position_ids = jnp.pad(position_ids, (0, bucket_len - actual_len))

        logits, hidden, self.kv_caches = self._jit_prefill_backbone(
            params["backbone"], embeds, position_ids,
            self.kv_caches,
            jnp.array(meta["kv_lens"]),
            jnp.array(meta["page_indices"]),
            jnp.array(meta["cu_q_lens"]),
            jnp.array(meta["distribution"]),
        )
        # Trim back to actual length
        logits = logits[:actual_len]
        hidden = hidden[:actual_len]
        logits.block_until_ready()

        t2 = _time.time()

        # Extract last-token logits per request
        qo_indptr = lm_inputs["qo_indptr"]
        last_indices = jnp.array([qo_indptr[i + 1] - 1 for i in range(n_req)])
        last_logits = logits[last_indices]
        last_hidden = hidden[last_indices]

        # Sample codebook-0
        c0_ids, self._rng_key = self._sample_fn(last_logits, self._rng_key)

        if self.has_depth_transformer:
            self._run_depth_and_writeback(requests, c0_ids, last_hidden, n_req)

        t3 = _time.time()
        self.logger.info(
            f"Prefill: prep={t1-t0:.3f}s backbone={t2-t1:.3f}s "
            f"depth={t3-t2:.3f}s total={t3-t0:.3f}s (tokens={input_ids.shape[0]})"
        )

        return None

    # ------------------------------------------------------------------
    # LM decode
    # ------------------------------------------------------------------

    _decode_call_count = 0

    def run_lm_decode(
        self, requests: List[Request], lm_inputs: LMInputs,
    ) -> Optional[Coroutine]:
        if len(requests) == 0:
            return None

        import time as _time
        t0 = _time.time()

        cfg = self.jax_config
        n_req = len(requests)

        # Convert inputs
        input_ids = self._torch_to_jax(lm_inputs["input_ids"], dtype=jnp.int32)
        position_ids = self._torch_to_jax(lm_inputs["position_ids"], dtype=jnp.int32)
        if lm_inputs["input_features"] is not None:
            input_features = self._torch_to_jax(lm_inputs["input_features"], dtype=jnp.bfloat16)
        else:
            input_features = jnp.zeros(
                (input_ids.shape[0], cfg.backbone_hidden), dtype=jnp.bfloat16
            )
        if lm_inputs["input_masks"] is not None:
            input_masks = self._torch_to_jax(lm_inputs["input_masks"], dtype=jnp.bool_)
        else:
            input_masks = jnp.ones((input_ids.shape[0], 17), dtype=jnp.bool_)

        # Compute RPA metadata
        meta = self._compute_rpa_metadata(lm_inputs, n_req, is_prefill=False)

        if self.has_depth_transformer and hasattr(self, "_jit_decode"):
            # Full JIT path: backbone + sample + depth in one compiled graph
            all_ids, ci_embed_sum, self.kv_caches, self._rng_key = self._jit_decode(
                self.jax_params,
                self.kv_caches,
                input_ids, position_ids, input_masks, input_features,
                jnp.array(meta["kv_lens"]),
                jnp.array(meta["page_indices"]),
                jnp.array(meta["cu_q_lens"]),
                jnp.array(meta["distribution"]),
                self._rng_key,
            )

            # Writeback
            n_cb = self.model.n_codebooks
            output_ids_np = np.zeros((n_req, n_cb), dtype=np.int64)
            output_ids_np[:, 0:16] = np.array(all_ids)
            output_ids_np[:, -1] = self.model.config.tts_pad_token_id
            ci_embed_np = np.array(ci_embed_sum).astype(np.float32)
            mdl_dtype = torch.bfloat16

            for i, req in enumerate(requests):
                out_torch = torch.from_numpy(output_ids_np[i:i + 1]).to(torch.long)
                req.input_tokens = torch.zeros(1, n_cb, dtype=torch.long)
                req.input_tokens[0, 0] = int(output_ids_np[i, 0])
                if not getattr(req, "is_input_streaming", False):
                    req.input_tokens[0, -1] = self.model.config.tts_pad_token_id
                req.input_masks = torch.ones(n_cb, dtype=torch.bool)[None, :]
                req.input_features = torch.zeros(1, self.model.hidden_size, dtype=mdl_dtype)
                req.input_features[:] += torch.from_numpy(ci_embed_np[i:i + 1]).to(mdl_dtype)
                req.lm_output_tokens.append(out_torch)
                if int(output_ids_np[i, 0]) != self.model.stop_token_id:
                    req.lm_output_audio_tokens.append(out_torch)
                else:
                    req.done_lm_generation = True
                    req.finish_reason = "stop_id_encountered"

            for req in requests:
                if req.next_position_id > self.model.max_tokens:
                    req.done_lm_generation = True
                    req.finish_reason = "max_tokens_reached"
        else:
            # Eager fallback
            embeds = embed_inputs(
                self.jax_params["backbone"], input_ids, input_masks, input_features
            )
            logits, hidden, self.kv_caches = backbone_forward_rpa(
                self.jax_params["backbone"], embeds, position_ids,
                self.kv_caches,
                jnp.array(meta["kv_lens"]),
                jnp.array(meta["page_indices"]),
                jnp.array(meta["cu_q_lens"]),
                jnp.array(meta["distribution"]),
                cfg,
            )
            c0_ids, self._rng_key = self._sample_fn(logits, self._rng_key)
            if self.has_depth_transformer:
                self._run_depth_and_writeback(requests, c0_ids, hidden, n_req)

        t1 = _time.time()
        self._decode_call_count += 1
        if self._decode_call_count <= 5 or self._decode_call_count % 50 == 0:
            self.logger.info(f"Decode #{self._decode_call_count}: {(t1-t0)*1000:.1f}ms (bs={n_req})")

        return None

    # ------------------------------------------------------------------
    # Depth + writeback
    # ------------------------------------------------------------------

    def _run_depth_and_writeback(
        self, requests: List[Request],
        c0_ids: jnp.ndarray, backbone_hidden: jnp.ndarray, bs: int,
    ):
        cfg = self.jax_config
        params = self.jax_params

        max_depth_seq = 17
        depth_shape = (
            bs, cfg.depth_layers, max_depth_seq, cfg.depth_kv_heads, cfg.head_dim,
        )
        depth_kv_k = jnp.zeros(depth_shape, dtype=jnp.bfloat16)
        depth_kv_v = jnp.zeros(depth_shape, dtype=jnp.bfloat16)

        # Use JIT-compiled depth forward (avoids re-tracing fori_loop each call)
        if not hasattr(self, "_jit_depth"):
            sample_fn = self._sample_fn

            @jax.jit
            def _jit_depth_fn(dp, ec, bh, c0, dkk, dkv, rng):
                return depth_forward_fused(
                    dp, ec, bh, c0, dkk, dkv, cfg, rng,
                    sample_fn=sample_fn,
                )
            self._jit_depth = _jit_depth_fn

        all_cb_ids, ci_embed_sum, _, _, self._rng_key = self._jit_depth(
            params["depth"], params["backbone"]["embed_codec"],
            backbone_hidden, c0_ids,
            depth_kv_k, depth_kv_v, self._rng_key,
        )

        n_cb = self.model.n_codebooks
        output_ids_np = np.zeros((bs, n_cb), dtype=np.int64)
        output_ids_np[:, 0] = np.array(c0_ids)
        output_ids_np[:, 1:16] = np.array(all_cb_ids)
        output_ids_np[:, -1] = self.model.config.tts_pad_token_id

        ci_embed_np = np.array(ci_embed_sum).astype(np.float32)
        mdl_dtype = torch.bfloat16

        for i, req in enumerate(requests):
            out_torch = torch.from_numpy(output_ids_np[i:i + 1]).to(torch.long)

            req.input_tokens = torch.zeros(1, n_cb, dtype=torch.long)
            req.input_tokens[0, 0] = int(output_ids_np[i, 0])
            if not getattr(req, "is_input_streaming", False):
                req.input_tokens[0, -1] = self.model.config.tts_pad_token_id
            req.input_masks = torch.ones(n_cb, dtype=torch.bool)[None, :]
            req.input_features = torch.zeros(1, self.model.hidden_size, dtype=mdl_dtype)
            req.input_features[:] += torch.from_numpy(ci_embed_np[i:i + 1]).to(mdl_dtype)

            req.lm_output_tokens.append(out_torch)
            if int(output_ids_np[i, 0]) != self.model.stop_token_id:
                req.lm_output_audio_tokens.append(out_torch)
            else:
                req.done_lm_generation = True
                req.finish_reason = "stop_id_encountered"

        for req in requests:
            if req.next_position_id > self.model.max_tokens:
                req.done_lm_generation = True
                req.finish_reason = "max_tokens_reached"

    # ------------------------------------------------------------------
    # Depth (integrated)
    # ------------------------------------------------------------------

    def run_lm_depth(self, output_ids, hidden_for_depth, requests):
        raise NotImplementedError("Depth is fused into prefill/decode")

    # ------------------------------------------------------------------
    # Detokenization
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
                new_tokens = req.lm_output_audio_tokens[
                    decode_idx:decode_idx + self.detokenize_interval
                ]
                if len(new_tokens) < self.detokenize_interval:
                    new_tokens.extend(
                        [new_tokens[-1]] * (self.detokenize_interval - len(new_tokens))
                    )
                token_ids.append(torch.cat(new_tokens, dim=0))
                if req.decoder_cache is not None:
                    decoder_caches.append(req.decoder_cache)
                request_chunk_mapping.append((req_idx, chunk_idx))

        if not token_ids:
            return

        token_ids = torch.stack(token_ids, dim=0).to(device="cpu", dtype=torch.long)

        from vox_serve.tokenizer.base import DecoderCache

        if decoder_caches:
            batched_cache = DecoderCache.cat(decoder_caches)
            audio_tensors = self.model.postprocess(token_ids, decoder_cache=batched_cache)
            for i, (req_idx, _) in enumerate(request_chunk_mapping):
                requests[req_idx].decoder_cache.copy_from(batched_cache[i:i + 1])
        else:
            audio_tensors = self.model.postprocess(token_ids)

        for i, (req_idx, chunk_idx) in enumerate(request_chunk_mapping):
            req = requests[req_idx]
            decode_idx = req.audio_decode_idx[chunk_idx]
            audio = audio_tensors[i].detach().cpu().float().numpy()
            audio_int16 = (audio * 32767).astype(np.int16)

            last_chunk_len = len(
                req.lm_output_audio_tokens[decode_idx:decode_idx + self.detokenize_interval]
            )
            if last_chunk_len < self.detokenize_interval:
                trim_len = int(
                    audio_int16.shape[1] * (last_chunk_len - 0.5) / self.detokenize_interval
                )
                audio_int16 = audio_int16[:, :trim_len]

            req.output_audio.put(audio_int16.tobytes())

        for req in requests:
            if req.done_lm_generation and req.audio_decode_idx and (
                req.audio_decode_idx[-1] + self.detokenize_interval
                >= len(req.lm_output_audio_tokens)
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
    # NVTX stubs
    # ------------------------------------------------------------------

    def nvtx_range_push(self, name: str):
        pass

    def nvtx_range_pop(self):
        pass
