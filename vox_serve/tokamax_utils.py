"""TPU attention wrappers for paged KV cache.

Provides drop-in replacements for the FlashInfer wrappers in flashinfer_utils.py,
implementing the same plan() / set_kv_cache() / run() interface so that model code
(e.g. qwen3_tts.py) can run on TPU without modification.

Both prefill and decode use pure PyTorch implementations that XLA compiles well
for TPU.  All hot-path methods (run/set_kv_cache) avoid .item()/.tolist() and
use bucketed tensor shapes so that XLA reuses cached HLO programs.
"""

import math
from dataclasses import dataclass
from typing import Union

import torch

from .utils import get_logger

try:
    import torch_xla  # noqa: F401

    HAS_TORCH_XLA = True
except ImportError:
    HAS_TORCH_XLA = False

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shape bucketing — limits the number of unique XLA compilations
# ---------------------------------------------------------------------------

_PAGE_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _bucket_pages(n: int) -> int:
    """Round *n* up to the next bucket value (power-of-2 friendly)."""
    for b in _PAGE_BUCKETS:
        if b >= n:
            return b
    return n


# ---------------------------------------------------------------------------
# KV cache containers  (separate K / V for tokamax)
# ---------------------------------------------------------------------------


@dataclass
class TPUKVCacheLayer:
    """Per-layer view into the TPU KV cache (separate K and V tensors).

    k: (num_pages, page_size, num_kv_heads, head_dim)
    v: (num_pages, page_size, num_kv_heads, head_dim)
    """

    k: torch.Tensor
    v: torch.Tensor


class TPUKVCache:
    """Indexable container of per-layer K/V cache tensors.

    Allocated once at worker startup.  The model iterates layers via
    ``kv_cache[layer_idx]``, which returns a ``TPUKVCacheLayer``.
    """

    def __init__(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        # k_cache, v_cache: (num_layers, num_pages, page_size, num_kv_heads, head_dim)
        self.k_cache = k_cache
        self.v_cache = v_cache

    def __getitem__(self, layer_idx: int) -> TPUKVCacheLayer:
        return TPUKVCacheLayer(k=self.k_cache[layer_idx], v=self.v_cache[layer_idx])


# ---------------------------------------------------------------------------
# Prefill wrapper  –  paged attention via vectorised gather + manual matmul
# ---------------------------------------------------------------------------


class TokmaxPrefillWrapper:
    """Paged prefill attention using pure PyTorch (XLA-friendly).

    Two code paths in ``run()``:
    * **uniform** (all requests have the same seq-len, e.g. depth transformer) —
      reshape only, no ``.item()``.
    * **non-uniform** (backbone prefill with different prompt lengths) —
      uses ``.item()`` in a for-loop, but this path only executes once per
      request so the one-time compilation cost is acceptable.
    """

    def __init__(
        self,
        n_qo_head: int,
        n_kv_head: int,
        n_state: int,
        page_size: int,
        device: torch.device,
    ):
        self.n_qo_head = n_qo_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_state // n_qo_head
        self.page_size = page_size
        self.device = device

        # Populated by plan()
        self.token_to_page: torch.Tensor | None = None
        self.token_to_cache: torch.Tensor | None = None
        self.qo_indptr: torch.Tensor | None = None
        self._n_req: int = 0
        self._total_tokens: int = 0

    # ---- plan ---------------------------------------------------------------

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        kv_len_budget: int | None = None,
    ) -> None:
        self.qo_indptr = qo_indptr
        n_req = qo_indptr.shape[0] - 1
        self._n_req = n_req

        # -- token-to-page / token-to-cache mapping (identical to FlashInfer) --
        starts = qo_indptr[:-1].to(torch.int32)
        lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)
        total_tokens = int(lens.sum().item())
        self._total_tokens = total_tokens

        num_pages_per_req = (paged_kv_indptr[1:] - paged_kv_indptr[:-1]).to(torch.int32)
        kv_lens = (num_pages_per_req - 1) * self.page_size + paged_kv_last_page_len

        seg = torch.repeat_interleave(torch.arange(n_req, dtype=torch.int32, device=self.device), lens)
        intra = torch.arange(total_tokens, dtype=torch.int32, device=self.device) - torch.repeat_interleave(
            starts, lens
        )

        start_new = kv_lens[seg] - lens[seg]
        g = start_new + intra

        page_off = torch.div(g, self.page_size, rounding_mode="floor").to(torch.int32)
        off_in_page = (g - page_off * self.page_size).to(torch.int32)
        abs_page_ptr = paged_kv_indptr[:-1][seg] + page_off

        self.token_to_page = paged_kv_indices[abs_page_ptr].to(self.device)
        self.token_to_cache = off_in_page.to(self.device)

        # -- precompute for vectorised run() ----------------------------------
        raw_max_pages = int(num_pages_per_req.max().item()) if n_req > 0 else 0
        raw_max_seq = int(lens.max().item()) if n_req > 0 else 0
        bucketed_max_pages = _bucket_pages(raw_max_pages)

        if kv_len_budget is not None:
            max_kv_len = kv_len_budget
        else:
            max_kv_len = bucketed_max_pages * self.page_size

        self._max_seq_len = raw_max_seq
        self._max_kv_len = max_kv_len
        self._bucketed_max_pages = bucketed_max_pages
        self._seq_lens = lens
        self._kv_lens = kv_lens

        # Build padded page-id tensor: (n_req, bucketed_max_pages)
        if n_req > 0 and bucketed_max_pages > 0:
            page_offsets = torch.arange(bucketed_max_pages, device=self.device, dtype=torch.int32)
            abs_indices = paged_kv_indptr[:-1].unsqueeze(1) + page_offsets.unsqueeze(0)  # (n_req, bmp)
            valid_mask = page_offsets.unsqueeze(0) < num_pages_per_req.unsqueeze(1)
            safe_max = max(paged_kv_indices.shape[0] - 1, 0)
            abs_indices = abs_indices.clamp(max=safe_max)
            self._page_ids = torch.where(
                valid_mask,
                paged_kv_indices[abs_indices.long()],
                torch.zeros(1, dtype=paged_kv_indices.dtype, device=self.device),
            )
        else:
            self._page_ids = torch.zeros(
                max(n_req, 1), max(bucketed_max_pages, 1), dtype=torch.int32, device=self.device,
            )

        # Store raw refs for backward compat
        self._paged_kv_indptr = paged_kv_indptr
        self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len

    def plan_compiled(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        total_tokens: int,
        max_seq_len: int,
        max_pages: int,
        kv_len_budget: int,
    ) -> None:
        """Plan without .item() — safe inside torch_xla.compile.

        All shape parameters are passed explicitly instead of derived from
        tensor values at trace time.
        """
        self.qo_indptr = qo_indptr
        n_req = qo_indptr.shape[0] - 1
        self._n_req = n_req

        starts = qo_indptr[:-1].to(torch.int32)
        lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)
        self._total_tokens = total_tokens

        num_pages_per_req = (paged_kv_indptr[1:] - paged_kv_indptr[:-1]).to(torch.int32)
        kv_lens = (num_pages_per_req - 1) * self.page_size + paged_kv_last_page_len

        seg = torch.repeat_interleave(torch.arange(n_req, dtype=torch.int32, device=self.device), lens)
        intra = torch.arange(total_tokens, dtype=torch.int32, device=self.device) - torch.repeat_interleave(
            starts, lens
        )

        start_new = kv_lens[seg] - lens[seg]
        g = start_new + intra

        page_off = torch.div(g, self.page_size, rounding_mode="floor").to(torch.int32)
        off_in_page = (g - page_off * self.page_size).to(torch.int32)
        abs_page_ptr = paged_kv_indptr[:-1][seg] + page_off

        self.token_to_page = paged_kv_indices[abs_page_ptr].to(self.device)
        self.token_to_cache = off_in_page.to(self.device)

        # Vectorised run() precomputation
        self._max_seq_len = max_seq_len
        self._max_kv_len = kv_len_budget
        self._bucketed_max_pages = max_pages
        self._seq_lens = lens
        self._kv_lens = kv_lens

        page_offsets = torch.arange(max_pages, device=self.device, dtype=torch.int32)
        abs_indices = paged_kv_indptr[:-1].unsqueeze(1) + page_offsets.unsqueeze(0)
        valid_mask = page_offsets.unsqueeze(0) < num_pages_per_req.unsqueeze(1)
        abs_indices = abs_indices.clamp(max=paged_kv_indices.shape[0] - 1)
        self._page_ids = torch.where(
            valid_mask,
            paged_kv_indices[abs_indices.long()],
            torch.zeros(1, dtype=paged_kv_indices.dtype, device=self.device),
        )

        self._paged_kv_indptr = paged_kv_indptr
        self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len

    # ---- set_kv_cache -------------------------------------------------------

    def set_kv_cache(self, kv_cache: TPUKVCacheLayer, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write K/V into separate cache tensors (vectorised scatter)."""
        page_idx = self.token_to_page
        cache_idx = self.token_to_cache
        kv_cache.k[page_idx, cache_idx] = k
        kv_cache.v[page_idx, cache_idx] = v

    # ---- run ----------------------------------------------------------------

    def run(self, q: torch.Tensor, kv_cache: TPUKVCacheLayer) -> torch.Tensor:
        """Run paged prefill attention.

        q: (total_tokens, n_qo_heads, head_dim)
        Returns: (total_tokens, n_qo_heads, head_dim)
        """
        n_req = self._n_req
        if n_req == 0:
            return torch.zeros(0, self.n_qo_head, self.head_dim, dtype=q.dtype, device=self.device)

        max_seq_len = self._max_seq_len
        max_kv_len = self._max_kv_len
        seq_lens = self._seq_lens       # tensor (n_req,)
        kv_lens = self._kv_lens         # tensor (n_req,)
        page_ids = self._page_ids       # (n_req, bucketed_max_pages)
        total_tokens = q.shape[0]

        # ---- build padded Q: (n_req, max_seq_len, heads, D) ----------------
        uniform = total_tokens == n_req * max_seq_len
        if uniform:
            # Fast path — all requests have the same seq len (depth transformer)
            q_batched = q.reshape(n_req, max_seq_len, self.n_qo_head, self.head_dim)
        else:
            # Non-uniform (backbone prefill, runs once) — for-loop is fine
            q_batched = torch.zeros(
                n_req, max_seq_len, self.n_qo_head, self.head_dim, dtype=q.dtype, device=self.device,
            )
            qo_indptr = self.qo_indptr
            for i in range(n_req):
                s = int(qo_indptr[i].item())
                e = int(qo_indptr[i + 1].item())
                q_batched[i, : e - s] = q[s:e]

        # ---- gather KV pages: (n_req, flat_kv_len, heads, D) ---------------
        k_pages = kv_cache.k[page_ids.long()]   # (n_req, bmp, page_dim, kv_heads, D)
        v_pages = kv_cache.v[page_ids.long()]

        kv_flat_len = page_ids.shape[1] * kv_cache.k.shape[1]  # bmp * actual page dim
        k_gathered = k_pages.reshape(n_req, kv_flat_len, self.n_kv_head, self.head_dim)
        v_gathered = v_pages.reshape(n_req, kv_flat_len, self.n_kv_head, self.head_dim)

        # Trim to max_kv_len (when kv_len_budget < bucketed pages * page_size)
        if kv_flat_len > max_kv_len:
            k_gathered = k_gathered[:, :max_kv_len]
            v_gathered = v_gathered[:, :max_kv_len]
            eff_kv_len = max_kv_len
        else:
            eff_kv_len = kv_flat_len

        # ---- build causal attention mask: (n_req, max_seq_len, eff_kv_len) --
        q_pos = torch.arange(max_seq_len, device=self.device)
        kv_pos = torch.arange(eff_kv_len, device=self.device)
        kv_offsets = kv_lens - seq_lens   # (n_req,)

        causal = kv_pos.view(1, 1, -1) <= (kv_offsets.view(-1, 1, 1) + q_pos.view(1, -1, 1))
        valid_kv = kv_pos.view(1, 1, -1) < kv_lens.view(-1, 1, 1)
        valid_q = q_pos.view(1, -1, 1) < seq_lens.view(-1, 1, 1)
        attn_mask = causal & valid_kv & valid_q   # (n_req, max_seq_len, eff_kv_len)

        # ---- attention ------------------------------------------------------
        q_4d = q_batched.transpose(1, 2)     # (n_req, heads, max_seq_len, D)
        k_4d = k_gathered.transpose(1, 2)    # (n_req, kv_heads, eff_kv_len, D)
        v_4d = v_gathered.transpose(1, 2)

        if self.n_qo_head != self.n_kv_head:
            n_groups = self.n_qo_head // self.n_kv_head
            k_4d = k_4d.repeat_interleave(n_groups, dim=1)
            v_4d = v_4d.repeat_interleave(n_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q_4d, k_4d.transpose(-2, -1)) * scale
        attn_mask_4d = attn_mask.unsqueeze(1)  # (n_req, 1, max_seq_len, eff_kv_len)
        attn_weights = attn_weights.masked_fill(~attn_mask_4d, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)  # padded Q rows → all -inf → nan
        out_4d = torch.matmul(attn_weights, v_4d)

        out_batched = out_4d.transpose(1, 2)  # (n_req, max_seq_len, heads, D)

        # ---- extract valid tokens ------------------------------------------
        if uniform:
            return out_batched.reshape(total_tokens, self.n_qo_head, self.head_dim)
        else:
            out = torch.zeros(total_tokens, self.n_qo_head, self.head_dim, dtype=q.dtype, device=self.device)
            qo_indptr = self.qo_indptr
            for i in range(n_req):
                s = int(qo_indptr[i].item())
                e = int(qo_indptr[i + 1].item())
                out[s:e] = out_batched[i, : e - s]
            return out


# ---------------------------------------------------------------------------
# Decode wrapper  –  vectorised gather-and-matmul (zero .item() in run())
# ---------------------------------------------------------------------------


class TokmaxDecodeWrapper:
    """Paged decode attention using vectorised PyTorch operations.

    ``plan()`` precomputes page-id and mask tensors with bucketed shapes.
    ``run()`` uses only these precomputed tensors — no ``.item()`` calls —
    so XLA reuses cached HLO programs across decode steps.

    For XLA graph reuse, call ``allocate_buffers()`` once per batch-size
    bucket during warmup. ``plan()`` will then write into those fixed
    tensor objects via ``copy_()`` instead of creating new ones.
    """

    def __init__(
        self,
        n_qo_head: int,
        n_kv_head: int,
        n_state: int,
        page_size: int,
        device: torch.device,
    ):
        self.n_qo_head = n_qo_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_state // n_qo_head
        self.n_gqa_groups = n_qo_head // n_kv_head
        self.page_size = page_size
        self.device = device
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Populated by plan()
        self.kv_cache_locations: torch.Tensor | None = None
        self.batch_size: int = 0

        # Pre-allocated output buffers keyed by (bs, bucketed_max_pages).
        # When set, plan() writes into these via copy_() so XLA always
        # sees the same tensor objects → guaranteed graph cache hit.
        self._output_bufs: dict | None = None

    def allocate_buffers(self, bs: int, max_pages_bucket: int) -> None:
        """Pre-allocate plan output buffers for a specific batch-size bucket."""
        if self._output_bufs is None:
            self._output_bufs = {}
        max_kv_len = max_pages_bucket * self.page_size
        self._output_bufs[bs] = {
            "kv_cache_locations": torch.zeros(bs, 2, dtype=torch.int32, device=self.device),
            "page_ids": torch.zeros(bs, max_pages_bucket, dtype=torch.int32, device=self.device),
            "attn_mask": torch.zeros(bs, max_kv_len, dtype=torch.bool, device=self.device),
            "max_kv_len": max_kv_len,
            "bucketed_max_pages": max_pages_bucket,
        }

    # ---- plan ---------------------------------------------------------------

    def plan(
        self,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Plan with dynamic bucketing (uses .item() — NOT safe inside torch_xla.compile)."""
        bs = paged_kv_indptr.shape[0] - 1
        self.batch_size = bs

        num_pages_per_req = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        raw_max_pages = int(num_pages_per_req.max().item()) if bs > 0 else 0
        bucketed_max_pages = _bucket_pages(raw_max_pages)

        self.plan_compiled(paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, bucketed_max_pages)

    def plan_compiled(
        self,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        max_pages: int,
    ) -> None:
        """Plan with fixed max_pages — no .item(), safe inside torch_xla.compile.

        All outputs are written into pre-allocated buffers (if ``allocate_buffers``
        was called) so the compiled graph always operates on the same tensor objects.
        """
        bs = paged_kv_indptr.shape[0] - 1
        self.batch_size = bs

        # KV cache write locations
        page_idx = paged_kv_indices[paged_kv_indptr[1:] - 1]
        pos_idx = paged_kv_last_page_len - 1
        kv_locs = torch.stack([page_idx, pos_idx], dim=1)

        # KV lengths
        num_pages_per_req = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        kv_lens = (num_pages_per_req - 1) * self.page_size + paged_kv_last_page_len

        max_kv_len = max_pages * self.page_size

        # Build page_ids: (bs, max_pages)
        page_offsets = torch.arange(max_pages, device=self.device, dtype=torch.int32)
        abs_indices = paged_kv_indptr[:-1].unsqueeze(1) + page_offsets.unsqueeze(0)
        valid_mask = page_offsets.unsqueeze(0) < num_pages_per_req.unsqueeze(1)
        abs_indices = abs_indices.clamp(max=paged_kv_indices.shape[0] - 1)
        page_ids = torch.where(
            valid_mask,
            paged_kv_indices[abs_indices.long()],
            torch.zeros(1, dtype=paged_kv_indices.dtype, device=self.device),
        )

        # Build attn_mask: (bs, max_kv_len)
        pos_range = torch.arange(max_kv_len, device=self.device, dtype=torch.int32)
        attn_mask = pos_range.unsqueeze(0) < kv_lens.unsqueeze(1)

        # Write into pre-allocated buffers if available
        buf = self._output_bufs.get(bs) if self._output_bufs else None
        if buf is not None and buf["bucketed_max_pages"] >= max_pages:
            buf["kv_cache_locations"].copy_(kv_locs)
            buf["page_ids"][:, :max_pages].copy_(page_ids)
            buf["attn_mask"][:, :max_kv_len].copy_(attn_mask)
            self.kv_cache_locations = buf["kv_cache_locations"]
            self._page_ids = buf["page_ids"]
            self._attn_mask = buf["attn_mask"]
            self._max_kv_len = buf["max_kv_len"]
        else:
            self.kv_cache_locations = kv_locs.to(self.device)
            self._page_ids = page_ids
            self._attn_mask = attn_mask
            self._max_kv_len = max_kv_len

    # ---- set_kv_cache -------------------------------------------------------

    def set_kv_cache(self, kv_cache: TPUKVCacheLayer, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write single token K/V per request into cache."""
        pages = self.kv_cache_locations[: self.batch_size, 0].long()
        positions = self.kv_cache_locations[: self.batch_size, 1].long()
        kv_cache.k[pages, positions] = k
        kv_cache.v[pages, positions] = v

    # ---- run ----------------------------------------------------------------

    def run(self, q: torch.Tensor, kv_cache: TPUKVCacheLayer) -> torch.Tensor:
        """Paged decode attention — fully vectorised, zero .item() calls.

        q: (batch_size, n_qo_heads, head_dim)
        Returns: (batch_size, n_qo_heads, head_dim)
        """
        bs = self.batch_size
        page_ids = self._page_ids       # (bs, bucketed_max_pages)
        max_kv_len = self._max_kv_len
        attn_mask = self._attn_mask     # (bs, max_kv_len)

        # Gather all KV pages at once: (bs, bmp, page_size, kv_heads, D)
        k_pages = kv_cache.k[page_ids.long()]
        v_pages = kv_cache.v[page_ids.long()]

        # Reshape to (bs, max_kv_len, kv_heads, D)
        k_flat = k_pages.reshape(bs, max_kv_len, self.n_kv_head, self.head_dim)
        v_flat = v_pages.reshape(bs, max_kv_len, self.n_kv_head, self.head_dim)

        # q: (bs, heads, 1, D)
        q_4d = q.unsqueeze(2)

        # k, v → (bs, kv_heads, max_kv_len, D)
        k_4d = k_flat.transpose(1, 2)
        v_4d = v_flat.transpose(1, 2)

        # GQA head expansion
        if self.n_gqa_groups > 1:
            k_4d = k_4d.repeat_interleave(self.n_gqa_groups, dim=1)
            v_4d = v_4d.repeat_interleave(self.n_gqa_groups, dim=1)

        # Attention scores: (bs, heads, 1, max_kv_len)
        attn_weights = torch.matmul(q_4d, k_4d.transpose(-2, -1)) * self.scale

        # Mask padding positions
        mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, max_kv_len)
        attn_weights = attn_weights.masked_fill(~mask_4d, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # (bs, heads, 1, D) → (bs, heads, D)
        out = torch.matmul(attn_weights, v_4d).squeeze(2)
        return out


# ---------------------------------------------------------------------------
# Union type matching FlashInferWrapper
# ---------------------------------------------------------------------------

TokmaxWrapper = Union[TokmaxPrefillWrapper, TokmaxDecodeWrapper]
