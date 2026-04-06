"""TPU attention wrappers for paged KV cache.

Provides drop-in replacements for the FlashInfer wrappers in flashinfer_utils.py,
implementing the same plan() / set_kv_cache() / run() interface so that model code
(e.g. qwen3_tts.py) can run on TPU without modification.

Both prefill and decode use pure PyTorch implementations that XLA compiles well
for TPU. The paging logic (page allocation, token-to-page mapping) mirrors
FlashInfer's approach with separate K/V cache tensors.
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
# Prefill wrapper  –  paged attention via gather + scaled_dot_product_attention
# ---------------------------------------------------------------------------


class TokmaxPrefillWrapper:
    """Paged prefill attention backed by ``tokamax.dot_product_attention``."""

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
        self._jax_paging_info = None
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
    ) -> None:
        self.qo_indptr = qo_indptr
        n_req = qo_indptr.shape[0] - 1
        self._n_req = n_req

        # -- token-to-page / token-to-cache mapping (identical to FlashInfer) --
        starts = qo_indptr[:-1].to(torch.int32)
        lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)
        total_tokens = int(lens.sum().item())
        self._total_tokens = total_tokens

        num_pages_after = (paged_kv_indptr[1:] - paged_kv_indptr[:-1]).to(torch.int32)
        kv_len_after = (num_pages_after - 1) * self.page_size + paged_kv_last_page_len

        seg = torch.repeat_interleave(torch.arange(n_req, dtype=torch.int32, device=self.device), lens)
        intra = torch.arange(total_tokens, dtype=torch.int32, device=self.device) - torch.repeat_interleave(
            starts, lens
        )

        start_new = kv_len_after[seg] - lens[seg]
        g = start_new + intra

        page_off = torch.div(g, self.page_size, rounding_mode="floor").to(torch.int32)
        off_in_page = (g - page_off * self.page_size).to(torch.int32)
        abs_page_ptr = paged_kv_indptr[:-1][seg] + page_off

        self.token_to_page = paged_kv_indices[abs_page_ptr].to(self.device)
        self.token_to_cache = off_in_page.to(self.device)

        # -- build tokamax PagingInfo ------------------------------------------
        num_active_pages = num_pages_after  # (n_req,)
        max_pages = int(num_active_pages.max().item()) if n_req > 0 else 0

        # Build per-request active_page_indices (n_req, max_pages)
        active_page_indices = torch.zeros(n_req, max_pages, dtype=torch.int32, device=self.device)
        for i in range(n_req):
            start_idx = int(paged_kv_indptr[i].item())
            end_idx = int(paged_kv_indptr[i + 1].item())
            n_pages_i = end_idx - start_idx
            active_page_indices[i, :n_pages_i] = paged_kv_indices[start_idx:end_idx]

        # Store paging metadata for run() to gather KV pages
        self._paged_kv_indptr = paged_kv_indptr
        self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len

    # ---- set_kv_cache -------------------------------------------------------

    def set_kv_cache(self, kv_cache: TPUKVCacheLayer, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write K/V into separate cache tensors.

        kv_cache: TPUKVCacheLayer (per-layer)
        k, v: (n_token, n_kv_heads, head_dim)
        """
        page_idx = self.token_to_page
        cache_idx = self.token_to_cache
        kv_cache.k[page_idx, cache_idx] = k
        kv_cache.v[page_idx, cache_idx] = v

    # ---- run ----------------------------------------------------------------

    def run(self, q: torch.Tensor, kv_cache: TPUKVCacheLayer) -> torch.Tensor:
        """Run paged prefill attention via gather + scaled_dot_product_attention.

        q: (total_tokens, n_qo_heads, head_dim)
        kv_cache: TPUKVCacheLayer with k, v of shape (num_pages, page_size, n_kv_heads, head_dim)
        Returns: (total_tokens, n_qo_heads, head_dim)
        """
        n_req = self._n_req
        qo_indptr = self.qo_indptr
        indptr = self._paged_kv_indptr
        indices = self._paged_kv_indices
        last_page_len = self._paged_kv_last_page_len

        seq_lens = (qo_indptr[1:] - qo_indptr[:-1]).tolist()
        max_seq_len = max(seq_lens) if seq_lens else 0

        # Determine max KV length across the batch
        num_pages_per_req = (indptr[1:] - indptr[:-1]).to(torch.int32)
        max_kv_len = int(((num_pages_per_req - 1) * self.page_size + last_page_len).max().item())

        # Gather KV into contiguous batched tensors and build per-request Q
        q_batched = torch.zeros(
            n_req, max_seq_len, self.n_qo_head, self.head_dim, dtype=q.dtype, device=self.device
        )
        k_gathered = torch.zeros(
            n_req, max_kv_len, self.n_kv_head, self.head_dim, dtype=q.dtype, device=self.device
        )
        v_gathered = torch.zeros(
            n_req, max_kv_len, self.n_kv_head, self.head_dim, dtype=q.dtype, device=self.device
        )

        for i in range(n_req):
            # Fill Q
            q_start = int(qo_indptr[i].item())
            q_end = int(qo_indptr[i + 1].item())
            q_len = q_end - q_start
            q_batched[i, :q_len] = q[q_start:q_end]

            # Gather KV pages
            kv_start = int(indptr[i].item())
            kv_end = int(indptr[i + 1].item())
            page_ids = indices[kv_start:kv_end]
            n_pages = kv_end - kv_start
            last_len = int(last_page_len[i].item())

            for p_idx in range(n_pages - 1):
                dst_start = p_idx * self.page_size
                dst_end = dst_start + self.page_size
                pid = page_ids[p_idx]
                k_gathered[i, dst_start:dst_end] = kv_cache.k[pid, : self.page_size]
                v_gathered[i, dst_start:dst_end] = kv_cache.v[pid, : self.page_size]

            if n_pages > 0:
                dst_start = (n_pages - 1) * self.page_size
                pid = page_ids[n_pages - 1]
                k_gathered[i, dst_start : dst_start + last_len] = kv_cache.k[pid, :last_len]
                v_gathered[i, dst_start : dst_start + last_len] = kv_cache.v[pid, :last_len]

        # Transpose to (batch, heads, seq_len, head_dim) for SDPA
        q_4d = q_batched.transpose(1, 2)  # (n_req, n_qo_heads, max_seq_len, D)
        k_4d = k_gathered.transpose(1, 2)  # (n_req, n_kv_heads, max_kv_len, D)
        v_4d = v_gathered.transpose(1, 2)

        # Repeat KV heads for GQA
        if self.n_qo_head != self.n_kv_head:
            n_groups = self.n_qo_head // self.n_kv_head
            k_4d = k_4d.repeat_interleave(n_groups, dim=1)
            v_4d = v_4d.repeat_interleave(n_groups, dim=1)

        # Use scaled_dot_product_attention with causal mask
        out_4d = torch.nn.functional.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, is_causal=True,
        )  # (n_req, n_qo_heads, max_seq_len, D)

        # Transpose back and flatten to (total_tokens, n_qo_heads, D)
        out_batched = out_4d.transpose(1, 2)  # (n_req, max_seq_len, n_qo_heads, D)

        out = torch.zeros(self._total_tokens, self.n_qo_head, self.head_dim, dtype=q.dtype, device=self.device)
        for i in range(n_req):
            q_start = int(qo_indptr[i].item())
            q_end = int(qo_indptr[i + 1].item())
            q_len = q_end - q_start
            out[q_start:q_end] = out_batched[i, :q_len]

        return out


# ---------------------------------------------------------------------------
# Decode wrapper  –  simple PyTorch gather-and-matmul (XLA compiles well)
# ---------------------------------------------------------------------------


class TokmaxDecodeWrapper:
    """Paged decode attention using standard PyTorch operations.

    For decode (one token per request), a simple gather-matmul is efficient
    and avoids tokamax's xla_chunked seq_len alignment constraints.
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
        # For gathering KV pages per request
        self._paged_kv_indptr: torch.Tensor | None = None
        self._paged_kv_indices: torch.Tensor | None = None
        self._paged_kv_last_page_len: torch.Tensor | None = None

    # ---- plan ---------------------------------------------------------------

    def plan(
        self,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.batch_size = paged_kv_indptr.shape[0] - 1
        self._paged_kv_indptr = paged_kv_indptr
        self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len

        # Compute KV cache write locations for decode (same as FlashInfer decode wrapper)
        page_idx = paged_kv_indices[paged_kv_indptr[1:] - 1]
        pos_idx = paged_kv_last_page_len - 1
        self.kv_cache_locations = torch.stack([page_idx, pos_idx], dim=1).to(self.device)

    # ---- set_kv_cache -------------------------------------------------------

    def set_kv_cache(self, kv_cache: TPUKVCacheLayer, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write single token K/V per request into cache.

        kv_cache: TPUKVCacheLayer (per-layer)
        k, v: (batch_size, n_kv_heads, head_dim)
        """
        pages = self.kv_cache_locations[: self.batch_size, 0].long()
        positions = self.kv_cache_locations[: self.batch_size, 1].long()
        kv_cache.k[pages, positions] = k
        kv_cache.v[pages, positions] = v

    # ---- run ----------------------------------------------------------------

    def run(self, q: torch.Tensor, kv_cache: TPUKVCacheLayer) -> torch.Tensor:
        """Paged decode attention via gather + matmul.

        q: (batch_size, n_qo_heads, head_dim)
        kv_cache: TPUKVCacheLayer with k, v of shape (num_pages, page_size, n_kv_heads, head_dim)
        Returns: (batch_size, n_qo_heads, head_dim)
        """
        bs = self.batch_size
        indptr = self._paged_kv_indptr
        indices = self._paged_kv_indices
        last_page_len = self._paged_kv_last_page_len

        # Determine max KV length across the batch for padding
        num_pages_per_req = (indptr[1:] - indptr[:-1]).to(torch.int32)  # (bs,)
        max_kv_len = int(((num_pages_per_req - 1) * self.page_size + last_page_len).max().item())

        # Gather KV into contiguous tensors: (bs, max_kv_len, n_kv_heads, head_dim)
        k_gathered = torch.zeros(bs, max_kv_len, self.n_kv_head, self.head_dim, dtype=q.dtype, device=self.device)
        v_gathered = torch.zeros(bs, max_kv_len, self.n_kv_head, self.head_dim, dtype=q.dtype, device=self.device)

        # Build attention mask for variable KV lengths
        attn_mask = torch.zeros(bs, max_kv_len, dtype=q.dtype, device=self.device)

        for i in range(bs):
            start = int(indptr[i].item())
            end = int(indptr[i + 1].item())
            page_ids = indices[start:end]
            n_pages = end - start
            last_len = int(last_page_len[i].item())
            kv_len = (n_pages - 1) * self.page_size + last_len

            # Gather full pages
            for p_idx in range(n_pages - 1):
                dst_start = p_idx * self.page_size
                dst_end = dst_start + self.page_size
                pid = page_ids[p_idx]
                k_gathered[i, dst_start:dst_end] = kv_cache.k[pid, : self.page_size]
                v_gathered[i, dst_start:dst_end] = kv_cache.v[pid, : self.page_size]

            # Gather last (partial) page
            if n_pages > 0:
                dst_start = (n_pages - 1) * self.page_size
                pid = page_ids[n_pages - 1]
                k_gathered[i, dst_start : dst_start + last_len] = kv_cache.k[pid, :last_len]
                v_gathered[i, dst_start : dst_start + last_len] = kv_cache.v[pid, :last_len]

            attn_mask[i, :kv_len] = 1.0

        # Standard attention: Q @ K^T / sqrt(d) -> softmax -> @ V
        # q: (bs, n_qo_heads, head_dim) -> (bs, n_qo_heads, 1, head_dim)
        q_4d = q.unsqueeze(2)

        # Repeat KV heads for GQA: (bs, max_kv_len, n_kv_heads, D) -> (bs, n_qo_heads, max_kv_len, D)
        k_4d = k_gathered.transpose(1, 2)  # (bs, n_kv_heads, max_kv_len, D)
        v_4d = v_gathered.transpose(1, 2)  # (bs, n_kv_heads, max_kv_len, D)
        if self.n_gqa_groups > 1:
            k_4d = k_4d.repeat_interleave(self.n_gqa_groups, dim=1)  # (bs, n_qo_heads, max_kv_len, D)
            v_4d = v_4d.repeat_interleave(self.n_gqa_groups, dim=1)

        # (bs, n_qo_heads, 1, max_kv_len)
        attn_weights = torch.matmul(q_4d, k_4d.transpose(-2, -1)) * self.scale

        # Apply mask: set padding positions to -inf
        # attn_mask: (bs, max_kv_len) -> (bs, 1, 1, max_kv_len)
        mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights.masked_fill(mask_4d == 0, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # (bs, n_qo_heads, 1, head_dim) -> (bs, n_qo_heads, head_dim)
        out = torch.matmul(attn_weights, v_4d).squeeze(2)
        return out


# ---------------------------------------------------------------------------
# Union type matching FlashInferWrapper
# ---------------------------------------------------------------------------

TokmaxWrapper = Union[TokmaxPrefillWrapper, TokmaxDecodeWrapper]
