"""Paged and sequential attention for JAX TPU inference.

Uses the vLLM ragged paged attention (RPA) Pallas kernel for backbone
attention. The RPA kernel handles KV cache write + attention in a single
fused call, supporting both prefill and decode.

For the depth transformer (short sequences), uses tokamax XLA attention.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import tokamax

from .rpa import get_kv_cache_shape, ragged_paged_attention

# ---------------------------------------------------------------------------
# Sequential attention (for depth transformer — small sequences)
# ---------------------------------------------------------------------------

def sequential_attn_fn(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    kv_cache_k: jnp.ndarray,
    kv_cache_v: jnp.ndarray,
    seq_start: jnp.ndarray,
    seq_len_so_far: int,
    layer_idx: int,
    n_heads: int = 16,
    n_kv_heads: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sequential attention with causal mask via tokamax XLA.

    Used for the depth transformer where sequences are short (max ~17 tokens).

    KV cache format: (max_seq_len, n_kv_heads, head_dim)

    Args:
        q: (new_tokens, n_heads, head_dim)
        k: (new_tokens, n_kv_heads, head_dim)
        v: (new_tokens, n_kv_heads, head_dim)
        kv_cache_k: (max_seq_len, n_kv_heads, head_dim)
        kv_cache_v: (max_seq_len, n_kv_heads, head_dim)
        seq_start: scalar — where to write new tokens in cache
        seq_len_so_far: not used
        layer_idx: not used (cache is per-layer)
        n_heads: number of Q heads
        n_kv_heads: number of KV heads
    Returns:
        (attn_output, new_kv_cache_k, new_kv_cache_v)
        attn_output: (new_tokens, n_heads, head_dim)
    """
    new_len = q.shape[0]
    max_seq_len = kv_cache_k.shape[0]

    # Write new K/V into cache
    indices = jnp.arange(new_len) + seq_start
    new_kv_cache_k = kv_cache_k.at[indices].set(k)
    new_kv_cache_v = kv_cache_v.at[indices].set(v)

    total_len = seq_start + new_len

    # Use full cache with masking (fori_loop compatible)
    all_k = new_kv_cache_k  # (max_seq_len, n_kv_heads, head_dim)
    all_v = new_kv_cache_v

    # tokamax expects (batch, seq, heads, dim)
    q_4d = q[None, :, :, :]
    k_4d = all_k[None, :, :, :]
    v_4d = all_v[None, :, :, :]

    # Causal + validity mask
    q_positions = jnp.arange(new_len) + seq_start
    k_positions = jnp.arange(max_seq_len)
    causal_mask = q_positions[:, None] >= k_positions[None, :]
    valid_mask = k_positions[None, :] < total_len
    mask = (causal_mask & valid_mask)[None, None, :, :]

    # XLA for small sequences (depth transformer)
    attn_out = tokamax.dot_product_attention(
        q_4d, k_4d, v_4d,
        mask=mask,
        implementation="xla",
    )

    return attn_out[0], new_kv_cache_k, new_kv_cache_v


# ---------------------------------------------------------------------------
# RPA-based attention (for backbone — paged KV cache)
# ---------------------------------------------------------------------------

def rpa_attn_fn(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    kv_cache: jnp.ndarray,
    kv_lens: jnp.ndarray,
    page_indices: jnp.ndarray,
    cu_q_lens: jnp.ndarray,
    distribution: jnp.ndarray,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backbone attention using the vLLM ragged paged attention kernel.

    This handles KV cache write + attention in a single fused Pallas call.
    Supports mixed prefill + decode batches.

    Args:
        q: (total_tokens, n_heads, head_dim)
        k: (total_tokens, n_kv_heads, head_dim)
        v: (total_tokens, n_kv_heads, head_dim)
        kv_cache: (num_pages, page_size, n_kv_heads*2//packing, packing, head_dim)
        kv_lens: (max_num_seqs,) int32
        page_indices: (max_num_seqs * pages_per_seq,) int32
        cu_q_lens: (max_num_seqs + 1,) int32
        distribution: (3,) int32 — [num_decode, end_prefill, end_mixed]
        n_heads, n_kv_heads, head_dim: attention config
    Returns:
        (attn_output, updated_kv_cache)
        attn_output: (total_tokens, n_heads, head_dim)
    """
    sm_scale = 1.0 / math.sqrt(head_dim)

    output, updated_kv_cache = ragged_paged_attention(
        q, k, v, kv_cache,
        kv_lens, page_indices, cu_q_lens, distribution,
        sm_scale=sm_scale,
        use_causal_mask=True,
    )

    return output, updated_kv_cache


# ---------------------------------------------------------------------------
# KV cache allocation
# ---------------------------------------------------------------------------

def allocate_kv_cache(
    num_layers: int,
    num_pages: int,
    page_size: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> list[jnp.ndarray]:
    """Allocate merged paged KV cache for the backbone (RPA format).

    Returns:
        List of per-layer KV cache arrays, each shaped:
        (num_pages, page_size, n_kv_heads*2//packing, packing, head_dim)
    """
    shape = get_kv_cache_shape(num_pages, page_size, n_kv_heads, head_dim, dtype)
    return [jnp.zeros(shape, dtype=dtype) for _ in range(num_layers)]


def allocate_depth_kv_cache(
    num_layers: int,
    max_seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Allocate sequential KV cache for the depth transformer.

    Returns:
        (kv_cache_k, kv_cache_v) — each (batch_size, num_layers, max_seq_len, n_kv_heads, head_dim)
    """
    shape = (batch_size, num_layers, max_seq_len, n_kv_heads, head_dim)
    return jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype)
