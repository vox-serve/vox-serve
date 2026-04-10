"""On-device sampling for JAX TPU inference.

All functions are pure and JIT-compatible — no .item() calls, no host transfers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def greedy_sample(logits: jnp.ndarray, rng_key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Greedy (argmax) sampling.

    Args:
        logits: (batch_size, vocab_size)
        rng_key: unused, passed through for API consistency
    Returns:
        (token_ids, rng_key) — token_ids: (batch_size,) int32
    """
    return jnp.argmax(logits, axis=-1).astype(jnp.int32), rng_key


def top_k_sample(
    logits: jnp.ndarray,
    rng_key: jnp.ndarray,
    k: int = 50,
    temperature: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-k sampling.

    Args:
        logits: (batch_size, vocab_size)
        rng_key: PRNG key
        k: number of top candidates
        temperature: sampling temperature
    Returns:
        (token_ids, new_rng_key) — token_ids: (batch_size,) int32
    """
    rng_key, sample_key = jax.random.split(rng_key)
    logits = logits / temperature

    # Get top-k values and indices
    top_values, top_indices = jax.lax.top_k(logits, k)  # (bs, k)

    # Sample from top-k
    sampled_idx = jax.random.categorical(sample_key, top_values, axis=-1)  # (bs,)

    # Map back to vocab indices
    token_ids = jnp.take_along_axis(
        top_indices, sampled_idx[:, None], axis=-1
    ).squeeze(-1)

    return token_ids.astype(jnp.int32), rng_key


def top_p_sample(
    logits: jnp.ndarray,
    rng_key: jnp.ndarray,
    p: float = 0.9,
    temperature: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-p (nucleus) sampling.

    Args:
        logits: (batch_size, vocab_size)
        rng_key: PRNG key
        p: cumulative probability threshold
        temperature: sampling temperature
    Returns:
        (token_ids, new_rng_key) — token_ids: (batch_size,) int32
    """
    rng_key, sample_key = jax.random.split(rng_key)
    logits = logits / temperature

    # Sort descending
    sorted_indices = jnp.argsort(-logits, axis=-1)  # (bs, vocab)
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)

    # Cumulative softmax probabilities
    sorted_probs = jax.nn.softmax(sorted_logits.astype(jnp.float32), axis=-1)
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

    # Mask tokens beyond threshold (keep at least the top-1)
    mask = cumulative_probs - sorted_probs > p  # shift by one to include the boundary token
    sorted_logits = jnp.where(mask, jnp.finfo(logits.dtype).min, sorted_logits)

    # Sample
    sampled_idx = jax.random.categorical(sample_key, sorted_logits, axis=-1)  # (bs,)
    token_ids = jnp.take_along_axis(
        sorted_indices, sampled_idx[:, None], axis=-1
    ).squeeze(-1)

    return token_ids.astype(jnp.int32), rng_key


def apply_repetition_penalty(
    logits: jnp.ndarray,
    repetition_cache: jnp.ndarray,
    penalty: float = 1.0,
) -> jnp.ndarray:
    """Apply repetition penalty to logits.

    Args:
        logits: (batch_size, vocab_size)
        repetition_cache: (batch_size, window_size) int32 — recent token IDs
        penalty: repetition penalty factor (1.0 = no penalty)
    Returns:
        penalized logits: (batch_size, vocab_size)
    """
    if penalty == 1.0:
        return logits

    # Gather logits at cached positions
    # repetition_cache: (bs, window) → gather from logits
    gathered = jnp.take_along_axis(logits, repetition_cache, axis=-1)  # (bs, window)

    # Apply penalty: divide positive logits, multiply negative logits
    penalized = jnp.where(gathered > 0, gathered / penalty, gathered * penalty)

    # Scatter back
    logits = logits.at[
        jnp.arange(logits.shape[0])[:, None],
        repetition_cache
    ].set(penalized)

    return logits


def update_repetition_cache(
    cache: jnp.ndarray,
    new_ids: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Shift repetition cache and append new token IDs.

    Args:
        cache: (batch_size, window_size) int32
        new_ids: (batch_size,) or (batch_size, 1) int32
        window_size: cache window size
    Returns:
        updated cache: (batch_size, window_size)
    """
    new_ids = new_ids.reshape(-1, 1)
    return jnp.concatenate([cache[:, 1:], new_ids], axis=-1)
