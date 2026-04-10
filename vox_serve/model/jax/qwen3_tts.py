"""Pure JAX model functions for Qwen3-TTS inference.

All functions are stateless (pure) and designed to work inside jax.jit.
Linear weights are stored as (in_features, out_features) — the JAX convention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .attention import sequential_attn_fn
from .params import Qwen3TTSConfig

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """RMSNorm: x * rsqrt(mean(x², -1) + eps) * weight."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    x_normed = x * jax.lax.rsqrt(variance + eps)
    return (x_normed * weight).astype(x.dtype)


def silu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.silu(x)


def linear(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray | None = None) -> jnp.ndarray:
    """Linear projection: x @ weight [+ bias]. Weight shape: (in, out)."""
    y = jnp.dot(x, weight)
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# RoPE (non-interleaved, matches FlashInfer / HuggingFace convention)
# ---------------------------------------------------------------------------

def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Negate-and-swap the two halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(
    q: jnp.ndarray,
    k: jnp.ndarray,
    positions: jnp.ndarray,
    rope_theta: float = 1_000_000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE to query and key tensors.

    Args:
        q: (seq_len, n_heads, head_dim)
        k: (seq_len, n_kv_heads, head_dim)
        positions: (seq_len,) int32
        rope_theta: base frequency
    Returns:
        (q_rotated, k_rotated) with same shapes
    """
    head_dim = q.shape[-1]
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    # positions: (seq_len,) → freqs: (seq_len, head_dim/2)
    freqs = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos = jnp.cos(freqs)[:, None, :]  # (seq_len, 1, head_dim/2)
    sin = jnp.sin(freqs)[:, None, :]
    # Expand to full head_dim
    cos = jnp.concatenate([cos, cos], axis=-1)  # (seq_len, 1, head_dim)
    sin = jnp.concatenate([sin, sin], axis=-1)

    q_out = (q * cos + _rotate_half(q) * sin).astype(q.dtype)
    k_out = (k * cos + _rotate_half(k) * sin).astype(k.dtype)
    return q_out, k_out


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def mlp(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """SiLU-gated MLP: down(silu(gate(x)) * up(x))."""
    gate = linear(x, params["gate_proj"])
    up = linear(x, params["up_proj"])
    return linear(silu(gate) * up, params["down_proj"])


# ---------------------------------------------------------------------------
# Attention (non-paged, for simplicity — paged version in attention.py)
# ---------------------------------------------------------------------------

def attention_proj(
    layer_params: dict,
    x: jnp.ndarray,
    positions: jnp.ndarray,
    config: Qwen3TTSConfig,
    is_backbone: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project x to q, k, v and apply RoPE + QK norms.

    Args:
        layer_params: params["layers"][i]["attn"]
        x: (seq_len, hidden)
        positions: (seq_len,)
    Returns:
        q: (seq_len, n_heads, head_dim)
        k: (seq_len, n_kv_heads, head_dim)
        v: (seq_len, n_kv_heads, head_dim)
    """
    attn = layer_params
    n_heads = config.backbone_heads if is_backbone else config.depth_heads
    n_kv_heads = config.backbone_kv_heads if is_backbone else config.depth_kv_heads
    head_dim = config.head_dim
    rope_theta = config.rope_theta

    q = linear(x, attn["q_proj"])  # (seq_len, n_heads * head_dim)
    k = linear(x, attn["k_proj"])  # (seq_len, n_kv_heads * head_dim)
    v = linear(x, attn["v_proj"])  # (seq_len, n_kv_heads * head_dim)

    # Reshape for per-head norm
    q = q.reshape(-1, head_dim)
    k = k.reshape(-1, head_dim)
    q = rms_norm(q, attn["q_norm"], config.rms_norm_eps)
    k = rms_norm(k, attn["k_norm"], config.rms_norm_eps)

    # Reshape to (seq_len, n_heads, head_dim)
    seq_len = x.shape[0]
    q = q.reshape(seq_len, n_heads, head_dim)
    k = k.reshape(seq_len, n_kv_heads, head_dim)
    v = v.reshape(seq_len, n_kv_heads, head_dim)

    # Apply RoPE
    q, k = apply_rope(q, k, positions, rope_theta)

    return q, k, v


def attention_output(attn_params: dict, attn_out: jnp.ndarray) -> jnp.ndarray:
    """Apply output projection. attn_out: (seq_len, n_heads * head_dim)."""
    return linear(attn_out, attn_params["o_proj"])


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

def decoder_layer(
    layer_params: dict,
    x: jnp.ndarray,
    positions: jnp.ndarray,
    kv_cache_k: jnp.ndarray,
    kv_cache_v: jnp.ndarray,
    page_table: jnp.ndarray,
    page_len: int,
    layer_idx: int,
    config: Qwen3TTSConfig,
    is_backbone: bool = True,
    attn_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One transformer decoder layer.

    Args:
        layer_params: params["backbone"]["layers"][i] or params["depth"]["layers"][i]
        x: (seq_len, hidden)
        positions: (seq_len,)
        kv_cache_k, kv_cache_v: per-layer KV cache arrays
        page_table: page indices for this request
        page_len: current valid length in last page
        layer_idx: layer index
        config: model config
        is_backbone: backbone (True) or depth (False)
        attn_fn: callable(q, k, v, kv_cache_k, kv_cache_v, page_table, page_len, layer_idx)
                 → (attn_output, new_kv_cache_k, new_kv_cache_v)
    Returns:
        (output, new_kv_cache_k, new_kv_cache_v)
    """
    n_heads = config.backbone_heads if is_backbone else config.depth_heads
    head_dim = config.head_dim

    # Pre-norm + attention
    residual = x
    x_normed = rms_norm(x, layer_params["input_layernorm"], config.rms_norm_eps)
    q, k, v = attention_proj(layer_params["attn"], x_normed, positions, config, is_backbone)

    attn_out, new_k_cache, new_v_cache = attn_fn(
        q, k, v, kv_cache_k, kv_cache_v, page_table, page_len, layer_idx
    )

    # attn_out: (seq_len, n_heads, head_dim) → (seq_len, n_heads * head_dim)
    attn_out = attn_out.reshape(x.shape[0], n_heads * head_dim)
    x = residual + attention_output(layer_params["attn"], attn_out)

    # Post-norm + MLP
    residual = x
    x_normed = rms_norm(x, layer_params["post_attn_layernorm"], config.rms_norm_eps)
    x = residual + mlp(layer_params["mlp"], x_normed)

    return x, new_k_cache, new_v_cache


# ---------------------------------------------------------------------------
# Backbone forward
# ---------------------------------------------------------------------------

def embed_inputs(
    params: dict,
    input_ids: jnp.ndarray,
    input_masks: jnp.ndarray,
    input_features: jnp.ndarray,
) -> jnp.ndarray:
    """Compute input embeddings for the backbone.

    Args:
        params: params["backbone"]
        input_ids: (batch_size, n_codebooks) — col 0 = codec0, col -1 = text token
        input_masks: (batch_size, n_codebooks) — bool, col -1 = True means codec+text
        input_features: (batch_size, hidden) — speaker/ICL features
    Returns:
        (batch_size, hidden) embeddings
    """
    # Text embedding + projection
    text_ids = input_ids[:, -1]  # (bs,)
    text_embeds = params["embed_text"][text_ids]  # (bs, hidden)
    tp = params["text_projection"]
    text_embeds = linear(silu(linear(text_embeds, tp["fc1"]["weight"], tp["fc1"]["bias"])),
                         tp["fc2"]["weight"], tp["fc2"]["bias"])

    # Codec embedding from column 0
    codec_ids = input_ids[:, 0]  # (bs,)
    codec_embeds = params["embed_codec"][codec_ids]  # (bs, hidden)

    # Combine: if mask[:, -1] is True → text + codec, else text only
    needs_codec = input_masks[:, -1:].astype(text_embeds.dtype)  # (bs, 1)
    embeds = text_embeds + codec_embeds * needs_codec

    # Add input features (speaker embeddings, depth codec sums)
    embeds = embeds + input_features

    return embeds


def backbone_forward(
    params: dict,
    input_embeds: jnp.ndarray,
    positions: jnp.ndarray,
    kv_cache_k: list[jnp.ndarray],
    kv_cache_v: list[jnp.ndarray],
    page_table: jnp.ndarray,
    page_len: int,
    config: Qwen3TTSConfig,
    attn_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]]:
    """Run the 28-layer backbone transformer.

    Args:
        params: params["backbone"]
        input_embeds: (seq_len, hidden) — already embedded
        positions: (seq_len,)
        kv_cache_k, kv_cache_v: list of per-layer cache arrays
        page_table: page indices
        page_len: valid length in last page
        config: model config
        attn_fn: attention function (see decoder_layer)
    Returns:
        (logits, hidden_states, new_kv_cache_k, new_kv_cache_v)
        logits: (seq_len, codec_vocab)
        hidden_states: (seq_len, hidden)
    """
    hidden = input_embeds
    new_k_caches = []
    new_v_caches = []

    for i in range(config.backbone_layers):
        hidden, new_k, new_v = decoder_layer(
            layer_params=params["layers"][i],
            x=hidden,
            positions=positions,
            kv_cache_k=kv_cache_k[i],
            kv_cache_v=kv_cache_v[i],
            page_table=page_table,
            page_len=page_len,
            layer_idx=i,
            config=config,
            is_backbone=True,
            attn_fn=attn_fn,
        )
        new_k_caches.append(new_k)
        new_v_caches.append(new_v)

    hidden = rms_norm(hidden, params["norm"], config.rms_norm_eps)
    logits = linear(hidden, params["codec_head"])  # (seq_len, codec_vocab)

    return logits, hidden, new_k_caches, new_v_caches


def backbone_forward_rpa(
    params: dict,
    input_embeds: jnp.ndarray,
    positions: jnp.ndarray,
    kv_caches: list[jnp.ndarray],
    kv_lens: jnp.ndarray,
    page_indices: jnp.ndarray,
    cu_q_lens: jnp.ndarray,
    distribution: jnp.ndarray,
    config: Qwen3TTSConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
    """Backbone forward using vLLM RPA kernel for attention.

    Args:
        params: params["backbone"]
        input_embeds: (total_tokens, hidden)
        positions: (total_tokens,)
        kv_caches: list of per-layer KV cache arrays (RPA merged format)
        kv_lens: (max_num_seqs,) int32
        page_indices: (max_num_seqs * pages_per_seq,) int32
        cu_q_lens: (max_num_seqs + 1,) int32
        distribution: (3,) int32
        config: model config
    Returns:
        (logits, hidden, new_kv_caches)
    """
    from .attention import rpa_attn_fn

    hidden = input_embeds
    new_kv_caches = []

    for i in range(config.backbone_layers):
        # Pre-norm
        residual = hidden
        x_normed = rms_norm(hidden, params["layers"][i]["input_layernorm"], config.rms_norm_eps)

        # QKV projection + norms + RoPE
        q, k, v = attention_proj(
            params["layers"][i]["attn"], x_normed, positions, config, is_backbone=True
        )

        # RPA: fused KV cache write + attention
        attn_out, updated_kv = rpa_attn_fn(
            q, k, v, kv_caches[i],
            kv_lens, page_indices, cu_q_lens, distribution,
            n_heads=config.backbone_heads,
            n_kv_heads=config.backbone_kv_heads,
            head_dim=config.head_dim,
        )
        new_kv_caches.append(updated_kv)

        # O projection + residual
        n_heads = config.backbone_heads
        attn_out = attn_out.reshape(hidden.shape[0], n_heads * config.head_dim)
        hidden = residual + attention_output(params["layers"][i]["attn"], attn_out)

        # Post-norm + MLP
        residual = hidden
        x_normed = rms_norm(hidden, params["layers"][i]["post_attn_layernorm"], config.rms_norm_eps)
        hidden = residual + mlp(params["layers"][i]["mlp"], x_normed)

    hidden = rms_norm(hidden, params["norm"], config.rms_norm_eps)
    logits = linear(hidden, params["codec_head"])

    return logits, hidden, new_kv_caches


# Keep the old backbone_forward_stacked for backward compatibility
backbone_forward_stacked = backbone_forward_rpa  # alias


# ---------------------------------------------------------------------------
# Top-level JIT-able functions
# ---------------------------------------------------------------------------

def decode_step(
    params: dict,
    kv_caches: list[jnp.ndarray],
    input_ids: jnp.ndarray,
    position_ids: jnp.ndarray,
    input_masks: jnp.ndarray,
    input_features: jnp.ndarray,
    kv_lens: jnp.ndarray,
    page_indices: jnp.ndarray,
    cu_q_lens: jnp.ndarray,
    distribution: jnp.ndarray,
    rng_key: jnp.ndarray,
    config: Qwen3TTSConfig,
    sample_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray, list[jnp.ndarray], jnp.ndarray]:
    """Complete decode step: backbone + sample + depth.

    All in one function for JIT compilation with donate_argnums on KV caches.

    Returns:
        (all_codebook_ids, ci_embed_sum, new_kv_caches, rng_key)
        all_codebook_ids: (bs, 16) — col 0 = cb0, cols 1-15 = depth codebooks
    """
    bs = input_ids.shape[0]

    # Embed
    embeds = embed_inputs(params["backbone"], input_ids, input_masks, input_features)

    # Backbone with RPA
    logits, hidden, kv_caches = backbone_forward_rpa(
        params["backbone"], embeds, position_ids,
        kv_caches, kv_lens, page_indices, cu_q_lens, distribution,
        config,
    )

    # Sample codebook-0
    c0_ids, rng_key = sample_fn(logits, rng_key)

    # Depth
    max_depth_seq = 17
    depth_shape = (
        bs, config.depth_layers, max_depth_seq,
        config.depth_kv_heads, config.head_dim,
    )
    depth_kv_k = jnp.zeros(depth_shape, dtype=jnp.bfloat16)
    depth_kv_v = jnp.zeros(depth_shape, dtype=jnp.bfloat16)

    all_cb_ids, ci_embed_sum, _, _, rng_key = depth_forward_fused(
        params["depth"], params["backbone"]["embed_codec"],
        hidden, c0_ids,
        depth_kv_k, depth_kv_v, config, rng_key,
        sample_fn=sample_fn,
    )

    # Combine: (bs, 16) — cb0 + 15 depth codebooks
    all_ids = jnp.concatenate([c0_ids[:, None], all_cb_ids], axis=1)

    return all_ids, ci_embed_sum, kv_caches, rng_key


# ---------------------------------------------------------------------------
# Depth transformer forward (with fori_loop)
# ---------------------------------------------------------------------------

def _depth_transformer_single(
    depth_params: dict,
    input_embeds: jnp.ndarray,
    positions: jnp.ndarray,
    kv_k: jnp.ndarray,
    kv_v: jnp.ndarray,
    seq_start: jnp.ndarray,
    codebook_idx: jnp.ndarray,
    config: Qwen3TTSConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run 5-layer depth transformer for ONE batch item.

    Args:
        depth_params: params["depth"]
        input_embeds: (tokens, depth_hidden) — tokens for this one item
        positions: (tokens,) int32
        kv_k: (depth_layers, max_seq, n_kv_heads, head_dim)
        kv_v: same
        seq_start: scalar — write position in cache
        codebook_idx: scalar — which lm_head (0-14)
        config: model config
    Returns:
        (logits, new_kv_k, new_kv_v)
        logits: (tokens, depth_vocab)
    """
    hidden = input_embeds
    n_heads = config.depth_heads
    n_kv_heads = config.depth_kv_heads
    head_dim = config.head_dim

    new_kv_k = kv_k
    new_kv_v = kv_v

    for layer_i in range(config.depth_layers):
        residual = hidden
        x_normed = rms_norm(hidden, depth_params["layers"][layer_i]["input_layernorm"],
                            config.rms_norm_eps)
        q, k, v = attention_proj(
            depth_params["layers"][layer_i]["attn"], x_normed, positions, config, is_backbone=False
        )
        attn_out, layer_k, layer_v = sequential_attn_fn(
            q, k, v,
            new_kv_k[layer_i], new_kv_v[layer_i],
            seq_start=seq_start, seq_len_so_far=0, layer_idx=layer_i,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
        )
        new_kv_k = new_kv_k.at[layer_i].set(layer_k)
        new_kv_v = new_kv_v.at[layer_i].set(layer_v)

        attn_out = attn_out.reshape(hidden.shape[0], n_heads * head_dim)
        hidden = residual + attention_output(depth_params["layers"][layer_i]["attn"], attn_out)

        residual = hidden
        x_normed = rms_norm(hidden, depth_params["layers"][layer_i]["post_attn_layernorm"],
                            config.rms_norm_eps)
        hidden = residual + mlp(depth_params["layers"][layer_i]["mlp"], x_normed)

    hidden = rms_norm(hidden, depth_params["norm"], config.rms_norm_eps)
    lm_head_weight = depth_params["lm_heads"][codebook_idx]
    logits = linear(hidden, lm_head_weight)

    return logits, new_kv_k, new_kv_v


def _depth_batched(
    depth_params: dict,
    input_embeds: jnp.ndarray,
    positions: jnp.ndarray,
    kv_k: jnp.ndarray,
    kv_v: jnp.ndarray,
    seq_start: jnp.ndarray,
    codebook_idx: jnp.ndarray,
    config: Qwen3TTSConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched depth transformer via vmap over batch dimension.

    Args:
        input_embeds: (bs, tokens, depth_hidden)
        positions: (bs, tokens) int32
        kv_k: (bs, depth_layers, max_seq, n_kv_heads, head_dim)
        kv_v: same
        seq_start: scalar (shared across batch)
        codebook_idx: scalar (shared across batch)
    Returns:
        (logits, new_kv_k, new_kv_v) with leading bs dimension
    """
    # Close over depth_params, config, seq_start, codebook_idx; vmap over batch arrays
    def _single(embeds, pos, kk, kv):
        return _depth_transformer_single(
            depth_params, embeds, pos, kk, kv, seq_start, codebook_idx, config
        )

    return jax.vmap(_single)(input_embeds, positions, kv_k, kv_v)


def depth_forward_fused(
    depth_params: dict,
    backbone_embed_codec: jnp.ndarray,
    backbone_hidden: jnp.ndarray,
    c0_ids: jnp.ndarray,
    depth_kv_k: jnp.ndarray,
    depth_kv_v: jnp.ndarray,
    config: Qwen3TTSConfig,
    rng_key: jnp.ndarray,
    sample_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fused depth loop: all 15 codebook iterations in one jax.lax.fori_loop.

    This compiles into a SINGLE HLO graph covering all 15 iterations.

    Each batch item has independent depth attention (no cross-batch attention).
    Uses vmap internally to parallelize across the batch.

    Args:
        depth_params: params["depth"]
        backbone_embed_codec: params["backbone"]["embed_codec"] — (3072, 2048)
        backbone_hidden: (bs, backbone_hidden_dim)
        c0_ids: (bs,) int32 — sampled codebook-0 token IDs
        depth_kv_k: (bs, depth_layers, max_depth_seq, n_kv_heads, head_dim)
        depth_kv_v: same
        config: model config
        rng_key: PRNG key
        sample_fn: (logits, rng_key) → (token_ids, new_rng_key)
    Returns:
        (all_codebook_ids, ci_embed_sum, new_depth_kv_k, new_depth_kv_v, rng_key)
        all_codebook_ids: (bs, 15) int32
        ci_embed_sum: (bs, backbone_hidden_dim)
    """
    bs = backbone_hidden.shape[0]
    num_codebooks = config.num_codebooks  # 15

    # Project backbone hidden → depth hidden
    proj_w = depth_params["projection"]["weight"]
    proj_b = depth_params["projection"]["bias"]
    depth_hidden = linear(backbone_hidden, proj_w, proj_b)  # (bs, depth_hidden)

    # c0 embedding from BACKBONE codec_embedding, projected to depth dim
    c0_embed = backbone_embed_codec[c0_ids]  # (bs, backbone_hidden)
    c0_depth = linear(c0_embed, proj_w, proj_b)  # (bs, depth_hidden)

    # Allocate output buffers
    all_ids = jnp.zeros((bs, num_codebooks), dtype=jnp.int32)
    ci_embed_sum = jnp.zeros((bs, config.backbone_hidden), dtype=jnp.bfloat16)

    # --- Iteration 0 (prefill: 2 tokens per batch item) ---
    # input: (bs, 2, depth_hidden), positions: (bs, 2)
    first_input = jnp.stack([depth_hidden, c0_depth], axis=1)  # (bs, 2, depth_hidden)
    first_positions = jnp.broadcast_to(
        jnp.array([0, 1], dtype=jnp.int32)[None, :], (bs, 2)
    )

    logits_first, depth_kv_k, depth_kv_v = _depth_batched(
        depth_params, first_input, first_positions,
        depth_kv_k, depth_kv_v,
        seq_start=jnp.int32(0), codebook_idx=jnp.int32(0),
        config=config,
    )
    # logits_first: (bs, 2, depth_vocab) — take last token per item
    logits_first = logits_first[:, -1, :]  # (bs, depth_vocab)

    token_ids, rng_key = sample_fn(logits_first, rng_key)
    all_ids = all_ids.at[:, 0].set(token_ids)
    ci_embed = depth_params["codec_embeddings"][0][token_ids]
    ci_embed_sum = ci_embed_sum + ci_embed

    # --- Iterations 1-14 (decode: 1 token per batch item, via fori_loop) ---

    def _body(i, carry):
        all_ids, ci_embed_sum, rng_key, kv_k, kv_v = carry

        prev_ids = all_ids[:, i - 1]
        ci_emb = depth_params["codec_embeddings"][i - 1][prev_ids]  # (bs, backbone_hidden)
        ci_depth = linear(ci_emb, proj_w, proj_b)  # (bs, depth_hidden)

        # (bs, 1, depth_hidden), positions (bs, 1)
        ci_input = ci_depth[:, None, :]
        positions = jnp.full((bs, 1), i + 1, dtype=jnp.int32)

        logits, kv_k, kv_v = _depth_batched(
            depth_params, ci_input, positions,
            kv_k, kv_v,
            seq_start=jnp.int32(i + 1), codebook_idx=jnp.int32(i),
            config=config,
        )
        logits = logits[:, 0, :]  # (bs, depth_vocab)

        token_ids, rng_key = sample_fn(logits, rng_key)
        all_ids = all_ids.at[:, i].set(token_ids)

        ci_embed = depth_params["codec_embeddings"][i][token_ids]
        ci_embed_sum = ci_embed_sum + ci_embed

        return (all_ids, ci_embed_sum, rng_key, kv_k, kv_v)

    init_carry = (all_ids, ci_embed_sum, rng_key, depth_kv_k, depth_kv_v)
    all_ids, ci_embed_sum, rng_key, depth_kv_k, depth_kv_v = jax.lax.fori_loop(
        1, num_codebooks, _body, init_carry
    )

    return all_ids, ci_embed_sum, depth_kv_k, depth_kv_v, rng_key
