import math
from typing import Protocol

import flashinfer
import torch

from .utils import get_logger

logger = get_logger(__name__)


class FlashInferWrapper(Protocol):
    """Paged attention as model code sees it (implemented by ``vox_serve.rt.core.RtAttention``)."""

    def set_kv_cache(self, kv_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Writes the K/V of this step's rows into ``kv_cache`` (one layer: pages, 2, page size, heads, dim)."""

    def run(self, q: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """Attention of this step's queries ``q`` over the paged ``kv_cache``."""


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Wrapper for FlashInfer RMSNorm operation.

    Args:
        hidden_states: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Epsilon value for numerical stability

    Returns:
        Normalized tensor of the same shape as hidden_states
    """
    if torch.compiler.is_compiling():
        # FlashInfer's ops have no fake-tensor kernels; under torch.compile use the same fp32 math
        # in plain torch so Inductor can trace (and fuse) it.
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (x * weight.float()).to(hidden_states.dtype)
    return flashinfer.norm.rmsnorm(
        input=hidden_states,
        weight=weight,
        eps=eps,
    )


def apply_rope_pos_ids(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_ids: torch.Tensor,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
    interleave: bool = False,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper for FlashInfer RoPE application with position IDs.

    Args:
        query_states: Query states tensor
        key_states: Key states tensor
        position_ids: Position IDs tensor
        rope_scale: Rope scaling factor
        rope_theta: Rope theta parameter
        interleave: Whether to interleave the RoPE
        **kwargs: Additional parameters for specific RoPE variants

    Returns:
        Tuple of (rotated_query_states, rotated_key_states)
    """
    # Filter out kwargs that are meant for specific RoPE variants
    llama31_params = {}
    other_params = {}

    for key, value in kwargs.items():
        if key in ['low_freq_factor', 'high_freq_factor', 'old_context_len']:
            llama31_params[key] = value
        else:
            other_params[key] = value

    if torch.compiler.is_compiling():
        return _rope_pos_ids_torch(
            query_states, key_states, position_ids, rope_scale, rope_theta, interleave, llama31_params, **other_params
        )

    # Use LLaMA 3.1 variant if specific parameters are provided
    if llama31_params:
        return flashinfer.rope.apply_llama31_rope_pos_ids(
            query_states,
            key_states,
            pos_ids=position_ids,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            interleave=interleave,
            **llama31_params
        )
    else:
        return flashinfer.rope.apply_rope_pos_ids(
            query_states,
            key_states,
            pos_ids=position_ids,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            interleave=interleave,
            **other_params
        )


def _rope_pos_ids_torch(q, k, pos_ids, rope_scale, rope_theta, interleave, llama31_params, rotary_dim=None):
    """Plain-torch equivalent of ``flashinfer.rope.apply_(llama31_)rope_pos_ids`` (fp32 math), used
    while tracing with torch.compile. q, k: (nnz, heads, head_dim); pos_ids: (nnz,)."""
    head_dim = q.shape[-1]
    rotary_dim = rotary_dim or head_dim
    inv_freq = rope_theta ** -(torch.arange(0, rotary_dim, 2, device=q.device, dtype=torch.float32) / rotary_dim)
    if llama31_params:
        low_ff = llama31_params.get("low_freq_factor", 1.0)
        high_ff = llama31_params.get("high_freq_factor", 4.0)
        old_ctx = llama31_params.get("old_context_len", 8192)
        smooth = ((old_ctx * inv_freq / (2 * math.pi)) - low_ff) / (high_ff - low_ff)
        smooth = smooth.clamp(0.0, 1.0)
        inv_freq = (1 - smooth) * inv_freq / rope_scale + smooth * inv_freq
    else:
        inv_freq = inv_freq / rope_scale
    freqs = pos_ids.float()[:, None] * inv_freq[None, :]  # (nnz, rotary_dim / 2)
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]

    def rotate(x):
        rot, rest = x[..., :rotary_dim].float(), x[..., rotary_dim:]
        if interleave:
            x1, x2 = rot[..., 0::2], rot[..., 1::2]
            out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)
        else:
            x1, x2 = rot.chunk(2, dim=-1)
            out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return torch.cat((out.to(x.dtype), rest), dim=-1) if rotary_dim < head_dim else out.to(x.dtype)

    return rotate(q), rotate(k)
