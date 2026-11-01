"""Segmented multi-head attention for chunked flow-matching decoders (Triton).

Every query row ``r`` (a chunk of ``n_q`` frames) attends, without masking inside segments, to the
concatenation of three key/value segments:

* a shared prefix (e.g. a voice prompt): row ``p_row[r]`` of ``pk`` / ``pv``, first ``p_len[r]`` keys;
* its history (e.g. a ring of earlier chunks): row ``h_row[r]`` of ``hk`` / ``hv``, first ``h_len[r]``;
* its own chunk: row ``r`` of ``ck`` / ``cv``, first ``c_len[r]`` keys.

Row indices and lengths are read on device, so one launch is CUDA-graph safe for any mix of
requests. Keys / values are ``[rows, T, n_heads * head_dim]`` views (any row / time stride, unit
channel stride), already rotated if the model uses rotary embeddings.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _segment(
    acc, m_i, l_i, q, k_base, v_base, stride_t, length, qk_scale, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr
):
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    for start in range(0, length, BLOCK_N):
        n = start + offs_n
        valid = n < length
        k = tl.load(k_base + n[None, :] * stride_t + offs_d[:, None], mask=valid[None, :], other=0.0)
        s = tl.dot(q, k) * qk_scale
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + n[:, None] * stride_t + offs_d[None, :], mask=valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, m_i, l_i


@triton.jit
def _seg_attn_kernel(
    Q,
    O,
    PK,
    PV,
    p_row,
    p_len,
    HK,
    HV,
    h_row,
    h_len,
    CK,
    CV,
    c_len,
    sq_r,
    sq_t,
    so_r,
    so_t,
    sp_r,
    sp_t,
    sh_r,
    sh_t,
    sc_r,
    sc_t,
    n_q,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    r = tl.program_id(0)
    head = tl.program_id(1)
    offs_m = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    col = head * HEAD_DIM
    q_mask = offs_m[:, None] < n_q
    q = tl.load(Q + r * sq_r + offs_m[:, None] * sq_t + col + offs_d[None, :], mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    pr = tl.load(p_row + r).to(tl.int64)
    acc, m_i, l_i = _segment(
        acc,
        m_i,
        l_i,
        q,
        PK + pr * sp_r + col,
        PV + pr * sp_r + col,
        sp_t,
        tl.load(p_len + r),
        qk_scale,
        HEAD_DIM,
        BLOCK_N,
    )
    hr = tl.load(h_row + r).to(tl.int64)
    acc, m_i, l_i = _segment(
        acc,
        m_i,
        l_i,
        q,
        HK + hr * sh_r + col,
        HV + hr * sh_r + col,
        sh_t,
        tl.load(h_len + r),
        qk_scale,
        HEAD_DIM,
        BLOCK_N,
    )
    rr = r.to(tl.int64)
    acc, m_i, l_i = _segment(
        acc,
        m_i,
        l_i,
        q,
        CK + rr * sc_r + col,
        CV + rr * sc_r + col,
        sc_t,
        tl.load(c_len + r),
        qk_scale,
        HEAD_DIM,
        BLOCK_N,
    )
    out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    tl.store(O + r * so_r + offs_m[:, None] * so_t + col + offs_d[None, :], out.to(O.dtype.element_ty), mask=q_mask)


def seg_attention(
    q: torch.Tensor,
    pk: torch.Tensor,
    pv: torch.Tensor,
    p_row: torch.Tensor,
    p_len: torch.Tensor,
    hk: torch.Tensor,
    hv: torch.Tensor,
    h_row: torch.Tensor,
    h_len: torch.Tensor,
    ck: torch.Tensor,
    cv: torch.Tensor,
    c_len: torch.Tensor,
    n_heads: int,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """q [R, n_q, H * D] -> [R, n_q, H * D]; see the module docstring for the segments. Index / length
    tensors are int32 [R]. Scale 1 / sqrt(D)."""
    R, n_q, width = q.shape
    head_dim = width // n_heads
    for t in (q, pk, pv, hk, hv, ck, cv):
        assert t.stride(-1) == 1, "channel stride must be 1"
    assert pk.stride() == pv.stride() and hk.stride() == hv.stride() and ck.stride() == cv.stride()
    if out is None:
        out = torch.empty(R, n_q, width, dtype=q.dtype, device=q.device)
    BLOCK_M = 64 if n_q > 32 else 32
    grid = (R, n_heads, triton.cdiv(n_q, BLOCK_M))
    _seg_attn_kernel[grid](
        q,
        out,
        pk,
        pv,
        p_row,
        p_len,
        hk,
        hv,
        h_row,
        h_len,
        ck,
        cv,
        c_len,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        pk.stride(0),
        pk.stride(1),
        hk.stride(0),
        hk.stride(1),
        ck.stride(0),
        ck.stride(1),
        n_q,
        math.log2(math.e) / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=64,
        num_warps=4,
        num_stages=2,
    )
    return out
