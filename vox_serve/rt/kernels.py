"""Sampling kernels recorded into the programs' CUDA graphs.

The kernels (``rust/csrc/sampling.cu``) are compiled into the ``vox_serve._rt`` extension module. These
wrappers take torch tensors and launch on the current stream of the tensors' device, so they are
recorded into the CUDA graph being captured. Without the extension (its optional build failed), the
same operations run as torch ops (``_TorchOps``): same filtering semantics, a different random stream.
"""

import importlib
import importlib.util
import os
from functools import lru_cache

import torch

# Logits dtype codes of rust/csrc/sampling.cu.
_DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


def _logits_layout(logits: torch.Tensor) -> tuple[int, int, int]:
    """(row stride, vocab size, rows) of logits [..., V] with a uniform stride between rows."""
    if logits.dtype not in _DTYPES:
        raise TypeError(f"unsupported logits dtype {logits.dtype}")
    if logits.stride(-1) != 1:
        raise ValueError("logits must be contiguous in the vocab dimension")
    V = logits.size(-1)
    if logits.dim() < 2:
        row_stride = V
    elif logits.dim() == 3 and logits.size(1) == 1:  # [B, 1, V] views may skip rows
        row_stride = logits.stride(0)
    else:
        row_stride = logits.stride(-2)
        if logits.dim() > 2 and logits.stride(0) != row_stride * logits.size(1):
            raise ValueError("unsupported logits layout")
    return row_stride, V, logits.numel() // V


def _check(t: torch.Tensor, dtype: torch.dtype, name: str) -> int:
    if t.dtype != dtype or not t.is_cuda:
        raise TypeError(f"{name}: expected a CUDA {dtype} tensor, got {t.dtype} on {t.device}")
    return t.data_ptr()


def _launch_args(t: torch.Tensor) -> tuple[int, int]:
    """(device index, current stream) for a launch on `t`'s device."""
    return t.device.index, torch.cuda.current_stream(t.device).cuda_stream


class _Ops:
    def __init__(self, rt):
        self._rt = rt

    def sample(self, logits, rows_per_req, active, temperature, top_k, top_p, min_p, seed, step, rng_salt, out):
        """Samples logits [B, R, V] (or [B*R, V] with rows_per_req=R) into out [B, out_stride] int32."""
        row_stride, V, n_rows = _logits_layout(logits)
        self._rt.sample(
            *_launch_args(logits),
            _DTYPES[logits.dtype],
            logits.data_ptr(),
            row_stride,
            V,
            n_rows,
            rows_per_req,
            _check(active, torch.int32, "active"),
            _check(temperature, torch.float32, "temperature"),
            _check(top_k, torch.int32, "top_k"),
            _check(top_p, torch.float32, "top_p"),
            _check(min_p, torch.float32, "min_p"),
            _check(seed, torch.int64, "seed"),
            _check(step, torch.int32, "step"),
            rng_salt,
            _check(out, torch.int32, "out"),
            out.stride(0),
        )

    def rep_penalty_apply(self, logits, rows_per_req, bits, slot, active, first, penalty):
        row_stride, V, n_rows = _logits_layout(logits)
        self._rt.rep_penalty_apply(
            *_launch_args(logits),
            _DTYPES[logits.dtype],
            logits.data_ptr(),
            row_stride,
            V,
            n_rows,
            rows_per_req,
            _check(bits, torch.int32, "bits"),
            bits.size(-1),
            _check(slot, torch.int32, "slot"),
            _check(active, torch.int32, "active"),
            _check(first, torch.int32, "first"),
            _check(penalty, torch.float32, "penalty"),
        )

    def rep_penalty_reset(self, bits, rows_per_req, slot, active, first):
        self._rt.rep_penalty_reset(
            *_launch_args(bits),
            _check(bits, torch.int32, "bits"),
            bits.size(-1),
            rows_per_req,
            slot.size(0),
            _check(slot, torch.int32, "slot"),
            _check(active, torch.int32, "active"),
            _check(first, torch.int32, "first"),
        )

    def rep_penalty_update(self, bits, rows_per_req, vocab, slot, active, tokens):
        self._rt.rep_penalty_update(
            *_launch_args(bits),
            _check(bits, torch.int32, "bits"),
            bits.size(-1),
            rows_per_req,
            vocab,
            slot.size(0),
            _check(slot, torch.int32, "slot"),
            _check(active, torch.int32, "active"),
            _check(tokens, torch.int32, "tokens"),
            tokens.stride(0),
        )

    def randn(self, out, seed, step, rng_salt):
        """Standard normal noise into out [N, K] (float32, contiguous) keyed by (seed, step, salt)."""
        if not out.is_contiguous():
            raise ValueError("randn: out must be contiguous")
        n = out.size(0)
        self._rt.randn(
            *_launch_args(out),
            _check(out, torch.float32, "out"),
            n,
            out.numel() // max(1, n),
            _check(seed, torch.int64, "seed"),
            _check(step, torch.int32, "step"),
            rng_salt,
        )


# ---- torch fallback ------------------------------------------------------------------------------

_Z_RANGE = 32.0  # tokens more than 32 nats below the best one are never sampled (as in the CUDA kernel)
_MASK53 = (1 << 53) - 1


def _mix64(x: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer on int64 tensors (wrapping multiplies; logical shifts via masks)."""
    x = x ^ ((x >> 30) & ((1 << 34) - 1))
    x = x * -4658895280553007687  # 0xbf58476d1ce4e5b9
    x = x ^ ((x >> 27) & ((1 << 37) - 1))
    x = x * -7723592293110705685  # 0x94d049bb133111eb
    return x ^ ((x >> 31) & ((1 << 33) - 1))


def _wrap64(v: int) -> int:
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >> 63 else v


def _uniform(seed: torch.Tensor, counter: torch.Tensor, salt: int) -> torch.Tensor:
    """Uniform floats in (0, 1], a pure function of (seed, counter, salt): graph-capturable and independent
    of batch composition and replay count, like the Philox stream of the CUDA kernels."""
    x = _mix64(seed.long() ^ _mix64(counter.long() * 4 + _wrap64(salt * 0x9E3779B97F4A7C15)))
    return (((x >> 11) & _MASK53) + 1).double().div_(float(1 << 53)).float()


class _TorchOps:
    """The operations of ``_Ops`` as torch ops (no compiled extension needed)."""

    def sample(self, logits, rows_per_req, active, temperature, top_k, top_p, min_p, seed, step, rng_salt, out):
        V = logits.size(-1)
        x = logits.reshape(-1, V).float()
        n_rows = x.shape[0]
        row = torch.arange(n_rows, device=x.device)
        req, sub = row // rows_per_req, row % rows_per_req
        best_v, best = x.max(-1)  # first index wins ties, like torch.argmax
        t = temperature[req]
        k = top_k[req].long()
        greedy = (t <= 0) | (k == 1)
        z = (best_v[:, None] - x) / torch.where(t > 0, t, torch.ones_like(t))[:, None]
        zs = z.sort(-1).values
        # Top-k: keep z <= z of the k-th best token (ties at the cut are kept), within the 32-nat range.
        k_eff = torch.where((k > 0) & (k < V), k, torch.full_like(k, V))
        zthr = zs.gather(1, (k_eff - 1)[:, None]).squeeze(1).clamp(max=_Z_RANGE)
        # Top-p on the top-k restricted distribution: the smallest prefix (by probability) reaching p.
        cum = torch.where(zs <= zthr[:, None], torch.exp(-zs), torch.zeros_like(zs)).cumsum(-1)
        p = top_p[req]
        target = (p * cum[:, -1])[:, None]
        cut = torch.searchsorted(cum, target).clamp(max=V - 1)
        zthr = torch.where((p > 0) & (p < 1), torch.minimum(zthr, zs.gather(1, cut).squeeze(1)), zthr)
        # Min-p: keep p_i >= min_p * p_max.
        mp = min_p[req]
        zthr = torch.where(mp > 0, torch.minimum(zthr, -torch.log(mp.clamp(min=1e-30))), zthr)
        # Inverse-CDF draw over the kept tokens in index order.
        w = torch.where(z <= zthr[:, None], torch.exp(-z), torch.zeros_like(z)).cumsum(-1)
        u = _uniform(seed[req], step[req].long() * 64 + sub, int(rng_salt))
        pick = torch.searchsorted(w, (u * w[:, -1])[:, None], right=True).squeeze(1)
        pick = torch.where(pick < V, pick, best)
        tok = torch.where(greedy, best, pick)
        tok = torch.where(active[req] != 0, tok, torch.zeros_like(tok)).int()
        out[:, :rows_per_req].copy_(tok.view(-1, rows_per_req)[: out.shape[0]])

    @staticmethod
    def _seen(bits, rows_per_req, slot, n_rows, V):
        """Seen-token mask [n_rows, V] of every logits row from bits [slots, rows_per_req, words]."""
        row = torch.arange(n_rows, device=bits.device)
        b = bits.view(-1, bits.size(-1))[slot[row // rows_per_req].long() * rows_per_req + row % rows_per_req]
        shift = torch.arange(32, device=bits.device, dtype=torch.int32)
        return (((b[:, :, None] >> shift) & 1) != 0).view(n_rows, -1)[:, :V]

    def rep_penalty_apply(self, logits, rows_per_req, bits, slot, active, first, penalty):
        V = logits.size(-1)
        rows = logits.reshape(-1, V)
        n_rows = rows.shape[0]
        req = torch.arange(n_rows, device=rows.device) // rows_per_req
        pen = penalty[req]
        on = (active[req] != 0) & (first[req] == 0) & (pen != 1) & (pen > 0)
        x = rows.float()
        new = torch.where(x > 0, x / pen[:, None], x * pen[:, None])
        hit = self._seen(bits, rows_per_req, slot, n_rows, V) & on[:, None]
        logits.copy_(torch.where(hit, new, x).to(logits.dtype).view(logits.shape))

    def rep_penalty_reset(self, bits, rows_per_req, slot, active, first):
        # Requests that are not reset clear the dummy slot (the last one) instead.
        dummy = torch.full_like(slot, bits.shape[0] - 1)
        bits.index_fill_(0, torch.where((active != 0) & (first != 0), slot, dummy).long(), 0)

    def rep_penalty_update(self, bits, rows_per_req, vocab, slot, active, tokens):
        n = slot.shape[0]
        words = bits.size(-1)
        tok = tokens[:, :rows_per_req].long()
        sub = torch.arange(rows_per_req, device=bits.device)
        row = slot.long()[:, None] * rows_per_req + sub[None, :]
        ok = (active != 0)[:, None] & (tok >= 0) & (tok < vocab)
        # Invalid entries write the dummy slot's (last) row, which no request reads.
        dummy = (bits.shape[0] - 1) * rows_per_req + sub[None, :].expand(n, -1)
        pos = torch.where(ok, row, dummy) * words + torch.where(ok, tok >> 5, torch.zeros_like(tok))
        flat = bits.view(-1)
        bit = torch.where(ok, torch.ones_like(tok) << (tok & 31), torch.zeros_like(tok)).int()
        flat.index_put_((pos.view(-1),), flat[pos.view(-1)] | bit.view(-1))

    def randn(self, out, seed, step, rng_salt):
        n, K = out.size(0), out.numel() // max(1, out.size(0))
        k = torch.arange(K, device=out.device)
        counter = step.long()[:, None] * (64 * 2 * K) + 2 * k[None, :]
        u1 = _uniform(seed[:, None].expand(n, K), counter, int(rng_salt))
        u2 = _uniform(seed[:, None].expand(n, K), counter + 1, int(rng_salt))
        out.view(n, K).copy_(torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * torch.pi * u2))


def native_kernels() -> bool:
    """Whether the compiled sampling kernels (the ``vox_serve._rt`` extension) are available."""
    return importlib.util.find_spec("vox_serve._rt") is not None


@lru_cache(maxsize=None)
def ops():
    """The compiled kernels if available (``VOX_TORCH_SAMPLING=1`` forces the torch fallback, for testing)."""
    if native_kernels() and os.environ.get("VOX_TORCH_SAMPLING") != "1":
        try:
            return _Ops(importlib.import_module("vox_serve._rt"))
        except ImportError:
            pass
    return _TorchOps()
