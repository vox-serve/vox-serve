"""FP8 (e4m3) weight quantization of LM linears: ``Fp8Linear`` and a reader for torchao FP8 checkpoints.

Weights are ``float8_e4m3fn`` ``[N, K]`` with one fp32 scale per output row (``W ~= weight * scale[:, None]``).
Activations are quantized per token on the fly (dynamic, absmax), and the GEMM runs on the FP8 tensor cores
through ``torch._scaled_mm``. Where cuBLAS supports row-wise scales (Hopper / data-center Blackwell), both
scales are applied in the GEMM epilogue; otherwise (e.g. Jetson Thor) the GEMM runs with unit scales and the
output is rescaled by one broadcast multiply, which costs an extra pass over the output (prefill-sized
batches pay for it; decode-sized batches are dominated by the halved weight reads).

``load_torchao_fp8`` reads a ``torch.save``'d state dict whose linears are torchao ``float8_weight_only``
tensors (``AffineQuantizedTensor`` / ``Float8AQTTensorImpl``) without importing torchao and without running
arbitrary pickled code: the unpickler resolves only an allow-list of globals.
"""

import collections
import mmap
import pickle
import struct
import warnings
import zipfile
from functools import lru_cache
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

FP8 = torch.float8_e4m3fn
# Activations are scaled to slightly below the e4m3 max (448) so bf16 rounding of x * (MAX / amax) never
# exceeds the representable range.
ACT_MAX = 440.0


@lru_cache(maxsize=None)
def rowwise_supported(device: torch.device) -> bool:
    """Whether ``torch._scaled_mm`` takes row-wise (per-token x per-channel) scales on ``device``."""
    try:
        a = torch.zeros(16, 32, device=device, dtype=FP8)
        b = torch.zeros(16, 32, device=device, dtype=FP8).t()
        torch._scaled_mm(
            a, b, scale_a=torch.ones(16, 1, device=device), scale_b=torch.ones(1, 16, device=device),
            out_dtype=torch.bfloat16,
        )  # fmt: skip
        return True
    except (RuntimeError, NotImplementedError):
        return False


@lru_cache(maxsize=None)
def _unit_scale(device: torch.device) -> torch.Tensor:
    return torch.ones((), device=device)


@lru_cache(maxsize=None)
def _inv_act_max(device: torch.device) -> torch.Tensor:
    return torch.full((), 1.0 / ACT_MAX, device=device)


def fp8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``x @ (weight * scale[:, None]).T (+ bias)``: x bf16 [..., K], weight fp8 [N, K], scale fp32 [N]."""
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    if not x.is_contiguous():
        x = x.contiguous()
    amax = torch.linalg.vector_norm(x, float("inf"), dim=-1, keepdim=True).clamp_(min=1e-4)
    xq = (x * (ACT_MAX / amax)).to(FP8)
    dev = x.device
    if rowwise_supported(dev):
        y = torch._scaled_mm(
            xq, weight.t(), scale_a=amax.float().mul_(1.0 / ACT_MAX), scale_b=scale[None], bias=bias,
            out_dtype=x.dtype,
        )  # fmt: skip
    else:
        # Tensor-wise scales only: 1 / ACT_MAX in the GEMM, then the per-token and per-channel scales
        # (in-place multiplies compute in fp32 and round once per multiply).
        y = torch._scaled_mm(xq, weight.t(), scale_a=_inv_act_max(dev), scale_b=_unit_scale(dev), out_dtype=x.dtype)
        y = y.mul_(amax).mul_(scale)
        if bias is not None:
            y = y.add_(bias)
    return y.view(*shape[:-1], -1)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], scale: Optional[torch.Tensor]):
    """``F.linear``, or ``fp8_linear`` when ``scale`` is given."""
    if scale is None:
        return F.linear(x, weight, bias)
    return fp8_linear(x, weight, scale, bias)


class Fp8Linear(nn.Module):
    """Drop-in for ``nn.Linear`` (no bias) with FP8 weights and per-output-channel scales."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=torch.bfloat16):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        self.register_buffer("bias", torch.empty(out_features, dtype=dtype) if bias else None)

    @classmethod
    def from_parts(cls, weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Wraps existing tensors (views allowed) without copying."""
        mod = cls.__new__(cls)
        nn.Module.__init__(mod)
        mod.out_features, mod.in_features = weight.shape
        mod.register_buffer("weight", weight)
        mod.register_buffer("weight_scale", scale)
        mod.register_buffer("bias", bias)
        return mod

    @classmethod
    def quantize(cls, lin: nn.Linear) -> "Fp8Linear":
        """Per-output-channel absmax quantization of a bf16 / fp32 ``nn.Linear``."""
        w = lin.weight.detach().float()
        scale = w.abs().amax(1).clamp_(min=1e-12) / torch.finfo(FP8).max
        weight = (w / scale[:, None]).to(FP8)
        bias = None if lin.bias is None else lin.bias.detach()
        return cls.from_parts(weight, scale, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_linear(x, self.weight, self.weight_scale, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8_e4m3"


# ---- torchao FP8 checkpoints (torch.save zip format) -------------------------------------------------


class Fp8Tensor:
    """A quantized weight read from a checkpoint: fp8 data [N, K] and fp32 per-row scale [N]."""

    def __init__(self, data: torch.Tensor, scale: torch.Tensor):
        self.data, self.scale = data, scale


class _AffineQuantizedTensor:
    def __init__(self, state: dict):
        impl = state["tensor_impl"]
        block = list(state["block_size"])
        data, scale = impl["float8_data"], impl["scale"]
        if impl.get("transposed"):
            raise ValueError("transposed torchao FP8 weights are not supported")
        if data.dtype != FP8 or data.dim() != 2 or block != [1, data.shape[1]]:
            raise ValueError(f"unsupported torchao FP8 layout: {data.dtype} {tuple(data.shape)} block {block}")
        self.value = Fp8Tensor(data, scale.float().reshape(-1))


class _Float8AQTTensorImpl:
    def __init__(self, state: dict):
        self.value = state


class _Float8Layout:
    """``torchao.dtypes.floatx.float8_layout.Float8Layout`` (a dataclass: rebuilt by NEWOBJ + BUILD)."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def _rebuild_from_type_v2(func, new_type, args, state):
    if new_type in (_AffineQuantizedTensor, _Float8AQTTensorImpl):
        return new_type(state).value
    raise pickle.UnpicklingError(f"unsupported tensor subclass {new_type}")


def _unsupported(*_args, **_kw):  # the wrapper-subclass constructor: _rebuild_from_type_v2 never calls it
    raise pickle.UnpicklingError("unexpected wrapper subclass")


_STORAGE_DTYPES = {
    "torch.UntypedStorage": torch.uint8,
    "torch.storage.UntypedStorage": torch.uint8,
    "torch.BFloat16Storage": torch.bfloat16,
    "torch.HalfStorage": torch.float16,
    "torch.FloatStorage": torch.float32,
    "torch.ByteStorage": torch.uint8,
}


class _StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


_GLOBALS = {
    "collections.OrderedDict": collections.OrderedDict,
    "torch._utils._rebuild_tensor_v2": torch._utils._rebuild_tensor_v2,
    "torch._utils._rebuild_tensor_v3": torch._utils._rebuild_tensor_v3,
    "torch._tensor._rebuild_from_type_v2": _rebuild_from_type_v2,
    "torch._utils._rebuild_wrapper_subclass": _unsupported,
    "torch.serialization._get_layout": torch.serialization._get_layout,
    "torch.device": torch.device,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float8_e4m3fn": FP8,
    "torch.uint8": torch.uint8,
    "torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor": _AffineQuantizedTensor,
    "torchao.dtypes.floatx.float8_layout.Float8AQTTensorImpl": _Float8AQTTensorImpl,
    "torchao.dtypes.floatx.float8_layout.Float8Layout": _Float8Layout,
    "torchao.quantization.quant_primitives.ZeroPointDomain": lambda v: v,
    **{k: _StorageType(v) for k, v in _STORAGE_DTYPES.items()},
}


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, storage):
        super().__init__(fh)
        self.storage = storage

    def find_class(self, module, name):
        try:
            return _GLOBALS[f"{module}.{name}"]
        except KeyError:
            raise pickle.UnpicklingError(f"checkpoint references unsupported global {module}.{name}") from None

    def persistent_load(self, pid):
        kind, stype, key, _location, _numel = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id {kind!r}")
        dtype = stype.dtype if isinstance(stype, _StorageType) else torch.uint8
        return self.storage(str(key), dtype)


def load_torchao_fp8(path: str) -> Dict[str, Union[torch.Tensor, Fp8Tensor]]:
    """State dict of a ``torch.save`` zip checkpoint, FP8 weights as ``Fp8Tensor``. Tensors are CPU views
    into a read-only memory map of the file (move the ones you keep to the device)."""
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl = next(n for n in names if n.endswith("/data.pkl") or n == "data.pkl")
    prefix = pkl[: -len("data.pkl")]
    fh = open(path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    def storage(key: str, dtype: torch.dtype):
        info = zf.getinfo(f"{prefix}data/{key}")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"compressed storage {key}")
        nlen, elen = struct.unpack("<HH", mm[info.header_offset + 26 : info.header_offset + 30])
        start = info.header_offset + 30 + nlen + elen
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only buffer: the tensors are never written
            raw = torch.frombuffer(mm, dtype=torch.uint8, count=info.file_size, offset=start)
        return torch.storage.TypedStorage(wrap_storage=raw.untyped_storage(), dtype=dtype, _internal=True)

    with zf.open(pkl) as f:
        state = _Unpickler(f, storage).load()
    return state
