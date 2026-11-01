"""Static buffers, FlashInfer attention and CUDA-graph capture for both serving runtimes.

The Rust runtime drives every captured graph through three kinds of memory exported from here:

* the *input arena*: one contiguous device buffer holding every per-step input (token ids, positions,
  page tables, sampling parameters, ...). Rust keeps a pinned host mirror with the same layout and
  uploads a whole step with a single ``cudaMemcpyAsync``.
* the *output arena*: one contiguous device buffer with every per-step output (sampled tokens, PCM),
  downloaded with a single ``cudaMemcpyAsync``.
* FlashInfer workspaces: Rust re-plans attention with FlashInfer's own host scheduler
  (``PrefillPlan``, linked into the Rust binary) and writes the result straight into the workspace
  that the captured kernels read from.

For the Rust runtime (``build_engine``) graphs are captured with ``keep_graph=True``: Rust
instantiates and launches them from the raw ``cudaGraph_t`` handle and never calls
``CUDAGraph.replay``, so torch never instantiates them. The Python runtime (``ProgramWorker``) uses
the builder in-process and launches graphs with ``CUDAGraph.replay``; torch instantiates each of its
graphs at the end of the capture, so no request pays for it.
"""

import importlib
import json
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import flashinfer
import torch

from ..utils import get_logger

logger = get_logger(__name__)

_DTYPE_NAMES = {
    torch.int32: "i32",
    torch.int64: "i64",
    torch.float32: "f32",
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.int16: "i16",
    torch.uint8: "u8",
    torch.bool: "bool",
}


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class Arena:
    """A contiguous device buffer carved into named, typed fields."""

    def __init__(self, name: str, device: torch.device, align: int = 128):
        self.name = name
        self.device = device
        self.align = align
        self._fields: Dict[str, Tuple[int, Tuple[int, ...], torch.dtype]] = {}
        self._size = 0
        self.storage: Optional[torch.Tensor] = None
        self._views: Dict[str, torch.Tensor] = {}

    def add(self, name: str, shape, dtype: torch.dtype) -> None:
        assert self.storage is None, "arena already finalized"
        assert name not in self._fields, f"duplicate arena field {name}"
        shape = tuple(int(s) for s in shape)
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        offset = _round_up(self._size, self.align)
        self._fields[name] = (offset, shape, dtype)
        self._size = offset + nbytes

    def finalize(self) -> None:
        self.storage = torch.zeros(_round_up(max(self._size, 1), self.align), dtype=torch.uint8, device=self.device)
        for name, (offset, shape, dtype) in self._fields.items():
            nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            self._views[name] = self.storage[offset : offset + nbytes].view(dtype).view(shape)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._views[name]

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def export(self) -> Dict[str, Any]:
        return {
            "ptr": self.storage.data_ptr(),
            "size": self.storage.numel(),
            "fields": {
                name: {"offset": off, "shape": list(shape), "dtype": _DTYPE_NAMES[dtype]}
                for name, (off, shape, dtype) in self._fields.items()
            },
        }


def add_attention_fields(arena: Arena, prefix: str, max_batch: int, max_rows: int, max_pages: int) -> None:
    """Page-table fields shared by every attention wrapper of one attention family."""
    arena.add(f"{prefix}qo_indptr", (max_batch + 1,), torch.int32)
    arena.add(f"{prefix}kv_indptr", (max_batch + 1,), torch.int32)
    arena.add(f"{prefix}kv_indices", (max_pages,), torch.int32)
    arena.add(f"{prefix}kv_last_page_len", (max_batch,), torch.int32)
    # Destination (page, offset-in-page) of every row's new K/V entry.
    arena.add(f"{prefix}kv_page", (max_rows,), torch.int64)
    arena.add(f"{prefix}kv_off", (max_rows,), torch.int64)


class RtAttention:
    """Graph-capturable FlashInfer paged attention with runtime-provided metadata.

    The attention model code sees (``vox_serve.flashinfer_utils.FlashInferWrapper``: ``set_kv_cache`` +
    ``run``). Page tables live in the input arena and the plan is recomputed by the runtime before every
    launch.
    """

    def __init__(
        self,
        kind: str,
        arena: Arena,
        prefix: str,
        float_workspace: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        batch_size: int,
        max_rows: int,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        assert kind in ("prefill", "decode")
        self.kind = kind
        self.prefix = prefix
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.batch_size = batch_size
        self.max_rows = max_rows if kind == "prefill" else batch_size
        self.causal = causal if kind == "prefill" else False
        self.dtype = dtype
        self.float_workspace = float_workspace

        qo = arena[f"{prefix}qo_indptr"][: batch_size + 1]
        kv_indptr = arena[f"{prefix}kv_indptr"][: batch_size + 1]
        kv_indices = arena[f"{prefix}kv_indices"]
        kv_last = arena[f"{prefix}kv_last_page_len"][:batch_size]
        self.kv_page = arena[f"{prefix}kv_page"][: self.max_rows]
        self.kv_off = arena[f"{prefix}kv_off"][: self.max_rows]

        if kind == "prefill":
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                float_workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=qo,
                paged_kv_indptr_buf=kv_indptr,
                paged_kv_indices_buf=kv_indices,
                paged_kv_last_page_len_buf=kv_last,
                backend="fa2",
            )
        else:
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                float_workspace,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=kv_indptr,
                paged_kv_indices_buffer=kv_indices,
                paged_kv_last_page_len_buffer=kv_last,
            )
        self._plan_for_capture()

    def _plan_for_capture(self) -> None:
        """Plan once on representative shapes; in CUDA-graph mode FlashInfer keeps the plan layout
        (padded batch size, split-kv, workspace offsets) fixed, so every runtime re-plan is compatible."""
        bs = self.batch_size
        kv_indptr = torch.arange(bs + 1, dtype=torch.int32)
        kv_indices = torch.zeros(bs, dtype=torch.int32)
        kv_last = torch.ones(bs, dtype=torch.int32)
        if self.kind == "prefill":
            per = max(1, self.max_rows // bs)
            qo = torch.arange(bs + 1, dtype=torch.int32) * per
            qo[-1] = self.max_rows
            self.wrapper.plan(
                qo_indptr=qo,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=kv_last,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                page_size=self.page_size,
                causal=self.causal,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
        else:
            self.wrapper.plan(
                indptr=kv_indptr,
                indices=kv_indices,
                last_page_len=kv_last,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
        torch.cuda.synchronize()

    # Kept out of torch.compile: Inductor functionalizes this scatter into a copy of the whole cache.
    @torch.compiler.disable
    def set_kv_cache(self, kv_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """kv_cache: (n_pages, 2, page_size, n_kv_heads, head_dim); k, v: (rows, n_kv_heads, head_dim)."""
        rows = k.shape[0]
        page = self.kv_page[:rows]
        off = self.kv_off[:rows]
        kv_cache[page, 0, off] = k
        kv_cache[page, 1, off] = v

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        return self.wrapper.run(q, kv_cache)

    def export(self) -> Dict[str, Any]:
        w = self.wrapper
        return {
            "kind": self.kind,
            "prefix": self.prefix,
            "batch_size": self.batch_size,
            "total_num_rows": int(w._plan_info[1].item()) if torch.is_tensor(w._plan_info) else int(w._plan_info[1]),
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "float_ws_ptr": w._float_workspace_buffer.data_ptr(),
            "float_ws_size": w._float_workspace_buffer.numel() * w._float_workspace_buffer.element_size(),
            "int_ws_ptr": w._int_workspace_buffer.data_ptr(),
            "int_ws_size": w._int_workspace_buffer.numel() * w._int_workspace_buffer.element_size(),
            "plan_info": [int(x) for x in (w._plan_info.tolist() if torch.is_tensor(w._plan_info) else w._plan_info)],
        }


@dataclass
class GraphEntry:
    stage: str
    key: List[int]
    graph: Optional[torch.cuda.CUDAGraph]  # None: run ``fn`` eagerly (EagerSpecBuilder)
    attention: Optional[RtAttention]
    fn: Optional[Callable[[], None]] = None

    def launch(self) -> None:
        if self.graph is not None:
            self.graph.replay()
        else:
            self.fn()


# Green-context splits per (device, detok SMs), shared by every engine of the process: engines that
# put their detokenizer on the same partition of a GPU then run on the same SMs.
_GREEN_SPLITS: Dict[Any, Any] = {}


def green_split(device: torch.device, dec_sms: int):
    """(streams, resources) of a two-way green-context split of `device`: [detok SMs, the rest]."""
    key = (torch.device(device).index, int(dec_sms))
    if key not in _GREEN_SPLITS:
        from flashinfer.green_ctx import split_device_green_ctx

        # FlashInfer wraps the green streams with torch.cuda.get_stream_from_external, which labels
        # them with the current device: make it the split's device.
        with torch.cuda.device(device):
            _GREEN_SPLITS[key] = split_device_green_ctx(torch.device(device), 1, int(dec_sms))
    return _GREEN_SPLITS[key]


class EngineSpecBuilder:
    """Captures a program's graphs, collects its arenas and ``meta`` and keeps the backing tensors alive.

    ``keep_graph``: keep each graph's ``cudaGraph_t`` uninstantiated for ``export`` (the Rust runtime);
    otherwise torch instantiates it at the end of the capture.
    """

    def __init__(
        self,
        device: torch.device,
        detok_device: Optional[torch.device] = None,
        overlap: bool = False,
        sm_modes: Optional[List[int]] = None,
        urgent_lane: bool = False,
        detok_sms: Optional[int] = None,
        keep_graph: bool = False,
    ):
        self.device = device
        self.keep_graph = keep_graph
        self.inputs = Arena("inputs", device)
        self.outputs = Arena("outputs", device)
        # Disaggregation: the detokenizer runs with its own arenas and stream, asynchronously to the
        # LM, either on its own device or (overlap) on the same one.
        self.detok_device = detok_device if detok_device is not None else device
        self.disaggregated = self.detok_device != device or overlap
        if self.disaggregated:
            self.dt_inputs = Arena("dt_inputs", self.detok_device)
            self.dt_outputs = Arena("dt_outputs", self.detok_device)
        else:
            self.dt_inputs, self.dt_outputs = self.inputs, self.outputs
        # Urgent detok lane: its own arenas (and, in the program, its own scratch state), so urgent
        # chunks can run while a background detok batch is still in flight.
        self.urgent_lane = urgent_lane
        if urgent_lane:
            assert self.disaggregated, "the urgent lane needs a separate detok lane"
            self.du_inputs = Arena("du_inputs", self.detok_device)
            self.du_outputs = Arena("du_outputs", self.detok_device)
        self.graphs: List[GraphEntry] = []
        self.meta: Dict[str, Any] = {}
        self.keep_alive: List[Any] = []
        self.pools: Dict[str, Any] = {}
        # SM modes: mode 0 runs every stage on the whole GPU; mode i > 0 splits the GPU into two green
        # contexts, the LM on one and the detokenizer on the other. Every graph is captured once per
        # mode on that mode's stream, which bakes the SM partition into the graph; the runtime then
        # chooses a partition by choosing which graph to launch.
        total = torch.cuda.get_device_properties(device).multi_processor_count
        self.modes: List[Dict[str, Any]] = [dict(lm_sms=total, dec_sms=total, lm_stream=None, dec_stream=None)]
        if detok_sms and self.detok_device != device:
            # Codec on a partition of another GPU (shared with every engine that uses the same split,
            # e.g. the LM engine of that GPU): mode 0's detok graphs are captured on that partition.
            streams, res = green_split(self.detok_device, detok_sms)
            self.modes[0]["dec_stream"] = streams[0]
            self.modes[0]["dec_sms"] = res[0].sm.smCount
            logger.info("detok on %s: %d-SM partition", self.detok_device, res[0].sm.smCount)
        if sm_modes:
            assert self.disaggregated and self.detok_device == device, "SM modes need the same-device overlap layout"
            for dec_sms in sm_modes:
                streams, res = green_split(device, dec_sms)
                self.keep_alive.append((streams, res))
                self.modes.append(
                    dict(
                        lm_sms=res[1].sm.smCount, dec_sms=res[0].sm.smCount, lm_stream=streams[1], dec_stream=streams[0]
                    )
                )
                logger.info(
                    "SM mode %d: LM %d SMs / detok %d SMs", len(self.modes) - 1, res[1].sm.smCount, res[0].sm.smCount
                )

    def pool(self, family: str):
        if family not in self.pools:
            self.pools[family] = torch.cuda.graph_pool_handle()
        return self.pools[family]

    def capture(
        self,
        stage: str,
        key: List[int],
        fn: Callable[[], None],
        attention: Optional[RtAttention] = None,
        family: str = "lm",
        warmup: int = 2,
        device: Optional[torch.device] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        device = device if device is not None else self.device
        stream = stream if stream is not None else torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph(keep_graph=self.keep_graph)
        with torch.cuda.device(device), torch.cuda.graph(graph, pool=self.pool(family), stream=stream):
            fn()
        torch.cuda.synchronize(device)
        self.graphs.append(GraphEntry(stage=stage, key=list(key), graph=graph, attention=attention, fn=fn))
        logger.info(
            "captured graph %s %s (allocated %.2f GiB, reserved %.2f GiB)",
            stage,
            key,
            torch.cuda.memory_allocated(device) / 2**30,
            torch.cuda.memory_reserved(device) / 2**30,
        )

    def export(self) -> str:
        spec = {
            "inputs": self.inputs.export(),
            "outputs": self.outputs.export(),
            "graphs": [
                {
                    "stage": g.stage,
                    "key": g.key,
                    "graph": g.graph.raw_cuda_graph(),
                    "attention": g.attention.export() if g.attention is not None else None,
                }
                for g in self.graphs
            ],
            "meta": dict(self.meta, sm_modes=[dict(lm_sms=m["lm_sms"], dec_sms=m["dec_sms"]) for m in self.modes]),
        }
        if self.disaggregated:
            spec["detok_inputs"] = self.dt_inputs.export()
            spec["detok_outputs"] = self.dt_outputs.export()
            spec["detok_device"] = self.detok_device.index
        if self.urgent_lane:
            spec["urgent_inputs"] = self.du_inputs.export()
            spec["urgent_outputs"] = self.du_outputs.export()
        return json.dumps(spec)


class EagerSpecBuilder(EngineSpecBuilder):
    """Keeps every stage as a function run eagerly instead of a CUDA graph (the Python runtime's
    ``--disable-cuda-graph``, e.g. to debug model code)."""

    def capture(
        self,
        stage: str,
        key: List[int],
        fn: Callable[[], None],
        attention: Optional[RtAttention] = None,
        family: str = "lm",
        warmup: int = 2,
        device: Optional[torch.device] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.graphs.append(GraphEntry(stage=stage, key=list(key), graph=None, attention=attention, fn=fn))


_ENGINE_KEEPALIVE: List[Any] = []


def _check_flashinfer_version() -> None:
    """Rust plans attention with the FlashInfer headers it was built against (rust/third_party/flashinfer);
    the captured kernels come from the installed flashinfer-python, so both must be the same version.
    Without the extension (Python runtime) FlashInfer plans for itself and there is nothing to check."""
    try:
        built = importlib.import_module("vox_serve._rt").FLASHINFER_VERSION
    except ImportError:
        return
    if built and built != flashinfer.__version__:
        raise RuntimeError(
            f"flashinfer-python {flashinfer.__version__} is installed, but the Rust runtime was built against "
            f"FlashInfer {built}: install flashinfer-python=={built}"
        )


def build_engine(config_json: str) -> str:
    """Entry point called by the Rust runtime (once per engine). Returns the engine spec as JSON."""
    from ..model import get_program

    _check_flashinfer_version()
    config = json.loads(config_json)
    # One engine per device (data parallel); hooks address their engine by index.
    config["engine_index"] = len(_ENGINE_KEEPALIVE)
    index = int(config.get("device", 0))
    torch.cuda.set_device(index)
    device = torch.device(f"cuda:{index}")
    with torch.no_grad():
        program = get_program(config["model"])(config, device)
        sm_modes = [int(x) for x in config.get("sm_modes") or []]
        urgent = bool(config.get("urgent_lane")) and program.supports_urgent_lane
        if config.get("urgent_lane") and not urgent:
            logger.warning("%s has no urgent detok lane; ignoring urgent_lane", type(program).__name__)
        builder = EngineSpecBuilder(
            device,
            program.detok_device,
            overlap=bool(config.get("overlap_detok")) or bool(sm_modes) or urgent,
            sm_modes=sm_modes,
            urgent_lane=urgent,
            detok_sms=config.get("detok_sms"),
            keep_graph=True,
        )
        program.build(builder)
    _ENGINE_KEEPALIVE.append((program, builder))
    return builder.export()


_HOOK_LOCK = threading.Lock()


def call_hook(engine: int, name: str, args_json: str) -> str:
    """Per-request preprocessing that still needs the PyTorch model (e.g. encoding reference audio).

    Called by the Rust runtime from request-handling threads, never from the engine step loop.
    Dispatches to ``program.hook_<name>(**args)`` of engine ``engine`` and returns the result as JSON.
    """
    program = _ENGINE_KEEPALIVE[engine][0]
    fn = getattr(program, f"hook_{name}", None)
    if fn is None:
        raise ValueError(f"{type(program).__name__} has no hook '{name}'")
    # Hooks may call into C libraries that release the GIL and are not thread-safe (espeak).
    with _HOOK_LOCK, torch.no_grad(), torch.cuda.device(program.device):
        return json.dumps(fn(**json.loads(args_json)))
