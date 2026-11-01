"""The worker of the Python runtime: runs a model's program (``vox_serve.model.<model>.program``).

The program is the model implementation both runtimes share. It owns the weights, the KV cache and the
per-request device state, builds prompts (``preprocess``) and captures every step as CUDA graphs over
its input / output arenas (``vox_serve.rt.program``). This worker fills the arenas from the scheduler's
``Request``s, replays the graphs (or runs the captured functions eagerly with ``--disable-cuda-graph``)
and turns their outputs back into request state, as the Rust runtime does for its own scheduler.

LM steps feed sampled tokens back on the device (rows with ``fb = 1`` read the request slot's last
token), so a step can be launched before the previous step's results have reached the CPU; with
``--async-scheduling`` they are collected one step later by the coroutine ``run_lm_*`` returns.
"""

import asyncio
import logging
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..model import get_program
from ..requests import Request
from ..rt.core import Arena, EagerSpecBuilder, EngineSpecBuilder, GraphEntry
from ..rt.prompt import TOKEN_AUDIO, TOKEN_STOP, ModelSpec
from ..utils import get_logger


class _HostArena:
    """Pinned host mirror of a device arena, uploaded with one copy. The copy reads the pinned memory
    asynchronously, so the mirror is written again only once the previous upload has run."""

    def __init__(self, arena: Arena):
        self.dev = arena
        self.storage = torch.zeros(arena.storage.numel(), dtype=torch.uint8).pin_memory()
        self.f: Dict[str, np.ndarray] = {}
        for name, (offset, shape, dtype) in arena._fields.items():
            if dtype != torch.bfloat16:
                nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                self.f[name] = self.storage[offset : offset + nbytes].view(dtype).view(shape).numpy()
        self.uploaded: Optional[torch.cuda.Event] = None

    def acquire(self) -> Dict[str, np.ndarray]:
        if self.uploaded is not None:
            self.uploaded.synchronize()
        return self.f

    def upload(self) -> None:
        self.dev.storage.copy_(self.storage, non_blocking=True)

    def mark_uploaded(self) -> None:
        """After every copy that reads host memory for this step (upload, attention plan) is enqueued."""
        if self.uploaded is None:
            self.uploaded = torch.cuda.Event()
        self.uploaded.record()


@dataclass(eq=False)
class _Step:
    """An LM step in flight: its requests (prefill entries first) and pinned copies of its outputs."""

    reqs: List[Request]
    n_prefill: int
    next_pos: List[int]  # each request's next position after this step
    tok: torch.Tensor
    kind: torch.Tensor
    done: torch.cuda.Event


def _pick(graphs: List[Tuple[int, GraphEntry]], n: int) -> Optional[Tuple[int, GraphEntry]]:
    """The smallest graph that fits ``n`` rows / requests."""
    return next(((k, g) for k, g in graphs if k >= n), None)


def final_chunk(done_lm: bool, start: int, n_frames: int, interval: int, step: int, holds_back: bool) -> bool:
    """Whether the codec chunk at frame ``start`` is the last one of a request with ``n_frames`` audio frames:
    its tokens are all generated (``done_lm``) and the chunk emits the rest of its audio. Chunks start
    ``step`` frames apart. A chunk emits its whole ``interval``-frame window, unless the codec takes
    ``dt_last`` (``holds_back``): then each chunk keeps look-ahead audio back for the next one, and only a
    chunk that no other follows (``start + step >= n_frames``, run with ``dt_last = 1``) emits all of it.
    The request is finished once its last chunk has been decoded."""
    return done_lm and start + (step if holds_back else interval) >= n_frames


class ProgramWorker:
    def __init__(
        self,
        model_name: str,
        max_batch_size: int,
        max_num_pages: Optional[int] = None,
        page_size: int = 64,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        repetition_penalty: float = None,
        repetition_window: int = None,
        cfg_scale: float = None,
        greedy: bool = False,
        enable_cuda_graph: bool = True,
        enable_nvtx: bool = False,
        enable_torch_compile: bool = False,
        unroll_depth_cuda_graph: bool = False,
        detokenizer_device: Optional[str] = None,
        dp_rank: int = 0,
        dp_size: int = 1,
        detokenize_interval: int = None,
        program_options: Optional[Dict[str, Any]] = None,
    ):
        base_logger = get_logger(__name__)
        if dp_size > 1:
            self.logger = logging.LoggerAdapter(base_logger, {"dp_rank": dp_rank})
            self.logger.process = lambda msg, kwargs: (f"[DP {dp_rank}/{dp_size}] {msg}", kwargs)
        else:
            self.logger = base_logger
        self.nvtx_enabled = enable_nvtx

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        config: Dict[str, Any] = {"model": model_name, "device": 0, "max_batch_size": max_batch_size}
        if detokenizer_device is not None:
            if torch.cuda.device_count() < 2:
                raise RuntimeError(
                    f"Disaggregation requires at least 2 GPUs, but only {torch.cuda.device_count()} available."
                )
            config["detok_device"] = torch.device(detokenizer_device).index
        overrides = {
            "page_size": page_size,
            "max_num_pages": max_num_pages,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "cfg_alpha": cfg_scale,
            "detok_interval": detokenize_interval,
        }
        config.update({k: v for k, v in overrides.items() if v is not None})
        config.update(program_options or {})  # kernel fusion (vox_serve.fusion)
        if enable_torch_compile:
            config["torch_compile"] = True
        if repetition_window is not None:
            self.logger.warning("--repetition-window is ignored: the penalty applies to every token generated so far")

        with torch.no_grad():
            self.program = get_program(model_name)(config, self.device)
            builder = (EngineSpecBuilder if enable_cuda_graph else EagerSpecBuilder)(
                self.device, self.program.detok_device
            )
            self.program.build(builder)
        self.detok_device = self.program.detok_device
        if self.detok_device != self.device:
            self.logger.info(f"Disaggregation mode: LM on {self.device}, detokenizer on {self.detok_device}")

        m = self.meta = dict(builder.meta)
        self.spec = ModelSpec.from_meta(m)
        self.sampling = self.spec.sampling
        if greedy:
            self.sampling.temperature = 0.0
        self.c = self.spec.n_codebooks
        stages: Dict[str, List[Tuple[int, GraphEntry]]] = {}
        for g in builder.graphs:
            if len(g.key) < 2 or g.key[1] == 0:  # mode 0: the whole GPU
                stages.setdefault(g.stage, []).append((g.key[0], g))
        for v in stages.values():
            v.sort(key=lambda t: t[0])
        self.prefill_graphs = stages["lm_prefill"]
        self.decode_graphs = stages["lm_decode"]
        self.detok_graphs = stages["detok"]

        self.max_batch_size = int(m["max_batch_size"])
        self.page_size = int(m["page_size"])
        self.max_context = int(m["max_context"])
        self.detok_samples = int(m["detok_samples"])
        self.audio_len_in_row = bool(m.get("audio_len_in_row"))
        self.max_prompt_len = min(self.prefill_graphs[-1][0], self.max_context - 1)
        self.dummy_slot = int(m["max_slots"])
        self.detok_slots = int(m.get("detok_slots") or m["max_slots"])
        self.free_slots = list(range(int(m["max_slots"])))[::-1]
        self.free_detok_slots = list(range(self.detok_slots))[::-1] if m.get("detok_stateful") else None
        scratch = int(m["scratch_page"])
        self.free_pages = [p for p in range(int(m["max_num_pages"]))[::-1] if p != scratch]

        self.lm_host = _HostArena(builder.inputs)
        self.out_tok, self.out_kind = builder.outputs["out_tok"], builder.outputs["out_kind"]
        self.out_pool: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.pending: "deque[_Step]" = deque()
        with torch.cuda.device(self.detok_device):
            self.dt_host = _HostArena(builder.dt_inputs)
            self.audio = builder.dt_outputs["audio"]
            self.audio_h = torch.zeros(self.audio.shape, dtype=self.audio.dtype).pin_memory()
            self.audio_done = torch.cuda.Event()
        self.detok_holds_back = "dt_last" in self.dt_host.f  # see final_chunk
        self.builder = builder  # keeps the graphs and arenas alive
        self.logger.info(
            "%s ready: %d decode / %d prefill / %d detok graphs%s",
            type(self.program).__name__,
            len(self.decode_graphs),
            len(self.prefill_graphs),
            len(self.detok_graphs),
            "" if enable_cuda_graph else " (run eagerly)",
        )

    # ---- what the schedulers read --------------------------------------------------------------

    @property
    def detokenize_interval(self) -> int:
        return int(self.meta["detok_interval"])

    @property
    def detokenize_overlap(self) -> int:
        return int(self.meta.get("detok_overlap") or 0)

    @property
    def supports_audio_input(self) -> bool:
        # The program's ``preprocess`` decides; requests it rejects fail.
        return True

    @property
    def available_batch_sizes(self) -> List[int]:
        return [k for k, _ in self.decode_graphs]

    @property
    def prefill_graph_batch_size(self) -> int:
        return self.max_batch_size

    @property
    def cuda_graph_seq_len_buckets(self) -> List[int]:
        return [k for k, _ in self.prefill_graphs]

    def tokenize_stream_text(self, text: str) -> List[int]:
        """Text tokens of an input-streaming text chunk."""
        return self.program.tokenize_stream_text(text)

    # ---- LM steps ------------------------------------------------------------------------------

    def prepare_lm_inputs(self, lm_requests: List[Request], detokenize_requests: List[Request]) -> Optional[Dict]:
        """Selects what of ``lm_requests`` runs this step (one prefill, if its prompt is built and a slot
        and KV pages are free, plus decodes that fit) and writes its inputs to the host arena."""
        for req in detokenize_requests:
            req.audio_decode_idx = req.next_audio_decode_idx.copy()

        max_rows = self.prefill_graphs[-1][0]
        reqs: List[Request] = []
        rows = 0
        for req in lm_requests:
            if req.done_lm_prefill or req.done_lm_generation:
                continue
            if self._admit(req):
                reqs.append(req)
                rows = req.input_length
                break
            if not req.done_lm_generation:
                break  # waits for a slot or KV pages (requests that failed are skipped)
        n_prefill = len(reqs)
        for req in lm_requests:
            if not req.done_lm_prefill or req.done_lm_generation:  # (the prefill above is not done_lm_prefill yet)
                continue
            if len(reqs) >= self.max_batch_size or (n_prefill and rows >= max_rows):
                break
            # Finished: the results of its last step are still being collected.
            if req.next_position_id > self.spec.max_tokens or req.next_position_id >= self.max_context:
                continue
            reqs.append(req)
            rows += 1
        if not reqs:
            return None

        size, graph = _pick(self.prefill_graphs if n_prefill else self.decode_graphs, rows if n_prefill else len(reqs))
        b = {"is_prefill": n_prefill > 0, "requests": reqs, "n_prefill": n_prefill, "graph": graph, "bucket": size}
        self._fill_lm(b)
        return b

    def run_lm_prefill(self, requests: List[Request], lm_inputs: Optional[Dict]) -> Optional[Coroutine]:
        return self._run_lm(lm_inputs)

    def run_lm_decode(self, requests: List[Request], lm_inputs: Optional[Dict]) -> Optional[Coroutine]:
        return self._run_lm(lm_inputs)

    def _fail(self, req: Request, reason: str) -> None:
        self.logger.warning("request %s failed: %s", req.request_id, reason)
        req.done_lm_prefill = req.done_lm_generation = True
        req.finish_reason = f"error: {reason}"
        self.free_kv_cache(req)

    def _admit(self, req: Request) -> bool:
        """Builds the prompt (once) and takes the request's slots and KV pages; False if it has to wait."""
        if req.lm_prompt is None:
            try:
                with torch.no_grad():
                    if req.is_input_streaming and self.spec.text_stream is None:
                        raise ValueError("input streaming is not supported by this model")
                    if req.is_input_streaming and req.first_text_token is not None:
                        req.lm_prompt = self.program.preprocess_streaming(
                            req.prompt, req.audio_path, req.model_kwargs, req.first_text_token
                        )
                    else:
                        req.is_input_streaming = False  # no text token to start from: the whole text is known
                        req.lm_prompt = self.program.preprocess(req.prompt, req.audio_path, req.model_kwargs)
            except Exception as e:  # noqa: BLE001  a bad request fails alone
                self._fail(req, str(e))
                return False
            req.input_length = len(req.lm_prompt.tokens) // self.c
            if req.input_length == 0 or req.input_length > self.max_prompt_len:
                self._fail(req, f"prompt length {req.input_length} out of range (max {self.max_prompt_len})")
                return False
        positions = min(max(req.input_length, self.spec.max_tokens + 1), self.max_context)
        n_pages = -(-positions // self.page_size)
        if not self.free_slots or self.free_detok_slots == [] or len(self.free_pages) < n_pages:
            return False
        req.slot = self.free_slots.pop()
        if self.free_detok_slots is not None:
            req.detok_slot = self.free_detok_slots.pop()
        req.kv_pages = self.free_pages[len(self.free_pages) - n_pages :]
        del self.free_pages[len(self.free_pages) - n_pages :]
        req.seed = random.getrandbits(32)
        return True

    def _next_text_token(self, req: Request, pad: int, eos: int) -> int:
        """Input streaming: the next queued text token, then EOS once the text is complete, else pad."""
        try:
            tok = req.pending_text_tokens.get_nowait()
            req.text_token_cursor += 1
            return tok
        except Exception:
            if req.text_complete and not req.eos_injected:
                req.eos_injected = True
                return eos
            return pad

    def _fill_lm(self, b: Dict) -> None:
        f = self.lm_host.acquire()
        c, ps, dummy = self.c, self.page_size, self.dummy_slot
        prefill_step = b["n_prefill"] > 0
        n_entries = b["graph"].attention.batch_size if prefill_step else b["bucket"]
        ts = self.spec.text_stream
        sp = self.sampling
        qo, kv, idx, last = [0], [0], [], []
        b["next_pos"] = []
        row = 0
        for e, req in enumerate(b["requests"]):
            prefill = e < b["n_prefill"]
            start, n = (0, req.input_length) if prefill else (req.next_position_id, 1)
            tok = f["tok"]
            if prefill:
                tok[row : row + n] = np.asarray(req.lm_prompt.tokens, dtype=np.int32).reshape(n, c)
            elif self.spec.decode_row is not None:
                tok[row] = self.spec.decode_row
            else:
                tok[row] = 0
            if not prefill and ts is not None and req.is_input_streaming:
                tok[row, ts[0]] = self._next_text_token(req, ts[1], ts[2])
            if "mask" in f:
                if prefill:
                    f["mask"][row : row + n] = np.asarray(req.lm_prompt.mask, dtype=np.int32).reshape(n, c)
                else:
                    f["mask"][row] = self.spec.decode_mask
            f["fb"][row : row + n] = 0 if prefill else 1
            f["row_slot"][row : row + n] = req.slot
            pos = np.arange(start, start + n)
            f["pos"][row : row + n] = pos
            pages = np.asarray(req.kv_pages, dtype=np.int64)
            f["lm_kv_page"][row : row + n] = pages[pos // ps]
            f["lm_kv_off"][row : row + n] = pos % ps
            kv_len = start + n
            n_pages = -(-kv_len // ps)
            idx.extend(req.kv_pages[:n_pages])
            last.append(kv_len - (n_pages - 1) * ps)
            qo.append(row + n)
            kv.append(len(idx))
            f["last_row"][e] = row + n - 1
            f["slot"][e] = req.slot
            f["active"][e] = 1
            f["first"][e] = int(prefill)
            f["temp"][e], f["top_k"][e], f["top_p"][e] = sp.temperature, sp.top_k, sp.top_p
            f["min_p"][e], f["rep_pen"][e] = sp.min_p, sp.repetition_penalty
            f["seed"][e], f["step"][e] = req.seed, req.lm_steps
            row += n
            if prefill:
                req.done_lm_prefill = True
            req.next_position_id = kv_len
            req.lm_steps += 1
            b["next_pos"].append(kv_len)
        # Padding entries: decode graphs give each one row on the scratch page, prefill graphs none.
        for e in range(len(b["requests"]), n_entries):
            if prefill_step:
                last.append(1)
                f["last_row"][e] = 0
            else:
                f["tok"][row] = 0
                if "mask" in f:
                    f["mask"][row] = 0
                f["fb"][row], f["row_slot"][row], f["pos"][row] = 0, dummy, 0
                f["lm_kv_page"][row], f["lm_kv_off"][row] = 0, 0
                idx.append(0)
                last.append(1)
                f["last_row"][e] = row
                row += 1
            qo.append(row)
            kv.append(len(idx))
            f["slot"][e], f["active"][e], f["first"][e] = dummy, 0, 0
        if row < b["bucket"]:  # padding rows of prefill graphs: computed, never attended to, KV to scratch
            n_rows = b["bucket"]
            f["tok"][row:n_rows] = 0
            if "mask" in f:
                f["mask"][row:n_rows] = 0
            f["fb"][row:n_rows], f["row_slot"][row:n_rows], f["pos"][row:n_rows] = 0, dummy, 0
            f["lm_kv_page"][row:n_rows], f["lm_kv_off"][row:n_rows] = 0, 0
        f["lm_qo_indptr"][: len(qo)] = qo
        f["lm_kv_indptr"][: len(kv)] = kv
        f["lm_kv_indices"][: len(idx)] = idx
        f["lm_kv_last_page_len"][: len(last)] = last
        b["plan"] = (qo, kv, idx, last)

    def _plan(self, b: Dict) -> None:
        a = b["graph"].attention
        qo, kv, idx, last = (torch.tensor(x, dtype=torch.int32) for x in b["plan"])
        if a.kind == "prefill":
            a.wrapper.plan(
                qo_indptr=qo,
                paged_kv_indptr=kv,
                paged_kv_indices=idx,
                paged_kv_last_page_len=last,
                num_qo_heads=a.num_qo_heads,
                num_kv_heads=a.num_kv_heads,
                head_dim_qk=a.head_dim,
                page_size=a.page_size,
                causal=a.causal,
                q_data_type=a.dtype,
                kv_data_type=a.dtype,
            )
        else:
            a.wrapper.plan(
                indptr=kv,
                indices=idx,
                last_page_len=last,
                num_qo_heads=a.num_qo_heads,
                num_kv_heads=a.num_kv_heads,
                head_dim=a.head_dim,
                page_size=a.page_size,
                q_data_type=a.dtype,
                kv_data_type=a.dtype,
            )

    @torch.no_grad()
    def _run_lm(self, b: Optional[Dict]) -> Optional[Coroutine]:
        if b is None:
            return None
        self.nvtx_range_push("lm_prefill" if b["is_prefill"] else "lm_decode")
        self.lm_host.upload()
        self._plan(b)
        self.lm_host.mark_uploaded()
        b["graph"].launch()
        n = len(b["requests"])
        tok, kind = self.out_pool.pop() if self.out_pool else self._new_out_buffers()
        tok[:n].copy_(self.out_tok[:n], non_blocking=True)
        kind[:n].copy_(self.out_kind[:n], non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        step = _Step(b["requests"], b["n_prefill"], b["next_pos"], tok, kind, done)
        self.pending.append(step)
        self.nvtx_range_pop()
        return self._collect(step)

    def _new_out_buffers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(self.out_tok.shape, dtype=self.out_tok.dtype).pin_memory(),
            torch.zeros(self.out_kind.shape, dtype=self.out_kind.dtype).pin_memory(),
        )

    async def _collect(self, step: _Step) -> None:
        while not step.done.query():
            await asyncio.sleep(0)
        # Steps are collected in launch order, whichever coroutine runs first.
        while step in self.pending:
            s = self.pending.popleft()
            s.done.synchronize()
            self._consume(s)
            self.out_pool.append((s.tok, s.kind))

    def _consume(self, s: _Step) -> None:
        c, preseed = self.c, self.spec.preseed
        toks, kinds = s.tok.numpy(), s.kind.numpy()
        for e, req in enumerate(s.reqs):
            if e < s.n_prefill and req.lm_prompt is not None and req.lm_prompt.features is not None:
                req.lm_prompt.features.release()  # the prefill has read the prompt's feature rows
            if req.done_lm_generation:
                continue  # launched before its previous step's results were collected
            frame = toks[e, :c].copy()
            req.lm_output_tokens.append(frame)
            kind = int(kinds[e])
            if kind == TOKEN_AUDIO:
                if not req.lm_output_audio_tokens and preseed > 0:
                    req.lm_output_audio_tokens.extend(np.zeros(c, dtype=frame.dtype) for _ in range(preseed))
                req.lm_output_audio_tokens.append(frame)
            elif kind == TOKEN_STOP:
                req.done_lm_generation, req.finish_reason = True, "stop_id_encountered"
            if s.next_pos[e] > self.spec.max_tokens:
                req.done_lm_generation, req.finish_reason = True, "max_tokens_reached"
            elif not req.done_lm_generation and s.next_pos[e] >= self.max_context:
                req.done_lm_generation, req.finish_reason = True, "max_context_reached"
            if req.done_lm_generation:
                self._release_lm(req)

    # ---- detokenization ------------------------------------------------------------------------

    @torch.no_grad()
    def run_detokenize(self, requests: List[Request]) -> None:
        chunks = [
            (req, start)
            for req in requests
            if not req.done_all
            for start in req.audio_decode_idx
            if start < len(req.lm_output_audio_tokens)
        ]
        max_bd = self.detok_graphs[-1][0]
        if self.free_detok_slots is not None:
            # A stateful codec decodes a request's chunks in order, so a batch holds at most one chunk
            # per request: the k-th chunks of the requests go in the k-th batches.
            waves: Dict[int, List[Tuple[Request, int]]] = {}
            seen: Dict[str, int] = {}
            for req, start in chunks:
                k = seen.get(req.request_id, 0)
                seen[req.request_id] = k + 1
                waves.setdefault(k, []).append((req, start))
            batches = [w[i : i + max_bd] for w in waves.values() for i in range(0, len(w), max_bd)]
        else:
            batches = [chunks[i : i + max_bd] for i in range(0, len(chunks), max_bd)]
        if batches:
            self.nvtx_range_push("detokenize")
            for batch in batches:
                self._detokenize(batch)
            self.nvtx_range_pop()
        for req in requests:
            if req.done_lm_generation and (
                not req.audio_decode_idx or self._final_chunk(req, req.audio_decode_idx[-1])
            ):
                req.done_all = True

    def _final_chunk(self, req: Request, start: int) -> bool:
        iv, n_frames = self.detokenize_interval, len(req.lm_output_audio_tokens)
        step = iv - self.detokenize_overlap
        return final_chunk(req.done_lm_generation, start, n_frames, iv, step, self.detok_holds_back)

    def _detokenize(self, chunks: List[Tuple[Request, int]]) -> None:
        bd, graph = _pick(self.detok_graphs, len(chunks))
        iv = self.detokenize_interval
        with torch.cuda.device(self.detok_device):
            f = self.dt_host.acquire()
            if "dt_len" in f and "dt_last" in f:  # valid frames of the window, and whether no chunk follows
                for k in range(bd):
                    if k < len(chunks):
                        req, start = chunks[k]
                        n_frames = len(req.lm_output_audio_tokens)
                        f["dt_len"][k] = min(start + iv, n_frames) - start
                        f["dt_last"][k] = int(self._final_chunk(req, start))
                    else:
                        f["dt_len"][k], f["dt_last"][k] = 0, 0
            if "dt_slot" in f and "dt_reset" in f:
                for k in range(bd):
                    if k < len(chunks):
                        req, start = chunks[k]
                        f["dt_slot"][k], f["dt_reset"][k] = req.detok_slot, int(start == 0)
                    else:
                        f["dt_slot"][k], f["dt_reset"][k] = self.detok_slots, 1
            dt = f["dt_tok"]
            for k, (req, start) in enumerate(chunks):
                frames = np.stack(req.lm_output_audio_tokens[start : start + iv])
                dt[k, : len(frames)] = frames
                dt[k, len(frames) :] = frames[-1]  # repeat the last frame
            self.dt_host.upload()
            self.dt_host.mark_uploaded()
            graph.launch()
            self.audio_h[: len(chunks)].copy_(self.audio[: len(chunks)], non_blocking=True)
            self.audio_done.record()
            self.audio_done.synchronize()

        n = self.detok_samples
        audio = self.audio_h.view(-1, n).numpy()
        for k, (req, start) in enumerate(chunks):
            real = min(len(req.lm_output_audio_tokens) - start, iv)
            if self.audio_len_in_row:
                length, skip = min(max(int(audio[k, n - 1]), 0), n - 1), 0
            else:
                length = n if real >= iv else int(n * (real - 0.5) / iv)
                # The pre-seeded frames are trimmed after the tail trim (an early stop may leave nothing).
                skip = min(self.spec.preseed * n // iv, length) if start == 0 else 0
            req.output_audio.put(audio[k, skip:length].tobytes())

    # ---- request resources ---------------------------------------------------------------------

    def _release_lm(self, req: Request) -> None:
        if req.slot is not None:
            self.free_slots.append(req.slot)
            req.slot = None
        if req.kv_pages:
            self.free_pages.extend(req.kv_pages)
            req.kv_pages = []

    def free_kv_cache(self, req: Request) -> None:
        """Releases everything the request holds (called once it is done)."""
        self._release_lm(req)
        if req.detok_slot is not None:
            self.free_detok_slots.append(req.detok_slot)
            req.detok_slot = None
        if req.lm_prompt is not None:
            if req.lm_prompt.features is not None:
                req.lm_prompt.features.release()
            req.lm_prompt = None  # also drops what the prompt holds (``Prompt.hold``)

    def nvtx_range_push(self, name: str) -> None:
        if self.nvtx_enabled:
            torch.cuda.nvtx.range_push(name)

    def nvtx_range_pop(self) -> None:
        if self.nvtx_enabled:
            torch.cuda.nvtx.range_pop()
