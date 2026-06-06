# Plan: Remove the per-step `torch.nonzero` EOS sync and pipeline the Orpheus decode loop

- **Status:** Phases 0–2 + **Phase 3a (carried-batch sync restructure) IMPLEMENTED** (2026-06-06, uncommitted on `users/garv/orpheus_perf`). Verified: changed files `py_compile` clean; two adversarial reviews PASS (deferred-EOS primitive + carried-batch loop). NOT yet: runtime/audio-byte-identical validation (Phase 4). The full design + caveats for 3a live at `~/.claude/plans/sorted-weaving-galaxy.md`.
- **Branch:** `users/garv/orpheus_perf`
- **Owner:** garv
- **Created:** 2026-06-06
- **Related memory:** `orpheus-nonzero-eos-keystone-sync`, `deferred-eos-pipelines-async-not-sync`
- **Evidence traces:** `../harness/profiles/vox_host_label_10rps_.sqlite` (host-labeled, the primary one), `../harness/profiles/vox_10rps_.sqlite` (pre-label baseline)

---

## 1. Objective

Eliminate the per-step host/device synchronization on the Orpheus decode critical path so that
host-side scheduling (request intake, batch selection, FlashInfer `plan()`, and especially the
prefill **tokenizer**) overlaps the in-flight GPU decode instead of running strictly after it.

This is a **TTFA / streaming-viability** fix, not a raw decode-throughput fix: the decode compute
itself is GPU-bound (~5.9 ms/step at bs≈80–90) and does not get faster. The win is reclaiming the
GPU-idle inter-step transition (union GPU idle ≈ **18.8%**) and hiding the prefill tokenizer
(~0.8 ms pure-CPU on prefill steps) that currently inflates time-to-first-audio.

## 2. Background & evidence (from nsys analysis)

Measured on `vox_host_label_10rps_.sqlite` (10 rps, **sync** scheduler, the mode the harness runs):

- Step period ≈ **12 ms** median; union GPU busy ≈ **81%**, idle ≈ **18.8%**.
- The decode/vocoder overlap already works: stream 7 (LM) bubbles are largely filled by stream 13
  (SNAC vocoder). The remaining union-idle is the **inter-step transition** (decode i done, vocoder
  done, decode i+1 not yet launched because the CPU is finishing the EOS sync + collect + emit +
  next-step prep + relaunch).
- `host_run_sampling_task` NVTX range (wraps `asyncio.run(task)`): median **7.17 ms**, of which
  **~5.89 ms (82%) is a single `cudaStreamSynchronize`** (585 instances). Per-range CUDA API
  fingerprint = 1× `cudaLaunchKernel` + 1× `cudaMemcpyAsync` + 1× `cudaStreamSynchronize` = the
  signature of `torch.nonzero`.
- Next-step host phases (`host_prepare_requests`, `host_select_*`, `host_prepare_lm_inputs`) overlap
  the decode kernels **~0%** → strictly serial today.
- Prefill `host_prepare_lm_inputs` is **~1.08 ms** median (max ~2.0 ms), of which **~808 µs is pure
  CPU with no CUDA in flight** = the HuggingFace tokenizer in `model.preprocess`. Decode-step
  `host_prepare_lm_inputs` is only ~251 µs. (No extra NVTX marker needed to confirm this.)

> The `host_*` NVTX ranges referenced above were added in this branch
> (`worker/base.py: nvtx_range`/`nvtx_mark`; scheduler `_step`/`_step_async`). Re-enable with the
> existing `enable_nvtx` flag when re-profiling.

## 3. Root cause (the keystone)

The only host dependency in the decode loop is **EOS detection**:

- `OrpheusModel.sampling` (`vox_serve/model/orpheus.py:419`) returns `task = update_req_states()`,
  an `async def` (`orpheus.py:449-477`) whose body runs
  `stop_indices = torch.nonzero(stop_mask, ...)` at **`orpheus.py:451`** (plus `idx.item()` at
  `orpheus.py:459` when an EOS is actually hit).
- `torch.nonzero` has a **data-dependent output shape**, so PyTorch launches a count kernel,
  `cudaMemcpyAsync`'s the count to host, and `cudaStreamSynchronize`s to read it. Because the
  default stream is FIFO and the decode-graph replay + sampling kernels were enqueued *before* the
  count kernel, that synchronize **drains the entire decode→sample chain** → its duration ≈ the
  decode latency (~5.9 ms), not nonzero's own cost.
- Sync `_step` runs this **inline** via `asyncio.run(task)` (`scheduler/base.py:225`), right after
  launching decode(i) → it eats the full decode latency on the CPU thread every step, so no
  next-step work can overlap.

Everything else in the loop is GPU-resident: `req.input_tokens = output_ids[i:i+1]` and
`lm_output_audio_tokens.append(...)` are GPU views (`orpheus.py:447,454-455`), and the next decode's
`input_ids` buffer is filled GPU→GPU (`cuda_graph_worker.py` `run_lm_decode`). So removing this one
sync makes the autoregressive feed fully GPU-resident.

**Current-tree note:** as of this branch there is **no** deferred-EOS machinery in git
(no `apply_pending_stops` / `stop_event` / `VOX_DEFER_EOS`). The profiled build is the **inline
`torch.nonzero`** baseline. The deferred-EOS work described in
`deferred-eos-pipelines-async-not-sync` lived elsewhere and is not present here; this plan
re-implements it on this branch.

## 4. Options considered & decision

### Route 1 — Deferred GPU stop-mask + CUDA event (CHOSEN)
Keep `stop_mask` on GPU, async-copy it to a pinned host buffer, record an event, and read it at the
**top of the next step** (event already complete → ~µs). Apply the 1-step fix-up then.

### Route 2 — Coarse-interval / GPU done-counter, sync rarely (REJECTED)
Sync only every N steps or when a cheap GPU counter trips.
- Finished requests keep occupying batch slots + KV pages until the next sync → wasted decode at
  bs≈80–90 **and** delayed admission of the next queued request → **hurts TTFA**, the opposite of
  the goal.
- Still needs a host read of the counter (just rarer); data-dependent control flow.
- Unbounded EOS latency → variable over-generation + trimming complexity + audio-artifact risk.

### Decision & justification
**Route 1.** It preserves exact per-step EOS semantics (bounded 1-step latency, a known-correct
fix-up — pop the trailing EOS token before any detokenize selection sees it), replaces the
data-dependent `torch.nonzero` with a **fixed-size async copy**, is mode-independent, and reuses the
existing pinned-ring pattern (`cuda_graph_worker.py:87-97`). Route 2 trades the wrong axis (sync
frequency) and regresses TTFA via slot/KV occupancy.

## 5. Critical constraint — deferral alone does NOT pipeline sync mode

Hard lesson from `deferred-eos-pipelines-async-not-sync`: a 1-step deferred read pipelines the
**async** loop but **not** the **sync** loop. In sync `_step`, if you wait the stop at the top of
the next step and then do all prep + decode enqueue *after* it, there is no decode in flight during
the wait → still ~neutral. The complete sync fix requires the **carried-batch structure**: enqueue
decode(N) *before* waiting stop(N-1), with batch selection lagged one step. That is exactly the
`_step_async` shape (`scheduler/base.py:283-298`, `asyncio.gather(run_model(), run_scheduling())`).

Therefore the plan is staged: Phase 1–2 deliver the primitive (and immediately fix async); Phase 3
delivers the sync pipelining (or adopts async). Phase 4 measures whether the overlap actually
flipped, to avoid repeating the known dead-end.

## 6. Implementation plan

### Phase 0 — A/B guard
Gate the new path behind an env flag (e.g. `VOX_DEFER_EOS`, default **off** initially) so old-vs-new
run in the same binary, mirroring the `VOX_B3_OFFLOAD` pattern (`scheduler/base.py:142`). Remove the
flag once validated.

### Phase 1 — Make the EOS read asynchronous (the primitive)
- **`OrpheusModel.sampling`** (`orpheus.py:419`): delete the `update_req_states` coroutine. Do the
  sync-free, GPU-resident bookkeeping eagerly — the `input_tokens` / `lm_output_audio_tokens`
  GPU-view appends (`orpheus.py:447,454-455`) and the host-only `max_tokens` check. Compute
  `stop_mask = output_ids[:,0] == stop_token_id` and **return the GPU mask** instead of a task. No
  `torch.nonzero`, no `.item()`.
- **Worker** `run_lm_decode` / `run_lm_prefill` (`cuda_graph_worker.py:1150` / `1010`; base
  fallback `worker/base.py:478` / `399`): after sampling, copy `stop_mask` into a **rotating pinned
  host buffer** and `record()` a `torch.cuda.Event` on the default stream (non-blocking). Return a
  small handle `(mask_slot, event)`. Reuse the pinned-ring idea from `_audio_pinned_ring`
  (`cuda_graph_worker.py:87-97`).
- **`apply_deferred_stops(prev_requests, handle)`** (new; worker or model method): `event.synchronize()`
  (already complete a step later → ~µs), read the pinned mask, and for each set bit apply the
  fix-up — pop the trailing EOS token from `lm_output_audio_tokens`, set `done_lm_generation` and
  `finish_reason="stop_id_encountered"`. Idempotent w.r.t. the `max_tokens` path.

### Phase 2 — Wire deferral into both loops
- **Sync `_step`** (`scheduler/base.py:157`): carry `(prev_lm_requests, prev_handle)` as loop state;
  call `apply_deferred_stops` at the **top** of the step (before `_select_*`); delete the inline
  `asyncio.run(task)` (`scheduler/base.py:225`).
- **Async `_step_async`** (`scheduler/base.py:228`): route through the same `apply_deferred_stops` so
  both modes share one mechanism (drops the async-only coroutine special-casing).

After Phase 1–2: async mode is fixed; sync mode is correct but still ~serial (expected) → Phase 3.

### Phase 3 — Pipeline the sync loop (the actual sync win) — choose ONE
- **3a (restructure):** reshape `_step` to the `_step_async` carried-batch shape — enqueue decode(N)
  before applying stop(N-1), with batch selection lagged one step (a request that stopped at N-1
  takes one optimistic extra decode at N, then drops; consistent with the Phase-1 pop). Reference:
  `run_model()` / `run_scheduling()` at `scheduler/base.py:283-298`.
- **3b (adopt async, lower risk):** validate `--async-scheduling` (harness defaults to sync;
  `async_scheduling=False` at `scheduler/base.py:42`) end-to-end on the sweep. If async is correct
  and faster, switch the harness and skip 3a.
- **Decision gate:** pick 3a vs 3b from the Phase-4 metrics on a 3b trial. Prefer 3b if async passes
  correctness + perf; fall back to 3a if async has event-loop/throughput issues.

## 7. Correctness considerations & edge cases

- **1-step EOS latency / trimming:** the EOS token is appended at step N and popped at the top of
  N+1 by `apply_deferred_stops`, before `_select_detokenize_requests` runs — so detokenize never
  emits EOS-token audio. Verify against the detokenize selection logic
  (`scheduler/base.py:_select_detokenize_requests`, ~`:397`) and the `done_all` gating.
- **Optimistic over-generation (Phase 3 only):** with lagged selection a stopped request decodes one
  extra token. Confirm it is dropped before detokenize and not emitted.
- **Final-step flush:** ensure the last in-flight mask is applied so completion messages send and KV
  pages free (`free_kv_cache`); don't strand the last request.
- **`max_tokens` termination:** stays host-side and position-based (no sync) — keep idempotent with
  the deferred stop path so a request isn't double-finalized.
- **Other models:** `qwen3_tts` / depth-transformer models have their own `sampling` and a depth
  loop (`has_depth_transformer`); scope this change to Orpheus first, keep the flag model-agnostic
  but default-off elsewhere.

## 8. Validation (re-profile with host labels)

Capture a `vox_host_label`-style run with `enable_nvtx` on, then confirm:
1. `host_run_sampling_task`'s ~5.9 ms `cudaStreamSynchronize` is gone (shrinks to a µs-scale
   already-complete event wait). Query: per-range sum of `*Synchronize` CUDA-runtime time inside the
   range (the analysis used `CUPTI_ACTIVITY_KIND_RUNTIME` joined to `StringIds`).
2. `host_prepare_requests` / `host_select_*` / `host_prepare_lm_inputs` overlap with stream-7 decode
   kernels flips from ~0% toward high (Phase 3 only). Query: intersect each host NVTX range with
   `CUPTI_ACTIVITY_KIND_KERNEL` where `streamId=7`.
3. Union GPU idle drops below 18.8%; TTFA (first-audio latency) improves on the harness metric.
4. **Audio output is byte-identical** to the baseline build — the correctness gate (no dropped/extra
   tokens). The prior deferred-EOS work was byte-identical; match that.

Reusable nsys query primitives (Python `sqlite3`): worker thread = `globalTid` of any
`lm_decode_bs%` NVTX row; step cadence = spacing between `host_step_begin` marks; GPU streams are
`7` (LM) and `13` (SNAC vocoder).

## 9. Risks & mitigations

- **Async underdelivers / has quirks** → keep Phase 1–2 shippable on their own; Phase 3a is the
  fallback if 3b (async) fails.
- **Deferral changes audio** → byte-identical gate in Phase 4; flag-gated A/B.
- **Detokenize selection sees the EOS token** → apply stops strictly before selection; covered by
  Phase 7 checks.
- **Scope creep into other models** → Orpheus-only first, default-off elsewhere.

## 10. Expected payoff (calibration)

- Decode compute (~5.9 ms/step) is the floor and does **not** shrink.
- Recoverable: ~0.6 ms/decode-step of host scheduling + ~1–2 ms/prefill-step of tokenizer (the TTFA
  lever) + part of the ~18.8% union GPU-idle transition.
- Net: meaningful TTFA / streaming-viability improvement and lower GPU idle; not a step-time
  transformation.

## 10.5 Implementation notes (what actually landed, 2026-06-06)

Flag `VOX_DEFER_EOS` defaults **ON** (`"1"`; set `=0` for the legacy inline `torch.nonzero` path) — a
deviation from the Phase-0 "default off" wording, chosen so the new path is exercised by default on
this dev branch. The legacy path is the A/B baseline.

- **`OrpheusModel.__init__`** (`orpheus.py`): `self._defer_eos = os.environ.get("VOX_DEFER_EOS","1") != "0"` (added `import os`).
- **`OrpheusModel.sampling`** (`orpheus.py`): coroutine deleted. Eager GPU-resident appends
  (`input_tokens`, `lm_output_tokens`, `lm_output_audio_tokens`) + host-only `max_tokens` + repetition
  cache; computes `stop_mask = output_ids[:,0]==stop_token_id`. Deferred → `return output_ids, stop_mask`;
  legacy → inline `torch.nonzero` + `return output_ids, None`.
- **`vox_serve/worker/base.py`**: new module-level `@dataclass DeferredStopHandle(requests, host_mask, event)`;
  `ModelWorker.__init__` gets `_stop_mask_pinned_ring` (2 pinned bool buffers) + `_stop_mask_pinned_idx`;
  new `_make_stop_handle(requests, stop_mask)` (async pinned copy + `torch.cuda.Event.record()`) and
  `apply_deferred_stops(handle)` (`event.synchronize()`; per set bit pop trailing EOS token + finalize,
  guarded `if not done_lm_generation`). `run_lm_prefill`/`run_lm_decode` non-depth branches:
  `if torch.is_tensor(second): return self._make_stop_handle(requests, second) else: return second`.
- **`cuda_graph_worker.py`**: same tensor→handle dispatch in both `run_lm_*` non-depth `else` branches
  (sets `task` then existing `return task`).
- **`worker/__init__.py`**: exports `DeferredStopHandle`.
- **`scheduler/base.py`**: imports `DeferredStopHandle`; `__init__` carries `self._pending_stop_handle=None`.
  `_step` **renamed `_step_pipelined`** (Phase-3 light rename; `run_forever` updated). Top of step calls
  `apply_deferred_stops(self._pending_stop_handle)` (NVTX `host_apply_deferred_stops`) before selection;
  end of step `if isinstance(task, DeferredStopHandle): self._pending_stop_handle = task` else legacy
  `elif task is not None: asyncio.run(task)`. `_step_async` wired analogously (dropped `asyncio.create_task`;
  `run_scheduling` dispatches handle→`apply_deferred_stops` / coroutine→`await`) — **wired, not validated**.

**Phase 3a (carried-batch sync) landed 2026-06-06 — what's there:**
- `worker/base.py`: `DeferredStopHandle` gained `audio_lens`; `_make_stop_handle` captures per-request
  `len(lm_output_audio_tokens)`; `apply_deferred_stops` truncates `del list[audio_lens[i]-1:]` (removes
  EOS + the optimistic over-generation token) instead of a single `pop()`.
- `scheduler/base.py`: new `_step_carried(handle, lm_requests, detokenize_requests, lm_inputs)` —
  enqueues decode(K) first, then applies stop(K−1) + selects batch(K+1) + `prepare_lm_inputs(K+1)`
  while decode(K) runs (no default-stream sync between enqueue and next-step prep). `run_forever` gets
  an `elif self._carried_sync:` carry-loop. `_select_detokenize_requests` not-done branch subtracts
  `self._overgen_lag` (frontier margin). `__init__` computes `_carried_sync` (env `VOX_CARRIED_SYNC`
  default on AND not async AND `model._defer_eos`), `_overgen_lag` (1 iff carried, else 0 → serial path
  byte-identical), `_idle_sleep_s`. Backpressure: next-step apply event-sync / vocoder collect / idle
  sleep (the decode-only `torch.cuda.synchronize()` was removed). Gated to the deferred path (asserts
  `new_handle` is not a coroutine). `_step_pipelined` retained for A/B.

**Resume here:** Phase 4 validation — run the harness sync sweep with `VOX_CARRIED_SYNC=1` (default) vs
`=0` (serial) vs `VOX_DEFER_EOS=0` (legacy baseline). Confirm audio byte-identical (or compare
post-truncation `lm_output_audio_tokens` streams if the harness isn't seeded/greedy). Re-profile with
`enable_nvtx`: the ~5.9ms `cudaStreamSynchronize` should be gone, and `host_prepare_lm_inputs_*`/
`host_select_*` should now OVERLAP stream-7 decode kernels (~0% → high) — the 3a success signal.
**Open risk to exercise:** caveat G — the optimistic extra decode can request a KV page under full
pressure (`empty_pages.get_nowait()` could raise). Nothing is committed yet.

## 11. Open questions

- Does `--async-scheduling` already pass the harness correctness + perf bar today (would make Phase
  3b a near-free win)?
- Is there a non-Orpheus model in the harness sweep that needs the same treatment, or is Orpheus the
  only `torch.nonzero`-EOS offender?
- Pinned-ring depth: 2 slots (prev/cur) is sufficient for a 1-step defer; confirm no emitter-style
  backlog requires more.
