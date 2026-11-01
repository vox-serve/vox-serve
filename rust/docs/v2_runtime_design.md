# VoxServe v2 runtime: asynchronous lanes and SM partitions

How the pieces of the Rust runtime's asynchronous engine fit together, and the flags that control it.

## Overview

```
HTTP / bench ─► Scheduler (one host thread per GPU)
                 │  select_lm()          select_detok_v2()        select_urgent()
                 ▼                       ▼                        ▼
           LM lane (stream)        Detok lane (stream)       Urgent lane (high-priority stream)
           lm_decode[bs, mode]     detok[bd, mode]           detok_urgent[bd ≤ 8, 0]
           lm_depth[bs, mode]      gated on LM events        own arenas + codec work state
                 └── green-context SM partitions, baked into each graph ──┘   all SMs
```

* **Two lanes, one batch in flight each.** `step_async` (rust/src/scheduler.rs) collects whichever
  lane's completion event fired, launches the next LM step if the LM lane is idle and the next detok
  batch if the detok lane is idle, and otherwise blocks on the lane(s) in flight. The LM lane never
  waits for the host to read the previous step: rows feed back `last_tok` on the GPU.
* **SM modes.** Mode 0 runs both lanes on the whole GPU (ordinary streams that share SMs). Mode i > 0
  is a green-context split created at start-up (`--sm-modes 24,48,68` = detok SM counts). Every decode
  and detok graph is captured once per mode on that mode's stream, so the partition is baked into the
  graph (verified: a graph captured on a green-context stream keeps its SM set when the raw graph is
  re-instantiated by Rust and launched on an ordinary stream). Choosing a partition = choosing which
  graph to launch; nothing is resized on the hot path. Prefill graphs exist only for mode 0.
* **Split LM steps.** `--lm-segments depth` captures a Qwen3-TTS step as two graphs, backbone + codebook-0
  head (`lm_decode`) and the depth transformer (`lm_depth`). `--lm-segments layers:<k>` additionally
  splits the backbone into groups of k layers (`lm_decode` = group 0, `lm_seg [size, mode, i]` = group i,
  sharing group 0's attention plan); rows travel between groups in a static buffer. The LM lane records
  the gate event before the graph the gate names. Other programs opt in by implementing `decode_head` /
  `decode_depth` (depth split) and `backbone_range` (layer groups).
* **Urgent lane** (`--urgent-lane`). A third lane with its own input / output arenas, its own graphs
  (batch 1–8, whole GPU, own memory pool) and, for Qwen3-TTS, its own codec work state, on a
  high-priority stream. First chunks and chunks whose playback deadline is within `--dec-urgent-ms` go
  there at once, even while a long background detok batch runs on the partition (non-streaming
  requests have no playback deadline and are never urgent); a request never has two chunks in flight
  (stateful codecs decode in order). Supported where the detokenizer keeps no shared scratch state
  between calls, or keeps one per lane: Qwen3-TTS, Orpheus, Zonos, CSM.
* **GPU-side ordering.** `cudaStreamWaitEvent` only, no host waits: a detok batch can wait for the
  latest `depth_start` (`--gate depth`) or the LM's `done`; the LM lane can wait for the detok lane's
  `done` (exclusive prefill / urgent batches).
* **Tracing.** `--trace-dir` brackets every graph launch with timing events and writes one JSON line per
  launch (`lane, stage, key=[size, mode], n, t0, t1`) plus one line per scheduling decision. The same
  tracer covers the v1 (synchronous) path, so both are measured the same way.
  `rust/scripts/v2_trace_report.py` turns a trace into busy time, overlap and per-stage durations.

## Policy knobs (`vox-serve` flags of the Rust runtime)

| Flag | Effect |
|---|---|
| `--async-engines` | Asynchronous LM and detok lanes (implies `--overlap-detok`); mode 0 only unless `--sm-modes` |
| `--sm-modes 24,48,68` | Green-context splits by detok SM count (implies `--async-engines`) |
| `--sm-mode <m>` / `auto:<rows>:<m>` / `adaptive` | Fixed mode; mode `m` once at least `rows` requests are decoding; or the mode with the lowest predicted time per step from latencies measured online per (mode, size), exploring unmeasured modes every 8 steps |
| `--lm-segments depth` / `layers:<k>` | Split Qwen3-TTS steps before the depth transformer, and optionally into k-layer groups |
| `--gate depth` / `layer:<i>` | Non-urgent detok batches start only once the LM's latest step reached its depth graph / layer group i |
| `--dec-batch N` | Continuation chunks wait until at least N are ready, then take all ready (first chunks and near-deadline chunks never wait) |
| `--dec-max-batch N` | Largest non-urgent batch: bounds how long an urgent chunk waits behind a background batch |
| `--dec-flush-ms`, `--dec-urgent-ms` | Deadline margin that flushes a partial batch / makes a chunk urgent |
| `--prefill-exclusive` | Prefill steps never overlap a detok batch |
| `--urgent-exclusive` | Batches with urgent chunks run on all SMs while the LM lane waits |
| `--stream-priorities` | LM stream at high priority, detok stream at low priority |
| `--urgent-lane` | Third detok lane for urgent chunks on all SMs, high priority (implies `--async-engines`) |
| `--slack-urgency` | First chunks are due `--deadline-startup-ms` after arrival and compete by deadline (instead of always urgent); the urgency margin adds the measured latency of the largest detok batch |
| `--scheduler-type deadline` | With `--async-engines`: the deadline policy's LM selection (prefill deferral, `--deadline-max-pending-chunks`) with v2 detok selection |
| `--compile-detok` | torch.compile the detokenizer before capture (fuses elementwise / copy work) |
| `--enable-torch-compile` | torch.compile the LM pieces (now including the split depth graph) |
| `--torch-compile-targets a,b` | Restrict it to these LM functions (`embed`, `backbone`, `decode_tokens`, `decode_head`, `decode_depth`, `backbone_range`) |
| `--time-share` | Strict time sharing on the v2 lanes: LM steps and detok batches alternate on the whole GPU, detok batching still applies (the "batching without colocation" comparison point) |
| `--max-tokens N` | Per-request token limit (prompt included) instead of the model's default |
| `--max-detok-batch-size N` | Largest captured detok batch (default: max batch size); bounds codec graph memory |
| `--trace-dir DIR` | Per-launch GPU timing trace |

Detok selection (`select_detok_v2`, `v2::detok_take`): candidates are sorted urgent first (first chunk
or playback deadline within `--dec-urgent-ms`), then by playback deadline. Urgent chunks go at once,
padded with other ready chunks up to their graph bucket; otherwise the batch waits for `--dec-batch`
chunks unless a deadline comes due (`--dec-flush-ms` plus the measured batch latency) or the LM has
nothing left to produce. One chunk per request per batch keeps stateful codecs in order.

A partition switch happens only while the detok lane is idle (a batch keeps the mode it started in).

## Fusion

* `--compile-detok`: torch.compile of the detokenizer before capture. Qwen3-TTS: `decode_chunk` only
  (compiling the whole detok step made Inductor copy the entire per-slot state store on every call);
  flow-matching decoders (CosyVoice2, Chatterbox, GLM-4-Voice, Step-Audio-2): the flow estimator
  (`forward` / `forward_chunk`), since the HiFT vocoder keeps NumPy-backed state Dynamo cannot guard;
  other models: `detokenize`.
* Qwen3-TTS codec: the RVQ codebooks are normalized once at start-up instead of on every call.
* `--fuse-depth` (Qwen3-TTS): the depth transformer with merged Q/K/V and gate/up GEMMs, grouped-query
  SDPA without materialized repeated KV heads, and each step's output head as a static view instead of
  an `index_select` copy.
* `--fuse-talker` (Qwen3-TTS): talker layers with merged Q/K/V and gate/up GEMMs and the post-attention
  residual add fused into its RMSNorm (FlashInfer `fused_add_rmsnorm`).
* `--fuse-lm` (Orpheus, CSM, CosyVoice2, Chatterbox, Step-Audio-2, Voxtral): sibling `q/k/v` and
  `gate/up` projections become one GEMM each (`vox_serve/fusion/projections.py`). Model forwards are untouched: the
  first projection computes the merged GEMM and its siblings return their slice when called with the same
  input tensor (anything else falls back to its own weight slice). Token-identical to the unfused model.
  Ignored with `--enable-torch-compile`. Zonos and GLM-4-Voice already use merged projections.
* `--enable-torch-compile` covers `decode_head` / `decode_depth` / `backbone_range` when the step is split.
  With `--fuse-talker`, the compiled talker uses a plain add + RMSNorm that Inductor fuses (the FlashInfer
  fused op has no fake-tensor kernel).

## Not done yet

* Segmenting the codec into stage graphs (a codec batch is one graph, so it cannot be interrupted;
  `--urgent-lane` sidesteps this by running urgent chunks next to it instead).
* Memory: every SM mode captures its own codec graphs and warms them up at the largest batch. With
  Step-Audio-2's flow decoder at batch 32 a second mode does not fit in 80 GB (`--max-detok-batch-size 16`
  does).
* An urgent lane for the other models: Voxtral and Step-Audio-2 keep one shared codec work cache (needs a
  second one, as done for Qwen3-TTS); the flow-matching decoders' state handling is not yet audited.
* Layer groups for models other than Qwen3-TTS, and mode switches inside a step (each step uses one
  mode; switching mid-step would need the codec lane to switch at the same boundary).
* An elastic persistent decoder kernel (research option in the original plan).
