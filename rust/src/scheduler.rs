//! Request lifecycle, batching policy and per-step input/output marshalling.
//!
//! Mirrors `vox_serve/scheduler/{base,online,offline}.py` + the worker's `prepare_lm_inputs` /
//! `run_detokenize`, but every step is a handful of pinned-memory writes, one FlashInfer plan, one
//! H2D copy, graph launches and one D2H copy.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use bytes::Bytes;
use crossbeam_channel::{Receiver, RecvTimeoutError, TryRecvError};
use tokio::sync::mpsc::UnboundedSender;

use crate::deadline::{self, DeadlineConfig};
use crate::engine::{Engine, Field, Gate, GraphId, HostArena};
use crate::models::{FeatureLease, Model, Prompt, SamplingParams, TextStreamSpec, TokenKind};
use crate::v2::{self, GateMode, V2Config};

#[derive(Debug)]
pub enum Event {
    Audio(Bytes),
    Done(String),
    Error(String),
}

pub struct Submit {
    pub id: String,
    pub prompt: Prompt,
    pub sampling: SamplingParams,
    pub seed: u64,
    pub streaming: bool,
    pub tx: UnboundedSender<Event>,
    /// Text tokens arriving after submission (input streaming); closing the channel ends the text.
    pub text_rx: Option<Receiver<Vec<i32>>>,
}

/// Text still to be fed to an input-streaming request's decode rows.
struct TextIn {
    rx: Receiver<Vec<i32>>,
    queue: VecDeque<i32>,
    complete: bool,
    eos_sent: bool,
}

impl TextIn {
    fn poll(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(t) => self.queue.extend(t),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.complete = true;
                    break;
                }
            }
        }
    }

    /// Decoding waits while the text is incomplete and nothing is queued.
    fn ready(&self) -> bool {
        self.complete || !self.queue.is_empty()
    }

    fn next(&mut self, spec: TextStreamSpec) -> i32 {
        if let Some(t) = self.queue.pop_front() {
            t
        } else if self.complete && !self.eos_sent {
            self.eos_sent = true;
            spec.eos
        } else {
            spec.pad
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Policy {
    /// `input_streaming`: the Python runtime's name; every Rust policy serves input streaming.
    #[value(alias = "input_streaming")]
    Base,
    Online,
    Offline,
    Deadline,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub policy: Policy,
    pub max_prefills_per_step: usize,
    pub deadline: DeadlineConfig,
    pub v2: V2Config,
    /// Requests this engine holds (admitted, not yet finished), published for load-aware dispatch.
    pub load: Arc<AtomicUsize>,
}

struct Req {
    id: String,
    /// Unique per scheduler; identifies the request across `reqs` reshuffles (in-flight detok).
    key: u64,
    tx: UnboundedSender<Event>,
    streaming: bool,
    prompt: Vec<i32>,
    prompt_mask: Option<Vec<i32>>,
    prompt_len: usize,
    /// Feature-pool rows read by the prefill; released once it has run.
    features: Option<Arc<FeatureLease>>,
    /// Model resources the request uses until it finishes (e.g. a voice slot).
    _hold: Option<Arc<dyn std::any::Any + Send + Sync>>,
    sampling: SamplingParams,
    seed: i64,
    text_in: Option<TextIn>,

    slot: Option<i32>,
    /// Detokenizer state slot (stateful codecs only); held until the request finishes.
    dt_slot: Option<i32>,
    pages: Vec<i32>,
    /// Position id of the next LM input row (== number of KV entries written so far).
    next_pos: usize,
    /// Number of tokens sampled so far (RNG step counter).
    steps: i32,
    done_lm: bool,
    finish_reason: &'static str,

    /// Audio frames, flattened `[n_frames, C]`.
    audio: Vec<i32>,
    n_frames: usize,
    /// Start index of the last scheduled detokenization chunk.
    last_detok: Option<usize>,

    received: Instant,
    /// Estimated client playback end; rebased when a late chunk arrives.
    playback_end: Option<Instant>,
    /// Next chunk start after the last emitted chunk (does not advance at launch).
    emitted_next: usize,
    first_send: Option<Instant>,
    total_sent_secs: f64,
    last_chunk_secs: f64,
    pressing: bool,
}

impl Req {
    fn next_detok(&self, step: usize) -> usize {
        self.last_detok.map_or(0, |i| i + step)
    }
}

/// Arena field handles used by the LM / detok stages.
struct Fields {
    tok: Field<i32>,
    fb: Field<i32>,
    row_slot: Field<i32>,
    pos: Field<i32>,
    mask: Option<Field<i32>>,
    qo_indptr: Field<i32>,
    kv_indptr: Field<i32>,
    kv_indices: Field<i32>,
    kv_last_page_len: Field<i32>,
    kv_page: Field<i64>,
    kv_off: Field<i64>,
    last_row: Field<i32>,
    slot: Field<i32>,
    active: Field<i32>,
    first: Field<i32>,
    temp: Field<f32>,
    top_k: Field<i32>,
    top_p: Field<f32>,
    min_p: Field<f32>,
    rep_pen: Field<f32>,
    seed: Field<i64>,
    step: Field<i32>,
    dt_tok: Field<i32>,
    dt_slot: Option<Field<i32>>,
    dt_reset: Option<Field<i32>>,
    dt_len: Option<Field<i32>>,
    dt_last: Option<Field<i32>>,
    out_tok: Field<i32>,
    /// `TokenKind` code of every sampled row, written by the program's graph.
    out_kind: Field<i32>,
    audio: Field<i16>,
}

impl Fields {
    fn resolve(e: &Engine) -> Result<Self> {
        let i = &e.inputs;
        let o = &e.outputs;
        let (di, dout) = e.detok.as_ref().map_or((i, o), |d| (&d.inputs, &d.outputs));
        Ok(Self {
            tok: i.field("tok")?,
            fb: i.field("fb")?,
            row_slot: i.field("row_slot")?,
            pos: i.field("pos")?,
            mask: if i.has("mask") { Some(i.field("mask")?) } else { None },
            qo_indptr: i.field("lm_qo_indptr")?,
            kv_indptr: i.field("lm_kv_indptr")?,
            kv_indices: i.field("lm_kv_indices")?,
            kv_last_page_len: i.field("lm_kv_last_page_len")?,
            kv_page: i.field("lm_kv_page")?,
            kv_off: i.field("lm_kv_off")?,
            last_row: i.field("last_row")?,
            slot: i.field("slot")?,
            active: i.field("active")?,
            first: i.field("first")?,
            temp: i.field("temp")?,
            top_k: i.field("top_k")?,
            top_p: i.field("top_p")?,
            min_p: i.field("min_p")?,
            rep_pen: i.field("rep_pen")?,
            seed: i.field("seed")?,
            step: i.field("step")?,
            dt_tok: di.field("dt_tok")?,
            dt_slot: if di.has("dt_slot") { Some(di.field("dt_slot")?) } else { None },
            dt_reset: if di.has("dt_reset") { Some(di.field("dt_reset")?) } else { None },
            dt_len: if di.has("dt_len") { Some(di.field("dt_len")?) } else { None },
            dt_last: if di.has("dt_last") { Some(di.field("dt_last")?) } else { None },
            out_tok: o.field("out_tok")?,
            out_kind: o.field("out_kind")?,
            audio: dout.field("audio")?,
        })
    }
}

/// Field handles of one detok lane's arenas.
struct DtFields {
    tok: Field<i32>,
    slot: Option<Field<i32>>,
    reset: Option<Field<i32>>,
    len: Option<Field<i32>>,
    last: Option<Field<i32>>,
    audio: Field<i16>,
}

impl DtFields {
    fn resolve(i: &HostArena, o: &HostArena) -> Result<Self> {
        Ok(Self {
            tok: i.field("dt_tok")?,
            slot: if i.has("dt_slot") { Some(i.field("dt_slot")?) } else { None },
            reset: if i.has("dt_reset") { Some(i.field("dt_reset")?) } else { None },
            len: if i.has("dt_len") { Some(i.field("dt_len")?) } else { None },
            last: if i.has("dt_last") { Some(i.field("dt_last")?) } else { None },
            audio: o.field("audio")?,
        })
    }
}

/// What one step runs on the LM.
struct LmBatch {
    graph: GraphId,
    /// Further backbone layer groups of a layer-split step (run after `graph`, before `depth`).
    segs: Vec<GraphId>,
    /// Depth graph of a split step (runs last).
    depth: Option<GraphId>,
    /// Captured size of `graph`: rows (prefill) or batch entries (decode).
    bucket: usize,
    /// Indices into `reqs`, prefills first.
    reqs: Vec<usize>,
    n_prefill: usize,
}

/// An LM step in flight on the asynchronous lane (requests by key: `reqs` may change meanwhile).
struct LmInflight {
    keys: Vec<u64>,
    n_prefill: usize,
    exclusive: bool,
    split: bool,
}

struct DetokChunk {
    req: usize,
    start: usize,
    /// Row of the detok batch.
    row: usize,
}

pub struct Scheduler {
    engine: Engine,
    model: Arc<Model>,
    cfg: SchedulerConfig,
    f: Fields,

    c: usize,
    max_bs: usize,
    max_slots: usize,
    page_size: usize,
    max_context: usize,
    detok_interval: usize,
    detok_step: usize,
    detok_samples: usize,
    /// The last sample of each audio row holds its valid length (variable-length chunks).
    audio_len_in_row: bool,
    /// Samples of audio a chunk advances (playback-time estimates).
    chunk_samples: usize,
    max_detok_bs: usize,
    /// Mode-0 (whole GPU) graph tables.
    decode_graphs: Vec<(usize, GraphId)>,
    prefill_graphs: Vec<(usize, GraphId)>,
    detok_graphs: Vec<(usize, GraphId)>,
    /// Per SM mode: decode, depth (split LM steps only) and detok graph tables.
    decode_by_mode: Vec<Vec<(usize, GraphId)>>,
    depth_by_mode: Vec<Vec<(usize, GraphId)>>,
    /// Per SM mode and decode size: layer groups 1.. of a layer-split step.
    segs_by_mode: Vec<Vec<(usize, Vec<GraphId>)>>,
    detok_by_mode: Vec<Vec<(usize, GraphId)>>,
    detok_buckets: Vec<usize>,

    // asynchronous (v2) lanes
    lm_inflight: Option<LmInflight>,
    /// SM mode of the latest decode-only LM step (prefill steps run on all SMs and leave it as is).
    decode_mode: usize,
    dt_mode: usize,
    dt_exclusive: bool,
    dt_launched: Option<(Instant, usize)>,
    /// Host-observed detok batch latency per bucket (EWMA, seconds).
    dt_cost: Vec<(usize, f64)>,
    calib: v2::Calibration,
    lm_launched: Option<(Instant, usize, usize)>,
    /// Urgent detok lane (`--urgent-lane`): graphs, arena fields and the batch in flight.
    urgent_graphs: Vec<(usize, GraphId)>,
    uf: Option<DtFields>,
    u_inflight: Option<Vec<(u64, usize)>>,

    free_slots: Vec<i32>,
    /// Free detokenizer state slots; `None` for stateless detokenizers.
    free_dt_slots: Option<Vec<i32>>,
    detok_slots: usize,
    free_pages: Vec<i32>,
    reqs: Vec<Req>,
    rx: Receiver<Submit>,
    next_key: u64,
    /// Disaggregated detok batch still running: (request key, chunk start) per row.
    inflight: Option<Vec<(u64, usize)>>,

    prefill_deferred_since: Option<Instant>,
    trace_batches: bool,

    // scratch (reused every step)
    qo_h: Vec<i32>,
    kv_h: Vec<i32>,
    pub stats: Stats,
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub steps: u64,
    pub host_time: Duration,
    pub wait_time: Duration,
    pub prefill_deferrals: u64,
    pub throttled_rows: u64,
}

impl Stats {
    fn delta(&self, before: &Stats) -> Stats {
        Stats {
            steps: self.steps - before.steps,
            host_time: self.host_time - before.host_time,
            wait_time: self.wait_time - before.wait_time,
            prefill_deferrals: self.prefill_deferrals - before.prefill_deferrals,
            throttled_rows: self.throttled_rows - before.throttled_rows,
        }
    }
}

fn meta_usize(m: &serde_json::Value, k: &str) -> Result<usize> {
    m.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| anyhow::anyhow!("meta.{k} missing"))
}

impl Scheduler {
    pub fn new(engine: Engine, model: Arc<Model>, cfg: SchedulerConfig, rx: Receiver<Submit>) -> Result<Self> {
        let m = engine.meta.clone();
        let f = Fields::resolve(&engine)?;
        // Graph keys are [size, sm_mode]; mode 0 runs on the whole GPU.
        let n_modes = m.get("sm_modes").and_then(|v| v.as_array()).map_or(1, |a| a.len().max(1));
        let by_mode = |stage: &str| -> Vec<Vec<(usize, GraphId)>> {
            let mut t = vec![Vec::new(); n_modes];
            for (k, id) in engine.graph_keys(stage) {
                let mode = k.get(1).copied().unwrap_or(0);
                if mode < n_modes {
                    t[mode].push((k[0], id));
                }
            }
            for v in &mut t {
                v.sort();
            }
            t
        };
        let decode_by_mode = by_mode("lm_decode");
        let depth_by_mode = by_mode("lm_depth");
        // Per mode: (decode size, [(group index, graph)]) before sorting by group.
        type SegGroups = Vec<(usize, Vec<(usize, GraphId)>)>;
        let mut segs_by_mode: Vec<SegGroups> = vec![Vec::new(); n_modes];
        for (k, id) in engine.graph_keys("lm_seg") {
            let (bs, mode, gi) = (k[0], k.get(1).copied().unwrap_or(0), k.get(2).copied().unwrap_or(1));
            if mode >= n_modes {
                continue;
            }
            match segs_by_mode[mode].iter_mut().find(|(b, _)| *b == bs) {
                Some((_, v)) => v.push((gi, id)),
                None => segs_by_mode[mode].push((bs, vec![(gi, id)])),
            }
        }
        let segs_by_mode: Vec<Vec<(usize, Vec<GraphId>)>> = segs_by_mode
            .into_iter()
            .map(|t| {
                t.into_iter()
                    .map(|(bs, mut v)| {
                        v.sort();
                        (bs, v.into_iter().map(|(_, id)| id).collect())
                    })
                    .collect()
            })
            .collect();
        let detok_by_mode = by_mode("detok");
        let urgent_graphs = by_mode("detok_urgent")[0].clone();
        let uf = match &engine.urgent {
            Some(u) if !urgent_graphs.is_empty() => Some(DtFields::resolve(&u.inputs, &u.outputs)?),
            _ => None,
        };
        let decode_graphs = decode_by_mode[0].clone();
        let prefill_graphs = by_mode("lm_prefill")[0].clone();
        let detok_graphs = detok_by_mode[0].clone();
        let detok_buckets: Vec<usize> = detok_graphs.iter().map(|(k, _)| *k).collect();
        if cfg.v2.enabled && engine.detok.is_none() {
            bail!("the asynchronous engine loop needs a separate detok lane (--overlap-detok or --codec-gpu)");
        }
        if decode_graphs.is_empty() || prefill_graphs.is_empty() || detok_graphs.is_empty() {
            bail!("engine is missing graphs: {:?}", engine.describe());
        }
        let max_slots = meta_usize(&m, "max_slots")?;
        let max_num_pages = meta_usize(&m, "max_num_pages")?;
        let scratch = meta_usize(&m, "scratch_page")? as i32;
        let detok_interval = meta_usize(&m, "detok_interval")?;
        let overlap = meta_usize(&m, "detok_overlap").unwrap_or(0);
        let stateful = m.get("detok_stateful").and_then(|v| v.as_bool()).unwrap_or(false);
        let detok_slots = meta_usize(&m, "detok_slots").unwrap_or(max_slots);
        if stateful && (f.dt_slot.is_none() || f.dt_reset.is_none()) {
            bail!("stateful detokenizer without dt_slot/dt_reset inputs");
        }
        Ok(Self {
            c: model.n_codebooks(),
            max_bs: meta_usize(&m, "max_batch_size")?,
            max_slots,
            page_size: meta_usize(&m, "page_size")?,
            max_context: meta_usize(&m, "max_context")?,
            detok_interval,
            detok_step: detok_interval - overlap,
            detok_samples: meta_usize(&m, "detok_samples")?,
            audio_len_in_row: m.get("audio_len_in_row").and_then(|v| v.as_bool()).unwrap_or(false),
            chunk_samples: meta_usize(&m, "chunk_samples").or_else(|_| meta_usize(&m, "detok_samples"))?,
            max_detok_bs: detok_graphs.last().unwrap().0,
            decode_graphs,
            prefill_graphs,
            detok_graphs,
            decode_by_mode,
            depth_by_mode,
            segs_by_mode,
            detok_by_mode,
            detok_buckets,
            lm_inflight: None,
            decode_mode: 0,
            dt_mode: 0,
            dt_exclusive: false,
            dt_launched: None,
            dt_cost: Vec::new(),
            calib: v2::Calibration::default(),
            lm_launched: None,
            urgent_graphs,
            uf,
            u_inflight: None,
            free_slots: (0..max_slots as i32).rev().collect(),
            free_dt_slots: stateful.then(|| (0..detok_slots as i32).rev().collect()),
            detok_slots,
            free_pages: (0..max_num_pages as i32).rev().filter(|&p| p != scratch).collect(),
            reqs: Vec::new(),
            rx,
            next_key: 0,
            inflight: None,
            prefill_deferred_since: None,
            trace_batches: std::env::var_os("VOX_BATCH_TRACE").is_some(),
            qo_h: Vec::new(),
            kv_h: Vec::new(),
            stats: Stats::default(),
            engine,
            model,
            cfg,
            f,
        })
    }

    pub fn max_prompt_len(&self) -> usize {
        self.prefill_graphs.last().unwrap().0.min(self.max_context - 1)
    }

    /// Runs until the submission channel is closed and every request has finished.
    pub fn run(mut self) -> Result<()> {
        let trace_queues = std::env::var_os("VOX_SCHEDULING_TRACE").is_some();
        let mut last_trace = Instant::now();
        let mut last_log = Instant::now();
        let mut logged = Stats::default();
        loop {
            if trace_queues && last_trace.elapsed() >= Duration::from_millis(100) {
                let pending: Vec<usize> = (0..self.reqs.len()).map(|i| self.pending_chunks(i)).collect();
                tracing::info!(
                    active = self.reqs.len(),
                    waiting_prefill = self.reqs.iter().filter(|r| r.steps == 0).count(),
                    submission_queue = self.rx.len(),
                    pending_chunks = pending.iter().sum::<usize>(),
                    max_pending_chunks = pending.iter().copied().max().unwrap_or(0),
                    codec_inflight_rows = self.inflight.as_ref().map_or(0, |rows| rows.len()),
                    prefill_deferrals = self.stats.prefill_deferrals,
                    throttled_rows = self.stats.throttled_rows,
                    "scheduling trace"
                );
                last_trace = Instant::now();
            }
            if last_log.elapsed() >= Duration::from_secs(10) && self.stats.steps != logged.steps {
                let d = self.stats.delta(&logged);
                tracing::info!(
                    "engine: {} steps, host {:.1} us/step, gpu wait {:.1} us/step, {} active, {} prefill deferrals, {} throttled rows",
                    d.steps,
                    d.host_time.as_secs_f64() * 1e6 / d.steps as f64,
                    d.wait_time.as_secs_f64() * 1e6 / d.steps as f64,
                    self.reqs.len(),
                    d.prefill_deferrals,
                    d.throttled_rows
                );
                logged = self.stats.clone();
                last_log = Instant::now();
                self.engine.trace_flush()?;
            }
            if self.reqs.is_empty() {
                match self.rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(s) => self.admit(s),
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => return self.shutdown(),
                }
            }
            loop {
                match self.rx.try_recv() {
                    Ok(s) => self.admit(s),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        if self.reqs.is_empty() {
                            return self.shutdown();
                        }
                        break;
                    }
                }
            }
            self.cfg.load.store(self.reqs.len(), Ordering::Relaxed);
            self.step()?;
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        if self.lm_inflight.is_some() {
            self.engine.wait_done()?;
        }
        if self.inflight.is_some() {
            self.engine.detok_wait()?;
        }
        if self.u_inflight.is_some() {
            self.engine.urgent_wait()?;
        }
        self.engine.trace_poll()?;
        self.engine.trace_flush()?;
        self.log_totals();
        Ok(())
    }

    fn log_totals(&self) {
        let s = &self.stats;
        if s.steps > 0 {
            tracing::info!(
                "engine totals: {} steps, host {:.1} us/step, gpu wait {:.1} us/step",
                s.steps,
                s.host_time.as_secs_f64() * 1e6 / s.steps as f64,
                s.wait_time.as_secs_f64() * 1e6 / s.steps as f64
            );
        }
    }

    fn admit(&mut self, s: Submit) {
        let len = s.prompt.tokens.len() / self.c;
        if len == 0 || len > self.max_prompt_len() {
            let _ =
                s.tx.send(Event::Error(format!("prompt length {len} out of range (max {})", self.max_prompt_len())));
            return;
        }
        self.next_key += 1;
        self.reqs.push(Req {
            id: s.id,
            key: self.next_key,
            tx: s.tx,
            streaming: s.streaming,
            prompt: s.prompt.tokens,
            prompt_mask: s.prompt.mask,
            prompt_len: len,
            features: s.prompt.features,
            _hold: s.prompt.hold,
            sampling: s.sampling,
            seed: s.seed as i64,
            text_in: s.text_rx.map(|rx| TextIn { rx, queue: VecDeque::new(), complete: false, eos_sent: false }),
            slot: None,
            dt_slot: None,
            pages: Vec::new(),
            next_pos: 0,
            steps: 0,
            done_lm: false,
            finish_reason: "unknown",
            audio: Vec::new(),
            n_frames: 0,
            last_detok: None,
            received: Instant::now(),
            playback_end: None,
            emitted_next: 0,
            first_send: None,
            total_sent_secs: 0.0,
            last_chunk_secs: 0.0,
            pressing: false,
        });
    }

    fn step(&mut self) -> Result<()> {
        let t0 = Instant::now();
        self.drop_disconnected();
        if self.cfg.policy == Policy::Online {
            self.update_pressing();
        }
        for r in &mut self.reqs {
            if let Some(t) = r.text_in.as_mut() {
                t.poll();
            }
        }
        if self.cfg.v2.enabled {
            return self.step_async(t0);
        }
        if self.engine.detok.is_some() {
            return self.step_disaggregated(t0);
        }
        let (chunks, completions) = self.select_detok();
        let lm = self.select_lm();

        let detok_graph = (!chunks.is_empty()).then(|| pick(&self.detok_graphs, chunks.len())).flatten();
        if let Some(g) = detok_graph {
            self.fill_detok(&chunks, key_of(&self.detok_graphs, g), 0);
        }
        if let Some(b) = &lm {
            self.fill_lm(b)?;
        }

        let ran = lm.is_some() || detok_graph.is_some();
        if ran {
            self.engine.upload_inputs()?;
            if let Some(g) = detok_graph {
                self.engine.launch_n(g, chunks.len())?;
            }
            if let Some(b) = &lm {
                self.engine.launch_n(b.graph, b.reqs.len())?;
            }
            if let Some(b) = &lm {
                self.engine.download_field(self.f.out_tok, b.reqs.len() * self.c)?;
                self.engine.download_field(self.f.out_kind, b.reqs.len())?;
            }
            if !chunks.is_empty() {
                self.engine.download_field(self.f.audio, chunks.len() * self.detok_samples)?;
            }
            self.engine.record_done()?;
            let t1 = Instant::now();
            self.stats.host_time += t1 - t0;
            self.engine.wait_done()?;
            self.stats.wait_time += t1.elapsed();
            self.stats.steps += 1;
        }
        let t2 = Instant::now();
        self.engine.trace_poll()?;

        // Detokenized audio (computed from tokens generated before this step's LM output).
        if !chunks.is_empty() {
            self.emit_audio(&chunks, 0);
        }
        let mut finished: Vec<usize> = completions;
        for (i, ch) in chunks.iter().enumerate() {
            let _ = i;
            let r = &self.reqs[ch.req];
            if r.done_lm && ch.start + self.detok_interval >= r.n_frames && !finished.contains(&ch.req) {
                finished.push(ch.req);
            }
        }
        if let Some(b) = &lm {
            self.consume_lm(b);
        }
        self.finish(finished);
        if ran {
            self.stats.host_time += t2.elapsed();
        } else {
            // Nothing runnable (e.g. waiting for KV pages); avoid a hot spin.
            std::thread::sleep(Duration::from_micros(200));
        }
        Ok(())
    }

    /// Disaggregated step: the detok batch runs on its own device and is only collected once its
    /// event has completed, so LM steps never wait for the codec.
    fn step_disaggregated(&mut self, t0: Instant) -> Result<()> {
        let mut finished = Vec::new();
        if self.inflight.is_some() && self.engine.detok_ready()? {
            let rows = self.inflight.take().unwrap();
            let chunks: Vec<DetokChunk> = rows
                .iter()
                .enumerate()
                .filter_map(|(row, &(key, start))| {
                    // Requests may have been dropped (client gone) meanwhile.
                    self.reqs.iter().position(|r| r.key == key).map(|req| DetokChunk { req, start, row })
                })
                .collect();
            self.emit_audio(&chunks, 0);
            for ch in &chunks {
                let r = &self.reqs[ch.req];
                if r.done_lm && ch.start + self.detok_interval >= r.n_frames {
                    finished.push(ch.req);
                }
            }
        }
        let mut launched = false;
        if self.inflight.is_none() {
            let (chunks, completions) = self.select_detok();
            finished.extend(completions);
            if let Some(g) = (!chunks.is_empty()).then(|| pick(&self.detok_graphs, chunks.len())).flatten() {
                self.fill_detok(&chunks, key_of(&self.detok_graphs, g), 0);
                if self.trace_batches {
                    tracing::info!(
                        batch = chunks.len(),
                        bucket = key_of(&self.detok_graphs, g),
                        first = chunks.iter().filter(|ch| self.reqs[ch.req].first_send.is_none()).count(),
                        "codec batch trace"
                    );
                }
                self.engine.detok_launch_async(g, self.f.audio, chunks.len() * self.detok_samples, chunks.len())?;
                self.inflight = Some(chunks.iter().map(|ch| (self.reqs[ch.req].key, ch.start)).collect());
                launched = true;
            }
        }

        let lm = self.select_lm();
        if let Some(b) = &lm {
            if self.trace_batches {
                tracing::info!(
                    batch = b.reqs.len(),
                    prefills = b.n_prefill,
                    bucket = if b.n_prefill > 0 {
                        key_of(&self.prefill_graphs, b.graph)
                    } else {
                        key_of(&self.decode_graphs, b.graph)
                    },
                    first = b.reqs.iter().filter(|&&i| self.reqs[i].first_send.is_none()).count(),
                    codec_inflight = self.inflight.as_ref().map_or(0, |v| v.len()),
                    "lm batch trace"
                );
            }
            self.fill_lm(b)?;
            self.engine.upload_inputs()?;
            self.engine.launch_n(b.graph, b.reqs.len())?;
            self.engine.download_field(self.f.out_tok, b.reqs.len() * self.c)?;
            self.engine.download_field(self.f.out_kind, b.reqs.len())?;
            self.engine.record_done()?;
            let t1 = Instant::now();
            self.stats.host_time += t1 - t0;
            self.engine.wait_done()?;
            self.stats.wait_time += t1.elapsed();
            self.stats.steps += 1;
        }
        let t2 = Instant::now();
        if let Some(b) = &lm {
            self.consume_lm(b);
        }
        self.finish(finished);
        if lm.is_some() {
            self.stats.host_time += t2.elapsed();
        } else if self.inflight.is_some() {
            // Only the codec has work: block on it instead of spinning.
            self.engine.detok_wait()?;
        } else if !launched {
            std::thread::sleep(Duration::from_micros(200));
        }
        Ok(())
    }

    // ---- asynchronous (v2) engine loop ---------------------------------------------------------

    /// One iteration of the asynchronous loop: collect whichever lane finished, launch the next LM
    /// step and the next detok batch if their lanes are idle, otherwise wait for a lane to finish.
    fn step_async(&mut self, t0: Instant) -> Result<()> {
        let mut finished = Vec::new();
        let mut progressed = false;
        self.engine.trace_poll()?;

        // Collect the detok lane.
        if self.inflight.is_some() && self.engine.detok_ready()? {
            let rows = self.inflight.take().unwrap();
            if let Some((t, bucket)) = self.dt_launched.take() {
                let secs = t.elapsed().as_secs_f64();
                match self.dt_cost.iter_mut().find(|(b, _)| *b == bucket) {
                    Some((_, c)) => *c = 0.8 * *c + 0.2 * secs,
                    None => self.dt_cost.push((bucket, secs)),
                }
                self.calib.detok(self.dt_mode, bucket, secs);
            }
            let chunks = self.chunks_of(&rows);
            self.emit_audio(&chunks, 0);
            self.push_finished(&chunks, &mut finished);
            self.dt_exclusive = false;
            progressed = true;
        }
        // Collect the urgent lane.
        if self.u_inflight.is_some() && self.engine.urgent_ready()? {
            let rows = self.u_inflight.take().unwrap();
            let chunks = self.chunks_of(&rows);
            self.emit_audio(&chunks, 1);
            self.push_finished(&chunks, &mut finished);
            progressed = true;
        }
        // Collect the LM lane.
        if self.lm_inflight.is_some() && self.engine.lm_ready()? {
            let f = self.lm_inflight.take().unwrap();
            if let Some((t, mode, bucket)) = self.lm_launched.take() {
                if f.n_prefill == 0 {
                    self.calib.lm_step(mode, bucket, t.elapsed().as_secs_f64());
                }
            }
            let reqs: Vec<usize> = f.keys.iter().filter_map(|&k| self.reqs.iter().position(|r| r.key == k)).collect();
            if reqs.len() == f.keys.len() {
                let b = LmBatch {
                    graph: self.decode_graphs[0].1,
                    segs: Vec::new(),
                    depth: None,
                    bucket: 0,
                    reqs,
                    n_prefill: f.n_prefill,
                };
                self.consume_lm(&b);
            } else {
                // A request was cancelled mid-step: consume the survivors one by one.
                for (e, &k) in f.keys.iter().enumerate() {
                    if let Some(i) = self.reqs.iter().position(|r| r.key == k) {
                        let b = LmBatch {
                            graph: self.decode_graphs[0].1,
                            segs: Vec::new(),
                            depth: None,
                            bucket: 0,
                            reqs: vec![i],
                            n_prefill: usize::from(e < f.n_prefill),
                        };
                        self.consume_lm_row(&b, e);
                    }
                }
            }
            progressed = true;
        }
        if !finished.is_empty() {
            self.finish(std::mem::take(&mut finished));
        }

        // Launch the next LM step.
        let mut launched = false;
        if self.lm_inflight.is_none() {
            if let Some(b) = self.select_lm() {
                self.launch_lm_async(b)?;
                launched = true;
            }
        }
        // Launch urgent chunks on the urgent lane, then the next batch on the detok lane.
        if self.uf.is_some() && self.u_inflight.is_none() {
            let chunks = self.select_urgent();
            if !chunks.is_empty() {
                self.launch_urgent(&chunks)?;
                launched = true;
            }
        }
        if self.inflight.is_none() {
            let (chunks, completions) = self.select_detok_v2();
            // Launch before retiring finished requests: `chunks` holds indices into `reqs`.
            if !chunks.is_empty() {
                self.launch_detok_async(&chunks)?;
                launched = true;
            }
            if !completions.is_empty() {
                self.finish(completions);
            }
        }
        let t1 = Instant::now();
        self.stats.host_time += t1 - t0;
        if !progressed && !launched {
            self.wait_any_lane()?;
            self.stats.wait_time += t1.elapsed();
        }
        Ok(())
    }

    /// Blocks until a lane finishes (or briefly sleeps when all are idle).
    fn wait_any_lane(&mut self) -> Result<()> {
        let (lm, dt, ut) = (self.lm_inflight.is_some(), self.inflight.is_some(), self.u_inflight.is_some());
        match (lm, dt, ut) {
            (false, false, false) => std::thread::sleep(Duration::from_micros(200)),
            (true, false, false) => self.engine.wait_done()?,
            (false, true, false) => self.engine.detok_wait()?,
            (false, false, true) => self.engine.urgent_wait()?,
            _ => loop {
                if (lm && self.engine.lm_ready()?)
                    || (dt && self.engine.detok_ready()?)
                    || (ut && self.engine.urgent_ready()?)
                {
                    break;
                }
                std::thread::yield_now();
            },
        }
        Ok(())
    }

    /// Rows of a collected detok batch, as chunks of the requests still alive.
    fn chunks_of(&self, rows: &[(u64, usize)]) -> Vec<DetokChunk> {
        rows.iter()
            .enumerate()
            .filter_map(|(row, &(key, start))| {
                self.reqs.iter().position(|r| r.key == key).map(|req| DetokChunk { req, start, row })
            })
            .collect()
    }

    fn push_finished(&self, chunks: &[DetokChunk], finished: &mut Vec<usize>) {
        for ch in chunks {
            let r = &self.reqs[ch.req];
            if r.done_lm && ch.start + self.detok_interval >= r.n_frames {
                finished.push(ch.req);
            }
        }
    }

    /// True while a chunk of request `key` is on a detok lane (chunks of a request decode in order).
    fn chunk_in_flight(&self, key: u64) -> bool {
        self.inflight.iter().chain(self.u_inflight.iter()).flatten().any(|&(k, _)| k == key)
    }

    /// Urgent-lane selection: the urgent ready chunks of requests with nothing in flight, earliest
    /// playback deadline first, up to the largest urgent graph.
    fn select_urgent(&mut self) -> Vec<DetokChunk> {
        let st = self.detok_step;
        let iv = self.detok_interval;
        let now = Instant::now();
        let mut cand: Vec<usize> = (0..self.reqs.len())
            .filter(|&i| {
                let r = &self.reqs[i];
                if (r.slot.is_none() && !r.done_lm) || self.chunk_in_flight(r.key) {
                    return false;
                }
                let next = r.next_detok(st);
                let ready = if r.done_lm { next < r.n_frames } else { next + iv <= r.n_frames };
                ready && self.is_urgent(i, now)
            })
            .collect();
        cand.sort_by(|&a, &b| {
            self.audio_deadline(a, now)
                .total_cmp(&self.audio_deadline(b, now))
                .then(self.reqs[a].key.cmp(&self.reqs[b].key))
        });
        cand.truncate(self.urgent_graphs.last().map_or(0, |(k, _)| *k));
        let mut chunks = Vec::with_capacity(cand.len());
        for (row, &i) in cand.iter().enumerate() {
            let start = self.reqs[i].next_detok(st);
            chunks.push(DetokChunk { req: i, start, row });
            let r = &mut self.reqs[i];
            r.last_detok = Some(r.last_detok.map_or(start, |x| x.max(start)));
        }
        chunks
    }

    fn launch_urgent(&mut self, chunks: &[DetokChunk]) -> Result<()> {
        let Some(g) = pick(&self.urgent_graphs, chunks.len()) else {
            bail!("no urgent detok graph for {} chunks", chunks.len())
        };
        let bucket = key_of(&self.urgent_graphs, g);
        self.fill_detok(chunks, bucket, 1);
        let audio = self.uf.as_ref().expect("urgent lane").audio;
        self.engine.urgent_launch(g, audio, chunks.len() * self.detok_samples, chunks.len())?;
        if self.engine.tracing() {
            self.engine.trace_note(serde_json::json!({"ev": "urgent", "n": chunks.len(), "bucket": bucket}));
        }
        self.u_inflight = Some(chunks.iter().map(|ch| (self.reqs[ch.req].key, ch.start)).collect());
        Ok(())
    }

    fn launch_lm_async(&mut self, mut b: LmBatch) -> Result<()> {
        let v2 = self.cfg.v2.clone();
        let decode_rows = b.reqs.len() - b.n_prefill;
        let n_modes = self.decode_by_mode.len();
        let mut mode = if b.n_prefill > 0 {
            0
        } else if v2.mode == v2::ModePolicy::Adaptive {
            let per_step = decode_rows as f64 / self.detok_step as f64;
            let dt_bucket =
                self.detok_buckets.iter().copied().find(|&x| x >= v2.dec_batch.max(1)).unwrap_or(self.max_detok_bs);
            self.calib.choose(n_modes, b.bucket, per_step, dt_bucket)
        } else {
            v2.mode_for(decode_rows, n_modes)
        };
        // Switch partitions only while the detok lane is idle (its batch keeps its mode).
        if b.n_prefill == 0 && self.inflight.is_some() && mode != self.dt_mode {
            mode = self.dt_mode;
        }
        if b.n_prefill == 0 {
            let table = &self.decode_by_mode[mode];
            b.graph = table.iter().find(|(k, _)| *k == b.bucket).map(|(_, g)| *g).unwrap_or(b.graph);
            b.depth = self.depth_by_mode[mode].iter().find(|(k, _)| *k == b.bucket).map(|(_, g)| *g);
            b.segs =
                self.segs_by_mode[mode].iter().find(|(k, _)| *k == b.bucket).map_or(Vec::new(), |(_, v)| v.clone());
        }
        let exclusive = v2.time_share || (b.n_prefill > 0 && v2.prefill_exclusive);
        self.fill_lm(&b)?;
        if (exclusive || self.dt_exclusive) && self.inflight.is_some() {
            self.engine.lm_wait_detok()?;
        }
        self.engine.upload_inputs()?;
        let n = b.reqs.len();
        // The gate event marks where non-urgent detok batches may start: before the depth graph,
        // or before backbone layer group i.
        let gate_at = match v2.gate {
            GateMode::Depth => Some(1 + b.segs.len()),
            GateMode::Layer(i) => Some(i.min(b.segs.len() + 1)),
            GateMode::None => None,
        };
        let mut graphs = vec![b.graph];
        graphs.extend(b.segs.iter().copied());
        graphs.extend(b.depth);
        for (k, &g) in graphs.iter().enumerate() {
            if gate_at == Some(k) && k > 0 {
                self.engine.record_depth_start()?;
            }
            self.engine.launch_n(g, n)?;
        }
        self.engine.download_field(self.f.out_tok, n * self.c)?;
        self.engine.download_field(self.f.out_kind, n)?;
        self.engine.record_done()?;
        if self.engine.tracing() {
            self.engine.trace_note(serde_json::json!({
                "ev": "lm", "rows": n, "prefill": b.n_prefill, "bucket": b.bucket, "mode": mode,
                "graphs": graphs.len(), "exclusive": exclusive, "active": self.reqs.len(),
            }));
        }
        if b.n_prefill == 0 {
            self.decode_mode = mode;
        }
        self.lm_launched = Some((Instant::now(), mode, b.bucket));
        self.stats.steps += 1;
        let keys = b.reqs.iter().map(|&i| self.reqs[i].key).collect();
        self.lm_inflight = Some(LmInflight {
            keys,
            n_prefill: b.n_prefill,
            exclusive,
            split: graphs.len() > 1 && gate_at.is_some_and(|k| k > 0 && k < graphs.len()),
        });
        Ok(())
    }

    fn launch_detok_async(&mut self, chunks: &[DetokChunk]) -> Result<()> {
        let v2 = self.cfg.v2.clone();
        let urgent = chunks.iter().filter(|ch| self.is_urgent(ch.req, Instant::now())).count();
        let exclusive = v2.time_share || (v2.urgent_exclusive && urgent > 0);
        let mode = if exclusive { 0 } else { self.background_mode() };
        let table = &self.detok_by_mode[mode.min(self.detok_by_mode.len() - 1)];
        let Some(g) = pick(table, chunks.len()) else { bail!("no detok graph for {} chunks", chunks.len()) };
        let bucket = key_of(table, g);
        self.fill_detok(chunks, bucket, 0);
        let inflight = self.lm_inflight.as_ref();
        let gate = if exclusive || inflight.is_some_and(|f| f.exclusive) {
            Gate::LmDone
        } else if v2.gate != GateMode::None && urgent == 0 && inflight.is_some_and(|f| f.split && f.n_prefill == 0) {
            Gate::DepthStart
        } else {
            Gate::None
        };
        self.engine.detok_launch_gated(g, self.f.audio, chunks.len() * self.detok_samples, gate, chunks.len())?;
        if self.engine.tracing() {
            self.engine.trace_note(serde_json::json!({
                "ev": "detok", "n": chunks.len(), "bucket": bucket, "urgent": urgent, "mode": mode,
                "gate": format!("{gate:?}"), "exclusive": exclusive,
            }));
        }
        self.dt_mode = mode;
        self.dt_exclusive = exclusive;
        self.dt_launched = Some((Instant::now(), bucket));
        self.inflight = Some(chunks.iter().map(|ch| (self.reqs[ch.req].key, ch.start)).collect());
        Ok(())
    }

    /// SM mode of a background detok batch: the split the mode policy gives decode steps, not the
    /// mode of the latest LM step. A batch launched right after a prefill step (mode 0) would
    /// otherwise run on all SMs and hold the following decode steps in mode 0 until it finishes,
    /// since partitions switch only while the detok lane is idle.
    fn background_mode(&self) -> usize {
        let decoding = self.reqs.iter().filter(|r| r.slot.is_some() && !r.done_lm).count();
        self.cfg.v2.background_mode(decoding.min(self.max_bs), self.decode_mode, self.decode_by_mode.len())
    }

    /// First chunks and chunks whose playback deadline is near. With `slack_urgency`, first chunks
    /// are urgent only near their startup deadline too, and the margin covers the largest batch.
    fn is_urgent(&self, i: usize, now: Instant) -> bool {
        let r = &self.reqs[i];
        if !self.cfg.v2.slack_urgency {
            return r.first_send.is_none() || self.audio_deadline(i, now) < self.cfg.v2.urgent;
        }
        let batch = self.dt_cost.iter().map(|(_, c)| *c).fold(0.0, f64::max);
        self.audio_deadline(i, now) < self.cfg.v2.urgent + batch
    }

    /// Detok selection for the asynchronous loop: urgent chunks go at once; other ready chunks wait
    /// until `dec_batch` of them are ready, a playback deadline comes due, or the LM has nothing
    /// left to produce. One chunk per request per batch (stateful codecs decode in order).
    fn select_detok_v2(&mut self) -> (Vec<DetokChunk>, Vec<usize>) {
        let st = self.detok_step;
        let iv = self.detok_interval;
        let now = Instant::now();
        let mut cand = Vec::new();
        let mut completions = Vec::new();
        for (i, r) in self.reqs.iter().enumerate() {
            if self.chunk_in_flight(r.key) {
                continue;
            }
            match detok_ready(r.slot.is_some(), r.done_lm, r.next_detok(st), r.n_frames, iv) {
                DetokReady::Chunk => cand.push(i),
                DetokReady::Complete => completions.push(i),
                DetokReady::Wait => {}
            }
        }
        cand.sort_by(|&a, &b| {
            let (ua, ub) = (self.is_urgent(a, now), self.is_urgent(b, now));
            ub.cmp(&ua)
                .then(self.audio_deadline(a, now).total_cmp(&self.audio_deadline(b, now)))
                .then(self.reqs[a].key.cmp(&self.reqs[b].key))
        });
        let n_urgent = cand.iter().filter(|&&i| self.is_urgent(i, now)).count();
        let lm_active = self.lm_inflight.is_some() || self.reqs.iter().any(|r| !r.done_lm);
        // Waiting for more chunks than the streams that can still produce one only adds delay: once
        // every such stream has a chunk ready, the batch goes.
        let streams = self
            .reqs
            .iter()
            .filter(|r| {
                (r.slot.is_some() || r.done_lm)
                    && !self.chunk_in_flight(r.key)
                    && (!r.done_lm || r.next_detok(st) < r.n_frames)
            })
            .count();
        let target = self.cfg.v2.dec_batch.min(streams.max(1));
        let due = !cand.is_empty() && {
            let bucket = self.detok_buckets.iter().copied().find(|&b| b >= target.max(1)).unwrap_or(self.max_detok_bs);
            let cost = self.dt_cost.iter().find(|(b, _)| *b == bucket).map_or(0.02, |(_, c)| *c);
            cand.iter().map(|&i| self.audio_deadline(i, now)).fold(f64::INFINITY, f64::min) - cost
                < self.cfg.v2.dec_flush
        };
        let cap = match self.cfg.v2.dec_max_batch {
            0 => self.max_detok_bs,
            m => m.min(self.max_detok_bs).max(n_urgent.min(self.max_detok_bs)),
        };
        let take = v2::detok_take(cand.len(), n_urgent, target, cap, lm_active, due, &self.detok_buckets);
        let mut chunks = Vec::with_capacity(take);
        for (row, &i) in cand.iter().take(take).enumerate() {
            let start = self.reqs[i].next_detok(st);
            chunks.push(DetokChunk { req: i, start, row });
            let r = &mut self.reqs[i];
            r.last_detok = Some(r.last_detok.map_or(start, |x| x.max(start)));
        }
        (chunks, completions)
    }

    fn drop_disconnected(&mut self) {
        let mut i = 0;
        while i < self.reqs.len() {
            if self.reqs[i].tx.is_closed() {
                let r = self.reqs.swap_remove(i);
                tracing::info!(request_id = %r.id, "request cancelled");
                self.release(r);
            } else {
                i += 1;
            }
        }
    }

    fn release(&mut self, mut r: Req) {
        if let Some(s) = r.slot.take() {
            self.free_slots.push(s);
        }
        if let (Some(s), Some(free)) = (r.dt_slot.take(), self.free_dt_slots.as_mut()) {
            free.push(s);
        }
        self.free_pages.append(&mut r.pages);
    }

    fn update_pressing(&mut self) {
        let now = Instant::now();
        for r in &mut self.reqs {
            r.pressing = is_pressing(r.streaming, r.done_lm, r.first_send, r.total_sent_secs, r.last_chunk_secs, now);
        }
    }

    // ---- selection --------------------------------------------------------------------------

    /// Returns detokenization chunks and the requests that only need a completion message.
    fn select_detok(&mut self) -> (Vec<DetokChunk>, Vec<usize>) {
        let iv = self.detok_interval;
        let st = self.detok_step;
        let cap = self.max_detok_bs;
        // Stateful codecs decode a request's chunks strictly in order, one per step.
        let per_req = if self.free_dt_slots.is_some() { 1 } else { usize::MAX };
        let mut chunks = Vec::new();
        let mut completions = Vec::new();

        let mut candidates = Vec::new();
        for (i, r) in self.reqs.iter().enumerate() {
            match detok_ready(r.slot.is_some(), r.done_lm, r.next_detok(st), r.n_frames, iv) {
                DetokReady::Chunk => candidates.push(i),
                DetokReady::Complete => completions.push(i),
                DetokReady::Wait => {}
            }
        }

        // Up to `budget` chunk start indices of request `r`.
        let assign =
            |r: &Req, budget: usize| chunk_starts(r.next_detok(st), r.n_frames, iv, st, r.done_lm, budget.min(per_req));

        match self.cfg.policy {
            Policy::Deadline => {
                let now = Instant::now();
                candidates.sort_by(|&a, &b| {
                    self.audio_deadline(a, now)
                        .total_cmp(&self.audio_deadline(b, now))
                        .then(self.reqs[a].key.cmp(&self.reqs[b].key))
                });
                // Always consume ready work, even when every stream has buffer. One
                // chunk per request preserves stateful ordering and batch fairness.
                for &i in candidates.iter().take(cap) {
                    chunks.push(DetokChunk { req: i, start: self.reqs[i].next_detok(st), row: 0 });
                }
            }
            Policy::Base => {
                for &i in candidates.iter().take(cap) {
                    let r = &self.reqs[i];
                    chunks.push(DetokChunk { req: i, start: r.next_detok(st), row: 0 });
                }
            }
            Policy::Offline => {
                if !offline_may_decode(self.reqs.iter().map(|r| (r.slot.is_some(), r.done_lm))) {
                    return (chunks, completions);
                }
                for &i in &candidates {
                    let budget = cap - chunks.len();
                    if budget == 0 {
                        break;
                    }
                    let starts = assign(&self.reqs[i], budget);
                    chunks.extend(starts.into_iter().map(|s| DetokChunk { req: i, start: s, row: 0 }));
                }
            }
            Policy::Online => {
                let (critical, relaxed): (Vec<usize>, Vec<usize>) =
                    candidates.iter().partition(|&&i| self.reqs[i].pressing);
                if critical.is_empty() {
                    return (chunks, completions);
                }
                let counts: Vec<usize> = critical
                    .iter()
                    .map(|&i| {
                        let r = &self.reqs[i];
                        let remaining = r.n_frames.saturating_sub(r.next_detok(st));
                        (remaining / st + usize::from(r.done_lm && remaining > 0)).min(per_req)
                    })
                    .collect();
                let assigned = online_budgets(counts, cap);
                for (k, &i) in critical.iter().enumerate() {
                    let budget = assigned[k].min(cap.saturating_sub(chunks.len()));
                    if budget == 0 {
                        continue;
                    }
                    let starts = assign(&self.reqs[i], budget);
                    chunks.extend(starts.into_iter().map(|s| DetokChunk { req: i, start: s, row: 0 }));
                }
                for &i in &relaxed {
                    let budget = cap.saturating_sub(chunks.len());
                    if budget == 0 {
                        break;
                    }
                    let starts = assign(&self.reqs[i], budget);
                    chunks.extend(starts.into_iter().map(|s| DetokChunk { req: i, start: s, row: 0 }));
                }
            }
        }
        for (k, ch) in chunks.iter_mut().enumerate() {
            ch.row = k;
            let r = &mut self.reqs[ch.req];
            r.last_detok = Some(r.last_detok.map_or(ch.start, |x| x.max(ch.start)));
        }
        (chunks, completions)
    }

    fn audio_deadline(&self, i: usize, now: Instant) -> f64 {
        let r = &self.reqs[i];
        let end = r.playback_end.unwrap_or(r.received + Duration::from_secs_f64(self.cfg.deadline.startup));
        if end >= now {
            (end - now).as_secs_f64()
        } else {
            -(now - end).as_secs_f64()
        }
    }

    fn pending_chunks(&self, i: usize) -> usize {
        let r = &self.reqs[i];
        deadline::ready_chunks(r.n_frames, r.emitted_next, self.detok_interval, self.detok_step)
    }

    fn lm_slack(&self, i: usize, now: Instant) -> f64 {
        let r = &self.reqs[i];
        let queued = self.pending_chunks(i);
        let target = r.emitted_next + queued * self.detok_step + self.detok_interval;
        let chunk_secs = self.chunk_samples as f64 / self.model.sample_rate() as f64;
        deadline::lm_slack(
            self.audio_deadline(i, now),
            queued,
            chunk_secs,
            target.saturating_sub(r.n_frames),
            &self.cfg.deadline,
        )
    }

    fn select_lm(&mut self) -> Option<LmBatch> {
        let max_rows = self.prefill_graphs.last().unwrap().0;
        let mut picked: Vec<usize> = Vec::new();
        let mut rows = 0usize;

        let now = Instant::now();
        let is_deadline = self.cfg.policy == Policy::Deadline;
        let decodable =
            |r: &Req| r.slot.is_some() && !r.done_lm && r.steps > 0 && r.text_in.as_ref().is_none_or(TextIn::ready);
        let mut decode_order = Vec::new();
        let mut min_slack = f64::INFINITY;
        if is_deadline {
            for i in 0..self.reqs.len() {
                // A prefill also delays collecting ready/in-flight audio. Protect
                // its deadline even when this request needs no more LM work.
                if self.pending_chunks(i) > 0 {
                    min_slack = min_slack.min(self.audio_deadline(i, now) - self.cfg.deadline.codec);
                }
                if decodable(&self.reqs[i]) {
                    if deadline::throttled(self.pending_chunks(i), &self.cfg.deadline) {
                        self.stats.throttled_rows += 1;
                    } else {
                        decode_order.push(i);
                    }
                }
            }
            decode_order.sort_by(|&a, &b| {
                self.lm_slack(a, now).total_cmp(&self.lm_slack(b, now)).then(self.reqs[a].key.cmp(&self.reqs[b].key))
            });
            min_slack = min_slack.min(decode_order.first().map_or(f64::INFINITY, |&i| self.lm_slack(i, now)));
        }
        let waiting = is_deadline && self.reqs.iter().any(|r| r.slot.is_none() && !r.done_lm);
        let deferred_for = self.prefill_deferred_since.map_or(0.0, |t| (now - t).as_secs_f64());
        let defer = is_deadline && waiting && deadline::defer_prefill(min_slack, deferred_for, &self.cfg.deadline);
        if defer {
            self.stats.prefill_deferrals += 1;
            self.prefill_deferred_since.get_or_insert(now);
        }
        if !waiting {
            self.prefill_deferred_since = None;
        }

        // Prefills (FIFO), subject to slot / KV-page availability.
        for i in 0..self.reqs.len() {
            if defer || picked.len() >= self.cfg.max_prefills_per_step.min(self.max_bs) {
                break;
            }
            let r = &self.reqs[i];
            if r.slot.is_some() || r.done_lm {
                continue;
            }
            if rows + r.prompt_len > max_rows {
                break;
            }
            let n_pages = kv_pages_needed(r.prompt_len, self.model.max_tokens(), self.max_context, self.page_size);
            let free_dt = self.free_dt_slots.as_ref().map(Vec::len);
            if !can_admit(self.free_slots.len(), free_dt, self.free_pages.len(), n_pages) {
                break;
            }
            let slot = self.free_slots.pop().unwrap();
            let dt_slot = self.free_dt_slots.as_mut().and_then(|f| f.pop());
            let at = self.free_pages.len() - n_pages;
            let pages: Vec<i32> = self.free_pages.drain(at..).collect();
            let r = &mut self.reqs[i];
            r.slot = Some(slot);
            r.dt_slot = dt_slot;
            r.pages = pages;
            rows += r.prompt_len;
            picked.push(i);
        }
        let n_prefill = picked.len();
        if n_prefill > 0 {
            self.prefill_deferred_since = None;
        }

        // Decodes: pressing first (online), then the rest.
        let room = |picked: &Vec<usize>, rows: usize| picked.len() < self.max_bs && (n_prefill == 0 || rows < max_rows);
        let decodable =
            |r: &Req| r.slot.is_some() && !r.done_lm && r.steps > 0 && r.text_in.as_ref().is_none_or(TextIn::ready);
        if is_deadline {
            for i in decode_order {
                if !room(&picked, rows) {
                    break;
                }
                picked.push(i);
                rows += 1;
            }
        } else if self.cfg.policy == Policy::Online {
            for pass in [true, false] {
                for i in 0..self.reqs.len() {
                    if !room(&picked, rows) {
                        break;
                    }
                    let r = &self.reqs[i];
                    if decodable(r) && r.pressing == pass && !picked[..n_prefill].contains(&i) {
                        picked.push(i);
                        rows += 1;
                    }
                }
            }
        } else {
            for i in 0..self.reqs.len() {
                if !room(&picked, rows) {
                    break;
                }
                if decodable(&self.reqs[i]) && !picked[..n_prefill].contains(&i) {
                    picked.push(i);
                    rows += 1;
                }
            }
        }
        if picked.is_empty() {
            return None;
        }
        let graph =
            if n_prefill > 0 { pick(&self.prefill_graphs, rows)? } else { pick(&self.decode_graphs, picked.len())? };
        let bucket =
            if n_prefill > 0 { key_of(&self.prefill_graphs, graph) } else { key_of(&self.decode_graphs, graph) };
        Some(LmBatch { graph, segs: Vec::new(), depth: None, bucket, reqs: picked, n_prefill })
    }

    // ---- input marshalling ------------------------------------------------------------------

    fn fill_lm(&mut self, b: &LmBatch) -> Result<()> {
        let c = self.c;
        let ps = self.page_size;
        let dummy = self.max_slots as i32;
        let is_prefill = b.n_prefill > 0;
        let (n_entries, n_rows) = if is_prefill {
            (self.engine.attention_batch_size(b.graph).unwrap(), b.bucket)
        } else {
            (b.bucket, b.bucket)
        };
        let f = &self.f;
        let inp = &mut self.engine.inputs;
        let text_spec = self.model.text_stream();

        self.qo_h.clear();
        self.kv_h.clear();
        self.qo_h.push(0);
        self.kv_h.push(0);
        let mut row = 0usize;
        let mut n_idx = 0usize;

        for (e, &ri) in b.reqs.iter().enumerate() {
            let prefill = e < b.n_prefill;
            let streamed = match (prefill, text_spec, self.reqs[ri].text_in.as_mut()) {
                (false, Some(spec), Some(t)) => Some((spec.column, t.next(spec))),
                _ => None,
            };
            let r = &self.reqs[ri];
            let (start, len) = if prefill { (0, r.prompt_len) } else { (r.next_pos, 1) };
            {
                let tok = inp.get_mut(f.tok);
                if prefill {
                    tok[row * c..(row + len) * c].copy_from_slice(&r.prompt[..len * c]);
                } else if let Some(d) = self.model.decode_row() {
                    tok[row * c..(row + 1) * c].copy_from_slice(d);
                } else {
                    tok[row * c..(row + 1) * c].fill(0);
                }
                if let Some((col, t)) = streamed {
                    tok[row * c + col] = t;
                }
            }
            if let Some(mf) = f.mask {
                let m = inp.get_mut(mf);
                if prefill {
                    let pm = r.prompt_mask.as_deref().expect("model with input masks must provide prompt masks");
                    m[row * c..(row + len) * c].copy_from_slice(&pm[..len * c]);
                } else {
                    m[row * c..(row + 1) * c].copy_from_slice(self.model.decode_mask().expect("decode mask"));
                }
            }
            inp.get_mut(f.fb)[row..row + len].fill(i32::from(!prefill));
            inp.get_mut(f.row_slot)[row..row + len].fill(r.slot.unwrap());
            for k in 0..len {
                let p = start + k;
                inp.get_mut(f.pos)[row + k] = p as i32;
                inp.get_mut(f.kv_page)[row + k] = r.pages[p / ps] as i64;
                inp.get_mut(f.kv_off)[row + k] = (p % ps) as i64;
            }
            let kv_len = start + len;
            let np = kv_len.div_ceil(ps);
            inp.get_mut(f.kv_indices)[n_idx..n_idx + np].copy_from_slice(&r.pages[..np]);
            n_idx += np;
            inp.get_mut(f.kv_last_page_len)[e] = (kv_len - (np - 1) * ps) as i32;
            self.qo_h.push((row + len) as i32);
            self.kv_h.push(n_idx as i32);

            inp.get_mut(f.last_row)[e] = (row + len - 1) as i32;
            inp.get_mut(f.slot)[e] = r.slot.unwrap();
            inp.get_mut(f.active)[e] = 1;
            inp.get_mut(f.first)[e] = i32::from(prefill);
            let s = &r.sampling;
            inp.get_mut(f.temp)[e] = s.temperature;
            inp.get_mut(f.top_k)[e] = s.top_k;
            inp.get_mut(f.top_p)[e] = s.top_p;
            inp.get_mut(f.min_p)[e] = s.min_p;
            inp.get_mut(f.rep_pen)[e] = s.repetition_penalty;
            inp.get_mut(f.seed)[e] = r.seed;
            inp.get_mut(f.step)[e] = r.steps;
            row += len;
        }

        // Padding entries: decode graphs give each one row on the scratch page; prefill graphs give
        // them zero rows.
        for e in b.reqs.len()..n_entries {
            if is_prefill {
                self.qo_h.push(row as i32);
                self.kv_h.push(n_idx as i32);
                inp.get_mut(f.kv_last_page_len)[e] = 1;
                inp.get_mut(f.last_row)[e] = 0;
            } else {
                inp.get_mut(f.tok)[row * c..(row + 1) * c].fill(0);
                if let Some(mf) = f.mask {
                    inp.get_mut(mf)[row * c..(row + 1) * c].fill(0);
                }
                inp.get_mut(f.fb)[row] = 0;
                inp.get_mut(f.row_slot)[row] = dummy;
                inp.get_mut(f.pos)[row] = 0;
                inp.get_mut(f.kv_page)[row] = 0;
                inp.get_mut(f.kv_off)[row] = 0;
                inp.get_mut(f.kv_indices)[n_idx] = 0;
                n_idx += 1;
                inp.get_mut(f.kv_last_page_len)[e] = 1;
                inp.get_mut(f.last_row)[e] = row as i32;
                row += 1;
                self.qo_h.push(row as i32);
                self.kv_h.push(n_idx as i32);
            }
            inp.get_mut(f.slot)[e] = dummy;
            inp.get_mut(f.active)[e] = 0;
            inp.get_mut(f.first)[e] = 0;
        }
        // Padding rows of prefill graphs: computed but never attended to; KV goes to scratch.
        if row < n_rows {
            inp.get_mut(f.tok)[row * c..n_rows * c].fill(0);
            if let Some(mf) = f.mask {
                inp.get_mut(mf)[row * c..n_rows * c].fill(0);
            }
            inp.get_mut(f.fb)[row..n_rows].fill(0);
            inp.get_mut(f.row_slot)[row..n_rows].fill(dummy);
            inp.get_mut(f.pos)[row..n_rows].fill(0);
            inp.get_mut(f.kv_page)[row..n_rows].fill(0);
            inp.get_mut(f.kv_off)[row..n_rows].fill(0);
        }
        inp.get_mut(f.qo_indptr)[..self.qo_h.len()].copy_from_slice(&self.qo_h);
        inp.get_mut(f.kv_indptr)[..self.kv_h.len()].copy_from_slice(&self.kv_h);

        if !is_prefill {
            // Decode graphs plan with one query row per entry.
            for (k, q) in self.qo_h.iter_mut().enumerate() {
                *q = k as i32;
            }
        }
        let (qo, kv) = (std::mem::take(&mut self.qo_h), std::mem::take(&mut self.kv_h));
        let res = self.engine.plan(b.graph, &qo, &kv);
        self.qo_h = qo;
        self.kv_h = kv;
        res
    }

    fn fill_detok(&mut self, chunks: &[DetokChunk], bd: usize, lane: usize) {
        let c = self.c;
        let iv = self.detok_interval;
        let (f_tok, f_slot, f_reset, f_len, f_last) = match (lane, &self.uf) {
            (1, Some(u)) => (u.tok, u.slot, u.reset, u.len, u.last),
            _ => (self.f.dt_tok, self.f.dt_slot, self.f.dt_reset, self.f.dt_len, self.f.dt_last),
        };
        if let (Some(fl), Some(fe)) = (f_len, f_last) {
            // Valid frames of the window, and whether no chunk follows it.
            let st = self.detok_step;
            let inp = self.engine.lane_inputs(lane);
            for k in 0..bd {
                let (len, last) = match chunks.get(k) {
                    Some(ch) => {
                        let r = &self.reqs[ch.req];
                        let len = (ch.start + iv).min(r.n_frames) - ch.start;
                        (len as i32, i32::from(r.done_lm && ch.start + st >= r.n_frames))
                    }
                    None => (0, 0),
                };
                inp.get_mut(fl)[k] = len;
                inp.get_mut(fe)[k] = last;
            }
        }
        if let (Some(fs), Some(fr)) = (f_slot, f_reset) {
            let dummy = self.detok_slots as i32;
            let inp = self.engine.lane_inputs(lane);
            for k in 0..bd {
                let (slot, reset) = match chunks.get(k) {
                    Some(ch) => {
                        let r = &self.reqs[ch.req];
                        (r.dt_slot.expect("detok slot"), i32::from(ch.start == 0))
                    }
                    None => (dummy, 1),
                };
                inp.get_mut(fs)[k] = slot;
                inp.get_mut(fr)[k] = reset;
            }
        }
        let dt = self.engine.lane_inputs(lane).get_mut(f_tok);
        for (k, ch) in chunks.iter().enumerate() {
            let r = &self.reqs[ch.req];
            let end = (ch.start + iv).min(r.n_frames);
            let dst = &mut dt[k * iv * c..(k + 1) * iv * c];
            let real = end - ch.start;
            dst[..real * c].copy_from_slice(&r.audio[ch.start * c..end * c]);
            let last = &r.audio[(end - 1) * c..end * c];
            for j in real..iv {
                dst[j * c..(j + 1) * c].copy_from_slice(last);
            }
        }
    }

    // ---- output handling --------------------------------------------------------------------

    fn emit_audio(&mut self, chunks: &[DetokChunk], lane: usize) {
        let n = self.detok_samples;
        let iv = self.detok_interval;
        let f_audio = match (lane, &self.uf) {
            (1, Some(u)) => u.audio,
            _ => self.f.audio,
        };
        let audio = self.engine.lane_outputs(lane).get(f_audio);
        let now = Instant::now();
        let sr = self.model.sample_rate() as f64;
        let preseed = self.model.audio_preseed_frames();
        for ch in chunks {
            let k = ch.row;
            let r = &mut self.reqs[ch.req];
            let real = (r.n_frames - ch.start).min(iv);
            let (len, skip) = if self.audio_len_in_row {
                ((audio[k * n + n - 1].max(0) as usize).min(n - 1), 0)
            } else {
                let mut len = n;
                if real < iv {
                    len = ((n as f64) * (real as f64 - 0.5) / iv as f64) as usize;
                }
                // Python trims the pre-seeded frames after the tail trim (an early stop may leave nothing).
                (len, if ch.start == 0 { (preseed * n / iv).min(len) } else { 0 })
            };
            let pcm = &audio[k * n + skip..k * n + len];
            let mut bytes = Vec::with_capacity(pcm.len() * 2);
            for s in pcm {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            if r.streaming {
                let secs = (len - skip) as f64 / sr;
                if secs > 0.0 {
                    let end = r.playback_end.unwrap_or(now).max(now);
                    r.playback_end = Some(end + Duration::from_secs_f64(secs));
                }
                r.first_send.get_or_insert(now);
                r.total_sent_secs += secs;
                r.last_chunk_secs = secs;
            }
            r.emitted_next = ch.start + self.detok_step;
            let _ = r.tx.send(Event::Audio(Bytes::from(bytes)));
        }
    }

    /// Consumes output row `e` of the last LM step for the single request in `b`.
    fn consume_lm_row(&mut self, b: &LmBatch, e: usize) {
        let c = self.c;
        let out = self.engine.outputs.get(self.f.out_tok);
        let tok: Vec<i32> = out[e * c..(e + 1) * c].to_vec();
        let kind = self.engine.outputs.get(self.f.out_kind)[e];
        let ri = b.reqs[0];
        let tmp = LmBatch {
            graph: b.graph,
            segs: Vec::new(),
            depth: None,
            bucket: 0,
            reqs: vec![ri],
            n_prefill: b.n_prefill,
        };
        // Reuse consume_lm on a one-row view of the output.
        let saved: Vec<i32> = self.engine.outputs.get(self.f.out_tok)[..c].to_vec();
        let saved_kind = self.engine.outputs.get(self.f.out_kind)[0];
        self.engine.outputs.get_mut(self.f.out_tok)[..c].copy_from_slice(&tok);
        self.engine.outputs.get_mut(self.f.out_kind)[0] = kind;
        self.consume_lm(&tmp);
        self.engine.outputs.get_mut(self.f.out_tok)[..c].copy_from_slice(&saved);
        self.engine.outputs.get_mut(self.f.out_kind)[0] = saved_kind;
    }

    fn consume_lm(&mut self, b: &LmBatch) {
        let c = self.c;
        let out = self.engine.outputs.get(self.f.out_tok);
        let kinds = self.engine.outputs.get(self.f.out_kind);
        let max_tokens = self.model.max_tokens();
        let preseed = self.model.audio_preseed_frames();
        let mut done = Vec::new();
        for (e, &ri) in b.reqs.iter().enumerate() {
            let tok = &out[e * c..(e + 1) * c];
            let r = &mut self.reqs[ri];
            r.next_pos = if e < b.n_prefill { r.prompt_len } else { r.next_pos + 1 };
            if e < b.n_prefill {
                r.features = None;
            }
            r.steps += 1;
            match TokenKind::from_code(kinds[e]) {
                TokenKind::Audio => {
                    if r.n_frames == 0 && preseed > 0 {
                        r.audio.resize(preseed * c, 0);
                        r.n_frames = preseed;
                    }
                    r.audio.extend_from_slice(tok);
                    r.n_frames += 1;
                }
                TokenKind::Stop => {
                    r.done_lm = true;
                    r.finish_reason = "stop_id_encountered";
                }
                TokenKind::Other => {}
            }
            if r.next_pos > max_tokens {
                r.done_lm = true;
                r.finish_reason = "max_tokens_reached";
            } else if !r.done_lm && r.next_pos >= self.max_context {
                r.done_lm = true;
                r.finish_reason = "max_context_reached";
            }
            if r.done_lm {
                done.push(ri);
            }
        }
        for ri in done {
            let r = &mut self.reqs[ri];
            if let Some(s) = r.slot.take() {
                self.free_slots.push(s);
            }
            self.free_pages.append(&mut r.pages);
        }
    }

    fn finish(&mut self, mut idx: Vec<usize>) {
        idx.sort_unstable();
        idx.dedup();
        for &i in idx.iter().rev() {
            let r = self.reqs.remove(i);
            tracing::info!(request_id = %r.id, finish_reason = r.finish_reason, steps = r.steps, audio_seconds = r.total_sent_secs, "request complete");
            let _ = r.tx.send(Event::Done(r.finish_reason.to_string()));
            self.release(r);
        }
    }
}

fn pick(graphs: &[(usize, GraphId)], n: usize) -> Option<GraphId> {
    graphs.iter().find(|(k, _)| *k >= n).map(|(_, g)| *g)
}

fn key_of(graphs: &[(usize, GraphId)], id: GraphId) -> usize {
    graphs.iter().find(|(_, g)| *g == id).map(|(k, _)| *k).unwrap()
}

// ---- policy decisions (pure, unit-tested below) ---------------------------------------------------

/// Online policy: a stream is pressing until its first chunk and from 1 s before its last received
/// chunk starts playing (`first_send` + the audio sent before that chunk). A non-streaming request is
/// pressing once its tokens are all generated: audio is only decoded together with pressing chunks, so
/// it would otherwise never be decoded while no stream is pressing.
fn is_pressing(
    streaming: bool,
    done_lm: bool,
    first_send: Option<Instant>,
    sent_secs: f64,
    last_chunk_secs: f64,
    now: Instant,
) -> bool {
    if !streaming {
        return done_lm;
    }
    match first_send {
        Some(first) => now + Duration::from_secs(1) >= first + Duration::from_secs_f64(sent_secs - last_chunk_secs),
        None => true,
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DetokReady {
    /// Not prefilled yet, or fewer than a chunk of new frames.
    Wait,
    /// A chunk can be decoded.
    Chunk,
    /// Everything is decoded: only the completion message is left.
    Complete,
}

/// Where a request stands for detokenization; `next` is the start frame of its next chunk.
fn detok_ready(admitted: bool, done_lm: bool, next: usize, n_frames: usize, iv: usize) -> DetokReady {
    if done_lm {
        if next < n_frames {
            DetokReady::Chunk
        } else {
            DetokReady::Complete
        }
    } else if admitted && next + iv <= n_frames {
        DetokReady::Chunk
    } else {
        DetokReady::Wait
    }
}

/// Start frames of up to `budget` chunks from `next`: full `iv`-frame windows `st` frames apart and,
/// once the tokens are all generated, the final partial window.
fn chunk_starts(
    mut next: usize,
    n_frames: usize,
    iv: usize,
    st: usize,
    done_lm: bool,
    mut budget: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    while budget > 0 && next + iv <= n_frames {
        out.push(next);
        next += st;
        budget -= 1;
    }
    if done_lm && budget > 0 && next < n_frames {
        out.push(next);
    }
    out
}

/// Online policy: chunks per pressing request (`counts` ready each) within a batch of `cap`: all of
/// them if they fit, else proportionally, at least one each while possible.
fn online_budgets(counts: Vec<usize>, cap: usize) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    if total <= cap {
        return counts;
    }
    let mut a: Vec<usize> = counts.iter().map(|&c| (c * cap / total).max(1)).collect();
    let mut sum: usize = a.iter().sum();
    while sum > cap {
        let mut changed = false;
        for x in a.iter_mut() {
            if *x > 1 {
                *x -= 1;
                sum -= 1;
                changed = true;
                if sum <= cap {
                    break;
                }
            }
        }
        // More pressing requests than slots: the caller stops at `cap` chunks.
        if !changed {
            break;
        }
    }
    a
}

/// Offline policy: audio is decoded only while no admitted request is generating, given `(admitted,
/// done_lm)` per request. Requests waiting for admission do not count: they may wait for the codec
/// state slot of a finished request, which is only freed once its audio is decoded.
fn offline_may_decode(mut reqs: impl Iterator<Item = (bool, bool)>) -> bool {
    !reqs.any(|(admitted, done_lm)| admitted && !done_lm)
}

/// KV pages a request reserves at admission: its prompt or `max_tokens + 1` positions, whichever is
/// longer, up to the maximum context (as `ProgramWorker` does).
fn kv_pages_needed(prompt_len: usize, max_tokens: usize, max_context: usize, page_size: usize) -> usize {
    prompt_len.max(max_tokens + 1).min(max_context).div_ceil(page_size)
}

/// Admission needs an LM slot, a codec state slot (stateful codecs: `free_dt_slots` is `Some`) and the
/// request's KV pages.
fn can_admit(free_slots: usize, free_dt_slots: Option<usize>, free_pages: usize, n_pages: usize) -> bool {
    free_slots > 0 && free_dt_slots != Some(0) && free_pages >= n_pages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressing() {
        let now = Instant::now();
        let ago = |s: f64| Some(now - Duration::from_secs_f64(s));
        // Non-streaming: pressing exactly once the tokens are all generated.
        assert!(!is_pressing(false, false, None, 0.0, 0.0, now));
        assert!(is_pressing(false, true, None, 0.0, 0.0, now));
        // A stream without audio yet.
        assert!(is_pressing(true, false, None, 0.0, 0.0, now));
        // Five 1 s chunks sent 0.5 s ago: the last one starts playing in 3.5 s.
        assert!(!is_pressing(true, false, ago(0.5), 5.0, 1.0, now));
        // Sent 3.5 s ago: the last chunk starts in 0.5 s.
        assert!(is_pressing(true, false, ago(3.5), 5.0, 1.0, now));
    }

    #[test]
    fn readiness() {
        use DetokReady::*;
        assert_eq!(detok_ready(false, false, 0, 10, 4), Wait, "not prefilled");
        assert_eq!(detok_ready(true, false, 4, 7, 4), Wait, "fewer than a chunk of new frames");
        assert_eq!(detok_ready(true, false, 4, 8, 4), Chunk);
        assert_eq!(detok_ready(true, true, 4, 5, 4), Chunk, "the final partial chunk");
        assert_eq!(detok_ready(true, true, 8, 8, 4), Complete);
        // A request that failed before admission (no audio) only needs its completion.
        assert_eq!(detok_ready(false, true, 0, 0, 4), Complete);
    }

    #[test]
    fn starts() {
        // While generating: full windows only.
        assert_eq!(chunk_starts(0, 10, 4, 4, false, 9), vec![0, 4]);
        // Once done: the partial window too.
        assert_eq!(chunk_starts(0, 10, 4, 4, true, 9), vec![0, 4, 8]);
        assert_eq!(chunk_starts(8, 10, 4, 4, true, 9), vec![8]);
        assert_eq!(chunk_starts(10, 10, 4, 4, true, 9), Vec::<usize>::new());
        // Overlapping windows (step < interval).
        assert_eq!(chunk_starts(0, 10, 4, 3, false, 9), vec![0, 3, 6]);
        // The budget (one chunk per step for stateful codecs).
        assert_eq!(chunk_starts(0, 10, 4, 4, true, 1), vec![0]);
        assert_eq!(chunk_starts(0, 10, 4, 4, true, 0), Vec::<usize>::new());
    }

    #[test]
    fn budgets() {
        assert_eq!(online_budgets(vec![2, 3], 8), vec![2, 3]);
        let b = online_budgets(vec![6, 2, 0], 4);
        assert!(b.iter().sum::<usize>() <= 4, "{b:?}");
        assert!(b[0] >= b[1], "{b:?}");
        // More pressing requests than slots: terminates; the caller stops at the batch size.
        assert_eq!(online_budgets(vec![3, 3, 3, 3], 2), vec![1, 1, 1, 1]);
    }

    #[test]
    fn offline_gate() {
        // (admitted, done_lm)
        assert!(offline_may_decode([(true, true), (false, false)].into_iter()), "waiting requests do not block");
        assert!(!offline_may_decode([(true, false), (true, true)].into_iter()), "a generating request blocks");
        assert!(offline_may_decode(std::iter::empty()));
    }

    #[test]
    fn admission() {
        assert_eq!(kv_pages_needed(10, 100, 2048, 64), 2); // 101 positions
        assert_eq!(kv_pages_needed(300, 100, 2048, 64), 5); // the prompt is longer
        assert_eq!(kv_pages_needed(10, 4096, 2048, 64), 32); // capped at the context
        assert!(can_admit(1, None, 2, 2));
        assert!(!can_admit(0, None, 2, 2), "no LM slot");
        assert!(!can_admit(1, Some(0), 2, 2), "no codec state slot");
        assert!(can_admit(1, Some(1), 2, 2));
        assert!(!can_admit(1, None, 1, 2), "not enough KV pages");
    }
}
