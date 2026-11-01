//! GPU executor: pinned arena mirrors, instantiated CUDA graphs and attention planners.
//!
//! One step = (plan attention) + one H2D copy of the input arena + graph launches + one D2H copy
//! of the output arena, all on a single stream. With disaggregation the detok stage has its own
//! arenas, stream and device (`DetokSide`) and runs asynchronously to the LM.
//!
//! The two lanes can be ordered on the GPU without host waits: the LM lane records `depth_start`
//! between the backbone and depth graphs of a split step and `done` after each step, and a detok
//! launch may wait on either (`Gate`); the LM lane may wait on the detok lane's `done`. With
//! tracing on, every graph launch is bracketed by timing events and written as one JSON line.

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};

use crate::cuda::{self, CudaEvent, Exec, Pinned, Stream};
use crate::flashinfer::Planner;
use crate::spec::{ArenaSpec, EngineSpec};

/// Element types that may be viewed in an arena field.
pub trait Elem: Copy + 'static {
    const DTYPE: &'static str;
}

impl Elem for i32 {
    const DTYPE: &'static str = "i32";
}
impl Elem for i64 {
    const DTYPE: &'static str = "i64";
}
impl Elem for f32 {
    const DTYPE: &'static str = "f32";
}
impl Elem for i16 {
    const DTYPE: &'static str = "i16";
}

/// Resolved location of an arena field (cheap to copy; avoids name lookups on the hot path).
#[derive(Debug, Clone, Copy)]
pub struct Field<T: Elem> {
    offset: usize,
    len: usize,
    _t: std::marker::PhantomData<T>,
}

pub struct HostArena {
    dev_ptr: u64,
    buf: Pinned,
    spec: ArenaSpec,
}

impl HostArena {
    fn new(spec: ArenaSpec) -> Result<Self> {
        Ok(Self { dev_ptr: spec.ptr, buf: Pinned::new(spec.size)?, spec })
    }

    pub fn field<T: Elem>(&self, name: &str) -> Result<Field<T>> {
        let f = self.spec.fields.get(name).ok_or_else(|| anyhow!("arena has no field '{name}'"))?;
        if f.dtype != T::DTYPE {
            bail!("arena field '{name}' has dtype {}, requested {}", f.dtype, T::DTYPE);
        }
        Ok(Field { offset: f.offset, len: f.shape.iter().product(), _t: std::marker::PhantomData })
    }

    pub fn has(&self, name: &str) -> bool {
        self.spec.fields.contains_key(name)
    }

    #[inline]
    pub fn get<T: Elem>(&self, f: Field<T>) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr().add(f.offset) as *const T, f.len) }
    }

    #[inline]
    pub fn get_mut<T: Elem>(&mut self, f: Field<T>) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr().add(f.offset) as *mut T, f.len) }
    }
}

struct GraphEntry {
    stage: String,
    key: Vec<usize>,
    exec: Exec,
    planner: Option<Planner>,
}

/// GPU-side precondition of a detok launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    None,
    /// Start once the LM's latest split step has entered its depth graph.
    DepthStart,
    /// Start once the LM's latest step has finished.
    LmDone,
}

#[derive(Debug, Clone, Default)]
pub struct EngineOpts {
    /// LM stream at high priority, detok stream at low priority.
    pub priorities: bool,
    /// Directory for the per-launch timing trace (`trace_<device>.jsonl`).
    pub trace_dir: Option<PathBuf>,
}

struct Pending {
    lane: &'static str,
    graph: usize,
    n: usize,
    host_ms: f64,
    start: CudaEvent,
    end: CudaEvent,
}

/// Per-launch GPU timing: start/end events relative to an origin event, written as JSON lines.
struct Tracer {
    origin: CudaEvent,
    host0: Instant,
    pending: VecDeque<Pending>,
    pool: Vec<CudaEvent>,
    out: std::io::BufWriter<std::fs::File>,
}

impl Tracer {
    fn event(&mut self) -> Result<CudaEvent> {
        self.pool.pop().map_or_else(CudaEvent::timed, Ok)
    }
}

/// Index of a graph inside the engine (resolved once at start-up).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphId(usize);

/// The detok stage on a separate device.
pub struct DetokSide {
    pub inputs: HostArena,
    pub outputs: HostArena,
    stream: Stream,
    done: CudaEvent,
    device: i32,
    upload_bytes: usize,
}

pub struct Engine {
    pub meta: serde_json::Value,
    pub inputs: HostArena,
    pub outputs: HostArena,
    pub detok: Option<DetokSide>,
    /// Urgent detok lane (same device as the detok lane): its own arenas and a high-priority stream,
    /// so urgent chunks need not wait for a background detok batch.
    pub urgent: Option<DetokSide>,
    device: i32,
    stream: Stream,
    graphs: Vec<GraphEntry>,
    done: CudaEvent,
    depth_start: CudaEvent,
    /// Bytes of the input arena actually uploaded each step (prefix covering every field).
    upload_bytes: usize,
    tracer: Option<Tracer>,
}

unsafe impl Send for Engine {}

impl Engine {
    /// Must be called on the thread that will drive the engine (sets its current device).
    pub fn new(spec: EngineSpec, device: i32, opts: &EngineOpts) -> Result<Self> {
        let new_stream =
            |high: bool| if opts.priorities { cuda::create_stream_with_priority(high) } else { cuda::create_stream() };
        let detok = match (spec.detok_inputs, spec.detok_outputs, spec.detok_device) {
            (Some(i), Some(o), Some(d)) => {
                cuda::set_device(d)?;
                Some(DetokSide {
                    upload_bytes: upload_size(&i),
                    inputs: HostArena::new(i)?,
                    outputs: HostArena::new(o)?,
                    stream: new_stream(false)?,
                    done: CudaEvent::new()?,
                    device: d,
                })
            }
            (None, None, None) => None,
            _ => bail!("incomplete detok arena spec"),
        };
        let urgent = match (spec.urgent_inputs, spec.urgent_outputs, &detok) {
            (Some(i), Some(o), Some(d)) => {
                cuda::set_device(d.device)?;
                Some(DetokSide {
                    upload_bytes: upload_size(&i),
                    inputs: HostArena::new(i)?,
                    outputs: HostArena::new(o)?,
                    stream: cuda::create_stream_with_priority(true)?,
                    done: CudaEvent::new()?,
                    device: d.device,
                })
            }
            (None, None, _) => None,
            _ => bail!("incomplete urgent detok arena spec"),
        };
        cuda::set_device(device)?;
        let stream = new_stream(true)?;
        let mut graphs = Vec::with_capacity(spec.graphs.len());
        for g in spec.graphs {
            let side = match g.stage.as_str() {
                "detok" => detok.as_ref(),
                "detok_urgent" => urgent.as_ref(),
                _ => None,
            };
            let on_detok = side.is_some();
            let s = match side {
                Some(d) => {
                    cuda::set_device(d.device)?;
                    d.stream
                }
                None => stream,
            };
            let exec = Exec::instantiate(g.graph, s)
                .with_context(|| format!("instantiating graph {} {:?}", g.stage, g.key))?;
            if on_detok {
                cuda::set_device(device)?;
            }
            let planner = g.attention.map(Planner::new).transpose()?;
            graphs.push(GraphEntry { stage: g.stage, key: g.key, exec, planner });
        }
        cuda::sync_stream(stream)?;
        for d in detok.iter().chain(urgent.iter()) {
            cuda::sync_stream(d.stream)?;
        }
        let upload_bytes = upload_size(&spec.inputs);
        let remote_detok = detok.as_ref().is_some_and(|d| d.device != device);
        if remote_detok && opts.trace_dir.is_some() {
            // Timing events are per device and the trace has one origin; not supported across GPUs.
            tracing::warn!("--trace-dir ignored: the detok lane is on another GPU");
        }
        let tracer = match &opts.trace_dir {
            Some(_) if remote_detok => None,
            Some(dir) => {
                std::fs::create_dir_all(dir)?;
                let origin = CudaEvent::timed()?;
                origin.record(stream)?;
                let f = std::fs::File::create(dir.join(format!("trace_{device}.jsonl")))?;
                Some(Tracer {
                    origin,
                    host0: Instant::now(),
                    pending: VecDeque::new(),
                    pool: Vec::new(),
                    out: std::io::BufWriter::new(f),
                })
            }
            None => None,
        };
        Ok(Self {
            meta: spec.meta,
            inputs: HostArena::new(spec.inputs)?,
            outputs: HostArena::new(spec.outputs)?,
            detok,
            urgent,
            device,
            stream,
            graphs,
            done: CudaEvent::new()?,
            depth_start: CudaEvent::new()?,
            upload_bytes,
            tracer,
        })
    }

    /// Launches on the LM lane, bracketed by timing events when tracing; `n` = real rows (trace only).
    pub fn launch_n(&mut self, id: GraphId, n: usize) -> Result<()> {
        let stream = self.stream;
        launch_traced(&mut self.tracer, &self.graphs, "lm", id, n, stream)
    }

    /// Marks the boundary between the backbone and depth graphs of a split LM step.
    pub fn record_depth_start(&self) -> Result<()> {
        self.depth_start.record(self.stream)
    }

    /// Makes the LM lane wait (on the GPU) for the detok lane's latest batch.
    pub fn lm_wait_detok(&self) -> Result<()> {
        match &self.detok {
            Some(d) => cuda::stream_wait(self.stream, &d.done),
            None => Ok(()),
        }
    }

    pub fn lm_ready(&self) -> Result<bool> {
        self.done.is_done()
    }

    /// Writes a host-side record (scheduling decision) into the trace.
    pub fn trace_note(&mut self, mut v: serde_json::Value) {
        if let Some(t) = self.tracer.as_mut() {
            v["host_ms"] = serde_json::json!(t.host0.elapsed().as_secs_f64() * 1e3);
            let _ = writeln!(t.out, "{v}");
        }
    }

    pub fn tracing(&self) -> bool {
        self.tracer.is_some()
    }

    /// Writes timing records of launches that have completed (in launch order per lane).
    pub fn trace_poll(&mut self) -> Result<()> {
        let Some(t) = self.tracer.as_mut() else { return Ok(()) };
        let mut keep = VecDeque::new();
        while let Some(p) = t.pending.pop_front() {
            if !p.end.is_done()? {
                keep.push_back(p);
                continue;
            }
            let t0 = CudaEvent::elapsed_ms(&t.origin, &p.start)?;
            let t1 = CudaEvent::elapsed_ms(&t.origin, &p.end)?;
            let g = &self.graphs[p.graph];
            let rec = serde_json::json!({"lane": p.lane, "stage": g.stage, "key": g.key, "n": p.n, "t0": t0, "t1": t1, "enq_ms": p.host_ms});
            writeln!(t.out, "{rec}")?;
            t.pool.push(p.start);
            t.pool.push(p.end);
        }
        t.pending = keep;
        Ok(())
    }

    pub fn trace_flush(&mut self) -> Result<()> {
        if let Some(t) = self.tracer.as_mut() {
            t.out.flush()?;
        }
        Ok(())
    }

    /// All captured keys of a stage, sorted ascending by the first key element.
    pub fn graph_keys(&self, stage: &str) -> Vec<(Vec<usize>, GraphId)> {
        let mut v: Vec<_> = self
            .graphs
            .iter()
            .enumerate()
            .filter(|(_, g)| g.stage == stage)
            .map(|(i, g)| (g.key.clone(), GraphId(i)))
            .collect();
        v.sort();
        v
    }

    pub fn attention_batch_size(&self, id: GraphId) -> Option<usize> {
        self.graphs[id.0].planner.as_ref().map(|p| p.batch_size())
    }

    pub fn plan(&mut self, id: GraphId, qo_indptr: &[i32], kv_indptr: &[i32]) -> Result<()> {
        let stream = self.stream;
        let g = &mut self.graphs[id.0];
        let p = g.planner.as_mut().ok_or_else(|| anyhow!("graph {} has no attention", g.stage))?;
        p.plan(qo_indptr, kv_indptr, stream)
    }

    pub fn upload_inputs(&mut self) -> Result<()> {
        unsafe { cuda::memcpy_h2d(self.inputs.dev_ptr, self.inputs.buf.as_ptr(), self.upload_bytes, self.stream) }
    }

    /// Downloads only `[offset, offset + len)` elements of an output field.
    pub fn download_field<T: Elem>(&mut self, f: Field<T>, len: usize) -> Result<()> {
        download(&mut self.outputs, f, len, self.stream)
    }

    /// Input arena holding the detok fields (the detok side's, if disaggregated).
    pub fn dt_inputs(&mut self) -> &mut HostArena {
        match &mut self.detok {
            Some(d) => &mut d.inputs,
            None => &mut self.inputs,
        }
    }

    /// Output arena holding the detok audio.
    pub fn dt_outputs(&self) -> &HostArena {
        match &self.detok {
            Some(d) => &d.outputs,
            None => &self.outputs,
        }
    }

    /// Disaggregated detok: upload its inputs, launch `id` and download `len` audio elements on the
    /// detok device without waiting; completion is reported by `detok_ready` / `detok_wait`.
    pub fn detok_launch_async(&mut self, id: GraphId, audio: Field<i16>, len: usize, n: usize) -> Result<()> {
        self.detok_launch_gated(id, audio, len, Gate::None, n)
    }

    /// Like `detok_launch_async`, but the detok lane first waits (on the GPU) for `gate`.
    pub fn detok_launch_gated(
        &mut self,
        id: GraphId,
        audio: Field<i16>,
        len: usize,
        gate: Gate,
        n: usize,
    ) -> Result<()> {
        let d = self.detok.as_mut().ok_or_else(|| anyhow!("engine is not disaggregated"))?;
        cuda::set_device(d.device)?;
        let r = (|| {
            match gate {
                Gate::None => {}
                Gate::DepthStart => cuda::stream_wait(d.stream, &self.depth_start)?,
                Gate::LmDone => cuda::stream_wait(d.stream, &self.done)?,
            }
            unsafe { cuda::memcpy_h2d(d.inputs.dev_ptr, d.inputs.buf.as_ptr(), d.upload_bytes, d.stream)? };
            launch_traced(&mut self.tracer, &self.graphs, "detok", id, n, d.stream)?;
            download(&mut d.outputs, audio, len, d.stream)?;
            d.done.record(d.stream)
        })();
        cuda::set_device(self.device)?;
        r
    }

    /// Inputs / outputs of detok lane `lane` (0 = the detok lane, 1 = the urgent lane).
    pub fn lane_inputs(&mut self, lane: usize) -> &mut HostArena {
        if lane == 1 && self.urgent.is_some() {
            return &mut self.urgent.as_mut().unwrap().inputs;
        }
        self.dt_inputs()
    }

    pub fn lane_outputs(&self, lane: usize) -> &HostArena {
        match (lane, &self.urgent) {
            (1, Some(u)) => &u.outputs,
            _ => self.dt_outputs(),
        }
    }

    /// Urgent lane: upload its inputs, launch `id`, download `len` audio elements, without waiting.
    pub fn urgent_launch(&mut self, id: GraphId, audio: Field<i16>, len: usize, n: usize) -> Result<()> {
        let d = self.urgent.as_mut().ok_or_else(|| anyhow!("no urgent detok lane"))?;
        cuda::set_device(d.device)?;
        let r = (|| {
            unsafe { cuda::memcpy_h2d(d.inputs.dev_ptr, d.inputs.buf.as_ptr(), d.upload_bytes, d.stream)? };
            launch_traced(&mut self.tracer, &self.graphs, "urgent", id, n, d.stream)?;
            download(&mut d.outputs, audio, len, d.stream)?;
            d.done.record(d.stream)
        })();
        cuda::set_device(self.device)?;
        r
    }

    pub fn urgent_ready(&self) -> Result<bool> {
        self.urgent.as_ref().map_or(Ok(true), |d| d.done.is_done())
    }

    pub fn urgent_wait(&self) -> Result<()> {
        self.urgent.as_ref().map_or(Ok(()), |d| d.done.synchronize())
    }

    pub fn detok_ready(&self) -> Result<bool> {
        self.detok.as_ref().map_or(Ok(true), |d| d.done.is_done())
    }

    pub fn detok_wait(&self) -> Result<()> {
        self.detok.as_ref().map_or(Ok(()), |d| d.done.synchronize())
    }

    pub fn record_done(&self) -> Result<()> {
        self.done.record(self.stream)
    }

    pub fn wait_done(&self) -> Result<()> {
        self.done.synchronize()
    }

    pub fn describe(&self) -> HashMap<String, Vec<Vec<usize>>> {
        let mut m: HashMap<String, Vec<Vec<usize>>> = HashMap::new();
        for g in &self.graphs {
            m.entry(g.stage.clone()).or_default().push(g.key.clone());
        }
        m
    }
}

fn launch_traced(
    tracer: &mut Option<Tracer>,
    graphs: &[GraphEntry],
    lane: &'static str,
    id: GraphId,
    n: usize,
    stream: Stream,
) -> Result<()> {
    let Some(t) = tracer.as_mut() else { return graphs[id.0].exec.launch(stream) };
    let (start, end) = (t.event()?, t.event()?);
    start.record(stream)?;
    graphs[id.0].exec.launch(stream)?;
    end.record(stream)?;
    let host_ms = t.host0.elapsed().as_secs_f64() * 1e3;
    t.pending.push_back(Pending { lane, graph: id.0, n, host_ms, start, end });
    Ok(())
}

fn download<T: Elem>(arena: &mut HostArena, f: Field<T>, len: usize, stream: Stream) -> Result<()> {
    let bytes = len.min(f.len) * std::mem::size_of::<T>();
    if bytes == 0 {
        return Ok(());
    }
    unsafe { cuda::memcpy_d2h(arena.buf.as_mut_ptr().add(f.offset), arena.dev_ptr + f.offset as u64, bytes, stream) }
}

/// Bytes of an input arena uploaded each step (prefix covering every field).
fn upload_size(spec: &ArenaSpec) -> usize {
    spec.fields.values().map(|f| f.offset + f.shape.iter().product::<usize>() * dtype_size(&f.dtype)).max().unwrap_or(0)
}

fn dtype_size(d: &str) -> usize {
    match d {
        "i64" => 8,
        "i32" | "f32" => 4,
        "bf16" | "f16" | "i16" => 2,
        _ => 1,
    }
}
