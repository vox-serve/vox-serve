//! Minimal CUDA runtime bindings: streams, pinned memory, async copies, graphs and events.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

use anyhow::{bail, Result};

pub type Stream = *mut c_void;
pub type Graph = *mut c_void;
pub type GraphExec = *mut c_void;
pub type Event = *mut c_void;

pub const MEMCPY_H2D: c_int = 1;
pub const MEMCPY_D2H: c_int = 2;
const STREAM_NON_BLOCKING: c_uint = 1;
const EVENT_DISABLE_TIMING: c_uint = 2;
const ERROR_NOT_READY: c_int = 600;

extern "C" {
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaGetErrorString(err: c_int) -> *const c_char;
    fn cudaStreamCreateWithFlags(stream: *mut Stream, flags: c_uint) -> c_int;
    fn cudaStreamSynchronize(stream: Stream) -> c_int;
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int, stream: Stream) -> c_int;
    fn cudaGraphInstantiate(exec: *mut GraphExec, graph: Graph, flags: u64) -> c_int;
    fn cudaGraphUpload(exec: GraphExec, stream: Stream) -> c_int;
    fn cudaGraphLaunch(exec: GraphExec, stream: Stream) -> c_int;
    fn cudaEventCreateWithFlags(event: *mut Event, flags: c_uint) -> c_int;
    fn cudaEventRecord(event: Event, stream: Stream) -> c_int;
    fn cudaEventSynchronize(event: Event) -> c_int;
    fn cudaEventQuery(event: Event) -> c_int;
    fn cudaEventElapsedTime(ms: *mut f32, start: Event, end: Event) -> c_int;
    fn cudaStreamWaitEvent(stream: Stream, event: Event, flags: c_uint) -> c_int;
    fn cudaStreamCreateWithPriority(stream: *mut Stream, flags: c_uint, priority: c_int) -> c_int;
    fn cudaDeviceGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> c_int;
}

pub fn check(code: c_int, what: &str) -> Result<()> {
    if code != 0 {
        let msg = unsafe { CStr::from_ptr(cudaGetErrorString(code)) }.to_string_lossy();
        bail!("{what} failed: {msg} ({code})");
    }
    Ok(())
}

pub fn set_device(dev: i32) -> Result<()> {
    check(unsafe { cudaSetDevice(dev) }, "cudaSetDevice")
}

pub fn create_stream() -> Result<Stream> {
    let mut s: Stream = std::ptr::null_mut();
    check(unsafe { cudaStreamCreateWithFlags(&mut s, STREAM_NON_BLOCKING) }, "cudaStreamCreate")?;
    Ok(s)
}

/// Non-blocking stream with the given priority (lower number = higher priority, clamped to the
/// device range); high-priority streams get their pending blocks dispatched first.
pub fn create_stream_with_priority(high: bool) -> Result<Stream> {
    let (mut least, mut greatest) = (0, 0);
    check(unsafe { cudaDeviceGetStreamPriorityRange(&mut least, &mut greatest) }, "cudaDeviceGetStreamPriorityRange")?;
    let mut s: Stream = std::ptr::null_mut();
    let prio = if high { greatest } else { least };
    check(unsafe { cudaStreamCreateWithPriority(&mut s, STREAM_NON_BLOCKING, prio) }, "cudaStreamCreateWithPriority")?;
    Ok(s)
}

/// Makes all later work on `stream` wait for the most recent record of `event` (GPU-side, no host wait).
pub fn stream_wait(stream: Stream, event: &CudaEvent) -> Result<()> {
    check(unsafe { cudaStreamWaitEvent(stream, event.0, 0) }, "cudaStreamWaitEvent")
}

pub fn sync_stream(s: Stream) -> Result<()> {
    check(unsafe { cudaStreamSynchronize(s) }, "cudaStreamSynchronize")
}

/// # Safety
/// `dst` must be a device pointer valid for `n` bytes and `src` a host buffer that stays alive
/// (and unmodified) until the copy has completed on `stream`.
pub unsafe fn memcpy_h2d(dst: u64, src: *const u8, n: usize, stream: Stream) -> Result<()> {
    check(cudaMemcpyAsync(dst as *mut c_void, src as *const c_void, n, MEMCPY_H2D, stream), "cudaMemcpyAsync(H2D)")
}

/// # Safety
/// See [`memcpy_h2d`].
pub unsafe fn memcpy_d2h(dst: *mut u8, src: u64, n: usize, stream: Stream) -> Result<()> {
    check(cudaMemcpyAsync(dst as *mut c_void, src as *const c_void, n, MEMCPY_D2H, stream), "cudaMemcpyAsync(D2H)")
}

/// Page-locked host buffer.
pub struct Pinned {
    ptr: *mut u8,
}

unsafe impl Send for Pinned {}

impl Pinned {
    pub fn new(len: usize) -> Result<Self> {
        let mut p: *mut c_void = std::ptr::null_mut();
        check(unsafe { cudaHostAlloc(&mut p, len.max(1), 0) }, "cudaHostAlloc")?;
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, len.max(1)) };
        Ok(Self { ptr: p as *mut u8 })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe { cudaFreeHost(self.ptr as *mut c_void) };
    }
}

pub struct Exec(GraphExec);

unsafe impl Send for Exec {}

impl Exec {
    /// Instantiates a graph captured by torch (`CUDAGraph(keep_graph=True).raw_cuda_graph()`).
    pub fn instantiate(graph: u64, stream: Stream) -> Result<Self> {
        let mut e: GraphExec = std::ptr::null_mut();
        check(unsafe { cudaGraphInstantiate(&mut e, graph as Graph, 0) }, "cudaGraphInstantiate")?;
        check(unsafe { cudaGraphUpload(e, stream) }, "cudaGraphUpload")?;
        Ok(Self(e))
    }

    pub fn launch(&self, stream: Stream) -> Result<()> {
        check(unsafe { cudaGraphLaunch(self.0, stream) }, "cudaGraphLaunch")
    }
}

pub struct CudaEvent(Event);

unsafe impl Send for CudaEvent {}

impl CudaEvent {
    pub fn new() -> Result<Self> {
        let mut e: Event = std::ptr::null_mut();
        check(unsafe { cudaEventCreateWithFlags(&mut e, EVENT_DISABLE_TIMING) }, "cudaEventCreate")?;
        Ok(Self(e))
    }

    /// Event that records a GPU timestamp (for `elapsed_ms`).
    pub fn timed() -> Result<Self> {
        let mut e: Event = std::ptr::null_mut();
        check(unsafe { cudaEventCreateWithFlags(&mut e, 0) }, "cudaEventCreate")?;
        Ok(Self(e))
    }

    /// Milliseconds between two completed timed events.
    pub fn elapsed_ms(start: &CudaEvent, end: &CudaEvent) -> Result<f32> {
        let mut ms = 0f32;
        check(unsafe { cudaEventElapsedTime(&mut ms, start.0, end.0) }, "cudaEventElapsedTime")?;
        Ok(ms)
    }

    pub fn record(&self, stream: Stream) -> Result<()> {
        check(unsafe { cudaEventRecord(self.0, stream) }, "cudaEventRecord")
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { cudaEventSynchronize(self.0) }, "cudaEventSynchronize")
    }

    pub fn is_done(&self) -> Result<bool> {
        let code = unsafe { cudaEventQuery(self.0) };
        if code == ERROR_NOT_READY {
            return Ok(false);
        }
        check(code, "cudaEventQuery")?;
        Ok(true)
    }
}
