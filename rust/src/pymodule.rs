//! The `vox_serve._rt` Python module: the runtime's command line (`serve`) and the launchers of the
//! sampling kernels (csrc/sampling.cu) that the Python programs record into their CUDA graphs
//! (vox_serve/rt/kernels.py passes tensor pointers, strides, device and stream).

use std::ffi::c_void;

use pyo3::exceptions::{PyRuntimeError, PySystemExit, PyValueError};
use pyo3::prelude::*;

use crate::cuda::{self, Stream};
use crate::{cli, models};

extern "C" {
    fn vox_sample(
        device: i32,
        stream: Stream,
        dtype: i32,
        logits: *const c_void,
        row_stride: i64,
        vocab: i32,
        n_rows: i64,
        rows_per_req: i64,
        active: *const c_void,
        temperature: *const c_void,
        top_k: *const c_void,
        top_p: *const c_void,
        min_p: *const c_void,
        seed: *const c_void,
        step: *const c_void,
        rng_salt: i64,
        out: *mut c_void,
        out_stride: i64,
    ) -> i32;
    fn vox_rep_penalty_apply(
        device: i32,
        stream: Stream,
        dtype: i32,
        logits: *mut c_void,
        row_stride: i64,
        vocab: i32,
        n_rows: i64,
        rows_per_req: i64,
        bits: *const c_void,
        words: i32,
        slot: *const c_void,
        active: *const c_void,
        first: *const c_void,
        penalty: *const c_void,
    ) -> i32;
    fn vox_rep_penalty_reset(
        device: i32,
        stream: Stream,
        bits: *mut c_void,
        words: i32,
        rows_per_req: i64,
        n_req: i32,
        slot: *const c_void,
        active: *const c_void,
        first: *const c_void,
    ) -> i32;
    fn vox_rep_penalty_update(
        device: i32,
        stream: Stream,
        bits: *mut c_void,
        words: i32,
        rows_per_req: i64,
        vocab: i64,
        n_req: i32,
        slot: *const c_void,
        active: *const c_void,
        tokens: *const c_void,
        tok_stride: i64,
    ) -> i32;
    fn vox_randn(
        device: i32,
        stream: Stream,
        out: *mut c_void,
        n: i32,
        k: i32,
        seed: *const c_void,
        step: *const c_void,
        rng_salt: i64,
    ) -> i32;
}

/// Runs the Rust runtime's command line (`argv[0]` is the program name) until the server stops.
#[pyfunction]
fn serve(py: Python<'_>, argv: Vec<String>) -> PyResult<()> {
    let args = match <cli::Args as clap::Parser>::try_parse_from(argv) {
        Ok(args) => args,
        Err(e) => {
            let _ = e.print();
            return Err(PySystemExit::new_err(e.exit_code()));
        }
    };
    // Engine threads take the GIL for Python hooks while this thread waits for the server.
    py.allow_threads(|| cli::run(args)).map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
}

/// Parses `argv` as `serve` would, without running anything: ValueError with the command-line error if
/// the Rust runtime does not take it (`vox-serve` then falls back to the Python runtime).
#[pyfunction]
fn parse_args(argv: Vec<String>) -> PyResult<()> {
    <cli::Args as clap::Parser>::try_parse_from(argv)
        .map(|_| ())
        .map_err(|e| PyValueError::new_err(e.render().to_string()))
}

/// The prompt rows the Rust prompt builder of an engine's `meta` (JSON) builds for a request, with
/// `first` an input-streaming request's first text token: `(tokens, mask)`. For checking a Rust
/// builder against the program's `preprocess`.
#[pyfunction]
#[pyo3(signature = (meta, text, audio_path=None, model_kwargs=None, first=None))]
fn build_prompt(
    meta: &str,
    text: String,
    audio_path: Option<String>,
    model_kwargs: Option<&str>,
    first: Option<i32>,
) -> PyResult<(Vec<i32>, Option<Vec<i32>>)> {
    let bad = |e: &dyn std::fmt::Display| PyValueError::new_err(e.to_string());
    let meta: serde_json::Value = serde_json::from_str(meta).map_err(|e| bad(&e))?;
    let model = models::load(&meta, None).map_err(|e| bad(&format!("{e:#}")))?;
    let model_kwargs = match model_kwargs {
        Some(k) => serde_json::from_str(k).map_err(|e| bad(&e))?,
        None => Default::default(),
    };
    let input = models::GenerateInput { text, audio_path, model_kwargs };
    let prompt = match first {
        Some(t) => model.preprocess_streaming(&input, t),
        None => model.preprocess(&input),
    }
    .map_err(|e| bad(&format!("{e:#}")))?;
    Ok((prompt.tokens, prompt.mask))
}

fn launched(code: i32, kernel: &str) -> PyResult<()> {
    cuda::check(code, kernel).map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
}

fn ptr(p: usize) -> *mut c_void {
    p as *mut c_void
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample(
    device: i32,
    stream: usize,
    dtype: i32,
    logits: usize,
    row_stride: i64,
    vocab: i32,
    n_rows: i64,
    rows_per_req: i64,
    active: usize,
    temperature: usize,
    top_k: usize,
    top_p: usize,
    min_p: usize,
    seed: usize,
    step: usize,
    rng_salt: i64,
    out: usize,
    out_stride: i64,
) -> PyResult<()> {
    let code = unsafe {
        vox_sample(
            device,
            ptr(stream),
            dtype,
            ptr(logits),
            row_stride,
            vocab,
            n_rows,
            rows_per_req,
            ptr(active),
            ptr(temperature),
            ptr(top_k),
            ptr(top_p),
            ptr(min_p),
            ptr(seed),
            ptr(step),
            rng_salt,
            ptr(out),
            out_stride,
        )
    };
    launched(code, "sample")
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rep_penalty_apply(
    device: i32,
    stream: usize,
    dtype: i32,
    logits: usize,
    row_stride: i64,
    vocab: i32,
    n_rows: i64,
    rows_per_req: i64,
    bits: usize,
    words: i32,
    slot: usize,
    active: usize,
    first: usize,
    penalty: usize,
) -> PyResult<()> {
    let code = unsafe {
        vox_rep_penalty_apply(
            device,
            ptr(stream),
            dtype,
            ptr(logits),
            row_stride,
            vocab,
            n_rows,
            rows_per_req,
            ptr(bits),
            words,
            ptr(slot),
            ptr(active),
            ptr(first),
            ptr(penalty),
        )
    };
    launched(code, "rep_penalty_apply")
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rep_penalty_reset(
    device: i32,
    stream: usize,
    bits: usize,
    words: i32,
    rows_per_req: i64,
    n_req: i32,
    slot: usize,
    active: usize,
    first: usize,
) -> PyResult<()> {
    let code = unsafe {
        vox_rep_penalty_reset(
            device,
            ptr(stream),
            ptr(bits),
            words,
            rows_per_req,
            n_req,
            ptr(slot),
            ptr(active),
            ptr(first),
        )
    };
    launched(code, "rep_penalty_reset")
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rep_penalty_update(
    device: i32,
    stream: usize,
    bits: usize,
    words: i32,
    rows_per_req: i64,
    vocab: i64,
    n_req: i32,
    slot: usize,
    active: usize,
    tokens: usize,
    tok_stride: i64,
) -> PyResult<()> {
    let code = unsafe {
        vox_rep_penalty_update(
            device,
            ptr(stream),
            ptr(bits),
            words,
            rows_per_req,
            vocab,
            n_req,
            ptr(slot),
            ptr(active),
            ptr(tokens),
            tok_stride,
        )
    };
    launched(code, "rep_penalty_update")
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn randn(
    device: i32,
    stream: usize,
    out: usize,
    n: i32,
    k: i32,
    seed: usize,
    step: usize,
    rng_salt: i64,
) -> PyResult<()> {
    let code = unsafe { vox_randn(device, ptr(stream), ptr(out), n, k, ptr(seed), ptr(step), rng_salt) };
    launched(code, "randn")
}

#[pymodule]
pub fn _rt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // FlashInfer version of the attention planner (third_party/flashinfer); must match the installed one.
    m.add("FLASHINFER_VERSION", env!("VOX_FLASHINFER_VERSION"))?;
    m.add_function(wrap_pyfunction!(serve, m)?)?;
    m.add_function(wrap_pyfunction!(build_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(parse_args, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(rep_penalty_apply, m)?)?;
    m.add_function(wrap_pyfunction!(rep_penalty_reset, m)?)?;
    m.add_function(wrap_pyfunction!(rep_penalty_update, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    Ok(())
}
