//! Start-up: load the model with the PyTorch implementation and capture the CUDA graphs
//! (`vox_serve.rt.build_engine`) in the interpreter that loaded the extension module. Afterwards
//! Python is only re-entered for per-request preprocessing hooks (`call_hook`, e.g. encoding
//! reference audio), never from the engine loop.

use anyhow::{anyhow, Result};
use pyo3::prelude::*;

use crate::spec::EngineSpec;

pub fn build_engine(config: &serde_json::Value) -> Result<EngineSpec> {
    let json = Python::with_gil(|py| -> PyResult<String> {
        let rt = py.import("vox_serve.rt")?;
        let out = rt.call_method1("build_engine", (config.to_string(),))?;
        out.extract::<String>()
    })
    .map_err(|e| py_error("build_engine", e))?;
    Ok(serde_json::from_str(&json)?)
}

/// Per-request Python hooks of one engine (`program.hook_<name>(**args)`).
#[derive(Debug, Clone, Copy)]
pub struct Hooks {
    engine: usize,
}

impl Hooks {
    pub fn from_meta(meta: &serde_json::Value) -> Self {
        Self { engine: meta["engine_index"].as_u64().unwrap_or(0) as usize }
    }

    /// Runs the hook in the embedded interpreter (blocking; takes the GIL).
    pub fn call(&self, name: &str, args: &serde_json::Value) -> Result<serde_json::Value> {
        let json = Python::with_gil(|py| -> PyResult<String> {
            py.import("vox_serve.rt")?
                .call_method1("call_hook", (self.engine, name, args.to_string()))?
                .extract::<String>()
        })
        .map_err(|e| py_error(name, e))?;
        Ok(serde_json::from_str(&json)?)
    }
}

fn py_error(what: &str, e: PyErr) -> anyhow::Error {
    Python::with_gil(|py| {
        if let Some(tb) = e.traceback(py) {
            if let Ok(s) = tb.format() {
                return anyhow!("python {what} failed: {e}\n{s}");
            }
        }
        anyhow!("python {what} failed: {e}")
    })
}
