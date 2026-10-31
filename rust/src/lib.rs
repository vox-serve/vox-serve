//! vox-serve Rust runtime.
//!
//! Start-up: Python loads the model and captures CUDA graphs. Serving: HTTP, scheduling, KV paging,
//! FlashInfer planning, graph launches and audio packaging, all in Rust. Built as the `vox_serve._rt`
//! extension module, which the package's `vox-serve` command runs.

pub mod cli;
mod cuda;
mod deadline;
mod engine;
mod flashinfer;
pub mod models;
mod pyinit;
pub mod pymodule;
mod scheduler;
mod server;
mod spec;
mod v2;
mod wav;
