//! Engine spec exported by `vox_serve.rt.build_engine` (JSON).

use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct FieldSpec {
    pub offset: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArenaSpec {
    pub ptr: u64,
    pub size: usize,
    pub fields: HashMap<String, FieldSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttentionSpec {
    pub batch_size: usize,
    pub total_num_rows: usize,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub float_ws_ptr: u64,
    pub float_ws_size: usize,
    pub int_ws_ptr: u64,
    pub int_ws_size: usize,
    pub plan_info: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphSpec {
    pub stage: String,
    pub key: Vec<usize>,
    pub graph: u64,
    pub attention: Option<AttentionSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EngineSpec {
    pub inputs: ArenaSpec,
    pub outputs: ArenaSpec,
    pub graphs: Vec<GraphSpec>,
    pub meta: serde_json::Value,
    /// Disaggregation: arenas of the detok stage on `detok_device`.
    #[serde(default)]
    pub detok_inputs: Option<ArenaSpec>,
    #[serde(default)]
    pub detok_outputs: Option<ArenaSpec>,
    #[serde(default)]
    pub detok_device: Option<i32>,
    /// Urgent detok lane: arenas of its own on `detok_device` (same layout as the detok arenas).
    #[serde(default)]
    pub urgent_inputs: Option<ArenaSpec>,
    #[serde(default)]
    pub urgent_outputs: Option<ArenaSpec>,
}
