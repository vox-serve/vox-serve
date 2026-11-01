use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use super::{meta_ids, GenerateInput, Prompt, PromptBuilder};
use crate::pyinit::Hooks;

/// Prompt built by the program's `preprocess` (`hook_preprocess`), one Python call per request. No
/// Rust needed to bring up a model; the call holds the GIL, so a model under heavy request rates
/// gets its own builder in Rust. Supports neither feature-pool rows nor input streaming.
pub struct Python {
    hooks: Hooks,
}

impl Python {
    pub fn new(meta: &Value) -> Self {
        Self { hooks: Hooks::from_meta(meta) }
    }
}

impl PromptBuilder for Python {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        let r = self.hooks.call(
            "preprocess",
            &json!({ "text": input.text, "audio_path": input.audio_path, "model_kwargs": input.model_kwargs }),
        )?;
        if !r["tokens"].is_array() {
            return Err(anyhow!("preprocess returned {r}"));
        }
        let mask = r["mask"].is_array().then(|| meta_ids(&r, "mask"));
        Ok(Prompt { tokens: meta_ids(&r, "tokens"), mask, ..Default::default() })
    }
}
