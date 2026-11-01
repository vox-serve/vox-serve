use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use super::{tokenizer, FeaturePool, GenerateInput, Prompt, PromptBuilder};
use crate::pyinit::Hooks;

/// Step-Audio-2-mini: chat-formatted text prompt, or (audio input) a prompt built by the
/// `encode_audio` hook whose audio-patch rows read encoded features from the feature pool
/// (mask value = pool row + 1).
pub struct StepAudio2 {
    hooks: Hooks,
    tokenizer: Tokenizer,
    segments: Vec<String>,
    pool: FeaturePool,
}

impl StepAudio2 {
    pub fn new(meta: &Value) -> Result<Self> {
        let segments = meta["prompt_segments"]
            .as_array()
            .ok_or_else(|| anyhow!("meta.prompt_segments missing"))?
            .iter()
            .map(|s| s.as_str().map(str::to_string).ok_or_else(|| anyhow!("bad prompt segment")))
            .collect::<Result<_>>()?;
        Ok(Self {
            hooks: Hooks::from_meta(meta),
            tokenizer: tokenizer(meta, "tokenizer_json")?,
            segments,
            pool: FeaturePool::new(meta["feature_pool_rows"].as_u64().unwrap_or(0) as usize),
        })
    }

    fn audio_prompt(&self, path: &str) -> Result<Prompt> {
        let r = self.hooks.call("encode_audio", &json!({ "path": path }))?;
        let key = r["key"].as_str().ok_or_else(|| anyhow!("encode_audio returned {r}"))?.to_string();
        let staged = || -> Result<Prompt> {
            let rows = r["rows"].as_u64().ok_or_else(|| anyhow!("encode_audio: rows missing"))? as usize;
            let lease = self.pool.alloc(rows, Duration::from_secs(60))?;
            self.hooks.call("features_commit", &json!({ "key": key, "offset": lease.offset }))?;
            let tokens: Vec<i32> = r["ids"]
                .as_array()
                .ok_or_else(|| anyhow!("encode_audio: ids missing"))?
                .iter()
                .map(|x| x.as_i64().map(|v| v as i32).ok_or_else(|| anyhow!("bad id")))
                .collect::<Result<_>>()?;
            let mut mask = vec![0; tokens.len()];
            let mut row = lease.offset as i32;
            for span in r["spans"].as_array().ok_or_else(|| anyhow!("encode_audio: spans missing"))? {
                let start = span[0].as_u64().unwrap_or(0) as usize;
                let n = span[1].as_u64().unwrap_or(0) as usize;
                for m in mask.iter_mut().skip(start).take(n) {
                    *m = row + 1;
                    row += 1;
                }
            }
            Ok(Prompt { tokens, mask: Some(mask), features: Some(Arc::new(lease)), ..Default::default() })
        };
        let res = staged();
        if res.is_err() {
            let _ = self.hooks.call("features_discard", &json!({ "key": key }));
        }
        res
    }
}

impl PromptBuilder for StepAudio2 {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        if let Some(path) = &input.audio_path {
            return self.audio_prompt(path);
        }
        let mut tokens = Vec::new();
        for seg in &self.segments {
            let text = seg.replacen("{prompt}", &input.text, 1);
            let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("tokenizer: {e}"))?;
            tokens.extend(enc.get_ids().iter().map(|&x| x as i32));
        }
        let mask = vec![0; tokens.len()];
        Ok(Prompt { tokens, mask: Some(mask), ..Default::default() })
    }
}
