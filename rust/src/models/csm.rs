use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use super::{meta_ids, tokenizer, GenerateInput, Prompt, PromptBuilder};

/// CSM-1B: 33 columns per row (32 Mimi codebooks + text), with a default two-speaker context.
pub struct Csm {
    tokenizer: Tokenizer,
    c: usize,
    context: Vec<i32>,
    context_mask: Vec<i32>,
}

impl Csm {
    pub fn new(meta: &Value) -> Result<Self> {
        Ok(Self {
            tokenizer: tokenizer(meta, "tokenizer_json")?,
            c: meta["n_codebooks"].as_u64().unwrap_or(33) as usize,
            context: meta_ids(meta, "context_tokens"),
            context_mask: meta_ids(meta, "context_mask"),
        })
    }
}

impl PromptBuilder for Csm {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        if input.audio_path.is_some() {
            bail!("CSM does not accept audio input");
        }
        let speaker = match input.model_kwargs.get("speaker") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            _ => "0".to_string(),
        };
        let enc =
            self.tokenizer.encode(format!("[{speaker}]{}", input.text), true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let n = enc.len();
        let mut tokens = Vec::with_capacity(self.context.len() + n * self.c);
        let mut mask = Vec::with_capacity(tokens.capacity());
        tokens.extend_from_slice(&self.context);
        mask.extend_from_slice(&self.context_mask);
        for &id in enc.get_ids() {
            tokens.extend(std::iter::repeat_n(0, self.c - 1));
            tokens.push(id as i32);
            mask.extend(std::iter::repeat_n(0, self.c - 1));
            mask.push(1);
        }
        Ok(Prompt { tokens, mask: Some(mask), ..Default::default() })
    }
}
