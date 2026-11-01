use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use super::{tokenizer, GenerateInput, Prompt, PromptBuilder};
use crate::pyinit::Hooks;

/// GLM-4-Voice: chat template around the text, or (speech input) around the `<|audio_N|>` tokens of
/// the clip, which the Python audio encoder produces (`hook_encode_audio`).
pub struct GlmVoice {
    hooks: Hooks,
    tokenizer: Tokenizer,
    text_template: String,
    speech_template: String,
}

impl GlmVoice {
    pub fn new(meta: &Value) -> Result<Self> {
        let s = |k: &str| meta[k].as_str().map(String::from).ok_or_else(|| anyhow!("meta.{k} missing"));
        Ok(Self {
            hooks: Hooks::from_meta(meta),
            tokenizer: tokenizer(meta, "tokenizer_json")?,
            text_template: s("text_prompt_template")?,
            speech_template: s("speech_prompt_template")?,
        })
    }
}

impl PromptBuilder for GlmVoice {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        let text = match &input.audio_path {
            Some(path) => {
                let codes = self.hooks.call("encode_audio", &json!({ "path": path }))?;
                let codes = codes.as_array().ok_or_else(|| anyhow!("encode_audio returned {codes}"))?;
                let mut user = String::from("<|begin_of_audio|>");
                for c in codes {
                    user.push_str(&format!("<|audio_{}|>", c.as_i64().unwrap_or(0)));
                }
                user.push_str("<|end_of_audio|>");
                self.speech_template.replacen("{prompt}", &user, 1)
            }
            None => self.text_template.replacen("{prompt}", &input.text, 1),
        };
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(Prompt { tokens: enc.get_ids().iter().map(|&x| x as i32).collect(), mask: None, ..Default::default() })
    }
}
