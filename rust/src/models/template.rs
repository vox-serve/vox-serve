use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use super::{meta_ids, tokenizer, GenerateInput, Prompt, PromptBuilder};

/// Request option substituted into the template as `{name}`.
struct Kwarg {
    name: String,
    default: String,
    /// Accepted values (any value if empty).
    choices: Vec<String>,
}

/// Generic single-codebook prompt from `meta["prompt_template"]`:
/// `prefix_ids + tokenize(text) + suffix_ids`, where `text` (e.g. `"{voice}: {text}"`) has every
/// `{kwarg}` replaced by the request's option (or its default), then `{text}` by the request text.
///
/// ```json
/// {"tokenizer_json": "...", "prefix_ids": [..], "suffix_ids": [..], "text": "{voice}: {text}",
///  "add_special_tokens": true, "kwargs": {"voice": {"default": "tara", "choices": ["tara", ..]}}}
/// ```
pub struct Template {
    tokenizer: Tokenizer,
    prefix: Vec<i32>,
    suffix: Vec<i32>,
    text: String,
    add_special_tokens: bool,
    kwargs: Vec<Kwarg>,
}

impl Template {
    pub fn new(meta: &Value) -> Result<Self> {
        if meta["n_codebooks"].as_u64() != Some(1) {
            bail!("the template prompt builder supports single-codebook models only");
        }
        let t = &meta["prompt_template"];
        if !t.is_object() {
            bail!("meta.prompt_template missing");
        }
        let strings = |v: &Value| -> Vec<String> {
            v.as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default()
        };
        let kwargs = t["kwargs"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(name, k)| Kwarg {
                        name: name.clone(),
                        default: k["default"].as_str().unwrap_or_default().to_string(),
                        choices: strings(&k["choices"]),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            tokenizer: tokenizer(t, "tokenizer_json")?,
            prefix: meta_ids(t, "prefix_ids"),
            suffix: meta_ids(t, "suffix_ids"),
            text: t["text"].as_str().unwrap_or("{text}").to_string(),
            add_special_tokens: t["add_special_tokens"].as_bool().unwrap_or(true),
            kwargs,
        })
    }
}

impl PromptBuilder for Template {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        if input.audio_path.is_some() {
            bail!("this model does not accept audio input");
        }
        let mut text = self.text.clone();
        for k in &self.kwargs {
            let value = match input.model_kwargs.get(&k.name) {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Number(n)) => n.to_string(),
                _ => k.default.clone(),
            };
            if !k.choices.is_empty() && !k.choices.contains(&value) {
                bail!("{} {value} is not available for this model", k.name);
            }
            text = text.replace(&format!("{{{}}}", k.name), &value);
        }
        // The request text goes in last, so braces in it are never substituted.
        let text = text.replacen("{text}", &input.text, 1);
        let enc = self.tokenizer.encode(text, self.add_special_tokens).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let mut tokens = Vec::with_capacity(self.prefix.len() + enc.len() + self.suffix.len());
        tokens.extend_from_slice(&self.prefix);
        tokens.extend(enc.get_ids().iter().map(|&x| x as i32));
        tokens.extend_from_slice(&self.suffix);
        Ok(Prompt { tokens, mask: None, ..Default::default() })
    }
}
