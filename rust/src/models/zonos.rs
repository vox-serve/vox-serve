use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use super::{GenerateInput, Prompt, PromptBuilder};
use crate::pyinit::Hooks;

const C: usize = 9;

/// Zonos: prefix rows (phonemes, then the constant conditioning rows) index the program's
/// precomputed prefix table; a final all-MASK code row starts generation.
pub struct Zonos {
    hooks: Hooks,
    mask_token: i32,
    n_phonemes: i32,
    n_const: i32,
    kinds: [i32; 2],
}

impl Zonos {
    pub fn new(meta: &Value) -> Result<Self> {
        let int = |v: &Value, k: &str| v[k].as_i64().map(|x| x as i32).ok_or_else(|| anyhow!("meta.{k} missing"));
        Ok(Self {
            hooks: Hooks::from_meta(meta),
            mask_token: int(meta, "mask_token")?,
            n_phonemes: int(meta, "n_phonemes")?,
            n_const: int(meta, "n_const_rows")?,
            kinds: [int(&meta["row_kinds"], "codes")?, int(&meta["row_kinds"], "prefix")?],
        })
    }
}

impl PromptBuilder for Zonos {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        if input.audio_path.is_some() {
            bail!("audio_path is not supported yet for this model");
        }
        let ids = self.hooks.call("phonemize", &json!({ "text": input.text }))?;
        let ids = ids.as_array().ok_or_else(|| anyhow!("phonemize returned {ids}"))?;
        let [codes, prefix] = self.kinds;
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        let mut push = |row: [i32; C], kind: i32| {
            tokens.extend_from_slice(&row);
            let mut m = [0; C];
            m[0] = kind;
            mask.extend_from_slice(&m);
        };
        for id in ids {
            let id = id.as_i64().ok_or_else(|| anyhow!("bad phoneme id {id}"))? as i32;
            let mut row = [0; C];
            row[0] = id.clamp(0, self.n_phonemes - 1);
            push(row, prefix);
        }
        for k in 0..self.n_const {
            let mut row = [0; C];
            row[0] = self.n_phonemes + k;
            push(row, prefix);
        }
        push([self.mask_token; C], codes);
        Ok(Prompt { tokens, mask: Some(mask), ..Default::default() })
    }
}
