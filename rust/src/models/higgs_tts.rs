use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use super::{meta_ids, tokenizer, GenerateInput, Prompt, PromptBuilder};
use crate::pyinit::Hooks;

const C: usize = 8;
/// Encoded reference clips kept (delayed code rows, keyed by a hash of the clip bytes).
const REF_CACHE: usize = 16;

/// Reference-clip cache: (clip hash, flattened delayed rows, last use) entries and the use counter.
type RefCache = (Vec<(u64, Arc<Vec<i32>>, u64)>, u64);

/// Higgs Audio v3 TTS. Prompt rows are `[id, 0, ..]` (text kind) or eight delayed codes (audio kind):
/// `<|tts|> [<|ref_text|> ref_text] [<|ref_audio|> ref codes] <|text|> text <|audio|>`, without the
/// bracketed parts for zero-shot requests (no reference audio). Output rows are delayed codes, or -1
/// once the request has finished.
pub struct HiggsTts {
    hooks: Hooks,
    tokenizer: Tokenizer,
    tts: i32,
    ref_text: i32,
    ref_audio: i32,
    text: i32,
    audio: i32,
    text_kind: i32,
    /// LRU of (clip hash, flattened delayed rows, last use).
    refs: Mutex<RefCache>,
    /// Serializes encodes, so one clip is encoded once.
    encode: Mutex<()>,
    /// Input mask of an audio row (reference codes).
    audio_mask: [i32; C],
}

impl HiggsTts {
    pub fn new(meta: &Value) -> Result<Self> {
        let int = |v: &Value, k: &str| v[k].as_i64().map(|x| x as i32).ok_or_else(|| anyhow!("meta.{k} missing"));
        let ids = &meta["special_ids"];
        let mut audio_mask = [0; C];
        audio_mask[0] = int(&meta["row_kinds"], "audio")?;
        Ok(Self {
            hooks: Hooks::from_meta(meta),
            tokenizer: tokenizer(meta, "tokenizer_json")?,
            tts: int(ids, "tts")?,
            ref_text: int(ids, "ref_text")?,
            ref_audio: int(ids, "ref_audio")?,
            text: int(ids, "text")?,
            audio: int(ids, "audio")?,
            text_kind: int(&meta["row_kinds"], "text")?,
            refs: Mutex::new((Vec::new(), 0)),
            encode: Mutex::new(()),
            audio_mask,
        })
    }

    fn encode_text(&self, s: &str) -> Result<Vec<i32>> {
        let enc = self.tokenizer.encode(s, false).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(enc.get_ids().iter().map(|&x| x as i32).collect())
    }

    fn lookup(&self, key: u64) -> Option<Arc<Vec<i32>>> {
        let mut g = self.refs.lock().unwrap();
        g.1 += 1;
        let now = g.1;
        g.0.iter_mut().find(|e| e.0 == key).map(|e| {
            e.2 = now;
            e.1.clone()
        })
    }

    /// Delayed code rows of reference clip `path`, encoding it on a cache miss.
    fn ref_codes(&self, path: &str) -> Result<Arc<Vec<i32>>> {
        let bytes = std::fs::read(path).map_err(|e| anyhow!("reading reference audio: {e}"))?;
        let mut h = DefaultHasher::new();
        bytes.hash(&mut h);
        let key = h.finish();
        if let Some(c) = self.lookup(key) {
            return Ok(c);
        }
        let _encoding = self.encode.lock().unwrap();
        if let Some(c) = self.lookup(key) {
            return Ok(c);
        }
        let r = self.hooks.call("encode_ref", &json!({ "path": path }))?;
        let codes = meta_ids(&r, "codes");
        if codes.is_empty() || !codes.len().is_multiple_of(C) {
            bail!("encode_ref returned {} codes", codes.len());
        }
        let codes = Arc::new(codes);
        let mut g = self.refs.lock().unwrap();
        let now = g.1;
        if g.0.len() >= REF_CACHE {
            let i = (0..g.0.len()).min_by_key(|&i| g.0[i].2).unwrap();
            g.0.swap_remove(i);
        }
        g.0.push((key, codes.clone(), now));
        Ok(codes)
    }
}

impl PromptBuilder for HiggsTts {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        let text_row = |id: i32, tokens: &mut Vec<i32>, mask: &mut Vec<i32>| {
            tokens.push(id);
            tokens.extend_from_slice(&[0; C - 1]);
            mask.push(self.text_kind);
            mask.extend_from_slice(&[0; C - 1]);
        };
        text_row(self.tts, &mut tokens, &mut mask);
        if let Some(path) = &input.audio_path {
            let codes = self.ref_codes(path)?;
            // The transcript only conditions together with the clip (as in the reference).
            if let Some(rt) = input.kwarg_str("ref_text").map(str::trim).filter(|t| !t.is_empty()) {
                text_row(self.ref_text, &mut tokens, &mut mask);
                for t in self.encode_text(rt)? {
                    text_row(t, &mut tokens, &mut mask);
                }
            }
            text_row(self.ref_audio, &mut tokens, &mut mask);
            tokens.extend_from_slice(&codes);
            for _ in 0..codes.len() / C {
                mask.extend_from_slice(&self.audio_mask);
            }
        }
        text_row(self.text, &mut tokens, &mut mask);
        for t in self.encode_text(&input.text)? {
            text_row(t, &mut tokens, &mut mask);
        }
        text_row(self.audio, &mut tokens, &mut mask);
        Ok(Prompt { tokens, mask: Some(mask), features: None, hold: None })
    }
}
