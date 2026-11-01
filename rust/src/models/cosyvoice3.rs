use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use super::cosyvoice2::text_normalize;
use super::{meta_ids, tokenizer, GenerateInput, Prompt, PromptBuilder};
use crate::pyinit::Hooks;

/// A reference voice encoded into one of the detokenizer's voice slots. Requests hold an `Arc`
/// until they finish; a slot is only re-encoded once no request holds its voice.
struct Voice {
    slot: i32,
    speech: Vec<i32>,
}

/// Voice slots: 0 holds the default voice, the others an LRU of reference clips keyed by content.
struct Voices {
    /// Per slot: (content hash, voice, last use).
    slots: Vec<Option<(u64, Arc<Voice>, u64)>>,
    clock: u64,
}

impl Voices {
    fn lookup(&mut self, key: u64) -> Option<Arc<Voice>> {
        self.clock += 1;
        let now = self.clock;
        self.slots.iter_mut().skip(1).flatten().find(|(k, _, _)| *k == key).map(|e| {
            e.2 = now;
            e.1.clone()
        })
    }

    /// An empty slot, or the least recently used one no request holds (its entry is dropped).
    fn reserve(&mut self) -> Option<usize> {
        if let Some(i) = (1..self.slots.len()).find(|&i| self.slots[i].is_none()) {
            return Some(i);
        }
        let i = (1..self.slots.len())
            .filter(|&i| self.slots[i].as_ref().is_some_and(|(_, v, _)| Arc::strong_count(v) == 1))
            .min_by_key(|&i| self.slots[i].as_ref().unwrap().2)?;
        self.slots[i] = None;
        Some(i)
    }
}

/// Fun-CosyVoice3 with zero-shot voice cloning. Prompt rows are `[id, n_text]` with mask
/// `[kind, voice slot]`: `[sos] system_prompt ref_text text [task] ref_speech` (all speech-kind
/// except the text). Without a reference transcript the prompt is `[sos] system_prompt text [task]`
/// (the reference's cross-lingual mode). Output rows are `[token, voice]`, `voice < 0` for tokens
/// that are fed back but not vocoded.
pub struct CosyVoice3 {
    hooks: Hooks,
    tokenizer: Tokenizer,
    sos: i32,
    task_id: i32,
    text_kind: i32,
    speech_kind: i32,
    system_prompt: String,
    default_text: String,
    voices: Mutex<Voices>,
    /// Serializes encodes (the Python hooks run one at a time anyway), so one clip is encoded once.
    encode: Mutex<()>,
}

impl CosyVoice3 {
    pub fn new(meta: &Value) -> Result<Self> {
        let int = |v: &Value, k: &str| v[k].as_i64().map(|x| x as i32).ok_or_else(|| anyhow!("meta.{k} missing"));
        let text_kind = int(&meta["row_kinds"], "text")?;
        let speech_kind = int(&meta["row_kinds"], "speech")?;
        let n_voices = meta["n_voices"].as_u64().ok_or_else(|| anyhow!("meta.n_voices missing"))? as usize;
        let dv = &meta["default_voice"];
        let mut slots = vec![None; n_voices.max(1)];
        slots[0] = Some((0, Arc::new(Voice { slot: 0, speech: meta_ids(dv, "speech_tokens") }), 0));
        Ok(Self {
            hooks: Hooks::from_meta(meta),
            tokenizer: tokenizer(meta, "tokenizer_json")?,
            sos: int(meta, "sos")?,
            task_id: int(meta, "task_id")?,
            text_kind,
            speech_kind,
            system_prompt: meta["system_prompt"].as_str().unwrap_or_default().to_string(),
            default_text: dv["ref_text"].as_str().unwrap_or_default().to_string(),
            voices: Mutex::new(Voices { slots, clock: 0 }),
            encode: Mutex::new(()),
        })
    }

    fn encode_text(&self, s: &str) -> Result<Vec<i32>> {
        let enc = self.tokenizer.encode(s, true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(enc.get_ids().iter().map(|&x| x as i32).collect())
    }

    /// The voice slot of reference clip `path`, encoding it on a cache miss.
    fn voice(&self, path: &str) -> Result<Arc<Voice>> {
        let bytes = std::fs::read(path).map_err(|e| anyhow!("reading reference audio: {e}"))?;
        let mut h = DefaultHasher::new();
        bytes.hash(&mut h);
        let key = h.finish();
        if let Some(v) = self.voices.lock().unwrap().lookup(key) {
            return Ok(v);
        }
        let _encoding = self.encode.lock().unwrap();
        let deadline = Instant::now() + Duration::from_secs(120);
        let slot = loop {
            let mut voices = self.voices.lock().unwrap();
            if let Some(v) = voices.lookup(key) {
                return Ok(v);
            }
            if let Some(i) = voices.reserve() {
                break i;
            }
            if Instant::now() >= deadline {
                bail!("all {} voice slots are in use by running requests", voices.slots.len() - 1);
            }
            drop(voices);
            // Running requests release their voices when they finish.
            std::thread::sleep(Duration::from_millis(20));
        };
        let r = self.hooks.call("encode_voice", &json!({ "path": path, "slot": slot }))?;
        let speech = meta_ids(&r, "speech_tokens");
        if speech.is_empty() {
            bail!("encode_voice returned no speech tokens");
        }
        let v = Arc::new(Voice { slot: slot as i32, speech });
        let mut voices = self.voices.lock().unwrap();
        let now = voices.clock;
        voices.slots[slot] = Some((key, v.clone(), now));
        Ok(v)
    }
}

impl PromptBuilder for CosyVoice3 {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        let ref_text = input.kwarg_str("ref_text").map(str::trim).filter(|t| !t.is_empty());
        let (voice, ref_text) = match &input.audio_path {
            Some(path) => (self.voice(path)?, ref_text.map(str::to_string)),
            None => {
                if ref_text.is_some() {
                    bail!("ref_text needs reference audio");
                }
                let v = self.voices.lock().unwrap().slots[0].as_ref().unwrap().1.clone();
                (v, Some(self.default_text.clone()))
            }
        };
        let text = text_normalize(&input.text)?;
        // The reference prepends the instruction to the prompt transcript (zero-shot) or to the text.
        let with_system = |t: &str| {
            if t.contains("<|endofprompt|>") {
                t.to_string()
            } else {
                format!("{}{t}", self.system_prompt)
            }
        };
        let (prefix, text_ids) = match &ref_text {
            Some(rt) => (self.encode_text(&with_system(rt))?, self.encode_text(&text)?),
            None => (Vec::new(), self.encode_text(&with_system(&text))?),
        };
        let n_text = text_ids.len() as i32;
        let slot = voice.slot;
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        let mut push = |id: i32, kind: i32| {
            tokens.extend_from_slice(&[id, n_text]);
            mask.extend_from_slice(&[kind, slot]);
        };
        push(self.sos, self.speech_kind);
        for &t in prefix.iter().chain(&text_ids) {
            push(t, self.text_kind);
        }
        push(self.task_id, self.speech_kind);
        if ref_text.is_some() {
            for &t in &voice.speech {
                push(t, self.speech_kind);
            }
        }
        let hold: Arc<dyn Any + Send + Sync> = voice;
        Ok(Prompt { tokens, mask: Some(mask), features: None, hold: Some(hold) })
    }
}
