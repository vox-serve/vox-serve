use std::any::Any;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use super::voices::VoiceSlots;
use super::{tokenizer, GenerateInput, Prompt, PromptBuilder};
use crate::pyinit::Hooks;

/// Chatterbox (English): `[cond rows][START text STOP][START_SPEECH x2]`. Rows are `[id, 0]` with
/// mask `[kind, voice slot]` (see `vox_serve/model/chatterbox/program.py`); reference audio is encoded
/// into a voice slot by the program's `encode_voice` hook.
pub struct Chatterbox {
    hooks: Hooks,
    tokenizer: Tokenizer,
    cond_len: usize,
    start_text: i32,
    stop_text: i32,
    start_speech: i32,
    kinds: [i32; 3],
    voices: VoiceSlots,
}

impl Chatterbox {
    pub fn new(meta: &Value) -> Result<Self> {
        let int = |k: &str| meta[k].as_i64().map(|x| x as i32).ok_or_else(|| anyhow!("meta.{k} missing"));
        let kind = |k: &str| {
            meta["row_kinds"][k].as_i64().map(|x| x as i32).ok_or_else(|| anyhow!("meta.row_kinds.{k} missing"))
        };
        Ok(Self {
            hooks: Hooks::from_meta(meta),
            tokenizer: tokenizer(meta, "tokenizer_json")?,
            cond_len: int("cond_len")? as usize,
            start_text: int("start_text")?,
            stop_text: int("stop_text")?,
            start_speech: int("start_speech")?,
            kinds: [kind("cond")?, kind("text")?, kind("speech_start")?],
            voices: VoiceSlots::new(meta["n_voices"].as_u64().unwrap_or(1) as usize),
        })
    }
}

/// Port of `ChatterboxModel._punc_norm`.
pub fn punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".into();
    }
    let mut chars = text.chars();
    let first = chars.next().unwrap();
    let mut t =
        if first.is_lowercase() { first.to_uppercase().chain(chars).collect::<String>() } else { text.to_string() };
    t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    for (old, new) in [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ] {
        t = t.replace(old, new);
    }
    let mut t = t.trim_end_matches(' ').to_string();
    if ![".", "!", "?", "-", ","].iter().any(|p| t.ends_with(p)) {
        t.push('.');
    }
    t
}

impl PromptBuilder for Chatterbox {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        let voice = match &input.audio_path {
            Some(path) => self.voices.get(path, |slot| {
                self.hooks.call("encode_voice", &json!({ "path": path, "slot": slot })).map(|_| ())
            })?,
            None => self.voices.default_voice(),
        };
        let text = punc_norm(&input.text).replace(' ', "[SPACE]");
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let [cond, txt, start] = self.kinds;
        let mut ids: Vec<i32> = (0..self.cond_len as i32).collect();
        let mut kinds = vec![cond; self.cond_len];
        ids.push(self.start_text);
        ids.extend(enc.get_ids().iter().map(|&x| x as i32));
        ids.push(self.stop_text);
        kinds.resize(ids.len(), txt);
        ids.extend([self.start_speech, self.start_speech]);
        kinds.extend([start, start]);
        let tokens = ids.iter().flat_map(|&i| [i, 0]).collect();
        let mask = kinds.iter().flat_map(|&k| [k, voice.slot]).collect();
        let hold: Arc<dyn Any + Send + Sync> = voice;
        Ok(Prompt { tokens, mask: Some(mask), features: None, hold: Some(hold) })
    }
}

#[cfg(test)]
mod tests {
    use super::punc_norm;

    #[test]
    fn punc_norm_matches_reference() {
        assert_eq!(punc_norm("hello   world"), "Hello world.");
        assert_eq!(punc_norm("wait... what; ok"), "Wait,  what,  ok.");
        assert_eq!(punc_norm("élan — “quoted” : x"), "Élan - \"quoted\", x.");
        assert_eq!(punc_norm("Done!"), "Done!");
        assert_eq!(punc_norm(""), "You need to add some text for me to talk.");
    }
}
