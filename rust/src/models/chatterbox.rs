use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use super::{tokenizer, GenerateInput, Prompt, PromptBuilder};

/// Chatterbox (English): `[cond rows][START text STOP][START_SPEECH x2]`, the row kind is passed
/// in the input mask (see `vox_serve/model/chatterbox/program.py`).
pub struct Chatterbox {
    tokenizer: Tokenizer,
    cond_len: usize,
    start_text: i32,
    stop_text: i32,
    start_speech: i32,
    kinds: [i32; 3],
}

impl Chatterbox {
    pub fn new(meta: &Value) -> Result<Self> {
        let int = |k: &str| meta[k].as_i64().map(|x| x as i32).ok_or_else(|| anyhow!("meta.{k} missing"));
        let kind = |k: &str| {
            meta["row_kinds"][k].as_i64().map(|x| x as i32).ok_or_else(|| anyhow!("meta.row_kinds.{k} missing"))
        };
        Ok(Self {
            tokenizer: tokenizer(meta, "tokenizer_json")?,
            cond_len: int("cond_len")? as usize,
            start_text: int("start_text")?,
            stop_text: int("stop_text")?,
            start_speech: int("start_speech")?,
            kinds: [kind("cond")?, kind("text")?, kind("speech_start")?],
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
        if input.audio_path.is_some() {
            bail!("Audio conditioning not implemented yet.");
        }
        let text = punc_norm(&input.text).replace(' ', "[SPACE]");
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let [cond, txt, start] = self.kinds;
        let mut tokens: Vec<i32> = (0..self.cond_len as i32).collect();
        let mut mask = vec![cond; self.cond_len];
        tokens.push(self.start_text);
        tokens.extend(enc.get_ids().iter().map(|&x| x as i32));
        tokens.push(self.stop_text);
        mask.resize(tokens.len(), txt);
        tokens.extend([self.start_speech, self.start_speech]);
        mask.extend([start, start]);
        Ok(Prompt { tokens, mask: Some(mask), ..Default::default() })
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
