use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use super::{tokenizer, GenerateInput, Prompt, PromptBuilder};

const C: usize = 37;

/// Voxtral-4B-TTS: `[BOS] [BEGIN_AUDIO] [AUDIO]*N [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT]
/// [BEGIN_AUDIO]` (mistral-common's `encode_speech_request`). `[AUDIO]` rows are voice rows whose
/// column 1 indexes the program's concatenated voice-embedding table.
pub struct VoxtralTts {
    tokenizer: Tokenizer,
    bos: i32,
    audio_token: i32,
    begin_audio: i32,
    text_to_audio: i32,
    audio_to_text: i32,
    /// Voice name -> (table offset, number of rows).
    voices: HashMap<String, (i32, usize)>,
    default_voice: String,
    kinds: [i32; 2],
}

impl VoxtralTts {
    pub fn new(meta: &Value) -> Result<Self> {
        let int = |v: &Value, k: &str| v[k].as_i64().map(|x| x as i32).ok_or_else(|| anyhow!("meta.{k} missing"));
        let voices: HashMap<String, (i32, usize)> = meta["voices"]
            .as_object()
            .ok_or_else(|| anyhow!("meta.voices missing"))?
            .iter()
            .map(|(k, v)| Ok((k.clone(), (int(v, "offset")?, int(v, "len")? as usize))))
            .collect::<Result<_>>()?;
        let default_voice = meta["default_voice"].as_str().ok_or_else(|| anyhow!("meta.default_voice missing"))?;
        if !voices.contains_key(default_voice) {
            bail!("default voice {default_voice} not in meta.voices");
        }
        Ok(Self {
            tokenizer: tokenizer(meta, "tokenizer_json")?,
            bos: int(meta, "bos")?,
            audio_token: int(meta, "audio_token")?,
            begin_audio: int(meta, "begin_audio")?,
            text_to_audio: int(meta, "text_to_audio")?,
            audio_to_text: int(meta, "audio_to_text")?,
            voices,
            default_voice: default_voice.to_string(),
            kinds: [int(&meta["row_kinds"], "text")?, int(&meta["row_kinds"], "voice")?],
        })
    }

    fn voice<'a>(&'a self, input: &'a GenerateInput) -> &'a str {
        let requested = match input.model_kwargs.get("voice") {
            Some(Value::String(s)) => Some(s.as_str()),
            Some(Value::Array(a)) => a.first().and_then(Value::as_str),
            _ => None,
        };
        match requested {
            Some(v) if self.voices.contains_key(v) => v,
            Some(v) => {
                tracing::warn!("Voxtral voice {v:?} not in registry; falling back to {:?}", self.default_voice);
                &self.default_voice
            }
            None => &self.default_voice,
        }
    }
}

impl PromptBuilder for VoxtralTts {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        if input.audio_path.is_some() {
            bail!("audio_path is not supported yet for this model");
        }
        let (offset, n_voice) = self.voices[self.voice(input)];
        let enc = self.tokenizer.encode(input.text.as_str(), false).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let [text, voice] = self.kinds;
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        let mut push = |tok: i32, kind: i32, voice_row: i32| {
            let mut row = [0i32; C];
            row[0] = tok;
            row[1] = voice_row;
            tokens.extend_from_slice(&row);
            let mut m = [0i32; C];
            m[0] = kind;
            mask.extend_from_slice(&m);
        };
        push(self.bos, text, 0);
        push(self.begin_audio, text, 0);
        for k in 0..n_voice {
            push(self.audio_token, voice, offset + k as i32);
        }
        push(self.text_to_audio, text, 0);
        for &id in enc.get_ids() {
            push(id as i32, text, 0);
        }
        push(self.audio_to_text, text, 0);
        push(self.begin_audio, text, 0);
        Ok(Prompt { tokens, mask: Some(mask), ..Default::default() })
    }
}
