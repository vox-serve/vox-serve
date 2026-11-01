use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use super::{FeatureLease, FeaturePool, GenerateInput, Prompt, PromptBuilder};
use crate::pyinit::Hooks;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelType {
    /// Preset speakers (speaker-id row).
    CustomVoice,
    /// Voice cloning: speaker-embedding row, plus reference text / codes in ICL mode.
    Base,
    /// Voice described by the instruction; no speaker row.
    VoiceDesign,
}

/// Reference voice of a Base-model request.
struct Voice {
    /// Feature-pool row holding the speaker embedding.
    lease: FeatureLease,
    /// ICL mode: reference transcript tokens and reference codes `[T][16]`.
    icl: Option<(Vec<i32>, Vec<Vec<i32>>)>,
}

/// Qwen3-TTS: rows are `[c0..c15, text]`; mask column `i < 16` adds the embedding of codebook `i`
/// to the text embedding (prompt rows use column 0, ICL reference rows and decode rows all 16). On
/// Base models a nonzero text-column mask adds feature-pool row `m - 1` (the speaker embedding).
pub struct Qwen3Tts {
    hooks: Hooks,
    model_type: ModelType,
    pool: FeaturePool,
    tokenizer: Tokenizer,
    c: usize,
    ids: HashMap<&'static str, i32>,
    languages: HashMap<String, i32>,
    speakers: HashMap<String, i32>,
    dialects: HashMap<String, String>,
    default_speaker: String,
    default_language: String,
    supports_instruct: bool,
    /// `<|im_start|>assistant\n` tokens.
    role: Vec<i32>,
}

fn int_map(v: &Value) -> HashMap<String, i32> {
    v.as_object()
        .map(|m| m.iter().filter_map(|(k, x)| Some((k.clone(), x.as_i64()? as i32))).collect())
        .unwrap_or_default()
}

impl Qwen3Tts {
    pub fn new(meta: &Value) -> Result<Self> {
        let tokenizer = super::tokenizer(meta, "tokenizer_json")?;
        let mut ids = HashMap::new();
        for k in [
            "tts_pad",
            "tts_bos",
            "tts_eos",
            "codec_pad",
            "codec_bos",
            "codec_think",
            "codec_nothink",
            "codec_think_bos",
            "codec_think_eos",
        ] {
            let v = meta[k].as_i64().ok_or_else(|| anyhow!("meta.{k} missing"))?;
            ids.insert(k, v as i32);
        }
        let dialects = meta["speaker_dialect"]
            .as_object()
            .map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect())
            .unwrap_or_default();
        let role = tokenizer
            .encode("<|im_start|>assistant\n", true)
            .map_err(|e| anyhow!("tokenizer: {e}"))?
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        if role.len() != 3 {
            bail!("unexpected tokenization of the role header: {role:?}");
        }
        let model_type = match meta["tts_model_type"].as_str().unwrap_or("custom_voice") {
            "custom_voice" => ModelType::CustomVoice,
            "base" => ModelType::Base,
            "voice_design" => ModelType::VoiceDesign,
            t => bail!("unknown Qwen3-TTS model type {t}"),
        };
        Ok(Self {
            hooks: Hooks::from_meta(meta),
            model_type,
            pool: FeaturePool::new(meta["feature_pool_rows"].as_u64().unwrap_or(0) as usize),
            role,
            tokenizer,
            c: meta["n_codebooks"].as_u64().unwrap_or(17) as usize,
            ids,
            languages: int_map(&meta["languages"]),
            speakers: int_map(&meta["speakers"]),
            dialects,
            default_speaker: meta["default_speaker"].as_str().unwrap_or("ryan").to_string(),
            default_language: meta["default_language"].as_str().unwrap_or("english").to_string(),
            supports_instruct: meta["supports_instruct"].as_bool().unwrap_or(false),
        })
    }

    /// Prompt rows around the synthesis text tokens `text`.
    fn build(&self, input: &GenerateInput, text: &[i32], streaming: bool) -> Result<Prompt> {
        if input.audio_path.is_some() && self.model_type != ModelType::Base {
            bail!("this Qwen3-TTS model does not accept audio input (voice cloning needs the Base model)");
        }
        let voice = match self.model_type {
            ModelType::Base => Some(self.encode_voice(input, streaming)?),
            _ => None,
        };
        let id = |k: &str| self.ids[k];
        let mut language = input.kwarg_str("language").unwrap_or(&self.default_language).to_lowercase();
        if language != "auto" && !self.languages.contains_key(&language) {
            tracing::warn!("language {language} not supported, using auto");
            language = "auto".into();
        }
        let speaker = (self.model_type == ModelType::CustomVoice).then(|| {
            let s = input.kwarg_str("speaker").unwrap_or(&self.default_speaker).to_lowercase();
            if self.speakers.contains_key(&s) {
                s
            } else {
                tracing::warn!("speaker {s} not supported, using {}", self.default_speaker);
                self.default_speaker.clone()
            }
        });
        let instruct = input.kwarg_str("instruct").filter(|s| self.supports_instruct && !s.is_empty());
        let mut language_id = (language != "auto").then(|| self.languages[&language]);
        if let Some(sp) = &speaker {
            if language == "chinese" || language == "auto" {
                if let Some(d) = self.dialects.get(sp) {
                    language_id = self.languages.get(d).copied().or(language_id);
                }
            }
        }

        let c = self.c;
        let mut p = Prompt { tokens: Vec::new(), mask: Some(Vec::new()), ..Default::default() };
        if let Some(ins) = instruct {
            for t in self.encode(&format!("<|im_start|>user\n{ins}<|im_end|>\n"))? {
                push_row(&mut p, c, t, None);
            }
        }
        for &t in &self.role {
            push_row(&mut p, c, t, None);
        }
        let prefix = match language_id {
            None => vec![id("codec_nothink"), id("codec_think_bos"), id("codec_think_eos")],
            Some(l) => vec![id("codec_think"), id("codec_think_bos"), l, id("codec_think_eos")],
        };
        for t in prefix {
            push_row(&mut p, c, id("tts_pad"), Some(t));
        }
        if let Some(sp) = &speaker {
            push_row(&mut p, c, id("tts_pad"), Some(self.speakers[sp]));
        }
        if let Some(v) = &voice {
            // Text embedding + speaker embedding (no codec embedding).
            push_row(&mut p, c, id("tts_pad"), None);
            *p.mask.as_mut().unwrap().last_mut().unwrap() = v.lease.offset as i32 + 1;
        }
        push_row(&mut p, c, id("tts_bos"), Some(id("codec_pad")));
        let icl = voice.as_ref().and_then(|v| v.icl.as_ref());
        if let Some((ref_text, _)) = icl {
            for &t in ref_text {
                push_row(&mut p, c, t, Some(id("codec_pad")));
            }
        }
        for (k, &t) in text.iter().enumerate() {
            let codec = if streaming && k + 1 == text.len() { id("codec_bos") } else { id("codec_pad") };
            push_row(&mut p, c, t, Some(codec));
        }
        if !streaming {
            push_row(&mut p, c, id("tts_eos"), Some(id("codec_pad")));
            push_row(&mut p, c, id("tts_pad"), Some(id("codec_bos")));
        }
        if let Some((_, codes)) = icl {
            let mask = p.mask.as_mut().unwrap();
            for frame in codes {
                p.tokens.extend_from_slice(frame);
                p.tokens.push(id("tts_pad"));
                mask.extend(std::iter::repeat_n(1, c - 1));
                mask.push(0);
            }
        }
        p.features = voice.map(|v| Arc::new(v.lease));
        Ok(p)
    }

    /// Runs the `encode_voice` hook and commits the speaker embedding to a feature-pool row.
    fn encode_voice(&self, input: &GenerateInput, streaming: bool) -> Result<Voice> {
        let mut x_vector_only = input.model_kwargs.get("x_vector_only_mode").and_then(Value::as_bool).unwrap_or(false);
        if streaming && !x_vector_only {
            // ICL rows need the whole text before the reference codes, which input streaming does not have.
            if input.audio_path.is_some() {
                bail!(
                    "Input streaming is not supported with ICL mode (voice cloning with reference audio). \
                     Please use x_vector_only_mode=True or disable input streaming."
                );
            }
            x_vector_only = true; // no reference audio: the default voice's speaker embedding
        }
        let r = self.hooks.call(
            "encode_voice",
            &json!({ "path": input.audio_path, "ref_text": input.kwarg_str("ref_text"), "x_vector_only": x_vector_only }),
        )?;
        let key = r["key"].as_str().ok_or_else(|| anyhow!("encode_voice returned {r}"))?.to_string();
        let staged = || -> Result<Voice> {
            let lease = self.pool.alloc(1, Duration::from_secs(60))?;
            self.hooks.call("features_commit", &json!({ "key": key, "offset": lease.offset }))?;
            let icl = match (r.get("codes"), r["ref_text"].as_str()) {
                (Some(Value::Array(frames)), Some(text)) => {
                    let ids = self.encode(&format!("<|im_start|>assistant\n{text}<|im_end|>\n"))?;
                    if ids.len() < 5 {
                        bail!("unexpected tokenization of the reference text");
                    }
                    let codes = frames
                        .iter()
                        .map(|f| {
                            let row: Option<Vec<i32>> =
                                f.as_array()?.iter().map(|x| x.as_i64().map(|v| v as i32)).collect();
                            row.filter(|r| r.len() == self.c - 1)
                        })
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| anyhow!("encode_voice: malformed reference codes"))?;
                    Some((ids[3..ids.len() - 2].to_vec(), codes))
                }
                _ => None,
            };
            Ok(Voice { lease, icl })
        };
        let res = staged();
        if res.is_err() {
            let _ = self.hooks.call("features_discard", &json!({ "key": key }));
        }
        res
    }

    fn encode(&self, s: &str) -> Result<Vec<i32>> {
        let enc = self.tokenizer.encode(s, true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(enc.get_ids().iter().map(|&x| x as i32).collect())
    }
}

/// Appends one row: `text` in the last column, optionally `codec` in column 0.
fn push_row(p: &mut Prompt, c: usize, text: i32, codec: Option<i32>) {
    let mask = p.mask.as_mut().unwrap();
    p.tokens.push(codec.unwrap_or(0));
    p.tokens.extend(std::iter::repeat_n(0, c - 2));
    p.tokens.push(text);
    mask.push(i32::from(codec.is_some()));
    mask.extend(std::iter::repeat_n(0, c - 1));
}

impl PromptBuilder for Qwen3Tts {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        let prompt_ids =
            self.encode(&format!("<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n", input.text))?;
        if prompt_ids.len() < 8 {
            bail!("unexpected tokenization of the prompt template");
        }
        self.build(input, &prompt_ids[3..prompt_ids.len() - 5], false)
    }

    fn tokenize_stream_text(&self, text: &str) -> Result<Vec<i32>> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(enc.get_ids().iter().map(|&x| x as i32).collect())
    }

    /// Streaming layout (`<|im_start|>assistant\n{text}` without the closing template): the last
    /// text row carries `codec_bos`, and there is no `tts_eos` row; the remaining text arrives on
    /// decode rows.
    fn preprocess_streaming(&self, input: &GenerateInput, first: i32) -> Result<Prompt> {
        self.build(input, &[first], true)
    }
}
