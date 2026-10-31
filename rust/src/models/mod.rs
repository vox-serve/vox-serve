//! The CPU side of a model: constants from the program's `meta` and the prompt builder. Everything
//! that runs on the GPU, including what a sampled row means (`out_kind`), lives in the captured graphs
//! (`vox_serve/model/<name>/program.py`), so the only per-model Rust code is prompt construction.
//!
//! The program's `meta["prompt_builder"]` (default: `meta["kind"]`) picks the builder: `template`
//! (prefix ids + tokenized template + suffix ids, configured in `meta["prompt_template"]`), `python`
//! (the program's `preprocess`, one Python call per request), or a model's own builder below.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

mod chatterbox;
mod cosyvoice2;
mod cosyvoice3;
mod csm;
mod features;
mod glm_voice;
mod higgs_tts;
mod python;
mod qwen3_tts;
mod step_audio_2;
mod template;
mod voxtral_tts;
mod zonos;

pub use features::{FeatureLease, FeaturePool};

/// Prompt rows, flattened `[L, n_codebooks]`, with an optional per-column input mask.
#[derive(Debug, Clone, Default)]
pub struct Prompt {
    pub tokens: Vec<i32>,
    pub mask: Option<Vec<i32>>,
    /// Feature-pool rows referenced by the mask; held until the prompt has been prefilled.
    pub features: Option<Arc<FeatureLease>>,
    /// Model resources the request uses until it finishes (e.g. a voice slot of the detokenizer).
    pub hold: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
}

impl SamplingParams {
    pub fn from_meta(v: &Value) -> Self {
        let f = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
        Self {
            temperature: f("temperature", 1.0),
            top_k: v.get("top_k").and_then(Value::as_i64).unwrap_or(0) as i32,
            top_p: f("top_p", 1.0),
            min_p: f("min_p", 0.0),
            repetition_penalty: f("repetition_penalty", 1.0),
        }
    }
}

/// A generation request as received over HTTP.
#[derive(Debug, Clone, Default)]
pub struct GenerateInput {
    pub text: String,
    pub audio_path: Option<String>,
    /// Model-specific options (voice, language, speaker, ...); `null` values mean "unset".
    pub model_kwargs: serde_json::Map<String, Value>,
}

impl GenerateInput {
    pub fn kwarg_str(&self, key: &str) -> Option<&str> {
        self.model_kwargs.get(key).and_then(Value::as_str)
    }
}

/// Host-injected text column of decode rows for input-streaming requests: the next queued text
/// token, then `eos` once the text is complete, then `pad`.
#[derive(Debug, Clone, Copy)]
pub struct TextStreamSpec {
    pub column: usize,
    pub pad: i32,
    pub eos: i32,
}

/// What a sampled row is, as the program's graph classified it (`out_kind`; the codes are
/// `TOKEN_AUDIO` / `TOKEN_STOP` / `TOKEN_OTHER` in `vox_serve/rt/prompt.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// An audio frame to be detokenized.
    Audio,
    /// End of generation (not detokenized).
    Stop,
    /// Generated and fed back, but not audio (e.g. interleaved text tokens).
    Other,
}

impl TokenKind {
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => TokenKind::Audio,
            1 => TokenKind::Stop,
            _ => TokenKind::Other,
        }
    }
}

/// Builds the prompt rows of a request (once per request, on the CPU).
pub trait PromptBuilder: Send + Sync {
    fn preprocess(&self, input: &GenerateInput) -> Result<Prompt>;
    /// Tokenizes a streamed text chunk (no special tokens).
    fn tokenize_stream_text(&self, _text: &str) -> Result<Vec<i32>> {
        bail!("input streaming is not supported by this model")
    }
    /// Prompt of an input-streaming request whose only text token so far is `first`.
    fn preprocess_streaming(&self, _input: &GenerateInput, _first: i32) -> Result<Prompt> {
        bail!("input streaming is not supported by this model")
    }
}

/// A model as the scheduler sees it: constants from `meta` plus the prompt builder.
pub struct Model {
    n_codebooks: usize,
    max_tokens: usize,
    sampling: SamplingParams,
    sample_rate: u32,
    decode_mask: Option<Vec<i32>>,
    decode_row: Option<Vec<i32>>,
    preseed: usize,
    text_stream: Option<TextStreamSpec>,
    prompts: Box<dyn PromptBuilder>,
}

impl Model {
    pub fn n_codebooks(&self) -> usize {
        self.n_codebooks
    }
    pub fn preprocess(&self, input: &GenerateInput) -> Result<Prompt> {
        self.prompts.preprocess(input)
    }
    /// Input mask of decode rows (models with input masks only).
    pub fn decode_mask(&self) -> Option<&[i32]> {
        self.decode_mask.as_deref()
    }
    /// Host-written token row of decode rows (columns not fed back on device); zeros if `None`.
    pub fn decode_row(&self) -> Option<&[i32]> {
        self.decode_row.as_deref()
    }
    /// Generation stops once the next position id exceeds this value.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    pub fn default_sampling(&self) -> SamplingParams {
        self.sampling
    }
    /// Output sample rate of the detokenizer (Hz).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    /// Zero frames inserted before the first audio frame so the first chunk is decoded early; their
    /// samples are trimmed from the first chunk's output.
    pub fn audio_preseed_frames(&self) -> usize {
        self.preseed
    }
    /// Input streaming support (text arriving while audio is generated).
    pub fn text_stream(&self) -> Option<TextStreamSpec> {
        self.text_stream
    }
    pub fn tokenize_stream_text(&self, text: &str) -> Result<Vec<i32>> {
        self.prompts.tokenize_stream_text(text)
    }
    pub fn preprocess_streaming(&self, input: &GenerateInput, first: i32) -> Result<Prompt> {
        self.prompts.preprocess_streaming(input, first)
    }
    /// Forces greedy decoding for every request (testing / parity checks).
    pub fn greedy(mut self) -> Self {
        self.sampling.temperature = 0.0;
        self
    }
}

/// The model of an engine spec's `meta`; `prompt_builder` overrides `meta["prompt_builder"]`.
pub fn load(meta: &Value, prompt_builder: Option<&str>) -> Result<Model> {
    let int = |k: &str| meta[k].as_u64().ok_or_else(|| anyhow!("meta.{k} missing")).map(|x| x as usize);
    let ids = |k: &str| meta.get(k).filter(|v| !v.is_null()).map(|_| meta_ids(meta, k));
    let builder = prompt_builder
        .or_else(|| meta["prompt_builder"].as_str())
        .or_else(|| meta["kind"].as_str())
        .ok_or_else(|| anyhow!("meta has neither prompt_builder nor kind"))?;
    let prompts: Box<dyn PromptBuilder> = match builder {
        "template" => Box::new(template::Template::new(meta)?),
        "python" => Box::new(python::Python::new(meta)),
        "chatterbox" => Box::new(chatterbox::Chatterbox::new(meta)?),
        "cosyvoice2" => Box::new(cosyvoice2::CosyVoice2::new(meta)?),
        "cosyvoice3" => Box::new(cosyvoice3::CosyVoice3::new(meta)?),
        "csm" => Box::new(csm::Csm::new(meta)?),
        "glm_voice" => Box::new(glm_voice::GlmVoice::new(meta)?),
        "higgs_tts" => Box::new(higgs_tts::HiggsTts::new(meta)?),
        "qwen3_tts" => Box::new(qwen3_tts::Qwen3Tts::new(meta)?),
        "step_audio_2" => Box::new(step_audio_2::StepAudio2::new(meta)?),
        "voxtral_tts" => Box::new(voxtral_tts::VoxtralTts::new(meta)?),
        "zonos" => Box::new(zonos::Zonos::new(meta)?),
        other => bail!("unknown prompt builder {other:?}"),
    };
    let ts = &meta["text_stream"];
    let text_stream = if ts.is_object() {
        let f = |k: &str| ts[k].as_i64().ok_or_else(|| anyhow!("meta.text_stream.{k} missing"));
        Some(TextStreamSpec { column: f("column")? as usize, pad: f("pad")? as i32, eos: f("eos")? as i32 })
    } else {
        None
    };
    Ok(Model {
        n_codebooks: int("n_codebooks")?,
        max_tokens: int("max_tokens")?,
        sampling: SamplingParams::from_meta(&meta["sampling"]),
        sample_rate: meta["sample_rate"].as_u64().unwrap_or(24000) as u32,
        decode_mask: ids("decode_mask"),
        decode_row: ids("decode_row"),
        preseed: meta["preseed_frames"].as_u64().unwrap_or(0) as usize,
        text_stream,
        prompts,
    })
}

pub(crate) fn meta_ids(v: &Value, key: &str) -> Vec<i32> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_i64).map(|x| x as i32).collect())
        .unwrap_or_default()
}

/// The tokenizer named by `meta[key]` (a tokenizer.json path).
pub(crate) fn tokenizer(meta: &Value, key: &str) -> Result<tokenizers::Tokenizer> {
    let path = meta[key].as_str().ok_or_else(|| anyhow!("meta.{key} missing"))?;
    tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow!("loading {path}: {e}"))
}
