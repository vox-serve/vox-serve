//! HTTP API compatible with `vox_serve/launch.py` (`/generate`, `/generate/stream/...`,
//! `/v1/audio/speech`, `/health`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Request, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use crossbeam_channel::{Sender, TrySendError};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::models::{GenerateInput, Model};
use crate::scheduler::{Event, Submit};
use crate::wav;

const TIMEOUT: Duration = Duration::from_secs(600);
/// Input streaming: text buffered before the prompt is built (as in `scheduler/input_streaming.py`).
const MIN_INITIAL_TEXT_CHARS: usize = 20;
/// Largest request body: multipart uploads and `/v1/audio/speech` JSON with base64 reference audio
/// (axum's default of 2 MiB rejects about 12 s of 44.1 kHz stereo WAV; the Python runtime has no
/// limit). 100 MiB is about 10 minutes of 44.1 kHz stereo 16-bit audio.
const MAX_BODY_BYTES: usize = 100 << 20;
/// A request whose event channel closed without `Done`: its engine's scheduler loop has stopped.
const ENGINE_STOPPED: &str = "Engine stopped before the request finished";

/// One engine (data-parallel replica): its scheduler queue and model logic (whose hooks and
/// feature pool belong to that engine).
pub struct Rank {
    pub tx: Sender<Submit>,
    pub model: Arc<Model>,
    /// Requests the engine holds, published by its scheduler.
    pub load: Arc<AtomicUsize>,
    /// Set when the engine's scheduler thread ends (a failed step, e.g. a CUDA error, or a panic):
    /// the rank takes no more requests and `/health` reports the server unhealthy.
    pub dead: Arc<AtomicBool>,
}

/// Marks a rank dead when dropped. Held by its scheduler thread, so the flag is set however the
/// thread ends.
pub struct MarkDeadOnDrop(pub Arc<AtomicBool>);

impl Drop for MarkDeadOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub struct AppState {
    ranks: Arc<Vec<Rank>>,
    next: Arc<AtomicUsize>,
    upload_dir: std::path::PathBuf,
    sessions: Arc<Mutex<HashMap<String, Session>>>,
}

impl AppState {
    pub fn new(ranks: Vec<Rank>, upload_dir: std::path::PathBuf) -> Self {
        Self { ranks: Arc::new(ranks), next: Default::default(), upload_dir, sessions: Default::default() }
    }

    /// Model logic for engine-independent work (tokenization, sample rate).
    fn model(&self) -> &Arc<Model> {
        &self.ranks[0].model
    }

    /// Least-loaded live replica (requests held by its engine plus those still queued for it); ties
    /// go round-robin. With one replica, or replicas of equal speed, this is round-robin dispatch;
    /// with replicas of unequal capacity (e.g. one LM on a whole GPU, one sharing its GPU with the
    /// codec) the faster one takes more requests. `None` once every engine has stopped.
    fn pick(&self) -> Option<&Rank> {
        let n = self.ranks.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        (0..n)
            .map(|k| &self.ranks[(start + k) % n])
            .filter(|r| !r.dead.load(Ordering::Relaxed))
            .min_by_key(|r| r.load.load(Ordering::Relaxed) + r.tx.len())
    }

    /// Ranks whose engine has stopped.
    fn dead_ranks(&self) -> Vec<usize> {
        (0..self.ranks.len()).filter(|&i| self.ranks[i].dead.load(Ordering::Relaxed)).collect()
    }
}

/// An input-streaming request. Text is buffered until enough has arrived to build the prompt (from
/// its first token); every later token goes through `text_tx` to the request's decode rows.
pub struct Session {
    input: GenerateInput,
    /// Moved into the submission.
    events_tx: Option<UnboundedSender<Event>>,
    /// Taken by `/audio`.
    events_rx: Option<UnboundedReceiver<Event>>,
    /// Dropped at `/end`, which completes the text once the scheduler has drained the channel.
    text_tx: Option<Sender<Vec<i32>>>,
    /// Moved into the submission.
    text_rx: Option<crossbeam_channel::Receiver<Vec<i32>>>,
    submitted: bool,
    ended: bool,
}

/// A prompt to build (outside the session lock: it may call Python hooks) and submit.
struct PendingStream {
    input: GenerateInput,
    first: i32,
    events: UnboundedSender<Event>,
    text_rx: crossbeam_channel::Receiver<Vec<i32>>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .route("/generate/stream/start", post(stream_start))
        .route("/generate/stream/{rid}/text", post(stream_text))
        .route("/generate/stream/{rid}/audio", get(stream_audio))
        .route("/generate/stream/{rid}/end", post(stream_end))
        .route("/v1/audio/speech", post(speech))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state)
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({"detail": msg.into()}))).into_response()
}

fn no_engine() -> Response {
    error(StatusCode::SERVICE_UNAVAILABLE, "No engine available (see the server log)")
}

/// 503 once any engine has stopped: a failed engine (e.g. after a CUDA error) does not come back
/// until the server is restarted. The other ranks keep serving meanwhile.
async fn health(State(state): State<AppState>) -> Response {
    let dead = state.dead_ranks();
    if dead.is_empty() {
        return Json(json!({"status": "healthy"})).into_response();
    }
    (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "unhealthy", "failed_engines": dead}))).into_response()
}

async fn submit(
    state: &AppState,
    input: GenerateInput,
    streaming: bool,
) -> Result<(String, UnboundedReceiver<Event>), Response> {
    // Tokenization is cheap, but some models call back into Python (audio encoders): keep it off
    // the async workers.
    let Some(rank) = state.pick() else { return Err(no_engine()) };
    let model = rank.model.clone();
    let prompt = tokio::task::spawn_blocking(move || model.preprocess(&input))
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| error(StatusCode::BAD_REQUEST, e.to_string()))?;
    let id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = unbounded_channel();
    let s = Submit {
        id: id.clone(),
        prompt,
        sampling: rank.model.default_sampling(),
        seed: rand::random::<u32>() as u64,
        streaming,
        tx,
        text_rx: None,
    };
    match rank.tx.try_send(s) {
        Ok(()) => Ok((id, rx)),
        Err(TrySendError::Full(_)) => Err(error(StatusCode::TOO_MANY_REQUESTS, "Server busy; please retry shortly")),
        Err(TrySendError::Disconnected(_)) => Err(error(StatusCode::SERVICE_UNAVAILABLE, "Server not ready")),
    }
}

/// Audio chunks until completion. An error, the request timeout or the channel closing without
/// `Done` (the engine stopped) ends the stream with an `Err`, which aborts the response (the chunked
/// body is never terminated) so that the client cannot mistake truncated audio for a complete one.
fn pcm_stream(rx: UnboundedReceiver<Event>) -> impl futures::Stream<Item = Result<Bytes, String>> {
    let deadline = Instant::now() + TIMEOUT;
    futures::stream::unfold(Some(rx), move |rx| async move {
        let mut rx = rx?;
        let left = deadline.saturating_duration_since(Instant::now());
        let err = match tokio::time::timeout(left, rx.recv()).await {
            Ok(Some(Event::Audio(b))) => return Some((Ok(b), Some(rx))),
            Ok(Some(Event::Done(_))) => return None,
            Ok(Some(Event::Error(e))) => e,
            Ok(None) => ENGINE_STOPPED.into(),
            Err(_) => "Generation timed out".into(),
        };
        Some((Err(err), None))
    })
}

fn streaming_wav(id: &str, sample_rate: u32, rx: UnboundedReceiver<Event>) -> Response {
    use futures::StreamExt;
    let head = futures::stream::once(async move { Ok(Bytes::copy_from_slice(&wav::header(sample_rate, 0))) });
    Response::builder()
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::CONTENT_DISPOSITION, format!("attachment; filename=stream_{}.wav", &id[..8]))
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(head.chain(pcm_stream(rx))))
        .unwrap()
}

async fn collect(mut rx: UnboundedReceiver<Event>) -> Result<Vec<u8>, Response> {
    let deadline = Instant::now() + TIMEOUT;
    let mut pcm = Vec::new();
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, rx.recv()).await {
            Ok(Some(Event::Audio(b))) => pcm.extend_from_slice(&b),
            Ok(Some(Event::Done(_))) => return Ok(pcm),
            Ok(Some(Event::Error(e))) => return Err(error(StatusCode::INTERNAL_SERVER_ERROR, e)),
            Ok(None) => return Err(error(StatusCode::INTERNAL_SERVER_ERROR, ENGINE_STOPPED)),
            Err(_) => return Err(error(StatusCode::INTERNAL_SERVER_ERROR, "Generation timed out")),
        }
    }
}

/// Form fields shared by `/generate` and `/generate/stream/start`.
#[derive(Default)]
struct Form {
    fields: HashMap<String, String>,
    audio_path: Option<String>,
    kwargs: Map<String, Value>,
}

fn truthy(v: &str) -> bool {
    matches!(v.to_ascii_lowercase().as_str(), "true" | "1" | "yes" | "on")
}

impl Form {
    fn insert(&mut self, name: String, value: String) {
        match name.as_str() {
            "language" | "speaker" | "ref_text" | "instruct" | "voice" => {
                self.kwargs.insert(name, Value::String(value));
            }
            "x_vector_only_mode" => {
                self.kwargs.insert(name, Value::Bool(truthy(&value)));
            }
            _ => {
                self.fields.insert(name, value);
            }
        }
    }
}

/// Parses a multipart or URL-encoded form (FastAPI `Form(...)` accepts both).
async fn parse_form(state: &AppState, req: Request) -> Result<Form, Response> {
    let bad = |e: &dyn std::fmt::Display| error(StatusCode::BAD_REQUEST, e.to_string());
    let multipart = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("multipart/form-data"));
    let mut form = Form::default();
    if !multipart {
        let body = axum::body::to_bytes(req.into_body(), 1 << 20).await.map_err(|e| bad(&e))?;
        for (k, v) in form_urlencoded::parse(&body) {
            form.insert(k.into_owned(), v.into_owned());
        }
        return Ok(form);
    }
    let mut mp = Multipart::from_request(req, &()).await.map_err(|e| bad(&e))?;
    while let Some(field) = mp.next_field().await.map_err(|e| bad(&e))? {
        let name = field.name().unwrap_or_default().to_string();
        if name == "audio" {
            let fname = field.file_name().unwrap_or("upload").replace('/', "_");
            let data = field.bytes().await.map_err(|e| bad(&e))?;
            if data.is_empty() {
                continue;
            }
            form.audio_path = Some(save_upload(state, &fname, &data).await?);
            continue;
        }
        let value = field.text().await.map_err(|e| bad(&e))?;
        form.insert(name, value);
    }
    Ok(form)
}

/// Writes uploaded audio to the upload directory (removed after the request timeout).
async fn save_upload(state: &AppState, name: &str, data: &[u8]) -> Result<String, Response> {
    let path = state.upload_dir.join(format!("{}_{name}", uuid::Uuid::new_v4()));
    if let Err(e) = tokio::fs::write(&path, data).await {
        return Err(error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
    }
    let p = path.clone();
    tokio::spawn(async move {
        tokio::time::sleep(TIMEOUT).await;
        let _ = tokio::fs::remove_file(p).await;
    });
    Ok(path.to_string_lossy().to_string())
}

async fn generate(State(state): State<AppState>, req: Request) -> Response {
    let mut form = match parse_form(&state, req).await {
        Ok(f) => f,
        Err(r) => return r,
    };
    let text = form.fields.remove("text");
    let streaming = form.fields.get("streaming").is_none_or(|v| truthy(v));
    let (audio_path, kwargs) = (form.audio_path, form.kwargs);
    let Some(text) = text else {
        return error(StatusCode::UNPROCESSABLE_ENTITY, "field 'text' is required");
    };
    let input = GenerateInput { text, audio_path, model_kwargs: kwargs };
    let (id, rx) = match submit(&state, input, streaming).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    if streaming {
        return streaming_wav(&id, state.model().sample_rate(), rx);
    }
    match collect(rx).await {
        Ok(pcm) if pcm.is_empty() => error(StatusCode::INTERNAL_SERVER_ERROR, "No audio generated"),
        Ok(pcm) => {
            let mut body = wav::header(state.model().sample_rate(), pcm.len() as u32).to_vec();
            body.extend_from_slice(&pcm);
            Response::builder()
                .header(header::CONTENT_TYPE, "audio/wav")
                .header(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{id}.wav\""))
                .body(Body::from(body))
                .unwrap()
        }
        Err(r) => r,
    }
}

#[derive(Deserialize)]
struct SpeechRequest {
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    extra_params: Option<Map<String, Value>>,
    /// Reference audio for voice cloning, as a base64 data URL (`data:audio/wav;base64,...`).
    #[serde(default)]
    ref_audio: Option<String>,
    /// Transcript of the reference audio.
    #[serde(default)]
    ref_text: Option<String>,
}

async fn speech(State(state): State<AppState>, Json(req): Json<SpeechRequest>) -> Response {
    let mut kwargs = Map::new();
    kwargs.insert("voice".into(), req.voice.map(Value::String).unwrap_or(Value::Null));
    kwargs.insert("language".into(), req.language.map(Value::String).unwrap_or(Value::Null));
    let cfg = req.extra_params.as_ref().and_then(|p| p.get("cfg_alpha")).cloned().unwrap_or(Value::Null);
    kwargs.insert("cfg_alpha".into(), cfg);
    if let Some(t) = req.ref_text {
        kwargs.insert("ref_text".into(), Value::String(t));
    }
    let audio_path = match req.ref_audio.as_deref() {
        Some(url) => {
            let Some(b64) = url.strip_prefix("data:").and_then(|u| u.split_once(";base64,")).map(|(_, d)| d) else {
                return error(StatusCode::BAD_REQUEST, "ref_audio must be a base64 data URL");
            };
            let data = match base64::decode(b64.trim()) {
                Ok(d) => d,
                Err(e) => return error(StatusCode::BAD_REQUEST, format!("ref_audio: {e}")),
            };
            match save_upload(&state, "ref_audio", &data).await {
                Ok(p) => Some(p),
                Err(r) => return r,
            }
        }
        None => None,
    };
    let input = GenerateInput { text: req.input, audio_path, model_kwargs: kwargs };
    let (id, rx) = match submit(&state, input, true).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    Response::builder()
        .header("x-request-id", id)
        .header("x-audio-sample-rate", state.model().sample_rate())
        .header(header::CONTENT_TYPE, "audio/pcm")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(pcm_stream(rx)))
        .unwrap()
}

// ---- input streaming --------------------------------------------------------------------------

async fn stream_start(State(state): State<AppState>, req: Request) -> Response {
    if state.model().text_stream().is_none() {
        return error(StatusCode::BAD_REQUEST, "input streaming is not supported by this model");
    }
    let form = match parse_form(&state, req).await {
        Ok(f) => f,
        Err(r) => return r,
    };
    let id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = unbounded_channel();
    let (text_tx, text_rx) = crossbeam_channel::unbounded();
    let session = Session {
        input: GenerateInput { text: String::new(), audio_path: form.audio_path, model_kwargs: form.kwargs },
        events_tx: Some(tx),
        events_rx: Some(rx),
        text_tx: Some(text_tx),
        text_rx: Some(text_rx),
        submitted: false,
        ended: false,
    };
    state.sessions.lock().unwrap().insert(id.clone(), session);
    // Abandoned sessions (and the requests they feed) are dropped after the request timeout.
    let (sessions, rid) = (state.sessions.clone(), id.clone());
    tokio::spawn(async move {
        tokio::time::sleep(TIMEOUT).await;
        sessions.lock().unwrap().remove(&rid);
    });
    Json(json!({"request_id": id})).into_response()
}

/// Called with the session locked once the prompt can be built: tokenizes the buffered text, queues
/// all but its first token, and hands out what the prompt needs.
#[allow(clippy::result_large_err)] // the error is the HTTP response itself
fn begin_submission(state: &AppState, s: &mut Session) -> Result<Option<PendingStream>, Response> {
    let tokens = (state.model().tokenize_stream_text(&s.input.text))
        .map_err(|e| error(StatusCode::BAD_REQUEST, e.to_string()))?;
    s.submitted = true;
    let events = s.events_tx.take().expect("unsubmitted session has an event sender");
    let Some((&first, rest)) = tokens.split_first() else {
        let _ = events.send(Event::Error("no text provided".into()));
        return Ok(None);
    };
    if let Some(tx) = &s.text_tx {
        let _ = tx.send(rest.to_vec());
    }
    let text_rx = s.text_rx.take().expect("unsubmitted session has a text receiver");
    Ok(Some(PendingStream { input: s.input.clone(), first, events, text_rx }))
}

async fn launch_stream(state: &AppState, p: PendingStream) -> Result<(), Response> {
    let Some(rank) = state.pick() else { return Err(no_engine()) };
    let model = rank.model.clone();
    let (input, first) = (p.input, p.first);
    let prompt = tokio::task::spawn_blocking(move || model.preprocess_streaming(&input, first))
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let prompt = match prompt {
        Ok(v) => v,
        Err(e) => {
            let _ = p.events.send(Event::Error(e.to_string()));
            return Err(error(StatusCode::BAD_REQUEST, e.to_string()));
        }
    };
    let sub = Submit {
        id: uuid::Uuid::new_v4().to_string(),
        prompt,
        sampling: rank.model.default_sampling(),
        seed: rand::random::<u32>() as u64,
        streaming: true,
        tx: p.events,
        text_rx: Some(p.text_rx),
    };
    match rank.tx.try_send(sub) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(_)) => Err(error(StatusCode::TOO_MANY_REQUESTS, "Server busy; please retry shortly")),
        Err(TrySendError::Disconnected(_)) => Err(error(StatusCode::SERVICE_UNAVAILABLE, "Server not ready")),
    }
}

async fn stream_text(State(state): State<AppState>, Path(rid): Path<String>, req: Request) -> Response {
    let text = match parse_form(&state, req).await {
        Ok(mut f) => f.fields.remove("text"),
        Err(r) => return r,
    };
    let Some(text) = text else {
        return error(StatusCode::UNPROCESSABLE_ENTITY, "field 'text' is required");
    };
    let pending = {
        let mut sessions = state.sessions.lock().unwrap();
        let Some(s) = sessions.get_mut(&rid) else {
            return error(StatusCode::NOT_FOUND, format!("Request {rid} not found"));
        };
        if s.ended {
            return error(StatusCode::BAD_REQUEST, format!("Request {rid} already completed"));
        }
        if text.is_empty() {
            None
        } else if !s.submitted {
            s.input.text.push_str(&text);
            if s.input.text.chars().count() < MIN_INITIAL_TEXT_CHARS {
                None
            } else {
                match begin_submission(&state, s) {
                    Ok(p) => p,
                    Err(r) => return r,
                }
            }
        } else {
            match state.model().tokenize_stream_text(&text) {
                Ok(t) => {
                    if let Some(tx) = &s.text_tx {
                        let _ = tx.send(t);
                    }
                    None
                }
                Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
            }
        }
    };
    if let Some(p) = pending {
        if let Err(r) = launch_stream(&state, p).await {
            return r;
        }
    }
    Json(json!({"status": "accepted", "request_id": rid})).into_response()
}

async fn stream_end(State(state): State<AppState>, Path(rid): Path<String>) -> Response {
    let pending = {
        let mut sessions = state.sessions.lock().unwrap();
        let Some(s) = sessions.get_mut(&rid) else {
            return error(StatusCode::NOT_FOUND, format!("Request {rid} not found"));
        };
        if s.ended {
            None
        } else {
            s.ended = true;
            let p = if s.submitted {
                None
            } else {
                match begin_submission(&state, s) {
                    Ok(p) => p,
                    Err(r) => return r,
                }
            };
            s.text_tx = None;
            p
        }
    };
    if let Some(p) = pending {
        if let Err(r) = launch_stream(&state, p).await {
            return r;
        }
    }
    Json(json!({"status": "completed", "request_id": rid})).into_response()
}

async fn stream_audio(State(state): State<AppState>, Path(rid): Path<String>) -> Response {
    let rx = {
        let mut sessions = state.sessions.lock().unwrap();
        let Some(s) = sessions.get_mut(&rid) else {
            return error(StatusCode::NOT_FOUND, format!("Request {rid} not found"));
        };
        let Some(rx) = s.events_rx.take() else {
            return error(StatusCode::BAD_REQUEST, format!("Audio for request {rid} is already being streamed"));
        };
        rx
    };
    streaming_wav(&rid, state.model().sample_rate(), rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tower::ServiceExt;

    fn rank() -> (Rank, crossbeam_channel::Receiver<Submit>) {
        let meta = json!({"prompt_builder": "python", "n_codebooks": 1, "max_tokens": 16});
        let model = Arc::new(crate::models::load(&meta, None).unwrap());
        let (tx, rx) = crossbeam_channel::bounded(4);
        (Rank { tx, model, load: Default::default(), dead: Default::default() }, rx)
    }

    fn audio() -> Bytes {
        Bytes::from_static(b"pcm")
    }

    #[tokio::test]
    async fn collect_fails_without_done() {
        let (tx, rx) = unbounded_channel();
        tx.send(Event::Audio(audio())).unwrap();
        tx.send(Event::Done("stop".into())).unwrap();
        assert_eq!(collect(rx).await.unwrap(), b"pcm");

        // The engine stopped mid-request: the scheduler dropped the sender without `Done`.
        let (tx, rx) = unbounded_channel();
        tx.send(Event::Audio(audio())).unwrap();
        drop(tx);
        assert_eq!(collect(rx).await.unwrap_err().status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn stream_aborts_without_done() {
        let (tx, rx) = unbounded_channel();
        tx.send(Event::Audio(audio())).unwrap();
        tx.send(Event::Done("stop".into())).unwrap();
        assert_eq!(pcm_stream(rx).collect::<Vec<_>>().await, [Ok(audio())]);

        let (tx, rx) = unbounded_channel();
        tx.send(Event::Audio(audio())).unwrap();
        drop(tx);
        assert_eq!(pcm_stream(rx).collect::<Vec<_>>().await, [Ok(audio()), Err(ENGINE_STOPPED.to_string())]);

        let (tx, rx) = unbounded_channel();
        tx.send(Event::Error("bad prompt".into())).unwrap();
        assert_eq!(pcm_stream(rx).collect::<Vec<_>>().await, [Err("bad prompt".to_string())]);
    }

    #[test]
    fn engine_panic_marks_rank_dead() {
        let dead = Arc::new(AtomicBool::new(false));
        let guard = MarkDeadOnDrop(dead.clone());
        let t = std::thread::spawn(move || {
            let _guard = guard;
            panic!("engine step failed");
        });
        assert!(t.join().is_err());
        assert!(dead.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn dead_ranks_are_unhealthy_and_skipped() {
        let (a, _rx_a) = rank();
        let (b, _rx_b) = rank();
        let state = AppState::new(vec![a, b], std::env::temp_dir());
        assert_eq!(health(State(state.clone())).await.status(), StatusCode::OK);

        state.ranks[0].dead.store(true, Ordering::Relaxed);
        assert_eq!(health(State(state.clone())).await.status(), StatusCode::SERVICE_UNAVAILABLE);
        for _ in 0..4 {
            assert!(std::ptr::eq(state.pick().unwrap(), &state.ranks[1]));
        }

        state.ranks[1].dead.store(true, Ordering::Relaxed);
        assert!(state.pick().is_none());
        let r = submit(&state, GenerateInput::default(), false).await.unwrap_err();
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    fn app(upload_dir: &std::path::Path) -> (Router, crossbeam_channel::Receiver<Submit>) {
        let (rank, rx) = rank();
        (router(AppState::new(vec![rank], upload_dir.to_path_buf())), rx)
    }

    async fn status(app: Router, req: axum::http::Request<Body>) -> StatusCode {
        app.oneshot(req).await.unwrap().status()
    }

    /// Bodies past axum's 2 MiB default reach the handlers (which reject these for other reasons).
    #[tokio::test]
    async fn accepts_bodies_over_2_mib() {
        let dir = std::env::temp_dir().join(format!("vox_rt_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let (app, _rx) = app(&dir);
        let big = 3 << 20;

        // Invalid base64 past the limit: 400 from the handler instead of 413 from `Json`.
        let body = json!({"input": "hi", "ref_audio": format!("data:audio/wav;base64,{}", "!".repeat(big))});
        let req = axum::http::Request::post("/v1/audio/speech")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        assert_eq!(status(app.clone(), req).await, StatusCode::BAD_REQUEST);

        // An uploaded clip past the limit is saved; the missing `text` field gets 422 instead of the
        // multipart error's 400.
        let mut body = b"--b\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"ref.wav\"\r\n\r\n".to_vec();
        body.resize(body.len() + big, 0);
        body.extend_from_slice(b"\r\n--b--\r\n");
        let req = axum::http::Request::post("/generate")
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=b")
            .body(Body::from(body))
            .unwrap();
        assert_eq!(status(app, req).await, StatusCode::UNPROCESSABLE_ENTITY);
        let saved: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().metadata().unwrap().len()).collect();
        assert_eq!(saved, [big as u64]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
