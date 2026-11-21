//! WebSocket speech streaming at `/v1/audio/speech/stream`, with the protocol of sglang-omni's
//! (<https://sgl-project.github.io/sglang-omni/basic_usage/tts.html#websocket-speech-streaming>)
//! served by vox-serve's input streaming.
//!
//! Client messages (JSON text frames):
//! - `{"type": "session.config", "session": {...}}`, first: `voice`, `language`, `response_format`
//!   (`"pcm"`, the default, or `"wav"`), `stream_audio` (default `false`), `split_granularity`
//!   (`"clause"`, the default, or `"sentence"`), and vox-serve's `ref_audio` (a base64 data URL),
//!   `ref_text`, `speaker`, `instruct`. `model` is ignored.
//! - `{"type": "input.text", "text": "..."}`, any number of times.
//! - `{"type": "input.commit"}`: ends the current segment's text (it is spoken in full, including text
//!   after its last boundary); the socket stays open for the next segment.
//! - `{"type": "input.done"}`: commits, and closes the session once all audio is sent.
//!
//! Server messages: `session.configured` (with `sample_rate`); per segment `audio.start`, the audio as
//! binary frames (16-bit mono PCM, or one WAV file with `"wav"`: with `stream_audio` as it is generated,
//! otherwise once per segment), `audio.done` and `input.committed` (`segment_index`, `segment_sentences`,
//! `total_sentences`); `session.done` after `input.done`; `error` (`message`).
//!
//! Each segment (the text between commits) is one input-streaming request, so a session can run for any
//! number of segments without filling the model's context. Segments are spoken in order; the next one
//! starts generating as soon as its text arrives. With `split_granularity: "sentence"`, text reaches the
//! model a sentence at a time (turn-based models such as Higgs TTS then read whole sentences per turn).

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use futures::stream::SplitSink;
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::time::Instant;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::models::GenerateInput;
use crate::scheduler::{sentence_cut, Event};
use crate::server::{
    api_error, decode_data_url, feed, finish, launch, streams_input, ApiError, AppState, Pending, Session,
    ENGINE_STOPPED, TIMEOUT,
};
use crate::wav;

/// Run-on limits of the sentence cut for models without turn-stream settings.
const MIN_WORDS: usize = 3;
const MAX_WORDS: usize = 20;

pub(crate) async fn speech_stream(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| run(state, socket))
}

#[derive(Deserialize, Default)]
struct SessionConfig {
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    speaker: Option<String>,
    #[serde(default)]
    instruct: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    stream_audio: bool,
    #[serde(default)]
    split_granularity: Option<String>,
    #[serde(default)]
    ref_audio: Option<String>,
    #[serde(default)]
    ref_text: Option<String>,
}

/// How segments' audio is sent.
#[derive(Clone, Copy)]
struct Output {
    stream_audio: bool,
    wav: bool,
    sample_rate: u32,
}

/// Work for the player, which sends each segment's audio and events in order.
enum Job {
    Segment {
        index: usize,
        events: UnboundedReceiver<Event>,
        sentences: oneshot::Receiver<usize>,
    },
    /// A commit without text.
    Empty {
        index: usize,
    },
    Done,
}

/// The segment whose text is arriving.
struct Segment {
    session: Session,
    /// All of its text (for the sentence count).
    text: String,
    /// Text not yet given to the model (sentence granularity).
    held: String,
    sentences: oneshot::Sender<usize>,
}

/// Reference audio written for the session, removed when the connection ends.
struct TempFile(std::path::PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

struct Configured {
    input: GenerateInput,
    sentences: bool,
    jobs: UnboundedSender<Job>,
    player: JoinHandle<()>,
    _ref_audio: Option<TempFile>,
}

enum Flow {
    Continue,
    /// `input.done`: wait for the player, then close.
    Done,
    /// A fatal error (already reported): close now.
    Close,
}

struct Conn {
    state: AppState,
    out: UnboundedSender<Message>,
    cfg: Option<Configured>,
    seg: Option<Segment>,
    index: usize,
}

fn text_frame(v: Value) -> Message {
    Message::Text(v.to_string().into())
}

impl Conn {
    fn send(&self, v: Value) {
        let _ = self.out.send(text_frame(v));
    }

    fn error(&self, msg: impl Into<String>) {
        self.send(json!({"type": "error", "message": msg.into()}));
    }

    async fn handle(&mut self, msg: Value) -> Flow {
        let kind = msg["type"].as_str().unwrap_or_default().to_string();
        if self.cfg.is_none() && kind != "session.config" {
            self.error("the first message must be session.config");
            return Flow::Close;
        }
        match kind.as_str() {
            "session.config" => self.configure(&msg["session"]).await,
            "input.text" => {
                let Some(text) = msg["text"].as_str() else {
                    self.error("input.text needs a string field 'text'");
                    return Flow::Continue;
                };
                self.text(text.to_string()).await;
                Flow::Continue
            }
            "input.commit" => {
                self.commit().await;
                Flow::Continue
            }
            "input.done" => {
                if self.seg.is_some() {
                    self.commit().await;
                }
                let _ = self.cfg.as_ref().expect("configured").jobs.send(Job::Done);
                Flow::Done
            }
            other => {
                self.error(format!("unknown message type {other:?}"));
                Flow::Continue
            }
        }
    }

    async fn configure(&mut self, session: &Value) -> Flow {
        if self.cfg.is_some() {
            self.error("the session is already configured");
            return Flow::Continue;
        }
        let fail = |conn: &Conn, msg: String| {
            conn.error(msg);
            Flow::Close
        };
        if !streams_input(self.state.model()) {
            return fail(self, "input streaming is not supported by this model".into());
        }
        let c: SessionConfig = if session.is_null() {
            SessionConfig::default()
        } else {
            match serde_json::from_value(session.clone()) {
                Ok(c) => c,
                Err(e) => return fail(self, format!("invalid session.config: {e}")),
            }
        };
        let wav = match c.response_format.as_deref().unwrap_or("pcm") {
            "pcm" => false,
            "wav" => true,
            f => return fail(self, format!("response_format {f:?} is not supported (pcm or wav)")),
        };
        if wav && c.stream_audio {
            return fail(self, "stream_audio needs response_format \"pcm\"".into());
        }
        let sentences = match c.split_granularity.as_deref().unwrap_or("clause") {
            "clause" => false,
            "sentence" => true,
            g => return fail(self, format!("split_granularity {g:?} is not supported (sentence or clause)")),
        };
        let ref_audio = match c.ref_audio.as_deref() {
            Some(url) => match self.write_ref_audio(url).await {
                Ok(f) => Some(f),
                Err(e) => return fail(self, e.msg),
            },
            None => None,
        };
        let mut kwargs = Map::new();
        for (k, v) in [
            ("voice", &c.voice),
            ("language", &c.language),
            ("speaker", &c.speaker),
            ("instruct", &c.instruct),
            ("ref_text", &c.ref_text),
        ] {
            if let Some(v) = v {
                kwargs.insert(k.into(), Value::String(v.clone()));
            }
        }
        let input = GenerateInput {
            text: String::new(),
            audio_path: ref_audio.as_ref().map(|f| f.0.to_string_lossy().to_string()),
            model_kwargs: kwargs,
        };
        let output = Output { stream_audio: c.stream_audio, wav, sample_rate: self.state.model().sample_rate() };
        let (jobs, jobs_rx) = unbounded_channel();
        let player = tokio::spawn(play(jobs_rx, self.out.clone(), output));
        self.cfg = Some(Configured { input, sentences, jobs, player, _ref_audio: ref_audio });
        self.send(json!({
            "type": "session.configured",
            "session": {
                "voice": c.voice,
                "language": c.language,
                "response_format": if wav { "wav" } else { "pcm" },
                "stream_audio": c.stream_audio,
                "split_granularity": if sentences { "sentence" } else { "clause" },
                "sample_rate": output.sample_rate,
            },
        }));
        Flow::Continue
    }

    async fn write_ref_audio(&self, url: &str) -> Result<TempFile, ApiError> {
        let data = decode_data_url(url)?;
        let path = self.state.upload_dir().join(format!("{}_ws_ref_audio", uuid::Uuid::new_v4()));
        tokio::fs::write(&path, data).await.map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(TempFile(path))
    }

    async fn text(&mut self, text: String) {
        let cfg = self.cfg.as_ref().expect("configured");
        if self.seg.is_none() {
            if text.trim().is_empty() {
                return; // leading whitespace of a segment
            }
            let mut session = Session::new(cfg.input.clone());
            let events = session.take_events().expect("new session has its events");
            let (tx, rx) = oneshot::channel();
            let _ = cfg.jobs.send(Job::Segment { index: self.index, events, sentences: rx });
            self.seg = Some(Segment { session, text: String::new(), held: String::new(), sentences: tx });
        }
        let sentences = cfg.sentences;
        let (min_words, max_words) =
            self.state.model().turn_stream().map_or((MIN_WORDS, MAX_WORDS), |t| (t.min_words, t.max_words));
        let seg = self.seg.as_mut().expect("segment");
        seg.text.push_str(&text);
        let ready = if sentences {
            seg.held.push_str(&text);
            match sentence_cut(&seg.held, min_words, max_words) {
                Some(cut) => seg.held.drain(..cut).collect(),
                None => return,
            }
        } else {
            text
        };
        let r = feed(&self.state, &mut seg.session, ready);
        self.run(r).await;
    }

    /// Ends the current segment's text (or reports an empty segment).
    async fn commit(&mut self) {
        let index = self.index;
        self.index += 1;
        let Some(mut seg) = self.seg.take() else {
            let _ = self.cfg.as_ref().expect("configured").jobs.send(Job::Empty { index });
            return;
        };
        let held = std::mem::take(&mut seg.held);
        let r = feed(&self.state, &mut seg.session, held);
        self.run(r).await;
        let r = finish(&self.state, &mut seg.session);
        self.run(r).await;
        let _ = seg.sentences.send(count_sentences(&seg.text));
    }

    /// Submits a request that is ready. Launch failures reach the client through the segment's events.
    async fn run(&self, r: Result<Option<Pending>, ApiError>) {
        match r {
            Ok(Some(p)) => {
                let _ = launch(&self.state, p).await;
            }
            Ok(None) => {}
            Err(e) => self.error(e.msg),
        }
    }
}

/// Sends each segment's audio and events in order.
async fn play(mut jobs: UnboundedReceiver<Job>, out: UnboundedSender<Message>, o: Output) {
    let send = |m: Message| out.send(m).is_ok();
    let mut total = 0;
    while let Some(job) = jobs.recv().await {
        let ok = match job {
            Job::Segment { index, mut events, sentences } => {
                let format = if o.wav { "wav" } else { "pcm" };
                if !send(text_frame(json!({
                    "type": "audio.start", "segment_index": index, "sample_rate": o.sample_rate, "format": format,
                }))) {
                    return;
                }
                let mut pcm = Vec::new();
                let deadline = Instant::now() + TIMEOUT;
                let error = loop {
                    let left = deadline.saturating_duration_since(Instant::now());
                    match tokio::time::timeout(left, events.recv()).await {
                        Ok(Some(Event::Audio(b))) => {
                            if o.stream_audio {
                                if !send(Message::Binary(b)) {
                                    return;
                                }
                            } else {
                                pcm.extend_from_slice(&b);
                            }
                        }
                        Ok(Some(Event::Done(_))) => break None,
                        Ok(Some(Event::Error(e))) => break Some(e),
                        Ok(None) => break Some(ENGINE_STOPPED.to_string()),
                        Err(_) => break Some("Generation timed out".to_string()),
                    }
                };
                drop(events); // cancels the request if it is still running (an error or timeout)
                if let Some(e) = error {
                    send(text_frame(json!({"type": "error", "message": e, "segment_index": index})));
                } else if !o.stream_audio && !pcm.is_empty() {
                    let body = if o.wav {
                        let mut b = wav::header(o.sample_rate, pcm.len() as u32).to_vec();
                        b.extend_from_slice(&pcm);
                        b
                    } else {
                        pcm
                    };
                    send(Message::Binary(body.into()));
                }
                let n = sentences.await.unwrap_or(0);
                total += n;
                send(text_frame(json!({"type": "audio.done", "segment_index": index})))
                    && send(text_frame(json!({
                        "type": "input.committed",
                        "segment_index": index,
                        "segment_sentences": n,
                        "total_sentences": total,
                    })))
            }
            Job::Empty { index } => send(text_frame(json!({
                "type": "input.committed", "segment_index": index, "segment_sentences": 0, "total_sentences": total,
            }))),
            Job::Done => {
                send(text_frame(json!({"type": "session.done", "total_sentences": total})));
                return;
            }
        };
        if !ok {
            return;
        }
    }
}

async fn write_all(mut sink: SplitSink<WebSocket, Message>, mut rx: UnboundedReceiver<Message>) {
    while let Some(m) = rx.recv().await {
        if sink.send(m).await.is_err() {
            return;
        }
    }
    let _ = sink.send(Message::Close(None)).await;
    let _ = sink.close().await;
}

async fn run(state: AppState, socket: WebSocket) {
    let (sink, mut incoming) = socket.split();
    let (out, out_rx) = unbounded_channel();
    let writer = tokio::spawn(write_all(sink, out_rx));
    let mut conn = Conn { state, out, cfg: None, seg: None, index: 0 };
    let mut done = false;
    while let Some(msg) = incoming.next().await {
        let text = match msg {
            Ok(Message::Text(t)) => t,
            Ok(Message::Binary(_)) => {
                conn.error("binary messages are not supported");
                continue;
            }
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(_) => continue, // ping / pong
        };
        let msg: Value = match serde_json::from_str(text.as_str()) {
            Ok(v) => v,
            Err(e) => {
                conn.error(format!("malformed JSON: {e}"));
                if conn.cfg.is_none() {
                    break;
                }
                continue;
            }
        };
        match conn.handle(msg).await {
            Flow::Continue => {}
            Flow::Done => {
                done = true;
                break;
            }
            Flow::Close => break,
        }
    }
    if let Some(cfg) = conn.cfg.take() {
        let mut player = cfg.player;
        if done {
            // Send the remaining audio unless the client goes away first.
            let gone = async {
                while let Some(Ok(m)) = incoming.next().await {
                    if matches!(m, Message::Close(_)) {
                        break;
                    }
                }
            };
            tokio::select! {
                _ = &mut player => {}
                _ = gone => player.abort(),
            }
        } else {
            // The client left (or a fatal error): dropping the queued segments cancels their requests.
            player.abort();
        }
        drop(cfg.jobs);
    }
    drop(conn);
    let _ = writer.await;
}

/// Sentences in a segment's text, for `input.committed` (text after the last sentence end counts as one).
fn count_sentences(text: &str) -> usize {
    let t = text.trim();
    if t.is_empty() {
        return 0;
    }
    let chars: Vec<char> = t.chars().collect();
    let mut n = 0;
    for (i, &c) in chars.iter().enumerate() {
        let next = chars.get(i + 1).copied();
        let ends = match c {
            '.' | '!' | '?' => next.is_none_or(char::is_whitespace),
            '。' | '！' | '？' => !matches!(next, Some('。' | '！' | '？')),
            _ => false,
        };
        n += ends as usize;
    }
    if !matches!(chars[chars.len() - 1], '.' | '!' | '?' | '。' | '！' | '？') {
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::count_sentences;
    use crate::models;
    use crate::scheduler::{Event, Submit};
    use crate::server::{router, AppState, Rank};
    use bytes::Bytes;
    use futures::{SinkExt, StreamExt};
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tokio_tungstenite::tungstenite::Message as Ws;

    /// A server with a turn-streaming model whose engine "speaks" each request as `P` plus the text of
    /// its later turns (the first turn is the prompt).
    async fn server() -> String {
        let dir = std::env::temp_dir().join(format!("vox_ws_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let tok = dir.join("tokenizer.json");
        let model = json!({"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"});
        let t = json!({"version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null, "model": model});
        std::fs::write(&tok, t.to_string()).unwrap();
        let meta = json!({"prompt_builder": "template", "n_codebooks": 1, "max_tokens": 64,
            "prompt_template": {"tokenizer_json": tok.to_string_lossy()},
            "turn_stream": {"eoc": 0, "min_eoc_step": 0, "tail_rows": 0}});
        let model = Arc::new(models::load(&meta, None).unwrap());
        let (tx, rx) = crossbeam_channel::bounded::<Submit>(16);
        std::thread::spawn(move || {
            for sub in rx {
                std::thread::spawn(move || {
                    let mut text = String::from("P");
                    for t in sub.turn_rx.expect("turn stream") {
                        text.push_str(&t);
                    }
                    let _ = sub.tx.send(Event::Audio(Bytes::from(text)));
                    let _ = sub.tx.send(Event::Done("stop".into()));
                });
            }
        });
        let rank = Rank { tx, model, load: Default::default(), dead: Default::default() };
        let app = router(AppState::new(vec![rank], dir));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("ws://{addr}/v1/audio/speech/stream")
    }

    /// Every server message until the socket closes: JSON events, and binary frames as `{"binary": ...}`.
    async fn session(url: &str, msgs: &[Value]) -> Vec<Value> {
        let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        for m in msgs {
            ws.send(Ws::Text(m.to_string().into())).await.unwrap();
        }
        let mut got = Vec::new();
        while let Some(Ok(m)) = ws.next().await {
            match m {
                Ws::Text(t) => got.push(serde_json::from_str(t.as_str()).unwrap()),
                Ws::Binary(b) => got.push(json!({"binary": String::from_utf8_lossy(&b)})),
                Ws::Close(_) => break,
                _ => {}
            }
        }
        got
    }

    fn text(t: &str) -> Value {
        json!({"type": "input.text", "text": t})
    }

    #[tokio::test]
    async fn segments_in_order() {
        let url = server().await;
        let mut msgs = vec![json!({"type": "session.config", "session": {"response_format": "pcm"}})];
        // Segment 0: the first turn is cut at the sentence end; the rest is a later turn.
        for w in "Hello there, my friend. How are you".split(' ') {
            msgs.push(text(&format!("{w} ")));
        }
        msgs.push(json!({"type": "input.commit"}));
        msgs.push(json!({"type": "input.commit"})); // empty segment 1
        msgs.push(text("Second segment here."));
        msgs.push(json!({"type": "input.done"})); // commits segment 2
        let got = session(&url, &msgs).await;
        let kinds: Vec<&str> = got.iter().map(|m| m["type"].as_str().unwrap_or("binary")).collect();
        assert_eq!(
            kinds,
            [
                "session.configured",
                "audio.start",
                "binary",
                "audio.done",
                "input.committed",
                "input.committed",
                "audio.start",
                "binary",
                "audio.done",
                "input.committed",
                "session.done"
            ]
        );
        assert_eq!(got[0]["session"]["sample_rate"], 24000);
        assert_eq!(got[2]["binary"], "P How are you ");
        assert_eq!(
            got[4],
            json!({"type": "input.committed", "segment_index": 0, "segment_sentences": 2, "total_sentences": 2})
        );
        assert_eq!(
            got[5],
            json!({"type": "input.committed", "segment_index": 1, "segment_sentences": 0, "total_sentences": 2})
        );
        assert_eq!(got[7]["binary"], "P"); // the whole text was the first turn
        assert_eq!(got[9]["segment_index"], 2);
        assert_eq!(got[9]["total_sentences"], 3);
    }

    #[tokio::test]
    async fn sentence_granularity_holds_clauses() {
        let url = server().await;
        let config = |g: &str| json!({"type": "session.config", "session": {"split_granularity": g}});
        let words = "One two three, four five six. Seven eight nine";
        let run = |g: &'static str| {
            let url = url.clone();
            async move {
                let mut msgs = vec![config(g)];
                msgs.extend(words.split(' ').map(|w| text(&format!("{w} "))));
                msgs.push(json!({"type": "input.done"}));
                session(&url, &msgs).await.into_iter().find(|m| m.get("binary").is_some()).unwrap()["binary"].clone()
            }
        };
        // Clause: the prompt is "One two three, "; sentence: "One two three, four five six. ".
        assert_eq!(run("clause").await, "P four five six. Seven eight nine ");
        assert_eq!(run("sentence").await, "P Seven eight nine ");
    }

    #[tokio::test]
    async fn protocol_errors() {
        let url = server().await;
        let got = session(&url, &[text("hi")]).await;
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["type"], "error");

        let bad = json!({"type": "session.config", "session": {"response_format": "wav", "stream_audio": true}});
        assert_eq!(session(&url, &[bad]).await[0]["type"], "error");

        let msgs = [json!({"type": "session.config"}), json!({"type": "bogus"}), json!({"type": "input.done"})];
        let kinds: Vec<Value> = session(&url, &msgs).await.into_iter().map(|m| m["type"].clone()).collect();
        assert_eq!(kinds, ["session.configured", "error", "session.done"]);
    }

    #[test]
    fn sentences() {
        assert_eq!(count_sentences("  "), 0);
        assert_eq!(count_sentences("Hello there"), 1);
        assert_eq!(count_sentences("Hello. How are you?"), 2);
        assert_eq!(count_sentences("It costs 3.5 dollars. Fine"), 2);
        assert_eq!(count_sentences("Wait... what?!"), 2);
        assert_eq!(count_sentences("你好。今天天气很好！"), 2);
    }
}
