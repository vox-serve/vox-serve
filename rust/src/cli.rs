//! Command line and start-up of the Rust runtime (`vox-serve`, through the `vox_serve._rt` extension
//! module): build one engine per GPU (Python), start a scheduler thread per engine, then serve HTTP or
//! run the bench.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;
use tracing::info;

use crate::models::GenerateInput;
use crate::scheduler::{Event, Policy, Scheduler, SchedulerConfig, Submit};
use crate::{deadline, engine, models, pyinit, server, v2, wav};

#[derive(Parser, Debug)]
#[command(name = "vox-serve", about = "VoxServe with the Rust serving runtime")]
pub struct Args {
    #[arg(long, default_value = "canopylabs/orpheus-3b-0.1-ft")]
    model: String,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long, default_value_t = 8)]
    max_batch_size: usize,
    #[arg(long, default_value_t = 64)]
    page_size: usize,
    #[arg(long)]
    max_num_pages: Option<usize>,
    #[arg(long, default_value_t = 2048)]
    max_context: usize,
    /// Comma-separated prefill row buckets (default: 256,512,1024,2048 capped at max-context).
    #[arg(long, value_delimiter = ',')]
    prefill_buckets: Vec<usize>,
    #[arg(long, value_enum, default_value = "base")]
    scheduler_type: Policy,
    #[arg(long, default_value_t = 1)]
    max_prefills_per_step: usize,
    /// Deadline policy: target time to first audio, measured from scheduler admission.
    #[arg(long, default_value_t = 500.0)]
    deadline_startup_ms: f64,
    /// Estimated LM step / codec batch / prefill batch costs (calibrate per model and device).
    #[arg(long, default_value_t = 10.0)]
    deadline_lm_step_ms: f64,
    #[arg(long, default_value_t = 10.0)]
    deadline_codec_ms: f64,
    #[arg(long, default_value_t = 30.0)]
    deadline_prefill_ms: f64,
    /// Maximum continuous prefill deferral while active decodes are at risk.
    #[arg(long, default_value_t = 100.0)]
    deadline_max_prefill_deferral_ms: f64,
    /// Deadline policy: pause LM requests with this many pending codec chunks (0 disables).
    #[arg(long, default_value_t = 0)]
    deadline_max_pending_chunks: usize,
    /// Data-parallel replicas: one engine (and scheduler thread) per GPU `0..dp_size`, requests
    /// dispatched round-robin.
    #[arg(long, default_value_t = 1)]
    dp_size: usize,
    /// Run the detokenizer on GPU 1 (LM on GPU 0), asynchronously to the LM steps.
    #[arg(long)]
    enable_disaggregation: bool,
    /// Run the detokenizer on its own stream of the same GPU, asynchronously to the LM steps.
    #[arg(long)]
    overlap_detok: bool,
    /// v2: run the LM and detok lanes asynchronously, each collected by its completion event
    /// (implies --overlap-detok).
    #[arg(long)]
    async_engines: bool,
    /// v2: SM partitions as detok SM counts, e.g. `24,48,68`; each becomes a green-context split
    /// whose graphs are captured up front (implies --async-engines). Mode 0 is always the whole GPU.
    #[arg(long, value_delimiter = ',')]
    sm_modes: Vec<usize>,
    /// v2: SM mode policy, `<mode>` or `auto:<min_decode_rows>:<mode>`.
    #[arg(long, default_value = "0")]
    sm_mode: v2::ModePolicy,
    /// v2: start non-urgent detok batches only once the LM's current step reaches its depth graph
    /// (`depth`) or layer group `i` (`layer:<i>`); needs --lm-segments depth / layers:<k>.
    #[arg(long, default_value = "none")]
    gate: v2::GateMode,
    /// v2: ready continuation chunks a detok batch waits for (0 = decode as soon as ready).
    #[arg(long, default_value_t = 0)]
    dec_batch: usize,
    /// v2: largest non-urgent detok batch (0 = largest captured size); bounds how long urgent
    /// chunks wait behind a background batch.
    #[arg(long, default_value_t = 0)]
    dec_max_batch: usize,
    /// v2: flush a partial detok batch when a playback deadline is this close (ms).
    #[arg(long, default_value_t = 50.0)]
    dec_flush_ms: f64,
    /// v2: chunks whose playback deadline is this close (ms) are urgent, like first chunks.
    #[arg(long, default_value_t = 100.0)]
    dec_urgent_ms: f64,
    /// v2: prefill steps and detok batches never overlap.
    #[arg(long)]
    prefill_exclusive: bool,
    /// v2: detok batches with urgent chunks run on all SMs while the LM waits.
    #[arg(long)]
    urgent_exclusive: bool,
    /// A third, high-priority detok lane on all SMs with its own arenas and graphs (batch sizes up
    /// to 8): first chunks and near-deadline chunks start at once, even while a background detok
    /// batch is running (implies --async-engines; Qwen3-TTS, Orpheus, Zonos, CSM, CosyVoice3).
    #[arg(long)]
    urgent_lane: bool,
    /// Largest urgent-lane batch (default 8): its graphs are captured at 1, 2, 4, ... up to it.
    #[arg(long)]
    urgent_max_batch: Option<usize>,
    /// v2: first chunks are due --deadline-startup-ms after arrival (scheduled by deadline like
    /// continuation chunks) instead of always urgent; the urgency margin adds the measured latency
    /// of the largest detok batch.
    #[arg(long)]
    slack_urgency: bool,
    /// v2: strict time sharing — LM steps and detok batches alternate on the whole GPU (each waits
    /// for the other on the GPU), keeping v2's detok batching (implies --async-engines).
    #[arg(long)]
    time_share: bool,
    /// Capture LM steps whole, split before the depth transformer (`depth`), or additionally as
    /// groups of k backbone layers (`layers:<k>`).
    #[arg(long, default_value = "whole")]
    lm_segments: String,
    /// Fuse the detokenizer with torch.compile before capturing its graphs.
    #[arg(long)]
    compile_detok: bool,
    /// Qwen3-TTS: run the depth transformer with merged QKV / gate-up GEMMs and grouped-query SDPA.
    #[arg(long)]
    fuse_depth: bool,
    /// Qwen3-TTS: run the talker with merged QKV / gate-up GEMMs and fused residual add + RMSNorm.
    #[arg(long)]
    fuse_talker: bool,
    /// Merge sibling q/k/v and gate/up projections of the LM into single GEMMs (Orpheus, CSM,
    /// CosyVoice2, Chatterbox, Step-Audio-2, Voxtral).
    #[arg(long)]
    fuse_lm: bool,
    /// Run the LM's decoder layers with merged GEMMs and fused add+RMSNorm, q/k-norm+RoPE+KV-write and
    /// SiLU-and-mul kernels (Higgs Audio v3, CosyVoice3).
    #[arg(long)]
    fuse_layers: bool,
    /// Fused codec kernels: Snake + conv bias + residual adds (Higgs Audio v3 DAC, CosyVoice3 HiFT),
    /// fused output conv (Higgs), one-launch history-ring writes and GELU GEMM epilogue (CosyVoice3).
    #[arg(long)]
    fuse_codec: bool,
    /// LM stream at high priority, detok stream at low priority.
    #[arg(long)]
    stream_priorities: bool,
    /// Write a per-launch GPU timing trace (JSON lines) into this directory.
    #[arg(long)]
    trace_dir: Option<PathBuf>,
    /// Compile the LM forward pieces with torch.compile before capturing their CUDA graphs
    /// (slower startup; no effect on the per-step host work).
    #[arg(long)]
    enable_torch_compile: bool,
    /// Restrict --enable-torch-compile to these LM functions (e.g. `decode_head,decode_depth`).
    #[arg(long, value_delimiter = ',')]
    torch_compile_targets: Vec<String>,
    /// Per-request token limit (prompt included), overriding the model's default.
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Sampling defaults of every request, overriding the model's.
    #[arg(long)]
    temperature: Option<f64>,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long)]
    top_k: Option<i64>,
    #[arg(long)]
    min_p: Option<f64>,
    #[arg(long)]
    repetition_penalty: Option<f64>,
    /// Log level (`debug`, `info`, `warning`, `error`; the RUST_LOG filter takes precedence).
    #[arg(long)]
    log_level: Option<String>,
    /// Multi-GPU: every replica's detokenizer runs on GPU `G` (asynchronous lanes). With
    /// --codec-sms N it runs on an N-SM green-context partition of that GPU, shared by all replicas,
    /// and the replica whose LM is on GPU `G` gets the remaining SMs; without it, on the whole GPU.
    /// `--dp-size 1 --codec-gpu 1` is LM/codec disaggregation; `--dp-size 2 --codec-gpu 1
    /// --codec-sms N` puts one LM on GPU 0 and one LM plus the shared codec on GPU 1.
    #[arg(long)]
    codec_gpu: Option<usize>,
    #[arg(long, default_value_t = 0)]
    codec_sms: usize,
    /// Largest detok batch captured (default: --max-batch-size); bounds codec graph memory.
    #[arg(long)]
    max_detok_batch_size: Option<usize>,
    /// Codec frames per detokenized chunk (Qwen3-TTS; default: the model's).
    #[arg(long)]
    detokenize_interval: Option<usize>,
    /// Classifier-free guidance scale (Voxtral-TTS; default: the model's).
    #[arg(long)]
    cfg_scale: Option<f64>,
    /// Run N concurrent requests through the scheduler (no HTTP), report timings and exit.
    #[arg(long)]
    bench_requests: Option<usize>,
    #[arg(
        long,
        default_value = "Hey there, this is a short benchmark sentence for the vox serve runtime written in rust."
    )]
    bench_text: String,
    /// Input audio file for the bench requests (speech-input models).
    #[arg(long)]
    bench_audio: Option<String>,
    /// Model kwargs of the bench requests, as a JSON object (e.g. '{"x_vector_only_mode": true}').
    #[arg(long)]
    bench_kwargs: Option<String>,
    /// Write the bench requests' audio as WAV files into this directory.
    #[arg(long)]
    bench_out: Option<PathBuf>,
    /// Greedy decoding for every request (testing).
    #[arg(long)]
    greedy: bool,
    /// Prompt builder instead of the model's own (`python`: the program's `preprocess`; `template`):
    /// for bringing up a model, or checking a Rust builder against Python.
    #[arg(long)]
    prompt_builder: Option<String>,
}

pub fn run(args: Args) -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            let level = args.log_level.as_deref().unwrap_or("info").to_lowercase();
            (if level == "warning" { "warn".to_string() } else { level }).into()
        }))
        .try_init();

    let deadline = deadline::DeadlineConfig {
        startup: args.deadline_startup_ms / 1000.0,
        lm_step: args.deadline_lm_step_ms / 1000.0,
        codec: args.deadline_codec_ms / 1000.0,
        prefill: args.deadline_prefill_ms / 1000.0,
        max_prefill_deferral: args.deadline_max_prefill_deferral_ms / 1000.0,
        max_pending_chunks: args.deadline_max_pending_chunks,
    };
    anyhow::ensure!(deadline.valid(), "deadline times must be finite and positive");
    anyhow::ensure!(args.max_prefills_per_step > 0, "max-prefills-per-step must be positive");
    let mut config = json!({
        "model": args.model,
        "max_batch_size": args.max_batch_size,
        "page_size": args.page_size,
        "max_context": args.max_context,
    });
    if let Some(n) = args.max_num_pages {
        config["max_num_pages"] = json!(n);
    }
    if !args.prefill_buckets.is_empty() {
        config["prefill_buckets"] = json!(args.prefill_buckets);
    }
    if args.enable_torch_compile {
        config["torch_compile"] = json!(true);
        if !args.torch_compile_targets.is_empty() {
            config["torch_compile_targets"] = json!(args.torch_compile_targets);
        }
    }
    if args.codec_gpu.is_some() && (args.enable_disaggregation || !args.sm_modes.is_empty()) {
        anyhow::bail!("--codec-gpu sets the detok placement itself: drop --enable-disaggregation / --sm-modes");
    }
    if args.enable_disaggregation {
        if args.dp_size > 1 {
            anyhow::bail!("--enable-disaggregation cannot be combined with --dp-size > 1");
        }
        config["detok_device"] = json!(1);
    }
    let async_engines = args.async_engines
        || !args.sm_modes.is_empty()
        || args.urgent_lane
        || args.time_share
        || args.codec_gpu.is_some();
    if args.overlap_detok || async_engines {
        config["overlap_detok"] = json!(true);
    }
    if !args.sm_modes.is_empty() {
        config["sm_modes"] = json!(args.sm_modes);
    }
    config["lm_segments"] = json!(args.lm_segments);
    if args.compile_detok {
        config["compile_detok"] = json!(true);
    }
    if args.fuse_depth {
        config["fuse_depth"] = json!(true);
    }
    if args.fuse_talker {
        config["fuse_talker"] = json!(true);
    }
    if args.fuse_lm {
        config["fuse_lm"] = json!(true);
    }
    if args.fuse_layers {
        config["fuse_layers"] = json!(true);
    }
    if args.fuse_codec {
        config["fuse_codec"] = json!(true);
    }
    if let Some(n) = args.max_tokens {
        config["max_tokens"] = json!(n);
    }
    for (key, v) in [
        ("temperature", args.temperature),
        ("top_p", args.top_p),
        ("min_p", args.min_p),
        ("repetition_penalty", args.repetition_penalty),
    ] {
        if let Some(v) = v {
            config[key] = json!(v);
        }
    }
    if let Some(k) = args.top_k {
        config["top_k"] = json!(k);
    }
    if let Some(n) = args.max_detok_batch_size {
        config["max_detok_batch_size"] = json!(n);
    }
    if let Some(n) = args.detokenize_interval {
        config["detok_interval"] = json!(n);
    }
    if let Some(v) = args.cfg_scale {
        config["cfg_alpha"] = json!(v);
    }
    if args.urgent_lane {
        config["urgent_lane"] = json!(true);
    }
    if let Some(n) = args.urgent_max_batch {
        config["urgent_max_batch"] = json!(n);
    }
    let v2cfg = v2::V2Config {
        enabled: async_engines,
        dec_batch: args.dec_batch,
        dec_max_batch: args.dec_max_batch,
        dec_flush: args.dec_flush_ms / 1000.0,
        urgent: args.dec_urgent_ms / 1000.0,
        gate: args.gate,
        mode: args.sm_mode,
        prefill_exclusive: args.prefill_exclusive,
        urgent_exclusive: args.urgent_exclusive,
        slack_urgency: args.slack_urgency,
        time_share: args.time_share,
    };
    let engine_opts = engine::EngineOpts { priorities: args.stream_priorities, trace_dir: args.trace_dir.clone() };
    let mut ranks = Vec::new();
    let mut threads = Vec::new();
    for rank in 0..args.dp_size {
        let mut config = config.clone();
        config["device"] = json!(rank);
        let mut v2cfg = v2cfg.clone();
        if let Some(g) = args.codec_gpu {
            if rank == g {
                // LM and the shared codec partition on the same GPU: the 1-GPU split.
                if args.codec_sms > 0 {
                    config["sm_modes"] = json!([args.codec_sms]);
                    v2cfg.mode = v2::ModePolicy::Fixed(1);
                }
            } else {
                config["detok_device"] = json!(g);
                if args.codec_sms > 0 {
                    config["detok_sms"] = json!(args.codec_sms);
                }
                v2cfg.mode = v2::ModePolicy::Fixed(0);
            }
        }
        let t = Instant::now();
        info!("building engine {rank} (python init) with {config}");
        let spec = pyinit::build_engine(&config)?;
        info!("engine {rank} spec ready in {:.1}s: {} graphs", t.elapsed().as_secs_f64(), spec.graphs.len());
        let mut model = models::load(&spec.meta, args.prompt_builder.as_deref())?;
        if args.greedy {
            model = model.greedy();
        }
        let model = Arc::new(model);

        let (tx, rx) = crossbeam_channel::bounded::<Submit>((args.max_batch_size * 2).max(1));
        let load = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cfg = SchedulerConfig {
            policy: args.scheduler_type,
            max_prefills_per_step: args.max_prefills_per_step,
            deadline: deadline.clone(),
            v2: v2cfg,
            load: load.clone(),
        };
        let opts = engine_opts.clone();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
        let sched_model = model.clone();
        let sched = std::thread::Builder::new().name(format!("vox-engine-{rank}")).spawn(move || -> Result<()> {
            let engine = match engine::Engine::new(spec, rank as i32, &opts) {
                Ok(e) => e,
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow::anyhow!("{e:#}")));
                    return Err(e);
                }
            };
            let s = Scheduler::new(engine, sched_model, cfg, rx)?;
            let _ = ready_tx.send(Ok(()));
            let r = s.run();
            if let Err(e) = &r {
                tracing::error!("engine {rank} loop failed: {e:#}");
            }
            r
        })?;
        ready_rx.recv().context("engine thread died")??;
        info!("engine {rank} ready");
        ranks.push(server::Rank { tx, model, load });
        threads.push(sched);
    }

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    if let Some(n) = args.bench_requests {
        rt.block_on(bench(n, &args, ranks))?;
    } else {
        let upload_dir = std::env::temp_dir().join("vox_serve_audio/uploads");
        std::fs::create_dir_all(&upload_dir)?;
        let state = server::AppState::new(ranks, upload_dir);
        let addr = format!("{}:{}", args.host, args.port);
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            info!("listening on http://{addr}");
            axum::serve(listener, server::router(state))
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
            anyhow::Ok(())
        })?;
    }
    drop(rt);
    for t in threads {
        t.join().map_err(|_| anyhow::anyhow!("engine thread panicked"))??;
    }
    Ok(())
}

async fn bench(n: usize, args: &Args, ranks: Vec<server::Rank>) -> Result<()> {
    let model_kwargs = match &args.bench_kwargs {
        Some(k) => serde_json::from_str(k)?,
        None => Default::default(),
    };
    let input = GenerateInput { text: args.bench_text.clone(), audio_path: args.bench_audio.clone(), model_kwargs };
    // Prompts may reference per-engine state (feature-pool rows): preprocess once per rank.
    let prompts = ranks.iter().map(|r| r.model.preprocess(&input)).collect::<Result<Vec<_>>>()?;
    let model = ranks[0].model.clone();
    let sampling = model.default_sampling();
    info!(
        "bench: {n} requests over {} engine(s), prompt {} rows",
        ranks.len(),
        prompts[0].tokens.len() / model.n_codebooks()
    );
    let t0 = Instant::now();
    let mut handles = Vec::new();
    for i in 0..n {
        let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
        let s = Submit {
            id: format!("bench-{i}"),
            prompt: prompts[i % ranks.len()].clone(),
            sampling,
            seed: 1234 + i as u64,
            streaming: true,
            tx: etx,
            text_rx: None,
        };
        let tx = ranks[i % ranks.len()].tx.clone();
        tokio::task::spawn_blocking(move || tx.send(s)).await??;
        handles.push(tokio::spawn(async move {
            let start = Instant::now();
            let mut ttfa = None;
            let mut pcm = Vec::<u8>::new();
            let mut reason = String::new();
            while let Some(ev) = erx.recv().await {
                match ev {
                    Event::Audio(b) => {
                        ttfa.get_or_insert(start.elapsed());
                        pcm.extend_from_slice(&b);
                    }
                    Event::Done(r) => {
                        reason = r;
                        break;
                    }
                    Event::Error(e) => {
                        reason = format!("error: {e}");
                        break;
                    }
                }
            }
            (ttfa, start.elapsed(), pcm, reason)
        }));
    }
    drop(prompts);
    drop(ranks);
    let mut total_audio = 0.0;
    for (i, h) in handles.into_iter().enumerate() {
        let (ttfa, e2e, pcm, reason) = h.await?;
        let secs = pcm.len() as f64 / 2.0 / model.sample_rate() as f64;
        total_audio += secs;
        info!(
            "req {i}: ttfa {:.1} ms, e2e {:.2} s, audio {:.2} s, finish {reason}",
            ttfa.map_or(f64::NAN, |d| d.as_secs_f64() * 1e3),
            e2e.as_secs_f64(),
            secs
        );
        if let Some(dir) = &args.bench_out {
            std::fs::create_dir_all(dir)?;
            let mut body = wav::header(model.sample_rate(), pcm.len() as u32).to_vec();
            body.extend_from_slice(&pcm);
            std::fs::write(dir.join(format!("bench_{i}.wav")), body)?;
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    info!("bench done: wall {wall:.2} s, audio {total_audio:.1} s, RTF x{:.1}", total_audio / wall);
    Ok(())
}
