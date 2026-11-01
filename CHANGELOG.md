# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-01

VoxServe v2: a Rust serving runtime and spatial GPU splitting by the urgency of audio work. See the
[release blog post](https://vox-serve.github.io/2026/11/01/voxserve-v2.html) and the new version of the
[paper](https://arxiv.org/abs/2602.00269).

### Added

- **Rust serving runtime** (the `vox_serve._rt` extension, shipped in the Linux x86_64 wheel): HTTP server,
  scheduling, KV paging, FlashInfer planning and CUDA graph launches with no Python on the per-step path.
  `vox-serve` uses it by default and falls back to the Python runtime when the extension is unavailable or an
  argument needs the Python runtime (`--runtime auto|rust|python`)
- **Spatial GPU splitting**: the LM and the codec run concurrently on separate SM partitions, and chunks a
  listener is waiting for (first chunks, chunks near their playback deadline) take an urgent lane on the whole
  GPU (`--overlap-detok`, `--sm-modes`, `--urgent-lane`, `--stream-priorities`); a playback-deadline scheduling
  policy (`--scheduler-type deadline`); data parallelism (`--dp-size`) and the codec on its own GPU (`--codec-gpu`)
- **Kernel fusion**: `--fuse-lm`, `--fuse-layers`, `--fuse-codec`, `--fuse-depth`, `--fuse-talker` and
  `--compile-detok` (model-specific)
- **Rust benchmark mode**: `--bench-requests N --bench-out DIR` generates N requests without HTTP and writes WAVs
- **New models**: Voxtral-4B-TTS (`voxtral-tts`), Fun-CosyVoice3-0.5B (`cosyvoice3`), Higgs Audio v3 TTS
  (`higgs-tts`), Higgs Audio v2 (`higgs-audio-v2`), VibeVoice-Realtime-0.5B (`vibevoice`) and XTTS-v2 (`xtts`)
- **Chatterbox voice cloning** from reference audio
- **Documentation**: serving runtimes, adding a model, CLI reference and concepts pages
- **Tests and CI**: CPU tests for the schedulers, codec batching and Rust vs Python prompt parity; Rust checks
  (rustfmt, clippy, unit tests); CI on pull requests; release builds of the sdist and a manylinux wheel

### Changed

- **Every model is implemented once, as a program** (`vox_serve/model/<name>/program.py`) that both runtimes
  serve; the per-model workers and the step methods of the model classes are gone (`docs/adding-a-model.rst`)
- The `vox-serve` command is now `vox_serve.cli:main`, which picks the runtime; `vox-serve --runtime python`
  (or `python -m vox_serve.launch`) runs the Python runtime with its previous arguments
- Chatterbox's default token budget is 1200 (was 200, which cut outputs at about 4 s), and its generated
  speech positions start from 1 as in the reference implementation

### Fixed

- Chatterbox and CosyVoice2 audio no longer drops to near-silence at every streamed chunk boundary
- Failed requests return their reason (Python runtime)
- Scheduling: look-ahead codec requests finish after their last chunk; non-streaming requests get no playback
  deadline
- Audio conversion to int16 saturates instead of wrapping

## [0.1.0] - 2026-02-09

### Added

- **Qwen3-TTS Model Support**: Full support for Qwen3-TTS-1.7B model with custom voice cloning capabilities
- **Input Text Streaming**: Real-time input text streaming mode for Qwen3 TTS, enabling lower latency for long-form generation
- **Playground UI**: Interactive web-based playground for testing TTS models with LLM chat mode support
- **Detokenizer Interval Configuration**: CLI argument `--detokenize-interval` for controlling audio chunk generation frequency (Qwen3 TTS)
- **Model-Specific Request Parameters**: Support for passing model-specific kwargs in client requests
- **Qwen3 TTS Detokenizer Caching**: Improved performance through caching in the detokenization pipeline
- **Documentation**: Comprehensive Sphinx documentation with per-model pages, GitHub Pages deployment

### Fixed

- Decoder cache shape issue for Qwen3 TTS
- Scheduler bug with excessive prefill operations
- CUDA graph compatibility issues for Qwen3 TTS
- Audio trimming logic improvements
- Head dimension definition for attention layers
- Input dtype handling for Qwen3 base model
- Depth model last token handling
- GLM detokenizer inference issues
- Playground compatibility for CSM model

### Changed

- Improved preprocessing for input streaming mode
- Enhanced playground UI appearance and usability

## [0.0.1] - 2026-01-15

### Added

- Initial release
- Core serving infrastructure with FastAPI and ZeroMQ
- Support for CSM-1B, Orpheus-3B, Zonos-v0.1, GLM-4-Voice-9B, Step-Audio-2-Mini, Chatterbox, CosyVoice2-0.5B
- CUDA graph optimization for decode phase
- PagedAttention with FlashInfer for efficient KV cache management
- Streaming and non-streaming response modes
- Audio watermarking support
