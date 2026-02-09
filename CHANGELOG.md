# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
