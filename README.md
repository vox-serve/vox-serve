<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/images/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/_static/images/logo-light.png">
    <img src="docs/_static/images/logo-dark.png" alt="VoxServe Logo" width="320" />
  </picture>
</p>

<h1 align="center">VoxServe</h1>
<p align="center"><strong>A High-Performance Serving System for Speech Language Models</strong></p>

<p align="center">
  <a href="https://pypi.org/project/vox-serve/"><img src="https://img.shields.io/pypi/v/vox-serve?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI&color=3775A9" alt="PyPI"></a>
  <a href="https://arxiv.org/abs/2602.00269"><img src="https://img.shields.io/badge/arXiv-2602.00269-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://vox-serve.github.io/vox-serve/"><img src="https://img.shields.io/badge/docs-online-009688?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"></a>
</p>

VoxServe delivers low-latency, high-throughput inference for Speech Language Models (SpeechLMs), including text-to-speech (TTS) and speech-to-speech (STS) models.

## News

- **[2025-02]** Blog post: [Light-Speed Qwen3-TTS Serving at Scale with VoxServe](https://vox-serve.github.io/2026/02/09/qwen3-tts-support.html)
- **[2025-02]** Paper released: [VoxServe: A Streaming-Centric Serving System for Speech Language Models](https://arxiv.org/abs/2602.00269)

## Quick Start

Install via pip and start the server:

```bash
pip install vox-serve
vox-serve --model <model-name> --port <port-number>
```

Or install from source:

```bash
git clone https://github.com/vox-serve/vox-serve.git
cd vox-serve
pip install -e .
python -m vox_serve.launch --model <model-name> --port <port-number>
```

Send requests to the server:

```bash
# Text-to-speech
curl -X POST "http://localhost:<port-number>/generate" \
  -F "text=Hello world" -F "streaming=true" -o output.wav

# Speech-to-speech (for models with audio input support)
curl -X POST "http://localhost:<port-number>/generate" \
  -F "text=Hello world" -F "@input.wav" -F "streaming=true" -o output.wav
```

See the [`examples/`](examples/) directory for more usage examples.

## Supported Models

VoxServe supports the following TTS and STS models:

| Model | Type | Link |
|-------|------|------|
| `chatterbox` | TTS | [Chatterbox TTS](https://huggingface.co/ResembleAI/chatterbox) |
| `cosyvoice2` | TTS | [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) |
| `csm` | TTS | [CSM-1B](https://huggingface.co/sesame/csm-1b) |
| `orpheus` | TTS | [Orpheus-3B](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) |
| `qwen3-tts` | TTS | [Qwen3-TTS-1.7B](https://huggingface.co/collections/Qwen/qwen3-tts) |
| `zonos` | TTS | [Zonos-v0.1](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) |
| `glm` | STS | [GLM-4-Voice-9B](https://huggingface.co/zai-org/glm-4-voice-9b) |
| `step` | STS | [Step-Audio-2-Mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) |

See the [models documentation](https://vox-serve.github.io/vox-serve/models.html) for detailed information. More models coming soon.

## Demos

### Ultra-Low Latency

VoxServe is optimized for real-time speech synthesis. The demo below shows a TTS request achieving **40 ms** Time-To-First-Audio (TTFA) on an NVIDIA H100 GPU with `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.

<a href="https://vimeo.com/1163095537">
  <img src="https://vumbnail.com/1163095537.jpg" alt="Ultra-Low Latency Demo" width="600">
</a>

### Real-Time LLM Integration

Qwen3-TTS supports incremental text input, enabling seamless integration with LLMs for voice chatbots. The demo below shows VoxServe connected to a local LLM with low end-to-end latency.

<a href="https://vimeo.com/1163095770">
  <img src="https://vumbnail.com/1163095770.jpg" alt="LLM Integration Demo" width="600">
</a>

## Playground

VoxServe includes a web-based playground for interactive testing. Use the browser UI to manage servers, generate audio, and view real-time logs.

![VoxServe Playground](examples/playground/static/images/playground-sample.png)

See [examples/playground/README.md](examples/playground/README.md) for setup instructions.
