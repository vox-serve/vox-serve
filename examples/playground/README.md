# VoxServe Playground

A web-based playground for interacting with VoxServe TTS server. Start the server, send requests, and play back generated audio with real-time latency metrics.

## Features

- **Server Management**: Start and stop VoxServe server with configurable parameters
- **GPU Selection**: Choose which GPUs to use via `CUDA_VISIBLE_DEVICES`
- **Model Selection**: Support for all VoxServe models including all Qwen3-TTS variants
- **Server Logs**: Real-time log display with syntax highlighting
- **Streaming Playback**: Real-time audio playback with Web Audio API
- **Latency Metrics**: Display time-to-first-audio (TTFA) for both network and playback
- **Proxy Architecture**: All requests routed through the playground backend

## Usage

1. Activate the virtual environment:
   ```bash
   source /home/keisuke/vox-serve/.venv/bin/activate
   ```

2. Start the playground server:
   ```bash
   cd /home/keisuke/vox-serve/examples/playground
   python app.py --port 7860
   ```

3. Open your browser to http://localhost:7860

4. Configure and start VoxServe:
   - Select a model
   - Choose GPU(s)
   - Set port and other parameters
   - Click "Start Server"

5. Generate audio:
   - Enter text to synthesize
   - Optionally upload reference audio (for voice cloning models)
   - Click "Generate Audio"

## Supported Models

### Standard Models
| Model | Description |
|-------|-------------|
| Orpheus-3B | General TTS with optional audio input |
| CSM-1B | Compact speech model |
| Zonos-v0.1 | Zonos text-to-speech |
| GLM-4-Voice-9B | Speech-to-speech (requires audio input) |
| Step-Audio-2-Mini | Speech-to-speech (requires audio input) |
| Chatterbox | Voice cloning (requires audio input) |
| CosyVoice2-0.5B | General TTS with optional audio input |

### Qwen3-TTS Models
All 6 models from the [Qwen3-TTS collection](https://huggingface.co/collections/Qwen/qwen3-tts):

| Model | Description |
|-------|-------------|
| Qwen3-TTS-1.7B-Base | Base TTS model |
| Qwen3-TTS-0.6B-Base | Lightweight base model |
| Qwen3-TTS-1.7B-CustomVoice | Voice cloning support |
| Qwen3-TTS-0.6B-CustomVoice | Lightweight voice cloning |
| Qwen3-TTS-1.7B-VoiceDesign | Voice design with instructions |
| Qwen3-TTS-0.6B-VoiceDesign | Lightweight voice design |

## Configuration

### Server Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Model | TTS model to use | orpheus |
| Port | Server port | 8000 |
| GPUs | GPU indices for CUDA_VISIBLE_DEVICES | [0] |
| Max Batch Size | Maximum inference batch size | 8 |
| Top-p | Nucleus sampling parameter | model default |
| Top-k | Top-k sampling parameter | model default |
| Temperature | Sampling temperature | model default |
| Max Tokens | Maximum tokens to generate | model default |
| CUDA Graph | Enable CUDA graph optimization | true |

### Model-Specific Parameters

| Model Type | Audio Input | Language | Speaker | Ref Text | Instruct |
|------------|-------------|----------|---------|----------|----------|
| Base models | optional | yes | yes | yes | - |
| CustomVoice | optional | yes | yes | yes | - |
| VoiceDesign | optional | yes | - | - | yes |

## Architecture

```
examples/playground/
├── app.py              # FastAPI backend with /api/generate proxy
├── server_manager.py   # VoxServe subprocess management
├── static/
│   ├── css/styles.css  # Styling with log display
│   └── js/
│       ├── audio-player.js  # Streaming audio playback
│       └── main.js          # UI logic with log polling
├── templates/
│   └── index.html      # Main page with logs section
└── README.md
```

### Request Flow

1. Browser sends request to `/api/generate` on playground server
2. Playground proxies request to VoxServe at `localhost:{port}/generate`
3. Audio chunks are streamed back through the proxy
4. Browser plays audio with Web Audio API

## Audio Buffering

The playground buffers ~100ms of audio before starting playback to ensure smooth playback without gaps. Two TTFA metrics are displayed:

- **TTFA (Network)**: Time from request to first audio byte received
- **TTFA (Playback)**: Time from request to first audio sample played (includes buffering delay)

## Server Logs

The logs section displays real-time output from the VoxServe server process with:
- Syntax highlighting for errors, warnings, and info messages
- Auto-scroll toggle
- Clear button to reset the log display
- 1-second polling interval

## Requirements

The playground uses dependencies already available in the VoxServe environment:
- FastAPI
- Uvicorn
- Jinja2 (for templates)
- aiohttp (for proxy streaming)
- Requests (for health checks)
