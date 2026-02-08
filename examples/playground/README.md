# VoxServe Playground

A web-based playground for interacting with VoxServe TTS server. Start the server, send requests, and play back generated audio with real-time latency metrics.

![VoxServe Playground](static/images/playground-sample.png)

## Features

- **Server Management**: Start and stop VoxServe server with configurable parameters
- **GPU Selection**: Choose which GPUs to use via `CUDA_VISIBLE_DEVICES`
- **Model Selection**: Support for all VoxServe models including all Qwen3-TTS variants
- **Server Logs**: Real-time log display with syntax highlighting
- **Streaming Playback**: Real-time audio playback with Web Audio API
- **Latency Metrics**: Display time-to-first-audio (TTFA) for both network and playback
- **Proxy Architecture**: All requests routed through the playground backend

## Usage

Start the playground server by running the following command in the playground directory:

```bash
cd examples/playground
python app.py --port 7860
```