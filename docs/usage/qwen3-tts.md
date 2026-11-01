# Qwen3-TTS

Qwen3-TTS is a state-of-the-art text-to-speech model from Alibaba's Qwen team. VoxServe supports the all three variants (Qwen3-TTS-1.7B) with input/output streaming inference.

## Model Variants

1. Custom Voice Model

Uses predefined speaker embeddings for consistent, high-quality voices.

```bash
vox-serve --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8000
```

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is a test.",
    "speaker": "ryan",  # Predefined speaker
    "language": "english"
})
```

See the [model config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/main/config.json) for a list of speakers and languages supported.

### 2. Base Model

Clone any voice using a reference audio sample and its transcript.

```bash
vox-serve --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 8000
```

```python
response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is a cloned voice.",
    "audio_path": "/path/to/reference.wav",
    "ref_text": "Transcript of the reference audio.",
    "language": "english"
})
```

This mode uses in-context learning to adapt the model to the reference voice.

### 3. Voice Design Mode

Generate voices based on natural language descriptions.

```bash
vox-serve --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --port 8000
```

```python
response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is a designed voice.",
    "instruct": "A warm, friendly female voice with a slight British accent.",
    "language": "english"
})
```

### Input Streaming

We also support input text streaming mode, ideal for connecting with text LLM to build a voice chatbot. For this, start the server with input streaming scheduler:

```bash
vox-serve --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 8000 --scheduler-type input_streaming
```

And see the example client script [here](https://github.com/vox-serve/vox-serve/tree/main/examples/input_streaming).

With input streaming, the Base model clones a voice from its speaker embedding only, because in-context learning needs the whole text up front: without reference audio it uses the default voice, and a request with reference audio must set `x_vector_only_mode` to true.
