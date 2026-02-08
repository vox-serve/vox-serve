# Chatterbox

Chatterbox is a text-to-speech model developed by Resemble AI, featuring a 0.5B LLM with flow matching and HiFT vocoder. It includes built-in Perth watermarking for audio authentication.

## Quickstart

Start the server:

```bash
python -m vox_serve.launch --model ResembleAI/chatterbox --port 8000
```

Generate speech:

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is Chatterbox speaking!"
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `streaming` | boolean | `false` | Enable streaming response |

## Examples

### Basic Usage

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "The quick brown fox jumps over the lazy dog."
})
```

### Streaming Audio

```python
import requests

with requests.post(
    "http://localhost:8000/generate",
    json={"text": "Hello world!", "streaming": True},
    stream=True
) as response:
    with open("output.wav", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
```

### Using curl

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Chatterbox!"}' \
  -o output.wav
```

## CLI Options

```bash
python -m vox_serve.launch \
  --model ResembleAI/chatterbox \
  --port 8000 \
  --temperature 0.7 \
  --top_p 0.9
```

## Architecture Notes

Chatterbox uses a LLaMA-based architecture with:
- Flow matching for high-quality audio generation
- HiFT vocoder for waveform synthesis
- Built-in Perth watermarking for audio authentication

Voice cloning via audio conditioning is planned for a future release.
