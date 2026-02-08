# Zonos

Zonos-v0.1 is a 1B parameter text-to-speech model developed by Zyphra, using a DAC (Descript Audio Codec) tokenizer for high-fidelity audio synthesis.

## Quickstart

Start the server:

```bash
python -m vox_serve.launch --model Zyphra/Zonos-v0.1-transformer --port 8000
```

Generate speech:

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is Zonos speaking!"
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
  -d '{"text": "Hello from Zonos!"}' \
  -o output.wav
```

## CLI Options

```bash
python -m vox_serve.launch \
  --model Zyphra/Zonos-v0.1-transformer \
  --port 8000 \
  --temperature 0.7 \
  --top_p 0.9
```

## Architecture Notes

Zonos uses a transformer backbone with DAC tokenizer, producing high-fidelity audio output. The model supports speaker embedding conditioning for voice customization (API support coming soon).
