# CosyVoice2

CosyVoice2-0.5B is a text-to-speech model developed by Alibaba FunAudioLLM team, using a 0.5B LLM with flow matching and HiFT vocoder for natural speech synthesis.

## Quickstart

Start the server:

```bash
python -m vox_serve.launch --model FunAudioLLM/CosyVoice2-0.5B --port 8000
```

Generate speech:

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is CosyVoice speaking!"
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `streaming` | boolean | `false` | Enable streaming response |

## Special Tokens

CosyVoice2 supports special tokens for expressive speech:

| Token | Description |
|-------|-------------|
| `[breath]` | Insert a breath |
| `[laughter]` | Insert laughter |
| `[cough]` | Insert a cough |
| `[sigh]` | Insert a sigh |
| `[noise]` | Insert background noise |
| `<strong>...</strong>` | Emphasize text |
| `<laughter>...</laughter>` | Speak with laughter |

## Examples

### Basic Usage

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "The quick brown fox jumps over the lazy dog."
})
```

### Expressive Speech

```python
import requests

# With special tokens
response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello! [breath] How are you doing today? <laughter>That's so funny!</laughter>"
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
  -d '{"text": "Hello from CosyVoice!"}' \
  -o output.wav
```

## CLI Options

```bash
python -m vox_serve.launch \
  --model FunAudioLLM/CosyVoice2-0.5B \
  --port 8000 \
  --temperature 0.7 \
  --top_p 0.9
```

## Architecture Notes

CosyVoice2 features:
- Qwen-based 0.5B LLM backbone
- Flow matching decoder for high-quality audio
- HiFT vocoder for efficient waveform synthesis
- Support for expressive speech via special tokens

Voice cloning via reference audio is planned for a future release.
