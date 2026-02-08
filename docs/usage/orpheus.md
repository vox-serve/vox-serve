# Orpheus

Orpheus is a 3B parameter text-to-speech model developed by Canopy Labs, using a SNAC audio tokenizer for high-quality speech synthesis.

## Quickstart

Start the server:

```bash
python -m vox_serve.launch --model canopylabs/orpheus-3b-0.1-ft --port 8000
```

Generate speech:

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is Orpheus speaking!",
    "voice": "tara"
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Available Voices

Orpheus supports 8 built-in voices:

| Voice | Description |
|-------|-------------|
| `tara` | Default female voice |
| `leah` | Female voice |
| `jess` | Female voice |
| `mia` | Female voice |
| `zoe` | Female voice |
| `leo` | Male voice |
| `dan` | Male voice |
| `zac` | Male voice |

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `voice` | string | `"tara"` | Voice to use for synthesis |
| `streaming` | boolean | `false` | Enable streaming response |

## Examples

### Basic Usage

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "The quick brown fox jumps over the lazy dog.",
    "voice": "leo"
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
  -d '{"text": "Hello from Orpheus!", "voice": "tara"}' \
  -o output.wav
```

## CLI Options

```bash
python -m vox_serve.launch \
  --model canopylabs/orpheus-3b-0.1-ft \
  --port 8000 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1
```
