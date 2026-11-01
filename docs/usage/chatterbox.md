# Chatterbox

Chatterbox is a text-to-speech model developed by Resemble AI, featuring a 0.5B LLM with flow matching and HiFT vocoder. It clones voices from a reference clip and includes built-in Perth watermarking for audio authentication.

## Quickstart

Start the server:

```bash
vox-serve --model ResembleAI/chatterbox --port 8000
```

Generate speech:

```python
import requests

response = requests.post("http://localhost:8000/generate", data={
    "text": "Hello, this is Chatterbox speaking!",
    "streaming": "false",
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `audio` | file | none | Reference clip of the voice to clone (at least 3 s; the first 10 s are used) |
| `streaming` | boolean | `true` | Stream the audio as it is generated |

## Examples

### Voice Cloning

```python
import requests

with open("reference.wav", "rb") as ref:
    response = requests.post(
        "http://localhost:8000/generate",
        data={"text": "Hello, this is a cloned voice.", "streaming": "false"},
        files={"audio": ("reference.wav", ref, "audio/wav")},
    )

with open("cloned.wav", "wb") as f:
    f.write(response.content)
```

The server keeps a few encoded voices (`n_voices` slots, the default voice included), so repeated
requests with the same clip skip the encoding.

### Streaming Audio

```python
import requests

with requests.post(
    "http://localhost:8000/generate",
    data={"text": "Hello world!", "streaming": "true"},
    stream=True,
) as response:
    with open("output.wav", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
```

### Using curl

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "text=Hello from Chatterbox!" -F "streaming=false" \
  -o output.wav

curl -X POST "http://localhost:8000/generate" \
  -F "text=Hello in a cloned voice!" -F "audio=@reference.wav" -F "streaming=false" \
  -o cloned.wav
```

## CLI Options

```bash
vox-serve \
  --model ResembleAI/chatterbox \
  --port 8000 \
  --temperature 0.7 \
  --top-p 0.9
```

## Architecture Notes

Chatterbox uses a LLaMA-based architecture with:
- Flow matching for high-quality audio generation (S3Gen, the same decoder as CosyVoice2)
- HiFT vocoder for waveform synthesis
- Built-in Perth watermarking for audio authentication
