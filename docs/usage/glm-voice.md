# GLM-4-Voice

GLM-4-Voice-9B is a speech-to-speech (STS) model developed by Z.ai (Zhipu AI), capable of both text-to-speech and speech-to-speech interactions with a 9B parameter LLM backbone.

## Quickstart

Start the server:

```bash
python -m vox_serve.launch --model zai-org/glm-4-voice-9b --port 8000
```

### Text-to-Speech

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is GLM-4-Voice speaking!"
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Speech-to-Speech

```python
import requests

with open("input.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:8000/generate",
        files={"audio": audio_file},
        data={"text": "Please respond to my question."}
    )

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text prompt or instruction |
| `audio` | file | optional | Input audio file for STS mode |
| `audio_path` | string | optional | Path to input audio file |
| `streaming` | boolean | `false` | Enable streaming response |

## Examples

### Text-to-Speech

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "The quick brown fox jumps over the lazy dog."
})
```

### Speech-to-Speech with Audio File

```python
import requests

# Using file upload
with open("question.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/generate",
        files={"audio": f},
        data={"text": "Answer the question in the audio."}
    )
```

### Speech-to-Speech with Audio Path

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "Respond to the audio message.",
    "audio_path": "/path/to/input.wav"
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

### Using curl (TTS)

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from GLM-4-Voice!"}' \
  -o output.wav
```

### Using curl (STS)

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "text=Respond to this audio" \
  -F "audio=@input.wav" \
  -o output.wav
```

## CLI Options

```bash
python -m vox_serve.launch \
  --model zai-org/glm-4-voice-9b \
  --port 8000 \
  --temperature 0.7 \
  --top_p 0.9
```

## Architecture Notes

GLM-4-Voice features:
- 9B parameter GLM-4 based backbone
- Audio encoder for processing input speech
- Flow matching decoder with HiFT vocoder
- Support for both text-to-speech and speech-to-speech modes
