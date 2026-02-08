# CSM

CSM-1B is a text-to-speech model developed by Sesame, featuring a depth-wise architecture with a Mimi audio tokenizer for natural conversational speech.

## Quickstart

Start the server:

```bash
python -m vox_serve.launch --model sesame/csm-1b --port 8000
```

Generate speech:

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "Hello, this is CSM speaking!",
    "speaker": 0
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Speaker Modes

CSM supports two speaker modes for conversational scenarios:

| Speaker | Description |
|---------|-------------|
| `0` | Conversational speaker A |
| `1` | Conversational speaker B |

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `speaker` | integer | `0` | Speaker ID (0 or 1) |
| `streaming` | boolean | `false` | Enable streaming response |

## Examples

### Basic Usage

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "text": "How are you doing today?",
    "speaker": 0
})
```

### Conversational Exchange

```python
import requests

# Speaker A
response_a = requests.post("http://localhost:8000/generate", json={
    "text": "Hello! How can I help you?",
    "speaker": 0
})

# Speaker B
response_b = requests.post("http://localhost:8000/generate", json={
    "text": "I'd like to know more about your services.",
    "speaker": 1
})
```

### Streaming Audio

```python
import requests

with requests.post(
    "http://localhost:8000/generate",
    json={"text": "Hello world!", "speaker": 0, "streaming": True},
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
  -d '{"text": "Hello from CSM!", "speaker": 0}' \
  -o output.wav
```

## CLI Options

```bash
python -m vox_serve.launch \
  --model sesame/csm-1b \
  --port 8000 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1
```

## Architecture Notes

CSM uses a depth-wise transformer architecture that processes audio tokens across multiple codebook levels, enabling high-quality natural speech synthesis with conversational prosody.
