# WebSocket Speech Streaming Example

`websocket_streaming.py` streams text word by word, as an LLM would, to `/v1/audio/speech/stream` and saves the
audio. The endpoint follows [sglang-omni's WebSocket protocol](https://sgl-project.github.io/sglang-omni/basic_usage/tts.html#websocket-speech-streaming)
and needs the Rust runtime and a model with input streaming (Qwen3-TTS, Higgs TTS).

```bash
vox-serve --model bosonai/higgs-tts-3-4b --port 8000
pip install websockets
python websocket_streaming.py --text "Sure, I can help with that. The station is two blocks north."
```

Each `--text` is one segment, ended with `input.commit`; the session ends with `input.done`:

```bash
python websocket_streaming.py --ref-audio voice.wav --ref-text "Transcript of voice.wav" \
    --text "First answer." --text "Second answer, in the same session." --granularity sentence
```

| Option | Meaning |
|---|---|
| `--url` | WebSocket URL (default `ws://localhost:8000/v1/audio/speech/stream`) |
| `--text` | one segment of text; repeat for several |
| `--delay` | seconds between words (default 0.05) |
| `--granularity` | `clause` (default) or `sentence`: the smallest piece of text the model reads |
| `--ref-audio`, `--ref-text` | reference voice for cloning, sent once per session |
| `--language` | language, for models that take one |
| `--output` | output WAV file |

The protocol in brief (see the [API reference](../../docs/api.rst) for all fields):

```text
client: {"type": "session.config", "session": {"response_format": "pcm", "stream_audio": true}}
server: {"type": "session.configured", "session": {..., "sample_rate": 24000}}
client: {"type": "input.text", "text": "Sure, "}  ... more text ...
client: {"type": "input.commit"}
server: {"type": "audio.start", "segment_index": 0, ...}, binary PCM frames, {"type": "audio.done", ...},
        {"type": "input.committed", "segment_index": 0, "segment_sentences": 2, "total_sentences": 2}
client: {"type": "input.done"}
server: {"type": "session.done"}
```
