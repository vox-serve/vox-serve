# Higgs Audio v3 TTS

[Higgs Audio v3 TTS](https://huggingface.co/bosonai/higgs-tts-3-4b) (`bosonai/higgs-tts-3-4b`): a Qwen3-4B
backbone over text tokens and the 8 delayed codebooks of the Higgs audio tokenizer (25 frames/s, 24 kHz).

```bash
vox-serve --model bosonai/higgs-tts-3-4b --port 8000
curl -X POST localhost:8000/generate -F "text=Hello world" -F "streaming=true" -o out.wav

# Voice cloning: reference audio and its transcript
curl -X POST localhost:8000/generate -F "text=Hello world" -F "audio=@ref.wav" \
  -F "ref_text=Transcript of the reference clip." -o out.wav
```

Inline control tags (`<|emotion:elation|>`, `<|prosody:pause|>`, `<|sfx:laughter|>`, ...) go in the text; see the
model's `PROMPTING.md`.

### Input Streaming

Text can be sent while audio is generated (e.g. from a text LLM), through the same endpoints as other models
(`/generate/stream/start`, `.../{id}/text`, `.../{id}/audio`, `.../{id}/end`; see the
[example client](https://github.com/vox-serve/vox-serve/tree/main/examples/input_streaming)). It needs the Rust
runtime, and no scheduler flag.

Higgs reads text in turns: the model's multi-turn layout `<|text|> chunk <|audio|> [audio] <|text|> chunk
<|audio|> [audio] ...` in one sequence, so the voice and prosody carry over and earlier turns stay in the
context. The request's first turn starts once about 20 characters have arrived; each later turn takes the text
that arrived while the previous turn's audio was generated, up to its last clause boundary (`. , ! ? ; :`), or up
to a word boundary for a run-on sentence of 10 or more words. A turn boundary costs a short pause, like a comma.
When the text arrives more slowly than it is spoken, the audio waits for it.

Every turn stays in the KV cache: a stream ends (`max_context_reached`) once text and audio fill
`--max-context` (2048 rows by default, about 70 seconds of speech). Raise it for long streams; the model was
trained with 8192.
