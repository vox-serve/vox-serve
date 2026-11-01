# XTTS-v2

[XTTS-v2](https://huggingface.co/coqui/XTTS-v2) is Coqui's multilingual text-to-speech model: a GPT-2 (30 layers) that
generates speech codes conditioned on a voice, and a HiFi-GAN vocoder that turns the GPT's hidden states into 24 kHz
audio. It speaks 17 languages and clones a voice from a few seconds of reference audio; 58 voices come with the
checkpoint.

The model weights are licensed under the [Coqui Public Model License](https://coqui.ai/cpml) (non-commercial use only).

## Quickstart

```bash
vox-serve --model coqui/XTTS-v2 --port 8000   # or --model xtts
```

```bash
curl -X POST localhost:8000/generate -F "text=Hello, this is XTTS speaking." -F "streaming=true" -o output.wav
```

## Request Options

| Field | Default | Description |
|-------|---------|-------------|
| `text` | required | Text to synthesize (at most 401 tokens, roughly 250 characters of English; split longer texts) |
| `language` | `en` | `en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh-cn`, `hu`, `ko`, `ja`, `hi` |
| `voice` (or `speaker`) | `Claribel Dervla` | A built-in voice (the names in the checkpoint's `speakers_xtts.pth`, listed in the server's start-up log) |
| `audio` | none | A reference clip to clone (the first 30 s are used); overrides `voice` |
| `streaming` | `true` | Stream the audio as it is generated |

```python
import requests

# A built-in voice, in Spanish.
r = requests.post("http://localhost:8000/generate",
                  data={"text": "Hola, ¿cómo estás?", "language": "es", "voice": "Ana Florence", "streaming": "false"})
open("output.wav", "wb").write(r.content)

# Voice cloning from a reference clip.
with open("reference.wav", "rb") as f:
    r = requests.post("http://localhost:8000/generate",
                      data={"text": "This voice is cloned.", "streaming": "false"},
                      files={"audio": ("reference.wav", f, "audio/wav")})
```

Text normalization reads numbers with [`num2words`](https://pypi.org/project/num2words/) (a dependency). Some
languages need optional packages: `pypinyin` (Chinese), `cutlet` (Japanese) and `ko-speech-tools` (Korean). Digits in
Chinese text are not converted to words.

## Sampling

The defaults are those of Coqui's `inference`: temperature 0.75, top-k 50, top-p 0.85 and a repetition penalty of 10
(the server's `--temperature`, `--top-k`, `--top-p` and `--repetition-penalty` override them). Generation stops at the
model's stop code or after 602 codes (about 28 s of audio).

## Implementation Notes

- The GPT runs in float16 (FlashInfer attention has no float32 kernels, and float16 keeps more precision than bfloat16
  over its 30 layers); the final norm and code head, the conditioning encoder, the speaker encoder and the vocoder run
  in float32, as in Coqui's inference.
- The vocoder decodes windows of 24 latents (4 of context, 16 emitted, 4 of look-ahead), aligned to the global
  mel-frame grid of its 22.05 to 24 kHz resampling and zero-padded after the request's end, so the streamed audio equals
  a whole-sequence decode. The first chunk holds 8 latents (about 0.37 s of audio) and is sent after 12 generated codes.
- Reference clips are encoded when the request is prepared (the 16 most recent clips are cached), on the server's
  prompt path: many concurrent requests with new reference clips queue behind that encoding.
