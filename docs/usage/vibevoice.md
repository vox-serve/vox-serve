# VibeVoice-Realtime

VibeVoice-Realtime-0.5B is a real-time, single-speaker text-to-speech model developed by Microsoft. A
Qwen2.5-0.5B is split in two: the lower layers encode the text, the upper layers generate speech as
continuous 64-d acoustic latents at 7.5 Hz through a diffusion head, and a causal acoustic decoder turns
them into 24 kHz audio. Text is consumed in windows of 5 tokens, each followed by 6 speech latents (0.8 s).

The model is primarily trained on English. Voices are not cloned from reference audio: the model ships
voice prompts as precomputed caches (the `.pt` files of the
[VibeVoice repository](https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model)),
which VoxServe downloads on first use.

## Quickstart

Start the server:

```bash
vox-serve --model microsoft/VibeVoice-Realtime-0.5B --port 8000
```

Generate speech:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "text=Hello, this is VibeVoice speaking!" \
  -F "voice=en-Emma_woman" \
  -F "streaming=true" \
  -o output.wav
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize (at most `max_text_tokens` tokens, 1024 by default) |
| `voice` (or `speaker`) | string | `en-Carter_man` | A loaded voice: its full name, or the speaker part (`carter`, `emma`, ...) |
| `streaming` | boolean | `true` | Stream the audio as it is generated |

Reference audio (`audio`) is rejected.

## Voices

The 25 voices of the repository are loaded: `en-Carter_man`, `en-Davis_man`, `en-Emma_woman`,
`en-Frank_man`, `en-Grace_woman`, `en-Mike_man`, `in-Samuel_man`, and two experimental voices each for German
(`de-`), French (`fr-`), Italian (`it-`), Japanese (`jp-`), Korean (`kr-`), Dutch (`nl-`), Polish (`pl-`),
Portuguese (`pt-`) and Spanish (`sp-`), e.g. `fr-Spk1_woman`. An unknown voice gets a 400 response listing them.

## Limits and options

- A request holds the voice prompt (150-510 positions), its text tokens and one position per latent frame
  (7.5 per second of audio) within the context: `--max-context` (Rust runtime, default 2048, up to the
  model's 8192) bounds the audio length, about 2.5 minutes at the default.
- `--detokenize-interval` sets the latent frames per decoded chunk (default 4, i.e. 533 ms; at most 10).
- The diffusion head runs 5 DPM-Solver steps with a classifier-free guidance scale of 1.5, as the reference
  demo does (program config keys `ddpm_steps` and `cfg_scale`; `voices` and `voices_dir` select the voice
  prompts, `max_text_tokens` bounds the text, 1024 by default).

## Notes

- Each LM step advances a request by one position: a window's text tokens go in one per step rather than
  five at once (the same sequence of rows), and speech steps run the diffusion head inside the step's
  CUDA graph, with the classifier-free-guidance condition from a second, speech-only context.
- The acoustic decoder keeps each convolution's history per request, so decoding in chunks computes the
  reference's frame-by-frame streaming output (up to bf16 rounding).
