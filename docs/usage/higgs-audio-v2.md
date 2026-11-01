# Higgs Audio v2

Higgs Audio v2 is a text-to-speech model developed by Boson AI: a Llama-3.2-3B backbone whose layers have a
separate FFN path for audio tokens, generating 8 delayed codebooks of the Higgs Audio tokenizer (25 frames/s,
24 kHz). It clones a voice from a reference clip and its transcript, and takes a scene description.

## Quickstart

Start the server:

```bash
vox-serve --model bosonai/higgs-audio-v2-generation-3B-base --port 8000
```

The weights (`bosonai/higgs-audio-v2-generation-3B-base`, about 11 GB) and the audio tokenizer
(`bosonai/higgs-audio-v2-tokenizer`) are downloaded from Hugging Face on first use.

Generate speech:

```python
import requests

response = requests.post("http://localhost:8000/generate", data={
    "text": "Hello, this is Higgs Audio speaking!",
    "streaming": "false",
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Voice Cloning

Send a reference clip together with its transcript (`ref_text`, required with `audio`):

```python
with open("reference.wav", "rb") as audio:
    response = requests.post(
        "http://localhost:8000/generate",
        data={
            "text": "This sentence is spoken in the reference voice.",
            "ref_text": "Transcript of the reference clip.",
            "streaming": "false",
        },
        files={"audio": ("reference.wav", audio, "audio/wav")},
    )
```

The reference is placed in the prompt as an earlier turn of the conversation, as in Boson AI's examples.
Clips are encoded once and cached (16 clips).

## Scene Description

`instruct` replaces the default scene description, `Audio is recorded from a quiet room.`:

```python
requests.post("http://localhost:8000/generate", data={
    "text": "Two coffees and a croissant, please.",
    "instruct": "Audio is recorded in a busy cafe with background chatter.",
})
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize (normalized as in Boson AI's example: half-width punctuation, no parentheses, a final period) |
| `audio` | file | none | Reference clip for voice cloning (up to 100 s) |
| `ref_text` | string | none | Transcript of the reference clip (required with `audio`) |
| `instruct` | string | `Audio is recorded from a quiet room.` | Scene description |
| `streaming` | boolean | `true` | Stream the response |

## Sampling

The defaults follow the model's `generation_config.json`: temperature 1.0, top-k 50, top-p 0.95, and
repetition-aware sampling (a code that already occurs twice among its codebook's last 7 frames is redrawn
from the untempered distribution). `--temperature`, `--top-k` and `--top-p` override the first three.

## Notes

- Both runtimes serve the model; prompts are built in Python (`"prompt_builder": "python"`).
- `--fuse-lm` merges the q/k/v and gate/up projections and `--fuse-codec` fuses the DAC codec kernels;
  `--fuse-layers` does not apply (the fused decoder stack has no dual FFN path).
- `--urgent-lane` is supported (the codec keeps no state between chunks).
- The weights are under the Boson Higgs Audio 2 Community License, based on the Meta Llama 3 Community
  License (see the model's `LICENSE` on Hugging Face).
