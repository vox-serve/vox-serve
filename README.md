# VoxServe: a serving system for SpeechLMs

**⚠️ This project is under heavy construction — interfaces are not stable, and performance tuning is in progress.**

VoxServe is a serving system for Speech Language Models (SpeechLMs). VoxServe provides low-latency & high-throughput inference for language models trained for speech tokens, specifically text-to-speech (TTS) and speech-to-speech (STS) models.

### Usage

Start the inference server with `launch.py`:

```bash
cd vox-serve
python -m vox-serve.launch --model <model-name> --port <port-number>
```

And call the server like this:

```bash
# Generate audio from text
curl -X POST "http://localhost:<port-number>/generate" -F "text=Hello world" -F "streaming=true" -o output.wav
```

We currently support the following TTS and STS models:

- `csm`: [CSM-1B](https://huggingface.co/sesame/csm-1b)
- `orpheus`: [Orpheus-3B](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)
- `zonos`: [Zonos-v0.1](https://huggingface.co/Zyphra/Zonos-v0.1-transformer)
- `glm`: [GLM-4-Voice-9B](https://huggingface.co/zai-org/glm-4-voice-9b)

And we are actively working on expanding the support.

`./examples` folder has more example usage.
