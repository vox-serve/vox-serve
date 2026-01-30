Core Concepts (from the paper)
==============================

This page distills the key ideas from ``paper.pdf`` into a practical overview of how VoxServe is designed
and what it optimizes for.

SpeechLM inference pipeline
---------------------------

Modern SpeechLMs usually combine:

- An LLM backbone that autoregressively emits *audio tokens*.
- A detokenizer (audio decoder) that reconstructs waveform chunks from those tokens.
- Optionally, a speech encoder for audio-conditioned tasks.

Many tokenizers use *multi-codebook* representations, where each audio segment is mapped to multiple
parallel token streams. This increases model diversity and makes serving harder to standardize.

Streaming metrics
-----------------

Streaming speech serving needs different metrics than text LLMs:

- **Time-To-First-Audio (TTFA):** time from request arrival to the first playable audio chunk.
- **Streaming viability:** once playback begins, each subsequent chunk must arrive before the previous
  chunk finishes playing. If the system falls behind, the stream stutters.

VoxServe design goals
---------------------

The paper focuses on two system goals:

1. Support diverse SpeechLM architectures under a common interface.
2. Provide high streaming performance (low TTFA and sustained viability).

Unified model interface
-----------------------

VoxServe abstracts SpeechLM execution into a consistent set of stages, which lets the scheduler and
worker optimize without model-specific rewrites:

- **Preprocess:** prompt formatting, tokenization, optional audio encoding, and request metadata setup.
- **LM forward:** the backbone LLM step over text/audio token tensors (plus optional features/masks).
- **Sampling:** converts logits to next-token decisions (temperature, top-k/top-p, repetition penalty).
- **Postprocess:** detokenizer step that turns tokens into waveform chunks.
- **Optional depth-wise decode:** used by some models that sample codebooks in sequence.

This interface also standardizes tensor shapes so GPU optimizations (e.g., CUDA graphs) can be reused
across different models.

Streaming-aware scheduling
--------------------------

The scheduler separates each request into two phases:

- **Startup phase:** prioritize getting the first audio chunk out quickly (TTFA-critical).
- **Steady-state phase:** keep streaming chunks on time (viability-critical).

Requests in steady-state are prioritized by their risk of missing a playback deadline. The scheduler
exploits slack when streams are comfortably ahead, and allocates resources to more urgent requests.

Asynchronous pipeline
---------------------

VoxServe uses asynchronous execution to overlap LM forward, sampling, and detokenization across
requests. This improves GPU utilization and reduces end-to-end latency under load.

Reported results
----------------

The paper reports significant throughput gains versus model-specific serving baselines, while keeping
latency comparable and preserving streaming viability.
