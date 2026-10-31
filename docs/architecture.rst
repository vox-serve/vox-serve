Architecture
============

High-level components
---------------------

Every model is implemented once, as a *program*: the GPU stages of a decoding step (backbone, depth
transformer, sampling, codec) captured as CUDA graphs over static buffers, plus its prompt building and
model constants. Two runtimes serve programs through the same HTTP API and command line (``vox-serve``):

- **Rust runtime** (default, the ``vox_serve._rt`` extension built from ``rust/``): HTTP server,
  scheduling policies (``base``, ``online``, ``offline``, ``deadline``), KV-cache paging, attention
  planning and graph launches, with no Python on the per-step path. It overlaps the LM and the codec
  on separate streams or SM partitions of the GPU and spreads replicas over several GPUs.
- **Python runtime**: a FastAPI server (``vox_serve/launch.py``) and one scheduler process per GPU,
  with the ``base``, ``online``, ``offline``, ``input_streaming`` and ``disaggregation`` policies and
  optional asynchronous scheduling; its worker replays the programs' graphs. ``vox-serve`` uses it when
  the extension is not installed (its build is optional: it needs a Rust toolchain and nvcc) or does
  not load, and for arguments only it takes; ``--runtime python`` selects it explicitly.

:doc:`runtimes` compares the two and describes how each one runs a step; :doc:`adding-a-model` describes the
program interface.

Repository map
--------------

- ``vox_serve/cli.py``: the ``vox-serve`` command (picks the runtime).
- ``vox_serve/launch.py``, ``vox_serve/scheduler/``, ``vox_serve/worker/``: the Python runtime (API server,
  scheduling policies, and ``ProgramWorker``, which fills a program's inputs and replays its graphs).
- ``vox_serve/model/<name>/``: one package per model. ``program.py`` is the implementation both runtimes
  run; ``model.py`` holds the PyTorch modules and weight loading it builds on. ``model/__init__.py`` maps
  model names to programs.
- ``vox_serve/rt/``: what programs share: ``program.py`` (the base program: buffers, graph capture,
  sampling, token kinds), ``core.py`` (arenas, attention, graph capture), ``kernels.py`` (sampling
  kernels, with a torch fallback) and ``prompt.py`` (prompts, model constants).
- ``vox_serve/fusion/``: fused kernels the programs switch on with ``--fuse-*`` flags (both runtimes).
- ``rust/``: the Rust runtime (``src/scheduler.rs``, ``src/server.rs``, ``src/engine.rs``) and the
  models' prompt builders in Rust (``src/models``), which must produce the same rows as the programs'
  ``preprocess``.
- ``vox_serve/tokenizer/`` and ``vox_serve/encoder/``: audio codecs and encoders.
- ``vox_serve/watermarker/``: optional watermarking.
- ``examples/``: sample clients. ``benchmark/``: load benchmarks.

Execution flow (simplified)
---------------------------

1. Start-up: the program loads the model and captures every step stage as CUDA graphs.
2. A request's prompt is built by ``Program.preprocess`` (or by the model's Rust prompt builder).
3. Each step, the scheduler picks the requests to prefill or decode and the audio chunks to decode,
   fills the step's inputs, plans attention and replays the graphs.
4. The graphs sample the next tokens and classify them (audio, stop, other); decoded audio chunks are
   streamed to the client.
