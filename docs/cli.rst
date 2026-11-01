CLI Reference
=============

The CLI entrypoint is ``vox-serve`` (installed via pip; also ``python -m vox_serve``).

Usage
-----

.. code-block:: bash

   vox-serve --model <model-name> --port 8000

Runtime
-------

``--runtime auto|rust|python``
  ``auto`` (default) serves with the Rust runtime, and with the Python runtime when the Rust extension
  is not installed or does not load, or when only the Python runtime takes the arguments (for example
  ``--async-scheduling``). Without the extension, flags only the Rust runtime has are ignored with a
  warning. ``rust`` and ``python`` select one runtime without fallback; ``python -m vox_serve.launch``
  is the Python runtime too.

Common arguments (both runtimes)
--------------------------------

``--model``
  Model name or Hugging Face repository. Default: ``canopylabs/orpheus-3b-0.1-ft``.

``--host`` / ``--port``
  Bind address and port of the HTTP server. Defaults: ``0.0.0.0``, ``8000``.

``--max-batch-size``
  Maximum number of requests decoded together. Default: ``8``.

``--page-size`` / ``--max-num-pages``
  KV-cache page size (default ``64``) and number of pages (default: enough for twice the batch at the
  maximum context).

``--scheduler-type``
  Scheduling policy: ``base`` (default), ``online`` (requests close to an audio underrun first) or
  ``offline`` (throughput first). The Python runtime also has ``input_streaming`` (needed there for the
  input-streaming endpoints; the Rust runtime serves them with every policy and treats the name as
  ``base``), the Rust runtime also ``deadline``. :doc:`runtimes` describes the policies.

``--max-tokens``
  Per-request token limit (prompt included), overriding the model's default.

``--temperature``, ``--top-p``, ``--top-k``, ``--min-p``, ``--repetition-penalty``
  Sampling parameters of every request, overriding the model's defaults.

``--greedy``
  Greedy decoding for every request (testing).

``--detokenize-interval``
  Codec frames per decoded audio chunk (Qwen3-TTS). Default: the model's.

``--cfg-scale``
  Classifier-free guidance scale (Voxtral-TTS). Default: the model's.

``--dp-size``
  Data-parallel replicas, one per GPU ``0..N-1``.

``--enable-disaggregation``
  Run the codec on GPU 1 and the LM on GPU 0.

``--enable-torch-compile``
  Compile the LM forward with ``torch.compile`` before capturing its graphs (slower start-up).

``--fuse-lm``, ``--fuse-layers``, ``--fuse-codec``, ``--fuse-depth``, ``--fuse-talker``, ``--compile-detok``
  Kernel fusion (``vox_serve/fusion``), model-specific: merged q/k/v and gate/up GEMMs (``--fuse-lm``), fused
  decoder layers (``--fuse-layers``: Higgs Audio v3, CosyVoice3), fused codec kernels (``--fuse-codec``:
  Higgs Audio v3, CosyVoice3), Qwen3-TTS depth transformer and talker (``--fuse-depth``, ``--fuse-talker``),
  and ``torch.compile`` of the codec (``--compile-detok``). ``vox-serve --help`` lists the models of each.

``--log-level``
  ``DEBUG``, ``INFO`` (default), ``WARNING`` or ``ERROR``.

Python runtime only
-------------------

``--async-scheduling``
  Schedule the next step while the GPU runs the current one.

``--disable-cuda-graph`` / ``--enable-cuda-graph``
  Run the model's step functions eagerly instead of replaying CUDA graphs (for debugging model code).
  ``--enable-cuda-graph`` is the default.

``--socket-suffix``
  Suffix of the IPC socket paths between the API server and the schedulers, to run several servers on
  one host.

``--enable-nvtx``
  NVTX ranges around the steps, for profiling.

``--repetition-window``, ``--unroll-depth-cuda-graph``
  Accepted for compatibility, without effect: the repetition penalty covers every generated token, and
  the depth transformer is part of each step's graph.

Rust runtime only
-----------------

``--max-context`` / ``--prefill-buckets``
  Maximum sequence length of a request (default ``2048``) and the prefill graph sizes (default
  ``256,512,1024,2048``, capped at the maximum context).

``--max-prefills-per-step``
  Requests prefilled together in one step (default ``1``).

``--max-detok-batch-size``
  Largest codec batch captured as a graph (default: ``--max-batch-size``); bounds codec graph memory.

``--torch-compile-targets``
  Restrict ``--enable-torch-compile`` to these LM functions (e.g. ``decode_head,decode_depth``).

``--codec-gpu`` / ``--codec-sms``
  Run the codec on another GPU (or an SM partition of one shared by several replicas).

``--overlap-detok``, ``--async-engines``, ``--sm-modes``, ``--sm-mode``, ``--urgent-lane``, ``--stream-priorities``
  Run the LM and the codec concurrently: on separate streams, or on SM partitions of the GPU, with a
  high-priority lane for first audio chunks (:ref:`runtimes:running the lm and the codec concurrently`).

``--urgent-max-batch``
  Largest urgent-lane batch (default ``8``).

``--lm-segments whole|depth|layers:<k>``, ``--gate none|depth|layer:<i>``
  Capture Qwen3-TTS steps in segments (before the depth transformer, or in groups of ``k`` layers), and
  start codec batches only once the LM step reaches the named segment. Segments need the asynchronous
  engine (``--async-engines`` or a flag that implies it).

``--dec-batch``, ``--dec-max-batch``, ``--dec-flush-ms``, ``--dec-urgent-ms``
  Codec batching of the asynchronous engine: the ready chunks a batch waits for (default ``0``: none), the
  largest batch without urgent chunks (default ``0``: the largest captured), how close a playback deadline
  flushes a partial batch (default ``50`` ms), and how close makes a chunk urgent (default ``100`` ms).

``--prefill-exclusive``, ``--urgent-exclusive``, ``--slack-urgency``, ``--time-share``
  Asynchronous-engine variants: prefills never overlap codec batches; urgent codec batches run on all SMs
  while the LM waits; first chunks are scheduled by deadline (``--deadline-startup-ms``) instead of always
  urgent; LM steps and codec batches alternate on the whole GPU.

``--deadline-startup-ms``, ``--deadline-lm-step-ms``, ``--deadline-codec-ms``, ``--deadline-prefill-ms``, ``--deadline-max-prefill-deferral-ms``, ``--deadline-max-pending-chunks``
  First-audio target and cost estimates of the ``deadline`` policy (calibrate per model and GPU), the
  longest prefill deferral, and the number of audio chunks waiting for the codec at which a request's LM
  steps pause (``0``, the default: never).

``--bench-requests N`` (with ``--bench-text``, ``--bench-audio``, ``--bench-kwargs``, ``--bench-out``)
  Run N concurrent requests without HTTP, report their timings, optionally write their audio, and exit.

``--trace-dir``
  Write the GPU time of every graph launch and every scheduling decision as JSON lines.

``--prompt-builder python|template``
  Build prompts with the program's ``preprocess`` (or the template builder) instead of the model's Rust
  prompt builder: for bringing up a model, or checking a Rust builder against Python
  (:doc:`adding-a-model`).

``vox-serve --runtime rust --help`` and ``vox-serve --runtime python --help`` list every flag;
``rust/docs/v2_runtime_design.md`` describes the asynchronous engine.
