Serving Runtimes
================

VoxServe has two serving runtimes. Both run the same model implementations (one *program* per model,
see :doc:`adding-a-model`), accept the same core arguments and serve the same HTTP API (:doc:`api`).

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Rust runtime (default)
     - Python runtime
   * - Selected with
     - ``vox-serve`` (or ``--runtime rust``)
     - ``--runtime python`` or ``python -m vox_serve.launch``; also the automatic fallback (below)
   * - Serving loop
     - Rust: an HTTP server and one scheduler thread per GPU, with no Python on the per-step path
     - Python: a FastAPI server plus one scheduler process per GPU, connected over ZMQ
   * - ``--scheduler-type``
     - ``base``, ``online``, ``offline``, ``deadline``
     - ``base``, ``online``, ``offline``, ``input_streaming``
   * - Overlap
     - LM and codec run concurrently: on separate streams, on SM partitions of the GPU, or with an
       urgent lane for first audio chunks
     - ``--async-scheduling`` schedules the next step while the GPU runs the current one
   * - Several GPUs
     - ``--dp-size``, ``--enable-disaggregation``, ``--codec-gpu`` / ``--codec-sms``
     - ``--dp-size``, ``--enable-disaggregation``
   * - Debugging
     - ``--bench-requests`` (requests without HTTP), ``--trace-dir``, ``--prompt-builder python``
     - ``--disable-cuda-graph`` (eager execution), ``--enable-nvtx``
   * - Needs
     - the ``vox_serve._rt`` extension, compiled at install time (Rust toolchain + CUDA compiler)
     - nothing beyond the Python package

Choosing a runtime
------------------

``vox-serve --runtime auto`` (the default) serves with the Rust runtime and switches to the Python runtime
when the Rust runtime cannot serve the command line:

- the extension is not installed (its build is optional and is skipped without a Rust toolchain or ``nvcc``)
  or does not load;
- the extension was built against a different FlashInfer version than the installed ``flashinfer-python``;
- an argument exists only in the Python runtime (``--async-scheduling``, ``--disable-cuda-graph``, ...).

The reason is printed on stderr. When the Python runtime serves because the extension is missing, arguments
that only the Rust runtime has are dropped with a warning. ``--runtime rust`` and ``--runtime python`` select
one runtime and never fall back, and an error after start-up never switches runtimes.

.. code-block:: bash

   vox-serve --model qwen3-tts                        # Rust runtime if available, else Python
   vox-serve --runtime rust --model qwen3-tts         # Rust runtime or an error explaining why not
   vox-serve --runtime python --model qwen3-tts       # Python runtime
   vox-serve --runtime rust --help                    # every Rust-runtime flag
   vox-serve --runtime python --help                  # every Python-runtime flag

Installing the Rust runtime
---------------------------

The extension ``vox_serve._rt`` is compiled by ``pip install`` from the ``rust/`` crate (setuptools-rust)
when these are available:

- a Rust toolchain (``cargo``; CI uses Rust 1.90, `rustup <https://rustup.rs>`_ installs it);
- the CUDA 12 toolkit with ``nvcc``, found through ``CUDA_HOME`` (default ``/usr/local/cuda-12``).

.. code-block:: bash

   git clone https://github.com/vox-serve/vox-serve.git && cd vox-serve
   pip install -e .                       # builds the extension (a few minutes), or skips it with a warning
   vox-serve --runtime rust --help        # fails with the reason if the extension is unavailable

Without these tools the installation still succeeds, and ``vox-serve`` serves with the Python runtime.
Python changes take effect immediately in an editable install; after changing Rust or CUDA sources, run
``pip install -e .`` again.

Build options (environment variables):

- ``VOX_CUDA_ARCHS``: comma-separated SM architectures of the sampling kernels, e.g. ``90`` for H100 only
  (default: ``80,86,89,90,100,120``, limited to those ``nvcc`` supports). One architecture builds faster.
- ``FLASHINFER_INCLUDE``: FlashInfer headers to plan attention with, instead of the copy vendored in
  ``rust/third_party/flashinfer``. Their version must equal the ``flashinfer-python`` package
  (``rust/third_party/flashinfer/update.sh`` re-vendors them).

The extension links the CUDA runtime (``libcudart.so.12``) dynamically and uses the copy PyTorch loads, so
the Rust runtime and PyTorch share CUDA graphs and streams. A wheel must never bundle its own
``libcudart`` (``auditwheel repair --exclude libcudart.so.12``).

How the Rust runtime works
--------------------------

1. **Start-up (Python).** ``vox-serve`` loads ``vox_serve._rt`` and hands it the command line. For every
   GPU replica, the extension calls ``vox_serve.rt.build_engine`` in the same interpreter: it builds the
   model's program, which loads the weights and captures every stage of a step as CUDA graphs over static
   buffers. The result is an *engine spec* (JSON): the device addresses and field layouts of the input and
   output arenas, a graph handle per stage and batch size, the attention configuration, and the program's
   ``meta`` (model constants and prompt-builder inputs).
2. **Serving (Rust).** An HTTP server (``rust/src/server.rs``) passes requests to one scheduler thread per
   replica (``rust/src/scheduler.rs``). Each step, the scheduler selects audio chunks to decode and a
   batch of requests to prefill or decode, writes the step's inputs into pinned host memory, copies them to
   the GPU in one transfer, plans paged-KV attention with FlashInfer's planner (compiled into the
   extension), launches the graphs (``rust/src/engine.rs``) and copies the outputs back. The graphs have
   already sampled the next tokens and classified them (``out_kind``: audio, stop or other), so the
   scheduler never interprets token ids; decoded audio chunks are streamed to the clients.
3. **Per-request Python.** Python only runs again to build prompts that need it: the ``python`` prompt
   builder (the program's ``preprocess``) and hooks such as reference-audio encoding or phonemization.
   Prompts of most models are built in Rust (``rust/src/models``).

Scheduling policies
^^^^^^^^^^^^^^^^^^^

- ``base`` (default): new requests are prefilled first, every running request decodes, and audio chunks
  are decoded as soon as they are ready.
- ``online``: a streaming request is *pressing* until its first audio chunk and again from a second
  before its client starts playing the last chunk it has received; a non-streaming request is pressing
  once its tokens are all generated. Pressing requests decode first, and audio is decoded only in steps that have a
  pressing chunk, which other ready chunks join while the batch has room.
- ``offline``: audio is decoded only while no admitted request is generating tokens; for batch jobs, not
  streaming.
- ``deadline``: audio chunks are decoded in order of their playback deadlines, and new requests by their
  first-audio deadline (``--deadline-startup-ms``); prefills are deferred, for at most
  ``--deadline-max-prefill-deferral-ms``, while running streams are about to underrun. Calibrate the
  ``--deadline-lm-step-ms`` / ``--deadline-codec-ms`` / ``--deadline-prefill-ms`` cost estimates per model
  and GPU.

Input-streaming requests (``/generate/stream/...``) work with every policy; ``input_streaming`` is accepted
as a name of ``base``.

Running the LM and the codec concurrently
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default a step runs the codec, then the LM, on one stream. These flags overlap them:

- ``--overlap-detok``: the codec runs on its own stream, asynchronously to the LM steps.
- ``--async-engines``: the LM and codec lanes each keep one batch in flight and are collected by their
  completion events.
- ``--sm-modes 24,48`` and ``--sm-mode``: SM partitions of the GPU (green contexts), given as the codec's SM
  counts. Every decode and codec graph is captured once per partition, so switching partitions only
  selects other graphs. For example, ``--sm-modes 32 --sm-mode 1`` gives the codec 32 SMs and the LM the
  rest; ``--sm-mode adaptive`` picks the partition from latencies measured while serving.
- ``--urgent-lane``: a third, high-priority codec lane on all SMs for first chunks and chunks close to
  their playback deadline (Qwen3-TTS, Orpheus, Zonos, CSM, CosyVoice3).
- ``--lm-segments depth`` / ``layers:<k>`` with ``--gate``: Qwen3-TTS steps captured in segments, so codec
  batches can start at a segment boundary.

The best partition and flags depend on the model and the GPU; measure with a load benchmark
(``benchmark/``). ``rust/docs/v2_runtime_design.md`` describes the asynchronous engine and every flag.

Several GPUs
^^^^^^^^^^^^

- ``--dp-size N``: one replica (engine and scheduler thread) per GPU ``0..N-1``.
- ``--enable-disaggregation``: the LM on GPU 0 and the codec on GPU 1.
- ``--codec-gpu G`` (with ``--codec-sms N``): the codec of every replica on GPU ``G``, optionally on an
  ``N``-SM partition of it that its replicas share.

Benchmarking without HTTP
^^^^^^^^^^^^^^^^^^^^^^^^^

``--bench-requests N`` runs ``N`` concurrent requests through the scheduler, logs each request's time to
first audio, latency, audio length and finish reason and the overall real-time factor, and exits.
``--bench-out DIR`` writes their audio, ``--greedy`` makes runs repeatable, and ``--bench-text`` /
``--bench-audio`` / ``--bench-kwargs`` set the request.

.. code-block:: bash

   vox-serve --model qwen3-tts --bench-requests 8 --bench-out /tmp/qwen3 --greedy

``--trace-dir DIR`` writes the GPU time of every graph launch and every scheduling decision as JSON lines.

The Python runtime
------------------

``vox_serve/launch.py`` starts a FastAPI server and one scheduler process per GPU
(``vox_serve/scheduler_entry.py``), which talk over ZMQ. ``Scheduler._step`` (``vox_serve/scheduler/base.py``)
admits and retires requests, selects the chunks to decode and the requests to run, and calls the worker;
the policies are subclasses (``online.py``, ``offline.py``, ``input_streaming.py``, and
``disaggregation.py`` for ``--enable-disaggregation``). The worker (``vox_serve/worker/program_worker.py``)
builds the same program as the Rust runtime, fills its input arenas from the requests, replays its graphs
(or, with ``--disable-cuda-graph``, runs the captured functions eagerly, which is useful when debugging
model code) and turns the outputs back into request state. Without the extension, the sampling kernels
come from a PyTorch fallback (``vox_serve/rt/kernels.py``).
