Development
===========

Local setup
-----------

.. code-block:: bash

   pip install -e .

This also builds the Rust runtime (the ``vox_serve._rt`` extension) when a Rust toolchain and the CUDA
compiler are available (:ref:`runtimes:installing the rust runtime`); otherwise it is skipped and
``vox-serve`` runs the Python runtime. Python changes take effect immediately. After changing anything under
``rust/``, run ``pip install -e .`` again; ``VOX_CUDA_ARCHS=90`` (one GPU architecture) makes that build
faster.

Run the server
--------------

.. code-block:: bash

   vox-serve --model <model-name> --port 8000                    # Rust runtime (Python if unavailable)
   vox-serve --runtime python --model <model-name> --port 8000   # Python runtime

Repository layout
-----------------

- ``vox_serve/model/<name>/``: the models, one program each, used by both runtimes (:doc:`adding-a-model`).
- ``vox_serve/rt/``: what the programs share (base program, graph capture, sampling kernels, prompts).
- ``vox_serve/fusion/``: fused kernels behind the ``--fuse-*`` flags.
- ``vox_serve/launch.py``, ``vox_serve/scheduler/``, ``vox_serve/worker/``: the Python runtime.
- ``vox_serve/cli.py``: the ``vox-serve`` command, which picks the runtime.
- ``rust/``: the Rust runtime (crate ``vox-rt``): ``src/server.rs`` (HTTP), ``src/scheduler.rs`` and
  ``src/deadline.rs`` (policies), ``src/v2.rs`` (asynchronous lanes and SM partitions), ``src/engine.rs``
  (graph launches), ``src/flashinfer.rs`` + ``csrc/fi_plan.cu`` (attention planning), ``csrc/sampling.cu``
  (sampling kernels, also used by the Python runtime), ``src/models/`` (prompt builders),
  ``third_party/flashinfer`` (vendored FlashInfer headers).

Checks
------

Python:

.. code-block:: bash

   ruff format .
   ruff check .

Rust (formatting settings in ``rust/rustfmt.toml``):

.. code-block:: bash

   cd rust
   cargo fmt
   cargo clippy --release --all-targets -- -D warnings
   cargo test --release

The Rust unit tests cover what runs without a GPU: scheduling-policy math, text normalization of the prompt
builders and the feature-pool allocator. Building the crate compiles CUDA sources, so ``nvcc`` is needed even
for these checks.

Continuous integration
----------------------

CI runs on CPU-only GitHub runners:

- ``.github/workflows/ci.yml``: ``ruff check``.
- ``.github/workflows/rust.yml``, with the Rust version pinned in ``RUST_TOOLCHAIN`` (bump it together with
  the fixes new ``clippy`` lints ask for):

  - ``cargo fmt --check``, ``cargo clippy -D warnings`` and ``cargo test``, after installing the CUDA
    compiler and runtime libraries (``.github/scripts/install-cuda-toolkit.sh``; no GPU or driver needed);
  - a wheel with the extension, installed in a clean environment, whose ``vox_serve._rt`` must load and
    parse a command line;
  - a wheel built without CUDA, which must not contain the extension and must fall back to the Python
    runtime.

- ``.github/workflows/docs.yml``: builds and publishes this documentation (pushes to ``main``).

There is no GPU test suite. Validate changes that touch the GPU path by serving with both runtimes:

.. code-block:: bash

   curl -X POST localhost:8000/generate -F "text=Hello world" -F "streaming=true" -o out.wav
   vox-serve --runtime rust --model <model-name> --bench-requests 4 --bench-out /tmp/out --greedy
   python rust/scripts/asr_check.py /tmp/out/bench_*.wav    # Whisper transcripts
   bash rust/scripts/smoke_model.sh <model-name>            # the two steps above

Load benchmarks against a running server are in ``benchmark/``, sample clients in ``examples/``.

Build documentation
-------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   make -C docs html
