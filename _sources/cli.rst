CLI Reference
=============

The CLI entrypoint is ``vox-serve`` (installed via pip), which maps to
``python -m vox_serve.launch``.

Usage
-----

.. code-block:: bash

   vox-serve --model <model-name> --port 8000

Arguments
---------

``--model``
  Model name or local path to load for inference.
  Default: ``canopylabs/orpheus-3b-0.1-ft``.

``--scheduler-type``
  Scheduler backend implementation. One of: ``base``, ``online``, ``offline``.

``--async-scheduling``
  Enable async scheduling mode, which overlaps request handling and scheduler work.

``--host``
  Bind address for the HTTP server. Default: ``0.0.0.0``.

``--port``
  TCP port for the HTTP server. Default: ``8000``.

``--max-batch-size``
  Maximum batch size used by the scheduler for inference.

``--max-num-pages``
  Maximum number of KV cache pages for the scheduler backend.

``--page-size``
  Size of each KV cache page (tokens per page).

``--top-p``
  Top-p (nucleus) sampling threshold. When set, tokens are sampled from the smallest set
  whose cumulative probability exceeds this value.

``--top-k``
  Top-k sampling threshold. When set, tokens are sampled from the k most likely candidates.

``--min-p``
  Min-p sampling threshold. Filters out tokens with probability below this value.

``--temperature``
  Sampling temperature to scale logits. Lower is more deterministic; higher is more random.

``--max-tokens``
  Maximum number of tokens to generate per request.

``--repetition-penalty``
  Penalize repeated tokens to reduce loops in generated output.

``--repetition-window``
  Window size for repetition penalty.

``--cfg-scale``
  Classifier-free guidance scale, where higher values strengthen conditioning.

``--greedy``
  Use greedy decoding (disables top-k/top-p/min-p/temperature sampling).

``--enable-cuda-graph``
  Enable CUDA graph optimization for the decode phase.

``--disable-cuda-graph``
  Disable CUDA graph optimization for the decode phase.

``--enable-disaggregation``
  Enable disaggregation mode (requires at least 2 GPUs).

``--dp-size``
  Enable data parallel mode with N replicas (N >= 1). Cannot be combined with
  ``--enable-disaggregation`` and requires N <= available GPUs.

``--enable-nvtx``
  Enable NVTX profiling for performance analysis.

``--enable-torch-compile``
  Enable ``torch.compile`` optimization for model inference.

``--log-level``
  Set log verbosity. One of: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.

``--socket-suffix``
  Append a suffix to IPC socket paths to avoid conflicts when running multiple instances.
