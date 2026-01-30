CLI Reference
=============

The CLI entrypoint is ``vox-serve`` (installed via pip), which maps to
``python -m vox_serve.launch``.

Usage
-----

.. code-block:: bash

   vox-serve --model <model-name> --port 8000

Common options
--------------

- ``--model``: Model name or path (default: ``canopylabs/orpheus-3b-0.1-ft``)
- ``--scheduler-type``: ``base`` | ``online`` | ``offline``
- ``--async-scheduling``: Enable async scheduling
- ``--host``: Bind address (default: ``0.0.0.0``)
- ``--port``: Port (default: ``8000``)
- ``--max-batch-size``: Max batch size
- ``--max-num-pages``: KV cache pages
- ``--page-size``: KV cache page size

Sampling options
----------------

- ``--top-p``
- ``--top-k``
- ``--min-p``
- ``--temperature``
- ``--max-tokens``
- ``--repetition-penalty``
- ``--repetition-window``
- ``--cfg-scale``
- ``--greedy``

Performance and profiling
-------------------------

- ``--enable-cuda-graph`` / ``--disable-cuda-graph``
- ``--enable-disaggregation``
- ``--dp-size``
- ``--enable-nvtx``
- ``--enable-torch-compile``
- ``--log-level``

IPC tuning
----------

- ``--socket-suffix``: Append a suffix to IPC socket paths to avoid conflicts.
