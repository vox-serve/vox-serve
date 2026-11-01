API Reference
=============

Base URL
--------

The server listens on ``http://<host>:<port>``.

POST /generate
--------------

Generate speech from text (and optional audio) and return a WAV response.

Parameters (multipart form)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``text`` (string, required): Input prompt.
- ``audio`` (file, optional): Input audio for STS-capable models. The Rust runtime accepts request
  bodies up to 100 MiB.
- ``streaming`` (bool, optional, default ``true``):
  - ``true`` streams WAV chunks as they are produced.
  - ``false`` returns a single WAV file after completion.

Response
^^^^^^^^

- Streaming mode: ``audio/wav`` with chunked transfer. With the Rust runtime, a request that fails
  after streaming has started (an error, the timeout, or its engine stopping) aborts the connection
  instead of ending the body cleanly, so truncated audio is never reported as complete.
- Non-streaming mode: ``audio/wav`` file download.

Examples
^^^^^^^^

.. code-block:: bash

   curl -X POST "http://localhost:8000/generate" \
     -F "text=Hello world" \
     -F "streaming=true" \
     -o output.wav

.. code-block:: bash

   curl -X POST "http://localhost:8000/generate" \
     -F "text=Hello world" \
     -F "audio=@input.wav" \
     -F "streaming=true" \
     -o output.wav

GET /health
-----------

Health check endpoint.

Response
^^^^^^^^

JSON payload:

.. code-block:: json

   {"status": "healthy"}

With the Rust runtime, once an engine has stopped (e.g. after a CUDA error) the endpoint returns 503
with ``{"status": "unhealthy", "failed_engines": [<rank>, ...]}`` until the server is restarted;
requests are routed to the remaining engines, or get 503 when none is left.

Python API
==========

Package overview
----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   vox_serve

Key modules
-----------

.. automodule:: vox_serve.cli
   :members:

.. automodule:: vox_serve.launch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vox_serve.rt.program
   :members: LMProgram
   :show-inheritance:
