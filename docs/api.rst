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

- Streaming mode: ``audio/wav`` with chunked transfer.
- Non-streaming mode: ``audio/wav`` file download.

Errors are JSON objects ``{"detail": "<reason>"}``. A request with input the model does not accept, such as
an unknown ``voice`` or ``speaker``, or audio for a model without audio input, gets ``400`` with the reason,
also in streaming mode: the streaming response starts only once the request's prompt is built. A busy
server answers ``429``; server errors and timeouts are ``500``.

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
