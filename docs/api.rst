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
- ``audio`` (file, optional): Input audio for STS-capable models.
- ``streaming`` (bool, optional, default ``true``):
  - ``true`` streams WAV chunks as they are produced.
  - ``false`` returns a single WAV file after completion.

Response
^^^^^^^^

- Streaming mode: ``audio/wav`` with chunked transfer.
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
