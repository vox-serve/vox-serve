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

With the Rust runtime, once an engine has stopped (e.g. after a CUDA error) the endpoint returns 503
with ``{"status": "unhealthy", "failed_engines": [<rank>, ...]}`` until the server is restarted;
requests are routed to the remaining engines, or get 503 when none is left.

WebSocket /v1/audio/speech/stream
---------------------------------

Speech from text that arrives over time (e.g. from a text LLM), with the protocol of
`sglang-omni's WebSocket speech streaming
<https://sgl-project.github.io/sglang-omni/basic_usage/tts.html#websocket-speech-streaming>`_. Rust runtime
only, for models that support input streaming (Qwen3-TTS, Higgs TTS). See the
`example client <https://github.com/vox-serve/vox-serve/tree/main/examples/websocket_streaming>`_.

Client messages (JSON text frames)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``{"type": "session.config", "session": {...}}`` (first): ``voice``, ``language``, ``response_format``
  (``"pcm"``, the default, or ``"wav"``), ``stream_audio`` (default ``false``; ``true`` needs ``"pcm"``),
  ``split_granularity`` (``"clause"``, the default, or ``"sentence"``); and for voice cloning ``ref_audio`` (a
  base64 data URL, ``data:audio/wav;base64,...``, sent once per session) and ``ref_text``. ``model`` is ignored.
- ``{"type": "input.text", "text": "..."}``: more text; send it with its spaces and punctuation.
- ``{"type": "input.commit"}``: ends the current segment. Its remaining text is spoken even without a final
  boundary, and the session stays open for the next segment.
- ``{"type": "input.done"}``: commits, and closes the session once all audio is sent.

Server messages
^^^^^^^^^^^^^^^

- ``session.configured``: the settings, with the output ``sample_rate``.
- Per segment, in order: ``audio.start`` (``segment_index``, ``sample_rate``, ``format``), the audio as binary
  frames, ``audio.done``, and ``input.committed`` (``segment_index``, ``segment_sentences``,
  ``total_sentences``). Audio is 16-bit little-endian mono PCM; with ``stream_audio`` it is sent as it is
  generated, otherwise as one frame per segment (a WAV file with ``"wav"``). An empty commit gets only
  ``input.committed``.
- ``session.done`` after ``input.done``; ``error`` (``message``) for invalid messages or a failed segment. An
  invalid ``session.config`` closes the session.

Each segment is one input-streaming request: the next segment starts generating as soon as its text arrives,
and its audio follows the previous segment's. ``split_granularity`` sets the smallest piece of text the model
reads: a clause, or a sentence (for turn-based models such as Higgs TTS, whole sentences per turn). Closing the
connection cancels its requests.

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
