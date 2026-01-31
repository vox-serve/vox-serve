Quickstart
==========

Prerequisites
-------------

- Python 3.12+
- A CUDA-capable GPU for most models (CPU-only runs are not guaranteed)

Install
-------

From PyPI:

.. code-block:: bash

   pip install vox-serve

From source:

.. code-block:: bash

   git clone https://github.com/vox-serve/vox-serve.git
   cd vox-serve
   pip install -e .

Run the server
--------------

.. code-block:: bash

   vox-serve --model <model-name> --port 8000

Or with the module entrypoint:

.. code-block:: bash

   python -m vox_serve.launch --model <model-name> --port 8000

Send a request
--------------

Text-to-speech (streaming):

.. code-block:: bash

   curl -X POST "http://localhost:8000/generate" \
     -F "text=Hello world" \
     -F "streaming=true" \
     -o output.wav

Speech-to-speech (when the model supports audio input):

.. code-block:: bash

   curl -X POST "http://localhost:8000/generate" \
     -F "text=Hello world" \
     -F "audio=@input.wav" \
     -F "streaming=true" \
     -o output.wav

Health check:

.. code-block:: bash

   curl -X GET "http://localhost:8000/health"

Notes
-----

- ``streaming=true`` returns a streaming WAV response with audio chunks.
- ``streaming=false`` returns a single WAV file after the request completes.
- When using data-parallel or disaggregation modes, ensure you have enough GPUs.
