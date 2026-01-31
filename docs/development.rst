Development
===========

Local setup
-----------

.. code-block:: bash

   pip install -e .

Run the server
--------------

.. code-block:: bash

   python -m vox_serve.launch --model <model-name> --port 8000

Formatting and linting
----------------------

.. code-block:: bash

   ruff format .
   ruff check .

Build documentation
-------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   make -C docs html

Manual validation
-----------------

Send a request using ``curl`` or the scripts in ``examples/``.
