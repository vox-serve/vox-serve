Architecture
============

High-level components
---------------------

VoxServe is organized as three cooperating roles (mirroring the paper):

- **API server:** FastAPI layer that accepts requests and streams audio responses.
- **Scheduler:** orchestrates request lifecycles and decides which step to run next.
- **Worker/model runtime:** executes model steps on GPU and runs the detokenizer.

Repository map
--------------

- ``vox_serve/launch.py``: API server entry point, CLI argument parsing.
- ``vox_serve/scheduler/``: scheduling policies and request orchestration.
- ``vox_serve/worker/``: GPU execution and pipeline logic.
- ``vox_serve/model/``: model wrappers and unified interface implementations.
- ``vox_serve/tokenizer/`` and ``vox_serve/encoder/``: audio/text tokenization and encoding utilities.
- ``vox_serve/watermarker/``: optional watermarking pipeline.
- ``examples/``: sample clients and usage patterns.
- ``benchmark/``: performance measurement scripts.

Execution flow (simplified)
---------------------------

1. The API server receives a request and enqueues it.
2. The scheduler decides which step to execute (LM forward, sampling, detokenizer).
3. The worker executes that step on GPU and returns the chunk.
4. The API server streams the chunk to the client.
