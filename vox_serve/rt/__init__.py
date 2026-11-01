"""What the models' programs share.

``program.py`` is the base of every model's program (``vox_serve/model/<name>/program.py``): static device
buffers, CUDA-graph capture of each step stage, sampling and token kinds. ``core.py`` builds a program's graphs
(``EngineSpecBuilder``) for either runtime: the Rust runtime (the ``vox_serve._rt`` extension, through
``build_engine``; it re-enters Python only for per-request hooks, ``call_hook``) or the Python runtime, whose
worker (``vox_serve.worker.ProgramWorker``) replays them from its scheduler.
"""

from .core import EngineSpecBuilder, build_engine, call_hook

__all__ = ["EngineSpecBuilder", "build_engine", "call_hook"]
