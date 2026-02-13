"""
Pool module for hardware resource management.

Pools manage compute resources (GPU memory, CUDA graphs, etc.) and execute
strategies asynchronously. They are agnostic about inference logic.
"""

from .base import Pool
from .cuda_graph_pool import CudaGraphPool

__all__ = [
    "Pool",
    "CudaGraphPool",
]
