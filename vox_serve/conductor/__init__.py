"""
Conductor module for distributed inference coordination.

Conductors orchestrate request lifecycle and coordinate strategy
execution across one or more Pools.
"""

from .base import Conductor
from .disaggregation import DisaggregatedConductor
from .offline import OfflineConductor
from .online import OnlineConductor

# Registry for conductor types
CONDUCTOR_REGISTRY = {
    "base": Conductor,
    "online": OnlineConductor,
    "offline": OfflineConductor,
    "disaggregated": DisaggregatedConductor,
}


def register_conductor(name: str, conductor_class: type):
    """Register a new conductor type."""
    CONDUCTOR_REGISTRY[name] = conductor_class


def load_conductor(conductor_type: str, **kwargs) -> Conductor:
    """
    Load a conductor by type.

    Args:
        conductor_type: Type of conductor to load
        **kwargs: Arguments to pass to conductor constructor

    Returns:
        Conductor instance
    """
    if conductor_type not in CONDUCTOR_REGISTRY:
        raise ValueError(
            f"Unknown conductor type: {conductor_type}. "
            f"Available types: {list(CONDUCTOR_REGISTRY.keys())}"
        )

    return CONDUCTOR_REGISTRY[conductor_type](**kwargs)


def list_supported_conductors():
    """List all supported conductor types."""
    return list(CONDUCTOR_REGISTRY.keys())


__all__ = [
    "Conductor",
    "OnlineConductor",
    "OfflineConductor",
    "DisaggregatedConductor",
    "load_conductor",
    "register_conductor",
    "list_supported_conductors",
    "CONDUCTOR_REGISTRY",
]
