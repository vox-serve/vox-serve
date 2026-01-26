from typing import Dict, Type

from .base import Scheduler
from .disaggregation import DisaggregationScheduler
from .offline import OfflineScheduler
from .online import OnlineScheduler

# Registry mapping scheduler type names to scheduler classes
SCHEDULER_REGISTRY: Dict[str, Type[Scheduler]] = {
    "base": Scheduler,
    "online": OnlineScheduler,
    "offline": OfflineScheduler,
    "disaggregation": DisaggregationScheduler,
}


def load_scheduler(
    scheduler_type: str,
    **kwargs
) -> Scheduler:
    """
    Load a scheduler instance based on the scheduler type.

    Args:
        scheduler_type: Type of scheduler to load ("base", "online", or "offline")
        **kwargs: Arguments to pass to the scheduler constructor

    Returns:
        Loaded scheduler instance

    Raises:
        ValueError: If scheduler type is not supported
    """
    scheduler_type_lower = scheduler_type.lower()

    if scheduler_type_lower not in SCHEDULER_REGISTRY:
        available_types = list(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unsupported scheduler type '{scheduler_type}'. Available types: {available_types}")

    scheduler_class = SCHEDULER_REGISTRY[scheduler_type_lower]
    return scheduler_class(**kwargs)


def register_scheduler(scheduler_type: str, scheduler_class: Type[Scheduler]) -> None:
    """
    Register a new scheduler class for a given type.

    Args:
        scheduler_type: Type name for the scheduler
        scheduler_class: Scheduler class to register
    """
    SCHEDULER_REGISTRY[scheduler_type.lower()] = scheduler_class


def list_supported_schedulers() -> Dict[str, Type[Scheduler]]:
    """
    Get a dictionary of all supported scheduler types and their classes.

    Returns:
        Dictionary mapping scheduler types to scheduler classes
    """
    return SCHEDULER_REGISTRY.copy()
