"""MAPS: model-agnostic predictability scoring for time series."""

from .core import (
    compute_predictability_functional,
    local_direction_coherence,
    maps_rolling_windows,
)

__all__ = [
    "compute_predictability_functional",
    "local_direction_coherence",
    "maps_rolling_windows",
]

__version__ = "1.0.0"

