"""Stable public API for fitpredict.

Import user-facing entry points from this module:

    from fitpredict import fit, predict, load_config, resolve_config

Subpackages contain implementation modules and lower-level extension helpers.
They may be useful for contributors, but the root package is the public API
that application code should depend on.
"""

from fitpredict.config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    load_config_file,
    loads_config,
    resolve_config,
)
from fitpredict.prediction import predict
from fitpredict.training import FitHistory, FitResult, fit

__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "FitHistory",
    "FitResult",
    "fit",
    "load_config",
    "load_config_file",
    "loads_config",
    "predict",
    "resolve_config",
]
