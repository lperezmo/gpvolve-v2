"""Fixation probability models and registry.

Importing this package registers all built-in fixation models in the global registry.
Look up by name via `get_fixation_model("sswm")`, or import the callable directly.
"""

from gpvolve.fixation.bloom import bloom_dms
from gpvolve.fixation.custom import register_fixation_model
from gpvolve.fixation.mccandlish import mccandlish
from gpvolve.fixation.moran import moran
from gpvolve.fixation.protocol import (
    get_fixation_model,
    list_fixation_models,
    validate_params,
)
from gpvolve.fixation.sswm import strong_selection_weak_mutation
from gpvolve.fixation.weak_mutation import weak_mutation

__all__ = [
    "bloom_dms",
    "get_fixation_model",
    "list_fixation_models",
    "mccandlish",
    "moran",
    "register_fixation_model",
    "strong_selection_weak_mutation",
    "validate_params",
    "weak_mutation",
]
