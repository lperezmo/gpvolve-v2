"""FixationModel Protocol and the model registry.

A fixation model is any callable that accepts two equal-shaped float64 arrays of
source and target fitnesses (plus optional keyword params) and returns an array of
fixation probabilities of the same shape. The registry is module-level and used by
`build_transition_matrix` to look up models by string name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gpvolve.exceptions import ModelError
from gpvolve.types import FixationModel

_REGISTRY: dict[str, FixationModel] = {}


def get_fixation_model(name: str) -> FixationModel:
    """Look up a registered fixation model by its registered name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ModelError(f"unknown fixation model {name!r}; known: {known}") from exc


def list_fixation_models() -> list[str]:
    """Return the sorted list of registered fixation model names."""
    return sorted(_REGISTRY)


def _attach_metadata(
    fn: Callable[..., NDArray[np.float64]],
    *,
    name: str,
    bounded_unit_interval: bool,
    required_params: frozenset[str],
) -> FixationModel:
    """Stamp the FixationModel Protocol attributes onto a plain function."""
    fn.name = name  # type: ignore[attr-defined]
    fn.bounded_unit_interval = bounded_unit_interval  # type: ignore[attr-defined]
    fn.required_params = required_params  # type: ignore[attr-defined]
    return fn  # type: ignore[return-value]


def register_fixation_model(
    *,
    name: str,
    bounded_unit_interval: bool,
    required_params: frozenset[str] | set[str] | None = None,
    aliases: tuple[str, ...] = (),
) -> Callable[[Callable[..., NDArray[np.float64]]], FixationModel]:
    """Decorator: register a callable as a fixation model in the global registry.

    The decorated callable must accept `(fitness_i, fitness_j, **params)` and return an
    array of the same shape. The decorator stamps `name`, `bounded_unit_interval`, and
    `required_params` onto the function so it satisfies the FixationModel Protocol at
    runtime.
    """
    required = frozenset(required_params or ())

    def decorator(fn: Callable[..., NDArray[np.float64]]) -> FixationModel:
        model = _attach_metadata(
            fn,
            name=name,
            bounded_unit_interval=bounded_unit_interval,
            required_params=required,
        )
        if not isinstance(model, FixationModel):
            raise ModelError(
                f"function {fn.__qualname__} does not satisfy the FixationModel Protocol"
            )
        if name in _REGISTRY:
            raise ModelError(f"fixation model {name!r} is already registered")
        _REGISTRY[name] = model
        for alias in aliases:
            if alias in _REGISTRY:
                raise ModelError(f"fixation model alias {alias!r} collides with an existing name")
            _REGISTRY[alias] = model
        return model

    return decorator


def validate_params(model: FixationModel, params: dict[str, Any]) -> None:
    """Raise ModelError if any required parameter is missing from `params`."""
    missing = model.required_params - params.keys()
    if missing:
        raise ModelError(
            f"fixation model {model.name!r} requires params {sorted(missing)}; got {sorted(params)}"
        )
