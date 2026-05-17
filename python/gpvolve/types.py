"""Shared Protocols, TypedDicts, and dataclasses for gpvolve-v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict


@runtime_checkable
class FixationModel(Protocol):
    """A vectorized fixation probability evaluator.

    Implementations accept two equal-shaped float64 arrays of source and target fitnesses
    plus model-specific keyword arguments, and return an array of fixation probabilities
    of the same shape.
    """

    name: str
    bounded_unit_interval: bool
    required_params: frozenset[str]

    def __call__(
        self,
        fitness_i: NDArray[np.float64],
        fitness_j: NDArray[np.float64],
        /,
        **params: Any,
    ) -> NDArray[np.float64]: ...


PathMethod = Literal["shortest", "greedy", "stochastic", "tpt"]


@dataclass(frozen=True)
class PathEnsemble:
    """A collection of evolutionary pathways with probabilities and provenance.

    See `SCHEMA.md` section 3 for the locked contract. `probabilities` must sum to a value
    in `[0, 1]`. `paths` is a tuple of tuples so that the dataclass is hashable.
    """

    paths: tuple[tuple[int, ...], ...]
    probabilities: NDArray[np.float64]
    source: int
    targets: tuple[int, ...]
    method: PathMethod
    metadata: dict[str, Any]


class ConvergenceStats(TypedDict, total=False):
    """Payload written to `PathEnsemble.metadata["convergence"]` for stochastic sampling."""

    ess: dict[int, float]
    rhat: dict[int, float]
    n_walkers: int
    n_chunks: int
    converged: bool
