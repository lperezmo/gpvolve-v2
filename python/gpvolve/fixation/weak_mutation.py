"""Weak-mutation (Lynch-Conery) fixation model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gpvolve.fixation._kernels import weak_mutation_kernel
from gpvolve.fixation.protocol import register_fixation_model


@register_fixation_model(
    name="weak_mutation",
    bounded_unit_interval=False,
    required_params=frozenset(),
)
def weak_mutation(
    fitness_i: NDArray[np.float64], fitness_j: NDArray[np.float64], /, **_params: object
) -> NDArray[np.float64]:
    """``max(0, (f_j - f_i) / f_i)``. Not bounded to [0, 1] for very large gaps."""
    return weak_mutation_kernel(fitness_i, fitness_j)
