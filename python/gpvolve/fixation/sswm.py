"""Strong-selection weak-mutation fixation model (Gillespie 1984)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gpvolve.fixation._kernels import sswm_kernel
from gpvolve.fixation.protocol import register_fixation_model


@register_fixation_model(
    name="sswm",
    bounded_unit_interval=True,
    required_params=frozenset(),
    aliases=("strong_selection_weak_mutation",),
)
def strong_selection_weak_mutation(
    fitness_i: NDArray[np.float64], fitness_j: NDArray[np.float64], /, **_params: object
) -> NDArray[np.float64]:
    """``pi_{i -> j} = 1 - exp(-(f_j - f_i) / f_i)`` for f_j > f_i, else 0."""
    return sswm_kernel(fitness_i, fitness_j)
