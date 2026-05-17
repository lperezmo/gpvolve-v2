"""Moran-process fixation model (Sella and Hirsch 2005)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gpvolve.fixation._kernels import moran_kernel
from gpvolve.fixation.protocol import register_fixation_model


@register_fixation_model(
    name="moran",
    bounded_unit_interval=True,
    required_params=frozenset({"population_size"}),
)
def moran(
    fitness_i: NDArray[np.float64],
    fitness_j: NDArray[np.float64],
    /,
    *,
    population_size: float,
    **_params: object,
) -> NDArray[np.float64]:
    """Moran fixation probability for effective population size ``N``."""
    return moran_kernel(fitness_i, fitness_j, population_size)
