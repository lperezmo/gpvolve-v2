"""McCandlish (2011) fixation model.

The original gpvolve and gpgraph-v2 spelled this `mcclandish`. v2 uses the correct
spelling `mccandlish` and keeps `mcclandish` as a registry alias for back-compat with
user scripts that pin the v1 name.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gpvolve.fixation._kernels import mccandlish_kernel
from gpvolve.fixation.protocol import register_fixation_model


@register_fixation_model(
    name="mccandlish",
    bounded_unit_interval=True,
    required_params=frozenset({"population_size"}),
    aliases=("mcclandish",),
)
def mccandlish(
    fitness_i: NDArray[np.float64],
    fitness_j: NDArray[np.float64],
    /,
    *,
    population_size: float,
    **_params: object,
) -> NDArray[np.float64]:
    """McCandlish (2011) fixation probability for effective population size ``N``."""
    return mccandlish_kernel(fitness_i, fitness_j, population_size)
