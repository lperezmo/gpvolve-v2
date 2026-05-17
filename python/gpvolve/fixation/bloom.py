"""Bloom (2017)-style empirical DMS fixation model.

The user supplies a precomputed ``pi_table`` matrix of fixation probabilities estimated
from deep mutational scanning preference data, plus an ``indices`` pair mapping the
input fitness arrays to rows and columns of the table.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gpvolve.fixation._kernels import bloom_kernel
from gpvolve.fixation.protocol import register_fixation_model


@register_fixation_model(
    name="bloom_dms",
    bounded_unit_interval=True,
    required_params=frozenset({"pi_table"}),
)
def bloom_dms(
    fitness_i: NDArray[np.float64],
    fitness_j: NDArray[np.float64],
    /,
    *,
    pi_table: NDArray[np.float64],
    indices: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None,
    **_params: object,
) -> NDArray[np.float64]:
    """Look up fixation probabilities from a DMS-derived preference table."""
    return bloom_kernel(fitness_i, fitness_j, pi_table=pi_table, indices=indices)
