"""Helpers for working with PCCA+ membership matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def metastable_sets(chi: NDArray[np.float64]) -> list[NDArray[np.int64]]:
    """Hard-assignment cluster membership: argmax across clusters.

    Returns a list of length ``n_clusters``; the ``j``-th entry is a sorted
    1-D ``int64`` array of the state indices that have ``chi[i, j]`` as their
    largest membership.
    """
    if chi.ndim != 2:
        raise ValueError("chi must be a 2-D membership matrix")
    assignment = np.argmax(chi, axis=1)
    out: list[NDArray[np.int64]] = []
    for j in range(chi.shape[1]):
        out.append(np.sort(np.where(assignment == j)[0]).astype(np.int64))
    return out


def crisp_assignments(chi: NDArray[np.float64]) -> NDArray[np.int64]:
    """Argmax cluster assignment, one int per state."""
    return np.asarray(np.argmax(chi, axis=1).astype(np.int64), dtype=np.int64)
