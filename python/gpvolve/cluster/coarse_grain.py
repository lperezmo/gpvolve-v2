"""Coarse-grained transition matrix on PCCA+ memberships.

The Galerkin projection of ``P`` onto the memberships ``chi`` (Roeblitz-Weber
2013, eq. 14) is

    P_coarse = (chi^T diag(pi) chi)^{-1} chi^T diag(pi) P chi

The resulting matrix is the row-stochastic transition matrix on the metastable
basins.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from gpvolve.markov.stationary import stationary_distribution


def coarse_grain(
    matrix: sp.spmatrix,
    chi: NDArray[np.float64],
    *,
    stationary: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Galerkin-coarse-grain ``matrix`` to the cluster space spanned by ``chi``.

    Parameters
    ----------
    matrix:
        Fine-grained row-stochastic transition matrix.
    chi:
        ``(n, k)`` membership matrix from :func:`pcca_plus`.
    stationary:
        Optional precomputed stationary distribution. If ``None``, computed.

    Returns
    -------
    ``(k, k)`` coarse-grained row-stochastic transition matrix.
    """
    pi = stationary_distribution(matrix) if stationary is None else stationary
    D = sp.diags(pi)
    chi_t_D = chi.T @ D
    A = chi_t_D @ chi
    B = chi_t_D @ matrix @ chi
    P_coarse = np.linalg.solve(A, B)
    # Cleanup: enforce nonneg and renormalize rows.
    P_coarse = np.maximum(P_coarse, 0.0)
    row_sums = P_coarse.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return np.asarray(P_coarse / row_sums, dtype=np.float64)
