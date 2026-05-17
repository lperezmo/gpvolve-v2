"""Validation helpers for transition matrices.

The MSM contract in ``SCHEMA.md`` requires:

- Rows sum to ``1.0 +/- 1e-12``.
- All entries in ``[0, 1]``.
- Optional: graph reachability / strong connectivity for ergodicity claims.

These checks are pure-Python on the sparse matrix and used by both
``build_transition_matrix`` (as a defensive postcondition) and by the user
when loading externally-built matrices.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from gpvolve.exceptions import NonStochasticError

_ROW_TOL = 1e-12
_BOUND_TOL = 1e-12


def assert_row_stochastic(matrix: sp.spmatrix, *, tol: float = _ROW_TOL) -> None:
    """Raise NonStochasticError if rows do not sum to 1.0 within ``tol``."""
    rows = np.asarray(matrix.sum(axis=1)).ravel()
    diff = np.abs(rows - 1.0).max() if rows.size else 0.0
    if diff > tol:
        raise NonStochasticError(
            f"transition matrix rows deviate from 1.0 by up to {diff:g} (tol={tol:g})"
        )


def assert_nonneg(matrix: sp.spmatrix, *, tol: float = _BOUND_TOL) -> None:
    """Raise NonStochasticError if any entry falls below ``-tol`` or above ``1 + tol``."""
    csr = matrix.tocsr()
    if csr.nnz == 0:
        return
    data = csr.data
    if np.any(data < -tol) or np.any(data > 1.0 + tol):
        worst_lo = float(data.min())
        worst_hi = float(data.max())
        raise NonStochasticError(
            f"transition matrix entries outside [0, 1]: min={worst_lo:g}, max={worst_hi:g}"
        )


def is_strongly_connected(matrix: sp.spmatrix) -> bool:
    """Return True if the underlying directed graph of ``matrix`` is strongly connected.

    Diagonal self-loops are ignored for the purpose of this check, so a matrix
    that is row-stochastic only through self-absorbing entries is not falsely
    reported as strongly connected.
    """
    n = matrix.shape[0]
    if n <= 1:
        return True
    csr = matrix.tocsr().copy()
    csr.setdiag(0)
    csr.eliminate_zeros()
    n_components, _labels = csgraph.connected_components(csr, directed=True, connection="strong")
    return bool(n_components == 1)


def assert_strongly_connected(matrix: sp.spmatrix) -> None:
    """Raise NonStochasticError if ``matrix`` is not strongly connected (ignoring self-loops)."""
    if not is_strongly_connected(matrix):
        raise NonStochasticError(
            "transition matrix is not strongly connected; ergodicity is not guaranteed"
        )
