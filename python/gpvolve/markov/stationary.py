"""Stationary distribution of a row-stochastic transition matrix.

Two paths:

- Power iteration on the transpose (cheap, sparse-friendly, converges fast for
  well-conditioned chains).
- ARPACK eigenvector at eigenvalue 1 (fallback for ill-conditioned chains or
  when the caller asks for it explicitly).

For reversible chains where detailed balance holds, both paths return the
unique left eigenvector with eigenvalue 1, normalized to sum to 1.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from gpvolve.exceptions import ConvergenceError

Method = Literal["power", "eigs", "auto"]

_DEFAULT_MAX_ITER = 10_000
_DEFAULT_TOL = 1e-12


def stationary_distribution(
    matrix: sp.spmatrix,
    *,
    method: Method = "auto",
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
) -> NDArray[np.float64]:
    """Return the stationary distribution ``pi`` of a row-stochastic matrix.

    Parameters
    ----------
    matrix:
        Row-stochastic transition matrix (shape ``(n, n)``).
    method:
        ``"power"`` uses normalized power iteration on ``P^T``. ``"eigs"`` uses
        ARPACK to find the largest left eigenvector. ``"auto"`` (default)
        starts with power iteration and falls back to ARPACK on non-convergence.
    max_iter:
        Maximum power-iteration steps before giving up. Ignored by ARPACK.
    tol:
        L1-norm tolerance on successive iterates for power iteration.

    Returns
    -------
    1-D ``float64`` numpy array of length ``n`` that sums to 1.
    """
    n = matrix.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)

    if method == "power":
        return _power_iteration(matrix, max_iter=max_iter, tol=tol)
    if method == "eigs":
        return _arpack_stationary(matrix)
    if method == "auto":
        try:
            return _power_iteration(matrix, max_iter=max_iter, tol=tol)
        except ConvergenceError:
            return _arpack_stationary(matrix)
    raise ValueError(f"unknown method {method!r}; expected one of 'power', 'eigs', 'auto'")


def _power_iteration(matrix: sp.spmatrix, *, max_iter: int, tol: float) -> NDArray[np.float64]:
    n = matrix.shape[0]
    pi = np.full(n, 1.0 / n, dtype=np.float64)
    pt = matrix.T.tocsr()
    for _ in range(max_iter):
        nxt = pt @ pi
        # Renormalize to guard against tiny floating drift.
        s = nxt.sum()
        if s <= 0:
            raise ConvergenceError("power iteration collapsed to zero vector; matrix is degenerate")
        nxt = nxt / s
        if np.abs(nxt - pi).sum() < tol:
            return np.asarray(nxt, dtype=np.float64)
        pi = nxt
    raise ConvergenceError(
        f"power iteration did not converge within {max_iter} steps at tol={tol:g}"
    )


def _arpack_stationary(matrix: sp.spmatrix) -> NDArray[np.float64]:
    pt = matrix.T.tocsr().astype(np.float64)
    try:
        _vals, vecs = spla.eigs(pt, k=1, which="LM", maxiter=10_000)
    except spla.ArpackNoConvergence as exc:
        raise ConvergenceError("ARPACK failed to converge on stationary eigenvector") from exc
    pi = np.asarray(vecs[:, 0].real, dtype=np.float64)
    # Sign convention: positive, sum to 1.
    s = pi.sum()
    if s < 0:
        pi = -pi
        s = -s
    if s <= 0:
        raise ConvergenceError("stationary eigenvector summed to zero")
    return np.asarray(pi / s, dtype=np.float64)
