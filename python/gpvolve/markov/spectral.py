"""Spectral analysis of an MSM transition matrix: timescales and MFPT.

Timescales are the relaxation timescales of the chain,
``tau_k = -1 / log|lambda_k|``, ordered from the slowest non-stationary mode
downward. The stationary mode (``lambda_1 = 1``) is excluded.

MFPT (mean first passage time) ``m_i = E[tau_target | X_0 = i]`` is the
solution of ``(I - P_off) m = 1`` where ``P_off`` is the transition matrix
with rows of the target set zeroed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from gpvolve.exceptions import ConvergenceError


def eigenvalues(matrix: sp.spmatrix, k: int = 10) -> NDArray[np.complex128]:
    """Return the ``k`` largest-magnitude eigenvalues of ``matrix``.

    Sorted by magnitude descending. The leading eigenvalue should equal 1 to
    floating-point tolerance for a valid row-stochastic chain.
    """
    n = matrix.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.complex128)
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return np.array([1.0 + 0j], dtype=np.complex128)
    # For small matrices, dense eig is faster and more reliable than ARPACK.
    if n <= 50:
        vals = np.linalg.eigvals(matrix.toarray())
    else:
        try:
            vals = spla.eigs(matrix.astype(np.float64), k=k_eff, which="LM", maxiter=10_000)[0]
        except spla.ArpackNoConvergence as exc:
            raise ConvergenceError("ARPACK failed to compute eigenvalues") from exc
        # Pad with 1.0 to recover the leading eigenvalue if ARPACK omitted it.
        if abs(np.max(np.abs(vals)) - 1.0) > 1e-6:
            vals = np.concatenate([vals, [1.0 + 0j]])
    order = np.argsort(-np.abs(vals))
    return vals[order][:k]


def timescales(matrix: sp.spmatrix, k: int = 10) -> NDArray[np.float64]:
    """Return the ``k`` slowest relaxation timescales, excluding the stationary mode.

    ``tau_l = -1 / log|lambda_l|`` for the ``l``-th eigenvalue with ``|lambda_l| < 1``.
    NaN entries are emitted for eigenvalues equal to 1 (the stationary mode).
    """
    vals = eigenvalues(matrix, k=k + 1)
    out: list[float] = []
    for v in vals:
        mag = abs(v)
        if mag >= 1.0 - 1e-12:
            continue
        if mag <= 0:
            out.append(float("inf"))
        else:
            out.append(-1.0 / float(np.log(mag)))
        if len(out) == k:
            break
    while len(out) < k:
        out.append(float("nan"))
    return np.asarray(out, dtype=np.float64)


def mfpt(matrix: sp.spmatrix, targets: int | Iterable[int]) -> NDArray[np.float64]:
    """Mean first passage time from every state to the target set.

    Solves ``(I - Q) m = 1`` where ``Q`` is ``matrix`` with rows in ``targets``
    set to zero. Returns a length-``n`` array; entries indexed by ``targets``
    are ``0`` by definition.
    """
    n = matrix.shape[0]
    target_set = {int(targets)} if isinstance(targets, int) else {int(t) for t in targets}
    if not target_set:
        raise ValueError("targets must be non-empty")
    if any(t < 0 or t >= n for t in target_set):
        raise ValueError(f"target index out of range for matrix of size {n}")

    mask = np.zeros(n, dtype=bool)
    for t in target_set:
        mask[t] = True

    Q_lil = matrix.tolil().copy()
    for t in target_set:
        Q_lil.rows[t] = []
        Q_lil.data[t] = []
    Q = Q_lil.tocsr()

    A = sp.eye(n, format="csr") - Q
    b = np.ones(n, dtype=np.float64)
    b[mask] = 0.0
    m = spla.spsolve(A, b)
    m = np.asarray(m, dtype=np.float64)
    m[mask] = 0.0
    return m


def mixing_time(matrix: sp.spmatrix, *, eps: float = 0.25) -> float:
    """Return the eps-mixing time bound from the spectral gap.

    Uses ``t_mix(eps) <= log(1 / (eps * pi_min)) / (1 - lambda_2)`` where
    ``lambda_2`` is the second-largest eigenvalue magnitude. Returns +inf if
    the spectral gap is degenerate.
    """
    vals = eigenvalues(matrix, k=2)
    if vals.size < 2:
        return float("inf")
    gap = 1.0 - float(abs(vals[1]))
    if gap <= 0:
        return float("inf")
    from gpvolve.markov.stationary import stationary_distribution

    pi = stationary_distribution(matrix)
    pi_min = float(pi.min())
    if pi_min <= 0:
        return float("inf")
    return float(np.log(1.0 / (eps * pi_min)) / gap)
