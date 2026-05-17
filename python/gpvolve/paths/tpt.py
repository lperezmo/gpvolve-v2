"""Transition Path Theory: committors, reactive flux, rate.

Berezhkovskii, Hummer, Szabo (2009) formulation. Notation:

- ``A``       : source set (indices).
- ``B``       : target set (indices).
- ``q_plus``  : forward committor, ``q+_i = P(reach B before A | X_0 = i)``.
- ``q_minus`` : backward committor; for reversible chains ``q-_i = 1 - q+_i``.
- ``f``       : reactive flux, ``f_ij = pi_i q-_i P_ij q+_j``, ``i != j``.
- ``f_net``   : net flux, ``f_net_ij = max(0, f_ij - f_ji)``.
- ``k_AB``    : rate ``k_AB = sum_{i in A, j not in A} pi_i P_ij q+_j``.

The forward committor solves the absorbing-boundary system

    (I - P_off) q = b

where ``P_off`` is ``P`` with rows in ``A`` and ``B`` zeroed and
``b = P (1_B)`` restricted to free rows; substituted boundary conditions
fold ``q_A = 0, q_B = 1`` into the right-hand side.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from gpvolve.exceptions import GpvolveError
from gpvolve.markov.stationary import stationary_distribution


def _coerce_set(x: int | Iterable[int], *, n: int, name: str) -> NDArray[np.int64]:
    if isinstance(x, int):
        arr = np.asarray([int(x)], dtype=np.int64)
    else:
        arr = np.asarray(sorted({int(v) for v in x}), dtype=np.int64)
    if arr.size == 0:
        raise GpvolveError(f"{name} must be non-empty")
    if (arr < 0).any() or (arr >= n).any():
        raise GpvolveError(f"{name} contains out-of-range index for matrix of size {n}")
    return arr


def forward_committor(
    matrix: sp.spmatrix,
    A: int | Iterable[int],
    B: int | Iterable[int],
) -> NDArray[np.float64]:
    """Forward committor probabilities ``q+_i`` for every state.

    ``q+_i = P(chain reaches B before A | X_0 = i)``. Boundary conditions:
    ``q+_A = 0``, ``q+_B = 1``.
    """
    n = matrix.shape[0]
    A_arr = _coerce_set(A, n=n, name="A")
    B_arr = _coerce_set(B, n=n, name="B")
    if np.intersect1d(A_arr, B_arr).size > 0:
        raise GpvolveError("A and B must be disjoint")

    boundary_mask = np.zeros(n, dtype=bool)
    boundary_mask[A_arr] = True
    boundary_mask[B_arr] = True
    free_mask = ~boundary_mask

    csr = matrix.tocsr()
    P_ff = csr[free_mask][:, free_mask]
    P_fB = csr[free_mask][:, B_arr]

    n_free = int(free_mask.sum())
    rhs = np.asarray(P_fB.sum(axis=1)).ravel()
    A_lhs = sp.eye(n_free, format="csr") - P_ff

    q_free = spla.spsolve(A_lhs, rhs) if n_free > 0 else np.zeros(0, dtype=np.float64)

    q = np.zeros(n, dtype=np.float64)
    q[B_arr] = 1.0
    q[free_mask] = q_free
    return np.clip(q, 0.0, 1.0)


def backward_committor(
    matrix: sp.spmatrix,
    A: int | Iterable[int],
    B: int | Iterable[int],
    *,
    stationary: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Backward committor probabilities ``q-_i`` for every state.

    Computed by solving the forward committor on the time-reversed chain
    ``P~_ij = pi_j P_ji / pi_i`` with the roles of ``A`` and ``B`` swapped.
    For reversible chains, ``q-_i = 1 - q+_i``.
    """
    n = matrix.shape[0]
    pi = stationary_distribution(matrix) if stationary is None else stationary
    if (pi <= 0).any():
        raise GpvolveError("backward committor requires strictly positive stationary distribution")
    # Build the reverse chain by swapping (i, j) -> (j, i) in the COO triples.
    # This is equivalent to diag(1/pi) @ P^T @ diag(pi) and yields the
    # time-reversed transition matrix. Solving the forward committor of the
    # reversed chain with A and B swapped gives the backward committor of the
    # original chain.
    coo = matrix.tocoo()
    rev_data = (pi[coo.col] / pi[coo.row]) * coo.data
    P_rev_t = sp.coo_matrix((rev_data, (coo.col, coo.row)), shape=(n, n)).tocsr()
    return forward_committor(P_rev_t, A=B, B=A)


def reactive_flux(
    matrix: sp.spmatrix,
    A: int | Iterable[int],
    B: int | Iterable[int],
    *,
    stationary: NDArray[np.float64] | None = None,
) -> sp.csr_matrix:
    """Reactive flux ``f_ij = pi_i q-_i P_ij q+_j`` as a sparse matrix.

    Diagonal entries are zero. The sum of out-of-A reactive flux equals the
    A-to-B rate (see :func:`rate`). The matrix has the same sparsity pattern
    as the off-diagonal of ``matrix``.
    """
    n = matrix.shape[0]
    pi = stationary_distribution(matrix) if stationary is None else stationary
    q_plus = forward_committor(matrix, A=A, B=B)
    q_minus = backward_committor(matrix, A=A, B=B, stationary=pi)

    coo = matrix.tocoo()
    mask = coo.row != coo.col
    rows = coo.row[mask]
    cols = coo.col[mask]
    data = pi[rows] * q_minus[rows] * coo.data[mask] * q_plus[cols]
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def net_flux(flux: sp.spmatrix) -> sp.csr_matrix:
    """Element-wise net flux ``max(0, f_ij - f_ji)`` from a reactive-flux matrix."""
    F = flux.tocsr()
    diff = F - F.T
    diff = diff.tocoo()
    keep = diff.data > 0
    return sp.coo_matrix((diff.data[keep], (diff.row[keep], diff.col[keep])), shape=F.shape).tocsr()


def rate(
    matrix: sp.spmatrix,
    A: int | Iterable[int],
    B: int | Iterable[int],
    *,
    stationary: NDArray[np.float64] | None = None,
) -> float:
    """A-to-B reactive rate ``k_AB = sum_{i in A, j not in A} pi_i P_ij q+_j``."""
    n = matrix.shape[0]
    A_arr = _coerce_set(A, n=n, name="A")
    pi = stationary_distribution(matrix) if stationary is None else stationary
    q_plus = forward_committor(matrix, A=A_arr, B=B)
    csr = matrix.tocsr()
    not_A = np.ones(n, dtype=bool)
    not_A[A_arr] = False
    sub = csr[A_arr][:, not_A]
    # sub: rows indexed by A, cols indexed by non-A states.
    contrib = sub.multiply(q_plus[not_A][np.newaxis, :])
    row_pi = pi[A_arr][:, np.newaxis]
    return float(np.asarray(contrib.multiply(row_pi).sum()).item())
