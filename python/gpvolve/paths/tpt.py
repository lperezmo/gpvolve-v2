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

from gpvolve.exceptions import ConvergenceError, GpvolveError
from gpvolve.markov.stationary import stationary_distribution

try:
    from gpvolve._rust import solve_bicgstab_csr as _rust_solve_bicgstab

    _RUST_BICGSTAB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extension
    _rust_solve_bicgstab = None  # type: ignore[assignment]
    _RUST_BICGSTAB_AVAILABLE = False

# scipy.sparse.linalg.spsolve uses a supernodal LU. Empirically (binary maps,
# Moran, pop_size=100) it scales as roughly O(n^2.5) once fill-in dominates:
# n=4096 -> 3 s, n=8192 -> 26 s, n=16384 timed out at 60 s. BiCGSTAB iterates
# in O(nnz) per step and converges in ~50 steps for these systems, so it wins
# everywhere except on toy test fixtures where FFI overhead dominates.
_BICGSTAB_SIZE_THRESHOLD = 256
_BICGSTAB_MAX_ITER = 2_000
_BICGSTAB_TOL = 1e-10


def _coerce_set(x: int | Iterable[int], *, n: int, name: str) -> NDArray[np.int64]:
    if isinstance(x, (int, np.integer)):
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

    if n_free == 0:
        q_free = np.zeros(0, dtype=np.float64)
    elif _RUST_BICGSTAB_AVAILABLE and n_free > _BICGSTAB_SIZE_THRESHOLD:
        A_csr = A_lhs.tocsr()
        x, _iters, converged = _rust_solve_bicgstab(
            np.ascontiguousarray(A_csr.indptr, dtype=np.int64),
            np.ascontiguousarray(A_csr.indices, dtype=np.int64),
            np.ascontiguousarray(A_csr.data, dtype=np.float64),
            np.ascontiguousarray(rhs, dtype=np.float64),
            _BICGSTAB_MAX_ITER,
            _BICGSTAB_TOL,
        )
        if not converged:
            # Iterative solver failed to hit tolerance; fall back to direct LU.
            # spsolve at this size is slow but reliable.
            q_free = spla.spsolve(A_lhs, rhs)
            if q_free is None or np.any(~np.isfinite(q_free)):
                raise ConvergenceError(
                    "forward committor BiCGSTAB and LU both failed to produce a finite solution"
                )
        else:
            q_free = np.asarray(x, dtype=np.float64)
    else:
        q_free = spla.spsolve(A_lhs, rhs)

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
        n_zero = int((pi <= 0).sum())
        raise GpvolveError(
            "backward committor is undefined: the stationary distribution has "
            f"{n_zero} non-positive entries out of {n}, so the chain is not ergodic. "
            "This commonly occurs when the fixation model produces absorbing states "
            "(e.g. SSWM on a single-peak landscape, where every fitness peak becomes "
            "a true sink with T_ii=1). Standard transition path theory requires an "
            "ergodic chain with strictly positive stationary distribution. Forward "
            "committor and rate are still well-defined for absorbing chains."
        )
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


def absorption_rate(
    matrix: sp.spmatrix,
    A: int | Iterable[int],
    B: int | Iterable[int],
    *,
    initial: NDArray[np.float64] | None = None,
) -> float:
    """MFPT-based rate ``k = 1 / E[tau_B | X_0 ~ initial]`` for absorbing chains.

    The standard reactive rate :func:`rate` is the long-time average flux from
    A to B in an ergodic chain. When the chain is absorbing (and the
    stationary distribution puts zero mass on A), that flux collapses to zero
    and the reactive rate becomes uninformative. The natural rate constant in
    that regime is the reciprocal of the expected absorption time
    (Hanggi-Talkner-Borkovec 1990; this is the chemical-kinetics rate
    constant for an irreversible A -> B process).

    Parameters
    ----------
    matrix:
        Row-stochastic transition matrix.
    A:
        Source state(s). Used as the support of the initial distribution
        when ``initial`` is None.
    B:
        Target absorbing set. Treated as absorbing for the MFPT computation
        (B states are zeroed out internally so the standard
        ``(I - Q) m = 1`` solve applies, regardless of whether the input
        matrix already had B absorbing).
    initial:
        Optional length-``n`` initial distribution. Must be non-negative and
        sum to 1 over its support. If None, a uniform distribution over
        ``A`` is used.

    Returns
    -------
    Rate constant ``k = 1 / E[tau_B]`` in inverse time-steps. ``+inf`` if
    every state in the support of the initial distribution is already in B
    (zero absorption time); ``0.0`` if ``B`` is unreachable from at least
    one state in the support.

    Notes
    -----
    The function uses :func:`gpvolve.mfpt`, which solves
    ``(I - Q) m = 1`` with ``Q = P[:, :]`` and target rows zeroed. The
    expected absorption time from ``initial`` is then
    ``E[tau_B] = sum_i initial_i m_i``. For chains with multiple competing
    absorbing classes, use :func:`gpvolve.conditional_mfpt` to condition the
    expectation on the event ``{absorbed in B}``.

    Comparison to :func:`rate`: the two agree when the chain is ergodic and
    the initial distribution is the stationary distribution restricted to A
    and renormalized, in the limit of slow A -> B transitions
    (Berezhkovskii-Hummer-Szabo 2009 reduces to Eyring/Kramers in that
    regime).

    References
    ----------
    Hanggi, P., Talkner, P., Borkovec, M. (1990). "Reaction-rate theory:
    fifty years after Kramers." *Reviews of Modern Physics* 62, 251-341.
    """
    from gpvolve.markov.absorbing import conditional_mfpt as _conditional_mfpt

    n = matrix.shape[0]
    A_arr = _coerce_set(A, n=n, name="A")
    B_arr = _coerce_set(B, n=n, name="B")
    if np.intersect1d(A_arr, B_arr).size > 0:
        raise GpvolveError("A and B must be disjoint")

    if initial is None:
        weights_A = np.full(A_arr.size, 1.0 / A_arr.size, dtype=np.float64)
        support_idx = A_arr
    else:
        full = np.asarray(initial, dtype=np.float64).copy()
        if full.shape != (n,):
            raise GpvolveError(f"initial must have shape ({n},); got {full.shape}")
        if (full < 0).any():
            raise GpvolveError("initial distribution must be non-negative")
        total = float(full.sum())
        if total <= 0:
            raise GpvolveError("initial distribution must have positive total mass")
        full = full / total
        support_mask = full > 0
        if not support_mask.any():
            raise GpvolveError("initial distribution has no support")
        support_idx = np.flatnonzero(support_mask).astype(np.int64)
        weights_A = full[support_idx]

    # Use conditional_mfpt so the rate is well-defined when other absorbing
    # classes compete with B (multi-peak landscapes); for chains where B is
    # the only sink, this reduces to ordinary mfpt(P, B).
    m_cond = _conditional_mfpt(matrix, A=support_idx.tolist(), B=B_arr.tolist())
    if np.any(np.isinf(m_cond)):
        # At least one supported state has zero probability of absorbing in B.
        return 0.0
    expected_tau = float(np.dot(weights_A, m_cond))
    if expected_tau <= 0:
        return float("inf")
    return 1.0 / expected_tau
