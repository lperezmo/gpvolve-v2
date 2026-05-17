"""Validation helpers for transition matrices.

The MSM contract in ``SCHEMA.md`` requires:

- Rows sum to ``1.0 +/- 1e-12``.
- All entries in ``[0, 1]``.
- Optional: graph reachability / strong connectivity for ergodicity claims.

These checks are pure-Python on the sparse matrix and used by both
``build_transition_matrix`` (as a defensive postcondition) and by the user
when loading externally-built matrices.

Beyond the basic shape checks, the module also exposes higher-level
predicates used throughout the rest of the library:

- :func:`is_irreducible` (alias of :func:`is_strongly_connected`).
- :func:`is_ergodic` for the standard finite-chain ergodicity criterion
  (irreducible + aperiodic). Aperiodicity is satisfied as soon as any
  diagonal entry is positive; gpvolve chains built via
  :func:`gpvolve.build_transition_matrix` always carry an absorption diagonal,
  so the check is exact for those chains.
- :func:`is_reversible` for the detailed-balance test
  ``pi_i P_ij = pi_j P_ji``.
- :func:`absorbing_states` for enumerating sinks (``T_ii ~ 1``); useful when
  routing through :mod:`gpvolve.markov.absorbing`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from numpy.typing import NDArray

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


def is_irreducible(matrix: sp.spmatrix) -> bool:
    """Return True if the chain is irreducible.

    Irreducibility of a finite-state Markov chain is equivalent to strong
    connectivity of its underlying directed graph (Norris 1997, Ch 1). This is
    an alias for :func:`is_strongly_connected`; the rename exists so that
    code reading "is irreducible" matches textbook language.
    """
    return is_strongly_connected(matrix)


def is_ergodic(matrix: sp.spmatrix) -> bool:
    """Return True if the chain is ergodic.

    A finite-state Markov chain is ergodic iff it is irreducible **and**
    aperiodic (Levin-Peres-Wilmer 2017, Ch 1). For a transition matrix with a
    positive diagonal, aperiodicity is automatic (a self-loop is a cycle of
    length 1, so the period of every state divides 1). The check used here is

    - ``is_irreducible(matrix)`` and
    - at least one diagonal entry is strictly positive.

    This is a **sufficient** criterion that matches the structure of every
    matrix built by :func:`gpvolve.build_transition_matrix`, since the
    absorption-diagonal rule guarantees ``T_ii > 0`` whenever the row has any
    deleterious neighbor. It is not the most general aperiodicity check (a
    chain with no self-loops can still be aperiodic), but no realistic gpvolve
    workflow produces such a chain.
    """
    n = matrix.shape[0]
    if n == 0:
        return True  # vacuous: no states, no failure to mix.
    if not is_irreducible(matrix):
        return False
    diag = np.asarray(matrix.diagonal()).ravel()
    return bool(np.any(diag > 0))


def is_reversible(
    matrix: sp.spmatrix,
    *,
    stationary: NDArray[np.float64] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> bool:
    """Return True if the chain satisfies detailed balance.

    Detailed balance (Kolmogorov's reversibility criterion, Norris 1997 Theorem
    1.7.1) requires ``pi_i P_ij == pi_j P_ji`` for every pair ``(i, j)``. The
    test compares ``|pi_i P_ij - pi_j P_ji|`` against ``atol + rtol *
    (pi_i P_ij + pi_j P_ji) / 2`` for every nonzero entry in the upper
    triangle.

    If ``stationary`` is None, :func:`stationary_distribution` is called; for
    non-ergodic chains where ``pi`` has zero entries the result is undefined
    and the function returns ``False`` (the standard detailed-balance test
    presumes a strictly positive stationary distribution).
    """
    n = matrix.shape[0]
    if n <= 1:
        return True
    if stationary is None:
        from gpvolve.markov.stationary import stationary_distribution

        stationary = stationary_distribution(matrix)
    pi = np.asarray(stationary, dtype=np.float64)
    if (pi <= 0).any():
        return False
    csr = matrix.tocsr()
    coo = csr.tocoo()
    for i, j, p_ij in zip(coo.row, coo.col, coo.data, strict=True):
        if i >= j:
            continue
        p_ji = float(csr[j, i])  # Direct lookup of P[j, i] in CSR.
        p_ij_f = float(p_ij)
        # Support-asymmetry check: if exactly one of P[i,j] and P[j,i] is zero
        # while the other is positive, detailed balance is violated (a chain
        # that can step i->j but not j->i is not reversible regardless of the
        # stationary distribution). This catches SSWM-style asymmetric kernels
        # without relying on the magnitude of the stationary entries (which
        # can underflow to numerically tiny but technically positive values).
        ij_zero = p_ij_f <= atol
        ji_zero = p_ji <= atol
        if ij_zero != ji_zero:
            return False
        if ij_zero and ji_zero:
            continue
        lhs = float(pi[i]) * p_ij_f
        rhs = float(pi[j]) * p_ji
        scale = atol + rtol * 0.5 * (lhs + rhs)
        if abs(lhs - rhs) > scale:
            return False
    return True


def absorbing_states(matrix: sp.spmatrix, *, tol: float = 1e-12) -> NDArray[np.int64]:
    """Return the indices of absorbing states (``T_ii >= 1 - tol`` and no off-diagonal mass).

    A state ``i`` is absorbing in a row-stochastic chain when ``P_ii = 1`` (and
    therefore ``P_ij = 0`` for ``j != i``); see Kemeny & Snell (1976) Ch 3. The
    function checks both conditions so that a state with ``T_ii`` numerically
    close to 1 but with a small off-diagonal leak is correctly **not** flagged
    as absorbing.
    """
    n = matrix.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    csr = matrix.tocsr()
    diag = np.asarray(csr.diagonal()).ravel()
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    off_diag = row_sums - diag
    absorbing = (diag >= 1.0 - tol) & (off_diag <= tol)
    return np.flatnonzero(absorbing).astype(np.int64)


def transient_states(
    matrix: sp.spmatrix, *, absorbing: Iterable[int] | None = None, tol: float = 1e-12
) -> NDArray[np.int64]:
    """Return the indices of transient states (complement of the absorbing set).

    If ``absorbing`` is None, the set is computed via :func:`absorbing_states`.
    This helper exists because :mod:`gpvolve.markov.absorbing` repeatedly needs
    "all states except the sinks" and the two-line idiom is error-prone.
    """
    n = matrix.shape[0]
    if absorbing is None:
        abs_idx = absorbing_states(matrix, tol=tol)
    else:
        abs_idx = np.asarray(sorted({int(i) for i in absorbing}), dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[abs_idx] = False
    return np.flatnonzero(mask).astype(np.int64)
