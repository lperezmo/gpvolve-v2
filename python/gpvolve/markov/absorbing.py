"""Absorbing-chain analysis: fundamental matrix, QSD, conditional MFPT.

For a finite Markov chain with one or more absorbing states, decompose the
transition matrix into

    P = | Q   R |
        | 0   I |

where ``Q`` is the substochastic block on transient states, ``R`` is the
transient-to-absorbing block, and ``I`` is the identity on absorbing states.
The standard absorbing-chain analytics (Kemeny & Snell, *Finite Markov
Chains*, 1976, Ch 3) follow:

- Fundamental matrix ``N = (I - Q)^{-1}``. Entry ``N_ij`` is the expected
  number of visits to transient state ``j`` starting from transient state
  ``i``, summed over all time before absorption.
- Expected time to absorption from transient state ``i`` is ``(N 1)_i``;
  exposed separately in :func:`gpvolve.mfpt`.
- Absorption probabilities ``B = N R``. Entry ``B_ij`` is the probability
  that a chain starting at transient ``i`` is eventually absorbed in
  absorbing state ``j``.
- Conditional MFPT: given that the chain is absorbed in a particular
  absorbing state (or set), the expected time conditional on that event.

The quasi-stationary distribution (QSD) is the left eigenvector of ``Q``
with the largest-magnitude eigenvalue ``lambda_1``; it is the limiting
distribution on transient states conditional on non-absorption (Darroch &
Seneta, "On quasi-stationary distributions in absorbing discrete-time
finite Markov chains", *J. Appl. Prob.* 2, 1965, pp 88-100). For irreducible
``Q`` (which holds whenever the transient class itself is strongly
connected), the QSD is unique.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from gpvolve.exceptions import ConvergenceError, GpvolveError


def _resolve_absorbing(matrix: sp.spmatrix, absorbing: Iterable[int] | None) -> NDArray[np.int64]:
    """Return absorbing state indices, computing them if not provided."""
    if absorbing is None:
        from gpvolve.markov.validation import absorbing_states

        return absorbing_states(matrix)
    return np.asarray(sorted({int(i) for i in absorbing}), dtype=np.int64)


def fundamental_matrix(
    matrix: sp.spmatrix,
    *,
    absorbing: Iterable[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Fundamental matrix ``N = (I - Q)^{-1}`` of an absorbing chain.

    Parameters
    ----------
    matrix:
        Row-stochastic transition matrix.
    absorbing:
        Indices of absorbing states. If ``None``, computed via
        :func:`gpvolve.markov.validation.absorbing_states`.

    Returns
    -------
    ``(N, transient_idx)`` where ``N`` is the fundamental matrix of shape
    ``(n_transient, n_transient)`` and ``transient_idx`` is the
    corresponding index array into the original state space. ``N[i, j]`` is
    the expected number of visits to ``transient_idx[j]`` starting from
    ``transient_idx[i]``, summed over all time before absorption.

    Raises
    ------
    GpvolveError
        If the chain has no absorbing states (``N`` is not defined for an
        ergodic chain). Use :func:`gpvolve.mfpt` for hitting times in an
        ergodic chain instead.

    Notes
    -----
    Uses a dense LU on the transient block because ``(I - Q)^{-1}`` is
    typically dense even when ``Q`` is sparse; gpvolve's transient blocks for
    L<=8 binary maps stay well under 256x256, so the dense path is the right
    choice. For very large transient classes (n > 2000) the caller should
    instead compute ``N v`` on demand via :func:`gpvolve.mfpt`-style sparse
    solves.

    References
    ----------
    Kemeny, J.G. & Snell, J.L. (1976). *Finite Markov Chains*. Springer,
    Ch 3, Theorem 3.2.4.
    """
    n = matrix.shape[0]
    abs_idx = _resolve_absorbing(matrix, absorbing)
    if abs_idx.size == 0:
        raise GpvolveError(
            "fundamental_matrix requires at least one absorbing state; "
            "this chain has none. For an ergodic chain, use gpvolve.mfpt to "
            "compute mean first passage times to a target set."
        )
    if abs_idx.size == n:
        # All states are absorbing; the transient block is empty.
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.int64)

    mask = np.ones(n, dtype=bool)
    mask[abs_idx] = False
    transient_idx = np.flatnonzero(mask).astype(np.int64)

    csr = matrix.tocsr()
    Q = csr[transient_idx][:, transient_idx]
    n_t = transient_idx.size
    A = sp.eye(n_t, format="csr") - Q
    # Solve (I - Q) N = I column-wise. For n_t < ~2000 this is fastest dense.
    rhs = np.eye(n_t, dtype=np.float64)
    N = spla.spsolve(A.tocsc(), sp.csc_matrix(rhs))
    if sp.issparse(N):
        N = N.toarray()
    return np.asarray(N, dtype=np.float64), transient_idx


def absorption_probabilities(
    matrix: sp.spmatrix,
    *,
    absorbing: Iterable[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Absorption probability matrix ``B = N R`` of Kemeny & Snell (1976).

    Returns
    -------
    ``(B, transient_idx, absorbing_idx)``. ``B[i, k]`` is the probability that
    a chain starting at transient state ``transient_idx[i]`` is eventually
    absorbed at ``absorbing_idx[k]``. Row sums equal 1 to within floating
    tolerance.

    Notes
    -----
    For chains with a single absorbing state, ``B`` collapses to a column of
    ones and conveys no information; for chains with multiple absorbing
    states (multi-peak SSWM landscapes, for instance) ``B`` is the standard
    way to read off "from which peak is this state in the basin of?"

    References
    ----------
    Kemeny, J.G. & Snell, J.L. (1976). *Finite Markov Chains*, Ch 3,
    Theorem 3.3.7.
    """
    n = matrix.shape[0]
    abs_idx = _resolve_absorbing(matrix, absorbing)
    if abs_idx.size == 0:
        raise GpvolveError("absorption_probabilities requires at least one absorbing state")
    if abs_idx.size == n:
        # All states absorb in themselves; B is the identity.
        return np.eye(n, dtype=np.float64), np.empty(0, dtype=np.int64), abs_idx

    N, transient_idx = fundamental_matrix(matrix, absorbing=abs_idx)
    csr = matrix.tocsr()
    R = csr[transient_idx][:, abs_idx].toarray()
    B = N @ R
    return np.asarray(B, dtype=np.float64), transient_idx, abs_idx


def conditional_mfpt(
    matrix: sp.spmatrix,
    A: int | Iterable[int],
    B: int | Iterable[int],
) -> NDArray[np.float64]:
    """Mean first passage time to ``B``, conditional on absorbing in ``B``.

    For a chain with multiple absorbing states (or competing target sets),
    plain :func:`gpvolve.mfpt` averages over trajectories that reach ``B``
    and trajectories that absorb elsewhere; ``conditional_mfpt`` returns the
    expectation taken only over the trajectories that *do* reach ``B``.

    The two agree when ``B`` is the only absorbing set in the chain.

    Parameters
    ----------
    matrix:
        Row-stochastic transition matrix.
    A:
        Source states. Returned values are the conditional MFPT entries at
        these indices.
    B:
        Target set. Treated as absorbing in the analysis; need not coincide
        with the chain's pre-existing absorbing states. If the chain has
        additional sinks outside ``B``, those are kept absorbing too and
        compete with ``B``.

    Returns
    -------
    1-D array of length ``len(A)``. Entries from states with zero
    absorption probability to ``B`` (unreachable targets) are ``+inf``;
    entries in ``A intersect B`` are ``0``.

    Notes
    -----
    Implementation uses Doob's h-transform: with ``h_i = Pr[absorbed in B |
    X_0 = i]``, the conditional process has transition kernel
    ``P_tilde_ij = P_ij h_j / h_i`` on the support ``{h > 0}``. The
    conditional MFPT to ``B`` is then a standard MFPT computation on
    ``P_tilde``.

    References
    ----------
    Norris, J.R. (1997). *Markov Chains*, Cambridge UP, Theorem 4.2.3
    (h-transform / Doob conditioning).
    """
    n = matrix.shape[0]
    A_arr = (
        np.asarray([int(A)], dtype=np.int64)
        if isinstance(A, (int, np.integer))
        else np.asarray(sorted({int(v) for v in A}), dtype=np.int64)
    )
    B_arr = (
        np.asarray([int(B)], dtype=np.int64)
        if isinstance(B, (int, np.integer))
        else np.asarray(sorted({int(v) for v in B}), dtype=np.int64)
    )
    if A_arr.size == 0 or B_arr.size == 0:
        raise GpvolveError("A and B must be non-empty")
    if np.intersect1d(A_arr, B_arr).size > 0:
        raise GpvolveError("A and B must be disjoint")

    # Make B absorbing in a local copy of the matrix.
    csr = matrix.tocsr().tolil()
    B_set = set(int(b) for b in B_arr)
    for b in B_arr:
        csr.rows[b] = [int(b)]
        csr.data[b] = [1.0]
    P_mod = csr.tocsr()

    # Compute h_i = Pr[chain absorbed in B | X_0 = i] in the modified chain.
    # Two structural pitfalls to handle:
    #   1. States in a closed transient component (no edge into the absorbing
    #      class) never absorb in B; their h is 0 by definition.
    #   2. States with positive absorption probability to a sink other than
    #      B contribute to the "competing absorption" mass; we treat all
    #      preexisting sinks as the alternative target.
    # Reach the absorbing class via reverse reachability on the directed
    # graph; states that can reach B have h > 0, others have h = 0.

    reach_check = P_mod.tocsr().copy()
    reach_check.setdiag(0)
    reach_check.eliminate_zeros()
    # Reverse reachability = standard BFS on the reverse graph.
    can_reach_B = np.zeros(n, dtype=bool)
    visited = np.zeros(n, dtype=bool)
    stack = list(int(b) for b in B_arr)
    while stack:
        s = stack.pop()
        if visited[s]:
            continue
        visited[s] = True
        can_reach_B[s] = True
        # Predecessors of s in P_mod (states with an edge into s).
        col = reach_check.getcol(s).tocoo()
        for pred in col.row:
            if not visited[pred]:
                stack.append(int(pred))

    h = np.zeros(n, dtype=np.float64)
    h[B_arr] = 1.0
    # Among states that can reach B, factor in competing sinks if any.
    from gpvolve.markov.validation import absorbing_states as _abs_states
    from gpvolve.paths.tpt import forward_committor

    other_sinks = np.asarray(
        [int(i) for i in _abs_states(P_mod) if int(i) not in B_set], dtype=np.int64
    )
    reach_mask = can_reach_B.copy()
    reach_mask[B_arr] = False  # B itself has h=1 by construction (above).
    if other_sinks.size > 0:
        # On the reachable set, q+ with A = other_sinks gives h.
        # forward_committor handles A-not-reaching-B via boundary substitution
        # cleanly when restricted to states that DO reach B (avoids the
        # disconnected closed-component singularity).
        q = forward_committor(P_mod, A=other_sinks, B=B_arr)
        # Substitute q for reachable-and-transient states, leaving 0 elsewhere.
        h[reach_mask] = q[reach_mask]
    else:
        # No competing sinks: every reachable state absorbs in B.
        h[reach_mask] = 1.0

    out = np.full(A_arr.size, np.inf, dtype=np.float64)
    # Source states already in B contribute 0.
    for k, a in enumerate(A_arr):
        if int(a) in B_set:
            out[k] = 0.0

    # Build conditioned chain on the transient support where h > 0 (excluding B).
    eps = 1e-15
    transient_mask = (h > eps) & ~np.isin(np.arange(n), B_arr)
    cond_idx = np.flatnonzero(transient_mask).astype(np.int64)
    if cond_idx.size == 0:
        return out

    # Doob h-transform on the transient block.
    Q_block = P_mod[cond_idx][:, cond_idx].toarray()
    h_cond = h[cond_idx]
    Q_tilde = Q_block * (h_cond[np.newaxis, :] / h_cond[:, np.newaxis])

    # Standard absorbing-chain MFPT on the h-transformed transient block:
    # (I - Q_tilde) m = 1, because exit to B (absorbed) is the residual.
    A_lhs = np.eye(cond_idx.size) - Q_tilde
    rhs = np.ones(cond_idx.size, dtype=np.float64)
    try:
        m_cond = np.linalg.solve(A_lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise ConvergenceError(
            "conditional MFPT linear solve failed; conditioned chain may be singular"
        ) from exc

    for k, a in enumerate(A_arr):
        if int(a) in B_set:
            continue
        if h[a] <= eps:
            out[k] = np.inf
            continue
        pos = int(np.searchsorted(cond_idx, int(a)))
        if pos < cond_idx.size and int(cond_idx[pos]) == int(a):
            out[k] = float(m_cond[pos])
        else:
            out[k] = np.inf
    return out


def quasi_stationary_distribution(
    matrix: sp.spmatrix,
    *,
    absorbing: Iterable[int] | None = None,
    method: str = "auto",
    max_iter: int = 10_000,
    tol: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.int64], float]:
    """Quasi-stationary distribution of an absorbing chain.

    Conditional on non-absorption, an absorbing finite chain whose transient
    block ``Q`` has a unique largest-magnitude eigenvalue ``lambda_1`` (with
    ``|lambda_1| < 1``) converges in distribution to the left Perron vector
    of ``Q``; this is the QSD (Darroch-Seneta 1965). Equivalently, ``u Q =
    lambda_1 u``, ``u >= 0``, ``sum u = 1``, where ``u`` lives on the
    transient class.

    Parameters
    ----------
    matrix:
        Row-stochastic transition matrix.
    absorbing:
        Indices of absorbing states. If ``None``, computed via
        :func:`gpvolve.markov.validation.absorbing_states`.
    method:
        ``"power"`` for left power iteration on ``Q``, ``"eigs"`` for
        ARPACK, or ``"auto"`` (default) which tries power and falls back
        to ARPACK on non-convergence.
    max_iter, tol:
        Power-iteration controls; ignored under ARPACK.

    Returns
    -------
    ``(qsd_full, transient_idx, lambda_1)``. ``qsd_full`` is a length-``n``
    array embedded in the original state space (zeros at absorbing states);
    ``transient_idx`` lists the transient indices; ``lambda_1`` is the
    Perron eigenvalue of ``Q``. ``1 - lambda_1`` is the per-step absorption
    rate from the QSD, so the expected absorption time from QSD is
    ``1 / (1 - lambda_1)``.

    Raises
    ------
    GpvolveError
        If the chain has no absorbing states or the transient block is
        empty.
    ConvergenceError
        If power iteration fails to converge and ARPACK is not requested.

    References
    ----------
    Darroch, J.N. & Seneta, E. (1965). "On quasi-stationary distributions in
    absorbing discrete-time finite Markov chains." *J. Appl. Prob.* 2,
    88-100.
    """
    n = matrix.shape[0]
    abs_idx = _resolve_absorbing(matrix, absorbing)
    if abs_idx.size == 0:
        raise GpvolveError("quasi_stationary_distribution requires at least one absorbing state")
    if abs_idx.size == n:
        raise GpvolveError("all states are absorbing; no transient class on which to define a QSD")

    mask = np.ones(n, dtype=bool)
    mask[abs_idx] = False
    transient_idx = np.flatnonzero(mask).astype(np.int64)

    csr = matrix.tocsr()
    Q = csr[transient_idx][:, transient_idx]

    if method in ("power", "auto"):
        try:
            u, lam = _qsd_power(Q, max_iter=max_iter, tol=tol)
        except ConvergenceError:
            if method == "power":
                raise
            u, lam = _qsd_arpack(Q)
    elif method == "eigs":
        u, lam = _qsd_arpack(Q)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'power', 'eigs', or 'auto'")

    full = np.zeros(n, dtype=np.float64)
    full[transient_idx] = u
    return full, transient_idx, float(lam)


def _qsd_power(Q: sp.spmatrix, *, max_iter: int, tol: float) -> tuple[NDArray[np.float64], float]:
    """Left power iteration on Q: u_{k+1} = u_k Q / |u_k Q|_1.

    Converges geometrically at rate |lambda_2 / lambda_1| for the leading
    real left eigenpair, which is positive on the transient class when Q is
    irreducible (Perron-Frobenius).
    """
    n = Q.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64), 0.0
    u = np.full(n, 1.0 / n, dtype=np.float64)
    Qt = Q.tocsr()  # we'll use u @ Qt = (Qt.T @ u.T).T
    for _ in range(max_iter):
        nxt = Qt.T @ u  # equals (u Q).T as a column
        nxt = np.asarray(nxt).ravel()
        s = nxt.sum()
        if s <= 0:
            raise ConvergenceError(
                "QSD power iteration collapsed to non-positive vector; "
                "transient block may be reducible"
            )
        lam = s  # u_k+1 = u_k Q normalized to sum 1; the unnormalized sum is lambda_1.
        nxt = nxt / s
        if np.abs(nxt - u).sum() < tol:
            return nxt, float(lam)
        u = nxt
    raise ConvergenceError(
        f"QSD power iteration did not converge in {max_iter} steps at tol={tol:g}"
    )


def _qsd_arpack(Q: sp.spmatrix) -> tuple[NDArray[np.float64], float]:
    """Largest left eigenpair of Q via ARPACK on Q^T."""
    Qt = Q.T.tocsr().astype(np.float64)
    try:
        vals, vecs = spla.eigs(Qt, k=1, which="LM", maxiter=10_000)
    except spla.ArpackNoConvergence as exc:
        raise ConvergenceError("ARPACK failed to compute QSD eigenvector") from exc
    u = np.asarray(vecs[:, 0].real, dtype=np.float64)
    if u.sum() < 0:
        u = -u
    s = u.sum()
    if s <= 0:
        raise ConvergenceError("QSD eigenvector summed to non-positive")
    return u / s, float(vals[0].real)
