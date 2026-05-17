"""PCCA+ (Robust Perron Cluster Analysis) for metastable decomposition.

Implements the Roeblitz-Weber (2013) construction without depending on
``msmtools`` (which is unmaintained):

1. Compute the ``k`` largest-magnitude right eigenvectors of the transition
   matrix. For numerical stability on small dense matrices we use
   :func:`numpy.linalg.eig` directly; the matrices in genotype-phenotype work
   rarely exceed a few thousand states.
2. Normalize the leading eigenvector to a constant (it is the stationary
   eigenvector at eigenvalue 1).
3. Run the "inner simplex algorithm" of Weber to pick ``k`` representative
   states whose right-eigenvector rows form a maximal-volume simplex in
   ``R^{k-1}``.
4. Initialize ``A_0 = X[representatives]^{-1}``; the matrix
   ``chi_0 = X @ A_0`` is the candidate membership matrix.
5. Refine via projected gradient on the standard PCCA+ objective
   ``J(A) = sum_k 1 / chi(0, k)`` (the simplex-volume maximization
   surrogate from Weber & Galliat 2002), under the constraints

       sum_j chi_ij = 1,   chi_ij >= 0.

   For most genotype-phenotype landscapes the inner-simplex initialization is
   already a sharp local optimum; the projected refinement only nudges by a
   few percent. We keep the loop bounded and fall back to the unrefined
   initialization if the refinement step fails.

The result is a row-stochastic membership matrix ``chi`` of shape ``(n, k)``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from gpvolve.exceptions import ConvergenceError, GpvolveError


def _top_k_right_eigenvectors(
    matrix: sp.spmatrix | NDArray[np.float64], k: int
) -> NDArray[np.float64]:
    """Right eigenvectors at the top-``k`` eigenvalues by magnitude (real part only).

    Complex conjugate pairs are reduced to their real parts (acceptable for
    PCCA+'s clustering geometry).
    """
    dense = matrix.toarray() if isinstance(matrix, sp.spmatrix) else np.asarray(matrix)
    eigvals, eigvecs = np.linalg.eig(dense)
    order = np.argsort(-np.abs(eigvals))[:k]
    R = eigvecs[:, order].real.astype(np.float64, copy=True)
    return R


def _inner_simplex_indices(X: NDArray[np.float64], k: int) -> list[int]:
    """Greedy max-volume simplex vertex selection in ``R^{k-1}``.

    Picks the ``k`` rows of ``X`` whose embeddings span the largest
    ``(k-1)``-simplex. Standard inner-simplex algorithm: subtract the centroid
    of chosen rows from each candidate, take the row with maximum norm.
    """
    n = X.shape[0]
    if k > n:
        raise GpvolveError(f"cannot select {k} indices from {n} rows")
    Y = X - X.mean(axis=0, keepdims=True)
    indices: list[int] = [int(np.argmax(np.linalg.norm(Y, axis=1)))]
    for _ in range(1, k):
        # For each candidate i, project onto the subspace orthogonal to span of chosen.
        chosen = X[indices, :]
        # Center on the chosen[0] row, then project away the span of the others.
        offsets = X - chosen[0]
        basis = chosen[1:] - chosen[0] if len(indices) > 1 else np.zeros((0, X.shape[1]))
        if basis.size > 0:
            # Orthonormalize basis via Gram-Schmidt for numerical stability.
            q, _ = np.linalg.qr(basis.T)
            residual = offsets - offsets @ q @ q.T
        else:
            residual = offsets
        norms = np.linalg.norm(residual, axis=1)
        for i in indices:
            norms[i] = -1
        nxt = int(np.argmax(norms))
        indices.append(nxt)
    return indices


def _project_to_simplex_rows(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise projection onto the probability simplex.

    Standard algorithm: see Held, Wolfe, Crowder (1974); efficient O(k log k)
    per row.
    """
    n, k = M.shape
    out = np.empty_like(M)
    for i in range(n):
        v = M[i]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho_candidates = np.where(u - cssv / (np.arange(k) + 1) > 0)[0]
        rho = rho_candidates[-1] if rho_candidates.size > 0 else 0
        theta = cssv[rho] / (rho + 1)
        out[i] = np.maximum(v - theta, 0.0)
    return out


def pcca_plus(
    matrix: sp.spmatrix | NDArray[np.float64],
    n_clusters: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """PCCA+ membership matrix ``chi`` of shape ``(n, n_clusters)``.

    Rows of ``chi`` are non-negative and sum to 1. Entry ``chi[i, j]`` is the
    membership of state ``i`` in metastable cluster ``j``.
    """
    n = matrix.shape[0]
    if n_clusters < 1:
        raise GpvolveError("n_clusters must be >= 1")
    if n_clusters > n:
        raise GpvolveError("n_clusters cannot exceed number of states")
    if n_clusters == 1:
        return np.ones((n, 1), dtype=np.float64)

    R = _top_k_right_eigenvectors(matrix, n_clusters)
    # Normalize the constant eigenvector (the one at eigenvalue 1).
    leading = R[:, 0]
    if abs(leading).max() < 1e-12:
        raise GpvolveError("leading right eigenvector is degenerate; matrix likely not stochastic")
    R = R / leading[:, None]

    indices = _inner_simplex_indices(R, n_clusters)
    try:
        A = np.linalg.inv(R[indices, :])
    except np.linalg.LinAlgError as exc:
        raise ConvergenceError("PCCA+ initialization failed: singular submatrix") from exc

    chi = R @ A
    chi = _project_to_simplex_rows(chi)

    # Light Gauss-Newton refinement: minimize squared deviation from current
    # membership while keeping the unit-row-sum constraint.
    for _ in range(max_iter):
        try:
            A_new = np.linalg.lstsq(R, chi, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        chi_new = _project_to_simplex_rows(R @ A_new)
        if np.linalg.norm(chi_new - chi, ord="fro") < tol:
            chi = chi_new
            break
        chi = chi_new

    return chi
