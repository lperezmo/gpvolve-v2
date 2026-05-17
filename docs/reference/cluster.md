# `gpvolve.cluster`

PCCA+ metastable clustering.

- `pcca_plus(matrix, n_clusters, *, max_iter=100, tol=1e-8) -> NDArray[float64]`
  Returns the `(n, n_clusters)` row-stochastic membership matrix `chi`.
  Rows non-negative, sum to 1.

- `metastable_sets(chi) -> list[NDArray[int64]]`
  Hard argmax assignment per cluster.

- `crisp_assignments(chi) -> NDArray[int64]`
  Argmax cluster label per state.

- `coarse_grain(matrix, chi, *, stationary=None) -> NDArray[float64]`
  Galerkin-projected `(k, k)` coarse-grained transition matrix.
