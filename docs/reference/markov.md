# `gpvolve.markov`

The MSM core: transition-matrix assembly, stationary, spectral analysis,
container.

![Relaxation timescales of the MSM from the subdominant eigenvalues, with a spectral gap](../assets/timescales-light.png#only-light)
![Relaxation timescales of the MSM from the subdominant eigenvalues, with a spectral gap](../assets/timescales-dark.png#only-dark)

The implied relaxation timescales come from the subdominant eigenvalues of `P`;
a gap between two of them is the signal that the chain has that many metastable
basins. Mean first passage times answer the complementary question of how long,
on average, the chain takes to first reach each state:

![Mean first passage time matrix between all state pairs shown as a heatmap](../assets/mfpt-matrix-light.png#only-light)
![Mean first passage time matrix between all state pairs shown as a heatmap](../assets/mfpt-matrix-dark.png#only-dark)

## `GenotypePhenotypeMSM`

Frozen-state container holding `(gpm, graph, transition_matrix, stationary,
fixation_model, fixation_params)`. See [SCHEMA](https://github.com/lperezmo/gpvolve-v2/blob/main/SCHEMA.md)
section 1. Construct with `from_graph(graph, fitness_column=..., fixation=..., **params)`.

## `build_transition_matrix(graph, *, fitness_column, fixation, self_loops="absorb", **params) -> csr_matrix`

Build a row-stochastic transition matrix from a graph and a fixation
model. The only `self_loops` mode is `"absorb"`: diagonal is
`1 - sum_j off-diagonal`. Off-diagonal entries are `pi_fix / k_max` where
`k_max` is the maximum out-degree. Raises `NonStochasticError` for
unbounded kernels, `ModelError` for missing params.

## `stationary_distribution(matrix, *, method="auto", max_iter=10_000, tol=1e-12) -> NDArray[float64]`

Power iteration on `P^T` (fast for well-conditioned chains); ARPACK
fallback (`method="eigs"`) for ill-conditioned ones. `method="auto"`
tries power first and falls back on `ConvergenceError`.

## `eigenvalues(matrix, k=10) -> NDArray[complex128]`

Top-`k` eigenvalues by magnitude. Dense `eig` for `n <= 50`; ARPACK for
larger matrices.

## `timescales(matrix, k=10) -> NDArray[float64]`

Relaxation timescales `tau_l = -1 / log|lambda_l|`, slowest first,
excluding the stationary mode.

## `mfpt(matrix, targets) -> NDArray[float64]`

Mean first passage time from every state to the target set. Entries
indexed by `targets` are zero.

## `mixing_time(matrix, *, eps=0.25) -> float`

Spectral-gap-based mixing-time bound.

## Validation helpers

- `is_strongly_connected(matrix) -> bool`
- `assert_strongly_connected(matrix)` (raises `NonStochasticError`)
- `assert_row_stochastic(matrix, *, tol=1e-12)` (raises `NonStochasticError`)
- `assert_nonneg(matrix, *, tol=1e-12)` (raises `NonStochasticError`)
