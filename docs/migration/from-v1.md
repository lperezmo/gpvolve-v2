# From gpvolve v1

gpvolve-v2 is a clean-break rewrite. There is no compatibility shim. The
table below maps every v1 public symbol to its v2 equivalent. v1 here
means [`harmslab/gpvolve`](https://github.com/harmslab/gpvolve) v0.2.0
(Aug 2020).

## Public API mapping

| v1 | v2 |
|---|---|
| `GenotypePhenotypeMSM(gpm)` | `GenotypePhenotypeMSM.from_graph(GenotypePhenotypeGraph.from_gpm(gpm), fitness_column=..., fixation=...)` |
| `msm.build_transition_matrix(model="moran", ...)` | `build_transition_matrix(graph, fitness_column=..., fixation="moran", population_size=N)` |
| `msm.stationary` (cached property) | `stationary_distribution(msm.transition_matrix)` |
| `msm.eigenvalues`, `msm.timescales` | `timescales(msm.transition_matrix, k=10)` |
| `msm.tpt(source, target)` | `forward_committor(P, A=..., B=...)`, then `reactive_flux(P, A=..., B=...)` |
| `msm.sample_paths(...)` | `sample_paths(msm, source, [target], convergence=ConvergenceCheck())` |
| `gpvolve.utils.*` | `gpvolve.analysis.*` and `gpvolve.io.*` |
| `gpvolve.cluster.pcca` (msmtools shim) | `gpvolve.cluster.pcca_plus` |
| `gpvolve.visualization.*` | `gpvolve.pyplot.*` |

## Fixation model rename

v1 used `model="mcclandish"` (misspelled). v2 uses
`fixation="mccandlish"`, with `mcclandish` retained as an alias only so
existing v1 scripts keep working. New code should use `mccandlish`.

## Six v1 bugs explicitly fixed

1. `build_transition_matrix` could be called before `apply_selection` and
   produced silently wrong results. v2 makes `fitness_column` a required
   keyword arg.
2. Row/column indexing was inconsistent between sparse and dense paths.
   v2 locks indices to `gpm.data.index` via `gpgraph-v2`'s int-keyed
   contract.
3. Self-loops were sometimes computed from the fixation probability of
   `i` to `i`, which is not well-defined. v2 always computes
   `P[i, i] = 1 - sum_j P[i, j]` after the off-diagonals are filled.
4. Stochastic path sampling convergence was a Euclidean distance on
   probability vectors, which conflates Monte Carlo error with mixing.
   v2 uses ESS + Gelman-Rubin R-hat.
5. PCCA+ was a thin wrapper on `msmtools`, which is unmaintained. v2
   reimplements the Roeblitz-Weber algorithm natively.
6. `msm.stationary` used an unstable eigenvector solve for ill-conditioned
   chains. v2 uses power iteration first and falls back to shifted ARPACK.

## Removed surface

- `gpvolve.flux.*` merged into `gpvolve.paths.tpt` and `gpvolve.paths.flux`.
- Direct dependency on `gpmap` (now `gpmap-v2`).
- Direct dependency on `gpgraph` (now `gpgraph-v2`).
- Direct dependency on `msmtools` (dropped entirely).
- Cython hot paths (replaced by Rust via PyO3 in upcoming releases).
