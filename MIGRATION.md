# Migration: gpvolve v1 -> gpvolve-v2

gpvolve-v2 is a clean-break rewrite. There is no compatibility shim. This page maps every
v1 public symbol to its v2 equivalent. v1 here means [`harmslab/gpvolve`](https://github.com/harmslab/gpvolve)
v0.2.0 (Aug 2020), which is the last published release.

## Public API mapping

| v1 | v2 | Notes |
|---|---|---|
| `GenotypePhenotypeMSM(gpm)` | `GenotypePhenotypeMSM.from_graph(GenotypePhenotypeGraph.from_gpm(gpm), fitness_column=..., fixation=...)` | Graph step is explicit |
| `msm.build_transition_matrix(model="moran", ...)` | `build_transition_matrix(graph, fitness_column=..., fixation="moran", population_size=N)` | Free function returning sparse CSR |
| `msm.stationary` (cached property) | `stationary_distribution(msm.transition_matrix)` | No hidden caching |
| `msm.eigenvalues`, `msm.timescales` | `timescales(msm.transition_matrix, k=10)` | Returns top-k decaying modes |
| `msm.tpt(source, target)` | `forward_committor(P, A=[source], B=[target])` then `reactive_flux(P, pi, q_plus, q_minus)` | Decomposed primitives |
| `msm.sample_paths(...)` | `sample_paths(msm, source, [target], n_walkers=..., convergence=ConvergenceCheck())` | Convergence-aware (ESS + R-hat) |
| `gpvolve.utils.*` | `gpvolve.analysis.*` and `gpvolve.io.*` | Reorganized |
| `gpvolve.cluster.pcca` (msmtools shim) | `gpvolve.cluster.pcca_plus` | Native, no msmtools |
| `gpvolve.visualization.*` | `gpvolve.pyplot.*` | Matches sibling naming |

## Fixation model rename

v1 used `model="mcclandish"` (misspelling). v2 uses `fixation="mccandlish"` (correct
spelling), with `mcclandish` retained as an alias only in the registry for back-compat
in user scripts that pin the v1 string. New code should use `mccandlish`.

## Six v1 bugs explicitly fixed

1. `build_transition_matrix()` could be called before `apply_selection()` and produced
   silently wrong results. v2 makes fitness a required keyword arg.
2. Row/column indexing was inconsistent between sparse and dense paths. v2 locks indices
   to `gpm.data.index` via `gpgraph-v2`'s int-keyed contract.
3. Self-loops were sometimes computed from fixation probability of i to i, which is not
   well defined. v2 always computes `P[i,i] = 1 - sum_j P[i,j]` after the off-diagonals
   are filled.
4. Stochastic path sampling convergence was a Euclidean distance on probability vectors,
   which conflated Monte Carlo error with mixing. v2 uses ESS + Gelman-Rubin R-hat.
5. PCCA+ was a thin wrapper on `msmtools`, which is unmaintained. v2 reimplements the
   Roeblitz-Weber algorithm natively.
6. `msm.stationary` was an unstable eigenvector solve for ill-conditioned chains. v2
   uses power iteration first and falls back to shifted ARPACK.

(Commit hashes will be filled in as each fix lands during the v2 rollout.)

## Removed surface

- `gpvolve.flux.*` -- merged into `gpvolve.paths.tpt` and `gpvolve.paths.flux`.
- Direct dependency on `gpmap` (no version pin); v2 requires `gpmap-v2`.
- Direct dependency on `gpgraph` (no version pin); v2 requires `gpgraph-v2`.
- Direct dependency on `msmtools`; dropped entirely.
- Cython hot paths; replaced by Rust via PyO3.
