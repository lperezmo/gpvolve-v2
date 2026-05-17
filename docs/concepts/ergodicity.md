# Ergodicity and reversibility

A finite-state Markov chain has a unique stationary distribution iff it is
**irreducible** (equivalently, the directed graph of non-zero transitions
is strongly connected). gpvolve-v2 ships two helpers in
`gpvolve.markov.validation`:

```python
from gpvolve.markov import is_strongly_connected, assert_strongly_connected

ok = is_strongly_connected(msm.transition_matrix)
assert_strongly_connected(msm.transition_matrix)  # raises NonStochasticError
```

Self-loops are ignored for the strong-connectivity check, so a matrix that
is row-stochastic only because every entry is absorbed by the diagonal does
not pass.

## Reversibility

A chain is reversible if detailed balance holds:
`pi_i P_ij = pi_j P_ji` for every pair `(i, j)`. Under reversibility:

- the backward committor equals `1 - q+`
- the symmetric similarity `D^{1/2} P D^{-1/2}` (with `D = diag(pi)`) is
  symmetric, so its eigenvectors are orthogonal
- PCCA+ has an analytic refinement step

gpvolve-v2 does not assume reversibility anywhere: the backward committor
is computed by explicitly building the time-reversed chain, and PCCA+ uses
the right eigenvectors of `P` directly rather than the symmetric form.
This makes the analyses work for SSWM/Moran chains on landscapes with
epistasis (which violate detailed balance) at the price of a slightly more
expensive linear algebra path.

## When connectivity fails

The genotype-phenotype graphs gpvolve-v2 works with are usually strongly
connected by construction (every edge in `gpgraph-v2` is added in both
directions). If you build a graph that drops edges for biological reasons
(e.g., one-step accessible only in the wildtype-to-mutant direction), check
the connectivity before running MSM analyses.
