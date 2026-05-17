# PCCA+ and metastable clustering

PCCA+ (Roeblitz and Weber 2013) decomposes an MSM into `k` metastable
basins. It is the spectral generalization of "find the deep valleys in the
landscape": given the top-`k` right eigenvectors of the transition matrix,
identify `k` vertices of a simplex such that the rows of the eigenvector
matrix project onto a row-stochastic membership matrix `chi` of shape
`(n_states, k)`.

## API

```python
from gpvolve import pcca_plus, metastable_sets, coarse_grain

chi = pcca_plus(msm.transition_matrix, n_clusters=3)
sets = metastable_sets(chi)  # hard argmax assignment per state
P_coarse = coarse_grain(msm.transition_matrix, chi)
```

`chi` rows sum to 1.0 within numerical tolerance and entries are
non-negative. `metastable_sets` returns one int array per cluster; together
they partition the state space.

`coarse_grain` applies the Galerkin projection
`P_coarse = (chi^T D chi)^{-1} chi^T D P chi` (with `D = diag(pi)`) to
produce a row-stochastic transition matrix on the metastable basins.

## Implementation notes

gpvolve-v2 implements PCCA+ from scratch (no `msmtools` dependency):

1. Compute the top-`k` right eigenvectors of `P` (dense eig for small `n`).
2. Normalize the leading eigenvector to a constant.
3. Inner-simplex algorithm to pick `k` representative states.
4. Form the candidate `chi` matrix and project onto the row-stochastic
   simplex.
5. Light Gauss-Newton refinement of the rotation matrix `A`.

The implementation works for non-reversible chains. For reversible chains
the result matches `msmtools` PCCA+ to within projection-step tolerance.
