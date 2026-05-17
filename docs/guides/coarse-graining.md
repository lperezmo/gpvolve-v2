# Coarse-graining with PCCA+

Reduce a large MSM to a small one over metastable basins, two function
calls:

```python
from gpvolve import pcca_plus, coarse_grain

chi = pcca_plus(msm.transition_matrix, n_clusters=3)
P_coarse = coarse_grain(msm.transition_matrix, chi)
```

`chi` is the `(n, k)` membership matrix (row-stochastic, non-negative).
`P_coarse` is the `(k, k)` row-stochastic Galerkin projection.

## Inspecting the basins

```python
from gpvolve import metastable_sets

sets = metastable_sets(chi)
for j, indices in enumerate(sets):
    print(f"cluster {j}: {len(indices)} states")
```

`metastable_sets` is the hard-argmax assignment. Combine it with
`find_peaks` to identify which basin each fitness peak ends up in.

## Picking `n_clusters`

The classic heuristic is the implied-timescales gap: compute timescales
`tau_k` for the first several modes, look for a clear separation between
`tau_{k-1}` and `tau_k`. The `k` for which the gap is largest is a
reasonable choice.

```python
from gpvolve import timescales

ts = timescales(msm.transition_matrix, k=10)
print(ts)
```

For small maps with obvious topology (two-peak landscape, single fitness
ridge) the right `k` is often pickable by eye.
