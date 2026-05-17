# Compute evolutionary pathways

Four families of pathways. Pick the one that matches your question.

## `shortest_paths` (deterministic, single path per target)

The shortest path is the path with maximum product of transition
probabilities (equivalently, minimum sum of $-\log P_{ij}$):

```python
from gpvolve import shortest_paths

ens = shortest_paths(msm, source=0, targets=[31, 63])
for path, prob in zip(ens.paths, ens.probabilities, strict=True):
    print(path, prob)
```

## `greedy_walk` (deterministic, single max-prob walk)

Greedy descent on the transition matrix: at each step, hop to the neighbor
with the largest off-diagonal `P_ij`. Stops on a target hit, fixed point,
or cycle:

```python
from gpvolve import greedy_walk

ens = greedy_walk(msm, source=0, targets=63)
print(ens.metadata["hit_target"])
```

## `dominant_pathways` (deterministic, top-k by reactive flux)

Decompose the reactive flux from `A` to `B` into bottleneck pathways:

```python
from gpvolve import reactive_flux, dominant_pathways

F = reactive_flux(msm.transition_matrix, A=0, B=63)
pathways = dominant_pathways(F, A=0, B=63, top_k=10)
```

Each returned `PathEnsemble` has its bottleneck flux in
`probabilities[0]`. The list is sorted by bottleneck flux descending.

## `sample_paths` (stochastic, ensemble with convergence check)

See the [stochastic sampling guide](stochastic-sampling.md).
