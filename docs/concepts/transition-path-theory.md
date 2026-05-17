# Transition path theory

TPT gives a probability-flux decomposition of an MSM between two state sets
`A` (source) and `B` (target). gpvolve-v2 implements the Berezhkovskii,
Hummer, Szabo (2009) formulation. No `msmtools` dependency.

## The four quantities

- **Forward committor** `q+_i = P(chain reaches B before A | X_0 = i)`.
  Solved as a linear system with boundary conditions `q+_A = 0`, `q+_B = 1`:

      (I - P_off) q = b

  where `P_off` is `P` with rows in `A` and `B` zeroed.
- **Backward committor** `q-_i = P(chain came from A before B | X_0 = i)`.
  Equal to `1 - q+_i` only for reversible chains. For non-reversible chains
  gpvolve-v2 builds the time-reversed transition matrix explicitly and runs
  the forward solver with swapped boundary sets.
- **Reactive flux** `f_ij = pi_i q-_i P_ij q+_j` for `i != j`. Nonnegative
  and supported on the off-diagonal of the graph.
- **Rate** `k_AB = sum_{i in A, j not in A} pi_i P_ij q+_j`.

## API

```python
from gpvolve import forward_committor, backward_committor, reactive_flux, rate

q_plus  = forward_committor(P, A=source, B=target)
q_minus = backward_committor(P, A=source, B=target)
F       = reactive_flux(P, A=source, B=target)
k       = rate(P, A=source, B=target)
```

All four accept either a single state or an iterable of states for both
`A` and `B`. They must be disjoint.

## Dominant pathways

`dominant_pathways(flux, A, B, top_k=10)` returns the top-k bottleneck
pathways through the reactive flux. The decomposition is the standard
Dijkstra-on-(-log(flux)) maximum-bottleneck-path algorithm, repeated with
the bottleneck flux subtracted at each step (the Karp 1980 procedure).
Pathways are returned as `PathEnsemble` instances sorted by bottleneck flux
descending.
