# Transition path theory

TPT gives a probability-flux decomposition of an MSM between two state sets
`A` (source) and `B` (target). gpvolve-v2 implements the Berezhkovskii,
Hummer, Szabo (2009) formulation. No `msmtools` dependency.

## The four quantities

**Forward committor.**
$q^{+}_{i} = \mathbb{P}(\text{chain reaches } B \text{ before } A \mid X_0 = i)$.
Solved as a linear system with boundary conditions $q^{+}_{A} = 0$ and
$q^{+}_{B} = 1$:

$$
(I - P_{\text{off}}) \, q \;=\; b
$$

where $P_{\text{off}}$ is $P$ with rows in $A$ and $B$ zeroed.

**Backward committor.**
$q^{-}_{i} = \mathbb{P}(\text{chain came from } A \text{ before } B \mid X_0 = i)$.
Equal to $1 - q^{+}_{i}$ only for reversible chains. For non-reversible
chains gpvolve-v2 builds the time-reversed transition matrix explicitly
and runs the forward solver with swapped boundary sets.

**Reactive flux.**

$$
f_{ij} \;=\; \pi_i \, q^{-}_{i} \, P_{ij} \, q^{+}_{j}, \qquad i \neq j
$$

Nonnegative and supported on the off-diagonal of the graph.

**Rate.**

$$
k_{AB} \;=\; \sum_{i \in A,\; j \notin A} \pi_i \, P_{ij} \, q^{+}_{j}
$$

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
