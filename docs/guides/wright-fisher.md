# Wright-Fisher simulation

Discrete-generation multinomial sampling on the genotype graph. Use this
when the SSWM regime does not apply (weak selection, large mutation rates,
polymorphic populations).

```python
from gpvolve.simulate import wright_fisher

history = wright_fisher(
    graph,
    fitness_column="phenotypes",
    population_size=1000,
    mutation_rate=1e-3,
    n_generations=200,
    initial_index=0,
    seed=0,
)
```

`history` is an `(n_generations + 1, n_states)` int array of per-generation
genotype counts. Row 0 is the initial population (`population_size`
individuals at `initial_index`).

## Algorithm

Each generation:

1. Compute reproductive weights `w_i = f_i * counts_i`.
2. Sample `population_size` offspring multinomially with probabilities
   proportional to `w`.
3. Mutate each offspring with probability `mutation_rate` to a uniformly
   chosen graph neighbor.

The implementation is pure numpy; no scipy or Rust dependencies.

## Gillespie alternative

For exact event times under SSWM-style mutation-and-fixation, use
`simulate.gillespie.gillespie_walk` instead. It samples waiting times from
the exponential of the summed fixation propensities and returns the visited
states with their cumulative times.
