# `gpvolve.simulate`

Population-dynamics simulation backends.

- `wright_fisher(graph, *, fitness_column, population_size, mutation_rate, n_generations, initial_index=0, seed=None) -> NDArray[int64]`
  Discrete-generation multinomial sampling. Returns
  `(n_generations + 1, n_states)` of per-generation genotype counts.

- `gillespie_walk(graph, *, fitness_column, source, targets, fixation="sswm", mu_per_step=1.0, max_steps=10_000, seed=None, **fixation_params) -> tuple[NDArray[float64], NDArray[int64]]`
  Exact Gillespie waiting-time trajectory under the chosen fixation
  kernel. Returns `(times, states)`.

- `slim_available() -> bool`
- `tskit_available() -> bool`

The SLiM and tskit backends are stubs in v2.0.0; install with
`pip install 'gpvolve-v2[sim]'` to make the availability probes return
`True`. The full SLiM integration is on the roadmap.
