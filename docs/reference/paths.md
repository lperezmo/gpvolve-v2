# `gpvolve.paths`

Path-finding, transition path theory, stochastic sampling.

## Pathway finders

- `shortest_paths(msm, source, targets) -> PathEnsemble` (method `"shortest"`)
- `greedy_walk(msm, source, targets, *, max_steps=None) -> PathEnsemble` (method `"greedy"`)
- `dominant_pathways(flux, A, B, *, top_k=10) -> list[PathEnsemble]` (method `"tpt"`)

## TPT primitives

- `forward_committor(matrix, A, B) -> NDArray[float64]`
- `backward_committor(matrix, A, B, *, stationary=None) -> NDArray[float64]`
- `reactive_flux(matrix, A, B, *, stationary=None) -> csr_matrix`
- `net_flux(flux) -> csr_matrix`
- `rate(matrix, A, B, *, stationary=None) -> float`

## Stochastic sampling

```python
@dataclass(frozen=True)
class ConvergenceCheck:
    ess_min: float = 200.0
    rhat_max: float = 1.01
    relevance_threshold: float = 1e-3
    chunk_size: int = 10_000
    max_walkers: int = 1_000_000
    n_chains: int = 4
    max_steps_per_walker: int | None = None
```

- `sample_paths(msm, source, targets, *, convergence=None, seed=None) -> PathEnsemble`
  (method `"stochastic"`; raises `ConvergenceError` if `max_walkers` is
  exhausted without satisfying both ESS and R-hat thresholds)
