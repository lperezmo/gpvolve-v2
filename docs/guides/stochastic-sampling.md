# Stochastic path sampling

`sample_paths` runs Monte Carlo walkers on the MSM until two convergence
criteria for each endpoint are simultaneously satisfied:

1. **ESS via Sokal-windowed integrated autocorrelation.** For each target
   `b`, the indicator `1[walker hit b]` is treated as a time series across
   walker IDs; the integrated autocorrelation time `tau_int` is estimated
   with the standard Sokal windowing. The effective sample size is
   `N / (2 * tau_int)`. Default threshold: `ess_min=200`.
2. **Gelman-Rubin R-hat across chains.** Walkers are split into `m >= 4`
   chains; R-hat is computed on the binary hit indicator. Default
   threshold: `rhat_max=1.01`.

Endpoints with empirical hit probability below `relevance_threshold` (default
`1e-3`) are excluded from the check, so vanishingly-rare targets do not
hold up convergence.

## v1 vs v2

v1 stopped on a Euclidean distance threshold over the empirical path
distribution. That criterion has no statistical interpretation: correlated
samples push the L2 norm down faster than the marginals actually converge,
the distribution is multinomial on a sparse support, and continuing to
sample after the marginals stabilize keeps reducing the L2 norm anyway.

v2 replaces it with ESS + R-hat. KL divergence with bootstrap CI is offered
as a separate diagnostic in `analysis.observables` but is not a stopping
rule (KL is unstable on sparse empirical multinomials with zeros).

## API

```python
from gpvolve import ConvergenceCheck, sample_paths

cc = ConvergenceCheck(
    ess_min=200.0,
    rhat_max=1.01,
    relevance_threshold=1e-3,
    chunk_size=10_000,
    max_walkers=1_000_000,
    n_chains=4,
)
ens = sample_paths(msm, source=0, targets=[31, 63], convergence=cc, seed=0)
```

The returned `PathEnsemble` has `metadata["convergence"]` matching SCHEMA
section 6: `{ess, rhat, n_walkers, n_chunks, converged}`. If the walker
budget is exhausted without convergence, `ConvergenceError` is raised; the
last-known `ess` and `rhat` are in the exception message.

## Recommended budgets

- Tiny maps (`n <= 64`): `chunk_size=200`, `ess_min=50` is usually enough.
- 2^10 to 2^14 binary maps: defaults are fine.
- Larger maps with deep metastable traps: increase `max_walkers` to `5e6`
  and consider running multiple seeds to verify R-hat across independent
  runs.
