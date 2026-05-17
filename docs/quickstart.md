# Quickstart

A 30-second tour. Build a 2-site binary map, hand it to gpgraph-v2 for the
neighbor graph, then to gpvolve-v2 for the Markov state model.

```python
from gpmap import GenotypePhenotypeMap
from gpgraph import GenotypePhenotypeGraph
from gpvolve import (
    GenotypePhenotypeMSM,
    forward_committor,
    reactive_flux,
    rate,
    sample_paths,
    timescales,
)

gpm = GenotypePhenotypeMap(
    wildtype="00",
    genotypes=["00", "01", "10", "11"],
    phenotypes=[1.0, 1.5, 1.2, 2.0],
)
graph = GenotypePhenotypeGraph.from_gpm(gpm)
msm = GenotypePhenotypeMSM.from_graph(
    graph, fitness_column="phenotypes", fixation="moran", population_size=100
)
```

`msm` is a dataclass with five locked attributes (see [SCHEMA](https://github.com/lperezmo/gpvolve-v2/blob/main/SCHEMA.md)):

- `transition_matrix`: row-stochastic `csr_matrix`, rows sum to 1.0 within 1e-12
- `stationary`: the stationary distribution
- `gpm`, `graph`: the upstream containers
- `fixation_model`, `fixation_params`: how `transition_matrix` was built

## Spectral analysis

```python
ts = timescales(msm.transition_matrix, k=3)
print(ts)
```

## Transition path theory

```python
q_plus = forward_committor(msm.transition_matrix, A=0, B=3)
flux = reactive_flux(msm.transition_matrix, A=0, B=3)
k_AB = rate(msm.transition_matrix, A=0, B=3)
```

## Stochastic path sampling with convergence

```python
ens = sample_paths(msm, source=0, targets=3, seed=0)
print(ens.metadata["convergence"])
```

`sample_paths` keeps drawing walker chunks until the ESS and Gelman-Rubin
R-hat for each endpoint both clear their thresholds. The default
`ConvergenceCheck()` has `ess_min=200` and `rhat_max=1.01`. Override either
by passing your own `ConvergenceCheck`. See the
[Stochastic sampling guide](guides/stochastic-sampling.md) for the full
story.

## Next steps

- [Build an MSM from a GP map](guides/build-msm.md)
- [Compute evolutionary pathways](guides/compute-pathways.md)
- [Stochastic sampling](guides/stochastic-sampling.md)
- [Coarse-graining with PCCA+](guides/coarse-graining.md)
