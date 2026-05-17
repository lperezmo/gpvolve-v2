"""Landing page."""

from __future__ import annotations

import streamlit as st

st.markdown("### gpvolve-v2")

st.markdown(
    """
A Rust-accelerated toolkit for **Markov-chain evolutionary dynamics** on
[gpmap-v2](https://github.com/lperezmo/gpmap-v2) genotype-phenotype maps.
Built on top of [gpgraph-v2](https://github.com/lperezmo/gpgraph-v2) for
the static graph topology; gpvolve adds the dynamics layer.

### Three-line quickstart

```python
from gpmap import GenotypePhenotypeMap
from gpgraph import GenotypePhenotypeGraph
from gpvolve import GenotypePhenotypeMSM, forward_committor

gpm = GenotypePhenotypeMap(
    wildtype="00",
    genotypes=["00", "01", "10", "11"],
    phenotypes=[1.0, 1.5, 1.2, 2.0],
)
graph = GenotypePhenotypeGraph.from_gpm(gpm)
msm = GenotypePhenotypeMSM.from_graph(
    graph, fitness_column="phenotypes", fixation="moran", population_size=100
)
q = forward_committor(msm.transition_matrix, A=0, B=3)
```

### What the pages show

- **MSM builder** -- pick a landscape and a fixation model; inspect the
  resulting transition matrix and stationary distribution.
- **TPT explorer** -- pick source and target states; compute committors,
  reactive flux, and the A-to-B rate.
- **Stochastic sampler** -- run rayon-parallel walker ensembles and watch
  the ESS / Gelman-Rubin convergence criteria fire.
- **PCCA+ clustering** -- decompose the chain into metastable basins
  without ``msmtools``.
- **Benchmarks** -- live Rust-vs-scipy timings for the walker sampler
  and the BiCGSTAB committor solver.
"""
)
