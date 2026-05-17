# gpvolve-v2

Markov-chain evolutionary dynamics on genotype-phenotype maps.

Given a genotype-phenotype map (`gpmap-v2`) and a graph over it (`gpgraph-v2`), gpvolve-v2
builds a row-stochastic transition matrix using a fixation model (SSWM, Moran, McCandlish,
Bloom DMS, weak-mutation) and analyzes the resulting Markov state model:

- stationary distribution and relaxation timescales
- transition path theory: forward and backward committors, reactive flux
- evolutionary pathways: shortest, greedy, dominant by flux, stochastic ensembles
- PCCA+ metastable clustering
- fitness peak and valley detection
- Wright-Fisher and Gillespie simulation backends

Hot paths (transition matrix assembly and stochastic walker loops) are Rust-accelerated;
spectral analysis stays in scipy.

## Where to start

- New here? Read [Quickstart](quickstart.md).
- Need to install? See [Installation](installation.md).
- Coming from v1? See [Migration from v1](migration/from-v1.md).
- Want theory? Start with the [MSM primer](concepts/msm-primer.md).
