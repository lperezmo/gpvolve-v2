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

Two hot loops live in Rust via PyO3 + rayon: the stochastic walker sampler (~500x
faster at 2^14 states) and the BiCGSTAB committor solver (~1700x faster at 2^14 vs
`scipy.sparse.linalg.spsolve`). Transition-matrix assembly, stationary distributions,
TPT setup and PCCA+ stay in vectorized numpy/scipy where they already run under a
second on 2^14 maps.

## Where to start

- New here? Read [Quickstart](quickstart.md).
- Need to install? See [Installation](installation.md).
- Coming from v1? See [Migration from v1](migration/from-v1.md).
- Want theory? Start with the [MSM primer](concepts/msm-primer.md).
