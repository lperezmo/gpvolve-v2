# gpvolve-v2 Streamlit showcase

A multi-page Streamlit app touring the gpvolve-v2 API end to end.
Deployed at [gpvolve-v2.streamlit.app](https://gpvolve-v2.streamlit.app).

## Pages

1. **Intro** -- what gpvolve-v2 is and a three-line quickstart.
2. **MSM builder** -- pick a landscape, fixation model and population
   size; see the transition matrix and stationary distribution.
3. **TPT explorer** -- source / target picker, committor, reactive flux
   heatmap, top dominant pathways.
4. **Stochastic sampler** -- run rayon-parallel walker ensembles, watch
   the ESS and Gelman-Rubin convergence stats land.
5. **PCCA+ clustering** -- decompose the chain into metastable basins
   and inspect the coarse-grained transition matrix.
6. **Benchmarks** -- live timings for the two Rust hot paths
   (walker sampling, BiCGSTAB committor).
7. **About** -- pointers to the sibling v2 packages and what is in
   Rust vs Python.

## Run locally

```bash
uv sync
uv run maturin develop --release
uv run streamlit run examples/streamlit/showcase.py
```

## Streamlit Cloud

`requirements.txt` pins everything Streamlit Cloud needs. The Rust crate
is built from sdist via maturin when Cloud installs gpvolve-v2; no
extra setup required.
