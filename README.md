# gpvolve-v2

[![CI](https://github.com/lperezmo/gpvolve-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/lperezmo/gpvolve-v2/actions/workflows/ci.yml)
[![Documentation](https://github.com/lperezmo/gpvolve-v2/actions/workflows/docs.yml/badge.svg)](https://lperezmo.github.io/gpvolve-v2/)
[![PyPI](https://img.shields.io/pypi/v/gpvolve-v2.svg)](https://pypi.org/project/gpvolve-v2/)
[![Python](https://img.shields.io/pypi/pyversions/gpvolve-v2.svg)](https://pypi.org/project/gpvolve-v2/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gpvolve-v2.streamlit.app)

Markov-chain evolutionary dynamics on genotype-phenotype maps. Rust-accelerated.

gpvolve-v2 lifts a [gpgraph-v2](https://github.com/lperezmo/gpgraph-v2) `GenotypePhenotypeGraph`
into a Markov state model and analyzes evolutionary dynamics on top of it. Given a fitness
function and a fixation model (SSWM, Moran, McCandlish, Bloom DMS, weak-mutation), it builds
a row-stochastic transition matrix and computes stationary distributions, relaxation
timescales, mean first passage times, transition path theory committors and reactive flux,
dominant pathways, stochastic walker trajectories, PCCA+ metastable sets, and fitness peaks.

This is a clean-break rewrite of [harmslab/gpvolve](https://github.com/harmslab/gpvolve)
(dormant since 2020) with selective architectural inspiration from the
[harmsm/gpvolve](https://github.com/harmsm/gpvolve) fork (dormant since 2022). The two
hot paths that actually scaled badly in pure Python live in Rust via PyO3 + rayon: the
stochastic walker sampler (~500x faster at 2^14 states) and the BiCGSTAB committor
solver (~1700x faster at 2^14 vs `scipy.sparse.linalg.spsolve`). Everything else
(transition matrix assembly, stationary distributions, TPT setup, PCCA+) stays in
vectorized numpy/scipy where it already runs under a second on 2^14 maps. See
[`benchmarks/README.md`](benchmarks/README.md) for the measured numbers and the rationale
for what is and is not in Rust.

## Try it in the browser

The [Streamlit showcase](https://gpvolve-v2.streamlit.app) tours every public surface
interactively: MSM builder, TPT explorer, stochastic sampler with live ESS / R-hat,
PCCA+ clustering, and a live Rust-vs-Python benchmark tab. Source under
[`examples/streamlit/`](examples/streamlit/).

## Why v2

- **Drops the dormant `msmtools` dependency.** PCCA+ and TPT are reimplemented natively.
- **Fast where it matters.** Walker sampling and the committor solver run in Rust with
  rayon parallelism; the rest stays in vectorized numpy/scipy.
- **Sound convergence.** Stochastic path sampling stops on a real criterion (effective sample
  size + Gelman-Rubin R-hat), not a Euclidean distance heuristic.
- **Typed.** Full type hints, `mypy --strict` in CI.
- **Modern tooling.** `uv` + `maturin` + `pyproject.toml`. Releases via
  `python-semantic-release`. OIDC-based PyPI publishing.
- **Consumes the v2 family.** Hard deps on `gpmap-v2` and `gpgraph-v2`; speaks their locked
  `SCHEMA.md` contracts.

## Install

```bash
pip install gpvolve-v2
```

Or with uv:

```bash
uv add gpvolve-v2
```

Plotting support is optional. For matplotlib:

```bash
pip install "gpvolve-v2[plot]"
```

Optional simulation backends (SLiM, tskit):

```bash
pip install "gpvolve-v2[sim]"
```

Python 3.11+. Prebuilt wheels ship for Linux (x86_64, aarch64), macOS (x86_64, aarch64),
and Windows (x64).

## Status

Pre-release. See `CHANGELOG.md` for progress and `docs/migration/from-v1.md` for the
v1 to v2 API map.
