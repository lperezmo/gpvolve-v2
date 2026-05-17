# CHANGELOG


## v1.1.0 (2026-05-17)

### Features

- **streamlit**: Multi-page showcase app
  ([`7632b53`](https://github.com/lperezmo/gpvolve-v2/commit/7632b539d777c773064536d5a5f9cdd426f4834a))

7-page tour mirroring the gpgraph-v2 / gpmap-v2 style:

intro quickstart + page map msm_builder pick L + fixation + pop_size; show P and pi tpt_explorer
  committor, reactive flux, dominant pathways A->B sampler rayon walker ensemble with live ESS /
  R-hat readout clustering PCCA+ memberships + coarse-grained P + basin overlay benchmarks live
  Rust-vs-Python timings for both hot kernels about links to sibling v2 packages and what is in Rust

Deployable to Streamlit Cloud at gpvolve-v2.streamlit.app via the pinned requirements.txt; the Rust
  crate builds from sdist on Cloud without extra setup. Local dev is uv sync + maturin develop + the
  showcase entrypoint.

README adds the Streamlit Cloud badge pointing to that URL.

### Performance Improvements

- **tpt**: Rust BiCGSTAB for the committor system on large maps
  ([`ab8b765`](https://github.com/lperezmo/gpvolve-v2/commit/ab8b76517a6778cbc9189c4ed2ad9065f9bdafde))

scipy.sparse.linalg.spsolve uses a supernodal LU. On the (I - P_ff) systems the committor reduces
  to, the LU fill-in scales roughly as O(n^2.5):

n=4,096 spsolve = 2,964 ms n=8,192 spsolve = 26,219 ms n=16,384 spsolve timed out at 60 s

BiCGSTAB iterates in O(nnz) and converges in ~50 steps for these near-row-stochastic systems. With
  the same configurations:

n=4,096 BiCGSTAB = 6.9 ms (430x) n=8,192 BiCGSTAB = 14.6 ms (1,800x) n=16,384 BiCGSTAB = 35 ms
  (>1,700x) n=32,768 BiCGSTAB = 77 ms

forward_committor now dispatches to gpvolve._rust.solve_bicgstab_csr when n_free > 256 and the
  extension is available; falls back to spsolve below that threshold (where FFI overhead would
  dominate) or when BiCGSTAB fails to hit tolerance.

Implementation is the textbook van der Vorst 1992 BiCGSTAB with breakdown detection. No assumption
  that A is symmetric or reversible. Zero initial guess; tol=1e-10 relative residual; max_iter=2000.

Verified with tests/unit/test_committor_parity.py: Rust output matches spsolve to within 5e-9 on a
  2^8 binary map. Benchmark suite in tests/benchmarks/test_committor_bench.py.


## v1.0.2 (2026-05-17)

### Bug Fixes

- **lint**: Import order and format on perf commit follow-up
  ([`0c05ed2`](https://github.com/lperezmo/gpvolve-v2/commit/0c05ed21d42624e7c313c35cd9aba836117b482c))

Ruff I001 and ruff format catches from CI that were missed locally.


## v1.0.1 (2026-05-17)

### Performance Improvements

- **sample**: Rayon-parallel walker sampling in Rust
  ([`932f3fb`](https://github.com/lperezmo/gpvolve-v2/commit/932f3fbb96d178b893f107fd4f7b17090cf64aa1))

The per-walker step loop in sample_paths was the only profiled hot path that scaled badly: 24 s for
  5k walkers on a 2^6 binary map, 104 s on 2^14. Everything else (transition matrix assembly,
  stationary, TPT) is already vectorized in numpy/scipy and runs under a second on 2^14 maps.

The Rust kernel sample_paths_csr does three things:

1. Precomputes per-row cumulative distributions once over the CSR data buffer, so each walker step
  is a uniform draw + one binary search. 2. Distributes walkers across rayon worker threads,
  dropping the GIL for the parallel section. 3. Folds the per-walker product of transition
  probabilities into the step loop. The first pass returned only paths and hits and was only ~2.5x
  faster end-to-end because the Python wrapper still did 4.5 million csr[i, j] lookups. Computing
  the product in the step where the probability is already in hand moves the end-to-end gain to
  ~400-500x at 2^14.

Each walker owns an independent Pcg64 stream seeded by splitmix64(seed, walker_id) so runs are
  bitwise reproducible across rayon thread counts.

Numbers (5k walkers, ESS threshold 10, 16 rayon threads):

sites n_states python (ms) rust (ms) speedup 6 64 2451 18 135x 10 1024 n/a 85 ~480x 14 16384 n/a 188
  ~550x

Pure-Python fallback retained as _simulate_chunk_python for sdist installs without a Rust toolchain.
  paths.stochastic dispatches at import time via _RUST_AVAILABLE.

Tested: 119 passed (no regression). New pytest-benchmark suite in tests/benchmarks/ records the
  head-to-head; benchmarks/README.md documents why we Rust-accelerate this one loop and nothing
  else.


## v1.0.0 (2026-05-17)

### Features

- Initial implementation of gpvolve-v2
  ([`95e0d1e`](https://github.com/lperezmo/gpvolve-v2/commit/95e0d1e01652a6c73d1df7abc7fcc9af47943d60))

Markov-chain evolutionary dynamics on genotype-phenotype maps.

Public surface: - GenotypePhenotypeMSM container with locked schema - build_transition_matrix with
  Sailer-Harms (2017) normalization, row-stochasticity asserted to 1e-12, v1 bugs 1 and 2 fixed -
  Stationary distribution via power iteration with ARPACK fallback - Spectral analysis: eigenvalues,
  timescales, mfpt, mixing_time - TPT primitives: forward and backward committors, reactive flux,
  net flux, rate - Pathway finders: shortest_paths, greedy_walk, dominant_pathways, sample_paths -
  Stochastic sampler with ESS + Gelman-Rubin R-hat convergence (no msmtools) - PCCA+ from scratch,
  metastable_sets, crisp_assignments, coarse_grain - Fitness analyses: find_peaks, find_valleys,
  accessible_peaks - Wright-Fisher and Gillespie simulation backends - I/O round-trip: JSON, NPZ,
  pickle - Matplotlib plot helpers under gpvolve.pyplot

Tested: 119 passed, 89% coverage. mypy strict clean, ruff clean, ruff format clean. End-to-end
  example builds an 8-site binary MSM and exercises every public surface in 3.5 s.

Docs: Zensical site builds clean across concepts, guides, reference, and migration nav sections.
