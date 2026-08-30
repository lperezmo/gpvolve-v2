# CHANGELOG


## v1.2.1 (2026-08-30)

### Bug Fixes

- **deps**: Patch crossbeam-epoch pointer formatting unsoundness
  ([`8fbefcc`](https://github.com/lperezmo/gpvolve-v2/commit/8fbefcc3029e54e6d97974129b2e2cf3df0800b3))

- **deps**: Refresh vulnerable Python dependency locks
  ([`082ec03`](https://github.com/lperezmo/gpvolve-v2/commit/082ec0388f56f9090e565a65b8630de59931d933))

### Chores

- Bump pyo3, starlette, python-multipart for security alerts
  ([`c035d57`](https://github.com/lperezmo/gpvolve-v2/commit/c035d57edb1d2d37285df84cf1945fad8e1e2f71))

Resolve open Dependabot alerts across both ecosystems.

pip (uv.lock, transitive via streamlit optional extra): - starlette 1.0.0 -> 1.3.1 -
  python-multipart 0.0.28 -> 0.0.32

rust (Cargo.toml + Cargo.lock): - pyo3 0.28 -> 0.29 (numpy bumped to 0.29 in lockstep)

cargo check passes with no source changes; uv lock resolves cleanly.

- Replace broken static.streamlit.io badge with shields.io
  ([`e056040`](https://github.com/lperezmo/gpvolve-v2/commit/e0560401a85162f37c8255c81bae8b0e28a1220c))

- Ruff format + sort conftest imports for CI lint
  ([`a41d2b5`](https://github.com/lperezmo/gpvolve-v2/commit/a41d2b5f185be1e70fb8ef673d26180c17e8e57d))

CI lint on the prior push caught: - conftest.py: imports inside the fuji_5_sswm_msm fixture were not
  grouped by stdlib/third-party order. - absorbing.py and test_absorbing.py: lines longer than the
  project's ruff format width.

All four CI checks now pass locally: ruff check, ruff format --check, mypy, pytest
  --cov-fail-under=80 (170 passed, 87% coverage).

- **docs, streamlit**: Inline math instead of code for symbolic identifiers
  ([`041bc81`](https://github.com/lperezmo/gpvolve-v2/commit/041bc8109f15bc334a1ffbdf15e09054d5f1c1f8))

Several spots wrote mathematical identifiers in markdown code spans, which makes them render as
  fixed-width code rather than italic math with proper subscripts. The R-hat / Tau / N rendering is
  fine; the green-highlighted spots in the user-flagged screenshots are these:

- examples/streamlit/app_pages/sampler.py: \text{ess\_min} and \text{rhat\_max} inside the LaTeX
  math block looked like awkward italics with mid-name escaped underscores. Moved the variable names
  out of math and into markdown code spans so they read as Python config names: ESS ... \geq
  `ess_min`, R-hat \leq `rhat_max`. - docs/concepts/fixation-models.md: pair of fitnesses `(f_i,
  f_j)` was a code span; switched to $(f_i, f_j)$ so the subscripts render. Same for the bare
  allele/background indices `i` and `j`. - docs/guides/compute-pathways.md: `P_ij` -> $P_{ij}$. -
  docs/guides/build-msm.md: `f_i == f_j` -> $f_i = f_j$ in the prose describing what the kernel is
  not evaluated at.

PCCA+ row-stochastic mid-word wrap on narrow mobile (third screenshot) is a Zensical/CSS hyphenation
  artifact, not a markdown issue; leaving the source alone so we do not regress wider viewports.

- **streamlit**: Fix sampler crash, theme config, tighter headers
  ([`ef7aabe`](https://github.com/lperezmo/gpvolve-v2/commit/ef7aabe7c347579f9ca75d2bed1cf4c1d191e46a))

- examples/streamlit/.streamlit/config.toml: themed colors + Inter and JetBrains Mono typography
  copied from epistasis-v2 so the showcase matches the rest of the v2 family instead of the default
  look. - app_pages/sampler.py: fix ValueError "unexpected '{' in field name" caught by Streamlit
  Cloud at runtime. The page composed a raw string containing LaTeX braces (\tau_{\text{int}},
  \text{ess\_min}) and then called .format() on it; Python's format parser saw the LaTeX braces as
  malformed field names and refused. Split the markdown into a static raw-string block for the math
  and a separate f-string for the dynamic Rust-backend status so the two never collide. -
  showcase.py: inject .block-container { padding-top: 1.25rem } via st.markdown so the showcase
  loses the chunky default top whitespace, matching the epistasis-v2 entrypoint pattern. - All 8
  pages: replace st.title("...") with st.markdown("### ...") so the page headers are H3 rather than
  the oversized default H1.

### Documentation

- Add light/dark gallery images to docs and README
  ([`c002f83`](https://github.com/lperezmo/gpvolve-v2/commit/c002f83cb93dfbe08dddbc38b715f869ada8e656))

Add transparent-background figures that adapt to light and dark themes: docs pages pair them with
  #only-light / #only-dark, the README uses <picture> with prefers-color-scheme (absolute raw URLs
  so they also resolve on PyPI, where <picture> degrades to the light <img>).

Images: stationary-distribution hero graph, transition-matrix heatmap, TPT committor gradient and
  reactive flux on the graph, PCCA+ metastable clusters, stochastic walker trajectories, sampler
  convergence (ESS and R-hat), relaxation timescales, MFPT matrix, and the Rust-vs-Python benchmark.
  Transparent backgrounds blend into any page background without a visible seam.

Docs-only change; no package code touched.

- Make graph node labels legible in both light and dark mode
  ([`0704139`](https://github.com/lperezmo/gpvolve-v2/commit/0704139b05f0a475e7ff72952b9a97866a81e078))

The Hamming-graph figures (stationary distribution, committor, reactive flux, walker trajectories)
  colored each node by a scalar (pi, q+, phenotype) but drew the genotype label and node outline in
  a single per-variant ink: black in the light PNG, white in the dark PNG. That ink matched some of
  the node fills exactly, so labels vanished: black text on the near-black low-pi nodes in light
  mode, white text on the yellow peak node in dark mode.

Node label and outline color are now chosen per node from each node's own fill luminance (dark ink
  on light fills, light ink on dark fills). The fill is identical in both PNG variants, so the
  chosen ink is too, and every label stays readable on whatever page background the docs or README
  use. No manual color overrides needed.


## v1.2.0 (2026-05-17)

### Chores

- **examples**: Absorbing-chain streamlit page + cli demo
  ([`c9b6b6f`](https://github.com/lperezmo/gpvolve-v2/commit/c9b6b6f6bfd7f1a3eb04c6a44396939b33af1cd2))

- examples/02_absorbing_chain_toolkit.py: runnable end-to-end demo of the new toolkit on the L=5
  SSWM bug case. Prints ergodicity flags, the standard-TPT failure mode, then the absorbing-chain
  answers (absorption rate, conditional MFPT, fundamental matrix trace, QSD Perron eigenvalue with
  metastable lifetime). About 79 steps from AAAAA to TTTTT, QSD peaks at TTTTA (the one-mutation
  bottleneck). - examples/streamlit/app_pages/absorbing_analysis.py (new): showcase page exposing
  ergodicity flags, absorbing-state list, absorption rate, conditional MFPT, QSD scatter, and
  fundamental-matrix heatmap. Gracefully handles ergodic chains with an info message. -
  examples/streamlit/app_pages/tpt_explorer.py: side-by-side display of reactive rate and absorption
  rate; on absorbing chains, the reactive rate underflows to zero so the new MFPT-based rate is the
  meaningful number. The warning text now explains the contrast instead of silently showing 0.0. -
  examples/streamlit/showcase.py: wires the new page into the nav.

- **streamlit**: Render math with LaTeX in showcase pages
  ([`96cbb04`](https://github.com/lperezmo/gpvolve-v2/commit/96cbb043c5a7f20c13dd86d1451674c9c6681302))

Switch prose-style expressions to KaTeX math in st.markdown so the landing copy on the TPT explorer,
  MSM builder, sampler, and PCCA+ clustering pages renders as proper equations instead of
  code-styled text.

- **tests**: Cover validators, absorbing-chain toolkit, edge cases
  ([`b9b3ab1`](https://github.com/lperezmo/gpvolve-v2/commit/b9b3ab1a0ed2639270838a760b6a84c2b57b7758))

New test files: - tests/unit/test_validation.py covers is_ergodic (including a 3-cycle that
  exercises "irreducible but periodic"), is_reversible on Moran (detailed balance holds) and SSWM
  (rejected via support asymmetry), and absorbing_states with multi-peak landscapes. -
  tests/unit/test_absorbing.py covers fundamental_matrix against a closed-form 2x2 inverse,
  absorption_probabilities against the analytic 5/6 vs 1/6 split on the two-sink chain,
  quasi_stationary_distribution with QSD lifetime cross-checked against MFPT-from-QSD,
  conditional_mfpt agreeing with mfpt when B is the only sink, absorption_rate finiteness on the L=5
  SSWM bug case, plus the four edge cases requested: multi-peak SSWM (reducible), singleton A and B
  at landscape extremes, numerically tiny stationary entries (L=4 SSWM), and a fully disconnected
  block-diagonal chain.

Extensions to tests/unit/test_paths.py: backward_committor raises the new informative error on a
  non-ergodic chain, forward_committor and rate still work on the same absorbing chain.

Fixtures (tests/conftest.py): rugged_gpm_8 / rugged_graph_8 with two deliberate local maxima, and
  fuji_5_sswm_msm reproducing the L=5 SSWM bug case end to end.

170/170 tests pass.

### Documentation

- Align README and Zensical site with shipped Rust hot paths
  ([`fbaa5b5`](https://github.com/lperezmo/gpvolve-v2/commit/fbaa5b5edec5fa4cb96e5015524e334a0ab59c13))

- README and docs/index.md were claiming transition-matrix assembly was Rust-accelerated. It is not;
  only the walker sampler and the BiCGSTAB committor solver are. Replaced with the actual scope plus
  the measured speedup numbers, with a pointer to benchmarks/README.md. - README adds a Try it in
  the browser section pointing at the Streamlit showcase the badge advertises. -
  docs/installation.md: replaced the misleading single line about the [streamlit] extra with a
  section explaining where the showcase lives and how to run it locally vs install just the runtime
  deps. - docs/reference/changelog.md was a hand-written 1.0.0-only mirror of CHANGELOG.md. Replaced
  with a thin pointer to the auto-generated CHANGELOG.md and the GitHub Releases page so it cannot
  drift again.

- Extend ergodicity guide with absorbing-chain toolkit
  ([`a8d59ec`](https://github.com/lperezmo/gpvolve-v2/commit/a8d59ec4260af6392fbb9be8f0a2568d2f197c3d))

Adds a top-level summary of the new validators (is_irreducible, is_ergodic, is_reversible) with the
  aperiodicity criterion explained, and a substantial new section on absorbing chains covering:

- the Kemeny-Snell block decomposition of P into Q, R, and I, - a function table mapping
  fundamental_matrix, absorption_probabilities, quasi_stationary_distribution, conditional_mfpt, and
  absorption_rate to their textbook references, - a worked example wiring the toolkit end to end on
  an SSWM chain, - a contrast between the reactive rate (TPT, ergodic) and the absorption rate
  (MFPT, absorbing) explaining when each applies and why the reactive rate reads zero on SSWM
  chains.

- Render equations with MathJax across concepts and guides
  ([`44f09af`](https://github.com/lperezmo/gpvolve-v2/commit/44f09af4226901f66cd5014615cfc5297aa10725))

Wire up MathJax via extra_javascript and rewrite the prose-style equations in the MSM primer,
  transition path theory, PCCA+ clustering, ergodicity, and compute-pathways pages as proper inline
  and display LaTeX. Indented code-block equations no longer render as monospace fragments.

### Features

- **markov,paths**: Comprehensive absorbing-chain toolkit
  ([`7f78a5d`](https://github.com/lperezmo/gpvolve-v2/commit/7f78a5d557197c7ed9333c636d75a2c78472e7c9))

Adds first-class support for absorbing Markov chains, which arise naturally under SSWM dynamics
  (every local fitness maximum becomes a true sink). The standard TPT machinery requires an ergodic
  chain with strictly positive stationary distribution and refuses to operate on absorbing chains;
  this toolkit fills the gap with textbook constructs.

Library additions - markov.validation: is_irreducible, is_ergodic, is_reversible, absorbing_states,
  transient_states. is_ergodic checks irreducibility plus a positive diagonal (sufficient
  aperiodicity criterion). is_reversible does a support-asymmetry test before the detailed-balance
  equality, so it correctly rejects SSWM kernels even when the stationary distribution underflows to
  numerically tiny but technically positive values. - markov.absorbing (new module):
  fundamental_matrix N = (I - Q)^-1 (Kemeny and Snell 1976), absorption_probabilities B = N R for
  basin assignment in multi-peak landscapes, quasi_stationary_distribution (Darroch and Seneta 1965)
  returning the metastable Perron pair on the transient class, conditional_mfpt via Doob's
  h-transform with reverse-reachability handling for closed transient components. - paths.tpt:
  absorption_rate(P, A, B, *, initial=None) returning 1/E[tau_B] (Hanggi, Talkner, Borkovec 1990)
  for the irreversible A to B rate constant. Routes through conditional_mfpt so it works on
  multi-peak landscapes too.

Bug fixes - backward_committor: error now names the cause (non-ergodic chain, absorbing states
  common under SSWM on single-peak landscapes) and points the caller at forward_committor and rate
  which remain well-defined. - mfpt: previously returned nan when the chain had absorbing states
  outside the target set (singular (I - Q) m = 1 system). Now raises GpvolveError pointing at
  conditional_mfpt. Behavior change: callers checking np.isnan(m) need to wrap in try/except
  instead. - _coerce_set in paths.tpt: accepts numpy integer scalars in addition to Python int, so
  absorbing_states()[k] flows directly into A or B.

References - Kemeny and Snell 1976, Finite Markov Chains, Ch 3. - Darroch and Seneta 1965, J. Appl.
  Prob. 2, 88 to 100. - Hanggi, Talkner, Borkovec 1990, Rev. Mod. Phys. 62, 251 to 341. - Norris
  1997, Markov Chains, Ch 4 (h-transform).


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
