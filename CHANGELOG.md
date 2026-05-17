# CHANGELOG


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
