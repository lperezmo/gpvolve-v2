# Changelog

## 1.0.0 (2026)

Initial release of the v2 line. Clean-break rewrite of the dormant
`harmslab/gpvolve` and `harmsm/gpvolve` Python packages.

### Added

- `GenotypePhenotypeMSM` container with locked schema (SCHEMA section 1).
- `build_transition_matrix` with the Sailer-Harms (2017) discrete-time
  normalization and the row-stochastic invariant asserted to 1e-12.
- Stationary distribution via power iteration with ARPACK fallback.
- Spectral analysis: `eigenvalues`, `timescales`, `mfpt`, `mixing_time`.
- Transition path theory: `forward_committor`, `backward_committor`,
  `reactive_flux`, `net_flux`, `rate`, `dominant_pathways`.
- Pathway finders: `shortest_paths`, `greedy_walk`, `sample_paths` with
  ESS + Gelman-Rubin convergence (no msmtools).
- PCCA+ from scratch: `pcca_plus`, `metastable_sets`, `crisp_assignments`,
  `coarse_grain`.
- Fitness analyses: `find_peaks`, `find_valleys`, `accessible_peaks`.
- Simulation backends: `wright_fisher`, `gillespie_walk`, optional `slim`
  and `tskit_io` stubs.
- I/O round-trips: JSON, NPZ, pickle.
- Matplotlib plot helpers under `gpvolve.pyplot`.

### Fixed (relative to v1)

1. Row/column indexing aligned with `gpm.data.index` (int-keyed contract
   shared with `gpgraph-v2`).
2. Diagonal entries computed as `1 - sum_j off-diagonal`; never evaluate
   the fixation kernel at `f_i == f_j`.
3. Misspelled `mcclandish` kept as an alias of `mccandlish`.
4. Stationary solver no longer assumes reversibility.
5. Convergence check for path sampling replaced with ESS + R-hat.
6. `msmtools` dependency dropped (TPT and PCCA+ reimplemented natively).

### Migration

See [Migration from v1](../migration/from-v1.md) for the symbol map.
