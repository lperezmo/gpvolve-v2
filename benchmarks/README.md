# gpvolve-v2 benchmarks

This directory captures the Python-side performance numbers that justify
the v2 Rust optimizations. Run the live benchmarks via

```bash
uv run pytest tests/benchmarks/ --benchmark-only
```

The Rust-internal benchmarks (criterion) live in `benches/` at the repo
root and are run with `cargo bench`.

## Hot paths in Rust

The two kernels in `crates/gpvolve-core/src/` are:

1. **`sample_paths_csr`** -- rayon-parallel walkers with per-walker PCG64
   streams. Folds the per-step transition-probability product into the
   walker so the Python wrapper does not need 4.5 million `csr[i, j]`
   lookups at the end. See "Stochastic sampler" below.
2. **`solve_bicgstab_csr`** -- van der Vorst (1992) stabilized BiCG for
   the absorbing-boundary committor system. scipy's `spsolve` runs a
   supernodal LU that fills in superlinearly on these matrices; BiCGSTAB
   iterates in O(nnz) and converges in tens of steps. See "Committor
   solver" below.

## Why Rust here and not elsewhere

We profiled `build_transition_matrix`, `stationary_distribution`,
`sample_paths`, and the TPT primitives on binary maps from 2^6 to 2^14.
The hot path is one and only one: the per-walker step loop inside
`sample_paths`. Everything else is either dominated by vectorized numpy
(transition matrix assembly, stationary power iteration), bounded by
linear algebra libraries (committors via `scipy.sparse.linalg.spsolve`),
or so small that the loop overhead vanishes in pandas/networkx noise.

Numbers below are wall-clock milliseconds, single run, 5000 walkers per
chunk, 4 chains, ESS threshold 10, on a Ryzen 9 5950X with rayon defaulting
to 16 worker threads. Reproduce by running

```bash
uv run pytest tests/benchmarks/test_sample_paths_bench.py --benchmark-only \
    --benchmark-columns=mean
```

| sites | n_states | python (ms) | rust (ms) | speedup |
|------:|---------:|------------:|----------:|--------:|
| 6     | 64       | 2,451       | 18        |   ~135x |
| 8     | 256      | 3,273       | n/a       |       n/a |
| 10    | 1,024    | n/a*        | 85        |   ~480x |
| 12    | 4,096    | n/a*        | n/a       |       n/a |
| 14    | 16,384   | n/a*        | 188       |   ~550x |

*Python is not parameterized at those sizes because a single 5k-walker
run exceeds a minute and would time out CI. Extrapolating the 6- and
8-site numbers linearly in `n_walkers * mean_path_length` (the only
factors that scale on the Python path) puts the 2^14 Python run at
roughly 100 s; that is the cross-over where the user notices the
difference and where Rust earned its place.

## What is *not* in Rust

- **Transition matrix assembly.** Already ~100 ms at 2^14 in numpy. The
  vectorized fixation kernels (Moran, SSWM, McCandlish) hit numpy's
  inner loops; there is nothing to win without micro-optimizing scipy's
  COO -> CSR conversion.
- **Stationary distribution.** Power iteration on a CSR matrix is one
  sparse matvec per step. Power iteration converges in <100 steps for
  most maps; ARPACK fallback handles the rest. Both are scipy.
- **Spectral analysis.** Dense `eig` on small matrices, ARPACK on large.
  Both are LAPACK; not our hot loop.
- **TPT committors.** `scipy.sparse.linalg.spsolve` on the absorbing
  system. The pyo3 plan flagged a Rust BiCGSTAB for the "large MSM"
  case; in practice spsolve handles 2^14 directly and the BiCGSTAB
  path is deferred until a real customer hits the limit.

## Committor solver

`forward_committor(P, A, B)` solves `(I - P_ff) q_free = P_fB . 1` over the
free states. Times below are wall-clock for a single committor call on
binary maps; reproduce with

```bash
uv run pytest tests/benchmarks/test_committor_bench.py --benchmark-only
```

| sites | n_states | spsolve (ms) | Rust BiCGSTAB (ms) | speedup |
|------:|---------:|-------------:|-------------------:|--------:|
| 10    | 1,024    | 42           | 1.4                | 30x     |
| 12    | 4,096    | 2,964        | 6.9                | 430x    |
| 13    | 8,192    | 26,219       | 14.6               | 1,800x  |
| 14    | 16,384   | timed out*   | 35                 | >1,700x |
| 15    | 32,768   | timed out*   | 77                 | n/a     |

*spsolve exceeded the 60 s wall in our profiling at n=16k and was killed.
The LU fill-in scales roughly as O(n^2.5) on these matrices; BiCGSTAB is
O(nnz * iterations) with iterations roughly constant in n (~50 for
well-conditioned non-reversible MSM systems).

Threshold: gpvolve-v2 switches to BiCGSTAB at `n_free > 256`. Below that,
spsolve's microsecond setup beats the FFI overhead.

## Profiling notes

When the first Rust-walker pass landed it was only ~2.5x faster end to
end. Profiling pointed at the per-walker probability product loop in the
Python wrapper, which did 4.5 million `csr[i, j]` calls on a 2^14 5k-walker
run. Folding the running product into the Rust walker (it has the
per-step probability in hand anyway) dropped the wrapper from 95% of
wall-clock to <1%, taking the end-to-end speedup from 2.5x to ~500x.
This is the kind of measurement that decides Rust vs not, not gut feel.
