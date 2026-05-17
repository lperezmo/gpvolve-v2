//! gpvolve-core: Rust hot-path primitives for gpvolve-v2.
//!
//! Exposed as `gpvolve._rust` inside the Python package.
//!
//! Scope (locked 2026-05-16, revisited 2026-05-17):
//!   - Stochastic path sampling (`sample_paths_csr`): rayon-parallel walkers
//!     over a CSR-encoded row-stochastic matrix, with per-walker rand_pcg
//!     streams.
//!
//! Transition-matrix assembly and the dense committor solve stay in numpy /
//! scipy. Profiling on 2^14 binary maps shows the Python paths at <2 s end
//! to end; only the stochastic walker loop (24 s for 5k walkers on a 2^6
//! map, scaling linearly in n_walkers and steps) earns Rust acceleration.

use pyo3::prelude::*;

mod sample;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample::sample_paths_csr, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
