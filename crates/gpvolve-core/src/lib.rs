//! gpvolve-core: Rust hot-path primitives for gpvolve-v2.
//!
//! Exposed as `gpvolve._rust` inside the Python package.
//!
//! Scope (locked with Luis 2026-05-16):
//!   - Transition matrix COO assembly with vectorized fixation kernels.
//!   - Stochastic path sampling with per-walker rng streams.
//!   - Sparse iterative committor solver for large MSMs.
//!
//! Pure-Python paths remain authoritative; Rust kernels accelerate the hot
//! loops once the Python implementation is feature-complete and tested.

use pyo3::prelude::*;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
