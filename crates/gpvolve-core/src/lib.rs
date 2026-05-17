//! gpvolve-core: Rust hot-path primitives for gpvolve-v2.
//!
//! Exposed as `gpvolve._rust` inside the Python package.
//!
//! Scope (last revised 2026-05-17):
//!   - `sample_paths_csr` -- rayon-parallel walker sampling with per-walker
//!     PCG64 streams and the per-step probability product folded in.
//!   - `solve_bicgstab_csr` -- sparse BiCGSTAB for the absorbing-boundary
//!     committor system on maps too large for scipy's LU (>~8k states).
//!
//! Transition-matrix assembly stays in numpy (already vectorized);
//! stationary distributions stay in scipy (power iteration + ARPACK).
//! Profiling on 2^14 binary maps places those under a second end to end.

use pyo3::prelude::*;

mod committor;
mod sample;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample::sample_paths_csr, m)?)?;
    m.add_function(wrap_pyfunction!(committor::solve_bicgstab_csr, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
