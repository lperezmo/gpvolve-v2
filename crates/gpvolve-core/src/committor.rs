//! Sparse BiCGSTAB for the TPT committor linear system.
//!
//! The forward-committor equation reduces to a sparse linear system over the
//! "free" states (those not in A or B):
//!
//!     (I - P_ff) q_free = P_fB . 1
//!
//! For maps below ~4k states scipy.sparse.linalg.spsolve handles this directly
//! via supernodal LU; we just call into scipy from the Python layer there.
//! Above ~8k states the LU fill-in blows up superlinearly (3 s at 8k, >>1 min
//! at 16k on our binary-map benchmarks). BiCGSTAB iterations are O(nnz) and
//! converge in a few hundred steps for well-conditioned MSM systems.
//!
//! This kernel is generic over the CSR matrix; the Python caller pre-builds
//! `A = I - P_ff` and passes its CSR representation. No assumption that `A`
//! is reversible or symmetric.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[inline]
fn csr_matvec(
    indptr: &[i64],
    indices: &[i64],
    data: &[f64],
    x: &[f64],
    out: &mut [f64],
) {
    let n = out.len();
    for i in 0..n {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;
        let mut s = 0.0f64;
        for k in start..end {
            s += data[k] * x[indices[k] as usize];
        }
        out[i] = s;
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Solve `A x = b` with stabilized biconjugate gradient (van der Vorst 1992).
///
/// Returns `(x, iterations, converged)`. Convergence is declared when
/// `||r|| <= tol * ||b||`. Stops at `max_iter` either way; the caller decides
/// what to do if `converged` is false.
fn bicgstab(
    indptr: &[i64],
    indices: &[i64],
    data: &[f64],
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, usize, bool) {
    let n = b.len();
    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec();
    let r_hat = r.clone();
    let bnorm = norm2(b).max(1e-300);

    let mut rho: f64 = 1.0;
    let mut alpha: f64 = 1.0;
    let mut omega: f64 = 1.0;
    let mut v = vec![0.0f64; n];
    let mut p = vec![0.0f64; n];
    let mut s = vec![0.0f64; n];
    let mut t = vec![0.0f64; n];

    for it in 0..max_iter {
        let rho_new = dot(&r_hat, &r);
        if rho_new.abs() < 1e-300 {
            // Breakdown: restart by resetting r_hat would help; we accept the
            // current x and report non-convergence.
            return (x, it, false);
        }
        if it == 0 {
            p.copy_from_slice(&r);
        } else {
            let beta = (rho_new / rho) * (alpha / omega);
            for k in 0..n {
                p[k] = r[k] + beta * (p[k] - omega * v[k]);
            }
        }
        csr_matvec(indptr, indices, data, &p, &mut v);
        let denom = dot(&r_hat, &v);
        if denom.abs() < 1e-300 {
            return (x, it, false);
        }
        alpha = rho_new / denom;
        for k in 0..n {
            s[k] = r[k] - alpha * v[k];
        }
        if norm2(&s) <= tol * bnorm {
            for k in 0..n {
                x[k] += alpha * p[k];
            }
            return (x, it + 1, true);
        }
        csr_matvec(indptr, indices, data, &s, &mut t);
        let tt = dot(&t, &t);
        if tt.abs() < 1e-300 {
            return (x, it, false);
        }
        omega = dot(&t, &s) / tt;
        for k in 0..n {
            x[k] += alpha * p[k] + omega * s[k];
            r[k] = s[k] - omega * t[k];
        }
        if norm2(&r) <= tol * bnorm {
            return (x, it + 1, true);
        }
        if omega.abs() < 1e-300 {
            return (x, it + 1, false);
        }
        rho = rho_new;
    }
    (x, max_iter, false)
}

/// Solve `A x = b` for a CSR-encoded sparse `A` with BiCGSTAB.
///
/// Returns `(x, iterations, converged)`. `tol` is a relative residual
/// threshold: convergence when `||A x - b|| / ||b|| <= tol`.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, b, max_iter=1000, tol=1e-10))]
pub fn solve_bicgstab_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, bool)> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    let data_slice = data.as_slice()?;
    let b_slice = b.as_slice()?;

    let (x, iters, converged) = py.detach(|| {
        bicgstab(
            indptr_slice,
            indices_slice,
            data_slice,
            b_slice,
            max_iter,
            tol,
        )
    });

    let arr = Array1::from_vec(x).into_pyarray(py);
    Ok((arr, iters, converged))
}
