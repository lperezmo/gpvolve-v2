//! Rayon-parallel stochastic walker sampling.
//!
//! Each walker owns an independent `Pcg64` stream seeded by SplitMix64(seed,
//! walker_id) so runs with the same seed are bit-identical regardless of the
//! number of rayon threads.
//!
//! The walkers traverse a CSR-encoded row-stochastic transition matrix. We
//! precompute the per-row cumulative distribution once, then each step is
//! one uniform draw plus a binary search into the row slice. This is the
//! inner loop the Python version spends >95% of its time on.

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;

/// SplitMix64: stable per-walker stream seeding (Steele, Lea, Flood 2014).
#[inline]
fn splitmix64(seed: u64, walker_id: u64) -> u64 {
    let mut z = seed.wrapping_add(walker_id.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sample one step from row `current`'s cumulative distribution.
///
/// Returns `(next_state, normalized_prob)`. `next_state == -1` signals a dead
/// row (zero outgoing mass); the caller stops the walker. The probability is
/// the un-cumulated row entry divided by the row sum, so multiplying these
/// across a walk gives the path's product of transition probabilities.
#[inline]
fn sample_step(
    cumdist: &[f64],
    data: &[f64],
    indices: &[i64],
    indptr_current: usize,
    indptr_next: usize,
    rng: &mut Pcg64,
) -> (i64, f64) {
    let start = indptr_current;
    let end = indptr_next;
    if start == end {
        return (-1, 0.0);
    }
    let row = &cumdist[start..end];
    let last = row[row.len() - 1];
    if last <= 0.0 {
        return (-1, 0.0);
    }
    let u: f64 = rng.random::<f64>() * last;
    let mut lo = 0usize;
    let mut hi = row.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if row[mid] < u {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let chosen = lo.min(row.len() - 1);
    let prob = data[start + chosen] / last;
    (indices[start + chosen], prob)
}

/// Walk a single trajectory until it hits any target, exhausts `max_steps`,
/// or reaches a dead row. Records every visited state, per-target hit
/// indicator, and the product of transition probabilities along the walk.
fn walk_one(
    cumdist: &[f64],
    data: &[f64],
    indptr: &[i64],
    indices: &[i64],
    source: i64,
    targets: &[i64],
    max_steps: usize,
    seed: u64,
    walker_id: u64,
) -> (Vec<i64>, Vec<u8>, f64) {
    let mut rng = Pcg64::seed_from_u64(splitmix64(seed, walker_id));
    let mut path: Vec<i64> = Vec::with_capacity(max_steps + 1);
    let mut hits = vec![0u8; targets.len()];
    let mut prob: f64 = 1.0;
    path.push(source);
    let mut current = source as usize;
    for _ in 0..max_steps {
        let (next, p) = sample_step(
            cumdist,
            data,
            indices,
            indptr[current] as usize,
            indptr[current + 1] as usize,
            &mut rng,
        );
        if next < 0 {
            break;
        }
        prob *= p;
        if next as usize == current {
            path.push(next);
            continue;
        }
        current = next as usize;
        path.push(next);
        if let Some(target_idx) = targets.iter().position(|&t| t == next) {
            hits[target_idx] = 1;
            break;
        }
    }
    (path, hits, prob)
}

/// Sample `n_walkers` independent trajectories in parallel.
///
/// Returns
/// -------
/// flat_paths : 1-D int64 array
///     Concatenated state sequences. Walker `w`'s path is
///     `flat_paths[offsets[w]..offsets[w] + lengths[w]]`.
/// lengths : 1-D int64 array of length `n_walkers`
///     Per-walker path lengths.
/// hits : 2-D uint8 array of shape `(n_walkers, n_targets)`
///     `hits[w, t] = 1` iff walker `w` ended at target index `t`.
/// probabilities : 1-D float64 array of length `n_walkers`
///     Per-walker product of transition probabilities along the walk.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, source, targets, n_walkers, max_steps, seed))]
#[allow(clippy::too_many_arguments)]
pub fn sample_paths_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    source: i64,
    targets: PyReadonlyArray1<'py, i64>,
    n_walkers: usize,
    max_steps: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    let data_slice = data.as_slice()?;
    let targets_slice = targets.as_slice()?;

    let mut cumdist = vec![0.0f64; data_slice.len()];
    let n_states = indptr_slice.len() - 1;
    for i in 0..n_states {
        let start = indptr_slice[i] as usize;
        let end = indptr_slice[i + 1] as usize;
        let mut acc = 0.0f64;
        for k in start..end {
            acc += data_slice[k];
            cumdist[k] = acc;
        }
    }

    let results: Vec<(Vec<i64>, Vec<u8>, f64)> = py.detach(|| {
        (0..n_walkers)
            .into_par_iter()
            .map(|w| {
                walk_one(
                    &cumdist,
                    data_slice,
                    indptr_slice,
                    indices_slice,
                    source,
                    targets_slice,
                    max_steps,
                    seed,
                    w as u64,
                )
            })
            .collect()
    });

    let lengths_vec: Vec<i64> = results.iter().map(|(p, _, _)| p.len() as i64).collect();
    let probs_vec: Vec<f64> = results.iter().map(|(_, _, p)| *p).collect();
    let total_len: usize = lengths_vec.iter().map(|&l| l as usize).sum();
    let mut flat_paths = Vec::with_capacity(total_len);
    let mut hits_flat = Vec::with_capacity(n_walkers * targets_slice.len());
    for (path, hits, _) in &results {
        flat_paths.extend_from_slice(path);
        hits_flat.extend_from_slice(hits);
    }

    let lengths_arr = Array1::from_vec(lengths_vec);
    let probs_arr = Array1::from_vec(probs_vec);
    let flat_arr = Array1::from_vec(flat_paths);
    let hits_arr = Array2::from_shape_vec((n_walkers, targets_slice.len()), hits_flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("hits reshape: {e}")))?;

    Ok((
        flat_arr.into_pyarray(py),
        lengths_arr.into_pyarray(py),
        hits_arr.into_pyarray(py),
        probs_arr.into_pyarray(py),
    ))
}

