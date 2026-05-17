"""Type stubs for the Rust extension module `gpvolve._rust`.

Functions exported here are the rayon-parallel hot paths called from the
Python wrappers in `gpvolve.paths.stochastic`. Pure-Python fallbacks live
alongside in each module; the wrapper picks the Rust path when the
extension is built.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__version__: str

def sample_paths_csr(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    source: int,
    targets: NDArray[np.int64],
    n_walkers: int,
    max_steps: int,
    seed: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.uint8],
    NDArray[np.float64],
]:
    """Sample `n_walkers` independent walkers on a CSR-encoded row-stochastic matrix.

    Returns `(flat_paths, lengths, hits, probabilities)`:

    - `flat_paths[offsets[w]:offsets[w] + lengths[w]]` is walker `w`'s state sequence.
    - `lengths[w]` is the per-walker trajectory length.
    - `hits[w, t] = 1` iff walker `w` ended at target index `t`.
    - `probabilities[w]` is the product of transition probabilities along the walk.

    Each walker owns an independent PCG64 stream seeded from
    `splitmix64(seed, walker_id)`, so the result is bitwise reproducible
    regardless of the rayon thread count.
    """
    ...

def solve_bicgstab_csr(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    b: NDArray[np.float64],
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> tuple[NDArray[np.float64], int, bool]:
    """Solve `A x = b` for a CSR-encoded sparse `A` with stabilized BiCG.

    Returns `(x, iterations, converged)`. Convergence is declared when the
    relative residual `||A x - b|| / ||b||` is at or below `tol`. The system
    is solved with zero initial guess; the caller may right-precondition if
    convergence stalls on stiff systems.
    """
    ...
