"""Vectorized fixation kernels with overflow-safe branches.

Ported from gpgraph-v2's `fixation.py` so numeric agreement holds at the boundaries.
The public model entry points in sibling modules wrap these kernels with the registry
metadata required by the FixationModel Protocol.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_MAXEXP = float(np.finfo(float).maxexp)


def _as_float_array(x: object) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def sswm_kernel(
    fitness_i: NDArray[np.float64], fitness_j: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gillespie (1984) strong-selection weak-mutation fixation probability.

    ``pi_{i -> j} = 1 - exp(-s_ij)`` where ``s_ij = (f_j - f_i) / f_i``. Returns 0 when
    f_j <= f_i. Overflow protection uses a log2 decomposition.
    """
    fi = _as_float_array(fitness_i)
    fj = _as_float_array(fitness_j)
    if np.any(fi <= 0) or np.any(fj <= 0):
        raise ValueError("fitness values must be > 0")

    a = fj - fi
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_log = np.log2(np.where(a > 0, a, 1.0)) - np.log2(fi)
    sij = np.where(
        a <= 0,
        0.0,
        np.where(ratio_log > _MAXEXP, np.inf, np.power(2.0, ratio_log)),
    )
    return np.where(a <= 0, 0.0, 1.0 - np.exp(-sij))


def weak_mutation_kernel(
    fitness_i: NDArray[np.float64], fitness_j: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Weak-mutation (Lynch-Conery) limit: pi = max(0, (f_j - f_i) / f_i).

    This is not a true fixation probability when f_j is much larger than f_i; users
    should treat it as a relative weighting in that regime.
    """
    fi = _as_float_array(fitness_i)
    fj = _as_float_array(fitness_j)
    if np.any(fi <= 0):
        raise ValueError("fitness_i must be > 0")
    delta = (fj - fi) / fi
    return np.maximum(delta, 0.0)


def _moran_safe(
    fi: NDArray[np.float64], fj: NDArray[np.float64], n: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Moran fixation probability with overflow guards (per gpgraph-v2 v1 branches)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a = np.log2(fi) - np.log2(fj)
        log2_abs_a = np.log2(np.abs(a))
        huge = log2_abs_a + np.log2(n) > _MAXEXP
        asymptotic_pos = np.where(n > _MAXEXP, 0.0, 1.0 / np.power(2.0, np.minimum(n, _MAXEXP)))
        asymptotic_neg = np.where(
            -a > _MAXEXP, 1.0, 1.0 - np.power(2.0, np.clip(a, -_MAXEXP, _MAXEXP))
        )
        huge_out = np.where(a > 0, asymptotic_pos, asymptotic_neg)

        b = a * n
        b_overflow = b > _MAXEXP
        power_a = np.power(2.0, np.clip(a, -_MAXEXP, _MAXEXP))
        power_b = np.power(2.0, np.clip(b, -_MAXEXP, _MAXEXP))
        safe_denom = np.where(power_b == 1.0, 1.0, 1.0 - power_b)
        regular = np.where(b_overflow, np.power(2.0, a - b), (1.0 - power_a) / safe_denom)

    return np.where(huge, huge_out, regular)


def moran_kernel(
    fitness_i: NDArray[np.float64],
    fitness_j: NDArray[np.float64],
    population_size: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sella and Hirsch (2005) Moran-process fixation probability.

    For population_size == 1 returns 1.0 by convention. For f_i == f_j the removable
    0/0 singularity is resolved by averaging evaluations at two slightly perturbed
    operating points.
    """
    fi = _as_float_array(fitness_i)
    fj = _as_float_array(fitness_j)
    n = _as_float_array(population_size)
    if np.any(fi <= 0) or np.any(fj <= 0):
        raise ValueError("fitness values must be > 0")
    if np.any(n < 1):
        raise ValueError("population_size must be >= 1")

    shape = np.broadcast_shapes(fi.shape, fj.shape, n.shape)
    fi_b = np.broadcast_to(fi, shape).astype(np.float64, copy=True)
    fj_b = np.broadcast_to(fj, shape).astype(np.float64, copy=True)
    n_b = np.broadcast_to(n, shape).astype(np.float64, copy=True)

    eq_mask = fi_b == fj_b
    if np.any(eq_mask):
        eval_a = _moran_safe(fi_b * 0.99999, fj_b, n_b)
        eval_b = _moran_safe(fi_b, fj_b * 0.99999, n_b)
        averaged = 0.5 * (eval_a + eval_b)
    else:
        averaged = np.zeros_like(fi_b)

    direct = _moran_safe(fi_b, fj_b, n_b)
    out = np.where(eq_mask, averaged, direct)
    return np.where(n_b == 1, 1.0, out)


def _mccandlish_safe(
    fi: NDArray[np.float64], fj: NDArray[np.float64], n: NDArray[np.float64]
) -> NDArray[np.float64]:
    """McCandlish (2011) fixation probability with overflow guards."""
    a = fj - fi
    power_coeff = -2.0 * np.log2(np.e)
    l2_power_coeff = np.log2(2.0 * np.log2(np.e))

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log2_abs_a = np.log2(np.abs(a))

        can_do_2a = log2_abs_a + l2_power_coeff <= _MAXEXP
        can_do_exp_2a = can_do_2a & ((a * power_coeff) <= _MAXEXP)
        can_do_2aN = log2_abs_a + np.log2(n) + l2_power_coeff <= _MAXEXP
        can_do_exp_2aN = can_do_2aN & ((a * n * power_coeff) <= _MAXEXP)

        regular_mask = can_do_exp_2a & can_do_exp_2aN

        neg2a = np.where(can_do_2a, a * power_coeff, 0.0)
        exp_neg2a = np.where(can_do_exp_2a, np.power(2.0, np.clip(neg2a, -_MAXEXP, _MAXEXP)), 0.0)
        neg2aN = np.where(can_do_2aN, a * n * power_coeff, 0.0)
        exp_neg2aN = np.where(
            can_do_exp_2aN, np.power(2.0, np.clip(neg2aN, -_MAXEXP, _MAXEXP)), 0.0
        )

        regular = np.where(
            regular_mask,
            (1.0 - exp_neg2a) / np.where(exp_neg2aN == 1.0, 1.0, 1.0 - exp_neg2aN),
            0.0,
        )

        pos_branch = np.where(can_do_exp_2a, 1.0 - exp_neg2a, 1.0)
        asymptotic = np.where(a > 0, pos_branch, 0.0)

    return np.where(regular_mask, regular, asymptotic)


def mccandlish_kernel(
    fitness_i: NDArray[np.float64],
    fitness_j: NDArray[np.float64],
    population_size: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """McCandlish (2011) fixation probability.

    ``pi = (1 - exp(-2*(f_j - f_i))) / (1 - exp(-2*N*(f_j - f_i)))`` with overflow-
    protected branches. For population_size == 1 returns 1.0 by convention.
    """
    fi = _as_float_array(fitness_i)
    fj = _as_float_array(fitness_j)
    n = _as_float_array(population_size)
    if np.any(fi <= 0) or np.any(fj <= 0):
        raise ValueError("fitness values must be > 0")
    if np.any(n < 1):
        raise ValueError("population_size must be >= 1")

    shape = np.broadcast_shapes(fi.shape, fj.shape, n.shape)
    fi_b = np.broadcast_to(fi, shape).astype(np.float64, copy=True)
    fj_b = np.broadcast_to(fj, shape).astype(np.float64, copy=True)
    n_b = np.broadcast_to(n, shape).astype(np.float64, copy=True)

    eq_mask = fi_b == fj_b
    if np.any(eq_mask):
        eval_a = _mccandlish_safe(fi_b * 0.99999, fj_b, n_b)
        eval_b = _mccandlish_safe(fi_b, fj_b * 0.99999, n_b)
        averaged = 0.5 * (eval_a + eval_b)
    else:
        averaged = np.zeros_like(fi_b)

    direct = _mccandlish_safe(fi_b, fj_b, n_b)
    out = np.where(eq_mask, averaged, direct)
    return np.where(n_b == 1, 1.0, out)


def bloom_kernel(
    fitness_i: NDArray[np.float64],
    fitness_j: NDArray[np.float64],
    pi_table: NDArray[np.float64],
    indices: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None,
) -> NDArray[np.float64]:
    """Bloom (2017)-style empirical DMS fixation probability.

    ``pi_table`` is a precomputed (N, N) matrix of fixation probabilities estimated from
    deep mutational scanning preference data. ``indices`` provides the (i_idx, j_idx)
    index pairs into the table; if None, the kernel assumes the caller has already
    aligned ``fitness_i`` and ``fitness_j`` with the table's row/column ordering and
    falls back to the SSWM kernel as a sanity layer.
    """
    fi = _as_float_array(fitness_i)
    fj = _as_float_array(fitness_j)
    if np.any(fi <= 0) or np.any(fj <= 0):
        raise ValueError("fitness values must be > 0")

    if indices is None:
        return sswm_kernel(fi, fj)

    i_idx, j_idx = indices
    return _as_float_array(pi_table[i_idx, j_idx])
