"""Stochastic walker sampling with statistically grounded convergence checks.

v1's path sampler stopped on a Euclidean distance threshold over the empirical
path distribution, which is statistically meaningless: nearby steps in the
walk are correlated, the distribution is multinomial on a sparse support, and
the cost grows linearly with sample count even after the marginals have
converged.

v2 replaces it with two complementary, well-known stopping rules run in
tandem:

1. **ESS via Sokal-windowed integrated autocorrelation.** For each target ``b``,
   we treat the per-walker indicator ``1[walker reached b]`` as a time series
   across walker IDs and estimate the integrated autocorrelation time
   ``tau_int`` with the standard Sokal windowing. Effective sample size is
   ``ESS = N / (2 * tau_int)``. Require ``ESS >= ess_min`` (default 200) for
   every endpoint whose empirical hit probability is above
   ``relevance_threshold``.
2. **Gelman-Rubin R-hat across walker chains.** Split walkers into
   ``m >= 4`` chains, compute the standard ``R-hat`` statistic on the binary
   hit indicator. Require ``R-hat <= 1.01``.

KL divergence with bootstrap CI is offered as a diagnostic in
``analysis.observables.path_distribution_kl`` but is **not** used as a stopping
rule, since KL is unstable on sparse empirical multinomials with zeros.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gpvolve.exceptions import ConvergenceError, GpvolveError
from gpvolve.markov.msm import GenotypePhenotypeMSM
from gpvolve.types import PathEnsemble


@dataclass(frozen=True)
class ConvergenceCheck:
    """Stopping criteria for :func:`sample_paths`.

    See SCHEMA.md section 6 for the convergence stats payload that lands in
    ``PathEnsemble.metadata["convergence"]``.
    """

    ess_min: float = 200.0
    rhat_max: float = 1.01
    relevance_threshold: float = 1e-3
    chunk_size: int = 10_000
    max_walkers: int = 1_000_000
    n_chains: int = 4
    max_steps_per_walker: int | None = None


def _sokal_tau_int(series: NDArray[np.float64], *, c: float = 5.0) -> float:
    """Sokal-windowed integrated autocorrelation time of a 1-D series.

    Returns 0.5 for series with zero variance (so ``ESS = N / (2*0.5) = N``).
    Window size grows until ``window > c * tau``, the standard Sokal rule.
    """
    n = series.size
    if n < 2:
        return 0.5
    x = series - series.mean()
    var = float((x * x).mean())
    if var <= 0:
        return 0.5
    # FFT-based autocorrelation.
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=size)
    acf = np.fft.irfft(f * np.conj(f), n=size)[:n].real
    acf = acf / acf[0]
    tau = 0.5
    for w in range(1, n):
        tau += acf[w]
        if w >= c * tau:
            break
    return max(tau, 0.5)


def _split_rhat(series: NDArray[np.float64], *, m: int) -> float:
    """Gelman-Rubin R-hat over ``m`` equal-sized chains."""
    n_total = series.size
    if m < 2 or n_total < 2 * m:
        return float("nan")
    n = n_total // m
    chains = series[: m * n].reshape(m, n)
    if not np.isfinite(chains).all():
        return float("nan")
    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean()
    B = n * float(((chain_means - grand_mean) ** 2).sum()) / (m - 1)
    W_per_chain = chains.var(axis=1, ddof=1)
    W = float(W_per_chain.mean())
    if W <= 0:
        # All chains constant. R-hat is 1 if they agree, otherwise inf.
        return 1.0 if np.allclose(chain_means, grand_mean) else float("inf")
    var_hat = ((n - 1) / n) * W + (1.0 / n) * B
    return float(np.sqrt(var_hat / W))


def _simulate_chunk(
    msm: GenotypePhenotypeMSM,
    *,
    source: int,
    target_set: frozenset[int],
    n_walkers: int,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, ...]], dict[int, NDArray[np.uint8]]]:
    """Run ``n_walkers`` independent walkers from ``source``; record paths + hits."""
    csr = msm.transition_matrix.tocsr()
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data

    paths: list[tuple[int, ...]] = []
    hits: dict[int, NDArray[np.uint8]] = {
        t: np.zeros(n_walkers, dtype=np.uint8) for t in target_set
    }

    for w in range(n_walkers):
        current = source
        trail: list[int] = [current]
        seen_target = False
        for _ in range(max_steps):
            start = indptr[current]
            end = indptr[current + 1]
            row_cols = indices[start:end]
            row_probs = data[start:end]
            # Renormalize defensively against tiny FP drift.
            s = row_probs.sum()
            if s <= 0:
                break
            probs = row_probs / s
            nxt = int(rng.choice(row_cols, p=probs))
            if nxt == current:
                # Self-loop step. Record it and stop if it stays. Continue so the
                # walker has a chance to exit via the diagonal absorption later.
                trail.append(nxt)
                continue
            current = nxt
            trail.append(current)
            if current in target_set:
                hits[current][w] = 1
                seen_target = True
                break
        if not seen_target:
            # No hit. Still record the path.
            pass
        paths.append(tuple(trail))

    return paths, hits


def sample_paths(
    msm: GenotypePhenotypeMSM,
    source: int,
    targets: int | Iterable[int],
    *,
    convergence: ConvergenceCheck | None = None,
    seed: int | None = None,
) -> PathEnsemble:
    """Monte Carlo walker ensemble from ``source`` until any of ``targets`` is hit.

    Sampling proceeds in chunks of ``convergence.chunk_size`` walkers. After
    each chunk, ESS and R-hat are evaluated for the binary hit indicators on
    each endpoint. Returns a :class:`PathEnsemble` whose ``metadata["convergence"]``
    matches SCHEMA section 6. Raises :class:`ConvergenceError` if the maximum
    walker budget is reached without convergence.
    """
    if isinstance(targets, int):
        target_list: tuple[int, ...] = (int(targets),)
    else:
        target_list = tuple(sorted({int(t) for t in targets}))
    if not target_list:
        raise GpvolveError("targets must be non-empty")
    target_set = frozenset(target_list)
    if int(source) in target_set:
        raise GpvolveError("source cannot be in targets")

    n = msm.n_states
    if any(t < 0 or t >= n for t in target_set):
        raise GpvolveError("target index out of range")
    if not (0 <= int(source) < n):
        raise GpvolveError("source index out of range")

    cc = convergence or ConvergenceCheck()
    max_steps = cc.max_steps_per_walker or max(50, 10 * n)
    rng = np.random.default_rng(seed)

    all_paths: list[tuple[int, ...]] = []
    hits_per_target: dict[int, list[NDArray[np.uint8]]] = {t: [] for t in target_set}
    n_chunks = 0
    converged = False
    ess: dict[int, float] = {}
    rhat: dict[int, float] = {}

    while len(all_paths) < cc.max_walkers:
        chunk_paths, chunk_hits = _simulate_chunk(
            msm,
            source=int(source),
            target_set=target_set,
            n_walkers=cc.chunk_size,
            max_steps=max_steps,
            rng=rng,
        )
        all_paths.extend(chunk_paths)
        for t in target_set:
            hits_per_target[t].append(chunk_hits[t])
        n_chunks += 1

        # Reevaluate ESS + R-hat on the aggregated indicators.
        ess = {}
        rhat = {}
        ok = True
        for t in target_set:
            series = np.concatenate(hits_per_target[t]).astype(np.float64)
            empirical_hit = float(series.mean())
            tau = _sokal_tau_int(series)
            ess_t = series.size / (2.0 * tau)
            ess[t] = ess_t
            rhat[t] = _split_rhat(series, m=cc.n_chains)
            if empirical_hit < cc.relevance_threshold:
                continue
            if ess_t < cc.ess_min:
                ok = False
            if not np.isnan(rhat[t]) and rhat[t] > cc.rhat_max:
                ok = False
        if ok:
            converged = True
            break

    if not converged:
        raise ConvergenceError(
            f"sampler did not converge within {cc.max_walkers} walkers; ess={ess}, rhat={rhat}"
        )

    n_walkers = len(all_paths)
    # Build path-level probabilities: per-walker product of transition probs.
    from itertools import pairwise

    csr = msm.transition_matrix.tocsr()
    probs = np.empty(n_walkers, dtype=np.float64)
    for k, p in enumerate(all_paths):
        prod = 1.0
        for i, j in pairwise(p):
            prod *= float(csr[i, j])
        probs[k] = prod

    metadata = {
        "convergence": {
            "ess": ess,
            "rhat": rhat,
            "n_walkers": n_walkers,
            "n_chunks": n_chunks,
            "converged": converged,
        }
    }
    return PathEnsemble(
        paths=tuple(all_paths),
        probabilities=probs,
        source=int(source),
        targets=target_list,
        method="stochastic",
        metadata=metadata,
    )
