"""Exact SSWM simulation via Gillespie waiting times.

Strong-selection weak-mutation regime: each fixation event is instantaneous on
the timescale of population dynamics. The waiting time between fixation events
is exponential with rate equal to the sum of outgoing transition propensities.

Propensities are the per-edge mutation-and-fixation product:

    a_ij = mu_per_step * fix(f_i, f_j)

where ``mu_per_step`` is a constant per-attempt mutation rate (default
``1.0`` so the returned times are in units of attempts). The next event is
chosen with probability ``a_ij / sum_l a_il``.

Returns the trajectory ``(times, states)`` from ``source`` until any of
``targets`` is hit (or ``max_steps`` is exceeded).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from gpvolve.exceptions import GpvolveError
from gpvolve.fixation.protocol import get_fixation_model

if TYPE_CHECKING:
    from gpgraph import GenotypePhenotypeGraph


def gillespie_walk(
    graph: GenotypePhenotypeGraph,
    *,
    fitness_column: str,
    source: int,
    targets: int | Iterable[int],
    fixation: str = "sswm",
    mu_per_step: float = 1.0,
    max_steps: int = 10_000,
    seed: int | None = None,
    **fixation_params: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Run a single Gillespie trajectory.

    Returns ``(times, states)`` arrays of length ``L+1`` (initial state plus
    ``L`` events). ``times[0] = 0``; ``times[k]`` is the cumulative time at the
    ``k``-th event. ``states`` lists the visited state indices.
    """
    if mu_per_step <= 0:
        raise GpvolveError("mu_per_step must be > 0")
    if isinstance(targets, int):
        target_set: set[int] = {int(targets)}
    else:
        target_set = {int(t) for t in targets}
    if int(source) in target_set:
        raise GpvolveError("source cannot be in targets")

    model = get_fixation_model(fixation)
    fitness = np.asarray(graph.gpm.data[fitness_column].to_numpy(), dtype=np.float64)
    rng = np.random.default_rng(seed)

    times: list[float] = [0.0]
    states: list[int] = [int(source)]
    current = int(source)

    for _ in range(max_steps):
        neighbors = np.asarray(
            sorted(int(v) for v in graph.successors(current) if int(v) != current),
            dtype=np.int64,
        )
        if neighbors.size == 0:
            break
        fi = np.full(neighbors.size, fitness[current], dtype=np.float64)
        fj = fitness[neighbors]
        probs = np.asarray(model(fi, fj, **fixation_params), dtype=np.float64)
        propensities = mu_per_step * probs
        total = propensities.sum()
        if total <= 0:
            break
        dt = float(rng.exponential(1.0 / total))
        next_local = int(rng.choice(neighbors.size, p=propensities / total))
        current = int(neighbors[next_local])
        times.append(times[-1] + dt)
        states.append(current)
        if current in target_set:
            break

    return np.asarray(times, dtype=np.float64), np.asarray(states, dtype=np.int64)
