"""Deterministic max-probability greedy walk on an MSM.

At each step we transition to the neighbor maximizing the outgoing transition
probability (excluding self-loops). Stops when the walker reaches a target, a
fixed-point self-loop, or a cycle.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from gpvolve.exceptions import GpvolveError
from gpvolve.markov.msm import GenotypePhenotypeMSM
from gpvolve.types import PathEnsemble


def greedy_walk(
    msm: GenotypePhenotypeMSM,
    source: int,
    targets: int | Iterable[int],
    *,
    max_steps: int | None = None,
) -> PathEnsemble:
    """Deterministic greedy walk from ``source`` until any target is reached.

    Returns a :class:`PathEnsemble` with ``method="greedy"``. ``paths`` contains
    one path; ``probabilities`` is the cumulative product along that path. If
    the walk ends without hitting a target (cycle or fixed-point), the resulting
    path still ends at the last visited node and the metadata records
    ``hit_target=False``.
    """
    if isinstance(targets, int):
        target_list: tuple[int, ...] = (int(targets),)
    else:
        target_list = tuple(int(t) for t in targets)
    target_set = set(target_list)
    if not target_set:
        raise GpvolveError("targets must be non-empty")

    n = msm.n_states
    csr = msm.transition_matrix.tocsr()
    if max_steps is None:
        max_steps = 10 * n

    visited: list[int] = [int(source)]
    visited_set: set[int] = {int(source)}
    prob = 1.0
    hit_target = int(source) in target_set
    if not hit_target:
        for _ in range(max_steps):
            current = visited[-1]
            row = csr.getrow(current)
            indices = row.indices
            data = row.data
            mask = indices != current
            if not mask.any():
                break
            j_local = int(np.argmax(data[mask]))
            best_idx = int(indices[mask][j_local])
            prob *= float(data[mask][j_local])
            visited.append(best_idx)
            if best_idx in visited_set and best_idx not in target_set:
                # Cycle without hitting target.
                break
            visited_set.add(best_idx)
            if best_idx in target_set:
                hit_target = True
                break

    return PathEnsemble(
        paths=(tuple(visited),),
        probabilities=np.asarray([prob], dtype=np.float64),
        source=int(source),
        targets=target_list,
        method="greedy",
        metadata={"hit_target": hit_target},
    )
