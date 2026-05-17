"""Fitness peak and valley detection on a genotype-phenotype graph.

A *peak* is a node whose fitness exceeds every neighbor's. A *valley* is the
mirror image. Both are computed directly from the graph topology and the
fitness values, no MSM machinery required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from gpvolve.exceptions import GpvolveError

if TYPE_CHECKING:
    from gpgraph import GenotypePhenotypeGraph


def _fitness_array(graph: GenotypePhenotypeGraph, fitness_column: str) -> NDArray[np.float64]:
    try:
        col = graph.gpm.data[fitness_column]
    except KeyError as exc:
        raise GpvolveError(f"fitness_column {fitness_column!r} not in gpm.data") from exc
    return np.asarray(col.to_numpy(), dtype=np.float64)


def find_peaks(graph: GenotypePhenotypeGraph, *, fitness_column: str) -> list[int]:
    """Return the sorted list of indices whose fitness is strictly larger than every neighbor's."""
    fitness = _fitness_array(graph, fitness_column)
    peaks: list[int] = []
    for i in range(graph.number_of_nodes()):
        neighbors = list(graph.successors(i))
        if not neighbors:
            continue
        if all(fitness[i] > fitness[j] for j in neighbors):
            peaks.append(int(i))
    return sorted(peaks)


def find_valleys(graph: GenotypePhenotypeGraph, *, fitness_column: str) -> list[int]:
    """Return the sorted list of indices whose fitness is strictly smaller than every neighbor's."""
    fitness = _fitness_array(graph, fitness_column)
    valleys: list[int] = []
    for i in range(graph.number_of_nodes()):
        neighbors = list(graph.successors(i))
        if not neighbors:
            continue
        if all(fitness[i] < fitness[j] for j in neighbors):
            valleys.append(int(i))
    return sorted(valleys)


def accessible_peaks(
    graph: GenotypePhenotypeGraph,
    source: int,
    *,
    fitness_column: str,
) -> list[int]:
    """Peaks reachable from ``source`` by strictly-increasing-fitness edges."""
    fitness = _fitness_array(graph, fitness_column)
    accessible: set[int] = set()
    peaks_set = set(find_peaks(graph, fitness_column=fitness_column))
    # BFS along fitness-increasing edges.
    frontier: list[int] = [int(source)]
    seen: set[int] = {int(source)}
    while frontier:
        nxt: list[int] = []
        for u in frontier:
            if u in peaks_set:
                accessible.add(u)
            for v in graph.successors(u):
                if fitness[v] > fitness[u] and v not in seen:
                    seen.add(int(v))
                    nxt.append(int(v))
        frontier = nxt
    if int(source) in peaks_set:
        accessible.add(int(source))
    return sorted(accessible)
