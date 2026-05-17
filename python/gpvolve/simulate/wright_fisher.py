"""Discrete-generation Wright-Fisher sampling on a genotype-phenotype graph.

Each generation, ``N`` haploid genomes are sampled with replacement from the
current population. Selection: each parent's relative reproductive weight is
its fitness. Mutation: each offspring's genotype mutates to a graph neighbor
with probability ``mu`` per individual per generation; if no mutation occurs
the genotype is inherited.

The returned trajectory is the per-generation count of each genotype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from gpvolve.exceptions import GpvolveError

if TYPE_CHECKING:
    from gpgraph import GenotypePhenotypeGraph


def wright_fisher(
    graph: GenotypePhenotypeGraph,
    *,
    fitness_column: str,
    population_size: int,
    mutation_rate: float,
    n_generations: int,
    initial_index: int = 0,
    seed: int | None = None,
) -> NDArray[np.int64]:
    """Run a discrete-generation Wright-Fisher chain on the graph.

    Returns an ``(n_generations + 1, n_states)`` ``int64`` array of per-generation
    counts. Row ``0`` is the initial population (``population_size`` individuals
    at ``initial_index``).
    """
    if population_size < 1:
        raise GpvolveError("population_size must be >= 1")
    if not 0.0 <= mutation_rate <= 1.0:
        raise GpvolveError("mutation_rate must lie in [0, 1]")
    if n_generations < 0:
        raise GpvolveError("n_generations must be >= 0")

    n = graph.number_of_nodes()
    if not (0 <= initial_index < n):
        raise GpvolveError("initial_index out of range")

    fitness = np.asarray(graph.gpm.data[fitness_column].to_numpy(), dtype=np.float64)
    if (fitness < 0).any():
        raise GpvolveError("fitness values must be non-negative")

    # Precompute neighbor lists.
    neighbor_lists: list[NDArray[np.int64]] = []
    for i in range(n):
        nbrs = sorted(int(v) for v in graph.successors(i) if int(v) != i)
        neighbor_lists.append(np.asarray(nbrs, dtype=np.int64))

    rng = np.random.default_rng(seed)
    history = np.zeros((n_generations + 1, n), dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    counts[initial_index] = population_size
    history[0] = counts

    for gen in range(1, n_generations + 1):
        # Reproductive weights.
        w = fitness * counts
        total = w.sum()
        if total <= 0:
            history[gen:] = counts
            break
        probs = w / total
        sampled = rng.multinomial(population_size, probs)

        # Mutation step.
        if mutation_rate > 0:
            new_counts = sampled.copy()
            for i in np.flatnonzero(sampled).tolist():
                if neighbor_lists[i].size == 0:
                    continue
                k = int(rng.binomial(int(sampled[i]), mutation_rate))
                if k == 0:
                    continue
                new_counts[i] -= k
                # Distribute the k mutants uniformly across neighbors.
                draws = rng.choice(neighbor_lists[i], size=k, replace=True)
                idx, freqs = np.unique(draws, return_counts=True)
                new_counts[idx] += freqs
            counts = new_counts
        else:
            counts = sampled
        history[gen] = counts

    return history
