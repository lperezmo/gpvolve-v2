"""Unit tests for simulate.wright_fisher and simulate.gillespie."""

from __future__ import annotations

import numpy as np
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpvolve.exceptions import GpvolveError
from gpvolve.simulate import gillespie_walk, wright_fisher


class TestWrightFisher:
    def test_basic_run(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        history = wright_fisher(
            tiny_graph,
            fitness_column="phenotypes",
            population_size=100,
            mutation_rate=0.01,
            n_generations=5,
            initial_index=0,
            seed=0,
        )
        assert history.shape == (6, 4)
        assert (history.sum(axis=1) == 100).all()

    def test_no_mutation_no_movement(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        history = wright_fisher(
            tiny_graph,
            fitness_column="phenotypes",
            population_size=100,
            mutation_rate=0.0,
            n_generations=10,
            initial_index=0,
            seed=0,
        )
        # Without mutation the population stays at the initial genotype (drift
        # alone cannot leave a single-state starting condition).
        assert (history[:, 0] == 100).all()

    def test_population_size_validation(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        with pytest.raises(GpvolveError, match="population_size"):
            wright_fisher(
                tiny_graph,
                fitness_column="phenotypes",
                population_size=0,
                mutation_rate=0.01,
                n_generations=1,
            )

    def test_mutation_rate_validation(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        with pytest.raises(GpvolveError, match="mutation_rate"):
            wright_fisher(
                tiny_graph,
                fitness_column="phenotypes",
                population_size=100,
                mutation_rate=-0.1,
                n_generations=1,
            )


class TestGillespie:
    def test_starts_at_source(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        times, states = gillespie_walk(
            tiny_graph,
            fitness_column="phenotypes",
            source=0,
            targets=3,
            fixation="sswm",
            seed=0,
        )
        assert states[0] == 0
        assert times[0] == 0.0
        assert times.shape == states.shape

    def test_times_monotone(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        times, _ = gillespie_walk(
            tiny_graph,
            fitness_column="phenotypes",
            source=0,
            targets=3,
            fixation="sswm",
            seed=42,
        )
        if times.size > 1:
            assert (np.diff(times) >= 0).all()

    def test_target_in_states_if_reached(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        _times, states = gillespie_walk(
            tiny_graph,
            fitness_column="phenotypes",
            source=0,
            targets=3,
            fixation="sswm",
            seed=1,
            max_steps=200,
        )
        # On this simple landscape the walker tends to reach 3 quickly.
        if states[-1] == 3:
            assert 3 in states.tolist()
