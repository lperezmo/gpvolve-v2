"""Unit tests for paths (shortest, greedy) and TPT (committors, flux, rate)."""

from __future__ import annotations

import numpy as np
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpvolve import (
    GenotypePhenotypeMSM,
    backward_committor,
    build_transition_matrix,
    dominant_pathways,
    forward_committor,
    greedy_walk,
    net_flux,
    rate,
    reactive_flux,
    shortest_paths,
)


class TestForwardCommittor:
    def test_boundary_conditions(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        q = forward_committor(T, A=0, B=3)
        assert q[0] == 0.0
        assert q[3] == 1.0

    def test_in_unit_interval(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=1000,
        )
        q = forward_committor(T, A=0, B=63)
        assert q.shape == (64,)
        assert (q >= 0).all() and (q <= 1).all()
        assert q[0] == 0.0
        assert q[63] == 1.0

    def test_set_a_set_b(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=1000,
        )
        q = forward_committor(T, A=[0, 1], B=[62, 63])
        for a in (0, 1):
            assert q[a] == 0.0
        for b in (62, 63):
            assert q[b] == 1.0

    def test_disjoint_required(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        with pytest.raises(Exception, match="disjoint"):
            forward_committor(T, A=[0, 1], B=[1, 2])


class TestBackwardCommittor:
    def test_boundary_conditions(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        q_minus = backward_committor(T, A=0, B=3)
        assert q_minus.shape == (4,)
        # q-_A = 1 (came from A by definition), q-_B = 0.
        assert q_minus[0] == 1.0
        assert q_minus[3] == 0.0


class TestReactiveFlux:
    def test_nonneg(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        F = reactive_flux(T, A=0, B=3)
        assert F.shape == (4, 4)
        assert F.tocsr().data.min() >= -1e-12

    def test_diagonal_zero(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        F = reactive_flux(T, A=0, B=3)
        diag = np.asarray(F.diagonal())
        np.testing.assert_array_equal(diag, np.zeros(4))


class TestNetFlux:
    def test_nonneg_no_two_way(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        F = reactive_flux(T, A=0, B=3)
        N = net_flux(F)
        assert N.data.min() > 0 if N.nnz > 0 else True
        # Net flux is anti-symmetric in its support: if (i, j) is in N then (j, i) is not.
        rows, cols = N.nonzero()
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            assert N[c, r] == 0.0


class TestRate:
    def test_positive(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        k = rate(T, A=0, B=3)
        assert k > 0
        assert k < 1


class TestShortestPaths:
    def test_returns_path_ensemble(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = GenotypePhenotypeMSM.from_graph(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        ens = shortest_paths(msm, source=0, targets=3)
        assert ens.method == "shortest"
        assert ens.source == 0
        assert ens.targets == (3,)
        assert len(ens.paths) == 1
        assert ens.paths[0][0] == 0
        assert ens.paths[0][-1] == 3
        assert 0.0 < ens.probabilities[0] <= 1.0

    def test_multiple_targets(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        msm = GenotypePhenotypeMSM.from_graph(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=1000,
        )
        ens = shortest_paths(msm, source=0, targets=[31, 63])
        assert ens.method == "shortest"
        assert len(ens.paths) == 2
        for p, t in zip(ens.paths, [31, 63], strict=True):
            assert p[0] == 0
            assert p[-1] == t


class TestGreedyWalk:
    def test_starts_at_source(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = GenotypePhenotypeMSM.from_graph(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        ens = greedy_walk(msm, source=0, targets=3)
        assert ens.paths[0][0] == 0
        assert ens.method == "greedy"

    def test_hit_metadata(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        msm = GenotypePhenotypeMSM.from_graph(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=1000,
        )
        ens = greedy_walk(msm, source=0, targets=63)
        assert "hit_target" in ens.metadata


class TestDominantPathways:
    def test_top_k_sorted_by_bottleneck(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=50,
        )
        # Pick the highest-phenotype state as B so reactive flux is non-trivial.
        phenos = binary_graph_64.gpm.data["phenotypes"].to_numpy()
        target = int(phenos.argmax())
        F = reactive_flux(T, A=0, B=target)
        pathways = dominant_pathways(F, A=0, B=target, top_k=5)
        assert 0 < len(pathways) <= 5
        bottlenecks = [float(p.probabilities[0]) for p in pathways]
        assert bottlenecks == sorted(bottlenecks, reverse=True)
        for p in pathways:
            assert p.paths[0][0] == 0
            assert p.paths[0][-1] == target
            assert p.method == "tpt"
