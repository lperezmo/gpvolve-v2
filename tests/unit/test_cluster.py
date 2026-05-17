"""Unit tests for PCCA+ clustering and analysis.peaks."""

from __future__ import annotations

import numpy as np
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap
from gpvolve import (
    accessible_peaks,
    build_transition_matrix,
    coarse_grain,
    find_peaks,
    find_valleys,
    metastable_sets,
    pcca_plus,
)
from gpvolve.cluster.metastable import crisp_assignments as _crisp


class TestPCCAPlus:
    def test_row_stochastic_membership(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=50,
        )
        chi = pcca_plus(T, n_clusters=3)
        assert chi.shape == (64, 3)
        np.testing.assert_allclose(chi.sum(axis=1), 1.0, atol=1e-8)
        assert (chi >= -1e-10).all()

    def test_single_cluster_trivial(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        chi = pcca_plus(T, n_clusters=1)
        np.testing.assert_array_equal(chi, np.ones((4, 1)))

    def test_more_clusters_than_states_raises(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        with pytest.raises(Exception, match="cannot exceed"):
            pcca_plus(T, n_clusters=100)


class TestMetastableHelpers:
    def test_metastable_sets_partition(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=50,
        )
        chi = pcca_plus(T, n_clusters=3)
        sets = metastable_sets(chi)
        assert len(sets) == 3
        # Union covers all states.
        all_states = np.sort(np.concatenate(sets))
        np.testing.assert_array_equal(all_states, np.arange(64))
        # No overlap.
        total = sum(len(s) for s in sets)
        assert total == 64

    def test_crisp_assignments_in_range(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=50,
        )
        chi = pcca_plus(T, n_clusters=3)
        labels = _crisp(chi)
        assert labels.shape == (64,)
        assert labels.min() >= 0 and labels.max() < 3


class TestCoarseGrain:
    def test_row_stochastic_output(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=50,
        )
        chi = pcca_plus(T, n_clusters=3)
        P_coarse = coarse_grain(T, chi)
        assert P_coarse.shape == (3, 3)
        np.testing.assert_allclose(P_coarse.sum(axis=1), 1.0, atol=1e-8)
        assert (P_coarse >= -1e-10).all()


class TestPeaks:
    def test_flat_landscape_no_peaks(self) -> None:
        gpm = GenotypePhenotypeMap(
            wildtype="00",
            genotypes=["00", "01", "10", "11"],
            phenotypes=[1.0, 1.0, 1.0, 1.0],
        )
        graph = GenotypePhenotypeGraph.from_gpm(gpm)
        assert find_peaks(graph, fitness_column="phenotypes") == []
        assert find_valleys(graph, fitness_column="phenotypes") == []

    def test_single_peak(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        # phenotypes = [1.0, 1.5, 1.2, 2.0]; node 3 ("11") is the unique peak.
        peaks = find_peaks(tiny_graph, fitness_column="phenotypes")
        assert peaks == [3]
        valleys = find_valleys(tiny_graph, fitness_column="phenotypes")
        assert valleys == [0]

    def test_accessible_peaks_from_source(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        # From node 0 ("00"), the only peak is node 3 ("11"). Both via 01 or 10.
        accessible = accessible_peaks(tiny_graph, source=0, fitness_column="phenotypes")
        assert accessible == [3]
