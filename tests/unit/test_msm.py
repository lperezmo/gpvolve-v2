"""Unit tests for the MSM container, stationary, and spectral primitives."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from gpgraph import GenotypePhenotypeGraph
from gpvolve import (
    GenotypePhenotypeMSM,
    build_transition_matrix,
    eigenvalues,
    mfpt,
    mixing_time,
    stationary_distribution,
    timescales,
)


class TestStationaryDistribution:
    def test_sums_to_one(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=1000,
        )
        pi = stationary_distribution(T)
        assert pi.shape == (64,)
        assert abs(pi.sum() - 1.0) < 1e-10
        assert (pi >= 0).all()

    def test_invariance_pi_p_equals_pi(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=1000,
        )
        pi = stationary_distribution(T)
        residual = pi @ T - pi
        assert np.abs(residual).max() < 1e-9

    def test_power_method(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        pi = stationary_distribution(T, method="power")
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_eigs_method(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        pi = stationary_distribution(T, method="eigs")
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_uniform_on_flat_landscape(self) -> None:
        from gpmap import GenotypePhenotypeMap

        gpm = GenotypePhenotypeMap(
            wildtype="000",
            genotypes=["000", "001", "010", "011", "100", "101", "110", "111"],
            phenotypes=[1.0] * 8,
        )
        graph = GenotypePhenotypeGraph.from_gpm(gpm)
        T = build_transition_matrix(
            graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        pi = stationary_distribution(T)
        np.testing.assert_allclose(pi, 1.0 / 8, atol=1e-8)


class TestEigenvalues:
    def test_leading_eigenvalue_is_one(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        vals = eigenvalues(T, k=4)
        assert abs(abs(vals[0]) - 1.0) < 1e-10

    def test_ordered_by_magnitude(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=1000,
        )
        vals = eigenvalues(T, k=10)
        mags = np.abs(vals)
        assert (np.diff(mags) <= 1e-12).all()


class TestTimescales:
    def test_excludes_stationary_mode(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        ts = timescales(T, k=3)
        # All timescales should be positive and finite (no stationary mode bleeding through).
        finite = ts[~np.isnan(ts)]
        assert (finite > 0).all()

    def test_returns_k_entries(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        ts = timescales(T, k=5)
        assert ts.shape == (5,)


class TestMFPT:
    def test_target_has_zero_mfpt(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        m = mfpt(T, targets=3)
        assert m[3] == 0.0

    def test_other_states_positive(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        m = mfpt(T, targets=3)
        assert (m[[0, 1, 2]] > 0).all()

    def test_set_target(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        m = mfpt(T, targets=[2, 3])
        assert m[2] == 0.0
        assert m[3] == 0.0
        assert (m[[0, 1]] > 0).all()

    def test_rejects_empty_targets(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        with pytest.raises(ValueError, match="non-empty"):
            mfpt(T, targets=[])

    def test_rejects_out_of_range(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        with pytest.raises(ValueError, match="out of range"):
            mfpt(T, targets=42)


class TestMixingTime:
    def test_positive_finite(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        t = mixing_time(T)
        assert t > 0 and t < float("inf")


class TestMSMContainer:
    def test_from_graph_builds_coherent_state(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = GenotypePhenotypeMSM.from_graph(
            tiny_graph,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=10,
        )
        assert msm.n_states == 4
        assert isinstance(msm.transition_matrix, sp.csr_matrix)
        assert msm.fixation_model == "moran"
        assert msm.fixation_params["population_size"] == 10
        assert abs(msm.stationary.sum() - 1.0) < 1e-10
        row_sums = np.asarray(msm.transition_matrix.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_repr_contains_size_and_model(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = GenotypePhenotypeMSM.from_graph(
            tiny_graph,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=10,
        )
        r = repr(msm)
        assert "n_states=4" in r
        assert "moran" in r

    def test_fixation_params_is_mapping(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = GenotypePhenotypeMSM.from_graph(
            tiny_graph,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=10,
        )
        # We expose params as a Mapping; concrete type is a plain dict so the
        # whole MSM stays picklable.
        from collections.abc import Mapping

        assert isinstance(msm.fixation_params, Mapping)
        assert dict(msm.fixation_params) == {"population_size": 10}
