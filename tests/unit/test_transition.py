"""Unit tests for ``markov.build_transition_matrix``.

These tests pin the row-stochasticity contract from ``SCHEMA.md`` section 2,
the regression for v1 bug 1 (int-keyed alignment), and the regression for
v1 bug 2 (diagonal computed as 1 - row_sum, not by evaluating the fixation
kernel at ``f_i == f_j``).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap
from gpvolve import build_transition_matrix
from gpvolve.exceptions import ModelError, NonStochasticError


class TestRowStochasticity:
    """Section 2.1 of SCHEMA.md: rows sum to 1.0 within 1e-12."""

    def test_moran_tiny(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        assert T.shape == (4, 4)
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_sswm_tiny(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(tiny_graph, fitness_column="phenotypes", fixation="sswm")
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_mccandlish_tiny(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph,
            fitness_column="phenotypes",
            fixation="mccandlish",
            population_size=100,
        )
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_alias_mcclandish_back_compat(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        """v1 misspelling 'mcclandish' must still resolve."""
        T = build_transition_matrix(
            tiny_graph,
            fitness_column="phenotypes",
            fixation="mcclandish",
            population_size=100,
        )
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_weak_mutation_rejected_as_unbounded(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        """weak_mutation returns unbounded selection coefficients; not row-stochastic."""
        with pytest.raises(NonStochasticError, match="not bounded"):
            build_transition_matrix(
                tiny_graph, fitness_column="phenotypes", fixation="weak_mutation"
            )

    def test_binary_map_64(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=1000,
        )
        assert T.shape == (64, 64)
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


class TestBounds:
    """Section 2.2 of SCHEMA.md: every entry in [0, 1]."""

    def test_all_entries_in_unit_interval(self, binary_graph_64: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            binary_graph_64,
            fitness_column="phenotypes",
            fixation="moran",
            population_size=1000,
        )
        data = T.tocsr().data
        assert data.min() >= 0.0 - 1e-12
        assert data.max() <= 1.0 + 1e-12


class TestSparsityPattern:
    """Section 2.3 of SCHEMA.md: sparsity = graph.edges + diagonal."""

    def test_sparsity_pattern_matches_graph(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        coo = T.tocoo()
        nz_pairs = set(zip(coo.row.tolist(), coo.col.tolist(), strict=True))
        expected_offdiag = {(int(u), int(v)) for u, v in tiny_graph.edges()}
        expected_diag = {(i, i) for i in range(tiny_graph.number_of_nodes())}
        assert nz_pairs == expected_offdiag | expected_diag


class TestV1Bug1IndexAlignment:
    """v1 bug 1 regression: row/col indexed by gpm.data.index."""

    def test_int_keyed_rows_match_gpm_data(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        gpm = tiny_graph.gpm
        # If row i corresponds to gpm.data.iloc[i], then T[0, :] for the
        # wildtype "00" should have non-zero off-diagonal only at neighbors
        # of node 0 in the graph. We verify by mapping back through the graph.
        for i in range(gpm.data.shape[0]):
            row = T.getrow(i)
            nz_cols = set(row.indices.tolist()) - {i}
            expected = {int(v) for _, v in tiny_graph.out_edges(i)}
            assert nz_cols == expected, f"row {i} mismatch: {nz_cols} != {expected}"


class TestV1Bug2DiagonalComputation:
    """v1 bug 2 regression: diagonal = 1 - sum(off-diagonals), never f(f_i, f_i)."""

    def test_diagonal_is_residual(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        for i in range(T.shape[0]):
            row = T.getrow(i).toarray().ravel()
            off_diag = row.sum() - row[i]
            expected_diag = 1.0 - off_diag
            assert abs(row[i] - expected_diag) < 1e-12

    def test_constant_landscape_diagonal_is_one_minus_offdiag_at_zero_selection(
        self,
    ) -> None:
        """On a flat landscape, fixation == neutral and self-loop residual is consistent."""
        gpm = GenotypePhenotypeMap(
            wildtype="00",
            genotypes=["00", "01", "10", "11"],
            phenotypes=[1.0, 1.0, 1.0, 1.0],
        )
        graph = GenotypePhenotypeGraph.from_gpm(gpm)
        T = build_transition_matrix(graph, fitness_column="phenotypes", fixation="sswm")
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


class TestErrorHandling:
    def test_unknown_model_raises(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        with pytest.raises(ModelError, match="unknown fixation model"):
            build_transition_matrix(tiny_graph, fitness_column="phenotypes", fixation="nope")

    def test_missing_required_params_raises(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        with pytest.raises(ModelError, match="requires params"):
            build_transition_matrix(tiny_graph, fitness_column="phenotypes", fixation="moran")

    def test_missing_fitness_column_raises(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        with pytest.raises(ModelError, match=r"not in gpm\.data"):
            build_transition_matrix(tiny_graph, fitness_column="missing", fixation="sswm")

    def test_invalid_self_loops_mode_raises(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        with pytest.raises(ModelError, match="self_loops"):
            build_transition_matrix(
                tiny_graph,
                fitness_column="phenotypes",
                fixation="sswm",
                self_loops="loose",  # type: ignore[arg-type]
            )


class TestSparsityFormat:
    def test_returns_csr_matrix(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        assert isinstance(T, sp.csr_matrix)
        assert T.dtype == np.float64
