"""Unit tests for chain validators.

Covers the predicates added in ``gpvolve.markov.validation``:

- ``is_irreducible`` / ``is_strongly_connected``
- ``is_ergodic``
- ``is_reversible`` (detailed-balance test)
- ``absorbing_states`` / ``transient_states``
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from gpgraph import GenotypePhenotypeGraph
from gpvolve import (
    absorbing_states,
    build_transition_matrix,
    is_ergodic,
    is_irreducible,
    is_reversible,
    transient_states,
)


class TestIsErgodic:
    def test_ergodic_moran_chain(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        """Moran at any finite N produces an ergodic chain."""
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        assert is_ergodic(T)

    def test_non_ergodic_sswm_chain(self) -> None:
        """SSWM on a single-peak landscape is non-ergodic (absorbing peak)."""
        from gpmap import GenotypePhenotypeMap

        gpm = GenotypePhenotypeMap(
            wildtype="00",
            genotypes=["00", "01", "10", "11"],
            phenotypes=[1.0, 1.5, 1.2, 2.0],
        )
        graph = GenotypePhenotypeGraph.from_gpm(gpm)
        T = build_transition_matrix(graph, fitness_column="phenotypes", fixation="sswm")
        assert not is_ergodic(T)

    def test_singleton_chain(self) -> None:
        """Single-state chains are trivially ergodic."""
        T = sp.csr_matrix(np.array([[1.0]]))
        assert is_ergodic(T)

    def test_empty_chain(self) -> None:
        """Zero-state chains are trivially ergodic (vacuous truth)."""
        T = sp.csr_matrix(np.empty((0, 0)))
        assert is_ergodic(T)

    def test_periodic_no_self_loop_is_not_ergodic(self) -> None:
        """A cycle with no self-loops is irreducible but periodic; the sufficient
        criterion correctly refuses to call it ergodic."""
        # 3-cycle: 0 -> 1 -> 2 -> 0. Period 3, no self-loops.
        T = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ]
            )
        )
        assert is_irreducible(T)
        assert not is_ergodic(T)


class TestIsReversible:
    def test_moran_at_n10_is_reversible(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        """Moran with detailed-balance fitness landscape satisfies detailed balance."""
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        assert is_reversible(T)

    def test_sswm_not_reversible(self) -> None:
        """SSWM violates detailed balance: P_ij > 0 implies f_j > f_i, so P_ji = 0
        but P_ij is not, breaking pi_i P_ij = pi_j P_ji unless both sides are 0."""
        from gpmap import GenotypePhenotypeMap

        gpm = GenotypePhenotypeMap(
            wildtype="00",
            genotypes=["00", "01", "10", "11"],
            phenotypes=[1.0, 1.5, 1.2, 2.0],
        )
        graph = GenotypePhenotypeGraph.from_gpm(gpm)
        T = build_transition_matrix(graph, fitness_column="phenotypes", fixation="sswm")
        # SSWM chains have pi with zero entries; is_reversible must return False
        # rather than crash on the detailed-balance test.
        assert not is_reversible(T)


class TestAbsorbingStates:
    def test_no_absorbing_in_ergodic_chain(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        assert absorbing_states(T).size == 0

    def test_sswm_single_peak(self) -> None:
        """SSWM on a single-peak chain has one absorbing state at the peak."""
        from gpmap import GenotypePhenotypeMap

        gpm = GenotypePhenotypeMap(
            wildtype="00",
            genotypes=["00", "01", "10", "11"],
            phenotypes=[1.0, 1.5, 1.2, 2.0],
        )
        graph = GenotypePhenotypeGraph.from_gpm(gpm)
        T = build_transition_matrix(graph, fitness_column="phenotypes", fixation="sswm")
        abs_idx = absorbing_states(T)
        assert abs_idx.tolist() == [3]  # peak at "11" -> index 3.

    def test_rugged_landscape_two_peaks(self, rugged_graph_8: GenotypePhenotypeGraph) -> None:
        """SSWM on a deliberately rugged landscape (two local maxima) yields TWO
        absorbing states. This exercises the multi-absorbing-state path."""
        T = build_transition_matrix(rugged_graph_8, fitness_column="phenotypes", fixation="sswm")
        abs_idx = absorbing_states(T)
        # Genotype "000" (index 0) and "111" (index 7) are local maxima.
        assert sorted(abs_idx.tolist()) == [0, 7]

    def test_transient_is_complement(self, rugged_graph_8: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(rugged_graph_8, fitness_column="phenotypes", fixation="sswm")
        abs_idx = absorbing_states(T)
        trans_idx = transient_states(T)
        n = T.shape[0]
        assert sorted(abs_idx.tolist() + trans_idx.tolist()) == list(range(n))
        assert set(abs_idx.tolist()).isdisjoint(set(trans_idx.tolist()))

    def test_tolerance_rejects_near_sink(self) -> None:
        """A state with T_ii = 1 - small_epsilon is not absorbing under default tol."""
        T = np.array(
            [
                [1.0 - 1e-6, 1e-6, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        T_sp = sp.csr_matrix(T)
        # Index 0 has T_00 = 1 - 1e-6 (NOT absorbing under default tol=1e-12).
        # Index 2 is genuinely absorbing.
        abs_idx = absorbing_states(T_sp)
        assert abs_idx.tolist() == [2]


@pytest.mark.parametrize(
    "fixation,population_size,expected_ergodic",
    [
        ("moran", 10, True),
        ("moran", 1000, True),
        ("mccandlish", 100, True),
        ("sswm", None, False),  # single-peak landscape -> always non-ergodic
    ],
)
def test_ergodicity_across_models(
    tiny_graph: GenotypePhenotypeGraph,
    fixation: str,
    population_size: int | None,
    expected_ergodic: bool,
) -> None:
    """Cross-check ergodicity expectations against fixation models on the same
    single-peak landscape."""
    kwargs: dict = {}
    if population_size is not None:
        kwargs["population_size"] = population_size
    T = build_transition_matrix(
        tiny_graph, fitness_column="phenotypes", fixation=fixation, **kwargs
    )
    assert is_ergodic(T) is expected_ergodic
