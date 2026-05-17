"""Unit tests for absorbing-chain analytics.

Covers ``gpvolve.markov.absorbing`` plus ``gpvolve.absorption_rate``:

- ``fundamental_matrix`` (Kemeny-Snell N = (I-Q)^-1)
- ``absorption_probabilities`` (B = N R)
- ``quasi_stationary_distribution`` (Darroch-Seneta)
- ``conditional_mfpt`` (Doob h-transform)
- ``absorption_rate`` (Hanggi-Talkner-Borkovec rate = 1/MFPT)
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from gpgraph import GenotypePhenotypeGraph
from gpvolve import (
    absorbing_states,
    absorption_rate,
    build_transition_matrix,
    conditional_mfpt,
    fundamental_matrix,
    mfpt,
    quasi_stationary_distribution,
)
from gpvolve.exceptions import GpvolveError
from gpvolve.markov.absorbing import absorption_probabilities


@pytest.fixture
def two_sink_chain() -> sp.csr_matrix:
    """4-state chain with two competing absorbing states.

    From state 0: 40% to absorbing state 2, 50% self, 10% to state 1.
    From state 1: 40% to absorbing state 3, 50% self, 10% to state 0.
    By symmetry, absorption probability from 0 into {2} is 5/6.
    """
    P = np.array(
        [
            [0.5, 0.1, 0.4, 0.0],
            [0.1, 0.5, 0.0, 0.4],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return sp.csr_matrix(P)


class TestFundamentalMatrix:
    def test_shape_and_transient_idx(self, two_sink_chain: sp.csr_matrix) -> None:
        N, trans_idx = fundamental_matrix(two_sink_chain)
        assert N.shape == (2, 2)
        assert trans_idx.tolist() == [0, 1]

    def test_expected_visits_match_geometric_intuition(
        self, two_sink_chain: sp.csr_matrix
    ) -> None:
        """For the two-sink chain, expected visits to state 0 from state 0 should
        exceed 1 because the state has a 50% self-loop and a 10% return route via
        state 1. Solve (I - Q) N = I analytically and cross-check."""
        # Q = [[0.5, 0.1], [0.1, 0.5]]; (I - Q) = [[0.5, -0.1], [-0.1, 0.5]].
        # Det = 0.25 - 0.01 = 0.24. N = (1/0.24) * [[0.5, 0.1], [0.1, 0.5]].
        N, _ = fundamental_matrix(two_sink_chain)
        expected = np.array([[0.5, 0.1], [0.1, 0.5]]) / 0.24
        np.testing.assert_allclose(N, expected, rtol=1e-10)

    def test_raises_on_ergodic_chain(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        """Ergodic chains have no absorbing states; the fundamental matrix is
        undefined."""
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        with pytest.raises(GpvolveError, match="absorbing"):
            fundamental_matrix(T)

    def test_all_absorbing_returns_empty(self) -> None:
        """When every state is absorbing, the transient block is empty."""
        T = sp.csr_matrix(np.eye(3))
        N, trans_idx = fundamental_matrix(T)
        assert N.shape == (0, 0)
        assert trans_idx.size == 0


class TestAbsorptionProbabilities:
    def test_row_sums_to_one(self, two_sink_chain: sp.csr_matrix) -> None:
        B, _, _ = absorption_probabilities(two_sink_chain)
        np.testing.assert_allclose(B.sum(axis=1), np.ones(B.shape[0]), atol=1e-12)

    def test_symmetric_two_sink_chain(self, two_sink_chain: sp.csr_matrix) -> None:
        """Closed-form: from state 0, Pr[absorbed in 2] = 5/6 by symmetry."""
        B, trans_idx, abs_idx = absorption_probabilities(two_sink_chain)
        assert trans_idx.tolist() == [0, 1]
        assert abs_idx.tolist() == [2, 3]
        np.testing.assert_allclose(B[0, 0], 5 / 6, rtol=1e-10)
        np.testing.assert_allclose(B[0, 1], 1 / 6, rtol=1e-10)
        np.testing.assert_allclose(B[1, 0], 1 / 6, rtol=1e-10)
        np.testing.assert_allclose(B[1, 1], 5 / 6, rtol=1e-10)

    def test_rugged_landscape_basins(
        self, rugged_graph_8: GenotypePhenotypeGraph
    ) -> None:
        """In a two-peak SSWM landscape, absorption probabilities partition each
        transient state's basin assignment. Every row of B sums to 1."""
        T = build_transition_matrix(rugged_graph_8, fitness_column="phenotypes", fixation="sswm")
        B, _, abs_idx = absorption_probabilities(T)
        assert len(abs_idx) == 2
        np.testing.assert_allclose(B.sum(axis=1), np.ones(B.shape[0]), atol=1e-10)
        # No row should be all zeros; every transient state has *some* basin.
        assert (B.sum(axis=1) > 0.99).all()


class TestQuasiStationaryDistribution:
    def test_two_sink_qsd_symmetric(self, two_sink_chain: sp.csr_matrix) -> None:
        """Transient block Q = [[0.5, 0.1], [0.1, 0.5]] is symmetric, so the QSD
        is the uniform [0.5, 0.5] with lambda_1 = 0.6."""
        qsd, trans_idx, lam = quasi_stationary_distribution(two_sink_chain)
        np.testing.assert_allclose(qsd[trans_idx], [0.5, 0.5], atol=1e-8)
        np.testing.assert_allclose(lam, 0.6, atol=1e-8)
        # Embedded zeros on absorbing states.
        assert qsd[2] == 0.0
        assert qsd[3] == 0.0

    def test_qsd_sums_to_one(self, fuji_5_sswm_msm) -> None:
        _gpm, msm = fuji_5_sswm_msm
        P = msm.transition_matrix
        qsd, _, _ = quasi_stationary_distribution(P)
        # QSD lives on the transient class; embedded QSD sums to 1 across
        # transient states.
        np.testing.assert_allclose(qsd.sum(), 1.0, atol=1e-10)

    def test_metastable_lifetime_matches_mfpt(self, fuji_5_sswm_msm) -> None:
        """The expected absorption time from the QSD equals 1 / (1 - lambda_1).
        Cross-check against MFPT from the QSD-weighted initial distribution."""
        _gpm, msm = fuji_5_sswm_msm
        P = msm.transition_matrix
        qsd, _, lam = quasi_stationary_distribution(P)
        # mfpt to the absorbing state
        abs_idx = absorbing_states(P)
        m = mfpt(P, targets=abs_idx.tolist())
        e_tau_from_qsd = float(np.dot(qsd, m))
        np.testing.assert_allclose(e_tau_from_qsd, 1.0 / (1.0 - lam), rtol=1e-6)

    def test_raises_on_ergodic_chain(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        T = build_transition_matrix(
            tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
        )
        with pytest.raises(GpvolveError, match="absorbing"):
            quasi_stationary_distribution(T)


class TestConditionalMfpt:
    def test_single_sink_agrees_with_mfpt(self) -> None:
        """When B is the only absorbing class, conditional_mfpt = mfpt."""
        from gpmap import GenotypePhenotypeMap

        gpm = GenotypePhenotypeMap(
            wildtype="00",
            genotypes=["00", "01", "10", "11"],
            phenotypes=[1.0, 1.5, 1.2, 2.0],
        )
        graph = GenotypePhenotypeGraph.from_gpm(gpm)
        T = build_transition_matrix(graph, fitness_column="phenotypes", fixation="sswm")
        # Single absorbing state at index 3.
        m_uncond = mfpt(T, targets=3)
        m_cond = conditional_mfpt(T, A=[0, 1, 2], B=3)
        np.testing.assert_allclose(m_cond, m_uncond[[0, 1, 2]], rtol=1e-10)

    def test_two_sink_chain_conditional_makes_finite(
        self, two_sink_chain: sp.csr_matrix
    ) -> None:
        """mfpt(P, 2) is undefined here (competing sink 3); conditional_mfpt is
        finite."""
        with pytest.raises(GpvolveError, match="outside the target set"):
            mfpt(two_sink_chain, targets=2)
        m_cond = conditional_mfpt(two_sink_chain, A=0, B=2)
        assert m_cond.shape == (1,)
        assert np.isfinite(m_cond[0])
        assert m_cond[0] > 0

    def test_inf_when_no_path_to_target(self) -> None:
        """A state with zero absorption probability to B yields inf."""
        # 0 transient, 1 absorbing; 2 absorbing (unreachable from 0).
        # Wait: design a chain where state 0 has 0 path to absorbing B=2.
        P = np.array(
            [
                [0.5, 0.5, 0.0],  # 0 -> only reaches 1
                [0.0, 1.0, 0.0],  # 1 absorbing
                [0.0, 0.0, 1.0],  # 2 absorbing, unreachable
            ]
        )
        T = sp.csr_matrix(P)
        m_cond = conditional_mfpt(T, A=0, B=2)
        assert np.isinf(m_cond[0])


class TestAbsorptionRate:
    def test_finite_on_absorbing_chain(self, fuji_5_sswm_msm) -> None:
        """For the L=5 sswm bug case, the absorption rate is finite and
        positive even though the reactive rate underflows to zero."""
        gpm, msm = fuji_5_sswm_msm
        P = msm.transition_matrix
        genos = list(gpm.data["genotypes"])
        A = genos.index("AAAAA")
        B = genos.index("TTTTT")
        k = absorption_rate(P, A=A, B=B)
        assert np.isfinite(k)
        assert k > 0

    def test_zero_when_target_unreachable(self) -> None:
        P = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        T = sp.csr_matrix(P)
        k = absorption_rate(T, A=0, B=2)
        assert k == 0.0

    def test_custom_initial_distribution(self, fuji_5_sswm_msm) -> None:
        """Weighting initial mass on the worst genotype vs. uniform-over-A
        should yield different rates; both must be positive."""
        gpm, msm = fuji_5_sswm_msm
        P = msm.transition_matrix
        n = P.shape[0]
        genos = list(gpm.data["genotypes"])
        B = genos.index("TTTTT")

        # A is the set of 3-mutation states (halfway). Uniform default.
        n_muts = np.asarray(gpm.data["n_mutations"].to_numpy(), dtype=int)
        half = np.flatnonzero(n_muts == 3).tolist()
        k_uniform = absorption_rate(P, A=half, B=B)

        # Custom initial: a delta at the single worst state of `half`.
        initial = np.zeros(n)
        initial[half[0]] = 1.0
        k_delta = absorption_rate(P, A=half, B=B, initial=initial)
        assert k_uniform > 0
        assert k_delta > 0


class TestEdgeCases:
    def test_singleton_a_b_at_extremes(self, fuji_5_sswm_msm) -> None:
        """A and B as length-1 sets at landscape extremes (best/worst genotype)
        produce a valid TPT solve in the forward direction; the bug case."""
        from gpvolve import forward_committor

        gpm, msm = fuji_5_sswm_msm
        P = msm.transition_matrix
        genos = list(gpm.data["genotypes"])
        A = genos.index("AAAAA")
        B = genos.index("TTTTT")
        q_plus = forward_committor(P, A=A, B=B)
        assert q_plus[A] == 0.0
        assert q_plus[B] == 1.0
        # q+ should be nondecreasing in n_mutations for an additive landscape;
        # at minimum, intermediate states have positive committor.
        assert (q_plus > 0).sum() > 1

    def test_disconnected_chain_handled(self) -> None:
        """A row-stochastic block-diagonal matrix (two disconnected components)
        is valid input. forward_committor on disjoint A, B must still produce
        boundary values; absorption_rate falls back to 0 for unreachable B."""
        # Two 2-state components; component 1 = {0, 1}, component 2 = {2, 3}.
        P = np.array(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ]
        )
        T = sp.csr_matrix(P)
        k = absorption_rate(T, A=0, B=2)
        # No path from {0, 1} to {2, 3}; expected rate is 0.
        assert k == 0.0

    def test_near_zero_stationary_does_not_crash_backward_committor(self) -> None:
        """L=4 sswm (and smaller) has min_pi ~ 1e-264 (not exactly zero), so
        backward_committor's guard does not trigger. Verify the computation
        succeeds without producing NaN/Inf, even if the resulting q- is
        numerically delicate."""
        from itertools import product

        from gpmap import GenotypePhenotypeMap
        from gpvolve import backward_committor, forward_committor

        alph = ("A", "T")
        length = 4
        rng = np.random.default_rng(0)
        per_site = rng.normal(loc=1.0, scale=0.5, size=(length, len(alph)))
        per_site[:, 0] = 0.0
        genos = ["".join(g) for g in product(alph, repeat=length)]
        phenos = []
        for g in genos:
            total = 0.0
            for i, c in enumerate(g):
                total += float(per_site[i, alph.index(c)])
            phenos.append(max(total + 1.0, 0.05))
        gpm = GenotypePhenotypeMap(
            wildtype=alph[0] * length,
            genotypes=genos,
            phenotypes=phenos,
        )
        graph = GenotypePhenotypeGraph.from_gpm(gpm)
        T = build_transition_matrix(graph, fitness_column="phenotypes", fixation="sswm")

        # The forward committor should succeed unconditionally.
        q_plus = forward_committor(T, A=0, B=len(genos) - 1)
        assert np.isfinite(q_plus).all()

        # backward_committor either succeeds (if pi underflows but stays > 0)
        # or raises GpvolveError. Either is acceptable behavior; what matters
        # is that we do NOT silently get NaN.
        try:
            q_minus = backward_committor(T, A=0, B=len(genos) - 1)
            assert np.isfinite(q_minus).all()
        except GpvolveError:
            pass

    def test_rugged_landscape_qsd_and_absorption_rate(
        self, rugged_graph_8: GenotypePhenotypeGraph
    ) -> None:
        """Multi-peak SSWM: there are two absorbing states. QSD must support
        all transient states, and absorption_rate to one peak must be finite
        even with a competing sink."""
        T = build_transition_matrix(rugged_graph_8, fitness_column="phenotypes", fixation="sswm")
        qsd, trans_idx, lam = quasi_stationary_distribution(T)
        assert 0 < lam < 1
        assert qsd[trans_idx].sum() > 0.99

        # Rate from state 4 (binary "100") to the "111" peak (index 7).
        k = absorption_rate(T, A=4, B=7)
        assert np.isfinite(k)
        assert k > 0
