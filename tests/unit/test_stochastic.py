"""Unit tests for ``paths.stochastic.sample_paths`` and ``ConvergenceCheck``.

Marginal-distribution agreement against the forward committor is left to the
property test in ``tests/property/test_path_sampler_marginal.py`` (50k walker
budget), since it dominates this file's runtime if duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpvolve import ConvergenceCheck, GenotypePhenotypeMSM, sample_paths
from gpvolve.exceptions import ConvergenceError, GpvolveError


def _make_msm(graph: GenotypePhenotypeGraph) -> GenotypePhenotypeMSM:
    return GenotypePhenotypeMSM.from_graph(
        graph, fitness_column="phenotypes", fixation="moran", population_size=50
    )


class TestSamplePaths:
    def test_basic_run(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = _make_msm(tiny_graph)
        ens = sample_paths(
            msm,
            source=0,
            targets=3,
            convergence=ConvergenceCheck(
                ess_min=10.0, chunk_size=200, max_walkers=4000, n_chains=4
            ),
            seed=0,
        )
        assert ens.method == "stochastic"
        assert ens.source == 0
        assert ens.targets == (3,)
        assert ens.probabilities.shape == (len(ens.paths),)
        for p in ens.paths:
            assert p[0] == 0

    def test_metadata_has_convergence_payload(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = _make_msm(tiny_graph)
        ens = sample_paths(
            msm,
            source=0,
            targets=3,
            convergence=ConvergenceCheck(
                ess_min=10.0, chunk_size=200, max_walkers=4000, n_chains=4
            ),
            seed=1,
        )
        cv = ens.metadata["convergence"]
        assert cv["converged"] is True
        assert cv["n_walkers"] > 0
        assert cv["n_chunks"] >= 1
        assert set(cv["ess"]) == {3}
        assert set(cv["rhat"]) == {3}

    def test_reproducible_with_seed(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = _make_msm(tiny_graph)
        kwargs = {
            "convergence": ConvergenceCheck(
                ess_min=10.0, chunk_size=200, max_walkers=4000, n_chains=4
            ),
            "seed": 12345,
        }
        a = sample_paths(msm, source=0, targets=3, **kwargs)
        b = sample_paths(msm, source=0, targets=3, **kwargs)
        assert a.paths == b.paths
        np.testing.assert_array_equal(a.probabilities, b.probabilities)

    def test_max_walkers_raises_on_unreachable(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = _make_msm(tiny_graph)
        # Very tight ESS target and tiny budget: should raise ConvergenceError.
        with pytest.raises(ConvergenceError):
            sample_paths(
                msm,
                source=0,
                targets=3,
                convergence=ConvergenceCheck(
                    ess_min=1e9, chunk_size=50, max_walkers=100, n_chains=4
                ),
                seed=0,
            )

    def test_source_cannot_be_target(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = _make_msm(tiny_graph)
        with pytest.raises(GpvolveError, match="source cannot be in targets"):
            sample_paths(msm, source=0, targets=0)

    def test_out_of_range_target(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = _make_msm(tiny_graph)
        with pytest.raises(GpvolveError, match="target index out of range"):
            sample_paths(msm, source=0, targets=99)
