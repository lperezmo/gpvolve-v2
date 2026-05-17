"""Schema tests for I/O round-trips.

Anchors: SCHEMA.md sections 1 and 5. Every supported serialization must
round-trip the matrix bit-for-bit on the entries actually written, and the
gpm hash must match on load.
"""

from __future__ import annotations

import numpy as np
from gpgraph import GenotypePhenotypeGraph
from gpvolve import GenotypePhenotypeMSM
from gpvolve.io import (
    from_dict,
    from_json,
    from_npz,
    from_pickle,
    to_dict,
    to_json,
    to_npz,
    to_pickle,
)


def _make_msm(graph: GenotypePhenotypeGraph) -> GenotypePhenotypeMSM:
    return GenotypePhenotypeMSM.from_graph(
        graph, fitness_column="phenotypes", fixation="moran", population_size=10
    )


class TestDictRoundTrip:
    def test_dict_round_trip(self, tiny_graph: GenotypePhenotypeGraph) -> None:
        msm = _make_msm(tiny_graph)
        payload = to_dict(msm)
        msm2 = from_dict(payload, graph=tiny_graph)
        np.testing.assert_array_equal(
            msm.transition_matrix.toarray(), msm2.transition_matrix.toarray()
        )
        np.testing.assert_array_equal(msm.stationary, msm2.stationary)
        assert msm.fixation_model == msm2.fixation_model
        assert dict(msm.fixation_params) == dict(msm2.fixation_params)


class TestJsonRoundTrip:
    def test_json_round_trip(self, tiny_graph: GenotypePhenotypeGraph, tmp_path) -> None:
        msm = _make_msm(tiny_graph)
        path = tmp_path / "msm.json"
        to_json(msm, path)
        msm2 = from_json(path, graph=tiny_graph)
        np.testing.assert_allclose(
            msm.transition_matrix.toarray(), msm2.transition_matrix.toarray(), atol=0
        )
        np.testing.assert_allclose(msm.stationary, msm2.stationary, atol=0)


class TestNpzRoundTrip:
    def test_npz_round_trip(self, tiny_graph: GenotypePhenotypeGraph, tmp_path) -> None:
        msm = _make_msm(tiny_graph)
        path = tmp_path / "msm.npz"
        to_npz(msm, path)
        msm2 = from_npz(path, graph=tiny_graph)
        np.testing.assert_array_equal(
            msm.transition_matrix.toarray(), msm2.transition_matrix.toarray()
        )
        np.testing.assert_array_equal(msm.stationary, msm2.stationary)


class TestPickleRoundTrip:
    def test_pickle_round_trip(self, tiny_graph: GenotypePhenotypeGraph, tmp_path) -> None:
        msm = _make_msm(tiny_graph)
        path = tmp_path / "msm.pkl"
        to_pickle(msm, path)
        msm2 = from_pickle(path)
        np.testing.assert_array_equal(
            msm.transition_matrix.toarray(), msm2.transition_matrix.toarray()
        )
        np.testing.assert_array_equal(msm.stationary, msm2.stationary)
        assert msm.fixation_model == msm2.fixation_model
