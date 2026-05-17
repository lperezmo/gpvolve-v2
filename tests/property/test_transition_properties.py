"""Hypothesis property tests for ``build_transition_matrix``.

Anchors: SCHEMA.md section 2.

- Row sums equal 1.0 to 1e-12 across the configuration space.
- Entries are in [0, 1].
- Diagonal equals 1 - sum(off-diagonals) for every row.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap
from gpvolve import build_transition_matrix
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def _make_binary_gpm(n_sites: int, phenotypes: list[float]) -> GenotypePhenotypeMap:
    genotypes = ["".join(bits) for bits in product("01", repeat=n_sites)]
    return GenotypePhenotypeMap(
        wildtype="0" * n_sites,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_sites=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    fixation=st.sampled_from(["sswm", "moran", "mccandlish"]),
    pop_size=st.integers(min_value=2, max_value=2000),
)
def test_row_sums_are_one(n_sites: int, seed: int, fixation: str, pop_size: int) -> None:
    rng = np.random.default_rng(seed)
    n = 2**n_sites
    phenotypes = np.exp(rng.normal(0.0, 0.5, size=n)).tolist()
    gpm = _make_binary_gpm(n_sites, phenotypes)
    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    if fixation in {"moran", "mccandlish"}:
        T = build_transition_matrix(
            graph, fitness_column="phenotypes", fixation=fixation, population_size=pop_size
        )
    else:
        T = build_transition_matrix(graph, fitness_column="phenotypes", fixation=fixation)
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-12)


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_sites=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_entries_in_unit_interval(n_sites: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 2**n_sites
    phenotypes = np.exp(rng.normal(0.0, 0.5, size=n)).tolist()
    gpm = _make_binary_gpm(n_sites, phenotypes)
    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    T = build_transition_matrix(
        graph, fitness_column="phenotypes", fixation="moran", population_size=100
    )
    data = T.tocsr().data
    assert data.min() >= 0.0 - 1e-12
    assert data.max() <= 1.0 + 1e-12


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_sites=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_diagonal_is_row_residual(n_sites: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 2**n_sites
    phenotypes = np.exp(rng.normal(0.0, 0.5, size=n)).tolist()
    gpm = _make_binary_gpm(n_sites, phenotypes)
    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    T = build_transition_matrix(
        graph, fitness_column="phenotypes", fixation="moran", population_size=100
    ).tocsr()
    for i in range(T.shape[0]):
        row = T.getrow(i).toarray().ravel()
        off_sum = row.sum() - row[i]
        assert abs(row[i] - (1.0 - off_sum)) < 1e-12


@pytest.mark.parametrize("fixation", ["sswm", "moran", "mccandlish"])
def test_flat_landscape_no_drift(fixation: str) -> None:
    """On a flat landscape the chain has no net drift (rows still sum to 1)."""
    gpm = _make_binary_gpm(3, [1.0] * 8)
    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    kwargs = {"population_size": 100} if fixation in {"moran", "mccandlish"} else {}
    T = build_transition_matrix(graph, fitness_column="phenotypes", fixation=fixation, **kwargs)
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)
