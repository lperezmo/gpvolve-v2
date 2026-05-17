"""Smoke tests for the pyplot module (matplotlib backend = Agg)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from gpgraph import GenotypePhenotypeGraph
from gpvolve import (
    GenotypePhenotypeMSM,
    build_transition_matrix,
    reactive_flux,
    timescales,
)
from gpvolve.pyplot import (
    draw_flux_heatmap,
    draw_landscape_1d,
    draw_stationary,
    draw_timescales,
    draw_transition_matrix,
)


def test_draw_transition_matrix(tiny_graph: GenotypePhenotypeGraph) -> None:
    msm = GenotypePhenotypeMSM.from_graph(
        tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
    )
    ax = draw_transition_matrix(msm)
    assert ax is not None


def test_draw_stationary(tiny_graph: GenotypePhenotypeGraph) -> None:
    msm = GenotypePhenotypeMSM.from_graph(
        tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
    )
    ax = draw_stationary(msm)
    assert ax is not None


def test_draw_flux_heatmap(tiny_graph: GenotypePhenotypeGraph) -> None:
    T = build_transition_matrix(
        tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
    )
    F = reactive_flux(T, A=0, B=3)
    ax = draw_flux_heatmap(F)
    assert ax is not None


def test_draw_timescales(tiny_graph: GenotypePhenotypeGraph) -> None:
    T = build_transition_matrix(
        tiny_graph, fitness_column="phenotypes", fixation="moran", population_size=10
    )
    ts = timescales(T, k=3)
    ax = draw_timescales(ts)
    assert ax is not None


def test_draw_landscape_1d(tiny_graph: GenotypePhenotypeGraph) -> None:
    ax = draw_landscape_1d(tiny_graph, fitness_column="phenotypes")
    assert ax is not None
