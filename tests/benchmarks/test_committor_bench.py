"""pytest-benchmark cases for the Rust BiCGSTAB committor solver.

Run with ``uv run pytest tests/benchmarks/test_committor_bench.py
--benchmark-only``. Comparing against scipy spsolve at sites>=13 is
intentionally skipped: spsolve grew past 26 s on a 2^13 map during the
profiling that justified writing BiCGSTAB in the first place. The
small-N spsolve cases run head-to-head for parity.
"""

from __future__ import annotations

from itertools import product

import gpvolve.paths.tpt as tpt
import numpy as np
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap
from gpvolve import build_transition_matrix, forward_committor


def _make_graph(sites: int) -> GenotypePhenotypeGraph:
    rng = np.random.default_rng(0)
    eff = rng.normal(0.0, 0.4, size=sites)
    gts = ["".join(b) for b in product("01", repeat=sites)]
    phens = [float(np.exp(sum(eff[i] for i, c in enumerate(g) if c == "1"))) for g in gts]
    gpm = GenotypePhenotypeMap(wildtype="0" * sites, genotypes=gts, phenotypes=phens)
    return GenotypePhenotypeGraph.from_gpm(gpm)


@pytest.mark.benchmark(group="committor_rust")
@pytest.mark.parametrize("sites", [10, 12, 14])
def test_bench_forward_committor_rust(benchmark, sites: int) -> None:
    """Time Rust-BiCGSTAB-backed forward committor across map sizes."""
    if not tpt._RUST_BICGSTAB_AVAILABLE:
        pytest.skip("gpvolve._rust not built")
    graph = _make_graph(sites)
    T = build_transition_matrix(
        graph, fitness_column="phenotypes", fixation="moran", population_size=100
    )
    target = int(np.argmax(graph.gpm.data["phenotypes"].to_numpy()))
    benchmark.pedantic(
        forward_committor,
        args=(T, 0, target),
        rounds=3,
        iterations=1,
    )


@pytest.mark.benchmark(group="committor_spsolve")
@pytest.mark.parametrize("sites", [10, 12])
def test_bench_forward_committor_spsolve(benchmark, sites: int, monkeypatch) -> None:
    """Time scipy-spsolve forward committor at sizes where it still completes."""
    monkeypatch.setattr(tpt, "_RUST_BICGSTAB_AVAILABLE", False)
    graph = _make_graph(sites)
    T = build_transition_matrix(
        graph, fitness_column="phenotypes", fixation="moran", population_size=100
    )
    target = int(np.argmax(graph.gpm.data["phenotypes"].to_numpy()))
    benchmark.pedantic(
        forward_committor,
        args=(T, 0, target),
        rounds=2,
        iterations=1,
    )
