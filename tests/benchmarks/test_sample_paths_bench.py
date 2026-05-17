"""pytest-benchmark cases for the Rust-accelerated walker sampler.

Run with ``uv run pytest tests/benchmarks -k bench``. These are skipped by
default in the main pytest run (collection filter in pyproject) since they
take seconds rather than milliseconds and would slow CI.

The "python" variants here intentionally only exercise small landscapes;
the pure-Python path is ~400x slower on 2^12+ maps and would push CI past
the timeout.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap

from gpvolve import ConvergenceCheck, GenotypePhenotypeMSM, sample_paths
import gpvolve.paths.stochastic as stochastic


def _build_msm(sites: int) -> GenotypePhenotypeMSM:
    rng = np.random.default_rng(0)
    eff = rng.normal(0.0, 0.4, size=sites)
    gts = ["".join(b) for b in product("01", repeat=sites)]
    phens = [
        float(np.exp(sum(eff[i] for i, c in enumerate(g) if c == "1"))) for g in gts
    ]
    gpm = GenotypePhenotypeMap(wildtype="0" * sites, genotypes=gts, phenotypes=phens)
    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    return GenotypePhenotypeMSM.from_graph(
        graph, fitness_column="phenotypes", fixation="moran", population_size=100
    )


@pytest.mark.benchmark(group="sample_paths_rust")
@pytest.mark.parametrize("sites", [6, 10, 14])
def test_bench_sample_paths_rust(benchmark, sites: int) -> None:
    """Time the Rust walker sampler for chunk_size=5000 walkers across map sizes."""
    if not stochastic._RUST_AVAILABLE:
        pytest.skip("gpvolve._rust not built")
    msm = _build_msm(sites)
    n = 2**sites
    target = int(np.argmax(msm.gpm.data["phenotypes"].to_numpy()))
    cc = ConvergenceCheck(ess_min=10, chunk_size=5000, max_walkers=5000, n_chains=4)
    benchmark.pedantic(
        sample_paths,
        args=(msm, 0, target),
        kwargs={"convergence": cc, "seed": 0},
        rounds=3,
        iterations=1,
    )
    assert n > 0


@pytest.mark.benchmark(group="sample_paths_python")
@pytest.mark.parametrize("sites", [6, 8])
def test_bench_sample_paths_python(benchmark, sites: int, monkeypatch) -> None:
    """Time the pure-Python fallback for a fair head-to-head on small maps."""
    monkeypatch.setattr(stochastic, "_RUST_AVAILABLE", False)
    msm = _build_msm(sites)
    target = int(np.argmax(msm.gpm.data["phenotypes"].to_numpy()))
    # Tighter budget so the Python fallback finishes in reasonable time.
    cc = ConvergenceCheck(ess_min=10, chunk_size=500, max_walkers=500, n_chains=4)
    benchmark.pedantic(
        sample_paths,
        args=(msm, 0, target),
        kwargs={"convergence": cc, "seed": 0},
        rounds=2,
        iterations=1,
    )
