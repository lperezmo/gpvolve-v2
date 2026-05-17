"""Shared pytest fixtures for gpvolve-v2.

Two canonical fixtures used across the test tree:

- ``tiny_graph`` : a 4-genotype, 2-site binary map with a handpicked rugged
  landscape. Used for unit tests where you can read the matrix entries by eye.
- ``binary_graph_64`` : a 2**6 = 64-genotype binary map with phenotypes drawn
  from a fixed-seed log-additive landscape. Used for property and golden tests
  where you want non-trivial topology but still fast to build.

Both fixtures return a ``gpgraph.GenotypePhenotypeGraph`` ready to feed into
``build_transition_matrix``.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap


@pytest.fixture
def tiny_gpm() -> GenotypePhenotypeMap:
    """4-genotype, 2-site binary map with a known fitness ordering."""
    return GenotypePhenotypeMap(
        wildtype="00",
        genotypes=["00", "01", "10", "11"],
        phenotypes=[1.0, 1.5, 1.2, 2.0],
    )


@pytest.fixture
def tiny_graph(tiny_gpm: GenotypePhenotypeMap) -> GenotypePhenotypeGraph:
    """4-genotype binary graph backing ``tiny_gpm``."""
    return GenotypePhenotypeGraph.from_gpm(tiny_gpm)


@pytest.fixture
def binary_gpm_64() -> GenotypePhenotypeMap:
    """2**6 = 64-genotype binary map with a deterministic log-additive landscape."""
    sites = 6
    rng = np.random.default_rng(42)
    site_effects = rng.normal(0.0, 0.5, size=sites)
    genotypes = ["".join(bits) for bits in product("01", repeat=sites)]
    phenotypes = []
    for g in genotypes:
        log_phi = sum(site_effects[i] for i, c in enumerate(g) if c == "1")
        phenotypes.append(float(np.exp(log_phi)))
    return GenotypePhenotypeMap(
        wildtype="0" * sites,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


@pytest.fixture
def binary_graph_64(binary_gpm_64: GenotypePhenotypeMap) -> GenotypePhenotypeGraph:
    """64-genotype binary graph backing ``binary_gpm_64``."""
    return GenotypePhenotypeGraph.from_gpm(binary_gpm_64)


@pytest.fixture
def rugged_gpm_8() -> GenotypePhenotypeMap:
    """8-genotype, 3-site binary map with TWO local fitness maxima.

    Constructed so that genotypes ``111`` and ``000`` are both local optima:
    every one-step neighbor of either has strictly lower fitness. Under SSWM
    the resulting chain has two absorbing states (one per peak), so the chain
    is reducible. Useful for exercising multi-peak / multi-absorbing-state
    code paths.
    """
    # Fitness values picked by hand so 000 and 111 dominate their neighbors.
    fits = {
        "000": 2.5,
        "001": 1.2,
        "010": 1.1,
        "100": 1.3,
        "011": 1.4,
        "101": 1.5,
        "110": 1.6,
        "111": 2.7,
    }
    genotypes = [
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    ]
    phenotypes = [fits[g] for g in genotypes]
    return GenotypePhenotypeMap(
        wildtype="000",
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


@pytest.fixture
def rugged_graph_8(rugged_gpm_8: GenotypePhenotypeMap) -> GenotypePhenotypeGraph:
    """8-genotype binary graph with two local maxima, backing ``rugged_gpm_8``."""
    return GenotypePhenotypeGraph.from_gpm(rugged_gpm_8)


@pytest.fixture
def fuji_5_sswm_msm():
    """L=5 SSWM Mount-Fuji MSM. Reproduces the bug case from the streamlit
    showcase: single absorbing peak at TTTTT, stationary distribution
    underflows to zero on the worst genotype AAAAA. Returns ``(gpm, msm)``.
    """
    from itertools import product

    import numpy as np
    from gpgraph import GenotypePhenotypeGraph
    from gpvolve import GenotypePhenotypeMSM

    alph = ("A", "T")
    length = 5
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
        stdeviations=[0.05] * len(genos),
    )
    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    msm = GenotypePhenotypeMSM.from_graph(graph, fitness_column="phenotypes", fixation="sswm")
    return gpm, msm
