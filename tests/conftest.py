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
