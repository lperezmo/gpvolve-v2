"""Cross-check: Rust BiCGSTAB committor agrees with scipy spsolve within tol.

The Rust solver only activates above ``_BICGSTAB_SIZE_THRESHOLD``. Below that
both forward_committor calls take the spsolve path so this test compares the
two solvers head-to-head by forcing each path with a monkeypatched threshold.
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


@pytest.mark.skipif(not tpt._RUST_BICGSTAB_AVAILABLE, reason="gpvolve._rust not built")
def test_bicgstab_matches_spsolve(monkeypatch) -> None:
    """On a 2^8 binary map, Rust BiCGSTAB and scipy spsolve return the same
    committor to within ``5 * tol`` of the relative residual.
    """
    graph = _make_graph(8)
    T = build_transition_matrix(
        graph, fitness_column="phenotypes", fixation="moran", population_size=100
    )
    target = int(np.argmax(graph.gpm.data["phenotypes"].to_numpy()))

    # Force spsolve path.
    monkeypatch.setattr(tpt, "_RUST_BICGSTAB_AVAILABLE", False)
    q_lu = forward_committor(T, A=0, B=target)

    # Force Rust path by dropping the threshold below the live n_free.
    monkeypatch.setattr(tpt, "_RUST_BICGSTAB_AVAILABLE", True)
    monkeypatch.setattr(tpt, "_BICGSTAB_SIZE_THRESHOLD", 0)
    q_rust = forward_committor(T, A=0, B=target)

    np.testing.assert_allclose(q_lu, q_rust, atol=5e-9)
    # Both must satisfy boundary conditions.
    assert q_rust[0] == 0.0
    assert q_rust[target] == 1.0
