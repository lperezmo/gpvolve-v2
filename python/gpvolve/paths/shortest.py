"""Shortest evolutionary path via NetworkX, weighted by edge ``-log(prob)``.

Shortest-path edge weights are the negative log of the transition probability,
so the "shortest" path is the path with the highest product of transition
probabilities. We attach those weights on-the-fly from the MSM rather than
mutating the graph.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np

from gpvolve.markov.msm import GenotypePhenotypeMSM
from gpvolve.types import PathEnsemble

_LOG_FLOOR = 1e-300


def _build_weighted_view(msm: GenotypePhenotypeMSM) -> nx.DiGraph:
    """Build a DiGraph with edge weight = -log(P_ij) over off-diagonal entries.

    Iterates the underlying graph's edges rather than the sparse matrix so that
    entries which underflowed to zero in the transition matrix are still
    traversable with a very high (but finite) weight.
    """
    if msm.graph is None:
        raise ValueError(
            "shortest_paths requires an MSM with an attached graph; "
            "load the MSM with a live graph argument"
        )
    out = nx.DiGraph()
    out.add_nodes_from(range(msm.n_states))
    csr = msm.transition_matrix.tocsr()
    for u, v in msm.graph.edges():
        i = int(u)
        j = int(v)
        if i == j:
            continue
        p = float(csr[i, j])
        out.add_edge(i, j, weight=float(-np.log(max(p, _LOG_FLOOR))))
    return out


def shortest_paths(
    msm: GenotypePhenotypeMSM,
    source: int,
    targets: int | Iterable[int],
) -> PathEnsemble:
    """Single shortest highest-probability path from ``source`` to each target.

    Returns a :class:`PathEnsemble` with ``method="shortest"``. The
    ``probabilities`` entry for each path is the product of transition
    probabilities along that path.
    """
    from itertools import pairwise

    target_list: tuple[int, ...] = (
        (int(targets),) if isinstance(targets, int) else tuple(int(t) for t in targets)
    )

    g = _build_weighted_view(msm)
    paths: list[tuple[int, ...]] = []
    probs: list[float] = []
    for t in target_list:
        try:
            node_path = nx.shortest_path(g, source=int(source), target=t, weight="weight")
        except nx.NetworkXNoPath:
            continue
        product = 1.0
        for u, v in pairwise(node_path):
            product *= float(msm.transition_matrix[u, v])
        paths.append(tuple(int(x) for x in node_path))
        probs.append(product)

    return PathEnsemble(
        paths=tuple(paths),
        probabilities=np.asarray(probs, dtype=np.float64),
        source=int(source),
        targets=target_list,
        method="shortest",
        metadata={},
    )
