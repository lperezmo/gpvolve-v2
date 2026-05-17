"""Dominant pathways from a reactive-flux matrix.

Decomposes the net reactive flux into pathways from source set A to target
set B by repeatedly extracting the highest-bottleneck path (Dijkstra on
``-log(flux)``), then subtracting its bottleneck capacity from the remaining
flux. The resulting list of paths is sorted by min-bottleneck flux descending.
This is the v1 ``dominant_pathways`` operation, made non-msmtools.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import scipy.sparse as sp

from gpvolve.exceptions import GpvolveError
from gpvolve.types import PathEnsemble

_LOG_FLOOR = 1e-300


def _to_digraph(flux: sp.spmatrix) -> nx.DiGraph:
    coo = flux.tocoo()
    g = nx.DiGraph()
    g.add_nodes_from(range(flux.shape[0]))
    for i, j, v in zip(coo.row, coo.col, coo.data, strict=True):
        if v <= 0:
            continue
        g.add_edge(int(i), int(j), flux=float(v), weight=float(-np.log(max(v, _LOG_FLOOR))))
    return g


def dominant_pathways(
    flux: sp.spmatrix,
    A: int | Iterable[int],
    B: int | Iterable[int],
    *,
    top_k: int = 10,
) -> list[PathEnsemble]:
    """Top-``k`` dominant pathways through the reactive-flux matrix.

    Each path is returned as a ``PathEnsemble`` of length 1 whose
    ``probabilities`` array holds the path's bottleneck flux (the smallest
    flux on any of its edges).

    Algorithm:

    1. Treat the flux matrix as a directed graph with edge weight
       ``-log(flux)``.
    2. Shortest path from any node in ``A`` to any node in ``B`` is the
       maximum-bottleneck path.
    3. Subtract the bottleneck flux from every edge on that path, drop edges
       that drop to zero, and repeat.
    """
    A_set = {int(A)} if isinstance(A, int) else {int(v) for v in A}
    B_set = {int(B)} if isinstance(B, int) else {int(v) for v in B}
    if not A_set or not B_set:
        raise GpvolveError("A and B must be non-empty")

    from itertools import pairwise

    g = _to_digraph(flux)

    # Add a super-source connected to A with infinite-flux edges, and a
    # super-target connected from B with infinite-flux edges. We use sentinels
    # safely outside the node index range.
    super_source = -1
    super_target = -2
    g.add_node(super_source)
    g.add_node(super_target)
    for a in A_set:
        g.add_edge(super_source, a, flux=float("inf"), weight=0.0)
    for b in B_set:
        g.add_edge(b, super_target, flux=float("inf"), weight=0.0)

    paths: list[PathEnsemble] = []
    targets = tuple(sorted(B_set))

    for _ in range(top_k):
        try:
            node_path = nx.shortest_path(
                g, source=super_source, target=super_target, weight="weight"
            )
        except nx.NetworkXNoPath:
            break
        real_path = tuple(int(v) for v in node_path if v not in {super_source, super_target})
        if len(real_path) < 2:
            break

        # Bottleneck flux along the real edges.
        bottleneck = float("inf")
        for u, v in pairwise(real_path):
            bottleneck = min(bottleneck, g[u][v]["flux"])
        if bottleneck <= 0 or bottleneck == float("inf"):
            break

        actual_source = int(real_path[0])
        paths.append(
            PathEnsemble(
                paths=(real_path,),
                probabilities=np.asarray([bottleneck], dtype=np.float64),
                source=actual_source,
                targets=targets,
                method="tpt",
                metadata={"bottleneck_flux": bottleneck},
            )
        )

        # Subtract bottleneck flux from each edge on the path; drop saturated edges.
        edges_to_drop: list[tuple[int, int]] = []
        for u, v in pairwise(real_path):
            g[u][v]["flux"] -= bottleneck
            if g[u][v]["flux"] <= 1e-15:
                edges_to_drop.append((u, v))
            else:
                g[u][v]["weight"] = float(-np.log(max(g[u][v]["flux"], _LOG_FLOOR)))
        for e in edges_to_drop:
            g.remove_edge(*e)

    paths.sort(key=lambda p: float(p.probabilities[0]), reverse=True)
    return paths
