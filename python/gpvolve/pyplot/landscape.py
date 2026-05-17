"""Fitness landscape plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gpgraph import GenotypePhenotypeGraph
    from matplotlib.axes import Axes


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required. Install with `uv pip install 'gpvolve-v2[plot]'`."
        ) from exc
    return plt


def draw_landscape_1d(
    graph: GenotypePhenotypeGraph,
    *,
    fitness_column: str,
    ax: Axes | None = None,
) -> Axes:
    """Scatter plot of fitness vs n_mutations."""
    plt = _require_matplotlib()
    if ax is None:
        _fig, ax = plt.subplots()
    fitness = np.asarray(graph.gpm.data[fitness_column].to_numpy(), dtype=np.float64)
    n_muts = np.asarray(graph.gpm.data["n_mutations"].to_numpy(), dtype=np.int64)
    ax.scatter(n_muts, fitness, alpha=0.6)
    ax.set_xlabel("n_mutations")
    ax.set_ylabel(fitness_column)
    ax.set_title("fitness landscape")
    return ax
