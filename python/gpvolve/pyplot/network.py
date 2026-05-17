"""MSM network plots (lightweight wrappers around matplotlib).

These are not first-class visualizations. They give the user a one-line
matplotlib preview for sanity-checking a small MSM; for publication-quality
figures, lean on gpgraph-v2's plotting layer (which exposes the layout) and
overlay MSM data manually.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gpvolve.markov.msm import GenotypePhenotypeMSM

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without [plot]
        raise ImportError(
            "matplotlib is required. Install with `uv pip install 'gpvolve-v2[plot]'`."
        ) from exc
    return plt


def draw_transition_matrix(msm: GenotypePhenotypeMSM, *, ax: Axes | None = None) -> Axes:
    """Heatmap of the (dense) transition matrix. For small MSMs only."""
    plt = _require_matplotlib()
    if ax is None:
        _fig, ax = plt.subplots()
    dense = msm.transition_matrix.toarray()
    im = ax.imshow(dense, cmap="viridis", aspect="auto")
    ax.figure.colorbar(im, ax=ax, label="P(i -> j)")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title(f"transition matrix (n={msm.n_states})")
    return ax


def draw_stationary(msm: GenotypePhenotypeMSM, *, ax: Axes | None = None) -> Axes:
    """Bar plot of the stationary distribution."""
    plt = _require_matplotlib()
    if ax is None:
        _fig, ax = plt.subplots()
    idx = np.arange(msm.n_states)
    ax.bar(idx, msm.stationary)
    ax.set_xlabel("state")
    ax.set_ylabel("pi(i)")
    ax.set_title("stationary distribution")
    return ax
