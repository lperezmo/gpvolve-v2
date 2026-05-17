"""Plots of reactive flux matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import scipy.sparse as sp

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required. Install with `uv pip install 'gpvolve-v2[plot]'`."
        ) from exc
    return plt


def draw_flux_heatmap(flux: sp.spmatrix, *, ax: Axes | None = None) -> Axes:
    """Heatmap of the (dense) reactive-flux matrix."""
    plt = _require_matplotlib()
    if ax is None:
        _fig, ax = plt.subplots()
    dense = flux.toarray()
    im = ax.imshow(dense, cmap="magma", aspect="auto")
    ax.figure.colorbar(im, ax=ax, label="flux")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("reactive flux")
    return ax
