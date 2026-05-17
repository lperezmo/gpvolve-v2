"""Implied-timescales plots (markovianity diagnostic)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

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


def draw_timescales(ts: NDArray[np.float64], *, ax: Axes | None = None) -> Axes:
    """Bar plot of timescales (slowest to fastest)."""
    plt = _require_matplotlib()
    if ax is None:
        _fig, ax = plt.subplots()
    finite = np.where(np.isnan(ts), 0.0, ts)
    ax.bar(np.arange(finite.size), finite)
    ax.set_xlabel("mode index")
    ax.set_ylabel("timescale")
    ax.set_yscale("log")
    ax.set_title("implied timescales")
    return ax
