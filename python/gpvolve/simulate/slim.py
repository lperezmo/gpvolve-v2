"""SLiM forward simulation backend (optional, [sim] extra).

Imports ``pyslim`` lazily. If the dependency is missing, every entrypoint
raises :class:`ImportError` with an install hint. Implementation is intentionally
minimal in v2.0.0 -- the full SLiM script generator from the harmsm fork is on
the roadmap.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = (
    "SLiM integration requires the optional 'sim' extra. "
    "Install with `uv pip install 'gpvolve-v2[sim]'` (pulls in pyslim and tskit)."
)


def _require_pyslim() -> Any:
    try:
        import pyslim
    except ImportError as exc:  # pragma: no cover - only hit without [sim]
        raise ImportError(_INSTALL_HINT) from exc
    return pyslim


def slim_available() -> bool:
    """Return True if ``pyslim`` is importable."""
    try:
        _require_pyslim()
    except ImportError:
        return False
    return True
