"""tskit tree-sequence I/O for SLiM-produced data (optional, [sim] extra).

Imports ``tskit`` lazily. Same install hint as :mod:`gpvolve.simulate.slim`.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = (
    "tskit integration requires the optional 'sim' extra. "
    "Install with `uv pip install 'gpvolve-v2[sim]'`."
)


def _require_tskit() -> Any:
    try:
        import tskit
    except ImportError as exc:  # pragma: no cover - only hit without [sim]
        raise ImportError(_INSTALL_HINT) from exc
    return tskit


def tskit_available() -> bool:
    """Return True if ``tskit`` is importable."""
    try:
        _require_tskit()
    except ImportError:
        return False
    return True
