"""Pickle round-trip for an MSM.

Used mainly for golden-test fixtures and quick development iterations.
Pickle is not a stable format across Python versions or library refactors --
prefer :mod:`gpvolve.io.json` or :mod:`gpvolve.io.npz` for archival data.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from gpvolve.markov.msm import GenotypePhenotypeMSM


def to_pickle(msm: GenotypePhenotypeMSM, path: str | Path) -> None:
    """Pickle an MSM to ``path``."""
    Path(path).write_bytes(pickle.dumps(msm, protocol=pickle.HIGHEST_PROTOCOL))


def from_pickle(path: str | Path) -> GenotypePhenotypeMSM:
    """Unpickle an MSM from ``path``."""
    loaded = pickle.loads(Path(path).read_bytes())
    if not isinstance(loaded, GenotypePhenotypeMSM):
        raise TypeError(f"pickle at {path!s} did not contain a GenotypePhenotypeMSM")
    return loaded


__all__ = ["from_pickle", "to_pickle"]
