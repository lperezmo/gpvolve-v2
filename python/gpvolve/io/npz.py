"""NumPy ``.npz`` archive serialization of an MSM.

Stores the same data as :mod:`gpvolve.io.json` but in a compact binary format.
The non-array metadata (fixation model + params + gpm_meta) is JSON-encoded
into a single ``meta`` entry so the file stays human-readable with ``np.load``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from gpvolve.exceptions import SchemaError
from gpvolve.io.json import SCHEMA_VERSION, _gpm_meta
from gpvolve.markov.msm import GenotypePhenotypeMSM


def to_npz(msm: GenotypePhenotypeMSM, path: str | Path) -> None:
    """Save ``msm`` to a NumPy ``.npz`` archive."""
    csr = msm.transition_matrix.tocsr()
    fixation_name = (
        msm.fixation_model
        if isinstance(msm.fixation_model, str)
        else getattr(msm.fixation_model, "name", "custom")
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "fixation_model": fixation_name,
        "fixation_params": dict(msm.fixation_params),
        "gpm_meta": _gpm_meta(msm.gpm),
    }
    np.savez_compressed(
        path,
        indptr=csr.indptr,
        indices=csr.indices,
        data=csr.data,
        shape=np.asarray(csr.shape, dtype=np.int64),
        stationary=msm.stationary,
        meta=np.asarray(json.dumps(meta)),
    )


def from_npz(path: str | Path, *, graph: Any | None = None) -> GenotypePhenotypeMSM:
    """Load an MSM from an ``.npz`` archive written by :func:`to_npz`."""
    archive = np.load(path, allow_pickle=False)
    meta = json.loads(str(archive["meta"]))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported schema_version: {meta.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    csr = sp.csr_matrix(
        (archive["data"], archive["indices"], archive["indptr"]),
        shape=tuple(int(v) for v in archive["shape"]),
    )
    pi = np.asarray(archive["stationary"], dtype=np.float64)

    if graph is not None:
        live_meta = _gpm_meta(graph.gpm)
        stored_hash = meta["gpm_meta"]["hash"]
        if live_meta["hash"] != stored_hash:
            raise SchemaError(
                f"gpm hash mismatch on load: stored {stored_hash}, live {live_meta['hash']}"
            )
        return GenotypePhenotypeMSM(
            gpm=graph.gpm,
            graph=graph,
            transition_matrix=csr,
            stationary=pi,
            fixation_model=meta["fixation_model"],
            fixation_params=meta["fixation_params"],
        )

    return GenotypePhenotypeMSM(
        gpm=None,
        graph=None,
        transition_matrix=csr,
        stationary=pi,
        fixation_model=meta["fixation_model"],
        fixation_params=meta["fixation_params"],
    )


__all__ = ["from_npz", "to_npz"]
