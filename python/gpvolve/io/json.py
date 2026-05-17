"""JSON serialization for ``GenotypePhenotypeMSM``.

The on-disk payload (SCHEMA.md section 5) is

    {
        "schema_version": 1,
        "transition_matrix": {
            "indptr": [...], "indices": [...], "data": [...], "shape": [n, n]
        },
        "stationary": [...],
        "fixation_model": "moran",
        "fixation_params": {...},
        "gpm_meta": {"hash": "...", "wildtype": "...", "n_genotypes": N}
    }

The full ``gpm`` is **not** serialized here. Users save the gpm independently
via ``gpmap.to_json`` and re-attach it on load. This avoids re-serializing the
underlying DataFrame, which gpmap-v2 already handles in its own format.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from gpvolve.exceptions import SchemaError
from gpvolve.markov.msm import GenotypePhenotypeMSM
from gpvolve.markov.transition import build_transition_matrix

SCHEMA_VERSION = 1


def _gpm_meta(gpm: Any) -> dict[str, Any]:
    """Stable hash of the gpm.data table plus light metadata."""
    table_bytes = gpm.data.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(table_bytes).hexdigest()
    return {
        "hash": f"sha256:{digest}",
        "wildtype": getattr(gpm, "wildtype", None),
        "n_genotypes": len(gpm.data),
    }


def to_dict(msm: GenotypePhenotypeMSM) -> dict[str, Any]:
    """Convert an MSM to its canonical JSON-able payload."""
    csr = msm.transition_matrix.tocsr()
    fixation_name = (
        msm.fixation_model
        if isinstance(msm.fixation_model, str)
        else getattr(msm.fixation_model, "name", "custom")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "transition_matrix": {
            "indptr": csr.indptr.tolist(),
            "indices": csr.indices.tolist(),
            "data": csr.data.tolist(),
            "shape": list(csr.shape),
        },
        "stationary": msm.stationary.tolist(),
        "fixation_model": fixation_name,
        "fixation_params": dict(msm.fixation_params),
        "gpm_meta": _gpm_meta(msm.gpm),
    }


def to_json(msm: GenotypePhenotypeMSM, path: str | Path) -> None:
    """Write a canonical JSON payload for ``msm`` to ``path``."""
    Path(path).write_text(json.dumps(to_dict(msm), indent=2))


def from_dict(payload: dict[str, Any], *, graph: Any | None = None) -> GenotypePhenotypeMSM:
    """Reconstruct an MSM from a payload dict.

    Parameters
    ----------
    payload:
        Output of :func:`to_dict`.
    graph:
        Optional :class:`gpgraph.GenotypePhenotypeGraph`. If supplied, the
        rebuilt MSM uses ``graph`` and ``graph.gpm`` as its containers (and the
        ``gpm_meta`` hash is checked against the live gpm). If ``None``, the
        MSM is reconstructed without a live graph attached -- useful for
        downstream analyses that only need the matrix and stationary vector.
    """
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported schema_version: {payload.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    tm = payload["transition_matrix"]
    csr = sp.csr_matrix(
        (
            np.asarray(tm["data"], dtype=np.float64),
            np.asarray(tm["indices"], dtype=np.int64),
            np.asarray(tm["indptr"], dtype=np.int64),
        ),
        shape=tuple(tm["shape"]),
    )
    pi = np.asarray(payload["stationary"], dtype=np.float64)
    fixation_model = payload["fixation_model"]
    fixation_params = dict(payload.get("fixation_params") or {})

    if graph is not None:
        live_meta = _gpm_meta(graph.gpm)
        stored_hash = payload["gpm_meta"]["hash"]
        if live_meta["hash"] != stored_hash:
            raise SchemaError(
                f"gpm hash mismatch on load: stored {stored_hash}, live {live_meta['hash']}"
            )
        return GenotypePhenotypeMSM(
            gpm=graph.gpm,
            graph=graph,
            transition_matrix=csr,
            stationary=pi,
            fixation_model=fixation_model,
            fixation_params=fixation_params,
        )

    # No live graph: leave gpm/graph as None placeholders. Downstream code
    # that needs them will fail loudly.
    return GenotypePhenotypeMSM(
        gpm=None,
        graph=None,
        transition_matrix=csr,
        stationary=pi,
        fixation_model=fixation_model,
        fixation_params=fixation_params,
    )


def from_json(path: str | Path, *, graph: Any | None = None) -> GenotypePhenotypeMSM:
    """Read a canonical JSON payload at ``path`` into an MSM."""
    payload = json.loads(Path(path).read_text())
    return from_dict(payload, graph=graph)


__all__ = ["SCHEMA_VERSION", "from_dict", "from_json", "to_dict", "to_json"]


# Silence unused import noise when build_transition_matrix is re-exported.
_ = build_transition_matrix
