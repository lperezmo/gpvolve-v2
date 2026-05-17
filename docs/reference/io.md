# `gpvolve.io`

Serialization round-trips for `GenotypePhenotypeMSM`. See [SCHEMA](https://github.com/lperezmo/gpvolve-v2/blob/main/SCHEMA.md)
section 5 for the canonical payload.

## JSON

- `to_dict(msm) -> dict`
- `to_json(msm, path) -> None`
- `from_dict(payload, *, graph=None) -> GenotypePhenotypeMSM`
- `from_json(path, *, graph=None) -> GenotypePhenotypeMSM`

If `graph` is supplied to a `from_*` call, the SHA-256 hash of
`graph.gpm.data` is checked against the payload's `gpm_meta.hash`. A
mismatch raises `SchemaError`.

## NumPy archive

- `to_npz(msm, path) -> None`
- `from_npz(path, *, graph=None) -> GenotypePhenotypeMSM`

## Pickle

- `to_pickle(msm, path) -> None`
- `from_pickle(path) -> GenotypePhenotypeMSM`

Pickle is not a stable format across Python versions or library refactors;
prefer JSON or NPZ for archival data.
