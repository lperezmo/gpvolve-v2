# gpvolve-v2 schema (locked contract)

This file documents the locked-in shapes and invariants that downstream code may rely on.
Changing any of these is a breaking change and triggers a major version bump.

Schema version: **1**.

## 1. `GenotypePhenotypeMSM` public surface

```python
@dataclass
class GenotypePhenotypeMSM:
    gpm: GenotypePhenotypeMap                # from gpmap-v2
    graph: GenotypePhenotypeGraph            # from gpgraph-v2
    transition_matrix: scipy.sparse.csr_matrix  # shape (n, n), dtype float64
    stationary: numpy.ndarray                # shape (n,), dtype float64, sums to 1
    fixation_model: str | FixationModel
    fixation_params: Mapping[str, Any]
```

Classmethod: `from_graph(graph, *, fitness_column, fixation, **params)`.

**Index alignment.** Row/column `i` of `transition_matrix` corresponds to `graph.nodes[i]`,
which corresponds to `gpm.data.iloc[i]`. This is the same int-keyed contract that
`gpgraph-v2` locks in its SCHEMA.md.

## 2. Transition matrix invariants

- Rows sum to `1.0 +/- 1e-12`.
- All entries in `[0, 1]`.
- Sparsity pattern equals `graph.edges` plus the diagonal.
- Diagonal entries are computed last as `1 - sum(off-diagonals)`. Direct evaluation of
  fixation probabilities for `i == j` is forbidden.

## 3. `PathEnsemble`

```python
@dataclass(frozen=True)
class PathEnsemble:
    paths: tuple[tuple[int, ...], ...]
    probabilities: numpy.ndarray   # shape (n_paths,), sums to <= 1
    source: int
    targets: tuple[int, ...]
    method: Literal["shortest", "greedy", "stochastic", "tpt"]
    metadata: Mapping[str, Any]
```

## 4. TPT outputs

- `forward_committor(P, A, B) -> NDArray[float64]` shape `(n,)`, range `[0, 1]`,
  `q[A] = 0`, `q[B] = 1`.
- `reactive_flux(P, pi, q_plus, q_minus) -> scipy.sparse.csr_matrix` shape `(n, n)`,
  nonnegative.
- `dominant_pathways(flux, A, B, top_k) -> list[PathEnsemble]` sorted by min bottleneck
  flux descending.

## 5. Serialization round-trip

`to_json(path)` writes:

```json
{
  "schema_version": 1,
  "transition_matrix": {"indptr": [...], "indices": [...], "data": [...], "shape": [n, n]},
  "stationary": [...],
  "fixation_model": "moran",
  "fixation_params": {...},
  "gpm_meta": {"hash": "...", "wildtype": "...", "n_genotypes": N}
}
```

The full `gpm` object is referenced by hash, not re-serialized. Users save it via
`gpmap.to_json` separately.

## 6. Convergence stats payload

For `PathEnsemble.metadata["convergence"]` when `method == "stochastic"`:

```python
{
    "ess": dict[int, float],        # per endpoint
    "rhat": dict[int, float],       # per endpoint
    "n_walkers": int,
    "n_chunks": int,
    "converged": bool,
}
```
