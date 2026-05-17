# Build an MSM from a GP map

Step-by-step. Assume you have a `gpmap.GenotypePhenotypeMap` already.

## 1. Build the graph

```python
from gpgraph import GenotypePhenotypeGraph

graph = GenotypePhenotypeGraph.from_gpm(gpm)
```

The graph is a NetworkX DiGraph: every directed edge `(i, j)` exists iff
the genotypes for nodes `i` and `j` are Hamming-1 neighbors (or, more
generally, satisfy whatever `neighbor_function` you passed). See the
`gpgraph-v2` docs for the alternatives.

## 2. Build the transition matrix

The free-function entrypoint:

```python
from gpvolve import build_transition_matrix

P = build_transition_matrix(
    graph,
    fitness_column="phenotypes",
    fixation="moran",
    population_size=1000,
)
```

`P` is a `scipy.sparse.csr_matrix`. Rows sum to 1.0 within 1e-12; every
entry is in `[0, 1]`; the sparsity pattern equals `graph.edges` plus the
diagonal. Diagonal entries are computed last as `1 - sum_j P_ij` (this is
the v1 bug 2 fix: never evaluate the fixation kernel at `f_i == f_j`).

## 3. Wrap it in an MSM

```python
from gpvolve import GenotypePhenotypeMSM

msm = GenotypePhenotypeMSM.from_graph(
    graph,
    fitness_column="phenotypes",
    fixation="moran",
    population_size=1000,
)
```

The `MSM` dataclass holds `(gpm, graph, transition_matrix, stationary,
fixation_model, fixation_params)`. The locked schema is in
[`SCHEMA.md`](https://github.com/lperezmo/gpvolve-v2/blob/main/SCHEMA.md).

## Picking a fixation model

- **`sswm`** is parameter-free and assumes the strong-selection
  weak-mutation regime.
- **`moran`** and **`mccandlish`** take `population_size` and reduce to
  SSWM in the large-N limit. McCandlish is the diffusion approximation.
- **`bloom_dms`** takes a `pi_table` argument: an `(L, alphabet)` table
  of empirical site preferences from a deep mutational scan.

The `weak_mutation` kernel is not bounded in `[0, 1]` and cannot be used
to build a row-stochastic matrix. Use `simulate.wright_fisher` or
`simulate.gillespie` for the weak-mutation regime instead.

## Saving and loading

```python
from gpvolve.io import to_json, from_json

to_json(msm, "msm.json")
loaded = from_json("msm.json", graph=graph)
```

`graph` is required on load if you want the live `gpm`/`graph` references
on the resulting MSM. The serialized JSON stores a SHA-256 of `gpm.data`
to catch mismatches; loading against a different gpm raises `SchemaError`.
