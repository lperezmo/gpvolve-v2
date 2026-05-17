"""Row-stochastic transition matrix assembly over a GenotypePhenotypeGraph.

The matrix is the discrete-time Markov chain on the genotype-phenotype graph.
For neighbors ``i, j`` the off-diagonal entry is

    T_ij = fix(f_i, f_j) / k_max

where ``k_max`` is the maximum out-degree across the graph and ``fix`` is the
fixation model evaluated on the source and target fitness values. The diagonal
is computed last as ``T_ii = 1 - sum_j T_ij`` so each row sums to ``1.0`` to
within ``1e-12``. See ``SCHEMA.md`` section 2 for the locked invariants.

This module fixes the two known v1 bugs:

1. **Row/column indexing.** Rows and columns are int-keyed by ``gpm.data.index``,
   identical to the contract that ``gpgraph-v2`` locks in its own SCHEMA. The
   v1 implementation mixed index conventions and could silently misalign with
   the underlying ``GenotypePhenotypeMap``.
2. **Self-loop computation.** The diagonal is computed strictly as the residual
   ``1 - sum(off-diagonals)``. v1 evaluated the fixation kernel at ``f_i == f_j``
   and added the result to the diagonal, which double-counted the absorption
   mass and broke row-stochasticity by up to a percent on rugged maps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from gpvolve.exceptions import ModelError, NonStochasticError
from gpvolve.fixation.protocol import get_fixation_model, validate_params
from gpvolve.types import FixationModel

if TYPE_CHECKING:
    from gpgraph import GenotypePhenotypeGraph

SelfLoopMode = Literal["absorb"]

_ROW_TOL = 1e-12
_BOUND_TOL = 1e-12


def _resolve_fixation(fixation: str | FixationModel | Callable[..., Any]) -> FixationModel:
    """Return a registered or user-supplied FixationModel."""
    if isinstance(fixation, str):
        return get_fixation_model(fixation)
    if isinstance(fixation, FixationModel):
        return fixation
    raise ModelError(
        "fixation must be a registered model name (str) or a FixationModel "
        f"satisfying the Protocol; got {type(fixation).__name__}"
    )


def _extract_fitness(graph: GenotypePhenotypeGraph, fitness_column: str) -> NDArray[np.float64]:
    """Pull a column from gpm.data, ordered by the graph node index."""
    gpm = graph.gpm
    try:
        column = gpm.data[fitness_column]
    except KeyError as exc:
        raise ModelError(f"fitness_column {fitness_column!r} not in gpm.data") from exc
    return np.asarray(column.to_numpy(), dtype=np.float64)


def build_transition_matrix(
    graph: GenotypePhenotypeGraph,
    *,
    fitness_column: str,
    fixation: str | FixationModel,
    self_loops: SelfLoopMode = "absorb",
    **params: Any,
) -> sp.csr_matrix:
    """Build a row-stochastic transition matrix from a graph and a fixation model.

    Parameters
    ----------
    graph:
        ``gpgraph.GenotypePhenotypeGraph`` instance. Nodes must be int-keyed and
        match the rows of ``graph.gpm.data``.
    fitness_column:
        Column in ``graph.gpm.data`` to read as the per-genotype fitness.
    fixation:
        Either a registered fixation model name (e.g. ``"moran"``) or a callable
        satisfying :class:`gpvolve.types.FixationModel`.
    self_loops:
        Currently only ``"absorb"`` is supported. The diagonal is computed last
        as ``T_ii = 1 - sum_j T_ij``.
    **params:
        Forwarded to the fixation model on every call (e.g. ``population_size``
        for Moran or McCandlish).

    Returns
    -------
    ``scipy.sparse.csr_matrix`` of shape ``(n, n)``, dtype ``float64``. Rows sum
    to ``1.0`` to within ``1e-12``. All entries lie in ``[0, 1]``. The sparsity
    pattern is the graph's edges plus the diagonal.
    """
    if self_loops != "absorb":
        raise ModelError(f"unsupported self_loops mode {self_loops!r}; only 'absorb' is supported")

    model = _resolve_fixation(fixation)
    validate_params(model, dict(params))
    if not model.bounded_unit_interval:
        raise NonStochasticError(
            f"fixation model {model.name!r} is not bounded in [0, 1]; "
            "cannot build a row-stochastic transition matrix"
        )

    fitness = _extract_fitness(graph, fitness_column)
    n = len(fitness)
    if graph.number_of_nodes() != n:
        raise ModelError(
            f"graph has {graph.number_of_nodes()} nodes but gpm.data has {n} rows; "
            "rebuild the graph from the gpm"
        )

    if graph.number_of_edges() == 0:
        diagonal = np.ones(n, dtype=np.float64)
        return sp.diags(diagonal, format="csr")

    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    src = edges[:, 0]
    dst = edges[:, 1]
    fi = fitness[src]
    fj = fitness[dst]

    probs = np.asarray(model(fi, fj, **params), dtype=np.float64)
    if probs.shape != fi.shape:
        probs = np.broadcast_to(probs, fi.shape).astype(np.float64, copy=True)

    # Clip tiny floating-point overshoots from kernels that round up at boundary.
    if np.any((probs < -_BOUND_TOL) | (probs > 1.0 + _BOUND_TOL)):
        bad = probs[(probs < -_BOUND_TOL) | (probs > 1.0 + _BOUND_TOL)]
        raise NonStochasticError(
            f"fixation model {model.name!r} returned values outside [0, 1] "
            f"(min={bad.min():g}, max={bad.max():g})"
        )
    probs = np.clip(probs, 0.0, 1.0)

    # Normalize so the matrix stays row-stochastic regardless of out-degree.
    # Mutation rate is 1/k_max per neighbor per step, the standard discrete-time
    # convention for fixation-walk Markov chains (Sailer & Harms 2017).
    out_degree = np.bincount(src, minlength=n)
    k_max = int(out_degree.max())
    if k_max == 0:
        diagonal = np.ones(n, dtype=np.float64)
        return sp.diags(diagonal, format="csr")
    off_diag = probs / k_max

    row_sum = np.bincount(src, weights=off_diag, minlength=n)
    if np.any(row_sum > 1.0 + _ROW_TOL):
        worst = row_sum.max()
        raise NonStochasticError(
            f"off-diagonal mass per row exceeds 1.0 (max={worst:g}); "
            "this should never happen for a bounded fixation model"
        )
    diag = 1.0 - row_sum
    # Floating-point cleanup at the boundary.
    diag = np.clip(diag, 0.0, 1.0)

    rows = np.concatenate([src, np.arange(n, dtype=np.int64)])
    cols = np.concatenate([dst, np.arange(n, dtype=np.int64)])
    vals = np.concatenate([off_diag, diag])

    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    matrix.eliminate_zeros()

    final_row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.allclose(final_row_sum, 1.0, atol=_ROW_TOL):
        worst = np.abs(final_row_sum - 1.0).max()
        raise NonStochasticError(f"row sums deviate from 1.0 by up to {worst:g} (tol={_ROW_TOL:g})")

    return matrix
