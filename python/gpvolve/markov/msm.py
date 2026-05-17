"""``GenotypePhenotypeMSM``: the dataclass container for an MSM on a GP graph.

See ``SCHEMA.md`` section 1 for the locked surface. Construction goes through
``GenotypePhenotypeMSM.from_graph``, which composes ``build_transition_matrix``
and ``stationary_distribution`` so the container holds a coherent snapshot of
the chain. No lazy/cached state, no v1-style mutation surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from gpvolve.markov.stationary import stationary_distribution
from gpvolve.markov.transition import build_transition_matrix
from gpvolve.types import FixationModel

if TYPE_CHECKING:
    from gpgraph import GenotypePhenotypeGraph
    from gpmap import GenotypePhenotypeMap


@dataclass
class GenotypePhenotypeMSM:
    """Markov state model over a genotype-phenotype graph.

    Attributes follow ``SCHEMA.md`` section 1. ``transition_matrix`` is
    row-stochastic to ``1e-12``; ``stationary`` sums to ``1.0``. Use
    :meth:`from_graph` to construct.
    """

    gpm: GenotypePhenotypeMap | None
    graph: GenotypePhenotypeGraph | None
    transition_matrix: sp.csr_matrix
    stationary: NDArray[np.float64]
    fixation_model: str | FixationModel
    fixation_params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_graph(
        cls,
        graph: GenotypePhenotypeGraph,
        *,
        fitness_column: str,
        fixation: str | FixationModel,
        **params: Any,
    ) -> GenotypePhenotypeMSM:
        """Build an MSM from a ``GenotypePhenotypeGraph`` and a fixation model."""
        matrix = build_transition_matrix(
            graph,
            fitness_column=fitness_column,
            fixation=fixation,
            **params,
        )
        pi = stationary_distribution(matrix)
        return cls(
            gpm=graph.gpm,
            graph=graph,
            transition_matrix=matrix,
            stationary=pi,
            fixation_model=fixation,
            fixation_params=dict(params),
        )

    @property
    def n_states(self) -> int:
        """Number of states in the chain (= number of genotypes in the gpm)."""
        return int(self.transition_matrix.shape[0])

    def __repr__(self) -> str:
        model_name = (
            self.fixation_model
            if isinstance(self.fixation_model, str)
            else getattr(self.fixation_model, "name", "custom")
        )
        return (
            f"<GenotypePhenotypeMSM n_states={self.n_states} "
            f"fixation={model_name!r} params={dict(self.fixation_params)!r}>"
        )
