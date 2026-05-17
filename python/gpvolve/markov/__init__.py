"""Markov state model primitives: transition matrices, stationary, spectral analysis."""

from gpvolve.markov.absorbing import (
    conditional_mfpt,
    fundamental_matrix,
    quasi_stationary_distribution,
)
from gpvolve.markov.msm import GenotypePhenotypeMSM
from gpvolve.markov.spectral import eigenvalues, mfpt, mixing_time, timescales
from gpvolve.markov.stationary import stationary_distribution
from gpvolve.markov.transition import build_transition_matrix
from gpvolve.markov.validation import (
    absorbing_states,
    assert_nonneg,
    assert_row_stochastic,
    assert_strongly_connected,
    is_ergodic,
    is_irreducible,
    is_reversible,
    is_strongly_connected,
    transient_states,
)

__all__ = [
    "GenotypePhenotypeMSM",
    "absorbing_states",
    "assert_nonneg",
    "assert_row_stochastic",
    "assert_strongly_connected",
    "build_transition_matrix",
    "conditional_mfpt",
    "eigenvalues",
    "fundamental_matrix",
    "is_ergodic",
    "is_irreducible",
    "is_reversible",
    "is_strongly_connected",
    "mfpt",
    "mixing_time",
    "quasi_stationary_distribution",
    "stationary_distribution",
    "timescales",
    "transient_states",
]
