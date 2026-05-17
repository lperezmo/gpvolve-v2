"""Markov state model primitives: transition matrices, stationary, spectral analysis."""

from gpvolve.markov.msm import GenotypePhenotypeMSM
from gpvolve.markov.spectral import eigenvalues, mfpt, mixing_time, timescales
from gpvolve.markov.stationary import stationary_distribution
from gpvolve.markov.transition import build_transition_matrix
from gpvolve.markov.validation import (
    assert_nonneg,
    assert_row_stochastic,
    assert_strongly_connected,
    is_strongly_connected,
)

__all__ = [
    "GenotypePhenotypeMSM",
    "assert_nonneg",
    "assert_row_stochastic",
    "assert_strongly_connected",
    "build_transition_matrix",
    "eigenvalues",
    "is_strongly_connected",
    "mfpt",
    "mixing_time",
    "stationary_distribution",
    "timescales",
]
