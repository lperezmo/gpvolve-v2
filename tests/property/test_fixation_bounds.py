"""Property tests: bounded fixation models must always return values in [0, 1]."""

from __future__ import annotations

import numpy as np
from gpvolve.fixation import mccandlish, moran, strong_selection_weak_mutation
from hypothesis import given, settings
from hypothesis import strategies as st

_fitness = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
_pop = st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(fi=_fitness, fj=_fitness)
@settings(max_examples=200)
def test_sswm_in_unit_interval(fi: float, fj: float) -> None:
    out = strong_selection_weak_mutation(np.array([fi]), np.array([fj]))
    assert 0.0 <= out[0] <= 1.0 + 1e-12


@given(fi=_fitness, fj=_fitness, n=_pop)
@settings(max_examples=200)
def test_moran_in_unit_interval(fi: float, fj: float, n: float) -> None:
    out = moran(np.array([fi]), np.array([fj]), population_size=n)
    assert 0.0 <= out[0] <= 1.0 + 1e-12


@given(fi=_fitness, fj=_fitness, n=_pop)
@settings(max_examples=200)
def test_mccandlish_in_unit_interval(fi: float, fj: float, n: float) -> None:
    out = mccandlish(np.array([fi]), np.array([fj]), population_size=n)
    assert 0.0 <= out[0] <= 1.0 + 1e-12


@given(fi=_fitness)
@settings(max_examples=100)
def test_sswm_monotone_in_target(fi: float) -> None:
    """For fixed source, SSWM is non-decreasing in target fitness."""
    targets = np.array([fi * 0.5, fi, fi * 1.1, fi * 2.0])
    out = strong_selection_weak_mutation(np.full_like(targets, fi), targets)
    assert np.all(np.diff(out) >= -1e-12)


@given(fi=_fitness, n=_pop)
@settings(max_examples=100)
def test_moran_monotone_in_target(fi: float, n: float) -> None:
    """For fixed source and population, Moran is non-decreasing in target fitness."""
    targets = np.array([fi * 0.5, fi * 0.9, fi, fi * 1.1, fi * 1.5])
    out = moran(np.full_like(targets, fi), targets, population_size=n)
    assert np.all(np.diff(out) >= -1e-9)
