"""Exception hierarchy for gpvolve."""

from __future__ import annotations


class GpvolveError(Exception):
    """Base class for all gpvolve-specific errors."""


class ModelError(GpvolveError):
    """Raised when a fixation model is misconfigured (missing params, invalid inputs)."""


class ConvergenceError(GpvolveError):
    """Raised when an iterative computation (path sampler, eigensolver) fails to converge."""


class NonStochasticError(GpvolveError):
    """Raised when a matrix expected to be row-stochastic violates the invariant."""


class SchemaError(GpvolveError):
    """Raised when a serialized artifact violates the locked SCHEMA.md contract."""
