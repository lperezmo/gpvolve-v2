"""Population-dynamics simulation backends (Wright-Fisher, Gillespie, optional SLiM)."""

from gpvolve.simulate.gillespie import gillespie_walk
from gpvolve.simulate.slim import slim_available
from gpvolve.simulate.tskit_io import tskit_available
from gpvolve.simulate.wright_fisher import wright_fisher

__all__ = [
    "gillespie_walk",
    "slim_available",
    "tskit_available",
    "wright_fisher",
]
