"""Shared helpers for the gpvolve-v2 Streamlit showcase."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import numpy as np
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap

from gpvolve import GenotypePhenotypeMSM


def make_fuji_gpm(
    length: int, alphabet: Iterable[str] = ("A", "T"), seed: int = 0
) -> GenotypePhenotypeMap:
    """Build an additive-with-noise Mount-Fuji-style gpm for demos.

    Same generator as gpgraph-v2's showcase so the two apps tell consistent
    stories about the same landscape.
    """
    alph = list(alphabet)
    genotypes = ["".join(g) for g in product(alph, repeat=length)]
    wildtype = alph[0] * length
    rng = np.random.default_rng(seed)
    per_site_effect = rng.normal(loc=1.0, scale=0.5, size=(length, len(alph)))
    per_site_effect[:, 0] = 0.0

    phenotypes = []
    for g in genotypes:
        total = 0.0
        for i, letter in enumerate(g):
            total += float(per_site_effect[i, alph.index(letter)])
        phenotypes.append(max(total + 1.0, 0.05))

    return GenotypePhenotypeMap(
        wildtype=wildtype,
        genotypes=genotypes,
        phenotypes=phenotypes,
        stdeviations=[0.05] * len(genotypes),
    )


def build_msm(
    length: int,
    fixation: str,
    population_size: float,
    *,
    seed: int = 0,
) -> tuple[GenotypePhenotypeMap, GenotypePhenotypeGraph, GenotypePhenotypeMSM]:
    """One-shot helper that pages reuse. Returns (gpm, graph, msm)."""
    gpm = make_fuji_gpm(length=length, seed=seed)
    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    kwargs: dict[str, float] = {}
    if fixation in {"moran", "mccandlish", "mcclandish"}:
        kwargs["population_size"] = float(population_size)
    msm = GenotypePhenotypeMSM.from_graph(
        graph, fitness_column="phenotypes", fixation=fixation, **kwargs
    )
    return gpm, graph, msm
