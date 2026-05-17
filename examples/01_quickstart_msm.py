"""End-to-end smoke test: build a 2^8 binary MSM and exercise every public surface.

Runtime target: < 30 s on a laptop. Used as the Phase 11 release-blocker check.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap

from gpvolve import (
    ConvergenceCheck,
    GenotypePhenotypeMSM,
    accessible_peaks,
    coarse_grain,
    dominant_pathways,
    find_peaks,
    forward_committor,
    pcca_plus,
    rate,
    reactive_flux,
    sample_paths,
    shortest_paths,
    timescales,
)


def main() -> None:
    sites = 8
    rng = np.random.default_rng(0)
    site_effects = rng.normal(0.0, 0.4, size=sites)
    genotypes = ["".join(bits) for bits in product("01", repeat=sites)]
    phenotypes = [
        float(np.exp(sum(site_effects[i] for i, c in enumerate(g) if c == "1")))
        for g in genotypes
    ]
    gpm = GenotypePhenotypeMap(
        wildtype="0" * sites, genotypes=genotypes, phenotypes=phenotypes
    )

    t0 = time.perf_counter()

    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    msm = GenotypePhenotypeMSM.from_graph(
        graph, fitness_column="phenotypes", fixation="moran", population_size=100
    )
    print(
        f"built MSM: n_states={msm.n_states}, nnz={msm.transition_matrix.nnz}, "
        f"stationary_sum={msm.stationary.sum():.6f}"
    )

    ts = timescales(msm.transition_matrix, k=5)
    print(f"top timescales: {ts}")

    target = int(np.argmax(phenotypes))
    print(f"global peak index: {target}, phenotype {phenotypes[target]:.4f}")

    q = forward_committor(msm.transition_matrix, A=0, B=target)
    print(f"forward committor range: [{q.min():.4f}, {q.max():.4f}]")

    flux = reactive_flux(msm.transition_matrix, A=0, B=target)
    k_AB = rate(msm.transition_matrix, A=0, B=target)
    print(f"reactive flux nnz={flux.nnz}, rate(A,B)={k_AB:.4e}")

    sp_ens = shortest_paths(msm, source=0, targets=target)
    print(
        f"shortest path: {sp_ens.paths[0]}, "
        f"probability={float(sp_ens.probabilities[0]):.4e}"
    )

    dom = dominant_pathways(flux, A=0, B=target, top_k=3)
    print(f"top {len(dom)} dominant pathways by bottleneck flux:")
    for p in dom:
        print(f"  {p.paths[0]}: bottleneck={float(p.probabilities[0]):.4e}")

    sample_ens = sample_paths(
        msm,
        source=0,
        targets=target,
        convergence=ConvergenceCheck(
            ess_min=50.0, chunk_size=500, max_walkers=5000, n_chains=4
        ),
        seed=0,
    )
    cv = sample_ens.metadata["convergence"]
    print(
        f"sampled {cv['n_walkers']} walkers in {cv['n_chunks']} chunks: "
        f"ess={cv['ess']}, rhat={cv['rhat']}, converged={cv['converged']}"
    )

    chi = pcca_plus(msm.transition_matrix, n_clusters=3)
    P_coarse = coarse_grain(msm.transition_matrix, chi)
    print(f"PCCA+: chi rows sum to {chi.sum(axis=1).mean():.6f}, P_coarse shape {P_coarse.shape}")

    peaks = find_peaks(graph, fitness_column="phenotypes")
    accessible = accessible_peaks(graph, source=0, fitness_column="phenotypes")
    print(f"found {len(peaks)} fitness peaks; {len(accessible)} accessible from wildtype")

    print(f"\ntotal runtime: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
