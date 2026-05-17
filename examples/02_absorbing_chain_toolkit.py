"""End-to-end demonstration of the absorbing-chain toolkit.

Constructs an SSWM Markov chain on a small Mount-Fuji-style fitness landscape,
which produces a non-ergodic chain with a single absorbing state at the peak.
The standard TPT machinery (backward committor, reactive flux, reactive rate)
breaks down on such chains; the absorbing-chain toolkit fills the gap.

References:
- Kemeny, J.G. & Snell, J.L. (1976). *Finite Markov Chains*. Springer.
- Darroch, J.N. & Seneta, E. (1965). "On quasi-stationary distributions in
  absorbing discrete-time finite Markov chains." J. Appl. Prob. 2, 88-100.
- Hanggi, P., Talkner, P., Borkovec, M. (1990). "Reaction-rate theory: fifty
  years after Kramers." Rev. Mod. Phys. 62, 251-341.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap
from gpvolve import (
    GenotypePhenotypeMSM,
    absorbing_states,
    absorption_rate,
    backward_committor,
    conditional_mfpt,
    forward_committor,
    fundamental_matrix,
    is_ergodic,
    is_reversible,
    mfpt,
    quasi_stationary_distribution,
    rate,
)
from gpvolve.exceptions import GpvolveError


def make_landscape(length: int, seed: int = 0) -> GenotypePhenotypeMap:
    alph = ("A", "T")
    rng = np.random.default_rng(seed)
    per_site = rng.normal(loc=1.0, scale=0.5, size=(length, len(alph)))
    per_site[:, 0] = 0.0
    genos = ["".join(g) for g in product(alph, repeat=length)]
    phenos = []
    for g in genos:
        total = sum(float(per_site[i, alph.index(c)]) for i, c in enumerate(g))
        phenos.append(max(total + 1.0, 0.05))
    return GenotypePhenotypeMap(
        wildtype=alph[0] * length,
        genotypes=genos,
        phenotypes=phenos,
        stdeviations=[0.05] * len(genos),
    )


def main() -> None:
    gpm = make_landscape(length=5)
    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    msm = GenotypePhenotypeMSM.from_graph(graph, fitness_column="phenotypes", fixation="sswm")
    P = msm.transition_matrix
    genos = list(gpm.data["genotypes"])

    print(f"Landscape: L=5, {P.shape[0]} states, SSWM dynamics.")
    print(f"  is_ergodic:    {is_ergodic(P)}")
    print(f"  is_reversible: {is_reversible(P)}")

    abs_idx = absorbing_states(P)
    print(f"  absorbing:     {abs_idx.tolist()} -> {[genos[i] for i in abs_idx]}")

    A = genos.index("AAAAA")
    B = genos.index("TTTTT")

    print()
    print("=== Standard TPT: breaks down on absorbing chains ===")
    q_plus = forward_committor(P, A=A, B=B)
    print(f"  forward committor  q+[A]={q_plus[A]:.3f}, q+[B]={q_plus[B]:.3f}  (works)")
    try:
        backward_committor(P, A=A, B=B)
    except GpvolveError as exc:
        print(f"  backward_committor refused: {exc.args[0].splitlines()[0]}")
    k_reactive = rate(P, A=A, B=B)
    print(f"  reactive rate k_AB:           {k_reactive:.3e}  (collapses to 0; pi_A = 0)")

    print()
    print("=== Absorbing-chain analytics ===")
    k_abs = absorption_rate(P, A=A, B=B)
    print(f"  absorption rate 1/MFPT:       {k_abs:.3e}  (the meaningful rate)")

    m = mfpt(P, targets=abs_idx.tolist())
    print(f"  E[tau_peak | X_0=AAAAA]:      {m[A]:.2f} time-steps")

    cmfpt = conditional_mfpt(P, A=A, B=B)
    print(f"  conditional MFPT(A->B):       {cmfpt[0]:.2f}  (equals MFPT since B is the only sink)")

    N, _trans = fundamental_matrix(P)
    print(f"  fundamental matrix N:         shape {N.shape}, trace {np.trace(N):.2f}")
    print("    -> expected number of visits to each transient state, summed over time.")

    qsd, _, lam = quasi_stationary_distribution(P)
    print(f"  QSD Perron eigenvalue:        lambda_1 = {lam:.6f}")
    print(f"  metastable lifetime:          1/(1 - lambda_1) = {1.0 / (1.0 - lam):.2f}")
    qsd_peak = int(np.argmax(qsd))
    print(f"  QSD peaks at:                 {genos[qsd_peak]} (qsd = {qsd[qsd_peak]:.4f})")


if __name__ == "__main__":
    main()
