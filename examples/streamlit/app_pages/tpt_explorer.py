"""Transition Path Theory explorer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gpvolve import (
    absorption_rate,
    backward_committor,
    dominant_pathways,
    forward_committor,
    is_ergodic,
    rate,
    reactive_flux,
)
from gpvolve.exceptions import GpvolveError
from utils import build_msm

st.title("Transition Path Theory")

st.markdown(
    r"""
Pick a source genotype $A$ and a target genotype $B$. We solve the
absorbing-boundary system for the forward committor $q^{+}$, build the
reactive flux

$$
f_{ij} \;=\; \pi_i \, q^{-}_{i} \, P_{ij} \, q^{+}_{j}
$$

compute the $A$-to-$B$ rate, and decompose the flux into bottleneck
pathways.

The committor solve dispatches to the Rust BiCGSTAB kernel when
`n_free > 256`; for the small landscapes in this UI it stays in
scipy's spsolve, but the API is identical.
"""
)

col = st.columns(3)
length = col[0].slider("L", 2, 5, 4)
fixation = col[1].selectbox("Fixation", ["moran", "sswm", "mccandlish"])
pop_size = col[2].slider("Population size", 2, 1000, 100)

gpm, graph, msm = build_msm(length=length, fixation=fixation, population_size=pop_size)
genotypes = list(gpm.data["genotypes"])

col = st.columns(2)
source_label = col[0].selectbox("Source A", genotypes, index=0)
target_label = col[1].selectbox("Target B", genotypes, index=len(genotypes) - 1)
source_idx = genotypes.index(source_label)
target_idx = genotypes.index(target_label)

if source_idx == target_idx:
    st.error("Source and target must be different states.")
    st.stop()

P = msm.transition_matrix
q_plus = forward_committor(P, A=source_idx, B=target_idx)
k_reactive = rate(P, A=source_idx, B=target_idx)
k_absorption = absorption_rate(P, A=source_idx, B=target_idx)
chain_ergodic = is_ergodic(P)

flux = None
pathways: list = []
backward_error: str | None = None
try:
    backward_committor(P, A=source_idx, B=target_idx)
    flux = reactive_flux(P, A=source_idx, B=target_idx)
    pathways = dominant_pathways(flux, A=source_idx, B=target_idx, top_k=5)
except GpvolveError as exc:
    backward_error = str(exc)

col = st.columns(4)
col[0].metric("reactive rate k_AB", f"{k_reactive:.3e}")
col[1].metric("absorption rate 1/MFPT", f"{k_absorption:.3e}")
col[2].metric("reactive edges", flux.nnz if flux is not None else "n/a")
col[3].metric("dominant pathways", len(pathways) if flux is not None else "n/a")

if backward_error is not None:
    st.warning(
        "**Reactive flux and dominant pathways are unavailable for this chain.**\n\n"
        f"{backward_error}\n\n"
        "The forward committor $q^{+}$ is still shown below. For absorbing chains "
        "the MFPT-based **absorption rate** $1 / E[\\tau_B | X_0 = A]$ is the "
        "natural rate constant (Hanggi-Talkner-Borkovec 1990); the reactive rate "
        "above will read 0 because $\\pi_A = 0$ for the absorbing dynamics. To "
        "explore the full reactive-flux decomposition, try the `moran` or "
        "`mccandlish` fixation models, which produce ergodic chains at finite "
        "population size."
    )
elif not chain_ergodic:
    st.info(
        "The chain is non-ergodic but the standard TPT solve still succeeded "
        "(stationary mass is numerically positive everywhere). Reactive and "
        "absorption rates are both shown for comparison."
    )

n_muts = np.asarray(gpm.data["n_mutations"].to_numpy(), dtype=int)
if flux is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    q_ax, flux_ax = axes[0], axes[1]
else:
    fig, q_ax = plt.subplots(1, 1, figsize=(7, 5))
    flux_ax = None

q_ax.scatter(n_muts, q_plus, c=q_plus, cmap="viridis", s=60)
q_ax.axhline(0, color="grey", linestyle=":")
q_ax.axhline(1, color="grey", linestyle=":")
q_ax.set_xlabel("n_mutations")
q_ax.set_ylabel("q+ (forward committor)")
q_ax.set_title(f"committor: {source_label} -> {target_label}")

if flux is not None and flux_ax is not None:
    dense_flux = flux.toarray()
    im = flux_ax.imshow(dense_flux, cmap="magma", aspect="auto")
    flux_ax.set_xlabel("j")
    flux_ax.set_ylabel("i")
    flux_ax.set_title(r"reactive flux $f_{ij}$")
    fig.colorbar(im, ax=flux_ax, label="flux")
st.pyplot(fig)

st.subheader("Top dominant pathways")
if flux is None:
    st.info(
        "Pathway decomposition requires the reactive flux, which is not defined "
        "for this chain (see warning above)."
    )
elif not pathways:
    st.info("No dominant pathway found (flux is effectively zero).")
else:
    rows = []
    for p in pathways:
        genotype_path = " -> ".join(genotypes[i] for i in p.paths[0])
        rows.append(
            {
                "bottleneck flux": float(p.probabilities[0]),
                "path": genotype_path,
            }
        )
    st.dataframe(rows, width="stretch")
