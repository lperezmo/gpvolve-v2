"""Transition Path Theory explorer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gpvolve import (
    backward_committor,
    dominant_pathways,
    forward_committor,
    rate,
    reactive_flux,
)
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
q_minus = backward_committor(P, A=source_idx, B=target_idx)
flux = reactive_flux(P, A=source_idx, B=target_idx)
k_AB = rate(P, A=source_idx, B=target_idx)
pathways = dominant_pathways(flux, A=source_idx, B=target_idx, top_k=5)

col = st.columns(3)
col[0].metric("rate k_AB", f"{k_AB:.3e}")
col[1].metric("reactive edges", flux.nnz)
col[2].metric("dominant pathways", len(pathways))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_muts = np.asarray(gpm.data["n_mutations"].to_numpy(), dtype=int)
axes[0].scatter(n_muts, q_plus, c=q_plus, cmap="viridis", s=60)
axes[0].axhline(0, color="grey", linestyle=":")
axes[0].axhline(1, color="grey", linestyle=":")
axes[0].set_xlabel("n_mutations")
axes[0].set_ylabel("q+ (forward committor)")
axes[0].set_title(f"committor: {source_label} -> {target_label}")

dense_flux = flux.toarray()
im = axes[1].imshow(dense_flux, cmap="magma", aspect="auto")
axes[1].set_xlabel("j")
axes[1].set_ylabel("i")
axes[1].set_title(r"reactive flux $f_{ij}$")
fig.colorbar(im, ax=axes[1], label="flux")
st.pyplot(fig)

st.subheader("Top dominant pathways")
if not pathways:
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
