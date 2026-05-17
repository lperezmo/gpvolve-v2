"""PCCA+ metastable clustering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gpvolve import coarse_grain, metastable_sets, pcca_plus
from utils import build_msm

st.title("PCCA+ metastable clustering")

st.markdown(
    r"""
PCCA+ (Roeblitz and Weber 2013) decomposes the chain into $k$ metastable
basins from the top-$k$ right eigenvectors of $P$. gpvolve-v2 implements
the inner-simplex initialization + projected refinement from scratch,
no ``msmtools`` dependency.

Pick a landscape size and the number of clusters. We show:

- the row-stochastic membership matrix $\chi$
- the hard-argmax assignment laid out by `n_mutations`
- the Galerkin-coarse-grained transition matrix on the basins,
  $P_{\text{coarse}} = (\chi^{\top} D \chi)^{-1} \chi^{\top} D P \chi$
  with $D = \operatorname{diag}(\pi)$
"""
)

col = st.columns(3)
length = col[0].slider("L", 2, 6, 4)
fixation = col[1].selectbox("Fixation", ["moran", "sswm", "mccandlish"])
pop_size = col[2].slider("Population size", 2, 1000, 100)

n_clusters = st.slider("n_clusters", 2, 6, 3)

gpm, graph, msm = build_msm(length=length, fixation=fixation, population_size=pop_size)

chi = pcca_plus(msm.transition_matrix, n_clusters=n_clusters)
sets = metastable_sets(chi)
labels = np.argmax(chi, axis=1)
P_coarse = coarse_grain(msm.transition_matrix, chi)

cols = st.columns(2)
cols[0].metric("States", msm.n_states)
cols[1].metric("Basin sizes", ", ".join(str(len(s)) for s in sets))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im0 = axes[0].imshow(chi.T, cmap="viridis", aspect="auto")
axes[0].set_xlabel("state index")
axes[0].set_ylabel("cluster")
axes[0].set_title("chi (memberships)")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(P_coarse, cmap="cividis", aspect="auto")
axes[1].set_xlabel("cluster j")
axes[1].set_ylabel("cluster i")
axes[1].set_title("coarse-grained P")
for i in range(P_coarse.shape[0]):
    for j in range(P_coarse.shape[1]):
        axes[1].text(
            j,
            i,
            f"{P_coarse[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if P_coarse[i, j] < 0.6 else "black",
            fontsize=10,
        )
st.pyplot(fig)

st.subheader("Landscape colored by basin")
phens = np.asarray(gpm.data["phenotypes"].to_numpy(), dtype=float)
n_muts = np.asarray(gpm.data["n_mutations"].to_numpy(), dtype=int)
fig2, ax2 = plt.subplots(figsize=(8, 4))
scatter = ax2.scatter(n_muts, phens, c=labels, cmap="tab10", s=80, edgecolor="k")
ax2.set_xlabel("n_mutations")
ax2.set_ylabel("phenotype")
ax2.set_title(f"metastable basins (k={n_clusters})")
legend1 = ax2.legend(*scatter.legend_elements(), title="cluster", loc="best")
ax2.add_artist(legend1)
st.pyplot(fig2)
