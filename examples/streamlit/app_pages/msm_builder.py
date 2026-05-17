"""Interactive MSM builder."""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from utils import build_msm

st.markdown("### MSM builder")

st.markdown(
    r"""
Pick a landscape size, a fixation model, and (for Moran/McCandlish) a
population size. The output is the row-stochastic transition matrix $P$
and its stationary distribution $\pi$. The matrix is exactly the object
the rest of the pages consume.
"""
)

col = st.columns(3)
length = col[0].slider("Sequence length L", min_value=2, max_value=6, value=4)
fixation = col[1].selectbox("Fixation model", ["moran", "sswm", "mccandlish"])
pop_size = col[2].slider("Population size", min_value=2, max_value=2000, value=100)

t0 = time.perf_counter()
gpm, graph, msm = build_msm(length=length, fixation=fixation, population_size=pop_size)
elapsed_ms = (time.perf_counter() - t0) * 1e3

col = st.columns(3)
col[0].metric("States", msm.n_states)
col[1].metric("Non-zeros in P", msm.transition_matrix.nnz)
col[2].metric("Build time", f"{elapsed_ms:.1f} ms")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
P_dense = msm.transition_matrix.toarray()
im = axes[0].imshow(P_dense, cmap="viridis", aspect="auto")
axes[0].set_xlabel("j")
axes[0].set_ylabel("i")
axes[0].set_title(f"transition matrix (n={msm.n_states})")
fig.colorbar(im, ax=axes[0], label="P(i -> j)")

n_muts = np.asarray(gpm.data["n_mutations"].to_numpy(), dtype=int)
order = np.argsort(n_muts)
axes[1].bar(np.arange(msm.n_states), msm.stationary[order], color="#4C72B0")
axes[1].set_xlabel("state (ordered by n_mutations)")
axes[1].set_ylabel("pi(i)")
axes[1].set_title("stationary distribution")
st.pyplot(fig)

with st.expander("show genotypes -> phenotypes"):
    table = gpm.data[["genotypes", "phenotypes", "n_mutations"]].copy()
    table.insert(0, "i", table.index)
    table["pi"] = msm.stationary
    st.dataframe(table, width="stretch", height=300)
