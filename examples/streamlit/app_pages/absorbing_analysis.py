"""Absorbing-chain analysis: fundamental matrix, QSD, and absorption rate.

Surfaces the toolkit in ``gpvolve.markov.absorbing`` for landscapes whose
dynamics produce absorbing states (SSWM is the canonical example: any local
fitness maximum becomes a true sink).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gpvolve import (
    absorbing_states,
    absorption_rate,
    conditional_mfpt,
    fundamental_matrix,
    is_ergodic,
    is_reversible,
    quasi_stationary_distribution,
)
from gpvolve.exceptions import GpvolveError
from utils import build_msm

st.title("Absorbing-chain analysis")

st.markdown(
    r"""
For chains with absorbing states, the standard TPT formulation breaks down
(the backward committor needs a strictly positive stationary distribution
and the reactive rate collapses to zero because $\pi_A = 0$). The
Kemeny-Snell absorbing-chain decomposition replaces it:

$$
P \;=\; \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix}
$$

with $Q$ on transient states. Tools below:

- **Fundamental matrix** $N = (I - Q)^{-1}$ -- expected visits to each
  transient state before absorption.
- **Quasi-stationary distribution** (Darroch-Seneta 1965) -- the
  metastable distribution on the transient class.
- **Absorption rate** $k = 1 / E[\tau_B]$ -- the chemical-kinetics rate
  constant (Hanggi-Talkner-Borkovec 1990).

Try `fixation=sswm` to see an absorbing chain; `moran` or `mccandlish`
produce ergodic chains where these tools refuse with an informative error.
"""
)

col = st.columns(3)
length = col[0].slider("L", 2, 5, 5)
fixation = col[1].selectbox("Fixation", ["sswm", "moran", "mccandlish"])
pop_size = col[2].slider("Population size", 2, 1000, 100)

gpm, graph, msm = build_msm(length=length, fixation=fixation, population_size=pop_size)
genotypes = list(gpm.data["genotypes"])
P = msm.transition_matrix

abs_idx = absorbing_states(P)

col = st.columns(4)
col[0].metric("states", P.shape[0])
col[1].metric("absorbing", abs_idx.size)
col[2].metric("ergodic", "yes" if is_ergodic(P) else "no")
col[3].metric("reversible", "yes" if is_reversible(P) else "no")

if abs_idx.size == 0:
    st.info(
        "This chain has no absorbing states; the toolkit on this page is "
        "undefined here. Switch to `sswm` to see the absorbing-chain "
        "analytics, or to the TPT explorer page for ergodic-chain TPT."
    )
    st.stop()

st.subheader("Absorbing states")
st.write(
    "The chain has "
    + f"**{abs_idx.size}** absorbing state(s): "
    + ", ".join(f"`{genotypes[i]}`" for i in abs_idx)
)

# Source / target selection.
col = st.columns(2)
source_label = col[0].selectbox(
    "Source A", genotypes, index=0, key="abs_source"
)
target_label = col[1].selectbox(
    "Target B (must be one of the absorbing peaks)",
    [genotypes[i] for i in abs_idx],
    key="abs_target",
)
source_idx = genotypes.index(source_label)
target_idx = genotypes.index(target_label)

if source_idx == target_idx:
    st.error("Source and target must be different states.")
    st.stop()

st.subheader("Kinetics: absorption rate and conditional MFPT")
try:
    k_abs = absorption_rate(P, A=source_idx, B=target_idx)
    cmfpt = conditional_mfpt(P, A=source_idx, B=target_idx)[0]
except GpvolveError as exc:
    st.error(f"Could not compute kinetics: {exc}")
    st.stop()

col = st.columns(2)
col[0].metric("absorption rate 1 / E[tau]", f"{k_abs:.3e}")
col[1].metric("E[tau_B | absorbed in B]", f"{cmfpt:.2f}")

# Quasi-stationary distribution.
st.subheader("Quasi-stationary distribution")
st.markdown(
    "Conditional on non-absorption, the chain converges to the QSD: the "
    "left Perron eigenvector of the transient block $Q$ with eigenvalue "
    "$\\lambda_1$. The metastable lifetime is $1 / (1 - \\lambda_1)$."
)

qsd, qsd_trans, lam = quasi_stationary_distribution(P)
col = st.columns(2)
col[0].metric("Perron eigenvalue lambda_1", f"{lam:.6f}")
col[1].metric("metastable lifetime", f"{1.0 / (1.0 - lam):.2f}" if lam < 1 else "inf")

n_muts = np.asarray(gpm.data["n_mutations"].to_numpy(), dtype=int)
fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(n_muts, qsd, c=qsd, cmap="viridis", s=80, edgecolors="k", linewidths=0.5)
for i in abs_idx:
    ax.axvline(int(n_muts[i]), color="red", linestyle=":", alpha=0.5)
ax.set_xlabel("Hamming distance from wildtype")
ax.set_ylabel("QSD mass")
ax.set_title("Quasi-stationary distribution on transient states")
fig.colorbar(sc, ax=ax, label="qsd")
st.pyplot(fig)

# Fundamental matrix heatmap.
st.subheader("Fundamental matrix: expected visits before absorption")
st.markdown(
    "$N_{ij}$ is the expected number of times the chain visits transient "
    "state $j$ starting from transient state $i$, summed over all time "
    "before absorption. The trace $\\sum_i N_{ii}$ is the expected total "
    "self-visits across the transient class."
)

N, trans_idx = fundamental_matrix(P)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(N, cmap="magma", aspect="auto")
ax.set_xlabel("transient state index j")
ax.set_ylabel("transient state index i")
ax.set_title(f"N = (I - Q)^-1, shape {N.shape}, trace = {np.trace(N):.2f}")
fig.colorbar(im, ax=ax, label="E[visits]")
st.pyplot(fig)

st.caption(
    "Row sums of N give E[time to absorption] from each transient state; "
    "off-diagonal entries reveal which transient states act as kinetic "
    "bottlenecks (high column sums) on the way to absorption."
)
