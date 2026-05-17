"""Stochastic walker ensemble with live convergence stats."""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gpvolve import ConvergenceCheck, sample_paths
import gpvolve.paths.stochastic as stochastic
from utils import build_msm

st.title("Stochastic walker sampler")

st.markdown(
    r"""
Run a rayon-parallel Monte Carlo walker ensemble from $A$ to $B$. The
sampler stops on **both** of the following per-endpoint criteria:

1. **ESS via Sokal autocorrelation**: $N / (2\, \tau_{\text{int}}) \geq \text{ess\_min}$.
2. **Gelman-Rubin R-hat across chains**: $\hat{R} \leq \text{rhat\_max}$.

The Rust backend is currently
**{rust_status}**; speedup vs the pure-Python fallback is typically
several hundred fold on rugged landscapes.
""".format(
        rust_status="enabled" if stochastic._RUST_AVAILABLE else "disabled (CPU fallback)"
    )
)

col = st.columns(3)
length = col[0].slider("L", 2, 6, 4)
fixation = col[1].selectbox("Fixation", ["moran", "sswm", "mccandlish"])
pop_size = col[2].slider("Population size", 2, 1000, 100)

gpm, graph, msm = build_msm(length=length, fixation=fixation, population_size=pop_size)
genotypes = list(gpm.data["genotypes"])

col = st.columns(2)
source_label = col[0].selectbox("Source A", genotypes, index=0)
target_label = col[1].selectbox(
    "Target B", genotypes, index=len(genotypes) - 1
)
source_idx = genotypes.index(source_label)
target_idx = genotypes.index(target_label)
if source_idx == target_idx:
    st.error("Source and target must be different states.")
    st.stop()

col = st.columns(3)
ess_min = col[0].slider("ess_min", 20, 1000, 200, step=20)
chunk = col[1].slider("chunk_size", 500, 20000, 5000, step=500)
budget = col[2].slider("max_walkers", chunk, 200000, max(20000, chunk), step=500)

cc = ConvergenceCheck(
    ess_min=float(ess_min),
    chunk_size=chunk,
    max_walkers=budget,
    n_chains=4,
)

t0 = time.perf_counter()
try:
    ens = sample_paths(msm, source=source_idx, targets=target_idx, convergence=cc, seed=0)
    converged = True
except Exception as exc:  # noqa: BLE001
    st.error(f"Sampler did not converge inside the budget: {exc}")
    converged = False
    ens = None
elapsed_ms = (time.perf_counter() - t0) * 1e3

if not converged or ens is None:
    st.stop()

cv = ens.metadata["convergence"]

col = st.columns(4)
col[0].metric("Walkers", cv["n_walkers"])
col[1].metric("Chunks", cv["n_chunks"])
col[2].metric("Hit rate", f"{ens.probabilities.size > 0 and float(np.mean([p[-1] == target_idx for p in ens.paths])):.3f}")
col[3].metric("Wall-clock", f"{elapsed_ms:.0f} ms")

ess_target = cv["ess"].get(target_idx, float("nan"))
rhat_target = cv["rhat"].get(target_idx, float("nan"))
col = st.columns(2)
col[0].metric("ESS at target", f"{ess_target:.1f}")
col[1].metric("R-hat at target", f"{rhat_target:.4f}")

st.subheader("Per-walker path-product distribution")
fig, ax = plt.subplots(figsize=(8, 4))
hit_mask = np.array([p[-1] == target_idx for p in ens.paths])
nonzero = ens.probabilities[(ens.probabilities > 0) & hit_mask]
if nonzero.size > 0:
    ax.hist(np.log10(nonzero), bins=40, color="#4C72B0")
    ax.set_xlabel("log10(path probability)")
    ax.set_ylabel("count")
    ax.set_title(f"path probabilities for walkers reaching {target_label}")
else:
    ax.text(0.5, 0.5, "no walker reached the target", ha="center", va="center")
    ax.set_axis_off()
st.pyplot(fig)
