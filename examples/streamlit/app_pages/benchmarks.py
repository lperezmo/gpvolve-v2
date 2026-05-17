"""Live benchmarks: Rust walker sampling and BiCGSTAB committor solve."""

from __future__ import annotations

import time
from itertools import product

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import streamlit as st
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap
from gpvolve import (
    ConvergenceCheck,
    GenotypePhenotypeMSM,
    build_transition_matrix,
    forward_committor,
    sample_paths,
)
import gpvolve.paths.stochastic as stochastic
import gpvolve.paths.tpt as tpt

st.title("Benchmarks")

st.markdown(
    """
Live timings on this machine for the two Rust hot paths. Hit ``Run`` after
adjusting the sliders; the rest of the showcase reuses the same kernels.
"""
)


def _make_msm(sites: int) -> GenotypePhenotypeMSM:
    rng = np.random.default_rng(0)
    eff = rng.normal(0.0, 0.4, size=sites)
    gts = ["".join(b) for b in product("01", repeat=sites)]
    phens = [
        float(np.exp(sum(eff[i] for i, c in enumerate(g) if c == "1"))) for g in gts
    ]
    gpm = GenotypePhenotypeMap(wildtype="0" * sites, genotypes=gts, phenotypes=phens)
    graph = GenotypePhenotypeGraph.from_gpm(gpm)
    return GenotypePhenotypeMSM.from_graph(
        graph, fitness_column="phenotypes", fixation="moran", population_size=100
    )


tab_sampler, tab_committor = st.tabs(["Walker sampler", "Committor solver"])

with tab_sampler:
    st.subheader("sample_paths: Rust rayon walkers vs Python")
    col = st.columns(2)
    max_pow = col[0].slider("max log2(n_states)", 6, 12, 10)
    n_walkers = col[1].slider("walkers per chunk", 200, 5000, 1000, step=200)
    if st.button("Run walker benchmark", key="run_walkers"):
        rows = []
        progress = st.progress(0.0)
        sizes = list(range(6, max_pow + 1))
        for i, sites in enumerate(sizes):
            msm = _make_msm(sites)
            target = int(np.argmax(msm.gpm.data["phenotypes"].to_numpy()))
            cc = ConvergenceCheck(
                ess_min=10.0, chunk_size=n_walkers, max_walkers=n_walkers, n_chains=4
            )

            t0 = time.perf_counter()
            sample_paths(msm, source=0, targets=target, convergence=cc, seed=0)
            t_rust = (time.perf_counter() - t0) * 1e3

            t_py = float("nan")
            # Only do Python at the small end; it scales linearly so 2^8 is
            # already ~3 s.
            if sites <= 8:
                original = stochastic._RUST_AVAILABLE
                stochastic._RUST_AVAILABLE = False
                try:
                    t0 = time.perf_counter()
                    sample_paths(msm, source=0, targets=target, convergence=cc, seed=0)
                    t_py = (time.perf_counter() - t0) * 1e3
                finally:
                    stochastic._RUST_AVAILABLE = original

            rows.append(
                {
                    "n_states": 2**sites,
                    "rust (ms)": round(t_rust, 1),
                    "python (ms)": round(t_py, 1) if not np.isnan(t_py) else "",
                }
            )
            progress.progress((i + 1) / len(sizes))

        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["n_states"], df["rust (ms)"], marker="o", label="rust")
        py_vals = pd.to_numeric(df["python (ms)"], errors="coerce")
        ax.plot(df["n_states"], py_vals, marker="s", label="python")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("n_states")
        ax.set_ylabel("time (ms)")
        ax.set_title("sample_paths")
        ax.legend()
        st.pyplot(fig)

with tab_committor:
    st.subheader("forward_committor: Rust BiCGSTAB vs scipy spsolve")
    max_pow_c = st.slider("max log2(n_states)", 6, 14, 12, key="committor_max")
    if st.button("Run committor benchmark", key="run_committor"):
        rows = []
        progress = st.progress(0.0)
        sizes = list(range(6, max_pow_c + 1))
        for i, sites in enumerate(sizes):
            msm = _make_msm(sites)
            target = int(np.argmax(msm.gpm.data["phenotypes"].to_numpy()))

            # Rust path (auto-dispatch above 256 states).
            t0 = time.perf_counter()
            forward_committor(msm.transition_matrix, A=0, B=target)
            t_rust = (time.perf_counter() - t0) * 1e3

            # Force spsolve.
            t_lu = float("nan")
            if sites <= 12:
                original = tpt._RUST_BICGSTAB_AVAILABLE
                tpt._RUST_BICGSTAB_AVAILABLE = False
                try:
                    t0 = time.perf_counter()
                    forward_committor(msm.transition_matrix, A=0, B=target)
                    t_lu = (time.perf_counter() - t0) * 1e3
                finally:
                    tpt._RUST_BICGSTAB_AVAILABLE = original

            rows.append(
                {
                    "n_states": 2**sites,
                    "rust bicgstab (ms)": round(t_rust, 2),
                    "scipy spsolve (ms)": round(t_lu, 2) if not np.isnan(t_lu) else "",
                }
            )
            progress.progress((i + 1) / len(sizes))

        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["n_states"], df["rust bicgstab (ms)"], marker="o", label="rust bicgstab")
        lu_vals = pd.to_numeric(df["scipy spsolve (ms)"], errors="coerce")
        ax.plot(df["n_states"], lu_vals, marker="s", label="scipy spsolve")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("n_states")
        ax.set_ylabel("time (ms)")
        ax.set_title("forward_committor")
        ax.legend()
        st.pyplot(fig)
