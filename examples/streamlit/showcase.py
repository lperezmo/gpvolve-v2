"""Entry point for the gpvolve-v2 Streamlit showcase."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="gpvolve-v2",
    page_icon=":dna:",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
      header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = [
    st.Page("app_pages/intro.py", title="Intro", icon=":material/home:"),
    st.Page("app_pages/msm_builder.py", title="MSM builder", icon=":material/hub:"),
    st.Page("app_pages/tpt_explorer.py", title="TPT explorer", icon=":material/route:"),
    st.Page(
        "app_pages/absorbing_analysis.py",
        title="Absorbing chains",
        icon=":material/track_changes:",
    ),
    st.Page(
        "app_pages/sampler.py",
        title="Stochastic sampler",
        icon=":material/casino:",
    ),
    st.Page(
        "app_pages/clustering.py",
        title="PCCA+ clustering",
        icon=":material/scatter_plot:",
    ),
    st.Page("app_pages/benchmarks.py", title="Benchmarks", icon=":material/speed:"),
    st.Page("app_pages/about.py", title="About", icon=":material/info:"),
]

nav = st.navigation(pages)
nav.run()
