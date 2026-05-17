"""About page."""

from __future__ import annotations

import streamlit as st

st.markdown("### About")

st.markdown(
    """
**gpvolve-v2** is a clean-break rewrite of
[harmslab/gpvolve](https://github.com/harmslab/gpvolve). It shares a
modernization pattern with its sister projects:

- [gpmap-v2](https://github.com/lperezmo/gpmap-v2) -- the
  genotype-phenotype map container.
- [gpgraph-v2](https://github.com/lperezmo/gpgraph-v2) -- the NetworkX
  graph layer gpvolve composes on top of.
- [epistasis-v2](https://github.com/lperezmo/epistasis-v2) -- the
  epistatic coefficient fitting library.

All four packages: `uv` + `maturin` + PyO3 + rayon for the genuinely hot
loops, vectorized numpy elsewhere, type-hinted public API,
`mypy --strict` in CI, `python-semantic-release` with OIDC-trusted PyPI
publishing. The Rust hot paths in this repo are:

- ``sample_paths_csr`` -- rayon-parallel walker sampling, ~500x over
  pure Python on 2^14 maps.
- ``solve_bicgstab_csr`` -- BiCGSTAB committor solver, ~1700x over
  ``scipy.sparse.linalg.spsolve`` at 2^14.

The dormant ``msmtools`` dependency was dropped entirely; PCCA+ and TPT
are implemented from scratch.
"""
)
