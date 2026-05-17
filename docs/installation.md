# Installation

Python 3.11+, prebuilt wheels for Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows (x64).

```bash
pip install gpvolve-v2
```

Or with uv:

```bash
uv add gpvolve-v2
```

## Optional extras

Plotting (matplotlib):

```bash
pip install "gpvolve-v2[plot]"
```

Simulation backends (pyslim, tskit):

```bash
pip install "gpvolve-v2[sim]"
```

## Streamlit showcase

The interactive multi-page tour lives at
[gpvolve-v2.streamlit.app](https://gpvolve-v2.streamlit.app). To run it locally:

```bash
git clone https://github.com/lperezmo/gpvolve-v2
cd gpvolve-v2
uv sync
uv run maturin develop --release
uv run --with streamlit --with matplotlib --with pandas \
    streamlit run examples/streamlit/showcase.py
```

The `[streamlit]` extra pulls in `streamlit` and `matplotlib` if you only want the
runtime deps for the app:

```bash
pip install "gpvolve-v2[streamlit]"
```

The showcase source itself lives under `examples/streamlit/` (not packaged on PyPI).

## From source

```bash
git clone https://github.com/lperezmo/gpvolve-v2
cd gpvolve-v2
uv sync
uv run maturin develop --release
```
