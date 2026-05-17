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

Streamlit demo app:

```bash
pip install "gpvolve-v2[streamlit]"
```

## From source

```bash
git clone https://github.com/lperezmo/gpvolve-v2
cd gpvolve-v2
uv sync
uv run maturin develop --release
```
