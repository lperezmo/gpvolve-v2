# Optional: SLiM integration

SLiM (Selection on Linked Mutations) is a forward population-genetic
simulator. gpvolve-v2 has a stub backend that wraps SLiM output and tskit
tree sequences. Install the optional `sim` extra:

```bash
uv pip install 'gpvolve-v2[sim]'
```

This pulls in `pyslim >=1.0` and `tskit >=0.5`. Without the extra, the
backend raises `ImportError` with the install hint.

```python
from gpvolve.simulate import slim_available, tskit_available

if slim_available() and tskit_available():
    ...  # run a SLiM-backed analysis
```

The v2.0.0 release ships only the availability probes. The full SLiM
script generator (ported from the harmsm/gpvolve fork) is on the roadmap;
follow [#TBD](https://github.com/lperezmo/gpvolve-v2/issues) for status.
