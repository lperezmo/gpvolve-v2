# Plotting

`gpvolve.pyplot` ships lightweight matplotlib wrappers for sanity-checking
MSM outputs. Install with the `[plot]` extra:

```bash
uv pip install 'gpvolve-v2[plot]'
```

## Available plots

```python
from gpvolve.pyplot import (
    draw_transition_matrix,
    draw_stationary,
    draw_flux_heatmap,
    draw_timescales,
    draw_landscape_1d,
)

draw_transition_matrix(msm)
draw_stationary(msm)
draw_flux_heatmap(flux)
draw_timescales(timescales(msm.transition_matrix, k=5))
draw_landscape_1d(graph, fitness_column="phenotypes")
```

These return the `matplotlib.axes.Axes` they drew on. They are intended
for quick inspection rather than publication; for publication-quality
figures, work directly with the `gpgraph-v2` layout API and overlay the
MSM data manually.
