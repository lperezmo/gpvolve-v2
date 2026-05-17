# `gpvolve.pyplot`

Lightweight matplotlib wrappers. Install with the `[plot]` extra.

- `draw_transition_matrix(msm, *, ax=None) -> Axes`
- `draw_stationary(msm, *, ax=None) -> Axes`
- `draw_flux_heatmap(flux, *, ax=None) -> Axes`
- `draw_timescales(ts, *, ax=None) -> Axes`
- `draw_landscape_1d(graph, *, fitness_column, ax=None) -> Axes`

Every function accepts an optional `ax` argument so you can compose the
plots with other matplotlib artists. They all return the `Axes` they drew
on.
