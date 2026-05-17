# `gpvolve.analysis`

Higher-level analyses on top of the MSM.

- `find_peaks(graph, *, fitness_column) -> list[int]`
  Indices of nodes whose fitness exceeds every neighbor's.

- `find_valleys(graph, *, fitness_column) -> list[int]`
  Indices of nodes whose fitness is below every neighbor's.

- `accessible_peaks(graph, source, *, fitness_column) -> list[int]`
  Peaks reachable from `source` by strictly-increasing-fitness edges.
