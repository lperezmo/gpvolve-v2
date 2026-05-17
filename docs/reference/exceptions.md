# Exceptions

All gpvolve-v2 exceptions descend from `GpvolveError`.

- `GpvolveError` -- base class.
- `ModelError` -- bad fixation model name, missing required params, unsupported
  self-loop mode, or invalid graph topology.
- `NonStochasticError` -- transition matrix violates row-stochasticity or
  bounds invariants.
- `ConvergenceError` -- a stationary solver or stochastic sampler did not
  converge inside the allotted budget.
- `SchemaError` -- serialized payload has a version mismatch or a
  gpm-hash mismatch.
