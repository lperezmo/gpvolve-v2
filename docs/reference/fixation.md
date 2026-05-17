# `gpvolve.fixation`

Fixation models and the registry.

## Protocol

```python
@runtime_checkable
class FixationModel(Protocol):
    name: str
    bounded_unit_interval: bool
    required_params: frozenset[str]

    def __call__(
        self,
        fitness_i: NDArray[float64],
        fitness_j: NDArray[float64],
        /,
        **params: Any,
    ) -> NDArray[float64]: ...
```

## Built-ins

- `strong_selection_weak_mutation(fi, fj) -> NDArray[float64]` (alias `sswm`)
- `moran(fi, fj, *, population_size) -> NDArray[float64]`
- `mccandlish(fi, fj, *, population_size) -> NDArray[float64]` (alias `mcclandish`)
- `bloom_dms(fi, fj, *, pi_table, indices=None) -> NDArray[float64]`
- `weak_mutation(fi, fj) -> NDArray[float64]` (unbounded; cannot build a row-stochastic matrix)

## Registry

- `get_fixation_model(name) -> FixationModel`
- `list_fixation_models() -> list[str]`
- `validate_params(model, params) -> None` (raises `ModelError`)
- `register_fixation_model(*, name, bounded_unit_interval, required_params, aliases=())` decorator
