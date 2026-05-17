# Fixation models

A fixation model maps a pair of fitnesses $(f_i, f_j)$ to a probability that
a new allele $j$ arising on background $i$ fixes in the population.
gpvolve-v2 ships five built-ins.

## Built-ins

| Name | Required params | Bounded in `[0, 1]`? | Reference |
|---|---|---|---|
| `sswm` (alias `strong_selection_weak_mutation`) | none | yes | Gillespie 1984 |
| `moran` | `population_size` | yes | Sella and Hirsch 2005 |
| `mccandlish` (alias `mcclandish` for v1 compat) | `population_size` | yes | McCandlish 2011 |
| `bloom_dms` | `pi_table` | yes | empirical (Bloom DMS) |
| `weak_mutation` | none | no | Lynch and Conery 2003 |

Only the bounded kernels can build a row-stochastic transition matrix. Passing
`weak_mutation` to `build_transition_matrix` raises `NonStochasticError`.

## Registering a custom kernel

```python
from gpvolve import register_fixation_model
import numpy as np

@register_fixation_model(
    name="my_kernel",
    bounded_unit_interval=True,
    required_params=frozenset({"sigma"}),
)
def my_kernel(fi, fj, sigma, **_):
    s = (fj - fi) / fi
    return 0.5 * (1.0 + np.tanh(s / sigma))
```

The decorator stamps the Protocol attributes (`name`,
`bounded_unit_interval`, `required_params`) onto the function and registers
it in the module-level registry. Look up the registration with
`gpvolve.get_fixation_model("my_kernel")`.

## v1 spelling

The original gpvolve had a misspelled kernel name (`mcclandish` instead of
`mccandlish`). v2 ships both spellings as aliases for the same function so
v1 scripts keep working.
