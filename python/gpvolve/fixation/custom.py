"""Public re-export of `register_fixation_model` for user-supplied custom models.

```python
import numpy as np
from numpy.typing import NDArray
from gpvolve.fixation import register_fixation_model

@register_fixation_model(
    name="boltzmann",
    bounded_unit_interval=True,
    required_params=frozenset({"temperature"}),
)
def boltzmann(
    fi: NDArray[np.float64],
    fj: NDArray[np.float64],
    /,
    *,
    temperature: float,
    **_: object,
) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-(fj - fi) / temperature))
```
"""

from __future__ import annotations

from gpvolve.fixation.protocol import register_fixation_model

__all__ = ["register_fixation_model"]
