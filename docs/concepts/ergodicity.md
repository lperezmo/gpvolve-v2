# Ergodicity and reversibility

A finite-state Markov chain has a unique stationary distribution iff it is
**irreducible** (equivalently, the directed graph of non-zero transitions
is strongly connected). It is **ergodic** when, additionally, it is
**aperiodic**: the greatest common divisor of cycle lengths in the chain's
underlying graph is 1. gpvolve-v2 exposes the standard predicates in
`gpvolve.markov` (re-exported at the top level):

```python
from gpvolve import (
    is_strongly_connected,
    is_irreducible,           # alias of is_strongly_connected
    is_ergodic,
    is_reversible,
    absorbing_states,
    transient_states,
)

is_strongly_connected(msm.transition_matrix)
is_ergodic(msm.transition_matrix)
is_reversible(msm.transition_matrix)
```

`is_strongly_connected` ignores self-loops, so a matrix that is row-stochastic
only because every row is absorbed by the diagonal does not pass.

`is_ergodic` is `is_irreducible(P) and any(diag(P) > 0)`. The diagonal check
is a sufficient aperiodicity criterion: a single positive diagonal entry
produces a length-1 cycle, which forces the period of every (communicating)
state to 1. Every matrix produced by `build_transition_matrix` carries an
absorption diagonal that satisfies this, so the check is exact for typical
gpvolve workflows. For chains constructed elsewhere with no self-loops, the
sufficient criterion is conservative and may report `False` on a chain that
is in fact aperiodic by a deeper cycle-gcd argument.

`is_reversible` performs the detailed-balance test
$\pi_i P_{ij} = \pi_j P_{ji}$ as a two-step check: support symmetry
(every nonzero $P_{ij}$ has a nonzero $P_{ji}$, which already fails for
SSWM-style asymmetric kernels), followed by the equality test on the
remaining pairs.

## Reversibility

## Reversibility

A chain is reversible if detailed balance holds:

$$
\pi_i \, P_{ij} \;=\; \pi_j \, P_{ji}
$$

for every pair $(i, j)$. Under reversibility:

- the backward committor equals $1 - q^{+}$
- the symmetric similarity $D^{1/2} P D^{-1/2}$ (with $D = \operatorname{diag}(\pi)$)
  is symmetric, so its eigenvectors are orthogonal
- PCCA+ has an analytic refinement step

gpvolve-v2 does not assume reversibility anywhere: the backward committor
is computed by explicitly building the time-reversed chain, and PCCA+ uses
the right eigenvectors of `P` directly rather than the symmetric form.
This makes the analyses work for SSWM/Moran chains on landscapes with
epistasis (which violate detailed balance) at the price of a slightly more
expensive linear algebra path.

## When connectivity fails

The genotype-phenotype graphs gpvolve-v2 works with are usually strongly
connected by construction (every edge in `gpgraph-v2` is added in both
directions). If you build a graph that drops edges for biological reasons
(e.g., one-step accessible only in the wildtype-to-mutant direction), check
the connectivity before running MSM analyses.

## Absorbing chains

When the chain has one or more absorbing states ($P_{ii} = 1$, no
off-diagonal mass) the standard TPT machinery refuses to operate: the
backward committor requires a strictly positive stationary distribution,
and on an absorbing chain $\pi$ puts mass 1 on the absorbing class and 0
on every transient state.

SSWM (`fixation="sswm"`) produces an absorbing chain on any landscape with
at least one local fitness maximum, since deleterious moves have fixation
probability 0 and a state with no beneficial neighbors becomes a true
sink. On a single-peak (Mount-Fuji-like) landscape SSWM produces *one*
absorbing state; on a rugged landscape with multiple local maxima, each
becomes its own sink and the chain is reducible.

gpvolve-v2 ships an absorbing-chain toolkit in `gpvolve.markov.absorbing`
(re-exported at the top level). The standard textbook decomposition
(Kemeny & Snell 1976, *Finite Markov Chains*, Ch 3) writes the matrix as

$$
P \;=\; \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix}
$$

with $Q$ the substochastic block on transient states and $R$ the
transient-to-absorbing block. Useful quantities follow:

| Function | Meaning |
| --- | --- |
| `fundamental_matrix(P)` | $N = (I-Q)^{-1}$. Entry $N_{ij}$ is the expected number of visits to transient state $j$ starting from transient $i$. |
| `absorption_probabilities(P)` | $B = N R$. Row $i$, column $k$ is $\Pr[\text{absorbed in } k \mid X_0 = i]$. Row sums equal 1. |
| `quasi_stationary_distribution(P)` | The Darroch-Seneta (1965) left Perron vector of $Q$. Conditional on non-absorption, the chain converges to this distribution; the lifetime in QSD is $1/(1 - \lambda_1)$. |
| `conditional_mfpt(P, A, B)` | $E[\tau_B \mid \text{absorbed in } B, X_0 = i]$ via Doob's h-transform (Norris 1997, Theorem 4.2.3). |
| `absorption_rate(P, A, B)` | $1/E[\tau_B \mid X_0 \sim \text{initial}]$. The chemical-kinetics rate constant for an irreversible A $\to$ B process (Hanggi-Talkner-Borkovec 1990). |

The `mfpt` solver raises `GpvolveError` when the chain has absorbing
states outside the target set, because in that regime the standard
$(I - Q) m = 1$ system is singular and the unconditional expectation
$E[\tau_B]$ diverges. Use `conditional_mfpt` instead.

### Worked example

```python
import numpy as np
from gpvolve import (
    absorbing_states,
    absorption_rate,
    fundamental_matrix,
    quasi_stationary_distribution,
)

# Build any SSWM chain on a single-peak landscape; here on L=5 binary map.
# msm = ...  # see examples/streamlit/utils
P = msm.transition_matrix

abs_idx = absorbing_states(P)            # e.g., [31] -> the fitness peak
N, transient = fundamental_matrix(P)     # 31x31 expected-visits matrix
qsd, _, lam = quasi_stationary_distribution(P)
# 1 - lam is the per-step absorption probability from the QSD;
# 1 / (1 - lam) is the metastable lifetime.

# Compare reactive rate (TPT) with absorption rate (MFPT-based):
A = 0   # genotype "AAAAA"
B = 31  # peak "TTTTT"
k_reactive   = gpvolve.rate(P, A=A, B=B)               # 0.0 on absorbing chains
k_absorption = absorption_rate(P, A=A, B=B)            # > 0; the meaningful rate
```

### Reactive rate vs. absorption rate

For an **ergodic** chain (Moran or McCandlish at any finite $N$),
`rate(P, A, B)` returns the long-time reactive flux

$$
k_{AB} \;=\; \sum_{i \in A, j \notin A} \pi_i P_{ij} q^{+}_j
$$

(Berezhkovskii-Hummer-Szabo 2009), and `absorption_rate(P, A, B)` returns
$1 / E[\tau_B]$. The two agree in the slow-transition limit
(Eyring/Kramers).

For an **absorbing** chain (e.g., SSWM), the reactive rate collapses to 0
because $\pi_A = 0$, and only `absorption_rate` carries useful kinetic
information. The streamlit TPT explorer shows both side-by-side and
explains the contrast when SSWM produces a non-ergodic chain.
