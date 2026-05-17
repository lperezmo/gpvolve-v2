# Markov chains on fitness landscapes

A short primer on what gpvolve-v2 is actually computing and why the choices
matter.

## The discrete-time chain

gpvolve-v2 models evolution on a genotype-phenotype map as a discrete-time
Markov chain on the genotype graph. The state space is the set of genotypes
in the map (one per row of `gpm.data`); the transition matrix `P` is
row-stochastic and supported on the graph's edges plus the diagonal.

Each off-diagonal entry has the form

$$
P_{ij} \;=\; \frac{\pi_{\text{fix}}(f_i, f_j)}{k_{\max}}
$$

where $\pi_{\text{fix}}$ is the fixation probability for a mutation from
genotype $i$ to a neighbor $j$, evaluated at fitnesses $f_i$ and $f_j$, and
$k_{\max}$ is the maximum out-degree of any node. The normalization by
$k_{\max}$ is the discrete-time convention of Sailer and Harms (2017): in
each time step the population attempts one mutation at one of $k_{\max}$
neighbor sites with equal probability, then accepts or rejects according to
the fixation kernel. The diagonal entry is the residual:

$$
P_{ii} \;=\; 1 - \sum_{j \,\sim\, i} P_{ij}
$$

so the chain is row-stochastic by construction. gpvolve-v2 asserts this to
`1e-12` after building the matrix and raises `NonStochasticError` if a
bounded fixation kernel ever returns a value outside `[0, 1]`. See
[SCHEMA](https://github.com/lperezmo/gpvolve-v2/blob/main/SCHEMA.md) section 2
for the full contract.

## Strong-selection weak-mutation

The "discrete time step" here is one mutation-and-fixation attempt, not a
generation. The model assumes the strong-selection weak-mutation regime
where adaptive substitutions are well-separated in time: at most one
mutation segregates in the population at a time, and selection on
between-allele competition is much faster than the mutation rate. Under that
regime, the population is monomorphic between fixation events and the chain
on genotypes is a faithful summary of the underlying Wright-Fisher process.

When selection is weak relative to drift, or when many mutations segregate
simultaneously, the SSWM picture breaks down. For those regimes use
`simulate.wright_fisher` (discrete generations with multinomial sampling)
or `simulate.gillespie` (exact event times with arbitrary fixation kernels)
instead of the MSM machinery.

## Reversibility

The chain is reversible only when the fixation kernel and the fitness
landscape together satisfy detailed balance. SSWM, Moran, and McCandlish on
log-additive landscapes give reversible chains; the same kernels on
landscapes with epistasis generally do not. The backward committor solver
(`backward_committor`) constructs the time-reversed chain explicitly so
TPT works in both cases. For PCCA+ clustering we use the right
eigenvectors of `P` directly rather than the symmetric similarity transform,
so non-reversible chains do not need a special path.

## When the matrix is large

For maps with 2^L states, `P` has order `L * 2^L` non-zeros. Up to about
2^16 states everything fits in memory comfortably. Above that, the
bottleneck is typically the dense spectral analysis used for `pcca_plus`,
which scales as the cube of the state count. The MSM container does not
eagerly compute eigenvectors; only the analyses that need them pay the cost.

## When to trust the numbers

Two failure modes to watch for:

1. **The graph is not strongly connected.** Then the stationary
   distribution is not unique and committors may be ill-defined on
   the disconnected components. Check with
   `gpvolve.markov.is_strongly_connected`.
2. **The fixation kernel underflows to zero.** With population sizes in the
   thousands and large fitness gaps, Moran and McCandlish can return values
   smaller than 1e-300. gpvolve-v2 keeps these entries in the sparsity
   pattern with very small but non-zero weights so path-finding still
   traverses them; spectral analysis and committor solves should still be
   stable, but the resulting timescales become astronomical.

## Where to read more

- McCandlish (2011) for the fixation kernel.
- Sailer and Harms (2017) for the discrete-time normalization convention.
- Berezhkovskii, Hummer, Szabo (2009) for the TPT formulation we implement.
- Roeblitz and Weber (2013) for PCCA+.
- Sokal (1989) for the integrated autocorrelation time used by
  `ConvergenceCheck`.
