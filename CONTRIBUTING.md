# Contributing to gpvolve-v2

## Dev setup

```bash
git clone https://github.com/lperezmo/gpvolve-v2
cd gpvolve-v2
uv sync
uv run maturin develop --release
uv run pytest
```

Python 3.11+, Rust stable (>=1.75), `uv`, and `maturin` are required.

## Commit message conventions

We use Conventional Commits with `python-semantic-release`. Allowed tags and their effect on
the version bump:

- `feat:` minor bump
- `fix:` patch bump
- `perf:` patch bump
- `refactor:`, `style:`, `docs:`, `test:`, `build:`, `ci:`, `chore:`, `revert:` no bump

Commits with a `!` after the type or a `BREAKING CHANGE:` footer trigger a major bump.

## Code style

- `ruff check` and `ruff format` must pass on `python/gpvolve` and `tests`.
- `mypy --strict` must pass on `python/gpvolve`.
- All public symbols need type hints.
- No em dashes anywhere. No emojis in code or commit messages.

## Tests

- `tests/unit/` -- one file per module, fast unit tests.
- `tests/property/` -- hypothesis-based property tests.
- `tests/schema/` -- enforces SCHEMA.md contracts.
- `tests/golden/` -- numeric agreement with published landmark results.
- `tests/benchmarks/` -- pytest-benchmark, runs in CI but not gating.

Run the full suite with `uv run pytest`. Run a single test file with
`uv run pytest tests/unit/test_transition.py -v`.
