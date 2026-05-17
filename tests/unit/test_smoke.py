"""Phase 0 smoke test: package imports and exposes a version."""

import gpvolve


def test_version_is_set() -> None:
    assert isinstance(gpvolve.__version__, str)
    assert len(gpvolve.__version__) > 0
