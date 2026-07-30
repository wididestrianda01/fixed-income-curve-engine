"""Smoke test: the package is importable and declares a version."""

import yieldcurve


def test_package_exposes_a_version_string() -> None:
    assert isinstance(yieldcurve.__version__, str)
    assert yieldcurve.__version__.count(".") == 2
