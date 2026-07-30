"""Smoke test: the package is importable and declares a version."""

import curveengine


def test_package_exposes_a_version_string() -> None:
    assert isinstance(curveengine.__version__, str)
    assert curveengine.__version__.count(".") == 2
