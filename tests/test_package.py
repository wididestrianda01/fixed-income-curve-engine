"""Smoke test: the package is importable and declares a single sourced version.

HYGIENE-07: pyproject.toml derives its version from ``yieldcurve.__version__``
(hatchling dynamic version), so the installed distribution metadata and the
package attribute are one source. This test guards against drift.
"""

from __future__ import annotations

import importlib.metadata

import yieldcurve


def test_package_exposes_a_version_string() -> None:
    assert isinstance(yieldcurve.__version__, str)
    assert yieldcurve.__version__.count(".") == 2


def test_installed_metadata_version_equals_package_version() -> None:
    assert importlib.metadata.version("yieldcurve") == yieldcurve.__version__
