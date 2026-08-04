"""Built-artifact contracts: wheel resources, sdist denylist, git hygiene.

Task 13 packaging gate. The artifact tests operate on the outputs of
``uv build --out-dir dist-review`` (see the Task 13 brief); they fail with a
clear message when no artifacts exist. The git-hygiene test needs no
artifacts and runs in the checkout.
"""

from __future__ import annotations

import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRS = (REPO_ROOT / "dist-review", REPO_ROOT / "dist")

# Path components or filenames that must never appear inside a built artifact:
# agent state, vector databases, caches, coverage output, worktrees, and local
# developer tooling (HYGIENE-01, HYGIENE-04, HYGIENE-05, SEC-03). The match is
# a substring on normalized member paths, so a nested ``foo/.coverage`` or a
# stray ``bar/__pycache__/baz.pyc`` is caught.
LOCAL_STATE_MARKERS = (
    ".claude-flow",
    ".swarm",
    ".codegraph",
    "graphify-out",
    "ruvector.db",
    ".worktrees",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    ".ipynb_checkpoints",
    ".coverage",
    "coverage.xml",
    "htmlcov",
    ".vscode",
    ".idea",
    ".serena",
    ".DS_Store",
    ".superpowers",
    "docs/superpowers",
    ".context",
    "dist-review",
    ".egg-info",
    "data/snapshots",
    "data/raw",
    "data/processed",
    "data/cache",
    ".parquet",
    ".pkl",
    ".joblib",
)


def find_built_wheel() -> Path:
    """The newest wheel under dist-review/ or dist/; fails with guidance."""
    for directory in ARTIFACT_DIRS:
        candidates = sorted(directory.glob("yieldcurve-*.whl"))
        if candidates:
            return candidates[-1]
    raise AssertionError(
        "no built wheel found; run `uv build --out-dir dist-review` first "
        f"(looked in {[str(d) for d in ARTIFACT_DIRS]})"
    )


def find_built_sdist() -> Path:
    """The newest sdist under dist-review/ or dist/; fails with guidance."""
    for directory in ARTIFACT_DIRS:
        candidates = sorted(directory.glob("yieldcurve-*.tar.gz"))
        if candidates:
            return candidates[-1]
    raise AssertionError(
        "no built sdist found; run `uv build --out-dir dist-review` first "
        f"(looked in {[str(d) for d in ARTIFACT_DIRS]})"
    )


def _wheel_members(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def _sdist_members(sdist: Path) -> list[str]:
    with tarfile.open(sdist) as archive:
        return archive.getnames()


def _assert_no_local_state(members: list[str], artifact: str) -> None:
    for member in members:
        for marker in LOCAL_STATE_MARKERS:
            assert marker not in member, f"{artifact}: {member} matches {marker!r}"


def test_wheel_contains_every_packaged_dataset_and_scenarios() -> None:
    """The wheel ships the frozen snapshot (every CSV the manifest lists, plus
    the manifest), the scenario configuration, and the type marker
    (DOC-10, HYGIENE-01)."""
    wheel = find_built_wheel()
    members = _wheel_members(wheel)

    with zipfile.ZipFile(wheel) as archive:
        manifest_bytes = archive.read("yieldcurve/data/snapshot_manifest.toml")
    manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
    datasets = manifest["datasets"]
    assert datasets, "the packaged manifest must list datasets"

    for name in datasets:
        assert f"yieldcurve/data/{name}.csv" in members, name

    assert "yieldcurve/data/snapshot_manifest.toml" in members
    assert "yieldcurve/risk/scenarios.toml" in members
    assert "yieldcurve/py.typed" in members


def test_wheel_contains_only_package_and_metadata_members() -> None:
    """The wheel is the package plus dist-info: no checkout files, tests, or
    documentation sneak in."""
    wheel = find_built_wheel()
    for member in _wheel_members(wheel):
        assert member.startswith("yieldcurve/") or ".dist-info/" in member, member


def test_wheel_has_no_local_state_paths() -> None:
    """The artifact-manifest contract applies to the wheel as well as the
    sdist (design spec section 7)."""
    wheel = find_built_wheel()
    _assert_no_local_state(_wheel_members(wheel), wheel.name)


def test_sdist_contains_sources_resources_and_lockfile() -> None:
    """The sdist is a faithful source archive: package sources and resources,
    the app, tests, notebooks, scripts, docs, the lockfile, and packaging
    metadata."""
    sdist = find_built_sdist()
    members = _sdist_members(sdist)
    expected_suffixes = (
        "src/yieldcurve/risk/scenarios.toml",
        "src/yieldcurve/data/snapshot_manifest.toml",
        "src/yieldcurve/py.typed",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "app/data.py",
        "tests/conftest.py",
        "notebooks/src/01-curve-construction.py",
        "scripts/topynb.py",
        "docs/hull-white-limitations.md",
    )
    for suffix in expected_suffixes:
        assert any(member.endswith(suffix) for member in members), suffix


def test_sdist_denylist_rejects_local_state() -> None:
    """The sdist excludes agent state, vector databases, caches, coverage
    output, worktrees, and local developer tooling (HYGIENE-01/04/05, SEC-03)."""
    sdist = find_built_sdist()
    _assert_no_local_state(_sdist_members(sdist), sdist.name)


def test_tracked_notebook_sources_are_visible_to_git() -> None:
    """No tracked notebook or notebook source may match an ignore rule: edits
    and new sources must show up in ``git status`` (HYGIENE-04)."""
    listed = subprocess.run(
        ["git", "ls-files", "notebooks/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in listed.stdout.splitlines() if line]
    assert tracked, "expected tracked notebook files"

    checked = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v", *tracked],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode != 0, (
        "tracked notebook sources must not be ignored; offending rules:\n" + checked.stdout
    )
