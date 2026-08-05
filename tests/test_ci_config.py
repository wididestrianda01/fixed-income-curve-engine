"""CI workflow invariants (Task 14 review): SHA pins, coverage isolation, frozen installs.

Deliberately a line-based parser: PyYAML is only a transitive dependency (via
pre-commit), not a declared project dependency. It understands this workflow's
flat two-space-indent job/step layout; nested step blocks would need parser
extensions, not weaker assertions.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
USES = re.compile(r"^\s*- uses:\s*([^@\s]+)@([0-9a-f]{40})\s*(?:#.*)?$")
JOB = re.compile(r"^  ([a-z0-9-]+):\s*$")


def _lines() -> list[str]:
    return WORKFLOW.read_text(encoding="utf-8").splitlines()


def _job_of(lines: list[str], index: int) -> str:
    for i in range(index, -1, -1):
        if m := JOB.match(lines[i]):
            return m.group(1)
    return "<top>"


def test_every_uses_is_a_full_sha_with_a_version_comment() -> None:
    lines = _lines()
    uses = [(i, USES.match(line)) for i, line in enumerate(lines) if "- uses:" in line]
    assert uses, "no pinned action found in ci.yml"
    for i, m in uses:
        assert m is not None, f"line {i}: unpinned action ref {lines[i].strip()}"
        assert lines[i - 1].strip().startswith("#"), (
            f"line {i}: uses lacks version-comment convention"
        )


def test_cov_flags_live_only_in_package_test_job() -> None:
    lines = _lines()
    for i, line in enumerate(lines):
        if "--cov" in line:
            assert _job_of(lines, i) == "package-test", f"line {i}: --cov outside package-test"


def test_every_install_is_the_frozen_lockfile_command() -> None:
    for i, line in enumerate(_lines()):
        if "uv sync" in line:
            assert line.strip().startswith("run: "), f"line {i}: uv sync outside a run step"
            assert "uv sync --frozen --extra dev --extra app --extra notebooks" in line, (
                f"line {i}: install command drifted from the frozen lockfile"
            )


def test_permissions_read_and_checkouts_do_not_persist_credentials() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^permissions:\s*\n\s+contents:\s*read\b", text, re.M)
    for i, line in enumerate(text.splitlines()):
        if "actions/checkout@" in line:
            step = "\n".join(text.splitlines()[i : i + 5])
            assert "persist-credentials: false" in step, f"line {i}: checkout persists credentials"


def test_no_continue_on_error_or_allow_failure() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error" not in text
    assert "allow-failure" not in text
