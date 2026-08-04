"""DATA_SOURCES.md is the human-readable provenance twin of the manifest.

The remediation design (Section 6) makes both records a contract: the
machine-readable ``snapshot_manifest.toml`` and ``DATA_SOURCES.md`` together
record observation/retrieval dates, publisher, primary URL, raw field meaning
and units, transformation, licence/redistribution status, the
public/constructed/illustrative classification, and known limitations for
every packaged dataset. This file pins the document's machine-readable
structure so a dataset cannot be added, relabelled, or dropped in one place
only.

Document conventions (the contract this file guards):

- every packaged dataset has a section headed ``## <manifest-name>`` where the
  name matches the strict identifier grammar ``[a-z][a-z0-9_]*``;
- each section body starts with ``**Classification:** public|constructed|illustrative``;
- each section carries a ``**Licence/redistribution:**`` line whose first word
  is ``verified`` or ``unverified``;
- each section's ``**Primary URL:**`` line carries the manifest's
  ``primary_url`` verbatim (as a backticked or linked URL);
- no stale CME redistribution language or network-refresh instructions remain.

The tests in this file pin exactly the fields above: dataset coverage,
classification, licence status, and primary URL. The remaining manifest fields
(dates, columns, units, sha256) are pinned against the packaged CSV bytes by
``tests/market/test_snapshot_contents.py``; neither file alone pins everything,
and together they cover the full provenance record.
"""

from __future__ import annotations

import re
import tomllib
from importlib import resources
from pathlib import Path

from yieldcurve.market.snapshot import Snapshot

DOC = Path(__file__).resolve().parents[2] / "DATA_SOURCES.md"
DATASET_HEADING_RE = re.compile(r"^## ([a-z][a-z0-9_]*)\s*$", re.MULTILINE)
CLASSIFICATION_RE = re.compile(
    r"^\*\*Classification:\*\*\s*(public|constructed|illustrative)\s*$", re.MULTILINE
)
LICENCE_RE = re.compile(r"^\*\*Licence/redistribution:\*\*\s*(verified|unverified)\b", re.MULTILINE)
PRIMARY_URL_RE = re.compile(
    r"^- \*\*Primary URL:\*\*\s*(.+?)(?=^- \*\*|\n\*\*Licence/redistribution:|\Z)",
    re.MULTILINE | re.DOTALL,
)
BACKTICKED_OR_LINKED_URL_RE = re.compile(r"`([^`]+)`|\((https?://[^)\s]+)\)")

_STALE_TOKENS = ("CME", "Information License Agreement", "Open Item", "refresh")


def _packaged_dataset_names() -> set[str]:
    """The manifest's authoritative dataset list, without constructing a Snapshot."""
    resource = resources.files("yieldcurve.data").joinpath("snapshot_manifest.toml")
    with resource.open("rb") as handle:
        manifest = tomllib.load(handle)
    return set(manifest["datasets"])


def _sections() -> dict[str, str]:
    """Map every ``## <identifier>`` heading to its section body.

    Headings that do not match the strict identifier grammar (for example
    "Known gaps and exclusions") are prose headings and are ignored; a
    lowercase-identifier heading is a dataset section by definition.
    """
    text = DOC.read_text(encoding="utf-8")
    matches = list(DATASET_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[match.end() : end]
    return sections


def test_data_sources_document_covers_exactly_the_packaged_datasets() -> None:
    """No packaged dataset is missing from the document and no document
    dataset section is orphaned: the manifest is the authoritative list."""
    assert set(_sections()) == _packaged_dataset_names()


def test_classification_matches_the_manifest(snapshot: Snapshot) -> None:
    """The document labels every dataset with the same taxonomy as the
    manifest: public, constructed, or illustrative."""
    sections = _sections()
    for name, block in snapshot.manifest["datasets"].items():
        match = CLASSIFICATION_RE.search(sections[name])
        assert match is not None, f"dataset {name!r} lacks a **Classification:** line"
        assert match.group(1) == block["classification"], name


def test_every_dataset_records_an_honest_licence_status(snapshot: Snapshot) -> None:
    """Each section states verified or unverified redistribution status; an
    assertion is never dressed up as a verified fact."""
    sections = _sections()
    for name in snapshot.manifest["datasets"]:
        match = LICENCE_RE.search(sections[name])
        assert match is not None, f"dataset {name!r} lacks a **Licence/redistribution:** line"
        assert match.group(1) in {"verified", "unverified"}


def test_primary_url_matches_the_manifest(snapshot: Snapshot) -> None:
    """The manifest's ``primary_url`` appears verbatim in the document section's
    **Primary URL:** line for every dataset: the two records cannot disagree
    on where a dataset comes from or is constructed."""
    sections = _sections()
    for name, block in snapshot.manifest["datasets"].items():
        match = PRIMARY_URL_RE.search(sections[name])
        assert match is not None, f"dataset {name!r} lacks a **Primary URL:** line"
        quoted = [a or b for a, b in BACKTICKED_OR_LINKED_URL_RE.findall(match.group(1))]
        assert block["primary_url"] in quoted, name


def test_no_stale_cme_or_refresh_language_remains() -> None:
    """The stale CME redistribution narrative and the promised-but-absent
    Riksbank/ECB refresh instructions are gone (HYGIENE-06, MKT-15)."""
    text = DOC.read_text(encoding="utf-8").lower()
    for token in _STALE_TOKENS:
        assert token.lower() not in text, f"stale token {token!r} still in DATA_SOURCES.md"
