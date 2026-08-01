"""Convert a percent-format ``.py`` source file into an executed ``.ipynb``.

Notebooks in this repository are authored as plain Python so they are
reviewable in a diff, then executed once so their outputs can be committed
(Phase 6 rule 3: GitHub renders outputs, it does not run kernels).

Cell markers::

    # %% [markdown]
    # Prose lines, each prefixed with "# ".

    # %%
    code_lines()

Usage::

    .venv/bin/python scripts/topynb.py notebooks/src/05-risk.py notebooks/05-risk.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

MARKDOWN_MARKER = "# %% [markdown]"
CODE_MARKER = "# %%"


def _split_cells(text: str) -> list[tuple[str, str]]:
    """Return ``(cell_type, source)`` pairs in file order."""
    cells: list[tuple[str, str]] = []
    kind = "code"
    buffer: list[str] = []

    def flush() -> None:
        source = "\n".join(buffer).strip("\n")
        if source.strip():
            cells.append((kind, source))

    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped in (MARKDOWN_MARKER, CODE_MARKER):
            flush()
            buffer = []
            kind = "markdown" if stripped == MARKDOWN_MARKER else "code"
            continue
        buffer.append(line)
    flush()
    return cells


def _uncomment(source: str) -> str:
    """Strip the leading ``# `` from every line of a markdown cell."""
    out = []
    for line in source.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def build(src: Path, dst: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}

    for index, (kind, source) in enumerate(_split_cells(src.read_text(encoding="utf-8"))):
        cell = (
            nbformat.v4.new_markdown_cell(_uncomment(source))
            if kind == "markdown"
            else nbformat.v4.new_code_cell(source)
        )
        # Deterministic ids: a fresh uuid per run would make every rebuild a diff.
        cell["id"] = f"cell{index:03d}"
        notebook.cells.append(cell)

    NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(dst.parent)}},
    ).execute()
    nbformat.write(notebook, str(dst))


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
