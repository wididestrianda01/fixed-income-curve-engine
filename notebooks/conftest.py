"""Offline notebook execution under ``pytest --nbmake``.

``nbmake`` executes each notebook in a separate ipykernel process, so a
pytest-side socket patch cannot reach the notebook code. This conftest puts
``_netblock/`` — a ``sitecustomize`` module that, at interpreter startup,
refuses outbound connection attempts (``connect``/``connect_ex``/``sendto``/
``sendmsg``) and name resolution (the ``socket`` DNS functions) — on
``PYTHONPATH`` before any kernel is spawned. Kernels inherit the parent
environment, and the ipykernel's own client communication runs over pyzmq,
which does not use the Python ``socket`` module, so kernels still start and
run while notebook code is guaranteed offline (TQ-09). The same mechanism
must be active when notebooks are regenerated from ``notebooks/src/*.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

_NETBLOCK = Path(__file__).parent / "_netblock"


def _register_netblock() -> None:
    parts = [str(_NETBLOCK)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


_register_netblock()
