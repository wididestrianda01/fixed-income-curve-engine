"""Offline notebook execution: refuse every outbound network connection.

``nbmake`` (and the source-to-notebook regeneration path) executes each
notebook in a separate ipykernel process, so a pytest-side ``socket`` patch can
never reach it. Python imports ``sitecustomize`` automatically at interpreter
startup, so putting this directory on ``PYTHONPATH`` before the kernel is
spawned installs the block inside the kernel before any notebook cell runs.

Why connections are refused rather than sockets created: the ipykernel and its
client stack initialize asyncio (whose event loop creates a ``socketpair``)
and jupyter_client (which creates a throwaway socket to find a free port)
after interpreter startup, so forbidding socket creation would break the
kernel itself. Forbidding outbound communication is the actual guarantee the
design demands — no notebook code can reach the network — and it lets the
kernel run. ``ssl`` is imported first because it subclasses
``socket.socket`` at import time.

The block mirrors the in-process fixture in ``tests/test_offline.py``
(``patch("socket.socket", side_effect=RuntimeError(...))``): the fixture can
forbid creation because ``ssl`` and asyncio are already initialized in the
pytest process; the kernel starts fresh, so the kernel-side block refuses
connections instead. Both raise ``RuntimeError("network call blocked ...")``.

``notebooks/conftest.py`` puts this directory on the kernel's ``PYTHONPATH``;
the same mechanism is used when regenerating the committed notebooks from
``notebooks/src/*.py``.
"""

from __future__ import annotations

import socket
import ssl  # noqa: F401  # imported before the swap: ssl subclasses socket.socket


class _BlockedSocket(socket.socket):
    """A socket that can be created but can never open an outbound connection."""

    def connect(self, address: object) -> None:
        raise RuntimeError("network call blocked (offline notebook execution)")

    def connect_ex(self, address: object) -> int:
        raise RuntimeError("network call blocked (offline notebook execution)")

    def sendto(self, *args: object, **kwargs: object) -> int:
        raise RuntimeError("network call blocked (offline notebook execution)")

    def sendmsg(self, *args: object, **kwargs: object) -> int:
        raise RuntimeError("network call blocked (offline notebook execution)")


def _blocked_getaddrinfo(*args: object, **kwargs: object) -> object:
    """Refuse name resolution so no DNS query can leak either."""
    raise RuntimeError("network call blocked (offline notebook execution)")


socket.socket = _BlockedSocket  # type: ignore[assignment]
socket.getaddrinfo = _blocked_getaddrinfo  # type: ignore[assignment]
