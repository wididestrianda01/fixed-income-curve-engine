"""Market data: the frozen packaged snapshot, read-only and fully offline.

Importing anything here must never perform network I/O. The default
:class:`Snapshot` loads packaged read-only resources through
``importlib.resources``, so a source checkout and an installed wheel behave
identically. The broken ECB, Riksbank, Riksgalden, and FRED HTTP adapters and
the ``refresh`` CLI were removed; there is no refresh path.
"""

from yieldcurve.market.snapshot import Snapshot

__all__ = ["Snapshot"]
