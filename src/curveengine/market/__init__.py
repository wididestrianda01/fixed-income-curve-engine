"""Market data adapters and the dated snapshot layer.

Importing anything here must never perform network I/O. Adapters fetch only when
their ``fetch_*`` function is called explicitly, which happens in
``curveengine.market.refresh`` and nowhere else.
"""
