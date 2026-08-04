"""Market data adapters and the packaged snapshot layer.

Importing anything here must never perform network I/O. The default snapshot is
packaged read-only resources loaded through ``importlib.resources``; the
network refresh CLI (``yieldcurve.market.refresh``) is the only module that
touches the network, and only when run by hand.
"""
