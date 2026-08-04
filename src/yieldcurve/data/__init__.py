"""Packaged, read-only frozen market-data snapshot resources.

The CSVs and ``snapshot_manifest.toml`` here are the one audited snapshot,
dated 2026-07-24. They are loaded through ``importlib.resources`` by
:mod:`yieldcurve.market.snapshot`, so a source checkout and an installed wheel
see identical data with no network access and no checkout-path fallback.
"""
