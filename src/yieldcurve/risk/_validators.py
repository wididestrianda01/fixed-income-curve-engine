"""Shared private validation for the risk analytics modules.

``sensitivities.py`` and ``keyrate.py`` enforce the same two contracts — a bump
must be a positive finite rate, and a normalized measure must not divide by a
materially zero present value — so the helpers live here once.

The scale contract (``instrument_scale``) and near-zero threshold
(``MIN_UNIT_PRICE``) they depend on stay in ``sensitivities.py`` and are
imported lazily inside :func:`_require_unit_price`: that module imports these
helpers, so a module-level import here would be circular.
"""

from __future__ import annotations

import math

from yieldcurve.instruments import Instrument


def _require_bump(bump: float, measure: str) -> None:
    if not math.isfinite(bump) or bump <= 0.0:
        raise ValueError(f"{measure} bump must be a positive finite rate, got {bump}")


def _require_unit_price(base: float, instrument: Instrument, measure: str) -> None:
    """Reject normalizing by a materially zero present value (error policy:
    the code must not normalize by near-zero PV)."""
    # Imported here, not at module level, to break the import cycle with
    # sensitivities.py, which imports these helpers itself.
    from yieldcurve.risk.sensitivities import MIN_UNIT_PRICE, instrument_scale

    if abs(base / instrument_scale(instrument)) < MIN_UNIT_PRICE:
        raise ValueError(
            f"{measure} is undefined for {type(instrument).__name__} with base price "
            f"{base:.6g}: materially zero present value; "
            "use bucket_exposure for the monetary sensitivity"
        )
