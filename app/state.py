"""The frozen selection the sidebar produces, and nothing else."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.market.snapshot import Snapshot

__all__ = ["AppState"]


@dataclass(frozen=True)
class AppState:
    """What every tab is allowed to know about the user's choices.

    There is no currency selector: the snapshot has SEK government data and USD swap data,
    and which one a tab uses is a property of the question that tab asks, not of a control.
    """

    snapshot: Snapshot
    asof: date
    method: InterpMethod
