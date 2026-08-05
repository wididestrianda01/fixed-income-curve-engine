"""The frozen selection the sidebar produces, and nothing else."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from yieldcurve.curves.interpolation import InterpMethod

__all__ = ["AppState"]


@dataclass(frozen=True)
class AppState:
    """What every tab is allowed to know about the user's choices.

    The snapshot is deliberately not part of the state: every tab reaches it through the
    cached ``app.data.load_snapshot`` bridge, and the sidebar validates it before any tab
    renders. There is no currency selector: the snapshot has SEK government data and USD
    swap data, and which one a tab uses is a property of the question that tab asks, not
    of a control.
    """

    asof: date
    method: InterpMethod
