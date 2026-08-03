"""Plotly helpers. Presentation only — no number is computed here."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

__all__ = ["bar_figure", "histogram_figure", "overlay_figure", "zero_curve_figure"]

_HEIGHT = 420
_MARGIN = {"l": 60, "r": 20, "t": 40, "b": 50}


def _layout(figure: go.Figure, *, y_title: str, x_title: str = "Maturity (years)") -> go.Figure:
    figure.update_layout(
        height=_HEIGHT,
        margin=_MARGIN,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return figure


def zero_curve_figure(times: Sequence[float], zeros: Sequence[float], *, label: str) -> go.Figure:
    """A single zero curve, rates on the y-axis in percent."""
    figure = go.Figure(
        go.Scatter(
            x=list(times),
            y=[z * 100.0 for z in zeros],
            mode="lines",
            name=label,
        )
    )
    return _layout(figure, y_title="Zero rate (%)")


def overlay_figure(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]], *, y_title: str
) -> go.Figure:
    """Several curves on one pair of axes, y values passed through unchanged."""
    figure = go.Figure()
    for name, (x, y) in series.items():
        figure.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", name=name))
    return _layout(figure, y_title=y_title)


def bar_figure(
    labels: Sequence[str], values: Sequence[float], *, y_title: str, x_title: str = ""
) -> go.Figure:
    """A categorical bar chart — key-rate ladders, ΔEVE by scenario."""
    figure = go.Figure(go.Bar(x=list(labels), y=list(values)))
    return _layout(figure, y_title=y_title, x_title=x_title)


def histogram_figure(
    values: npt.NDArray[np.float64], *, markers: Mapping[str, float], x_title: str
) -> go.Figure:
    """A P&L distribution with vertical markers for VaR and expected shortfall."""
    figure = go.Figure(go.Histogram(x=np.asarray(values, dtype=float), nbinsx=60))
    for name, position in markers.items():
        figure.add_vline(
            x=position,
            line_dash="dash",
            annotation_text=name,
            annotation_position="top",
        )
    return _layout(figure, y_title="Observations", x_title=x_title)
