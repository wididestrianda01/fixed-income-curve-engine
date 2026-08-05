"""Plotly helpers. Presentation only — no number is computed here.

One shared chart template and one semantic palette replace isolated defaults.
Line dash and markers supplement colour, so series stay distinguishable in
monochrome and for colour-blind readers; every axis carries its units, set by
the tab that owns the number's meaning. Material charts pair these figures
with an expandable data table and a concise text result in the calling tab.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

__all__ = ["ADVERSE", "PALETTE", "bar_figure", "histogram_figure", "overlay_figure", "series_style"]

_HEIGHT = 420
_MARGIN = {"l": 60, "r": 20, "t": 40, "b": 50}
_TEMPLATE = "plotly_white"
_FONT = {"family": "sans serif"}

# Semantic palette. The first colour is the primary/canonical series; the
# second marks the adverse direction (losses, worst scenario). Later colours
# cycle through further series.
PALETTE: tuple[str, ...] = (
    "#1f4e79",
    "#c0392b",
    "#2e8b57",
    "#8e44ad",
    "#d35400",
    "#16a085",
    "#7f8c8d",
)
ADVERSE: str = PALETTE[1]
_PILLAR_COLOR = "#9e9e9e"

_LINE_DASHES: tuple[str, ...] = ("solid", "dash", "dot", "longdash")
_MARKERS: tuple[str, ...] = ("circle", "square", "diamond", "triangle-up")


def series_style(index: int) -> dict[str, str]:
    """Colour plus line dash plus marker symbol for series ``index``.

    Colour is never the only encoding: dash and marker pattern repeat in the
    same cycle, so adjacent series stay distinct when printed in monochrome.
    """
    return {
        "color": PALETTE[index % len(PALETTE)],
        "dash": _LINE_DASHES[index % len(_LINE_DASHES)],
        "symbol": _MARKERS[index % len(_MARKERS)],
    }


def _layout(figure: go.Figure, *, y_title: str, x_title: str = "Maturity (years)") -> go.Figure:
    figure.update_layout(
        template=_TEMPLATE,
        font=_FONT,
        height=_HEIGHT,
        margin=_MARGIN,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return figure


def overlay_figure(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    y_title: str,
    x_title: str = "Maturity (years)",
    pillar_times: Sequence[float] | None = None,
) -> go.Figure:
    """Several curves on one pair of axes, y values passed through unchanged.

    ``pillar_times`` draws dotted vertical lines at the calibration pillars and
    adds a legend entry naming them, so the canonical input knots are visually
    distinguished from the interpolation overlay lines.
    """
    figure = go.Figure()
    for index, (name, (x, y)) in enumerate(series.items()):
        style = series_style(index)
        figure.add_trace(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                name=name,
                line={"color": style["color"], "dash": style["dash"]},
            )
        )
    if pillar_times:
        for t in pillar_times:
            figure.add_vline(x=t, line_dash="dot", line_color=_PILLAR_COLOR, opacity=0.8)
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Calibration pillar",
                line={"color": _PILLAR_COLOR, "dash": "dot"},
            )
        )
    return _layout(figure, y_title=y_title, x_title=x_title)


def bar_figure(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    y_title: str,
    x_title: str = "",
    text_format: str | None = None,
) -> go.Figure:
    """A categorical bar chart — key-rate ladders, ΔEVE by scenario.

    ``text_format`` prints the value above each bar, so the number is readable
    without the hover tooltip (text supplements colour).
    """
    trace = go.Bar(x=list(labels), y=list(values), marker_color=PALETTE[0])
    if text_format is not None:
        trace.text = [text_format.format(float(v)) for v in values]
        trace.textposition = "outside"
        trace.cliponaxis = False
    figure = go.Figure(trace)
    return _layout(figure, y_title=y_title, x_title=x_title)


def histogram_figure(
    values: npt.NDArray[np.float64], *, markers: Mapping[str, float], x_title: str
) -> go.Figure:
    """A P&L distribution with vertical markers for VaR and expected shortfall."""
    figure = go.Figure(
        go.Histogram(x=np.asarray(values, dtype=float), nbinsx=60, marker_color=PALETTE[0])
    )
    for index, (name, position) in enumerate(markers.items()):
        style = series_style(index + 1)
        figure.add_vline(
            x=position,
            line_dash=style["dash"],
            line_color=style["color"],
            annotation_text=name,
            annotation_position="top",
        )
    return _layout(figure, y_title="Observations", x_title=x_title)
