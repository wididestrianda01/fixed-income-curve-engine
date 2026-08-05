"""Unit tests for the shared chart template and semantic palette.

AppTest does not expose the serialized Plotly figures, so the presentation
contract — axis units, colour + dash + marker supplement, pillar markers,
value text on bars — is pinned directly on the figure objects the tabs render.
"""

from __future__ import annotations

from app.charts import (
    ADVERSE,
    PALETTE,
    bar_figure,
    histogram_figure,
    overlay_figure,
    series_style,
)


def test_series_style_cycles_colour_dash_and_marker() -> None:
    first, second = series_style(0), series_style(1)
    assert first["color"] == PALETTE[0]
    assert PALETTE[1] == ADVERSE
    # colour is never the only encoding: dash and marker repeat in the same cycle
    assert first["color"] != second["color"]
    assert first["dash"] != second["dash"]
    assert first["symbol"] != second["symbol"]


def test_overlay_figure_uses_the_shared_template_and_distinct_styles() -> None:
    figure = overlay_figure(
        {"A": ([0.0, 1.0], [1.0, 2.0]), "B": ([0.0, 1.0], [2.0, 3.0])},
        y_title="Zero rate (%)",
    )
    assert figure.layout.font.family == "sans serif"
    assert figure.layout.yaxis.title.text == "Zero rate (%)"
    assert figure.layout.xaxis.title.text == "Maturity (years)"
    colors = [t.line.color for t in figure.data]
    dashes = [t.line.dash for t in figure.data]
    assert colors[0] != colors[1]
    assert dashes[0] != dashes[1]


def test_overlay_figure_marks_calibration_pillars() -> None:
    figure = overlay_figure(
        {"A": ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])},
        y_title="Zero rate (%)",
        pillar_times=[0.5, 2.0],
    )
    vlines = [s for s in figure.layout.shapes if s.type == "line"]
    assert len(vlines) == 2
    names = [t.name for t in figure.data]
    assert "Calibration pillar" in names


def test_bar_figure_carries_units_and_optional_value_text() -> None:
    figure = bar_figure(["A", "B"], [1.0, 2.0], y_title="Illustrative ΔEVE (SEK)")
    assert figure.layout.yaxis.title.text == "Illustrative ΔEVE (SEK)"
    assert figure.data[0].text is None
    figure = bar_figure(
        ["A", "B"], [1.0, 2.0], y_title="Illustrative ΔEVE (SEK)", text_format="{:,.0f}"
    )
    assert tuple(figure.data[0].text) == ("1", "2")
    assert figure.data[0].textposition == "outside"


def test_histogram_markers_supplement_colour_with_dash() -> None:
    import numpy as np

    figure = histogram_figure(
        np.asarray([-1.0, 0.0, 1.0], dtype=float),
        markers={"VaR": -0.9, "ES": -0.7},
        x_title="Daily P&L (SEK)",
    )
    assert figure.layout.xaxis.title.text == "Daily P&L (SEK)"
    vlines = [s for s in figure.layout.shapes if s.type == "line"]
    assert len(vlines) == 2
    assert {s.line.dash for s in vlines} == {"dash", "dot"}
    annotations = [a.text for a in figure.layout.annotations if a.text]
    assert {"VaR", "ES"} <= set(annotations)
