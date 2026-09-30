"""One dark plotly theme for every chart in the app.

Colors follow a fixed role system so charts read as one family:

- ``SERIES``: identity colors, used in this fixed order and never cycled.
  Slot 1 (blue) is "this split", slot 2 (orange) is "everything else".
- ``UP`` / ``DOWN``: above / below the baseline. Always paired with a sign or
  arrow in the text, so the meaning never rests on color alone.
- Ink and chrome: text, gridlines and axes stay quiet so the data is the loud part.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
NEUTRAL = "#383835"

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
UP = "#0ca30c"
DOWN = "#d03b3b"
WARN = "#fab219"

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
TEMPLATE_NAME = "nbalab_dark"


def _template() -> go.layout.Template:
    axis = dict(
        gridcolor=GRID, gridwidth=1, linecolor=AXIS, zerolinecolor=AXIS, zerolinewidth=1,
        tickfont=dict(color=MUTED, size=12), title=dict(font=dict(color=INK_2, size=13)),
        showline=True, ticks="",
    )
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(family=FONT, color=INK_2, size=13),
            title=dict(font=dict(color=INK, size=16), x=0.0, xanchor="left"),
            colorway=SERIES,
            xaxis=axis,
            yaxis=axis,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2), orientation="h",
                        yanchor="bottom", y=1.02, xanchor="left", x=0),
            hoverlabel=dict(bgcolor="#262624", bordercolor=AXIS, font=dict(color=INK, family=FONT)),
            margin=dict(l=16, r=24, t=56, b=40),
            bargap=0.35,
        )
    )


def register() -> str:
    """Register the theme with plotly and make it the default. Returns its name."""
    pio.templates[TEMPLATE_NAME] = _template()
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


def signed_color(value: float) -> str:
    """UP for positive, DOWN for negative, muted gray for zero or NaN."""
    if value != value or value == 0:  # NaN check without numpy
        return MUTED
    return UP if value > 0 else DOWN


register()
