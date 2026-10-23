"""The Factor attribution section of a backtest's report page.

``factor_attribution_section`` draws a run's factor attribution from its
metrics block (``quantlab.risk.attribution.attribution_summary`` per segment)
and its per-bar store (``factor_attribution.zarr``) only: this module imports
no risk model. It reads, on the store's ``factor`` axis, each factor's
``group`` (country, industry, style) and display ``label`` (the factor name
when the store has none).

The section opens with six tiles (annualized log growth, the part from the
factors, risk-free plus trading, forecast and realized volatility, coverage)
and then sets return and risk side by side, row by row: by part (bars from
the zero axis), over time, the styles and the industries; the styles'
exposure over time and each part's return against its realized risk close
it. Every tile and chart carries a plain-English explanation in its
``title``, which the page shows on hover.

Examples
--------
>>> html = factor_attribution_section(
...     metrics["factor_attribution"], run.factor_attribution(), out_of_sample=False
... )
>>> "Return by part" in html
True
"""

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

__all__ = ["factor_attribution_section"]

#: The page's chart chrome, shared with ``quantlab.runs.backtest_report``.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CHART_FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
POSITIVE, NEGATIVE, MIDPOINT, TOTAL = "#2a78d6", "#e34948", "#f0efec", "#52514e"
LIGHT = "#86b6ef"

#: The parts a bar's return is split into, in their fixed colour order:
#: ``(key, label, colour)``; the first three are factor groups.
PARTS = (
    ("country", "Country", "#2a78d6"),
    ("industry", "Industry", "#eb6834"),
    ("style", "Style", "#1baf7a"),
    ("specific", "Specific", "#eda100"),
    ("uncovered", "Uncovered", "#e87ba4"),
    ("risk_free", "Risk-free", "#008300"),
    ("trading", "Trading", "#4a3aa7"),
)
_GROUPS = PARTS[:3]

#: How many industries each industry chart shows at the top and at the bottom.
INDUSTRIES_SHOWN = 10

#: Explanations shown on hover.
_TILE_TIPS = {
    "total": "The portfolio's growth per year in log terms. The factor attribution splits exactly "
             "this number into its sources.",
    "factor": "The part of the return explained by the portfolio's exposures to the risk model's "
              "factors: the market, industries and styles. Specific is the rest of the stock returns.",
    "rf_trading": "Interest on the invested money (the risk-free rate) plus everything trading changed: "
                  "fills at the open, fees, slippage and idle cash.",
    "forecast": "The volatility the risk model predicted for the portfolio held each day, averaged and "
                "annualized. The note is the factor part's share of it, squared (variance terms).",
    "realized": "The volatility the portfolio actually had, annualized. Compare it with the forecast "
                "to judge the risk model.",
    "coverage": "The share of the portfolio's holdings the risk model knows (has exposures for), on "
                "average. The rest of the return lands in Uncovered.",
}
_CHART_TIPS = {
    "return_by_part": "Each bar is how much one source added to (blue) or took from (red) the yearly log "
                      "growth; the bars add up to the grey Total. Country is the market, Industry and "
                      "Style are the portfolio's tilts, Specific is the stock picking the factors cannot "
                      "explain, Trading is fills, fees and idle cash.",
    "risk_by_part": "Each bar is how much one source adds to the predicted volatility (exposure x "
                    "volatility x correlation); they add up to the grey Forecast. A red bar hedges the "
                    "rest. The black bar is the volatility that actually happened.",
    "return_over_time": "The running total of each source over time. The black line is the portfolio "
                        "itself (log value): the coloured lines always add up to it.",
    "risk_over_time": "The predicted volatility each month, split by source (the bars add up to the "
                      "prediction), against the volatility that actually happened over the last 63 "
                      "bars (dotted).",
    "style_return": "Left: how strongly the portfolio leaned into each style (positive means more of it "
                    "than the market). Right: what that lean earned or lost per year.",
    "style_risk": "How much each style added to the predicted volatility (left) and to the volatility "
                  "that actually happened (right), in the same order as the chart beside it.",
    "industry_return": "The industries that added the most and those that cost the most per year. The "
                       "grey number beside each name is the portfolio's average net weight in it.",
    "industry_risk": "The industries adding the most to the predicted volatility and those reducing it "
                     "the most (hedging). The grey number is the average net weight.",
    "style_heatmap": "Each row is a style, each column a week. Blue means the portfolio leaned into the "
                     "style, red away from it; the stronger the colour, the bigger the lean.",
    "return_vs_risk": "For each source, what it earned per year (dark) against how much of the actual "
                      "volatility it caused (light). A long light bar with a short dark one is risk "
                      "taken without being paid.",
}


def factor_attribution_section(
    block: dict,
    attribution: xr.Dataset,
    *,
    out_of_sample: bool,
    bars_per_year: float | None = None,
    in_sample_range: tuple[str, str] | None = None,
) -> str:
    """Return the Factor attribution section's html: tiles, then return and risk side by side.

    Parameters
    ----------
    block : dict
        The run's ``factor_attribution`` metrics block (``whole`` and, for
        a model run, ``in_sample`` and ``out_of_sample``).
    attribution : xarray.Dataset
        The run's per-bar factor attribution (``factor_attribution.zarr``).
    out_of_sample : bool
        Whether the run has an in-sample part, so the tiles and the charts
        summarize the ``out_of_sample`` segment rather than ``whole``, as
        the page's headline numbers do.
    bars_per_year : float, optional
        Bars in a year, to annualize the per-bar forecasts drawn over time;
        252 when not given.
    in_sample_range : tuple of str, optional
        ``(first, last)`` bar labels of the in-sample part, shaded on the
        charts over time (which span the whole window).

    Returns
    -------
    str
        The section's html: plotly fragments reusing the page's plotly.js,
        the rest escaped; a note alone when the block has no segment or the
        store no bar.

    Examples
    --------
    >>> html = factor_attribution_section(block, attribution, out_of_sample=True)
    >>> '"name":"return_by_part"' in html, '"name":"risk_by_part"' in html
    (True, True)
    """
    name, segment = ("out-of-sample", block.get("out_of_sample")) if out_of_sample else (
        "whole window", block.get("whole"))
    if not isinstance(segment, dict):
        name, segment = "whole window", block.get("whole")
    if not isinstance(segment, dict) or attribution.sizes.get("timestamp", 0) == 0:
        return '<p class="note">No factor attribution to show: the run has no attributed bar.</p>'
    scale = float(np.sqrt(bars_per_year or 252))
    labels = _labels(attribution)
    groups = _groups(attribution)
    over = "over the whole window" + ("; grey: in-sample" if in_sample_range else "")
    rows = [
        (_card("Return by part", f"Annualized log growth, {name}; the bars add up to the total.",
               "return_by_part", _return_by_part(segment, groups)),
         _card("Risk by part", f"Contribution to the forecast volatility, and the realized volatility, {name}.",
               "risk_by_part", _risk_by_part(segment, groups))),
        (_card("Return over time", f"Cumulative log contribution of each part, {over}; black: log NAV.",
               "return_over_time", _shaded(_return_over_time(attribution, groups), in_sample_range)),
         _card("Risk over time", f"Forecast volatility by part, monthly mean, {over}; dotted: realized "
               "63-bar volatility.", "risk_over_time",
               _shaded(_risk_over_time(attribution, groups, scale), in_sample_range))),
        (_card("Styles: exposure and return", f"Mean net exposure and annualized contribution, {name}.",
               "style_return", _style_return(segment, attribution, labels)),
         _card("Styles: risk", f"Contribution to forecast and to realized volatility, {name}, same order.",
               "style_risk", _style_risk(segment, attribution, labels))),
        (_card("Industries: return", f"Best and worst {INDUSTRIES_SHOWN} by annualized contribution, {name}; "
               "grey: mean net weight over the whole window.", "industry_return",
               _industries(segment, attribution, labels, risk=False)),
         _card("Industries: risk", f"Largest and smallest {INDUSTRIES_SHOWN} contributions to forecast "
               f"volatility, {name}.", "industry_risk", _industries(segment, attribution, labels, risk=True))),
    ]
    body = "".join(
        f'<div class="pair">{left}{right}</div>' if left and right else left + right for left, right in rows
    )
    return (
        '<div class="fa">'
        + _tiles(segment, name)
        + '<div class="pair heads"><div class="colhead">Return</div><div class="colhead">Risk</div></div>'
        + body
        + _card("Style exposure over time", f"Weekly mean net exposure, {over}; blue long, red short.",
                "style_heatmap", _shaded(_style_heatmap(segment, attribution, labels), in_sample_range))
        + _card("Return against realized risk",
                f"Each part's annualized log growth and its contribution to realized volatility, {name}.",
                "return_vs_risk", _return_vs_risk(segment, groups))
        + _coverage_note(segment)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# page pieces
# ---------------------------------------------------------------------------


def _card(title: str, subtitle: str, tip: str, figure: go.Figure | None) -> str:
    """A chart in a card titled ``title``, its explanation on hover; nothing without a chart."""
    if figure is None:
        return ""
    body = figure.to_html(full_html=False, include_plotlyjs=False,
                          config={"responsive": True, "displaylogo": False})
    return (
        f'<div class="card"><h3 title="{html.escape(_CHART_TIPS[tip])}">{html.escape(title)} '
        f'<span class="i">ⓘ</span></h3><p class="sub">{html.escape(subtitle)}</p>{body}</div>'
    )


def _tiles(segment: dict, name: str) -> str:
    growth = segment["annualized_log_return"]
    volatility = segment["ex_ante_risk"]["volatility"]
    realized = segment["ex_post_risk"]["volatility"]
    coverage = segment["coverage"]

    def pct(value, signed=True):
        if not _finite(value):
            return "—"
        return f"{100 * value:+.2f}%" if signed else f"{100 * value:.2f}%"

    total, factor = volatility.get("total"), volatility.get("factor")
    share = factor**2 / total**2 if _finite(total) and _finite(factor) and total > 0 else None
    side = (growth.get("risk_free") + growth.get("trading")
            if _finite(growth.get("risk_free")) and _finite(growth.get("trading")) else None)
    least = coverage.get("min_covered_weight")
    tiles = [
        ("total", "Log growth / yr", pct(growth.get("total")), name),
        ("factor", "From factors", pct(growth.get("factor")), f"specific {pct(growth.get('specific'))}"),
        ("rf_trading", "Risk-free + trading", pct(side), f"uncovered {pct(growth.get('uncovered'))}"),
        ("forecast", "Forecast vol", pct(total, signed=False),
         "" if share is None else f"{share:.0%} of it from factors"),
        ("realized", "Realized vol", pct(realized, signed=False), "ex post"),
        ("coverage", "Coverage", pct(coverage.get("mean_covered_weight"), signed=False),
         f"min {least:.1%}" if _finite(least) else ""),
    ]
    return '<div class="tiles">' + "".join(
        f'<div class="tile" title="{html.escape(_TILE_TIPS[key])}"><div class="kl">{html.escape(label)}</div>'
        f'<div class="kv">{html.escape(value)}</div><div class="ks">{html.escape(note)}</div></div>'
        for key, label, value, note in tiles
    ) + "</div>"


def _coverage_note(segment: dict) -> str:
    note = (segment.get("coverage") or {}).get("note")
    return f'<p class="note">{html.escape(note)}</p>' if note else ""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _labels(attribution: xr.Dataset) -> dict[str, str]:
    """Each factor's display name: the store's ``label`` coordinate, else the factor name."""
    factors = [str(factor) for factor in attribution["factor"].values]
    if "label" in attribution.coords:
        return dict(zip(factors, (str(label) for label in attribution["label"].values)))
    return {factor: factor for factor in factors}


def _finite(value) -> bool:
    return value is not None and bool(np.isfinite(value))


def _groups(attribution: xr.Dataset) -> tuple:
    """The entries of ``_GROUPS`` the risk model has a factor in."""
    present = {str(group) for group in attribution["group"].values}
    return tuple(entry for entry in _GROUPS if entry[0] in present)


def _shaded(fig: go.Figure | None, in_sample_range) -> go.Figure | None:
    """Shade the in-sample range on a chart over time."""
    if fig is not None and in_sample_range is not None:
        fig.add_vrect(x0=in_sample_range[0], x1=in_sample_range[1], fillcolor="grey", opacity=0.15,
                      line_width=0, layer="below")
    return fig


def _members(attribution: xr.Dataset, group: str) -> list[str]:
    return [str(f) for f, g in zip(attribution["factor"].values, attribution["group"].values) if g == group]


def _parts(attribution: xr.Dataset, groups: tuple) -> dict[str, pd.Series]:
    """Each part's per-bar log contribution: the groups the model has, then the terms."""
    group = attribution["group"].values
    factor_log = attribution["factor_log_contribution"].values
    index = pd.DatetimeIndex(attribution["timestamp"].values)
    out = {key: pd.Series(factor_log[:, group == key].sum(axis=1), index=index) for key, _, _ in groups}
    for key, _, _ in PARTS[3:]:
        out[key] = pd.Series(attribution["log_contribution"].sel(term=key).values, index=index)
    return out


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def _figure(height: int, rows: int = 1, cols: int = 1, titles=None) -> go.Figure:
    fig = go.Figure() if rows == cols == 1 else make_subplots(
        rows=rows, cols=cols, shared_yaxes=True, horizontal_spacing=0.05, subplot_titles=titles)
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30 if titles else 10, b=10),
        paper_bgcolor="#fff", plot_bgcolor="#fff", showlegend=False,
        font=dict(family=CHART_FONT, size=12, color=INK2),
        hoverlabel=dict(bgcolor="#fff", bordercolor=GRID, font=dict(color=INK)),
        legend=dict(orientation="h", x=0, y=1.0, xanchor="left", yanchor="bottom", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_annotations(font=dict(size=11, color=MUTED))
    return fig


def _padded(values, pad: float = 1.45) -> list[float]:
    """An axis range holding the bars and the labels outside them, with 0 inside."""
    finite = [v for v in values if _finite(v)] or [0.0]
    low, high = min(min(finite), 0.0), max(max(finite), 0.0)
    span = (high - low) or 1.0
    return [low - span * (pad - 1) if low < 0 else -span * 0.05,
            high + span * (pad - 1) if high > 0 else span * 0.05]


def _signed(values) -> list[str]:
    return [POSITIVE if (v or 0) >= 0 else NEGATIVE for v in values]


def _parts_of(groups: tuple) -> tuple:
    """The groups the model has, then the four terms."""
    return groups + PARTS[3:]


def _zero(values) -> list[float]:
    return [float(v) if _finite(v) else 0.0 for v in values]


def _kept(values) -> list[float | None]:
    """Finite values as floats, the rest None: no bar is drawn and the label is a dash."""
    return [float(v) if _finite(v) else None for v in values]


def _labelled(values, fmt) -> list[str]:
    return [fmt(v) if v is not None else "—" for v in values]


def _return_by_part(segment: dict, groups: tuple) -> go.Figure:
    group_growth, terms = segment["group_annualized_log_return"], segment["annualized_log_return"]
    parts = _parts_of(groups)
    values = _kept([group_growth.get(key) if index < len(groups) else terms.get(key)
                    for index, (key, _, _) in enumerate(parts)] + [terms.get("total")])
    fig = _figure(330)
    fig.add_trace(go.Bar(
        x=[label for _, label, _ in parts] + ["Total"], y=values, name="return_by_part",
        marker=dict(color=_signed(values[:-1]) + [TOTAL], cornerradius=3),
        text=_labelled(values, lambda v: f"{v:+.1%}"), textposition="outside", cliponaxis=False,
        textfont=dict(color=INK2), hovertemplate="%{x}<br>%{y:+.2%} a year<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", range=_padded(values, 1.2))
    fig.update_layout(bargap=0.35)
    fig.add_hline(y=0, line=dict(color=INK2, width=1.2), layer="above")
    return fig


def _risk_by_part(segment: dict, groups: tuple) -> go.Figure:
    ante = segment["ex_ante_risk"]
    parts = _kept([ante["group_contribution"].get(key) for key, _, _ in groups]
                  + [ante["contribution"].get("specific")])
    totals = _kept([ante["volatility"].get("total"), segment["ex_post_risk"]["volatility"]])
    labels = [label for _, label, _ in groups] + ["Specific", "Forecast", "Realized"]
    values = parts + totals
    fig = _figure(330)
    fig.add_trace(go.Bar(
        x=labels, y=values, name="risk_by_part",
        marker=dict(color=[LIGHT if (v or 0) >= 0 else NEGATIVE for v in parts] + [TOTAL, INK], cornerradius=3),
        text=_labelled(parts, lambda v: f"{v:+.1%}") + _labelled(totals, lambda v: f"{v:.1%}"),
        textposition="outside", cliponaxis=False, textfont=dict(color=INK2),
        hovertemplate="%{x}<br>%{y:.2%} annualized volatility<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", range=_padded(values, 1.2))
    fig.update_layout(bargap=0.35)
    fig.add_hline(y=0, line=dict(color=INK2, width=1.2), layer="above")
    return fig


def _spread(values: list[float], gap: float) -> list[float]:
    """Label positions at least ``gap`` apart, keeping their order."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for low, high in zip(order[:-1], order[1:]):
        if out[high] - out[low] < gap:
            out[high] = out[low] + gap
    return out.tolist()


def _return_over_time(attribution: xr.Dataset, groups: tuple) -> go.Figure:
    parts = _parts(attribution, groups)
    curves = {key: series.cumsum() for key, series in parts.items()}
    total = sum(parts.values()).cumsum()
    first, last = total.index[0], total.index[-1]
    fig = _figure(380)
    labels = []
    for key, label, colour in _parts_of(groups):
        curve = curves[key]
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name=f"cumulative_{key}", mode="lines",
                                 line=dict(color=colour, width=2),
                                 hovertemplate=f"{label} %{{y:+.1%}}<extra></extra>"))
        labels.append((curve.values[-1], f"{label} {curve.values[-1]:+.0%}", INK2, colour))
    fig.add_trace(go.Scatter(x=total.index, y=total.values, name="cumulative_total", mode="lines",
                             line=dict(color=INK, width=3), hovertemplate="Total %{y:+.1%}<extra></extra>"))
    labels.append((total.values[-1], f"<b>Total {total.values[-1]:+.0%}</b>", INK, INK))
    every = [total.values] + [curve.values for curve in curves.values()]
    span = float(max(np.nanmax(v) for v in every) - min(np.nanmin(v) for v in every)) or 1.0
    for (_, text, ink, colour), at in zip(labels, _spread([y for y, *_ in labels], span * 0.055)):
        fig.add_annotation(x=last, y=at, text=f'<span style="color:{colour}">■</span> {text}', showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=11, color=ink))
    fig.update_layout(hovermode="x unified", margin=dict(l=10, r=130, t=10, b=10))
    fig.update_xaxes(range=[first, last])
    fig.update_yaxes(tickformat=".0%")
    fig.add_hline(y=0, line=dict(color=INK2, width=1.2), layer="above")
    return fig


def _risk_over_time(attribution: xr.Dataset, groups: tuple, scale: float) -> go.Figure:
    group = attribution["group"].values
    risk = attribution["factor_risk_contribution"].to_pandas()
    frame = pd.DataFrame({key: risk.loc[:, group == key].sum(axis=1, min_count=1) * scale
                          for key, _, _ in groups})
    frame["specific"] = attribution["specific_risk_contribution"].to_pandas() * scale
    monthly = frame.resample("ME").mean()
    realized = attribution["return"].to_pandas().rolling(63).std() * scale
    fig = _figure(380)
    for key, label, colour in groups + (PARTS[3],):
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[key], name=label, marker=dict(color=colour),
                             hovertemplate=f"%{{x|%Y-%m}} {label} %{{y:.1%}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=realized.index, y=realized.values, name="Realized (63-bar)", mode="lines",
                             line=dict(color=INK, width=2, dash="dot"),
                             hovertemplate="%{x|%Y-%m-%d} realized %{y:.1%}<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.1, showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
    low = float(np.nan_to_num(monthly.clip(upper=0).sum(axis=1).min(), posinf=0, neginf=0))
    high = max(float(np.nan_to_num(monthly.clip(lower=0).sum(axis=1).max(), posinf=0, neginf=0)),
               float(np.nan_to_num(realized.max(), posinf=0, neginf=0)))
    bounds, step = _ticked_range(low, high)
    fig.update_yaxes(tickformat=".0%", tick0=0, dtick=step, range=bounds,
                     title=dict(text="annualized volatility", font=dict(size=11, color=MUTED)))
    fig.add_hline(y=0, line=dict(color=INK2, width=1.2), layer="above")
    return fig


def _ticked_range(low: float, high: float, ticks: int = 8) -> tuple[list[float], float]:
    """An axis range from ``low`` <= 0 to ``high`` >= 0 and its round tick spacing.

    A part below 0 (a hedge) is small beside the market's spikes, so the range reaches a whole
    tick below 0 and the hedge gets a labelled tick; rounding noise below 0 does not count.
    """
    span = (high - low) or 1.0
    power = 10.0 ** np.floor(np.log10(span / ticks))
    step = float(next(m for m in (1, 2, 5, 10) if m * power >= span / ticks) * power)
    bottom = -step * np.ceil(-low / step) if low < -1e-9 * span else 0.0
    return [float(bottom), high * 1.05 or step], step


def _sorted_styles(segment: dict, attribution: xr.Dataset) -> list[str]:
    growth = segment["factor_annualized_log_return"]
    return sorted(_members(attribution, "style"), key=lambda name: growth[name] or 0.0)


def _bars(fig, labels, values, text, hover, name, col) -> None:
    fig.add_trace(go.Bar(y=labels, x=values, orientation="h", name=name,
                         marker=dict(color=_signed(values), cornerradius=3), text=text,
                         textposition="outside", cliponaxis=False, textfont=dict(color=INK2, size=11),
                         hovertemplate=hover), row=1, col=col)
    fig.add_vline(x=0, line=dict(color=INK2, width=1.2), layer="above", row=1, col=col)


def _style_return(segment: dict, attribution: xr.Dataset, labels: dict) -> go.Figure | None:
    names = _sorted_styles(segment, attribution)
    if not names:
        return None
    shown = [labels[n] for n in names]
    exposure = _zero([segment["style_mean_exposure"].get(n) for n in names])
    growth = _zero([segment["factor_annualized_log_return"][n] for n in names])
    fig = _figure(max(240, 28 * len(names) + 60), cols=2,
                  titles=("Mean net exposure", "Contribution, log growth / yr"))
    _bars(fig, shown, exposure, [f"{v:+.2f}" for v in exposure], "%{y}: exposure %{x:.2f}<extra></extra>",
          "style_exposure", 1)
    _bars(fig, shown, growth, [f"{v:+.1%}" for v in growth], "%{y}: %{x:+.2%} a year<extra></extra>",
          "style_contribution", 2)
    fig.update_xaxes(range=_padded(exposure), row=1, col=1)
    fig.update_xaxes(tickformat=".0%", range=_padded(growth), row=1, col=2)
    fig.update_layout(bargap=0.3)
    return fig


def _style_risk(segment: dict, attribution: xr.Dataset, labels: dict) -> go.Figure | None:
    names = _sorted_styles(segment, attribution)
    if not names:
        return None
    shown = [labels[n] for n in names]
    forecast = _kept([segment["ex_ante_risk"]["factor_contribution"].get(n) for n in names])
    realized = _kept([segment["ex_post_risk"]["factor_contribution"].get(n) for n in names])
    fig = _figure(max(240, 28 * len(names) + 60), cols=2,
                  titles=("Share of forecast volatility", "Share of realized volatility"))
    for col, values, name in ((1, forecast, "style_forecast_risk"), (2, realized, "style_realized_risk")):
        _bars(fig, shown, values, _labelled(values, lambda v: f"{v:+.2%}"),
              "%{y}: %{x:+.2%}<extra></extra>", name, col)
        fig.update_xaxes(tickformat=".1%", range=_padded(values), row=1, col=col)
    fig.update_layout(bargap=0.3)
    return fig


def _industries(segment: dict, attribution: xr.Dataset, labels: dict, *, risk: bool) -> go.Figure | None:
    names = _members(attribution, "industry")
    if not names:
        return None
    source = segment["ex_ante_risk"]["factor_contribution"] if risk else segment["factor_annualized_log_return"]
    values = {name: _zero([source.get(name)])[0] for name in names}
    held = attribution["gross_weight"] > 0
    weight = attribution["exposure"].sel(factor=names).where(held).mean("timestamp").values
    weight = dict(zip(names, np.nan_to_num(weight)))
    ranked = sorted(names, key=lambda name: -values[name])
    shown = ranked if len(ranked) <= 2 * INDUSTRIES_SHOWN else ranked[:INDUSTRIES_SHOWN] + ranked[-INDUSTRIES_SHOWN:]
    xs = [values[name] for name in shown]
    ys = [labels[name] for name in shown]
    unit = "of forecast vol" if risk else "a year"
    fig = _figure(max(240, 21 * len(shown) + 40))
    fig.add_trace(go.Bar(
        y=ys, x=xs, orientation="h", name="industry_risk" if risk else "industry_return",
        marker=dict(color=_signed(xs), cornerradius=3), text=[f"{v:+.2%}" for v in xs],
        textposition="outside", cliponaxis=False, textfont=dict(size=10, color=INK2),
        customdata=[weight[name] for name in shown],
        hovertemplate=f"%{{y}}<br>%{{x:+.2%}} {unit}<br>mean net weight %{{customdata:.1%}}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", tickvals=ys,
                     ticktext=[f"{html.escape(y)}  <span style='color:{MUTED}'>{weight[n]:.0%}</span>"
                               for y, n in zip(ys, shown)])
    fig.update_xaxes(tickformat=".2%" if risk else ".1%", nticks=6, range=_padded(xs, 1.3))
    fig.update_layout(bargap=0.25)
    fig.add_vline(x=0, line=dict(color=INK2, width=1.2), layer="above")
    return fig


def _style_heatmap(segment: dict, attribution: xr.Dataset, labels: dict) -> go.Figure | None:
    names = _sorted_styles(segment, attribution)[::-1]
    held = attribution["gross_weight"].to_pandas() > 0
    if not names or not held.any():
        return None
    weekly = attribution["exposure"].sel(factor=names).to_pandas()[held].resample("W-FRI").mean()
    limit = float(np.nanpercentile(np.abs(weekly.values), 98)) if np.isfinite(weekly.values).any() else 1.0
    limit = limit or 1.0
    fig = _figure(max(200, 26 * len(names) + 60))
    fig.add_trace(go.Heatmap(
        z=weekly.T.values, x=weekly.index, y=[labels[n] for n in names], zmid=0, zmin=-limit, zmax=limit,
        colorscale=[[0, NEGATIVE], [0.5, MIDPOINT], [1, POSITIVE]], ygap=2, name="style_exposure_heatmap",
        colorbar=dict(title=dict(text="exposure", font=dict(size=11, color=MUTED)), thickness=10),
        hovertemplate="%{y} · week of %{x|%Y-%m-%d}<br>exposure %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def _return_vs_risk(segment: dict, groups: tuple) -> go.Figure:
    group_growth, terms = segment["group_annualized_log_return"], segment["annualized_log_return"]
    post = segment["ex_post_risk"]
    parts = _parts_of(groups)
    labels = [label for _, label, _ in parts]
    growth = _kept([group_growth.get(key) if i < len(groups) else terms.get(key)
                    for i, (key, _, _) in enumerate(parts)])
    risk = _kept([post["group_contribution"].get(key) if i < len(groups) else post["term_contribution"].get(key)
                  for i, (key, _, _) in enumerate(parts)])
    fig = _figure(max(240, 44 * len(parts) + 60))
    for values, name, legend, colour in ((growth, "part_return", "Return, log growth / yr", POSITIVE),
                                         (risk, "part_realized_risk", "Share of realized volatility", LIGHT)):
        fig.add_trace(go.Bar(y=labels, x=values, orientation="h", name=legend, meta=name,
                             marker=dict(color=colour, cornerradius=3), text=_labelled(values, lambda v: f"{v:+.1%}"),
                             textposition="outside", cliponaxis=False, textfont=dict(size=10, color=INK2),
                             hovertemplate="%{y}: %{x:+.2%}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", range=_padded(growth + risk, 1.15))
    fig.add_vline(x=0, line=dict(color=INK2, width=1.2), layer="above")
    return fig
