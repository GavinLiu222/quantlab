"""HTML report writer for one backtest run.

``write_backtest_report`` renders a single self-contained page for a run: a
dark header naming the run, its window and benchmark, a sidebar of sections,
a row of headline cards and, per section, cards holding each chart beside
the tables that explain it. The tables compare the strategy
with the benchmark by group (returns, risk, risk-adjusted), list the excess
over the benchmark and the trading statistics, and, when the run has an
in-sample part, set the in-sample and out-of-sample slices side by side.
*In-sample* means the bars the model was trained on; *out-of-sample* means
the bars it never saw, which are the honest test, so the cards and the main
tables show the out-of-sample slice whenever the run has both. The sections
are Overview (the windows, in-sample against out-of-sample, the Performance
chart of equity, drawdown, monthly returns and their year-by-month heatmap,
and strategy against benchmark), Excess (cumulative excess return and excess
drawdown, and the relative table), Rolling (one-year rolling statistics),
Portfolio (turnover, holdings and exposure per rebalance, and the trading
table), Holdings when the run's holdings are given (each bar's targets,
holdings and cash, stepped through by its own small script from data
embedded in the page as JSON), Attribution for a model run, Factor attribution for a run with a
risk model (``quantlab.runs.factor_attribution_report``, drawn from the run's
metrics and per-bar ``factor_attribution.zarr`` only: this layer imports no
risk model) and Setup & notes. Every headline card, metric row and chart
carries a plain-English explanation in its ``title``, which the page shows
as a tooltip on hover. Charts are drawn on a white plot area with a light
grid. ``backtest_report_figure`` returns the Performance chart alone, as a
plotly figure.

The module holds a catalogue of the metrics it knows, with each one's
group, label, unit and definition, but it renders every metric it is given:
a key missing from the catalogue appears in an "Other" table with a unit
guessed from its name, and a value it cannot render becomes a dash. That is
deliberate, because the report is written inside a run's staging directory,
where an exception would discard the whole run rather than just the page.
For the same reason every string that comes from the run passes through
``html.escape`` before it reaches the page, and plotly.js is loaded from its
CDN so a run directory stays a few kilobytes (viewing the page needs network
access).

The page is drawn from what the run recorded: the account record of
``quantlab.runs.run_metrics`` (value, returns, fills, settlements, the
rebalance record, holdings, initial cash, bar interval and year), the run's
metrics, its target weights, its config mapping and its benchmark record,
plus one ``ReportExtras`` slot for what only some runs have. Every chart
input (turnover, the decided rebalances, the deepest drawdown, the Rolling
window, the timeline, the Setup lines, the benchmark curve) is derived here
from those, so an executor that simulates elsewhere (an event-driven replay
of a quantlab run) writes a page in exactly this format by handing over its
own record. This module imports only the standard library, numpy,
pandas, xarray, plotly, ``quantlab.runs.backtest_stats``,
``quantlab.runs.run_metrics`` and ``quantlab.runs.factor_attribution_report``.
"""

import bisect
import datetime
import html
import json
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

from quantlab.utils import date_range
from quantlab.utils.returns import one_bar_returns
from quantlab.runs import backtest_stats
from quantlab.runs.run_metrics import AccountRecord, BenchmarkRecord, decided_bars
from quantlab.runs.factor_attribution_report import (
    AXIS, CHART_FONT, GRID, INK2, MUTED, factor_attribution_section,
)

__all__ = [
    "ReportExtras",
    "backtest_report_figure",
    "write_backtest_report",
]

#: Shown in place of a value the run does not have. An em dash rather than a
#: hyphen so it cannot be read as the minus sign of a negative number.
DASH = "—"

_STYLE = """
  * { box-sizing: border-box; }
  body { margin: 0; background: #f4f5f7; color: #111827; font-size: 13px;
         font-family: Inter, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  header { position: sticky; top: 0; z-index: 5; background: #0f172a; color: #fff; padding: 12px 24px;
           display: flex; align-items: baseline; gap: 16px; }
  h1 { font-size: 15px; font-weight: 600; margin: 0; overflow: hidden; text-overflow: ellipsis;
       white-space: nowrap; }
  header .sub { color: #94a3b8; font-size: 12px; white-space: nowrap; }
  h2 { font-size: 13px; font-weight: 600; color: #111827; margin: 0 0 10px; }
  .card h2 ~ h2, .card table + h2, .card ul + h2 { margin-top: 18px; }
  h3 { font-size: 13px; font-weight: 600; color: #111827; margin: 0 0 2px; }
  h3 .i { color: #9ca3af; font-weight: 400; }
  p.sub { font-size: 12px; color: #6b7280; margin: 0 0 6px; }
  .shell { display: flex; min-height: calc(100vh - 46px); }
  nav { width: 200px; flex: none; padding: 20px 12px; position: sticky; top: 46px;
        height: calc(100vh - 46px); border-right: 1px solid #e5e7eb; background: #fff; }
  .tab { display: block; width: 100%; text-align: left; border: 0; background: none; padding: 9px 12px;
         border-radius: 8px; font: inherit; color: #374151; cursor: pointer; margin-bottom: 2px; }
  .tab:hover { background: #f3f4f6; }
  .tab.on { background: #eff6ff; color: #2563eb; font-weight: 600; }
  main { flex: 1; min-width: 0; padding: 20px 24px 60px; }
  .pane { display: none; } .pane.on { display: block; }
  .kpis, .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
                  margin-bottom: 20px; }
  .tiles { margin-bottom: 0; }
  .kpi, .tile { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 16px;
                box-shadow: 0 1px 2px rgba(0,0,0,.05); }
  .kl { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }
  .kv { font-size: 22px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
  .kv.pos { color: #059669; } .kv.neg { color: #dc2626; }
  .ks { font-size: 12px; color: #6b7280; margin-top: 2px; }
  .grid { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 16px; align-items: start;
          margin-bottom: 16px; }
  .grid2, .pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px;
                  align-items: start; margin-bottom: 16px; }
  .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 18px;
          box-shadow: 0 1px 2px rgba(0,0,0,.05); min-width: 0; overflow-x: auto; margin-bottom: 16px; }
  .grid > .card, .grid2 > .card, .pair > .card { margin-bottom: 0; }
  .fa { display: flex; flex-direction: column; gap: 16px; }
  .fa .pair, .fa .card { margin-bottom: 0; }
  .pair.heads { margin-bottom: -8px; }
  .colhead { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
  p.note { font-size: 12px; color: #6b7280; margin: 0; }
  table.summary { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  table.summary th { text-align: left; padding: 5px 16px 5px 0; font-weight: 500; color: #6b7280;
                     white-space: nowrap; border-bottom: 1px solid #f1f2f4; vertical-align: top; }
  table.summary td { padding: 5px 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere;
                     border-bottom: 1px solid #f1f2f4; }
  table.summary td .scroll { max-height: 4.8em; overflow-y: auto; font-size: 12px; padding-right: 4px; }
  table.metrics { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  table.metrics th, table.metrics td { padding: 5px 10px 5px 0; border-bottom: 1px solid #f1f2f4;
                                       white-space: nowrap; }
  table.metrics thead th { text-align: right; color: #6b7280; font-weight: 500; font-size: 11px;
                           text-transform: uppercase; letter-spacing: .04em; }
  table.metrics thead th:first-child { text-align: left; }
  table.metrics tbody th { text-align: left; font-weight: 400; color: #374151; }
  table.metrics td { text-align: right; font-variant-numeric: tabular-nums; }
  table.metrics td.better { font-weight: 600; color: #059669; }
  table.metrics tr.group td { text-align: left; font-size: 10.5px; color: #6b7280; text-transform: uppercase;
                              letter-spacing: .06em; padding-top: 12px; border-bottom: 1px solid #e5e7eb; }
  [data-tip], [title] { cursor: help; }
  table.metrics tbody th[data-tip] { text-decoration: underline dotted #9ca3af; text-underline-offset: 3px; }
  .kpi[data-tip]:hover, .tile[data-tip]:hover { border-color: #93c5fd; }
  #tip { position: fixed; z-index: 100; max-width: 340px; background: #0f172a; color: #f8fafc;
         font-size: 13px; line-height: 1.5; padding: 10px 12px; border-radius: 8px; pointer-events: none;
         box-shadow: 0 8px 24px rgba(0,0,0,.25); display: none; }
  #tip b { display: block; margin-bottom: 4px; font-weight: 600; }
  ul.notes { color: #4b5563; padding-left: 18px; line-height: 1.55; margin: 0; }
  .timeline { font-size: 12px; color: #333; }
  .timeline .caption { color: #6b7280; margin: 0 0 6px; }
  .timeline svg text { font-size: 10px; fill: #6b7280; }
  .timeline .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: #6b7280; margin-top: 4px; }
  .timeline .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                  vertical-align: -1px; margin-right: 4px; }
  @media (max-width: 900px) {
    .shell { flex-direction: column; }
    nav { width: auto; height: auto; position: static; display: flex; overflow-x: auto; padding: 8px 12px;
          border-right: 0; border-bottom: 1px solid #e5e7eb; }
    .tab { width: auto; white-space: nowrap; }
    main { padding: 16px; }
    .grid, .grid2, .pair { grid-template-columns: 1fr; }
  }
"""


@dataclass(frozen=True)
class ReportExtras:
    """What only some runs put on their page: the one optional slot of ``write_backtest_report``.

    Attributes
    ----------
    attribution : xarray.Dataset or None
        The attribution curves of a model run (``universe_value``,
        ``gross_value``, ``group_value``; ``quantlab.runs.backtest_attribution``).
        With them and an ``attribution`` block in the metrics the page gets
        an Attribution section: the excess split into its parts, the
        cumulative log growth of the strategy, before costs, the universe
        and the benchmark, and each score group's annualised return.
    factor_attribution : xarray.Dataset or None
        The per-bar factor attribution of a run with a risk model
        (``factor_attribution.zarr``). With it and a ``factor_attribution``
        block in the metrics the page gets a Factor attribution section
        (``quantlab.runs.factor_attribution_report``), summarizing the
        out-of-sample segment when the run has an in-sample part.
    tables : Mapping or None
        More tables after the metric tables: heading -> ordered mapping of
        row label to value, each value shown as an unknown metric is (a
        number by its magnitude, a percent when the label ends in ``[%]``, a
        string as it is). For statistics only an executor has, such as an
        event-driven replay's commissions.
    setup : Mapping or None
        Setup lines applied after the standard ones: a label the page
        already has is replaced in place (an executor states its own fee
        model under ``Fees``), a new one is appended.
    names : callable or None
        ``names(symbols, day)`` returns a ``(ticker, company)`` pair for each
        of ``symbols`` as of the date ``day``, the company ``None`` when
        unknown; the Holdings section shows symbols by them, a new span
        wherever a name changes. Without it a symbol is shown by its id.
    prices : xarray.DataArray or None
        The valuation prices on ``(timestamp, symbol)`` the engine marks the
        book with. With them the Holdings section shows each holding
        period's return (the symbol's price at the period's last close over
        its price at the signal bar's close, less one, carried forward over
        a bar without one) and the contributions by position size.

    Examples
    --------
    >>> extras = ReportExtras(tables={"Execution": {"Commissions": 12.5}}, setup={"Fees": "IBKR tiered"})
    >>> extras.setup["Fees"], extras.attribution is None
    ('IBKR tiered', True)
    """

    attribution: xr.Dataset | None = None
    factor_attribution: xr.Dataset | None = None
    tables: Mapping[str, Mapping] | None = None
    setup: Mapping[str, str] | None = None
    names: Callable[[list, datetime.date], list[tuple[str, str | None]]] | None = None
    prices: xr.DataArray | None = None


def write_backtest_report(
    account: AccountRecord,
    path: str | Path,
    *,
    title: str,
    metrics: Mapping | None = None,
    weights: xr.DataArray | None = None,
    config: Mapping | None = None,
    benchmark: BenchmarkRecord | None = None,
    folds: Sequence[Mapping] | None = None,
    extras: ReportExtras | None = None,
) -> None:
    """Write the HTML report of one backtest run to ``path``, drawn from its record.

    The page opens with a header (the run, its window and benchmark), a
    sidebar of sections and a row of headline numbers. Each section holds
    its charts beside the tables that explain them: Overview (the windows
    timeline, in-sample against out-of-sample when the run has an in-sample
    part, the Performance chart of NAV, drawdown, monthly returns and the
    monthly heatmap, and the strategy against the benchmark by group),
    Excess (cumulative excess return with a log / arithmetic toggle, the
    excess drawdown, and the excess over the benchmark), Rolling (one-year
    panels), Portfolio (turnover, holdings and exposure per rebalance, and
    trading), Holdings (each bar's targets, holdings and cash, when the
    account has holdings), Attribution and Factor attribution (from
    ``extras``) and Setup & notes. Every headline number, metric row and
    chart explains itself on hover. The headline numbers and the main tables
    are the out-of-sample slice when the run has an in-sample part, else the
    whole window. Drawdown is ``value / running max - 1``, negative below a
    peak, in every table, card and chart. A section, card or table whose
    input is missing is left off the page, and a metric the page does not
    know is still shown, in an "Other" table.

    Every chart input is derived from the arguments: the in-sample range
    (shaded) and the notes from the metrics; the turnover per rebalance fill
    bar (``backtest_stats.turnover`` of the account's fills, its settlements
    left out); the decided rebalances from its rebalance record; the Rolling
    window (``year_freq`` over the bar interval); the deepest drawdown,
    marked from its valley to its recovery (``backtest_stats.drawdown_span``
    of the value); the timeline; and the benchmark's curve.

    Parameters
    ----------
    account : AccountRecord
        What the account did (``quantlab.runs.run_metrics``): its value
        (drawn as is, so the page and the persisted equity carry the same
        numbers), returns, fills, settlements, rebalance record, initial
        cash, bar interval and year, and its holdings (each symbol's value
        at each bar's close over the book's value), which give the page a
        Holdings section.
    path : str | Path
        The ``report.html`` to write.
    title : str
        Page title, typically the run directory name.
    metrics : Mapping, optional
        The run's metrics, as ``metrics.json`` holds them: the slices
        ``whole``, ``in_sample`` and ``out_of_sample``, ``benchmark`` and
        ``relative``, ``execution`` and ``portfolio_construction``, the split
        keys (``in_sample_range`` is shaded), ``trained_checkpoint`` and
        ``notes`` (listed in the Notes section). A ``run_cv()`` run's mapping
        (``stitched``, ``folds``, ``notes``) is read through ``stitched``,
        and its ``folds`` are the timeline's rows.
    weights : xarray.DataArray, optional
        Target weights on ``(timestamp, symbol)``, an all-NaN row on a bar
        that holds; the Portfolio section draws the holdings and exposure of
        the decided rebalances' rows (a NaN cell counted at that symbol's
        last target) and the Holdings section the targets in force.
    config : Mapping, optional
        The run's config mapping (``BaseBacktester.get_config()``, a run
        directory's ``config.json``), read for the Setup lines: ``Bar
        interval``; ``Benchmark`` with its ``source``; the deepest drawdown;
        ``Model mode`` (or ``Signal`` for a run without a model);
        ``Rebalance every``; ``Portfolio construction`` (the rule's
        ``repr``), or ``Top N`` and ``Direction``; ``Fees``; ``Trained
        checkpoint``. Every key is read with ``.get()``. Without it the page
        has only the lines of ``extras.setup``.
    benchmark : BenchmarkRecord, optional
        The benchmark (``quantlab.runs.run_metrics``), drawn beside the
        portfolio and named by the metrics' ``benchmark`` symbol (else its
        own ``symbol``, else "benchmark"); its curve is its ``value``, or
        ``init_cash * cumprod(1 + returns)`` (bought and held from the
        account's capital) when it has none. A benchmark that does not cover
        the account's bars is left off.
    folds : sequence of Mapping, optional
        The timeline's rows of a run traded by several models, each with
        ``fold`` (its number), ``training_window``, ``traded`` (its first
        and last traded bar label) and ``in_sample_range``. Default: the
        rows of ``metrics["folds"]`` (a ``run_cv()`` run), else one "model"
        row when the metrics have a ``training_window``, else none.
    extras : ReportExtras, optional
        What only some runs have (attribution, extra tables, Setup lines,
        the Holdings section's names and prices).

    Examples
    --------
    >>> import pandas as pd, xarray as xr
    >>> from quantlab.runs.run_metrics import AccountRecord
    >>> bars = pd.bdate_range("2024-01-01", periods=5)
    >>> on_bars = dict(dims="timestamp", coords={"timestamp": bars})
    >>> account = AccountRecord(
    ...     value=xr.DataArray([100.0, 104.0, 98.0, 103.0, 110.0], **on_bars),
    ...     returns=xr.DataArray([0.0, 0.04, -0.0577, 0.051, 0.068], **on_bars),
    ...     orders=xr.Dataset(),
    ...     rebalances=xr.Dataset({"scheduled": ("timestamp", [False] * 5),
    ...                            "decided": ("timestamp", [False] * 5)}, coords={"timestamp": bars}),
    ...     init_cash=100.0, bar_interval="1D", year_freq=pd.Timedelta(days=252),
    ... )
    >>> write_backtest_report(
    ...     account, "demo.html", title="demo_run",
    ...     metrics={"whole": {"Total Return [%]": 10.0}, "notes": ["No borrow cost is modelled."]},
    ...     config={"rebalance_periods": 5, "fees": 0.001},
    ... )
    >>> page = open("demo.html").read()
    >>> "<h1>demo_run</h1>" in page, "<td>5 bars</td>" in page
    (True, True)
    """
    _render(account.value, path, title=title,
            **_page_inputs(account, metrics, weights, config, benchmark, folds, extras or ReportExtras()))


def backtest_report_figure(
    account: AccountRecord,
    *,
    metrics: Mapping | None = None,
    benchmark: BenchmarkRecord | None = None,
) -> go.Figure:
    """Return the Performance chart of a run's ``report.html`` as a plotly figure.

    Equity, drawdown and compounded monthly returns on a shared time axis,
    the in-sample range shaded, the deepest drawdown marked and the notes
    below; with a benchmark, its NAV, drawdown and monthly returns beside the
    portfolio's. The excess over the benchmark is not drawn here:
    ``report.html`` draws it on its own tab, apart from the drawdowns,
    because the drawdown of the relative NAV is a different quantity from
    either NAV's drawdown. The arguments mean what they mean for
    ``write_backtest_report``, whose Performance tab is this chart without
    the notes (the page lists them in its own section).

    Returns
    -------
    plotly.graph_objects.Figure
        The figure, not yet shown or written.

    Examples
    --------
    >>> import pandas as pd, xarray as xr
    >>> from quantlab.runs.run_metrics import AccountRecord
    >>> bars = pd.bdate_range("2024-01-01", periods=5)
    >>> on_bars = dict(dims="timestamp", coords={"timestamp": bars})
    >>> account = AccountRecord(
    ...     value=xr.DataArray([100.0, 104.0, 98.0, 103.0, 110.0], **on_bars),
    ...     returns=xr.DataArray([0.0, 0.04, -0.0577, 0.051, 0.068], **on_bars),
    ...     orders=xr.Dataset(),
    ...     rebalances=xr.Dataset({"scheduled": ("timestamp", [False] * 5),
    ...                            "decided": ("timestamp", [False] * 5)}, coords={"timestamp": bars}),
    ...     init_cash=100.0, bar_interval="1D", year_freq=pd.Timedelta(days=252),
    ... )
    >>> figure = backtest_report_figure(account)
    >>> sorted(trace.name for trace in figure.data)
    ['deepest_drawdown_end', 'deepest_drawdown_valley', 'drawdown', 'equity', 'monthly_return']
    """
    block = _block(metrics)
    return _performance_figure(account.value, notes=_notes(metrics),
                               **_chart_inputs(account, block, benchmark))


def _block(metrics: Mapping | None) -> dict:
    """The metric level carrying the split keys: ``stitched`` of a ``run_cv()`` run, else the metrics."""
    metrics = dict(metrics or {})
    stitched = metrics.get("stitched")
    return dict(stitched) if isinstance(stitched, Mapping) else metrics


def _notes(metrics: Mapping | None) -> list[str]:
    """The run's notes, ``metrics["notes"]``."""
    return list((metrics or {}).get("notes") or [])


def _chart_inputs(account: AccountRecord, block: Mapping, benchmark: BenchmarkRecord | None) -> dict:
    """The arguments the Performance chart shares with the page, from the record.

    ``in_sample_range`` (the block's, shaded), the account's ``returns`` and
    ``init_cash``, its deepest ``drawdown_span``, and with a benchmark its
    ``benchmark_value`` (its own, or bought and held from the account's
    capital), ``benchmark_returns`` and ``benchmark_name``.
    """
    inputs = {
        "in_sample_range": block.get("in_sample_range"),
        "returns": account.returns,
        "init_cash": account.init_cash,
        "drawdown_span": backtest_stats.drawdown_span(account.value),
    }
    if benchmark is not None:
        value = benchmark.value
        if value is None:
            value = account.init_cash * (1.0 + benchmark.returns.fillna(0.0)).cumprod("timestamp")
        info = block.get("benchmark") or {}
        inputs.update(
            benchmark_value=value,
            benchmark_returns=benchmark.returns,
            benchmark_name=info.get("symbol") or benchmark.symbol or "benchmark",
        )
    return inputs


def _fold_rows(metrics: Mapping | None) -> list[dict] | None:
    """The timeline rows of a ``run_cv()`` run's ``metrics["folds"]``, or ``None`` without folds.

    Each fold's number, its ``training_window`` and ``in_sample_range``, and
    the first and last bar it traded (its ``whole`` ``Start`` and ``End``).
    """
    folds = (metrics or {}).get("folds")
    if not isinstance(folds, list):
        return None
    return [
        {
            "fold": fold["fold"],
            "training_window": fold["metrics"].get("training_window"),
            "traded": tuple(date_range.bar_label(fold["metrics"]["whole"][key]) for key in ("Start", "End")),
            "in_sample_range": fold["metrics"].get("in_sample_range"),
        }
        for fold in folds
    ]


def _page_inputs(
    account: AccountRecord,
    metrics: Mapping | None,
    weights: xr.DataArray | None,
    config: Mapping | None,
    benchmark: BenchmarkRecord | None,
    folds: Sequence[Mapping] | None,
    extras: ReportExtras,
) -> dict:
    """Every input of the page but its value and title, derived from what the run recorded."""
    block = _block(metrics)
    chart = _chart_inputs(account, block, benchmark)
    summary = None
    if config is not None:
        summary = _summary(
            dict(config), block, bar_interval=account.bar_interval, drawdown_span=chart["drawdown_span"],
            benchmark_source=None if benchmark is None else benchmark.source,
        )
    if extras.setup:
        summary = {**(summary or {}), **extras.setup}
    rebalances = decided_bars(account.rebalances)
    inputs = {
        **chart,
        "notes": _notes(metrics),
        "summary": summary,
        "metrics": block,
        "windows": _windows(
            account.value.timestamp.values, block,
            _fold_rows(metrics) if folds is None else [dict(fold) for fold in folds],
        ),
        **_portfolio_inputs(weights, account, rebalances),
        "extra_tables": None if extras.tables is None else {k: dict(v) for k, v in extras.tables.items()},
        "attribution": extras.attribution,
        "factor_attribution": extras.factor_attribution,
    }
    if account.holdings is not None:
        inputs.update(_holdings_inputs(
            account.holdings, weights, rebalances=rebalances, label=extras.names, prices=extras.prices,
        ))
    return inputs


def _render(
    value: xr.DataArray,
    path: str | Path,
    *,
    in_sample_range: tuple[str, str] | None,
    notes: list[str],
    title: str,
    summary: dict[str, str] | None = None,
    metrics: dict | None = None,
    returns: xr.DataArray | None = None,
    init_cash: float | None = None,
    drawdown_span: dict | None = None,
    benchmark_value: xr.DataArray | None = None,
    benchmark_returns: xr.DataArray | None = None,
    benchmark_name: str = "benchmark",
    weights: xr.DataArray | None = None,
    turnover: xr.DataArray | None = None,
    rebalances=None,
    bars_per_year: float | None = None,
    windows: dict | None = None,
    extra_tables: dict[str, dict] | None = None,
    attribution: xr.Dataset | None = None,
    factor_attribution: xr.Dataset | None = None,
    holdings: xr.DataArray | None = None,
    holding_names: Mapping[str, Sequence[tuple[str, str, str]]] | None = None,
    holding_returns: xr.DataArray | None = None,
    holding_contributions: xr.DataArray | None = None,
) -> None:
    """Write the page from its derived inputs (``_page_inputs``); see ``write_backtest_report``.

    ``in_sample_range`` is shaded; ``summary`` is the ordered Setup lines,
    already formatted; ``metrics`` the metric level carrying the split keys;
    ``drawdown_span`` the deepest drawdown (``valley``, ``end``, ``bars``,
    ``depth``, ``recovered``), marked from its valley to its recovery;
    ``benchmark_value`` and ``benchmark_returns`` the benchmark on the same
    bars; ``turnover`` the turnover per rebalance fill bar; ``rebalances``
    the decided rebalances' bars; ``bars_per_year`` the Rolling window
    (252 when not given); ``windows`` the timeline (``_windows``);
    ``holdings``, ``holding_names``, ``holding_returns`` and
    ``holding_contributions`` the Holdings section (``_holdings_inputs``).
    """
    # The page lists the notes in its own section; inside the figure they
    # would be cut off at the plot's width.
    fig = _performance_figure(
        value,
        in_sample_range=in_sample_range,
        notes=None,
        returns=returns,
        init_cash=init_cash,
        drawdown_span=drawdown_span,
        benchmark_value=benchmark_value,
        benchmark_returns=benchmark_returns,
        benchmark_name=benchmark_name,
    )
    equity = value.to_pandas()
    reference = _aligned_benchmark(benchmark_value, equity)
    name = str(benchmark_name) if reference is not None else None
    _clean(fig)
    tabs = [("Performance", fig.to_html(full_html=False, include_plotlyjs="cdn",
                                        config={"responsive": True})
             + _heatmap_block(_monthly_heatmap_div(returns)))]
    extra = [
        ("Excess", _excess_figure(equity, reference, name) if name else None),
        ("Rolling", _rolling_figure(equity, reference, bars_per_year, name)),
        ("Portfolio", _portfolio_figure(weights, turnover, rebalances)),
    ]
    for label, extra_fig in extra:
        if extra_fig is not None:
            _shade(extra_fig, in_sample_range)
            tabs.append((label, _figure_div(extra_fig)))
    if holdings is not None:
        tabs.append(("Holdings", _holdings_section(
            holdings, weights, rebalances, holding_names, holding_returns, holding_contributions,
            returns,
        )))
    block = (metrics or {}).get("attribution")
    if attribution is not None and block:
        tabs.append(("Attribution", _attribution_table(block, name)
                     + _figure_div(_attribution_figure(equity, attribution, reference, block, name))))
    factor_block = (metrics or {}).get("factor_attribution")
    if factor_attribution is not None and factor_block:
        tabs.append(("Factor attribution", factor_attribution_section(
            factor_block, factor_attribution, out_of_sample=_has_in_sample(metrics), bars_per_year=bars_per_year,
            in_sample_range=in_sample_range,
        )))
    Path(path).write_text(
        _document(title, summary, metrics, tabs, notes, benchmark_name=name, windows=windows,
                  extra_tables=extra_tables),
        encoding="utf-8",
    )


def _text(value) -> str:
    """Render ``value`` as text, with a dash for ``None``."""
    return DASH if value is None else str(value)


def _component_repr(config: dict) -> str:
    """The ``repr`` of a portfolio component, from its ``get_config()`` mapping.

    ``ClassName(field=value, ...)`` in the mapping's order, a nested
    component (a mapping with a ``name``) written the same way. A list, as
    JSON holds a tuple field (several components, for example), is written
    as the tuple the config holds.
    """
    def value_repr(value) -> str:
        if isinstance(value, dict) and "name" in value:
            return _component_repr(value)
        if isinstance(value, (list, tuple)):
            items = [value_repr(item) for item in value]
            return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"
        return repr(value)

    fields = ", ".join(f"{key}={value_repr(value)}" for key, value in config.items() if key != "name")
    return f"{str(config.get('name', '')).rsplit('.', 1)[-1]}({fields})"


def _summary(
    config: dict,
    block: dict,
    *,
    bar_interval,
    drawdown_span: dict | None = None,
    benchmark_source: str | None = None,
) -> dict[str, str]:
    """Return the "Setup" lines of a run's page from its config mapping.

    Pure presentation: an ordered mapping of label to text, read from the
    run's config mapping and metric block, computing nothing. The lines,
    in order: ``Bar interval``; ``Benchmark`` when ``block`` has a
    ``benchmark`` record; ``Deepest drawdown (valley to recovery)`` with a
    ``drawdown_span``; ``Model mode``, or ``Signal`` for a run without a
    model (a ``block`` without ``out_of_sample_ranges``); ``Rebalance
    every``; ``Portfolio construction`` (the rule's ``repr``) when the
    config names a constructor, else ``Top N`` and ``Direction`` for those
    of the two the config gives a value (weights recording the selection
    they came from); ``Fees``;
    ``Trained checkpoint`` when ``block`` records one. Every key is read
    with ``.get()`` and a missing value is a dash, so a renamed key degrades
    the page instead of raising. A caller replaces a line by assigning to
    its key, which keeps the order (an executor states its own fee model
    under ``Fees``), and appends its own lines after them.

    Parameters
    ----------
    config : dict
        The run's config mapping, as ``BaseBacktester.get_config()`` returns
        it or a run directory's ``config.json`` holds it: ``model_mode``,
        ``rebalance_periods``, ``constructor`` (the rule's ``get_config()``)
        or ``top_n`` and ``direction``, and ``fees``. Read back from JSON a
        tuple in the rule's config is a list, and its ``repr`` says so.
    block : dict
        The metric level carrying the split keys: a run's metrics, or the
        ``stitched`` metrics of a ``run_cv()`` run.
    bar_interval
        The bar spacing, anything ``pd.Timedelta`` accepts.
    drawdown_span : dict, optional
        ``backtest_stats.drawdown_span`` of the run's value.
    benchmark_source : str, optional
        Where the benchmark was read from, shown after its name.

    Returns
    -------
    dict[str, str]
        The lines.

    Examples
    --------
    >>> config = {"model_mode": "load", "rebalance_periods": 5, "fees": 0.001,
    ...           "constructor": {"direction": "long_only", "top_n": 2, "score_label": None,
    ...                           "name": "quantlab.portfolio.predefined.top_n.TopNConstructor"}}
    >>> summary = _summary(config, {"out_of_sample_ranges": []}, bar_interval="1D")
    >>> for label, text in summary.items():
    ...     print(f"{label}: {text}")
    Bar interval: 1 days 00:00:00
    Model mode: load
    Rebalance every: 5 bars
    Portfolio construction: TopNConstructor(direction='long_only', top_n=2, score_label=None)
    Fees: 0.001
    """
    summary = {"Bar interval": str(pd.Timedelta(bar_interval))}
    benchmark = block.get("benchmark")
    if isinstance(benchmark, dict):
        where = f" ({benchmark_source})" if benchmark_source else ""
        summary["Benchmark"] = f"{_text(benchmark.get('symbol'))}{where}, buy and hold"
    if drawdown_span:
        bars = drawdown_span.get("bars")
        summary["Deepest drawdown (valley to recovery)"] = (
            f"{_text(drawdown_span.get('valley'))} .. "
            f"{_text(drawdown_span.get('end'))}, "
            f"{DASH if bars is None else f'{bars} trading days'}, "
            f"{'recovered' if drawdown_span.get('recovered') else 'not recovered by the last bar'}"
        )
    if "out_of_sample_ranges" in block:
        summary["Model mode"] = _text(config.get("model_mode"))
    else:
        summary["Signal"] = "precomputed weights (run_weights), no model"
    summary["Rebalance every"] = f"{config.get('rebalance_periods')} bars"
    # A cross-sectional config names its portfolio construction rule; a
    # weights config records the selection its weights came from, if any.
    constructor = config.get("constructor")
    if isinstance(constructor, dict):
        summary["Portfolio construction"] = _component_repr(constructor)
    else:
        for label, key in (("Top N", "top_n"), ("Direction", "direction")):
            if config.get(key) is not None:
                summary[label] = _text(config[key])
    summary["Fees"] = _text(config.get("fees"))
    if block.get("trained_checkpoint") is not None:
        summary["Trained checkpoint"] = _text(block["trained_checkpoint"])
    return summary


def _windows(timestamps, block: dict, folds: list[dict] | None = None) -> dict:
    """Return the timeline of a run's page.

    Parameters
    ----------
    timestamps : array_like of datetime64
        The run's bars; the first and last are the backtest window.
    block : dict
        The metric level carrying the split keys, as for ``_summary``:
        ``in_sample_range`` or ``in_sample_ranges``,
        ``out_of_sample_ranges`` and, for a model backtest,
        ``training_window``.
    folds : list[dict], optional
        One row per fold of a ``run_cv()`` run, in fold order, each with
        ``fold`` (its number), ``training_window``, ``traded`` (its first
        and last traded bar, as ``date_range.bar_label`` strings) and
        ``in_sample_range``. From a run directory's ``metrics.json`` the
        row of ``fold`` in ``metrics["folds"]`` is ``fold["fold"]``,
        ``fold["metrics"]["training_window"]``,
        ``fold["metrics"]["in_sample_range"]`` and the ``bar_label`` of
        ``fold["metrics"]["whole"]["Start"]`` and ``["End"]``. Without
        folds, a block with a ``training_window`` is one "model" row and a
        block without (a ``run_weights()`` run) none.

    Returns
    -------
    dict
        ``backtest``, ``bars``, ``in_sample``, ``out_of_sample`` and
        ``folds``, with the labels ``metrics.json`` carries.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=5).values
    >>> windows = _windows(bars, {
    ...     "training_window": ("2023-01-02", "2023-12-29"), "in_sample_range": None,
    ...     "out_of_sample_ranges": [("2024-01-01", "2024-01-05")],
    ... })
    >>> windows["backtest"], windows["bars"], [row["label"] for row in windows["folds"]]
    (('2024-01-01', '2024-01-05'), 5, ['model'])
    """
    timestamps = np.asarray(timestamps)
    traded = (date_range.bar_label(timestamps[0]), date_range.bar_label(timestamps[-1]))
    in_sample = list(block.get("in_sample_ranges") or [])
    if block.get("in_sample_range"):
        in_sample.append(block["in_sample_range"])
    if folds is not None:
        rows = [
            {
                "label": f"fold {fold['fold']}",
                "training": fold.get("training_window"),
                "traded": fold.get("traded"),
                "in_sample": fold.get("in_sample_range"),
            }
            for fold in folds
        ]
    elif "training_window" in block:
        rows = [
            {
                "label": "model",
                "training": block.get("training_window"),
                "traded": traded,
                "in_sample": block.get("in_sample_range"),
            }
        ]
    else:
        rows = []
    return {
        "backtest": traded,
        "bars": int(timestamps.size),
        "in_sample": in_sample,
        "out_of_sample": list(block.get("out_of_sample_ranges") or []),
        "folds": rows,
    }


def _portfolio_inputs(weights: xr.DataArray | None, account: AccountRecord, rebalances) -> dict:
    """Return the Portfolio and Rolling inputs of a run's page from its account record.

    ``weights`` (as given), ``turnover`` (``backtest_stats.turnover`` of
    the account's fills per rebalance fill bar, its settlements left out:
    the sum is the ``Total Turnover [%]`` row), ``rebalances`` (the decided
    rebalances' bars) and ``bars_per_year`` (the account's year over its bar
    interval, the Rolling window).
    """
    return {
        "weights": weights,
        "turnover": backtest_stats.turnover(
            account.orders, account.value, account.init_cash, settlements=account.settlements
        ),
        "rebalances": np.asarray(rebalances).astype("datetime64[ns]"),
        "bars_per_year": account.year_freq / pd.Timedelta(account.bar_interval),
    }


def _holdings_inputs(
    holdings: xr.DataArray,
    weights: xr.DataArray,
    *,
    rebalances,
    label: Callable[[list, datetime.date], list[tuple[str, str | None]]] | None = None,
    prices: xr.DataArray | None = None,
) -> dict:
    """Return the Holdings section's inputs of a run's page.

    The tab shows, on each bar, every symbol held or targeted by the last
    rebalance before it; ``label`` names those symbols as of each bar, and
    the names are handed to the page as spans, a new span wherever a
    symbol's name changes (a ticker change).

    Parameters
    ----------
    holdings : xarray.DataArray
        The holdings on ``(timestamp, symbol)``: each symbol's value at
        each bar's close over the book's value, cash included.
    weights : xarray.DataArray
        The target weights on ``(timestamp, symbol)``, a row per signal bar
        (all-NaN on a bar that holds), as ``write_backtest_report`` takes
        them.
    rebalances : array_like of datetime64
        The bars of the decided rebalances, from the run's rebalance record:
        the rows of ``weights`` whose targets come into force.
    label : callable, optional
        ``label(symbols, day)`` returns a ``(ticker, company)`` pair for
        each of ``symbols`` as of the date ``day``, the company ``None``
        when unknown. Without it the page shows each symbol by its id.
    prices : xarray.DataArray, optional
        The valuation prices on ``(timestamp, symbol)`` the engine marks
        the book with. A holding period is the bars one rebalance's targets
        are in force, from its fill to the bar before the next rebalance
        fills; its return is the symbol's price at the period's last close
        over its price at the signal bar's close, less one, prices carried
        forward over a bar without one (as the book is valued).

    Returns
    -------
    dict
        ``holdings``; ``holding_names`` (``None`` without ``label``): per
        symbol id (as ``str``), ``(first bar label, ticker, company)``
        spans in time order, each in use until the next, an unknown company
        empty; ``holding_returns`` (``None`` without ``prices``): on
        ``(timestamp, symbol)``, the return of the holding period each bar
        falls in, NaN before the first rebalance fills or without a price;
        and ``holding_contributions`` (``None`` without ``prices``): on
        ``(timestamp, rank)``, each bar's contributions (a name's holding at
        the previous close times its return over the bar), one per name
        held at the previous close, largest holding first, NaN past the
        bar's last name.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=3)
    >>> holdings = xr.DataArray([[0.0], [0.5], [0.6]], dims=("timestamp", "symbol"),
    ...                         coords={"timestamp": bars, "symbol": [13407]})
    >>> weights = xr.DataArray([[0.5], [np.nan], [np.nan]], dims=("timestamp", "symbol"),
    ...                        coords={"timestamp": bars, "symbol": [13407]})
    >>> def label(symbols, day):
    ...     ticker = "FB" if str(day) < "2024-01-03" else "META"
    ...     return [(ticker, "Meta Platforms") for _ in symbols]
    >>> _holdings_inputs(holdings, weights, rebalances=bars[:1].values, label=label)["holding_names"]
    {'13407': [('2024-01-02', 'FB', 'Meta Platforms'), ('2024-01-03', 'META', 'Meta Platforms')]}
    >>> prices = xr.DataArray([[10.0], [11.0], [12.1]], dims=("timestamp", "symbol"),
    ...                       coords={"timestamp": bars, "symbol": [13407]})
    >>> inputs = _holdings_inputs(holdings, weights, rebalances=bars[:1].values, prices=prices)
    >>> inputs["holding_returns"].values.round(2).ravel().tolist()
    [nan, 0.21, 0.21]
    >>> inputs["holding_contributions"].values.round(3).tolist()
    [[nan], [nan], [0.05]]
    """
    names = None
    if label is not None:
        names = {}
        held = np.nan_to_num(holdings.transpose("timestamp", "symbol").values.astype(np.float64))
        targets, _ = _targets_in_force(weights, holdings, rebalances)
        symbols = holdings.symbol.values
        for i, bar in enumerate(holdings.timestamp.values):
            shown = np.flatnonzero((held[i] != 0.0) | (targets[i] != 0.0))
            if shown.size == 0:
                continue
            day = pd.Timestamp(bar).date()
            for j, (ticker, company) in zip(shown, label([symbols[j] for j in shown], day)):
                span = (str(ticker), company or "")
                spans = names.setdefault(str(symbols[j]), [])
                if not spans or spans[-1][1:] != span:
                    spans.append((date_range.bar_label(bar), *span))
    returns = None if prices is None else _holding_period_returns(weights, holdings, prices, rebalances)
    contributions = None if prices is None else _ranked_contributions(holdings, prices)
    return {
        "holdings": holdings,
        "holding_names": names,
        "holding_returns": returns,
        "holding_contributions": contributions,
    }


def _performance_figure(
    value: xr.DataArray,
    *,
    in_sample_range: tuple[str, str] | None = None,
    notes: list[str] | None = None,
    returns: xr.DataArray | None = None,
    init_cash: float | None = None,
    drawdown_span: dict | None = None,
    benchmark_value: xr.DataArray | None = None,
    benchmark_returns: xr.DataArray | None = None,
    benchmark_name: str = "benchmark",
) -> go.Figure:
    """The Performance chart; ``backtest_report_figure`` and the page's Performance tab draw it.

    The parameters mean what they mean for ``_render``; ``notes`` are
    written below the chart.
    """
    equity = value.to_pandas()
    drawdown = equity / equity.cummax() - 1.0
    reference = _aligned_benchmark(benchmark_value, equity)

    rows = {"equity": 1, "drawdown": 2, "monthly": 3}
    fig = make_subplots(
        rows=len(rows),
        cols=1,
        shared_xaxes=True,
        row_heights=[0.54, 0.23, 0.23],
        vertical_spacing=0.04,
    )
    _add_equity(fig, equity, init_cash)
    _add_drawdown_span(fig, equity, drawdown_span)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="drawdown",
            mode="lines",
            line={"color": PORTFOLIO_COLOUR},
        ),
        row=rows["drawdown"],
        col=1,
    )
    if reference is not None:
        _add_benchmark(fig, equity, reference, benchmark_name, init_cash, rows)
    _add_monthly_returns(
        fig,
        returns,
        row=rows["monthly"],
        benchmark_returns=benchmark_returns if reference is not None else None,
        benchmark_name=benchmark_name,
    )

    if in_sample_range is not None:
        fig.add_vrect(
            x0=in_sample_range[0],
            x1=in_sample_range[1],
            row="all",
            col=1,
            fillcolor="grey",
            opacity=0.2,
            line_width=0,
        )
    if notes:
        fig.add_annotation(
            text="<br>".join(notes),
            xref="paper",
            yref="paper",
            x=0.0,
            y=-0.16,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
        )
    fig.update_yaxes(title_text="value", row=rows["equity"], col=1)
    fig.update_yaxes(
        title_text="drawdown", tickformat=".1%", row=rows["drawdown"], col=1
    )
    fig.update_yaxes(
        title_text="monthly return", tickformat=".1%", row=rows["monthly"], col=1
    )
    if reference is not None:
        # Only the curves that need telling apart carry a legend entry.
        for trace in fig.data:
            trace.showlegend = trace.name in _LEGEND_TRACES
    # Rotated y-axis titles must fit in their row. At plotly's default 450px
    # rows 2 and 3 are about 44px tall and the titles collide; at 900px they
    # are about 140px and every title fits.
    fig.update_layout(
        height=900,
        margin={"b": 140},
        showlegend=reference is not None,
        # Below the chart: in the page's half-width column a legend on top
        # wraps onto two rows and runs into the linear/log buttons.
        legend={"orientation": "h", "x": 0.0, "xanchor": "left", "y": -0.06, "yanchor": "top"},
        barmode="group",
        updatemenus=[_axis_toggle()],
    )
    return fig


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


#: Line colours of the two curves compared on every row: the portfolio in a
#: saturated blue, the benchmark in a neutral grey so the eye lands on the
#: portfolio first. The excess rows use their own accent.
PORTFOLIO_COLOUR = "#1f77b4"
BENCHMARK_COLOUR = "#8a8f98"
EXCESS_COLOUR = "#d9822b"

#: Trace names shown in the legend when a benchmark is drawn; everything
#: else (markers, the drawdown rows) is identified by its row title.
_LEGEND_TRACES = ("equity", "benchmark_equity", "monthly_return", "benchmark_monthly_return")


def _aligned_benchmark(
    benchmark_value: xr.DataArray | None, equity: pd.Series
) -> pd.Series | None:
    """Return the benchmark value on the equity index, or ``None`` to draw none.

    The backtester values both on the same bars; a benchmark that does not
    cover the equity index is left off the page rather than raising, because
    an exception here would discard the whole staged run directory.
    """
    if benchmark_value is None:
        return None
    series = benchmark_value.to_pandas()
    if not isinstance(series, pd.Series) or series.empty:
        return None
    series = series.reindex(equity.index)
    if not series.notna().all():
        return None
    return series


def _excess_curves(equity: pd.Series, reference: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return ``(excess return, excess drawdown)`` of the portfolio over the benchmark.

    The relative NAV is portfolio value over benchmark value. Both start from
    the same capital, so it is 1 before the first bar and its final value
    minus 1 is the ``Excess Return [%]`` metric of ``relative.whole`` (as a
    fraction here, a percent there); its drawdown is measured from a running
    peak that starts at 1, like ``Excess Max Drawdown [%]``.
    """
    relative = equity / reference
    excess = relative - 1.0
    peak = relative.cummax().clip(lower=1.0)
    return excess, relative / peak - 1.0


def _add_benchmark(
    fig,
    equity: pd.Series,
    reference: pd.Series,
    name: str,
    init_cash: float | None,
    rows: dict,
) -> None:
    """Add the benchmark NAV and its drawdown beside the portfolio's."""
    label = html.escape(str(name))
    if init_cash:
        multiple = reference.values / float(init_cash)
        hover = (
            f"%{{x}}<br>{label} value %{{y:,.2f}}"
            "<br>%{customdata:,.4f}x initial<extra></extra>"
        )
    else:
        multiple = [None] * len(reference)
        hover = f"%{{x}}<br>{label} value %{{y:,.2f}}<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=reference.index,
            y=reference.values,
            name="benchmark_equity",
            mode="lines",
            line={"color": BENCHMARK_COLOUR, "dash": "dash"},
            customdata=multiple,
            hovertemplate=hover,
        ),
        row=rows["equity"],
        col=1,
    )

    benchmark_drawdown = reference / reference.cummax() - 1.0
    fig.add_trace(
        go.Scatter(
            x=benchmark_drawdown.index,
            y=benchmark_drawdown.values,
            name="benchmark_drawdown",
            mode="lines",
            line={"color": BENCHMARK_COLOUR, "dash": "dash"},
            hovertemplate=f"%{{x}}<br>{label} drawdown %{{y:.2%}}<extra></extra>",
        ),
        row=rows["drawdown"],
        col=1,
    )


def _add_equity(fig, equity: pd.Series, init_cash: float | None) -> None:
    """Add the equity trace, with the multiple of ``init_cash`` in the hover text.

    ``y`` stays the persisted portfolio value so the page cannot disagree with
    the stored equity; the multiple of initial capital, which makes a run that
    compounded by orders of magnitude legible, rides along as ``customdata``.
    """
    if init_cash:
        multiple = equity.values / float(init_cash)
        hover = "%{x}<br>value %{y:,.2f}<br>%{customdata:,.4f}x initial<extra></extra>"
    else:
        multiple = [None] * len(equity)
        hover = "%{x}<br>value %{y:,.2f}<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="equity",
            mode="lines",
            line={"color": PORTFOLIO_COLOUR},
            customdata=multiple,
            hovertemplate=hover,
        ),
        row=1,
        col=1,
    )


#: Colour of both drawdown triangles. They are the two ends of one measurement
#: (the valley and the recovery of a single episode) and are told apart by
#: shape, up versus down, not by colour.
SPAN_COLOUR = "#8e44ad"


def _add_drawdown_span(fig, equity: pd.Series, span) -> None:
    """Add the two triangles marking the deepest drawdown to the equity row.

    The caller has chosen and measured the episode; this only draws it. The
    up triangle sits on the valley and the down triangle on the recovery bar,
    so the distance between them is the time from the bottom back to even.
    ``bars`` is a bar count (trading days on a daily panel) and the hover text
    says so; no calendar duration is shown, since the time axis spans more
    calendar days than the span lasts bars. Every key is read with ``.get``
    and an endpoint missing from the equity axis drops only its own marker,
    because an exception here would discard the whole run directory. The two
    markers are separate named traces: only the end marker can say that the
    drawdown never recovered.
    """
    if not span:
        return

    bars = span.get("bars")
    depth = span.get("depth")
    length = "an unknown number of" if bars is None else str(bars)
    depth_text = "" if depth is None else f"<br>depth {float(depth):.2%}"

    if span.get("recovered"):
        tail = (
            "deepest drawdown recovers here"
            f"<br>{length} trading days (bars) from its deepest point"
        )
    else:
        tail = (
            "deepest drawdown had not recovered by the last bar"
            f"<br>{length} trading days (bars) since its deepest point"
        )

    endpoints = (
        (
            "valley",
            "deepest_drawdown_valley",
            "triangle-up",
            "deepest drawdown bottoms here",
        ),
        ("end", "deepest_drawdown_end", "triangle-down", tail),
    )
    for key, name, marker_symbol, text in endpoints:
        label = span.get(key)
        if label is None:
            continue
        stamp = pd.Timestamp(str(label))
        if stamp not in equity.index:
            continue
        suffix = depth_text if key == "end" else ""
        fig.add_trace(
            go.Scatter(
                x=[stamp],
                y=[equity.loc[stamp]],
                name=name,
                mode="markers",
                marker={"symbol": marker_symbol, "size": 11, "color": SPAN_COLOUR},
                hovertemplate=f"%{{x}}<br>{text}{suffix}<extra></extra>",
            ),
            row=1,
            col=1,
        )


def _monthly_series(returns: xr.DataArray | None) -> pd.Series | None:
    """Return the compounded return of each calendar month, indexed by period.

    Both the monthly bar panel and the heatmap read this one series, so they
    cannot drift apart. Months are formed with ``to_period("M")`` rather than
    a resample alias, whose spelling changed across pandas versions. NaN bars
    are dropped first: ``prod`` skips NaN, so a month of only NaN bars would
    otherwise compound to a fabricated ``0.0`` instead of being absent.
    Returns ``None`` when there is nothing to compute.
    """
    if returns is None:
        return None
    series = returns.to_pandas().dropna()
    if series.empty:
        return None
    index = pd.DatetimeIndex(series.index)
    return (1.0 + series).groupby(index.to_period("M")).prod() - 1.0


def _add_monthly_returns(
    fig,
    returns: xr.DataArray | None,
    *,
    row: int = 3,
    benchmark_returns: xr.DataArray | None = None,
    benchmark_name: str = "benchmark",
) -> None:
    """Add the per-calendar-month compounded returns as a bar row.

    A short window legitimately yields one or two bars; that shows at a glance
    that a run's whole profit landed in a single month. Each bar is coloured
    by its sign with the same constants as the heatmap, so a month reads the
    same colour in both panels. With ``benchmark_returns`` the benchmark's
    months stand beside the portfolio's in grey.
    """
    monthly = _monthly_series(returns)
    if monthly is None or monthly.empty:
        return
    fig.add_trace(
        go.Bar(
            x=[period.to_timestamp() for period in monthly.index],
            y=monthly.values,
            marker={"color": [_sign_colour(value) for value in monthly.values]},
            name="monthly_return",
            hovertemplate="%{x|%Y-%m}<br>%{y:.2%}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    reference = _monthly_series(benchmark_returns)
    if reference is None or reference.empty:
        return
    label = html.escape(str(benchmark_name))
    fig.add_trace(
        go.Bar(
            x=[period.to_timestamp() for period in reference.index],
            y=reference.values,
            marker={"color": BENCHMARK_COLOUR},
            name="benchmark_monthly_return",
            hovertemplate=f"%{{x|%Y-%m}}<br>{label} %{{y:.2%}}<extra></extra>",
        ),
        row=row,
        col=1,
    )


#: The colours of a monthly return's sign, shared by the monthly bars and the
#: heatmap so the two views of the same numbers cannot disagree. Green is a
#: gain and red a loss (the Western convention). The midpoint is a neutral
#: grey, which the heatmap pins to zero with ``zmid=0.0``. The two poles have
#: matched luminance and stay separable under red-green colour-vision
#: deficiency; sign is also carried without colour, by bar direction and by
#: the hover percentage.
GAIN_COLOUR = "#1b8a5a"
LOSS_COLOUR = "#e03b30"
NEUTRAL_COLOUR = "#f0efec"

#: The heatmap's diverging colorscale: the most negative value is red, zero
#: (through ``zmid=0.0``) is grey, the most positive value is green.
RETURN_COLOURSCALE = (
    (0.0, LOSS_COLOUR),
    (0.5, NEUTRAL_COLOUR),
    (1.0, GAIN_COLOUR),
)


def _sign_colour(value: float) -> str:
    """Return the colour for one monthly return: gain, loss or neutral at zero.

    Exactly zero takes the neutral grey so that a month spent wholly in cash
    matches the heatmap's zero cell. A NaN fails both comparisons and falls
    through to neutral rather than raising.
    """
    if value > 0:
        return GAIN_COLOUR
    if value < 0:
        return LOSS_COLOUR
    return NEUTRAL_COLOUR


#: The heatmap's month columns, as two-digit strings so they sort and read the
#: same way and plotly treats them as categories.
MONTH_LABELS = [f"{month:02d}" for month in range(1, 13)]

#: Caption above the heatmap. Escaped on the way onto the page like every
#: other non-plotly string.
HEATMAP_CAPTION = "Monthly returns by year"


def _monthly_grid(monthly: pd.Series) -> tuple[list[int], list[list[float | None]]]:
    """Return ``(years, z)``: one row of twelve cells per year in ``monthly``.

    A month the run did not cover stays ``None``, which serialises to JSON
    null and renders as an empty cell rather than a fabricated ``0.0``.
    ``period.month`` is 1-based, hence the ``- 1`` on the column index.
    """
    years = sorted({period.year for period in monthly.index})
    row_of = {year: row for row, year in enumerate(years)}
    grid: list[list[float | None]] = [[None] * 12 for _ in years]
    for period, value in monthly.items():
        grid[row_of[period.year]][period.month - 1] = float(value)
    return years, grid


def _monthly_heatmap_div(returns: xr.DataArray | None) -> str:
    """Return the year-by-month returns heatmap as its own HTML fragment.

    Returns the empty string when there is nothing to draw. The numbers come
    from ``_monthly_series``, the same helper the bar row uses. It is a
    separate figure rather than a fourth subplot row because ``shared_xaxes``
    is figure-wide: a fourth row would bind the three datetime axes to this
    chart's categorical month axis. ``include_plotlyjs=False`` reuses the
    library the main div already loads.
    """
    monthly = _monthly_series(returns)
    if monthly is None or monthly.empty:
        return ""
    years, z = _monthly_grid(monthly)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=MONTH_LABELS,
            y=[str(year) for year in years],
            name="monthly_return_heatmap",
            colorscale=RETURN_COLOURSCALE,
            zmid=0.0,
            colorbar={"tickformat": ".1%"},
            hoverongaps=False,
            hovertemplate="%{y}-%{x}<br>%{z:.2%}<extra></extra>",
        )
    )
    # About 36px per year row plus room for the axis labels and margins, with
    # a floor so a one-year run is still a legible strip.
    fig.update_layout(
        height=max(220, 120 + 36 * len(years)),
        margin={"t": 20, "b": 40},
    )
    fig.update_xaxes(type="category", title_text="month")
    fig.update_yaxes(type="category", autorange="reversed", title_text="year")
    _clean(fig)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _axis_toggle() -> dict:
    """Return the linear/log button group for the equity axis.

    Linear is the default because a run that lost everything has a
    non-positive value that renders an empty log panel. Log is one click away
    and turns a curve that compounded by orders of magnitude from a flat line
    with a final spike into a readable slope.
    """
    return {
        "type": "buttons",
        "direction": "right",
        "x": 0.0,
        "y": 1.12,
        "xanchor": "left",
        "yanchor": "top",
        "showactive": True,
        "buttons": [
            {"label": "linear", "method": "relayout", "args": [{"yaxis.type": "linear"}]},
            {"label": "log", "method": "relayout", "args": [{"yaxis.type": "log"}]},
        ],
    }


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def _escape(value: object) -> str:
    """Return ``str(value)`` with every HTML-significant character escaped."""
    return html.escape(str(value))


def _flatten(value: dict, prefix: str = "") -> dict:
    """Flatten nested dicts into ``{"dotted.path": leaf}``.

    A sub-dict becomes rows under its own name, so the table needs no
    knowledge of any metric name.
    """
    flat: dict = {}
    for key, item in value.items():
        path = f"{prefix}{key}"
        if isinstance(item, dict):
            flat.update(_flatten(item, f"{path}."))
        else:
            flat[path] = item
    return flat


#: How a metric is displayed: ``(group, label, unit, definition)``. Only the
#: rows' presentation lives here; the values are whatever the run computed. A
#: key the catalogue does not know is still shown, in the "Other" group, with
#: a unit guessed from its name and type (see ``_guess_unit``).
_STRATEGY_METRICS: dict[str, tuple[str, str, str, str]] = {
    "Start Value": ("Returns", "Start value", "money", "The portfolio's value at the first bar."),
    "End Value": ("Returns", "End value", "money", "Portfolio value at the last bar."),
    "Total Return [%]": ("Returns", "Total return", "pct", "How much the portfolio grew over the whole window, compounding included: 42% means 1.00 became 1.42."),
    "Annualized Return [%]": ("Returns", "Annualised return", "pct", "The yearly growth rate that compounds to the same total return over the window."),
    "Annualized Volatility [%]": ("Risk", "Annualised volatility", "pct",
                                  "How much the daily returns swing, scaled to a year: the standard deviation of the per-bar returns, annualised."),
    "Max Drawdown [%]": ("Risk", "Max drawdown", "neg_pct",
                         "The worst fall from a previous high to a later low: what buying at the worst moment would have lost. Negative."),
    "Max Drawdown Duration": ("Risk", "Longest drawdown", "days",
                              "The longest time the portfolio stayed below a previous high before getting back to it, in calendar days."),
    "Value at Risk": ("Risk", "Value at risk (95%)", "frac_pct", "A bad day: on 95% of bars the return was better than this; the worst 5% were worse (the 5th percentile)."),
    "Skew": ("Risk", "Skew", "ratio", "Whether big moves tend to be gains (positive) or losses (negative). Negative skew means occasional sharp drops."),
    "Kurtosis": ("Risk", "Kurtosis", "ratio", "How often extreme bars happen compared with a normal distribution; above 0 means fatter tails, more surprises."),
    "Sharpe Ratio": ("Risk-adjusted", "Sharpe ratio", "ratio",
                     "Return per unit of risk: the annualised mean return over the annualised volatility (no risk-free rate subtracted). Higher is better; about 1 is good."),
    "Sortino Ratio": ("Risk-adjusted", "Sortino ratio", "ratio",
                      "Like the Sharpe ratio, but only the downside swings count as risk, so upside volatility is not penalised."),
    "Calmar Ratio": ("Risk-adjusted", "Calmar ratio", "ratio", "The annualised return over the max drawdown: how much was earned per unit of worst loss."),
    "Omega Ratio": ("Risk-adjusted", "Omega ratio", "ratio",
                    "All the gains over all the losses of the per-bar returns. Above 1, gains outweighed losses."),
    "Tail Ratio": ("Risk-adjusted", "Tail ratio", "ratio",
                   "The size of the best 5% of bars against the worst 5% (95th percentile over the absolute 5th). Above 1, big up bars were larger than big down bars."),
    "Common Sense Ratio": ("Risk-adjusted", "Common sense ratio", "ratio", "The profit factor times the tail ratio; above 1 suggests the edge survives the bad days."),
    "Rebalance Win Rate [%]": ("Win rates", "Rebalances with a gain", "pct1",
                               "The share of holding periods that ended with a gain. A period runs from the "
                               "fill of one decided rebalance to the day before the next one's; a delisting "
                               "settlement or a held rebalance does not start one."),
    "Monthly Win Rate [%]": ("Win rates", "Months with a gain", "pct1",
                             "The share of calendar months that ended with a gain."),
}

#: Which side of a strategy-vs-benchmark row is better: +1 when the larger
#: value is, -1 when the smaller is. A metric absent here (skew, kurtosis,
#: the start value) has no better side and marks nothing. Drawdowns and the
#: value at risk are negative, so the larger (closer to zero) is better.
_BETTER = {
    "End Value": 1, "Total Return [%]": 1, "Annualized Return [%]": 1,
    "Annualized Volatility [%]": -1, "Max Drawdown [%]": 1, "Max Drawdown Duration": -1,
    "Value at Risk": 1, "Sharpe Ratio": 1, "Sortino Ratio": 1, "Calmar Ratio": 1,
    "Omega Ratio": 1, "Tail Ratio": 1, "Common Sense Ratio": 1,
}

#: Strategy rows the "Relative to" table replaces when a benchmark ran.
_WITHOUT_BENCHMARK_ONLY = frozenset({"Rebalance Win Rate [%]", "Monthly Win Rate [%]"})

#: The strategy's trading rows, from the same slice as the strategy metrics.
_TRADING_METRICS: dict[str, tuple[str, str, str]] = {
    "Annualized Turnover [%]": ("Annualised turnover", "pct0",
                                "How much the rebalances bought and sold in a year, as a share of the portfolio "
                                "value: the total turnover over the window's length in years."),
    "Turnover per Rebalance [%]": ("Turnover per rebalance", "pct0",
                                   "Buys plus sells at the average decided rebalance, as a share of the portfolio: "
                                   "buying a full book from cash is 100%, replacing the whole book about 200%. A "
                                   "rebalance that kept its book counts as 0; a held rebalance is not counted."),
    "Total Turnover [%]": ("Total turnover", "pct0",
                           "Everything the rebalances bought and sold over the window, as a share of the portfolio "
                           "value before each fill. Delisting settlements are not counted."),
    "Traded Notional": ("Traded notional", "money", "The total value of every fill over the window."),
    "Total Fees Paid": ("Fees paid", "money", "The fees and slippage the simulation charged, in money."),
    "Max Gross Exposure [%]": ("Max gross exposure", "pct0", "The largest total of long plus short positions held, as a share of the portfolio value."),
    "Total Orders": ("Orders filled", "int", "The number of fills that actually happened over the window."),
    "Total Trades": ("Round trips", "int", "How many times a symbol was bought and later fully sold (entry to flat), open ones included."),
    "Total Closed Trades": ("Round trips closed", "int", "Round trips that ended with the position fully sold."),
    "Total Open Trades": ("Round trips open", "int", "Round trips still held at the last bar."),
    "Open Trade PnL": ("Open round-trip P&L", "money", "The profit, not yet taken, on the positions still held."),
    "Win Rate [%]": ("Round-trip win rate", "pct0", "The share of finished round trips that made money."),
    "Best Trade [%]": ("Best round trip", "pct", "The return of the best finished round trip."),
    "Worst Trade [%]": ("Worst round trip", "pct", "The return of the worst finished round trip."),
    "Avg Winning Trade [%]": ("Avg winning round trip", "pct", "The average return of the round trips that made money."),
    "Avg Losing Trade [%]": ("Avg losing round trip", "pct", "The average return of the round trips that lost money."),
    "Avg Winning Trade Duration": ("Avg winning round-trip duration", "days1",
                                   "How long the winning positions were held, on average."),
    "Avg Losing Trade Duration": ("Avg losing round-trip duration", "days1",
                                  "How long the losing positions were held, on average."),
    "Profit Factor": ("Profit factor", "ratio", "Money made on winning round trips over money lost on losing ones. Above 1, the trading made money."),
    "Expectancy": ("Expectancy", "money", "The average profit of a finished round trip, in money."),
}

#: The relative rows, from ``metrics["relative"]``.
_RELATIVE_METRICS: dict[str, tuple[str, str, str]] = {
    "Excess Return [%]": ("Excess return (geometric)", "spct",
                          "How far the portfolio ended ahead of (or behind) the benchmark: strategy value over benchmark value, minus 1, at the last bar."),
    "Annualized Excess Return [%]": ("Annualised excess return", "spct",
                                     "The excess over the benchmark expressed per year (the geometric excess compounded to one year)."),
    "Total Return Difference [%]": ("Total return difference (arithmetic)", "spct",
                                    "The strategy's total return minus the benchmark's; it differs from the "
                                    "geometric excess because of compounding."),
    "Excess Max Drawdown [%]": ("Excess max drawdown", "pct",
                                "The worst stretch of falling behind the benchmark: the deepest fall of "
                                "strategy value over benchmark value from its high. Negative."),
    "Tracking Error [%]": ("Tracking error", "pct", "How much the portfolio's returns wander from the benchmark's: the annualised standard deviation of the per-bar difference r - b."),
    "Information Ratio": ("Information ratio", "ratio", "Excess return per unit of tracking error: how consistently the portfolio beat the benchmark (the annualised mean of r - b over the tracking error). Above 0.5 is good."),
    "Beta": ("Beta", "ratio", "How much the portfolio moves when the benchmark moves 1%: the slope of its per-bar returns on the benchmark's."),
    "Correlation": ("Correlation", "ratio", "How closely the per-bar returns move with the benchmark's, from -1 (opposite) to 1 (in lockstep)."),
    "CAPM Alpha [%]": ("CAPM alpha", "spct", "The yearly return that moving with the benchmark does not explain: the annualised intercept of the beta regression."),
    "Win Rate vs Benchmark [%]": ("Bars beating the benchmark", "pct0", "The share of bars on which the portfolio did better than the benchmark."),
    "Rebalance Win Rate vs Benchmark [%]": ("Rebalances beating the benchmark", "pct1",
                                            "The share of holding periods (from one decided rebalance's fill to "
                                            "the next one's) in which the portfolio did better than the benchmark."),
    "Monthly Win Rate vs Benchmark [%]": ("Months beating the benchmark", "pct1",
                                          "The share of calendar months in which the portfolio did better than "
                                          "the benchmark."),
}

#: Keys every run carries that the page shows elsewhere (the dates-and-setup
#: table) or that are bookkeeping, so they make no "Other" row.
_HIDDEN_METRICS = frozenset({"Start", "End", "Period", "Bars"})

#: Relative keys that repeat the "Total return" row of the comparison table.
_RELATIVE_SHOWN_ELSEWHERE = frozenset({"Strategy Total Return [%]", "Benchmark Total Return [%]"})

#: The key rows of the in-sample vs out-of-sample table.
_SPLIT_METRICS = ("Total Return [%]", "Annualized Return [%]", "Annualized Volatility [%]",
                  "Sharpe Ratio", "Max Drawdown [%]")
_SPLIT_RELATIVE = ("Excess Return [%]", "Excess Max Drawdown [%]", "Information Ratio", "Beta")


def _number(value: object) -> float | None:
    """``value`` as a finite float, or ``None`` for anything that is not one.

    ``bool`` is rejected because it is an ``int`` subclass and would render as
    a plausible number; ``numbers.Real`` admits numpy scalars without
    importing numpy.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _days(value: object) -> float | None:
    """A duration (a ``pd.Timedelta`` or its string) in days, or ``None``."""
    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(delta) else delta / pd.Timedelta(days=1)


def _guess_unit(key: str, value: object) -> str:
    """The display unit of a metric the catalogues do not know."""
    if key.endswith("[%]"):
        return "pct"
    if value is pd.NaT or isinstance(value, pd.Timedelta):
        return "days1"
    if isinstance(value, bool):
        return "text"
    if isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, numbers.Real):
        return "num"
    if isinstance(value, str) and _days(value) is not None and "day" in value:
        return "days1"
    return "text"


def _format(value: object, unit: str) -> str:
    """Render one metric value in its unit; anything unrenderable becomes a dash.

    NaN and infinity become a dash rather than ``nan`` / ``inf``, which read
    like numbers. ``neg_pct`` shows a positive magnitude as a negative percent,
    so every drawdown on the page carries the same sign.
    """
    if value is None:
        return DASH
    if unit in ("days", "days1"):
        days = _days(value)
        if days is None:
            return DASH
        return f"{days:,.0f} d" if unit == "days" else f"{days:,.1f} d"
    if unit == "text":
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value)
        return text if text.strip() else DASH
    number = _number(value)
    if number is None:
        return DASH if not isinstance(value, str) or not value.strip() else str(value)
    if unit == "pct":
        return f"{number:,.2f}%"
    if unit == "spct":
        return f"{number:+,.2f}%" if number else "0.00%"
    if unit == "neg_pct":
        return f"{-abs(number):,.2f}%"
    if unit == "pct0":
        return f"{number:,.0f}%"
    if unit == "pct1":
        return f"{number:,.1f}%"
    if unit == "frac_pct":
        return f"{number * 100:,.2f}%"
    if unit == "ratio":
        return f"{number:,.2f}"
    if unit == "money":
        return f"{number:,.0f}"
    if unit == "int":
        return f"{int(round(number)):,}"
    # "num": a number of unknown meaning, legible at any magnitude.
    return f"{number:,.0f}" if abs(number) >= 1000 else f"{number:.4g}"


def _difference(later: object, earlier: object, unit: str) -> str:
    """``later - earlier`` in the unit's terms: percentage points for percents."""
    if unit in ("days", "days1"):
        a, b = _days(later), _days(earlier)
        return DASH if a is None or b is None else f"{a - b:+,.0f} d"
    a, b = _number(later), _number(earlier)
    if a is None or b is None:
        return DASH
    if unit == "neg_pct":
        a, b = -abs(a), -abs(b)
    if unit in ("pct", "spct", "neg_pct", "pct0", "pct1"):
        return f"{a - b:+,.2f} pp"
    if unit == "frac_pct":
        return f"{(a - b) * 100:+,.2f} pp"
    if unit == "money":
        return f"{a - b:+,.0f}"
    if unit == "int":
        return f"{int(round(a - b)):+,}"
    return f"{a - b:+,.2f}"


def _slice(block: object, name: str, *, fallback: bool) -> dict:
    """``block[name]`` as a flat dict; ``fallback`` fills a key missing from
    ``whole`` from ``out_of_sample``, for a run whose whole window is its
    out-of-sample part (vectorbt reports some period statistics per slice)."""
    if not isinstance(block, dict):
        return {}
    out = _flatten(block[name]) if isinstance(block.get(name), dict) else {}
    if fallback and isinstance(block.get("out_of_sample"), dict):
        for key, value in _flatten(block["out_of_sample"]).items():
            if out.get(key) is None:
                out[key] = value
    return out


def _has_in_sample(metrics: dict) -> bool:
    """Whether the run's metrics carry an in-sample slice with any value."""
    return isinstance(metrics.get("in_sample"), dict) and bool(metrics["in_sample"])


def _headline(block: object, metrics: dict) -> dict:
    """The slice the cards and the main tables show, from ``block``.

    With an in-sample part the out-of-sample slice, the bars the model never
    saw; otherwise the whole window, which is then all out-of-sample, gaps
    filled from ``out_of_sample``.
    """
    if _has_in_sample(metrics):
        return _slice(block, "out_of_sample", fallback=False)
    return _slice(block, "whole", fallback=True)


def _suffix(metrics: dict) -> str:
    """`` (out-of-sample)`` on the headings when the headline is not the whole window."""
    return " (out-of-sample)" if _has_in_sample(metrics) else ""


def _row(label: str, definition: str, cells: list[str], better: list[bool] | None = None) -> str:
    """One table row: the label with its definition on hover, then the cells;
    a cell whose ``better`` flag is set is marked as the better side."""
    flags = better or [False] * len(cells)
    tds = "".join(
        ('<td class="better">' if flag else "<td>") + f"{_escape(cell)}</td>"
        for cell, flag in zip(cells, flags + [False] * (len(cells) - len(flags)))
    )
    return f'<tr><th title="{_escape(definition)}">{_escape(label)}</th>{tds}</tr>'


def _table(heading: str, header: list[str], rows: list[str]) -> str:
    """A captioned metric table, or the empty string when it has no row."""
    if not any(not row.startswith('<tr class="group">') for row in rows):
        return ""
    head = "".join(f"<th>{_escape(cell)}</th>" for cell in header)
    return (
        f"  <h2>{_escape(heading)}</h2>\n"
        '  <table class="metrics">\n'
        f"    <thead><tr>{head}</tr></thead>\n"
        "    <tbody>\n      " + "\n      ".join(rows) + "\n    </tbody>\n"
        "  </table>\n"
    )


def _group_row(name: str, width: int) -> str:
    """A full-width row naming the group of the rows below it."""
    return f'<tr class="group"><td colspan="{width}">{_escape(name)}</td></tr>'


def _comparison_section(metrics: dict, benchmark_name: str | None) -> str:
    """Strategy (and benchmark and difference) by group, then an "Other" table.

    The columns are the headline slice (``_headline``) of the strategy and
    of ``metrics["benchmark"]``, and their difference.
    """
    strategy = _headline(metrics, metrics)
    benchmark = _headline(metrics.get("benchmark"), metrics) if benchmark_name else {}
    width = 4 if benchmark_name else 2
    rows, group = [], None
    for key, (name, label, unit, definition) in _STRATEGY_METRICS.items():
        if strategy.get(key) is None and benchmark.get(key) is None:
            continue
        if benchmark_name and key in _WITHOUT_BENCHMARK_ONLY:
            continue
        if name != group:
            rows.append(_group_row(name, width))
            group = name
        cells = [_format(strategy.get(key), unit)]
        better = None
        if benchmark_name:
            cells += [
                _format(benchmark.get(key), unit),
                _difference(strategy.get(key), benchmark.get(key), unit),
            ]
            better = _better_side(key, strategy.get(key), benchmark.get(key))
        rows.append(_row(label, definition, cells, better))
    header = ["", "Strategy"] + ([benchmark_name, "Difference"] if benchmark_name else [])
    heading = (f"Strategy vs {benchmark_name}" if benchmark_name else "Strategy") + _suffix(metrics)
    out = _table(heading, header, rows)

    return out + _other_section(metrics, benchmark_name, strategy, benchmark)


def _other_section(metrics: dict, benchmark_name: str | None, strategy: dict, benchmark: dict) -> str:
    """Every headline metric no catalogue knows, with a unit guessed from it.

    The strategy's keys are listed as they are; a benchmark or relative key
    is prefixed with where it came from, so the three sources share one
    table.
    """
    known = set(_STRATEGY_METRICS) | set(_TRADING_METRICS) | _HIDDEN_METRICS
    sources = [("", strategy, known)]
    if benchmark_name:
        sources += [
            (f"{benchmark_name}: ", benchmark, known),
            ("relative: ", _headline(metrics.get("relative"), metrics), set(_RELATIVE_METRICS) | _HIDDEN_METRICS | _RELATIVE_SHOWN_ELSEWHERE),
        ]
    rows = [
        _row(f"{prefix}{key}", "Reported by the run; the report has no description of it.",
             [_format(value, _guess_unit(key, value))])
        for prefix, block, catalogue in sources
        for key, value in block.items()
        if key not in catalogue
    ]
    return _table("Other", ["", "Value"], rows)


def _better_side(key: str, strategy: object, benchmark: object) -> list[bool] | None:
    """``[strategy is better, benchmark is better]``, or ``None`` when the row
    has no direction, a value is missing, or the two are equal."""
    direction = _BETTER.get(key)
    if direction is None:
        return None
    if key == "Max Drawdown Duration":
        a, b = _days(strategy), _days(benchmark)
    else:
        a, b = _number(strategy), _number(benchmark)
        if key == "Max Drawdown [%]" and a is not None and b is not None:
            a, b = -abs(a), -abs(b)
    if a is None or b is None or a == b:
        return None
    strategy_better = (a > b) == (direction > 0)
    return [strategy_better, not strategy_better]


def _relative_section(metrics: dict, benchmark_name: str | None) -> str:
    """The excess over the benchmark, from ``metrics["relative"]``."""
    if not benchmark_name:
        return ""
    relative = _headline(metrics.get("relative"), metrics)
    rows = [
        _row(label, definition, [_format(relative.get(key), unit)])
        for key, (label, unit, definition) in _RELATIVE_METRICS.items()
        if relative.get(key) is not None
    ]
    return _table(f"Relative to {benchmark_name}{_suffix(metrics)}", ["", "Value"], rows)


def _trading_section(metrics: dict) -> str:
    """Turnover, costs and round trips, then execution and construction counts."""
    strategy = _headline(metrics, metrics)
    rows = [
        _row(label, definition, [_format(strategy.get(key), unit)])
        for key, (label, unit, definition) in _TRADING_METRICS.items()
        if strategy.get(key) is not None
    ]
    execution = metrics.get("execution")
    if isinstance(execution, dict) and execution.get("rejected_order_count") is not None:
        rows.append(_row("Orders rejected",
                         "Orders that could not be filled because the symbol had no price at the next bar; "
                         "the holding was kept.",
                         [_format(execution["rejected_order_count"], "int")]))
    construction = metrics.get("portfolio_construction")
    if isinstance(construction, dict):
        if construction.get("failed_bar_count") is not None:
            rows.append(_row("Rebalances held after a failure",
                             "Rebalances on which the portfolio rule failed (for example, the optimiser "
                             "found no solution), so the backtest kept the previous holdings.",
                             [_format(construction["failed_bar_count"], "int")]))
        for event, record in construction.items():
            if isinstance(record, dict) and "count" in record:
                bars = record.get("bars")
                on = f" on {len(bars):,} bar(s)" if isinstance(bars, list) else ""
                rows.append(_row(f"Constructor event: {event}",
                                 "Symbols the portfolio constructor reported this event for, summed over "
                                 "the rebalance bars it happened on.",
                                 [_format(record["count"], "int") + on]))
    return _table(f"Trading{_suffix(metrics)}", ["", "Value"], rows)


def _split_section(metrics: dict, benchmark_name: str | None) -> str:
    """Key metrics per slice, only when the run has an in-sample part.

    The columns are the in-sample slice, the out-of-sample slice, their
    difference (out-of-sample minus in-sample, in percentage points for a
    percent) and the whole window.
    """
    if not _has_in_sample(metrics):
        return ""
    slices = ("in_sample", "out_of_sample", "whole")
    rows = []
    for block, keys, catalogue in (
        (metrics, _SPLIT_METRICS, {k: v[1:] for k, v in _STRATEGY_METRICS.items()}),
        (metrics.get("relative") if benchmark_name else None, _SPLIT_RELATIVE, _RELATIVE_METRICS),
    ):
        columns = [_slice(block, name, fallback=False) for name in slices]
        for key in keys:
            label, unit, definition = catalogue[key]
            if all(column.get(key) is None for column in columns):
                continue
            first, second, whole = (column.get(key) for column in columns)
            rows.append(_row(label, definition, [
                _format(first, unit), _format(second, unit), _difference(second, first, unit), _format(whole, unit),
            ]))
    return _table("In-sample vs out-of-sample", ["", "In-sample", "Out-of-sample", "Difference", "Whole"], rows)


#: What each headline card means, shown on hover.
_KPI_TIPS = {
    "Total return": "How much the portfolio grew over the window, compounding included: 42% means 1.00 "
                    "became 1.42. The note is the benchmark's, or the end value.",
    "Annualised return": "The yearly growth rate that compounds to the same total return.",
    "Excess return": "How far the portfolio ended ahead of (or behind) the benchmark: strategy value over "
                     "benchmark value, minus 1. The note is the same per year.",
    "Information ratio": "Excess return per unit of tracking error: how consistently the portfolio beat the "
                         "benchmark. Above 0.5 is good; negative means it lagged.",
    "Win rate": "The share of holding periods (from one decided rebalance's fill to the next one's) that "
                "beat the benchmark, or made money without one. The note is the same per calendar month.",
    "Sharpe ratio": "Return per unit of risk: the annualised mean return over the annualised volatility. "
                    "Higher is better; about 1 is good.",
    "Max drawdown": "The worst fall from a previous high to a later low: what buying at the worst moment "
                    "would have lost.",
    "Beta": "How much the portfolio moves when the benchmark moves 1%. The note is how closely they move "
            "together (correlation, -1 to 1).",
    "Volatility": "How much the daily returns swing, scaled to a year.",
    "Turnover / year": "How much the rebalances bought and sold in a year, as a share of the portfolio: 700% "
                       "replaces the book about 3.5 times a year. The note is the fees and slippage paid.",
}


def _card(label: str, value: str, note: str, sign: float | None = None) -> str:
    """One KPI card, its explanation in ``title``; ``sign`` colours the value green above zero, red below."""
    cls = "" if sign is None or sign == 0 else (" pos" if sign > 0 else " neg")
    tip = _KPI_TIPS.get(label)
    attribute = f' title="{_escape(tip)}"' if tip else ""
    return (
        f'<div class="kpi"{attribute}><div class="kl">{_escape(label)}</div>'
        f'<div class="kv{cls}">{_escape(value)}</div><div class="ks">{_escape(note)}</div></div>'
    )


def _kpi_section(metrics: dict | None, benchmark_name: str | None) -> str:
    """The row of headline numbers above the tables; empty without metrics."""
    if not isinstance(metrics, dict) or not isinstance(metrics.get("whole"), dict):
        return ""
    s = _headline(metrics, metrics)
    turnover = _card("Turnover / year", _format(s.get("Annualized Turnover [%]"), "pct0"),
                     f"fees {_format(s.get('Total Fees Paid'), 'money')}")
    if benchmark_name:
        b = _headline(metrics.get("benchmark"), metrics)
        r = _headline(metrics.get("relative"), metrics)
        cards = [
            _card("Total return", _format(s.get("Total Return [%]"), "pct"),
                  f"{benchmark_name} {_format(b.get('Total Return [%]'), 'pct')}"),
            _card("Excess return", _format(r.get("Excess Return [%]"), "spct"),
                  f"annualised {_format(r.get('Annualized Excess Return [%]'), 'spct')}",
                  _number(r.get("Excess Return [%]"))),
            _card("Information ratio", _format(r.get("Information Ratio"), "ratio"),
                  f"tracking error {_format(r.get('Tracking Error [%]'), 'pct')}"),
            _card("Win rate", _format(r.get("Rebalance Win Rate vs Benchmark [%]"), "pct1"),
                  f"monthly {_format(r.get('Monthly Win Rate vs Benchmark [%]'), 'pct1')}"),
            _card("Sharpe ratio", _format(s.get("Sharpe Ratio"), "ratio"),
                  f"{benchmark_name} {_format(b.get('Sharpe Ratio'), 'ratio')}"),
            _card("Max drawdown", _format(s.get("Max Drawdown [%]"), "neg_pct"),
                  f"{benchmark_name} {_format(b.get('Max Drawdown [%]'), 'neg_pct')}"),
            _card("Beta", _format(r.get("Beta"), "ratio"), f"correlation {_format(r.get('Correlation'), 'ratio')}"),
            turnover,
        ]
    else:
        cards = [
            # The out-of-sample slice ends on the last bar too, but only the
            # whole window reports the value there.
            _card("Total return", _format(s.get("Total Return [%]"), "pct"),
                  f"end value {_format(_slice(metrics, 'whole', fallback=True).get('End Value'), 'money')}"),
            _card("Annualised return", _format(s.get("Annualized Return [%]"), "pct"), ""),
            _card("Win rate", _format(s.get("Rebalance Win Rate [%]"), "pct1"),
                  f"monthly {_format(s.get('Monthly Win Rate [%]'), 'pct1')}"),
            _card("Sharpe ratio", _format(s.get("Sharpe Ratio"), "ratio"),
                  f"Sortino {_format(s.get('Sortino Ratio'), 'ratio')}"),
            _card("Max drawdown", _format(s.get("Max Drawdown [%]"), "neg_pct"),
                  f"longest {_format(s.get('Max Drawdown Duration'), 'days')}"),
            _card("Volatility", _format(s.get("Annualized Volatility [%]"), "pct"), "annualised"),
            turnover,
        ]
    return '  <div class="kpis">' + "".join(cards) + "</div>\n"


# ---------------------------------------------------------------------------
# excess, rolling and portfolio figures
# ---------------------------------------------------------------------------


def _excess_figure(equity: pd.Series, reference: pd.Series, benchmark_name: str) -> go.Figure:
    """Cumulative excess return (log / arithmetic toggle) above the excess drawdown.

    Log is ``cumsum(log((1 + r) / (1 + b)))``, whose ``exp`` minus 1 is the
    geometric excess of the tables; arithmetic is ``cumsum(r - b)``, whose
    slope is the mean excess per bar, the way a cumulative IC is read. The
    per-bar returns are the equity's and the benchmark's own changes, so the
    curve follows the drawn values exactly.
    """
    r = equity.pct_change().fillna(0.0)
    b = reference.pct_change().fillna(0.0)
    # A ratio at or below zero (a value that reached zero or went negative)
    # has no log; it becomes NaN rather than raising inside the staged run.
    ratio = (1.0 + r) / (1.0 + b)
    log_cum = ratio.where(ratio > 0).map(math.log).cumsum()
    arith_cum = (r - b).cumsum()
    _, excess_drawdown = _excess_curves(equity, reference)
    label = html.escape(str(benchmark_name))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(
        x=log_cum.index, y=log_cum.values, name="cumulative_excess_log", mode="lines",
        line={"color": EXCESS_COLOUR},
        hovertemplate=f"%{{x}}<br>cumulative log excess vs {label} %{{y:.2%}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=arith_cum.index, y=arith_cum.values, name="cumulative_excess_arithmetic", mode="lines",
        line={"color": PORTFOLIO_COLOUR}, visible=False,
        hovertemplate=f"%{{x}}<br>cumulative excess vs {label} %{{y:.2%}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=excess_drawdown.index, y=excess_drawdown.values, name="excess_drawdown", mode="lines",
        fill="tozeroy", line={"color": LOSS_COLOUR},
        hovertemplate=f"%{{x}}<br>excess drawdown vs {label} %{{y:.2%}}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0.0, line={"color": BENCHMARK_COLOUR, "width": 1}, row=1, col=1)
    fig.update_yaxes(title_text=f"cumulative excess vs {label}", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text=f"excess drawdown vs {label}", tickformat=".0%", row=2, col=1)
    fig.update_layout(
        height=620, showlegend=False,
        updatemenus=[{
            "type": "buttons", "direction": "right", "x": 1.0, "y": 1.1, "xanchor": "right",
            "yanchor": "top", "showactive": True,
            "buttons": [
                {"label": "Log", "method": "update", "args": [{"visible": [True, False, True]}]},
                {"label": "Arithmetic", "method": "update", "args": [{"visible": [False, True, True]}]},
            ],
        }],
    )
    return fig


def _attribution_table(block: dict, benchmark_name: str | None) -> str:
    """The excess split into its parts, as annualised log growth."""
    parts = block.get("decomposition") or {}
    against = benchmark_name or "the universe"
    rows = [
        _row(label, definition, [_format(100 * parts[key], "spct")])
        for key, label, definition in (
            ("universe", "Universe vs benchmark",
             "The equal-weighted universe over the benchmark: earned or lost whatever is picked."),
            ("selection", "Selection",
             "The weights simulated without costs over the equal-weighted universe: what the scores and the rule add."),
            ("costs", "Costs", "The strategy after fees and slippage over the same weights without them."),
            ("total", "Total", f"The strategy over {against}; the sum of the rows above."),
        )
        if parts.get(key) is not None
    ]
    return _table(f"Attribution vs {against} (annualised log growth; {block.get('groups')} score groups "
                  f"by {block.get('score_label')})", ["Part", "Log growth per year"], rows)


def _attribution_figure(equity: pd.Series, attribution: xr.Dataset, reference: pd.Series | None,
                        block: dict, benchmark_name: str | None) -> go.Figure:
    """Draw the attribution curves, the score groups' curves and their annualised log growth.

    The top panel is the cumulative log growth of the strategy, the same
    weights before costs, the equal-weighted universe and the benchmark; the
    middle one each score group's, from the lowest scores (lightest) to the
    highest; the bottom one each group's annualised log growth.
    """
    fig = make_subplots(rows=3, cols=1, row_heights=[0.42, 0.33, 0.25], vertical_spacing=0.08)
    curves = [("attribution_strategy", "strategy", equity, PORTFOLIO_COLOUR)]
    if "gross_value" in attribution:
        curves.append(("attribution_gross", "before costs", attribution["gross_value"].to_pandas(), GAIN_COLOUR))
    curves.append(("attribution_universe", "equal-weighted universe",
                   attribution["universe_value"].to_pandas(), EXCESS_COLOUR))
    if reference is not None:
        curves.append(("attribution_benchmark", str(benchmark_name), reference, BENCHMARK_COLOUR))
    for trace, label, curve, colour in curves:
        growth = np.log(curve / curve.iloc[0])
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth.values, name=trace, mode="lines", line={"color": colour},
            hovertemplate=f"%{{x}}<br>{html.escape(label)} log growth %{{y:.2%}}<extra></extra>",
        ), row=1, col=1)
    group_curves = attribution["group_value"]
    count = group_curves.sizes["group"]
    for i, group in enumerate(group_curves["group"].values):
        curve = group_curves.sel(group=group).to_pandas()
        growth = np.log(curve / curve.iloc[0])
        shade = 0.25 + 0.75 * i / max(count - 1, 1)
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth.values, name=f"attribution_group_{group}", mode="lines",
            line={"color": PORTFOLIO_COLOUR, "width": 1}, opacity=shade, showlegend=False,
            hovertemplate=f"%{{x}}<br>G{group} log growth %{{y:.2%}}<extra></extra>",
        ), row=2, col=1)
    growths = block.get("group_annualized_log_return") or []
    fig.add_trace(go.Bar(
        x=[f"G{i + 1}" for i in range(len(growths))], y=growths, name="group_annualized_log_return",
        marker={"color": [GAIN_COLOUR if (g or 0) >= 0 else LOSS_COLOUR for g in growths]},
        showlegend=False, hovertemplate="%{x}<br>annualised log growth %{y:.2%}<extra></extra>",
    ), row=3, col=1)
    fig.update_yaxes(title_text="cumulative log growth", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="score groups (G1 lowest)", tickformat=".0%", row=2, col=1)
    fig.update_yaxes(title_text="log growth per year", tickformat=".0%", row=3, col=1)
    fig.update_layout(height=900, showlegend=True, legend={"orientation": "h", "y": 1.06})
    return fig



def _notes_list(notes: list[str]) -> str:
    """The notes as a list, without a heading."""
    items = "\n".join(f"    <li>{_escape(note)}</li>" for note in notes)
    return f'  <ul class="notes">\n{items}\n  </ul>\n'

def _rolling_figure(equity: pd.Series, reference: pd.Series | None, bars_per_year: float | None,
                    benchmark_name: str) -> go.Figure | None:
    """Rolling one-year panels: excess return, IR and beta against the benchmark,
    or return, volatility and Sharpe ratio without one. ``None`` when the
    window is shorter than a year."""
    window = int(round(bars_per_year or TRADING_BARS_PER_YEAR))
    if window < 2 or len(equity) <= window:
        return None
    r = equity.pct_change()
    if reference is not None:
        b = reference.pct_change()
        excess = r - b
        relative = equity / reference
        panels = [
            ("rolling_excess_return", f"1y excess vs {benchmark_name}", relative / relative.shift(window) - 1.0, ".0%", 0.0),
            ("rolling_information_ratio", "1y information ratio",
             excess.rolling(window).mean() / excess.rolling(window).std() * math.sqrt(window), ".2f", 0.0),
            ("rolling_beta", "1y beta", r.rolling(window).cov(b) / b.rolling(window).var(), ".2f", 1.0),
        ]
    else:
        panels = [
            ("rolling_return", "1y return", equity / equity.shift(window) - 1.0, ".0%", 0.0),
            ("rolling_volatility", "1y volatility", r.rolling(window).std() * math.sqrt(window), ".0%", None),
            ("rolling_sharpe", "1y Sharpe ratio",
             r.rolling(window).mean() / r.rolling(window).std() * math.sqrt(window), ".2f", 0.0),
        ]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for row, (name, title, series, fmt, reference_line) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=name, mode="lines", line={"color": PORTFOLIO_COLOUR},
            hovertemplate=f"%{{x}}<br>{html.escape(title)} %{{y:{fmt}}}<extra></extra>",
        ), row=row, col=1)
        if reference_line is not None:
            fig.add_hline(y=reference_line, line={"color": BENCHMARK_COLOUR, "width": 1}, row=row, col=1)
        fig.update_yaxes(title_text=html.escape(title), tickformat=fmt, row=row, col=1)
    fig.update_layout(height=720, showlegend=False)
    return fig


#: The smallest absolute weight counted as a holding: an optimiser leaves
#: residues of a few basis points or less on symbols it does not want.
HOLDING_THRESHOLD = 1e-3


def _portfolio_figure(
    weights: xr.DataArray | None, turnover: xr.DataArray | None, rebalances=None
) -> go.Figure | None:
    """Per rebalance: turnover, holdings and exposure.

    Turnover is the series given (buys plus sells over the previous bar's
    value), drawn as a filled step line with its trailing one-year mean,
    because a bar one day wide vanishes on a multi-year axis. Holdings count
    the target weights of the decided rebalances (``rebalances``) of at
    least ``HOLDING_THRESHOLD`` in absolute value (a symbol a row leaves
    NaN, to keep it, counts at its last target),
    beside the effective number ``(sum |w|)^2 / sum w^2``, which residues do
    not move. Exposure is the gross (and net, when anything is short)
    exposure on an axis that always spans 0 to 100%, so a fully invested
    book's round-off is not magnified into noise.
    """
    if weights is None and turnover is None:
        return None
    rows: list[tuple[str, str, list, dict]] = []
    if turnover is not None and turnover.size:
        series = turnover.to_pandas()
        mean = series.rolling("365D").mean()
        rows.append(("turnover (buys + sells)", "per rebalance fill · line: trailing one-year mean", [
            go.Scatter(x=series.index, y=series.values, name="turnover", mode="lines",
                       line={"color": EXCESS_COLOUR, "width": 1, "shape": "hv"}, fill="tozeroy",
                       hovertemplate="%{x}<br>turnover %{y:.1%}<extra></extra>"),
            go.Scatter(x=mean.index, y=mean.values, name="turnover_1y_mean", mode="lines",
                       line={"color": PORTFOLIO_COLOUR, "width": 2},
                       hovertemplate="%{x}<br>trailing 1y mean %{y:.1%}<extra></extra>"),
        ], {"tickformat": ".0%", "rangemode": "tozero"}))
    if weights is not None and weights.ndim == 2:
        weights = weights.transpose("timestamp", ...)
        # A NaN cell keeps that symbol's holding, so a rebalance row counts
        # it at its last target.
        frame = weights.to_pandas().ffill().fillna(0.0).iloc[_rebalance_rows(weights, rebalances)]
        if not frame.empty:
            absolute = frame.abs()
            holdings = (absolute >= HOLDING_THRESHOLD).sum(axis=1).astype(float)
            gross = absolute.sum(axis=1)
            squares = (frame**2).sum(axis=1)
            effective = (gross**2 / squares).where(squares > 0, 0.0)
            top = float(max(holdings.max(), effective.max(), 1.0))
            rows.append(("holdings", f"solid: weights of at least {HOLDING_THRESHOLD:.1%} · dotted: effective number", [
                go.Scatter(x=holdings.index, y=holdings.values, name="holdings", mode="lines",
                           line={"color": PORTFOLIO_COLOUR, "shape": "hv"},
                           hovertemplate="%{x}<br>holdings %{y:.0f}<extra></extra>"),
                go.Scatter(x=effective.index, y=effective.values, name="effective_holdings", mode="lines",
                           line={"color": BENCHMARK_COLOUR, "shape": "hv", "dash": "dot"},
                           hovertemplate="%{x}<br>effective holdings %{y:.1f}<extra></extra>"),
            ], {"tickformat": ",.0f", "rangemode": "tozero",
                "dtick": max(1, math.ceil(top / 5))}))
            exposure = [
                go.Scatter(x=gross.index, y=gross.values, name="gross_exposure", mode="lines",
                           line={"color": GAIN_COLOUR, "shape": "hv"},
                           hovertemplate="%{x}<br>gross exposure %{y:.1%}<extra></extra>"),
            ]
            net = frame.sum(axis=1)
            if (frame < 0).any().any():
                exposure.append(go.Scatter(
                    x=net.index, y=net.values, name="net_exposure", mode="lines",
                    line={"color": BENCHMARK_COLOUR, "shape": "hv", "dash": "dash"},
                    hovertemplate="%{x}<br>net exposure %{y:.1%}<extra></extra>"))
            low = min(0.0, float(net.min()))
            high = max(1.0, float(gross.max()))
            pad = 0.05 * (high - low)
            key = "gross (solid) and net (dashed)" if len(exposure) > 1 else "gross"
            rows.append(("exposure", key, exposure, {"tickformat": ".0%", "range": [low - pad, high + pad]}))
    if not rows:
        return None
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.07)
    for row, (title, key, traces, axis) in enumerate(rows, start=1):
        for trace in traces:
            fig.add_trace(trace, row=row, col=1)
        fig.update_yaxes(title_text=title, row=row, col=1, **axis)
        suffix = "" if row == 1 else str(row)
        fig.add_annotation(text=html.escape(key), xref=f"x{suffix} domain", yref=f"y{suffix} domain",
                           x=0.0, y=1.0, xanchor="left", yanchor="bottom", showarrow=False,
                           font={"size": 11, "color": "#666"})
    fig.update_layout(height=240 * len(rows) + 80, showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# holdings
# ---------------------------------------------------------------------------


#: Holdings whose target weight is not zero but at most this in absolute value
#: are folded into one "Other" line of the Holdings tab: an optimiser leaves
#: such dust on symbols it does not want. A constant of the report rather than
#: a config field, so it never changes a run's identity.
DUST_THRESHOLD = 1e-4


def _rebalance_rows(weights: xr.DataArray, rebalances) -> np.ndarray:
    """Return the positions in ``weights`` of the decided rebalances, in time order."""
    stamps = pd.DatetimeIndex(weights.timestamp.values)
    wanted = pd.DatetimeIndex(np.asarray([] if rebalances is None else rebalances).astype("datetime64[ns]"))
    rows = stamps.get_indexer(wanted.unique().sort_values())
    return rows[rows >= 0]


def _targets_in_force(
    weights: xr.DataArray | None, holdings: xr.DataArray, rebalances
) -> tuple[np.ndarray, list[str | None]]:
    """Return the targets in force on each bar of ``holdings`` and the rebalance they came from.

    ``rebalances`` are the decided rebalances, rows of ``weights``; a signal
    at bar t fills at t + 1, so its targets are in force from the first bar
    after it, a NaN cell keeping that symbol's previous target. Before the
    first rebalance every target is 0 and the rebalance is ``None``.
    """
    bars = holdings.timestamp.values
    targets = np.zeros((bars.size, holdings.sizes["symbol"]))
    since: list[str | None] = [None] * bars.size
    if weights is None:
        return targets, since
    weights = weights.transpose("timestamp", "symbol").reindex(symbol=holdings.symbol.values)
    rows = weights.values.astype(np.float64)
    rebalances = _rebalance_rows(weights, rebalances)
    starts = np.searchsorted(bars, weights.timestamp.values[rebalances], side="right")
    current = np.zeros(rows.shape[1])
    for n, (k, start) in enumerate(zip(rebalances, starts)):
        current = np.where(np.isfinite(rows[k]), rows[k], current)
        end = starts[n + 1] if n + 1 < len(starts) else bars.size
        targets[start:end] = current
        since[start:end] = [date_range.bar_label(weights.timestamp.values[k])] * (end - start)
    return targets, since


def _holding_period_returns(
    weights: xr.DataArray, holdings: xr.DataArray, prices: xr.DataArray, rebalances
) -> xr.DataArray:
    """Return, on each bar of ``holdings``, the price return of the holding period it falls in.

    A period is the bars one decided rebalance's targets are in force
    (``rebalances``, rows of ``weights``); its return
    runs from the close of its signal bar to its last close on ``holdings``'
    bars, on ``prices`` carried forward. NaN before the first rebalance
    fills and where a symbol has no price.
    """
    bars = holdings.timestamp.values
    price = (
        prices.transpose("timestamp", "symbol")
        .reindex(symbol=holdings.symbol.values)
        .ffill("timestamp")
    )
    returns = np.full((bars.size, holdings.sizes["symbol"]), np.nan)
    signals = weights.timestamp.values[_rebalance_rows(weights, rebalances)]
    starts = np.searchsorted(bars, signals, side="right")
    for n, (signal, start) in enumerate(zip(signals, starts)):
        end = starts[n + 1] if n + 1 < len(starts) else bars.size
        if end <= start:
            continue
        opened = price.sel(timestamp=signal).values.astype(np.float64)
        closed = price.sel(timestamp=bars[end - 1], method="ffill").values.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns[start:end] = closed / opened - 1.0
    return xr.DataArray(returns, dims=("timestamp", "symbol"), coords={
        "timestamp": bars, "symbol": holdings.symbol.values,
    })


def _ranked_contributions(holdings: xr.DataArray, prices: xr.DataArray) -> xr.DataArray:
    """Return each bar's contributions, largest holding first, on ``(timestamp, rank)``.

    A name's contribution to a bar is its holding at the previous close
    times its return over the bar on ``prices`` carried forward (0 without
    a return). Each bar lists the names held at the previous close, ranked
    by the size of their holding (ties in symbol order), and is NaN past
    its last name; the first bar has none.
    """
    held = np.nan_to_num(holdings.transpose("timestamp", "symbol").values.astype(np.float64))
    start = np.zeros_like(held)
    start[1:] = held[:-1]
    price = (
        prices.transpose("timestamp", "symbol")
        .reindex(timestamp=holdings.timestamp.values, symbol=holdings.symbol.values)
        .ffill("timestamp")
        .values.astype(np.float64)
    )
    contribution = start * np.nan_to_num(one_bar_returns(price))
    ranked = np.full(held.shape, np.nan)
    for i in range(held.shape[0]):
        names = np.flatnonzero(start[i])
        names = names[np.argsort(-np.abs(start[i, names]), kind="stable")]
        ranked[i, : names.size] = contribution[i, names]
    width = int(np.isfinite(ranked).sum(axis=1).max(initial=0))
    return xr.DataArray(ranked[:, :width], dims=("timestamp", "rank"), coords={
        "timestamp": holdings.timestamp.values, "rank": np.arange(width),
    })


def _holdings_data(
    holdings: xr.DataArray,
    weights: xr.DataArray | None,
    rebalances,
    names: Mapping[str, Sequence[tuple[str, str, str]]] | None,
    holding_returns: xr.DataArray | None,
    contributions: xr.DataArray | None,
    book_returns: xr.DataArray | None,
) -> dict:
    """The JSON the Holdings tab embeds.

    ``days`` holds, per bar, its label ``d``, the rebalance its targets came
    from ``r``, the ``cash`` (1 less the holdings), ``other`` (the count,
    target and holding of the dust) and ``h``, one ``[name, target,
    holding, period return]`` row per other symbol held or targeted, largest
    holding first; ``name`` indexes ``names``, ``[ticker, company, symbol
    id]`` rows, and the period return (``holding_returns`` on the bar, ``None``
    where it is NaN or not given) is the symbol's return over the holding
    period the bar falls in.
    Each day also carries its summary: ``n``, the number of non-zero
    holdings (dust included); ``top``, the ten largest holdings by size
    summed (a short counts by its size); and ``new``, the positions in
    ``h`` of the names the day's rebalance targets that the previous one
    did not (a target counts when it is above ``DUST_THRESHOLD`` in size),
    ``None`` while there is no previous rebalance to compare with.
    ``summary`` holds the tab's tiles: bars, rebalances and holdings per bar.
    ``contribution`` (``None`` without ``contributions``) holds the
    ``ranked``, each bar's contributions largest holding first (the row
    of ``contributions`` without its NaN tail), and ``nav``, the book's return on each bar
    (``None`` without ``book_returns``), whose excess over the summed
    contributions is trading at the open and costs.
    """
    held = np.nan_to_num(holdings.transpose("timestamp", "symbol").values.astype(np.float64))
    targets, since = _targets_in_force(weights, holdings, rebalances)
    targets = np.nan_to_num(targets)
    period = (
        np.full(held.shape, np.nan) if holding_returns is None
        else holding_returns.transpose("timestamp", "symbol").values.astype(np.float64)
    )
    symbols = [str(symbol) for symbol in holdings.symbol.values]
    spans = {str(key): list(value) for key, value in (names or {}).items()}
    starts = {key: [str(span[0]) for span in value] for key, value in spans.items()}
    index: dict[tuple[str, str, str], int] = {}
    table: list[list[str]] = []

    def name(j: int, day: str) -> int:
        symbol, ticker, company = symbols[j], symbols[j], ""
        k = bisect.bisect_right(starts.get(symbol, []), day) - 1
        if k >= 0:
            ticker, company = str(spans[symbol][k][1]), str(spans[symbol][k][2])
        key = (ticker, company, symbol)
        if key not in index:
            index[key] = len(table)
            table.append(list(key))
        return index[key]

    days = []
    # The targets of the rebalance before the one in force: None until a
    # second rebalance is in force.
    previous: np.ndarray | None = None
    for i, bar in enumerate(holdings.timestamp.values):
        day = date_range.bar_label(bar)
        h, t = held[i], targets[i]
        if i > 0 and since[i] != since[i - 1]:
            previous = None if since[i - 1] is None else targets[i - 1]
        shown = (h != 0.0) | (t != 0.0)
        dust = shown & (t != 0.0) & (np.abs(t) <= DUST_THRESHOLD)
        kept = np.flatnonzero(shown & ~dust)
        kept = kept[np.argsort(-h[kept], kind="stable")]
        new = None
        if previous is not None:
            added = (np.abs(t) > DUST_THRESHOLD) & ~(np.abs(previous) > DUST_THRESHOLD)
            new = [k for k, j in enumerate(kept) if added[j]]
        days.append({
            "d": day,
            "r": since[i],
            "cash": float(1.0 - h.sum()),
            "n": int(np.count_nonzero(h)),
            "top": float(np.sort(np.abs(h))[::-1][:10].sum()),
            "new": new,
            "other": [int(dust.sum()), float(t[dust].sum()), float(h[dust].sum())],
            "h": [
                [name(j, day), float(t[j]), float(h[j]), float(period[i, j]) if np.isfinite(period[i, j]) else None]
                for j in kept
            ],
        })
    traded = [int(np.count_nonzero(held[i])) for i in range(len(days)) if since[i] is not None]
    summary = [
        ["Bars", f"{len(days):,}"],
        ["Rebalances", f"{len({r for r in since if r is not None}):,}"],
        ["Holdings per bar", f"{min(traded)}–{max(traded)} (avg {np.mean(traded):.0f})" if traded else DASH],
    ]
    contribution = None
    if contributions is not None:
        nav = None
        if book_returns is not None:
            nav = np.nan_to_num(book_returns.reindex(timestamp=holdings.timestamp.values).values.astype(np.float64))
        contribution = {
            "ranked": [
                row[np.isfinite(row)].tolist()
                for row in contributions.transpose("timestamp", "rank").values.astype(np.float64)
            ],
            "nav": None if nav is None else nav.tolist(),
        }
    return {"days": days, "names": table, "summary": summary, "dust": DUST_THRESHOLD,
            "contribution": contribution}


def _holdings_section(
    holdings: xr.DataArray,
    weights: xr.DataArray | None,
    rebalances,
    names: Mapping[str, Sequence[tuple[str, str, str]]] | None,
    holding_returns: xr.DataArray | None,
    contributions: xr.DataArray | None,
    book_returns: xr.DataArray | None,
) -> str:
    """The Holdings tab: summary tiles, day controls, the day's table, its data and script.

    The data is embedded as JSON, uncompressed, with every ``<`` escaped so
    no name can close the script element; the page's own small script draws
    the selected day from it.
    """
    data = _holdings_data(holdings, weights, rebalances, names, holding_returns, contributions, book_returns)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    tiles = "".join(
        f'<div class="tile"><div class="kl">{_escape(label)}</div><div class="kv">{_escape(value)}</div></div>'
        for label, value in data["summary"]
    )
    return (
        f"<style>{_HOLDINGS_STYLE}</style>\n"
        + ('<div class="hd-analysis" id="hd-analysis"></div>\n' if data["contribution"] else "")
        + f'<div class="tiles hd-tiles">{tiles}</div>\n'
        '<div class="hd-controls">'
        '<button id="hd-prev-reb" title="The first bar of the previous rebalance">&laquo; Rebalance</button>'
        '<button id="hd-prev" title="The previous bar (left arrow)">&lsaquo; Day</button>'
        '<input type="date" id="hd-date">'
        '<button id="hd-next" title="The next bar (right arrow)">Day &rsaquo;</button>'
        '<button id="hd-next-reb" title="The first bar of the next rebalance">Rebalance &raquo;</button>'
        '<input type="search" id="hd-filter" placeholder="Filter ticker or company">'
        '<button id="hd-csv">Download day as CSV</button></div>\n'
        '<input type="range" id="hd-slider" min="0" step="1">\n'
        '<div class="hd-daybar" id="hd-daybar"></div>\n'
        '<table class="metrics hd-table"><thead><tr>'
        '<th data-k="ticker">Ticker</th><th data-k="company">Company</th><th data-k="symbol">Symbol</th>'
        '<th data-k="t" title="The weight the last rebalance before the day asked for.">Target weight</th>'
        '<th data-k="h" title="The share of the book, cash included, at the day\'s close.">Holding</th>'
        '<th data-k="h" title="Upper, light bar: the target weight. Lower, dark bar: the holding. Blue is '
        'long, red short.">Target <span class="hd-key t"></span><span class="hd-key t neg"></span> / holding '
        '<span class="hd-key h"></span><span class="hd-key h neg"></span></th>'
        '<th data-k="p" title="The symbol\'s price return over the holding period the day falls in: the days the '
        'targets of this rebalance are in force, from the fill to the day before the next rebalance fills. It runs '
        'from the signal day\'s close to the period\'s last close, on the prices the book is valued at; with a '
        'rebalance every day it is the one day\'s return. Every day of a period shows the whole period\'s return; '
        'the last period ends with the run.">Period return</th></tr></thead>'
        '<tbody id="hd-rows"></tbody><tfoot id="hd-foot"></tfoot></table>\n'
        f'<script type="application/json" id="holdings-data">{payload}</script>\n'
        f"<script>{_HOLDINGS_SCRIPT}</script>"
        + (f"<script>{_HOLDINGS_ANALYSIS_SCRIPT}</script>" if data["contribution"] else "")
    )


_HOLDINGS_STYLE = """
  .hd-tiles { margin-bottom: 14px; }
  .hd-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; }
  .hd-controls button, .hd-controls input { font: inherit; color: #111827; background: #fff;
    border: 1px solid #e5e7eb; border-radius: 6px; padding: 5px 10px; }
  .hd-controls button { cursor: pointer; } .hd-controls button:hover { border-color: #2563eb; }
  #hd-slider { width: 100%; accent-color: #2563eb; margin: 4px 0 10px; }
  .hd-daybar { display: flex; flex-wrap: wrap; gap: 6px 20px; color: #6b7280; margin-bottom: 10px; }
  .hd-daybar b { color: #111827; font-variant-numeric: tabular-nums; }
  .hd-tag { display: inline-block; font-size: 10.5px; padding: 0 6px; border-radius: 4px; margin-left: 6px;
    background: #eff6ff; color: #2563eb; }
  .hd-tag.new { background: none; color: #059669; border: 1px solid #059669; }
  table.hd-table thead th { cursor: pointer; user-select: none; }
  table.hd-table td.l, table.hd-table thead th:nth-child(-n+3) { text-align: left; }
  table.hd-table td.l.co { white-space: normal; min-width: 160px; }
  table.hd-table tfoot td { color: #6b7280; }
  .hd-bar { position: relative; width: 140px; height: 10px; }
  .hd-bar span { position: absolute; left: 0; height: 4px; border-radius: 2px; }
  table.hd-table td.pos { color: #059669; } table.hd-table td.neg { color: #dc2626; }
  .hd-bar .t { top: 0; background: #93c5fd; } .hd-bar .h { top: 6px; background: #2563eb; }
  .hd-key { display: inline-block; width: 12px; height: 4px; border-radius: 2px; margin: 0 2px 2px 0; }
  .hd-key.t { background: #93c5fd; } .hd-key.h { background: #2563eb; }
  .hd-key.t.neg { background: #fca5a5; } .hd-key.h.neg { background: #dc2626; }
  .hd-bar .t.neg { background: #fca5a5; } .hd-bar .h.neg { background: #dc2626; }
  .hd-analysis { margin-bottom: 22px; padding-bottom: 18px; border-bottom: 1px solid #e5e7eb; }
  .hd-analysis h4 { margin: 0 0 4px; font-size: 14px; } .hd-analysis .hd-note { color: #6b7280; margin-bottom: 10px; }
  .hd-cuts { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .hd-cuts button { font: inherit; color: #111827; background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
    padding: 4px 10px; cursor: pointer; }
  .hd-cuts button.on { border-color: #2563eb; color: #2563eb; background: #eff6ff; }
  .hd-share { display: flex; align-items: center; gap: 4px; margin-bottom: 10px; }
  .hd-share b { min-width: 46px; text-align: right; font-variant-numeric: tabular-nums; }
  .hd-share input { flex: 1; margin-left: 12px; accent-color: #2563eb; }
  .hd-analysis .kv.pos { color: #059669; } .hd-analysis .kv.neg { color: #dc2626; }
  .hd-deciles { overflow-x: auto; margin-top: 14px; }
  table.hd-dec td, table.hd-dec th { text-align: right; } table.hd-dec td:first-child, table.hd-dec th:first-child { text-align: left; }
  table.hd-dec tr.all td { font-weight: 600; }
"""

#: Draws the Holdings tab's contribution analysis from its JSON: the return of
#: the largest positions (any share of the names, chosen on a slider) against the
#: rest and the residual, as tiles and cumulative lines, then the contributions
#: by size decile and year, each as the net, the gains or the losses, with the
#: share of name-days that lost. Contributions are summed over bars, not
#: compounded. Text is set with ``textContent``, never as markup.
_HOLDINGS_ANALYSIS_SCRIPT = """
(function () {
  var D = JSON.parse(document.getElementById('holdings-data').textContent);
  var C = D.contribution, root = document.getElementById('hd-analysis');
  if (!C || !root || !D.days.length) return;
  var dates = D.days.map(function (d) { return d.d; });
  var nav = C.nav, cut = 10, view = 'net', T = dates.length;
  // Prefix sums of each bar's ranked contributions: net, gains, losses and
  // losing names, so any share of the names is one lookup per bar.
  var PRE = { net: [], gain: [], loss: [], losers: [] }, N = [];
  C.ranked.forEach(function (row) {
    var n = row.length, p = { net: new Float64Array(n + 1), gain: new Float64Array(n + 1),
      loss: new Float64Array(n + 1), losers: new Float64Array(n + 1) };
    row.forEach(function (c, r) {
      p.net[r + 1] = p.net[r] + c;
      p.gain[r + 1] = p.gain[r] + (c > 0 ? c : 0);
      p.loss[r + 1] = p.loss[r] + (c < 0 ? c : 0);
      p.losers[r + 1] = p.losers[r] + (c < 0 ? 1 : 0);
    });
    Object.keys(PRE).forEach(function (k) { PRE[k].push(p[k]); });
    N.push(n);
  });
  // How many of n names, largest first, are in the top `share` percent:
  // the names of rank r (0 first) with r / n < share / 100.
  function count(n, share) { return share >= 100 ? n : Math.ceil(share * n / 100 - 1e-9); }
  var VIEWS = [['net', 'Net'], ['gain', 'Gains'], ['loss', 'Losses']];
  var pct = function (x, p) { return (x * 100).toFixed(p === undefined ? 1 : p) + '%'; };
  var signed = function (x, p) { return (x > 0 ? '+' : '') + pct(x, p); };
  var tone = function (x) { return x > 0 ? ' pos' : x < 0 ? ' neg' : ''; };
  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }
  function tile(label, value, notes, cls, tip) {
    var t = el('div', undefined, 'tile');
    if (tip) t.title = tip;
    t.appendChild(el('div', label, 'kl'));
    t.appendChild(el('div', value, 'kv' + (cls || '')));
    notes.forEach(function (n) { if (n) t.appendChild(el('div', n, 'hd-note')); });
    return t;
  }
  // Bar i's `key` (net, gain, loss, losers, or names) over the names from the
  // top `lower` to the top `upper` percent.
  function between(i, key, lower, upper) {
    var from = count(N[i], lower), to = count(N[i], upper);
    if (key === 'names') return to - from;
    return PRE[key][i][to] - PRE[key][i][from];
  }
  // `key` summed over the bars i for which keep(i) holds, in a share of the names.
  function total(key, lower, upper, keep) {
    var s = 0;
    for (var i = 0; i < T; i++) if (!keep || keep(i)) s += between(i, key, lower, upper);
    return s;
  }
  function losing(lower, upper, keep) {
    var n = total('names', lower, upper, keep);
    return n ? total('losers', lower, upper, keep) / n : 0;
  }
  root.appendChild(el('h4', 'Return by position size'));
  root.appendChild(el('div', 'Each day the names held at the previous close are ranked by the size of their holding; '
    + 'a name contributes its holding times its return over the day. The largest share of the names against the rest, '
    + 'summed over the days (not compounded): the net, or only the gains or only the losses. A losing name-day is '
    + 'a name that lost on a day. The residual is the return of the book less all contributions: trading at the '
    + 'open and costs.', 'hd-note'));
  var views = el('div', undefined, 'hd-cuts'), slider = el('div', undefined, 'hd-share');
  VIEWS.forEach(function (v) {
    var b = el('button', v[1]);
    b.dataset.view = v[0];
    b.addEventListener('click', function () { view = v[0]; draw(); decile(); });
    views.appendChild(b);
  });
  var range = el('input'), shown = el('b');
  range.type = 'range'; range.min = '0.1'; range.max = '99.9'; range.step = '0.1'; range.value = String(cut);
  range.title = 'The share of the names, largest holding first, counted as the top';
  slider.appendChild(el('span', 'Top '));
  slider.appendChild(shown);
  slider.appendChild(el('span', ' of names'));
  slider.appendChild(range);
  var pending = false;
  range.addEventListener('input', function () {
    cut = +range.value;
    shown.textContent = cut.toFixed(1) + '%';
    if (pending) return;
    pending = true;
    requestAnimationFrame(function () { pending = false; draw(); });
  });
  shown.textContent = cut.toFixed(1) + '%';
  root.appendChild(views);
  root.appendChild(slider);
  var tiles = el('div', undefined, 'tiles');
  root.appendChild(tiles);
  var chart = el('div', undefined, 'plotly-graph-div');
  root.appendChild(chart);
  var noun = function () { return view === 'net' ? '' : view === 'gain' ? ' (gains)' : ' (losses)'; };
  function draw() {
    views.querySelectorAll('button').forEach(function (b) { b.classList.toggle('on', b.dataset.view === view); });
    var top = [], rest = [], resid = [], book = [], a = 0, b = 0, r = 0, n = 0, label = cut.toFixed(1) + '%';
    dates.forEach(function (d, i) {
      var t = between(i, view, 0, cut), all = between(i, view, 0, 100);
      a += t; b += all - t;
      top.push(a); rest.push(b);
      if (nav) {
        n += nav[i]; book.push(n);
        r += nav[i] - between(i, 'net', 0, 100); resid.push(r);
      }
    });
    tiles.textContent = '';
    // A share of the return of the book reads only when there is a gain to share.
    var share = function (x) {
      return view === 'net' && nav && n > 0.01 ? pct(x / n, 0) + ' of the summed return of the book' : '';
    };
    var split = function (lower, upper) {
      return 'gains ' + signed(total('gain', lower, upper)) + ', losses ' + signed(total('loss', lower, upper));
    };
    var lost = function (lower, upper) { return pct(losing(lower, upper), 0) + ' of name-days lost'; };
    tiles.appendChild(tile('Top ' + label + ' of names' + noun(), signed(a), [share(a), split(0, cut), lost(0, cut)],
      tone(a), 'The contributions of the largest ' + label + ' of the names held at each previous close.'));
    tiles.appendChild(tile('Remaining ' + (100 - cut).toFixed(1) + '%' + noun(), signed(b),
      [share(b), split(cut, 100), lost(cut, 100)], tone(b), 'The contributions of the other names.'));
    if (nav && view === 'net') {
      tiles.appendChild(tile('Trading & costs', signed(r, 2), [share(r)], tone(r),
        'The return of the book less every contribution: fills at the open rather than the close, fees and slippage.'));
      tiles.appendChild(tile('Book', signed(n), ['summed daily returns, not compounded'], tone(n)));
    }
    var traces = [
      { x: dates, y: top, name: 'Top ' + label, line: { color: view === 'loss' ? '#dc2626' : '#2563eb', width: 2 } },
      { x: dates, y: rest, name: 'Remaining ' + (100 - cut).toFixed(1) + '%',
        line: { color: view === 'loss' ? '#fca5a5' : '#93c5fd', width: 2 } },
    ];
    if (nav && view === 'net') {
      traces.push({ x: dates, y: resid, name: 'Trading & costs', line: { color: '#9ca3af', dash: 'dot', width: 1.5 } });
      traces.push({ x: dates, y: book, name: 'Book', line: { color: '#111827', width: 1 } });
    }
    Plotly.react(chart, traces, {
      height: 300, margin: { l: 56, r: 12, t: 24, b: 32 }, hovermode: 'x unified',
      yaxis: { tickformat: '.0%', title: { text: 'summed contribution' + noun() }, gridcolor: '#f3f4f6' },
      xaxis: { gridcolor: '#f3f4f6' }, legend: { orientation: 'h', y: 1.15 },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { size: 11 },
    }, { displayModeBar: false, responsive: true });
  }
  // Size deciles by year.
  var years = [];
  dates.forEach(function (d) { if (years.indexOf(d.slice(0, 4)) < 0) years.push(d.slice(0, 4)); });
  var inYear = function (y) { return function (i) { return dates[i].slice(0, 4) === y; }; };
  var box = el('div', undefined, 'hd-deciles');
  box.appendChild(el('h4', 'Contribution by size decile and year'));
  box.appendChild(el('div', 'D1 is the largest 10% of the names held at each previous close, D10 the smallest; '
    + 'each cell sums the contributions of that decile over the year. The last row is the share of name-days that '
    + 'lost. The first bar has no previous close, so its return is all residual.', 'hd-note'));
  var table = el('table', undefined, 'metrics hd-dec');
  box.appendChild(table);
  var bars = el('div', undefined, 'plotly-graph-div');
  box.appendChild(bars);
  root.appendChild(box);
  function decile() {
    var groups = years.map(function (y) { return [y, inYear(y)]; }).concat([['Whole run', null]]);
    var cells = groups.map(function (g) {
      return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (k) { return total(view, k * 10, (k + 1) * 10, g[1]); });
    });
    var scale = Math.max(1e-12, Math.max.apply(null, cells.slice(0, -1).map(function (row) {
      return Math.max.apply(null, row.map(Math.abs));
    })));
    table.textContent = '';
    var head = el('tr');
    ['Year', 'Holdings per bar'].concat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(function (k) { return 'D' + k; }))
      .concat(view === 'net' && nav ? ['Held', 'Trading & costs', 'Book'] : ['Held'])
      .forEach(function (h) { head.appendChild(el('th', h)); });
    var thead = el('thead');
    thead.appendChild(head);
    table.appendChild(thead);
    var body = el('tbody');
    groups.forEach(function (g, row) {
      var keep = g[1], tr = el('tr', undefined, keep ? '' : 'all');
      var count = 0, names = total('names', 0, 100, keep);
      dates.forEach(function (d, i) { if (!keep || keep(i)) count++; });
      tr.appendChild(el('td', g[0]));
      tr.appendChild(el('td', (names / Math.max(1, count)).toFixed(0)));
      cells[row].forEach(function (x) {
        var td = el('td', signed(x)), a = Math.min(1, Math.abs(x) / scale) * 0.5;
        td.style.background = x >= 0 ? 'rgba(37,99,235,' + a.toFixed(2) + ')' : 'rgba(220,38,38,' + a.toFixed(2) + ')';
        tr.appendChild(td);
      });
      var held = cells[row].reduce(function (s, x) { return s + x; }, 0);
      tr.appendChild(el('td', signed(held)));
      if (view === 'net' && nav) {
        var book = 0;
        nav.forEach(function (x, i) { if (!keep || keep(i)) book += x; });
        tr.appendChild(el('td', signed(book - held, 2)));
        tr.appendChild(el('td', signed(book)));
      }
      body.appendChild(tr);
    });
    var lost = el('tr', undefined, 'all');
    lost.appendChild(el('td', 'Losing name-days'));
    lost.appendChild(el('td', ''));
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].forEach(function (k) { lost.appendChild(el('td', pct(losing(k * 10, (k + 1) * 10), 0))); });
    lost.appendChild(el('td', pct(losing(0, 100), 0)));
    if (view === 'net' && nav) { lost.appendChild(el('td', '')); lost.appendChild(el('td', '')); }
    body.appendChild(lost);
    table.appendChild(body);
    var whole = cells[cells.length - 1];
    Plotly.react(bars, [{
      type: 'bar', x: whole.map(function (x, k) { return 'D' + (k + 1); }),
      y: whole,
      marker: { color: whole.map(function (x) { return x < 0 ? '#dc2626' : '#2563eb'; }) },
      hovertemplate: '%{x}: %{y:+.1%}<extra></extra>',
    }], {
      height: 220, margin: { l: 56, r: 12, t: 28, b: 28 }, yaxis: { tickformat: '.0%', gridcolor: '#f3f4f6' },
      title: { text: 'Whole run: summed contributions' + noun() + ' by size decile', font: { size: 12 } },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { size: 11 },
    }, { displayModeBar: false, responsive: true });
  }
  draw();
  decile();
})();
"""

#: Draws the selected day of the Holdings tab from its JSON: stepping by bar,
#: by rebalance, by date, by slider and by the arrow keys (while the tab is
#: shown), sorting by any column, filtering by ticker or company and the
#: day's CSV. Text from the data is set with ``textContent``, never as markup.
_HOLDINGS_SCRIPT = """
(function () {
  var D = JSON.parse(document.getElementById('holdings-data').textContent);
  var days = D.days, N = D.names;
  var $ = function (id) { return document.getElementById(id); };
  if (!days.length) return;
  var pct = function (x, p) { return (x * 100).toFixed(p === undefined ? 2 : p) + '%'; };
  var rebIdx = [];
  days.forEach(function (d, i) { if (d.r !== null && (i === 0 || d.r !== days[i - 1].r)) rebIdx.push(i); });
  var cur = days.length - 1, sortKey = 'h', sortDir = -1;
  function rowsOf(i) {
    var fresh = {};
    (days[i].new || []).forEach(function (k) { fresh[k] = true; });
    return days[i].h.map(function (r, k) {
      var n = N[r[0]];
      return { ticker: n[0], company: n[1], symbol: n[2], t: r[1], h: r[2], p: r[3], isNew: !!fresh[k] };
    });
  }
  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }
  function bar(t, h, scale) {
    var div = el('div', undefined, 'hd-bar');
    [['t', t], ['h', h]].forEach(function (p) {
      var s = el('span', undefined, p[0] + (p[1] < 0 ? ' neg' : ''));
      s.style.width = (Math.abs(p[1]) / scale * 100).toFixed(1) + '%';
      div.appendChild(s);
    });
    return div;
  }
  function render() {
    var day = days[cur];
    var all = rowsOf(cur);
    var bits = [['Date', day.d], ['Targets from', day.r === null ? 'no rebalance yet' : day.r],
                ['Holdings', String(day.n)], ['Top-10 holding', pct(day.top, 1)], ['Cash', pct(day.cash)],
                ['Other (dust)', pct(day.other[2], 3)]];
    if (day.new !== null) bits.push(['New names since previous rebalance', String(day.new.length)]);
    var daybar = $('hd-daybar');
    daybar.textContent = '';
    bits.forEach(function (b, i) {
      var span = el('span', b[0] + ' ');
      span.appendChild(el('b', b[1]));
      if (i === 0 && rebIdx.indexOf(cur) >= 0) span.appendChild(el('span', 'rebalance filled', 'hd-tag'));
      daybar.appendChild(span);
    });
    var q = $('hd-filter').value.trim().toLowerCase();
    var rows = all.filter(function (r) {
      return !q || r.ticker.toLowerCase().indexOf(q) >= 0 || r.company.toLowerCase().indexOf(q) >= 0
        || r.symbol.toLowerCase().indexOf(q) >= 0;
    });
    rows.sort(function (a, b) {
      var x = a[sortKey], y = b[sortKey];
      if (x === null) x = -Infinity;
      if (y === null) y = -Infinity;
      return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
    });
    var scale = Math.max(1e-9, Math.max.apply(null, all.map(function (r) {
      return Math.max(Math.abs(r.t), Math.abs(r.h));
    }).concat([0])));
    var body = $('hd-rows');
    body.textContent = '';
    rows.forEach(function (r) {
      var tr = el('tr');
      var tk = el('td', r.ticker, 'l');
      if (r.isNew) tk.appendChild(el('span', 'new', 'hd-tag new'));
      tr.appendChild(tk);
      tr.appendChild(el('td', r.company, 'l co'));
      tr.appendChild(el('td', r.symbol, 'l'));
      tr.appendChild(el('td', pct(r.t)));
      tr.appendChild(el('td', pct(r.h)));
      var cell = el('td');
      cell.appendChild(bar(r.t, r.h, scale));
      tr.appendChild(cell);
      tr.appendChild(r.p === null ? el('td', '\u2013')
        : el('td', (r.p > 0 ? '+' : '') + pct(r.p), r.p > 0 ? 'pos' : r.p < 0 ? 'neg' : undefined));
      body.appendChild(tr);
    });
    var foot = $('hd-foot');
    foot.textContent = '';
    var extra = [['Cash', '', null, day.cash]];
    if (day.other[0]) extra.unshift(['Other', day.other[0] + ' holdings with a target of at most ' + D.dust,
                                     day.other[1], day.other[2]]);
    extra.forEach(function (x) {
      var tr = el('tr');
      tr.appendChild(el('td', x[0], 'l'));
      tr.appendChild(el('td', x[1], 'l co'));
      tr.appendChild(el('td', '', 'l'));
      tr.appendChild(el('td', x[2] === null ? '' : pct(x[2], 3)));
      tr.appendChild(el('td', pct(x[3], x[2] === null ? 2 : 3)));
      tr.appendChild(el('td'));
      tr.appendChild(el('td'));
      foot.appendChild(tr);
    });
    $('hd-date').value = day.d.slice(0, 10);
    $('hd-slider').value = cur;
  }
  function select(i) { cur = Math.max(0, Math.min(days.length - 1, i)); render(); }
  $('hd-slider').max = days.length - 1;
  $('hd-slider').addEventListener('input', function (e) { select(+e.target.value); });
  $('hd-date').min = days[0].d.slice(0, 10);
  $('hd-date').max = days[days.length - 1].d.slice(0, 10);
  $('hd-date').addEventListener('change', function (e) {
    var v = e.target.value;
    if (!v) return;
    var k = days.findIndex(function (d) { return d.d.slice(0, 10) >= v; });
    select(k < 0 ? days.length - 1 : k);
  });
  $('hd-prev').addEventListener('click', function () { select(cur - 1); });
  $('hd-next').addEventListener('click', function () { select(cur + 1); });
  $('hd-prev-reb').addEventListener('click', function () {
    var k = rebIdx.filter(function (r) { return r < cur; });
    select(k.length ? k[k.length - 1] : 0);
  });
  $('hd-next-reb').addEventListener('click', function () {
    var k = rebIdx.find(function (r) { return r > cur; });
    select(k === undefined ? days.length - 1 : k);
  });
  $('hd-filter').addEventListener('input', render);
  document.addEventListener('keydown', function (e) {
    if (!$('hd-rows').offsetParent) return;
    var t = e.target;
    if (t.tagName === 'INPUT' && t.type !== 'range') return;
    if (e.key === 'ArrowLeft') { select(cur - 1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { select(cur + 1); e.preventDefault(); }
  });
  document.querySelectorAll('.hd-table th[data-k]').forEach(function (th) {
    th.addEventListener('click', function () {
      var k = th.dataset.k;
      if (sortKey === k) sortDir = -sortDir;
      else { sortKey = k; sortDir = (k === 't' || k === 'h' || k === 'p') ? -1 : 1; }
      render();
    });
  });
  $('hd-csv').addEventListener('click', function () {
    var day = days[cur];
    var quote = function (s) { return '"' + String(s).replace(/"/g, '""') + '"'; };
    var lines = ['date,rebalance,ticker,company,symbol,target_weight,holding,period_return'];
    var line = function (a, b, c, t, h, p) {
      return [day.d, day.r === null ? '' : day.r, quote(a), quote(b), quote(c), t, h,
              p === null || p === undefined ? '' : p].join(',');
    };
    rowsOf(cur).forEach(function (r) { lines.push(line(r.ticker, r.company, r.symbol, r.t, r.h, r.p)); });
    if (day.other[0]) lines.push(line('Other', '', '', day.other[1], day.other[2]));
    lines.push(line('Cash', '', '', '', day.cash));
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([lines.join('\\n') + '\\n'], { type: 'text/csv' }));
    a.download = 'holdings_' + day.d.replace(/[^0-9A-Za-z-]/g, '_') + '.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
  });
  render();
})();
"""


def _shade(fig: go.Figure, in_sample_range) -> None:
    """Shade the in-sample range on every row, as the Performance figure does."""
    if in_sample_range is not None:
        fig.add_vrect(x0=in_sample_range[0], x1=in_sample_range[1], row="all", col=1,
                      fillcolor="grey", opacity=0.2, line_width=0)


#: Bars in a year when the caller does not say (US equity trading days).
TRADING_BARS_PER_YEAR = 252


def _figure_div(fig: go.Figure) -> str:
    """A figure as an HTML fragment reusing the plotly.js the page loads, on the page's chart style."""
    _clean(fig)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _clean(fig: go.Figure) -> None:
    """Draw ``fig`` on a white plot area with a light grid and the page's font."""
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font={"family": CHART_FONT, "color": INK2})
    for axes in (fig.update_xaxes, fig.update_yaxes):
        axes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont={"color": MUTED})

#: The timeline's bar kinds: ``(fold key, colour, legend text)``. A fold's
#: ``training`` window, the bars it ``traded`` out-of-sample and its
#: ``in_sample`` bars (traded inside its own training window).
_TIMELINE_KINDS = (
    ("training", "#c6dbef", "training"),
    ("traded", "#2b8a3e", "traded, out-of-sample"),
    ("in_sample", "#e03b30", "traded, in-sample"),
)
_TIMELINE_COLOUR = {key: colour for key, colour, _ in _TIMELINE_KINDS}

#: Timeline geometry in pixels: the drawing width, the row-label gutter and
#: the height the fold rows share before a row stops shrinking.
_TIMELINE_WIDTH = 420
_TIMELINE_GUTTER = 56
_TIMELINE_ROWS_HEIGHT = 260


def _span(pair: object) -> tuple[pd.Timestamp, pd.Timestamp, str] | None:
    """Parse a pair of bar labels for the timeline.

    Parameters
    ----------
    pair : object
        A ``[first, last]`` pair of bar labels, as ``metrics.json`` carries
        them.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp, str] or None
        The two bars as naive UTC timestamps, so labels with and without a
        UTC offset share one axis, and the pair as ``"first .. last"``
        text; ``None`` for anything that is not two parseable, ordered
        labels.
    """
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    try:
        start, end = pd.Timestamp(str(pair[0])), pd.Timestamp(str(pair[1]))
    except (TypeError, ValueError):
        return None
    if pd.isna(start) or pd.isna(end):
        return None
    start, end = (t.tz_convert(None) if t.tzinfo is not None else t for t in (start, end))
    if end < start:
        return None
    return start, end, f"{pair[0]} .. {pair[1]}"


def _ticks(low: pd.Timestamp, high: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
    """Choose the timeline's axis ticks.

    Parameters
    ----------
    low, high : pd.Timestamp
        The first and last instant on the axis, naive.

    Returns
    -------
    list[tuple[pd.Timestamp, str]]
        At most seven month starts inside the axis, each with its label:
        the year when the step is whole years, else ``YYYY-MM``.
    """
    months = (high.year - low.year) * 12 + high.month - low.month + 1
    step = next((m for m in (1, 3, 6, 12, 24, 36, 60, 120) if months / m <= 7), 240)
    first = pd.Timestamp(year=low.year, month=1, day=1)
    ticks = []
    for date in pd.date_range(first, high, freq=pd.DateOffset(months=step)):
        if date > low:
            ticks.append((date, str(date.year) if step % 12 == 0 else date.strftime("%Y-%m")))
    return ticks


def _timeline_section(windows: dict | None) -> str:
    """Draw the run's windows as an inline SVG timeline.

    With several folds (a walk-forward CV run) the top row is the backtest
    window, its out-of-sample bars green and the bars inside a training
    window red, and each fold has a row below it with its training window
    in light blue, the bars it traded in green and its in-sample bars red,
    so sliding and expanding folds read as a staircase of equal or growing
    bars. A model backtest has a single row, its training window beside
    the backtest window, and a run without a model a single row of traded
    bars. Every bar carries its dates as a tooltip; rows shrink as folds
    are added, and only every fifth fold is named once they get thin. The
    legend lists only the kinds drawn.

    Parameters
    ----------
    windows : dict or None
        The ``windows`` input of ``write_backtest_report``.

    Returns
    -------
    str
        The section's HTML, or the empty string without a parseable
        backtest window.
    """
    if not isinstance(windows, dict):
        return ""
    backtest = _span(windows.get("backtest"))
    if backtest is None:
        return ""
    folds = [fold for fold in windows.get("folds") or [] if isinstance(fold, dict)]
    spans = [backtest] + [
        span for fold in folds for span in (_span(fold.get("training")), _span(fold.get("traded"))) if span
    ]
    low, high = min(s[0] for s in spans), max(s[1] for s in spans)
    width = max((high - low).total_seconds(), 1.0)
    drawn: set[str] = set()

    def x(when: pd.Timestamp) -> float:
        """The horizontal position of ``when`` on the drawing."""
        return _TIMELINE_GUTTER + (when - low).total_seconds() / width * (_TIMELINE_WIDTH - _TIMELINE_GUTTER - 4)

    def bar(span: tuple, y: float, height: float, kind: str, tip: str) -> str:
        """One bar of ``kind`` over ``span`` with its dates as the tooltip."""
        drawn.add(kind)
        start, end, text = span
        return (f'<rect x="{x(start):.1f}" y="{y:.1f}" width="{max(1.0, x(end) - x(start)):.1f}" '
                f'height="{height:.1f}" fill="{_TIMELINE_COLOUR[kind]}"><title>{_escape(tip)} '
                f"{_escape(text)}</title></rect>")

    # The backtest row: out-of-sample runs and in-sample runs, or, for a run
    # without a split, the whole window as traded bars.
    out_of_sample = [s for s in map(_span, windows.get("out_of_sample") or []) if s]
    in_sample = [s for s in map(_span, windows.get("in_sample") or []) if s]
    split = bool(folds or out_of_sample)
    backtest_bars = [("traded", span, "out-of-sample") for span in out_of_sample] or [
        ("traded", backtest, "traded" if not split else "out-of-sample")
    ]
    backtest_bars += [("in_sample", span, "in-sample") for span in in_sample]

    # One fold: its training window joins the backtest row, whose traded
    # bars are the fold's, so the two rows would repeat each other.
    single = len(folds) <= 1
    rows = [(str(folds[0].get("label", "")) if folds else "backtest",
             ([("training", _span(folds[0].get("training")), "training")] if folds else []) + backtest_bars)]
    if not single:
        rows = [("backtest", backtest_bars)] + [
            (str(fold.get("label", "")),
             [(key, _span(fold.get(key)), tip) for key, tip in
              (("training", "training"), ("traded", "traded"), ("in_sample", "in-sample"))])
            for fold in folds
        ]

    row = 14.0 if single else max(4.0, min(12.0, _TIMELINE_ROWS_HEIGHT / len(folds)))
    gap = 2.0 if row > 6 else 1.0
    height = len(rows) * (row + gap) + 16
    parts = []
    for when, text in _ticks(low, high):
        parts.append(f'<line x1="{x(when):.1f}" x2="{x(when):.1f}" y1="0" y2="{height - 14:.1f}" stroke="#eee"/>'
                     f'<text x="{x(when):.1f}" y="{height - 3:.1f}" text-anchor="middle">{_escape(text)}</text>')
    for i, (name, bars) in enumerate(rows):
        y = i * (row + gap)
        # Fold rows are numbered from the second row; name every fifth when thin.
        if row >= 9 or i == 0 or (i - 1) % 5 == 0:
            parts.append(f'<text x="0" y="{y + row - 1:.1f}">{_escape(name)}</text>')
        prefix = "" if i == 0 and name == "backtest" else f"{name} "
        parts += [bar(span, y, row, kind, f"{prefix}{tip}") for kind, span, tip in bars if span]

    caption = f"Backtest {backtest[2]}"
    if _number(windows.get("bars")) is not None:
        caption += f" ({int(windows['bars']):,} bars)"
    trained = [s for s in (_span(fold.get("training")) for fold in folds) if s]
    if not single:
        kind = "expanding" if len(trained) > 1 and len({s[0] for s in trained}) == 1 else "sliding"
        caption += f"; {len(folds)} folds, {kind} training window"
    legend = "".join(
        f'<span><span class="sw" style="background:{colour}"></span>'
        f'{"traded" if key == "traded" and not split else text}</span>'
        for key, colour, text in _TIMELINE_KINDS
        if key in drawn
    )
    return (
        "  <h2>Windows</h2>\n"
        f'  <div class="timeline"><div class="caption">{_escape(caption)}</div>\n'
        f'  <svg viewBox="0 0 {_TIMELINE_WIDTH} {height:.0f}" width="100%" style="max-width:{_TIMELINE_WIDTH}px">'
        + "".join(parts) + "</svg>\n"
        f'  <div class="legend">{legend}</div></div>\n'
    )


#: Setup rows whose value can run to many lines (a rule with a factor risk
#: model in its parameters): shown in a box of a few lines that scrolls.
_SCROLLED_SUMMARY_LABELS = frozenset({"Portfolio construction"})


def _summary_section(summary: dict[str, str] | None) -> str:
    """Render the setup table, or the empty string when there is none."""
    if not summary:
        return ""

    def cell(label: str, text: str) -> str:
        value = _escape(text)
        if label in _SCROLLED_SUMMARY_LABELS:
            value = f'<div class="scroll">{value}</div>'
        return f"<td>{value}</td>"

    rows = "\n".join(
        f"      <tr><th>{_escape(label)}</th>{cell(label, text)}</tr>"
        for label, text in summary.items()
    )
    return (
        "  <h2>Setup</h2>\n"
        '  <table class="summary">\n'
        f"{rows}\n"
        "  </table>\n"
    )


def _notes_section(notes: list[str] | None) -> str:
    """Render the notes list, or the empty string when there are none."""
    if not notes:
        return ""
    return "  <h2>Notes</h2>\n" + _notes_list(notes)


def _extra_section(extra_tables: dict[str, dict] | None) -> str:
    """Render each extra table under its heading, values formatted as unknown metrics."""
    return "".join(
        _table(heading, ["", "Value"], [
            _row(label, "", [_format(value, _guess_unit(label, value))])
            for label, value in rows.items()
        ])
        for heading, rows in (extra_tables or {}).items()
    )


def _heatmap_block(heatmap: str) -> str:
    """The heatmap under its caption, or nothing when there is no heatmap."""
    return f"\n  <h2>{_escape(HEATMAP_CAPTION)}</h2>\n{heatmap}\n" if heatmap else ""


#: What each chart section shows, on hover of its title.
_CHART_TIPS = {
    "Performance": "The portfolio's value (and the benchmark's, dashed), its drawdown below the running high, "
                   "and each month's return. The grey band is the in-sample part; the triangles mark the "
                   "deepest drawdown's valley and recovery.",
    "Excess": "How far ahead of or behind the benchmark the portfolio was over time (switch between log and "
              "arithmetic), and the worst stretches of falling behind it.",
    "Rolling": "One-year rolling statistics, to see whether the result was steady or came in bursts: excess "
               "return, information ratio and beta against the benchmark, or return, volatility and Sharpe "
               "ratio without one.",
    "Portfolio": "Turnover at each rebalance fill, the number of holdings and the gross exposure on each decided rebalance.",
    "Attribution": "Where the excess came from: the universe against the benchmark, the selection against "
                   "the universe, and the costs; and how each score group of the universe did.",
    "Holdings": "Which positions made the return: the largest positions against the rest, over time and by size "
                "decile and year. Then what the book held at each day's close: each symbol's target weight from the last rebalance "
                "before the day, and its holding, its value over the whole book with cash. Rejected orders, "
                "delisting settlements, costs and price moves make the two differ. The period return is each "
                "symbol's price return over the holding period of the day's rebalance. Top-10 holding is the sum "
                "of the ten largest holdings by size, shorts included; cash is one less the net holdings, so "
                "a short sale's proceeds raise it above 100%.",
}

#: The sidebar's sections, in order.
_SECTIONS = ("Overview", "Excess", "Rolling", "Portfolio", "Holdings", "Attribution", "Factor attribution",
             "Setup & notes")


def _section_card(label: str, body: str) -> str:
    """A chart tab's html in a card titled by the tab, its explanation on hover."""
    tip = _CHART_TIPS.get(label, "")
    return (f'<div class="card"><h3 title="{_escape(tip)}">{_escape(label)} <span class="i">ⓘ</span></h3>\n'
            f"{body}\n</div>")


def _cards(*bodies: str, layout: str = "grid2") -> str:
    """The non-empty bodies as cards side by side, or nothing."""
    cards = [body if body.startswith('<div class="card">') else f'<div class="card">\n{body}</div>'
             for body in bodies if body.strip()]
    if not cards:
        return ""
    if len(cards) == 1:
        return cards[0] + "\n"
    return f'<div class="{layout}">\n' + "\n".join(cards) + "\n</div>\n"


def _document(
    title: str,
    summary: dict[str, str] | None,
    metrics: dict | None,
    tabs: list[tuple[str, str]],
    notes: list[str] | None,
    *,
    benchmark_name: str | None = None,
    windows: dict | None = None,
    extra_tables: dict[str, dict] | None = None,
) -> str:
    """Assemble the page: a header, a sidebar of sections, the KPI cards and each section's cards.

    ``tabs`` are ``(label, html)`` pairs whose html is plotly's own fragments,
    inserted verbatim; everything else comes from the run and is escaped.
    Each chart tab becomes a section holding its chart beside the tables
    that explain it: Overview (the windows, in-sample against out-of-sample,
    the Performance chart and strategy against benchmark), Excess (and the
    relative table), Rolling, Portfolio (and the trading and extra tables),
    Attribution, Factor attribution, and Setup & notes. A section without
    input is left out; a table whose chart is missing moves to Overview. The
    first section is shown; a switch asks plotly to resize the figures it
    reveals, which were laid out while hidden. Every element with a
    ``title`` shows it as a styled explanation on hover.
    """
    m = metrics if isinstance(metrics, dict) else {}
    charts = dict(tabs)
    comparison = (_comparison_section(m, benchmark_name)) if m else ""
    relative = _relative_section(m, benchmark_name) if m else ""
    trading = (_trading_section(m) if m else "") + _extra_section(extra_tables)
    split = _split_section(m, benchmark_name) if m else ""
    overview = _cards(_timeline_section(windows), split)
    if "Performance" in charts:
        overview += _cards(_section_card("Performance", charts["Performance"]), comparison, layout="grid")
    else:
        overview += _cards(comparison)
    if "Excess" not in charts:
        overview += _cards(relative)
    if "Portfolio" not in charts:
        overview += _cards(trading)
    sections = {
        "Overview": overview,
        "Excess": _cards(_section_card("Excess", charts["Excess"]), relative, layout="grid")
        if "Excess" in charts else "",
        "Rolling": _cards(_section_card("Rolling", charts["Rolling"])) if "Rolling" in charts else "",
        "Portfolio": _cards(_section_card("Portfolio", charts["Portfolio"]), trading, layout="grid")
        if "Portfolio" in charts else "",
        "Holdings": _cards(_section_card("Holdings", charts["Holdings"])) if "Holdings" in charts else "",
        "Attribution": _cards(_section_card("Attribution", charts["Attribution"]))
        if "Attribution" in charts else "",
        "Factor attribution": charts.get("Factor attribution", ""),
        "Setup & notes": _cards(_summary_section(summary), _notes_section(notes)),
    }
    shown = [(label, sections[label]) for label in _SECTIONS if sections[label].strip()]
    buttons = "".join(
        f'<button class="tab{" on" if i == 0 else ""}" data-tab="tab{i}">{_escape(label)}</button>'
        for i, (label, _) in enumerate(shown)
    )
    panes = "".join(
        f'\n  <div class="pane{" on" if i == 0 else ""}" id="tab{i}">\n{body}\n  </div>'
        for i, (_, body) in enumerate(shown)
    )
    span = _span((windows or {}).get("backtest"))
    window = f"{span[0].date()} → {span[1].date()}" if span else ""
    against = f" · vs {benchmark_name}" if benchmark_name else ""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{_escape(title)}</title>\n"
        f"  <style>{_STYLE}  </style>\n"
        "</head>\n"
        "<body>\n"
        f'  <header><h1>{_escape(title)}</h1><span class="sub">{_escape(window + against)}</span></header>\n'
        '  <div class="shell">\n'
        f"  <nav>{buttons}</nav>\n"
        f"  <main>\n{_kpi_section(metrics, benchmark_name)}{panes}\n  </main>\n"
        "  </div>\n"
        '  <div id="tip"></div>\n'
        f"  <script>{_TAB_SCRIPT}</script>\n"
        "</body>\n"
        "</html>\n"
    )


#: Switches the sections, resizes the figures a switch reveals, and shows every
#: ``title`` as a styled explanation on hover.
_TAB_SCRIPT = """
document.querySelectorAll('.tab').forEach(function (tab) {
  tab.addEventListener('click', function () {
    document.querySelectorAll('.tab').forEach(function (t) { t.classList.toggle('on', t === tab); });
    document.querySelectorAll('.pane').forEach(function (pane) {
      var on = pane.id === tab.dataset.tab;
      pane.classList.toggle('on', on);
      if (on && window.Plotly) {
        pane.querySelectorAll('.plotly-graph-div').forEach(function (div) { Plotly.Plots.resize(div); });
      }
    });
    window.scrollTo(0, 0);
  });
});
var tip = document.getElementById('tip');
document.querySelectorAll('main [title]').forEach(function (el) {
  if (el.closest('svg')) return;
  el.dataset.tip = el.getAttribute('title');
  el.removeAttribute('title');
});
function show(event) {
  var el = event.target.closest && event.target.closest('[data-tip]');
  if (!el || !el.dataset.tip) { tip.style.display = 'none'; return; }
  var head = el.querySelector('.kl') || (el.tagName === 'TH' || el.tagName === 'H3' ? el : null);
  var name = head ? head.textContent.replace('ⓘ', '').trim() : '';
  tip.textContent = '';
  if (name) { var b = document.createElement('b'); b.textContent = name; tip.appendChild(b); }
  tip.appendChild(document.createTextNode(el.dataset.tip));
  tip.style.display = 'block';
  place(event);
}
function place(event) {
  if (tip.style.display !== 'block') return;
  var r = tip.getBoundingClientRect(), x = event.clientX + 14, y = event.clientY + 14;
  if (x + r.width > innerWidth - 8) x = Math.max(8, event.clientX - r.width - 14);
  if (y + r.height > innerHeight - 8) y = Math.max(8, event.clientY - r.height - 14);
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
document.addEventListener('mouseover', show);
document.addEventListener('mousemove', place);
document.addEventListener('click', show);
"""
