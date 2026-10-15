"""Leaf tests for `quantlab/runs/backtest_report.py` (quick task 260915-sxx).

`write_backtest_report` is a leaf: it takes what a run recorded, hand-built (an
account record, metrics, a benchmark record, Setup lines) and writes one
self-contained HTML page. These tests call it directly with small synthetic
records (`tests/report_records.py`) -- no dataset, no model, no backtest run --
so the page composition is covered in milliseconds instead of behind a
two-minute end-to-end run. The run-level locks (the page really carries the persisted
run's dates and numbers) live in tests/test_backtest_persistence.py.

T-sxx-01: every value the caller interpolates into the page is escaped. The
title and the summary values come from a run directory name, from operator
notes and from a metrics mapping, so a page that pasted them raw would turn
whatever produced them into markup.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.runs.backtest_report import ReportExtras, write_backtest_report
from tests.report_records import account, benchmark, series

N_BARS = 12
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)


def _value(n: int = N_BARS) -> xr.DataArray:
    """A rising-then-falling portfolio value, so drawdown is really negative."""
    path = np.concatenate(
        [np.linspace(1.0, 1.4, n // 2), np.linspace(1.4, 1.1, n - n // 2)]
    )
    return xr.DataArray(
        1_000_000.0 * path,
        dims=("timestamp",),
        coords={"timestamp": BARS[:n]},
    )


def _returns(n: int = N_BARS) -> xr.DataArray:
    """Per-bar returns of `_value`, all inside one calendar month."""
    values = _value(n).values
    returns = np.concatenate([[np.nan], values[1:] / values[:-1] - 1.0])
    return xr.DataArray(
        returns, dims=("timestamp",), coords={"timestamp": BARS[:n]}
    )


def _nan_returns(n: int = N_BARS) -> xr.DataArray:
    """Returns with nothing to draw: no monthly bar, no heatmap."""
    return series(np.full(n, np.nan), BARS[:n])


def _write(
    tmp_path,
    *,
    value=None,
    returns=None,
    metrics=None,
    notes=("a note",),
    in_sample_range=None,
    summary=None,
    title="Backtester_20240101_000000_000000",
    init_cash=None,
    benchmark_value=None,
    benchmark_name=None,
) -> str:
    """Write the page of a hand-built record (returns default to none to draw), return it.

    ``notes`` and ``in_sample_range`` go into the metrics as a run records
    them; ``summary`` is the Setup lines (no config, so no standard ones).
    """
    value = _value() if value is None else value
    path = tmp_path / "report.html"
    page_metrics = None if metrics is None and not notes and in_sample_range is None else dict(metrics or {})
    if page_metrics is not None:
        page_metrics["notes"] = list(notes)
        if in_sample_range is not None:
            page_metrics["in_sample_range"] = in_sample_range
    write_backtest_report(
        account(value, returns=_nan_returns(value.sizes["timestamp"]) if returns is None else returns,
                init_cash=init_cash),
        path,
        title=title,
        metrics=page_metrics,
        benchmark=None if benchmark_value is None else benchmark(benchmark_value, symbol=benchmark_name),
        extras=ReportExtras(setup=summary),
    )
    return path.read_text(encoding="utf-8")


def test_a_minimal_record_writes_a_page(tmp_path):
    """A value curve, its in-sample range and a note are enough for a page."""
    html = _write(
        tmp_path,
        in_sample_range=("2024-01-01", "2024-01-04"),
        notes=["short-side returns are optimistic"],
        title="a run",
    )

    assert html.startswith("<!DOCTYPE html>")
    assert '"name":"equity"' in html and '"name":"drawdown"' in html
    assert '"type":"rect"' in html, "the in-sample band must still be shaded"
    assert "short-side returns are optimistic" in html


def test_a_page_without_a_summary_is_still_valid(tmp_path):
    html = _write(tmp_path, summary=None)
    assert "Dates and setup" not in html
    assert '"name":"equity"' in html


def test_the_summary_labels_and_values_appear_on_the_page(tmp_path):
    summary = {
        "Backtest window": "2024-01-01 .. 2024-01-16 (12 bars)",
        "Bar interval": "1 days 00:00:00",
        "Training window": "2023-06-01 .. 2023-12-29",
        "In-sample range": "—",
    }
    html = _write(tmp_path, summary=summary)

    for label, text in summary.items():
        assert f"<th>{label}</th>" in html, label
        assert f"<td>{text}</td>" in html, text
    # The block keeps the caller's order: the page reads top-down as the
    # caller composed it, not in dict-sorted order.
    positions = [html.index(f"<th>{label}</th>") for label in summary]
    assert positions == sorted(positions)


def test_the_portfolio_construction_parameters_scroll_in_a_short_box(tmp_path):
    """A long rule (a mean-variance optimiser with a factor risk model) does
    not take over the setup table: its full text is on the page, inside a box
    of bounded height that scrolls; the other rows are plain cells."""
    rule = "MeanVarianceOptimizer(" + ", ".join(f"field_{i}={i}" for i in range(400)) + ")"
    html = _write(tmp_path, summary={"Rebalance every": "5 bars", "Portfolio construction": rule})

    assert f'<td><div class="scroll">{rule}</div></td>' in html
    assert "<td>5 bars</td>" in html
    css = html[html.index(".scroll {"):]
    css = css[: css.index("}")]
    assert "max-height" in css and "overflow-y: auto" in css


def test_the_page_states_the_notes_as_text(tmp_path):
    notes = ["first note", "second note"]
    html = _write(tmp_path, notes=notes)
    for note in notes:
        assert f"<li>{note}</li>" in html


def test_angle_brackets_in_the_title_are_escaped(tmp_path):
    """T-sxx-01: a run directory name cannot inject markup.

    The title reaches the page only through the escaped `<h1>`; it is
    deliberately not handed to plotly's layout title, so no unescaped copy of
    it reaches the embedded JSON either. The raw text must be absent from the
    WHOLE page, not merely from the heading.
    """
    html = _write(tmp_path, title="run <img src=x onerror=alert(1)>")

    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<h1>run &lt;img src=x onerror=alert(1)&gt;</h1>" in html


def test_angle_brackets_in_a_summary_value_are_escaped(tmp_path):
    """T-sxx-01: a metrics value or a label cannot inject markup either."""
    html = _write(
        tmp_path,
        summary={"<b>label</b>": "<script>alert(1)</script>"},
    )

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<b>label</b>" not in html
    assert "&lt;b&gt;label&lt;/b&gt;" in html


def test_angle_brackets_in_a_note_are_escaped(tmp_path):
    html = _write(tmp_path, notes=["<i>borrow cost</i> is not modelled"])

    assert "<li>&lt;i&gt;borrow cost&lt;/i&gt; is not modelled</li>" in html


def test_the_report_is_one_self_contained_file_loading_plotly_from_the_cdn(tmp_path):
    """D-23: one file per run, whose only external reference is the CDN."""
    html = _write(tmp_path, notes=(), title="a run")

    assert [p.name for p in tmp_path.iterdir()] == ["report.html"]
    assert "cdn.plot.ly" in html


def test_the_module_is_a_leaf():
    """The only `quantlab.*` imports of the report module are `backtest_stats`,
    `run_metrics`, `date_range`, `returns` and `factor_attribution_report`.

    All are leaves (numpy, pandas, xarray, plotly; `backtest_stats` also reads
    the numpy-only Execution rules, `run_metrics` only `backtest_stats` and
    `date_range`), so the report module still drags in no quantlab layer
    (#115: it labels bars and computes turnover through them; #215: the Factor
    attribution section; the Holdings tab's contributions take their one-bar
    returns from `one_bar_returns`, numpy and xarray only; #279: it draws from
    the account and benchmark records of `run_metrics`).
    """
    import pathlib

    import quantlab.runs.backtest_report as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    offenders = [
        line
        for line in source.splitlines()
        if line.startswith(("from quantlab", "import quantlab"))
        and line
        not in (
            "from quantlab.utils import date_range",
            "from quantlab.utils.returns import one_bar_returns",
            "from quantlab.runs import backtest_stats",
            "from quantlab.runs.run_metrics import AccountRecord, BenchmarkRecord, decided_bars",
            "from quantlab.runs.factor_attribution_report import (",
        )
    ]
    assert offenders == []
    section = pathlib.Path(module.__file__).with_name("factor_attribution_report.py")
    assert not [
        line for line in section.read_text(encoding="utf-8").splitlines()
        if line.startswith(("from quantlab", "import quantlab"))
    ]


# ---------------------------------------------------------------------------
# The metric tables never raise and never lose a metric
# ---------------------------------------------------------------------------
#
# The report is written inside a run's staging directory: an exception here
# deletes the whole run, not just the report. The tables are now laid out
# from a catalogue of known metrics (group, label, unit, definition), so the
# guarantee is asserted by rendering mappings the catalogue has never seen:
# an unknown key lands in the "Other" table, and a value of any type renders
# or becomes a dash.


def _cells(html: str) -> dict[str, list[str]]:
    """Every metric row as `label -> [cell, ...]`, parsed back off the page."""
    rows = re.findall(r'<tr><th title="[^"]*">([^<]+)</th>((?:<td[^>]*>[^<]*</td>)+)</tr>', html)
    return {name: re.findall(r"<td[^>]*>([^<]*)</td>", cells) for name, cells in rows}


def test_an_unknown_metric_key_renders_in_other_and_raises_nothing(tmp_path):
    html = _write(tmp_path, metrics={"whole": {"Brand New Metric": 1.25}})

    assert "<h2>Other</h2>" in html
    assert _cells(html)["Brand New Metric"] == ["1.25"]


def test_an_unseen_nested_sub_dict_is_flattened_to_dotted_paths(tmp_path):
    html = _write(tmp_path, metrics={"whole": {"block": {"inner": 3, "deeper": {"x": 0.5}}}})

    cells = _cells(html)
    assert cells["block.inner"] == ["3"] and cells["block.deeper.x"] == ["0.5"]


def test_a_mapping_with_every_shipped_key_deleted_still_renders(tmp_path):
    html = _write(tmp_path, metrics={"whole": {}, "in_sample": None, "out_of_sample": None})

    assert html.startswith("<!DOCTYPE html>")


def test_a_null_or_missing_slice_renders_the_page(tmp_path):
    html = _write(
        tmp_path,
        metrics={"whole": {"Total Return [%]": 5.0}, "in_sample": None, "out_of_sample": {}},
    )

    assert _cells(html)["Total return"] == ["5.00%"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "  ", pd.NaT])
def test_unrenderable_values_become_a_dash_never_nan_or_none(tmp_path, value):
    html = _write(tmp_path, metrics={"whole": {"Sharpe Ratio": value, "Odd": value}})

    cells = _cells(html)
    assert cells.get("Sharpe ratio", ["—"]) == ["—"]
    assert cells["Odd"] == ["—"]
    tables = html.split("Plotly.newPlot")[0]
    assert ">nan<" not in tables and ">None<" not in tables and ">inf<" not in tables


def test_bools_ints_strings_timedeltas_and_numpy_scalars_render(tmp_path):
    html = _write(tmp_path, metrics={"whole": {
        "Flag": True, "Count": 7, "Label": "abc", "Wait": pd.Timedelta(days=3, hours=12),
        "Numpy Float": np.float64(0.5), "Numpy Int": np.int64(9),
        "Max Drawdown Duration": pd.Timedelta(days=375),
    }})

    cells = _cells(html)
    assert cells["Flag"] == ["true"] and cells["Count"] == ["7"] and cells["Label"] == ["abc"]
    assert cells["Wait"] == ["3.5 d"]
    assert cells["Numpy Float"] == ["0.5"] and cells["Numpy Int"] == ["9"]
    assert cells["Longest drawdown"] == ["375 d"]


def test_metric_names_and_values_are_escaped(tmp_path):
    html = _write(tmp_path, metrics={"whole": {"<b>key</b>": "<i>v</i>"}})

    assert "<b>key</b>" not in html and "&lt;b&gt;key&lt;/b&gt;" in html
    assert "<i>v</i>" not in html and "&lt;i&gt;v&lt;/i&gt;" in html


def test_no_metrics_mapping_means_no_tables_and_no_cards(tmp_path):
    html = _write(tmp_path, metrics=None, notes=())

    assert '<table class="metrics">' not in html and 'class="kpi"' not in html


# ---------------------------------------------------------------------------
# The difference column: strategy minus benchmark, decided by type alone
# ---------------------------------------------------------------------------
#
# Every operand is the raw object the report receives at render time, never
# the JSON shape `to_jsonable` later writes: a duration arrives as a
# pd.Timedelta, a missing value as NaN or NaT.


def _difference(tmp_path, key, strategy, benchmark) -> str:
    html = _write(
        tmp_path,
        metrics={"whole": {key: strategy}, "benchmark": {"whole": {key: benchmark}}},
        benchmark_value=_value() * 0 + 1_000_000.0,
        benchmark_name="SPY",
    )
    label = next(label for label, cells in _cells(html).items() if len(cells) == 3)
    return _cells(html)[label][2]


@pytest.mark.parametrize(
    "key, strategy, benchmark, expected",
    [
        ("Total Return [%]", 12.5, 10.0, "+2.50 pp"),
        ("Sharpe Ratio", np.float64(0.5), np.float64(0.75), "-0.25"),
        ("Max Drawdown [%]", 20.0, 30.0, "+10.00 pp"),
        ("Max Drawdown Duration", pd.Timedelta(days=10), pd.Timedelta(days=4), "+6 d"),
        ("Sharpe Ratio", float("nan"), 1.0, "—"),
        ("Sharpe Ratio", 1.0, float("inf"), "—"),
        ("Sharpe Ratio", True, False, "—"),
        ("Max Drawdown Duration", pd.NaT, pd.Timedelta(days=1), "—"),
    ],
)
def test_the_difference_is_decided_by_type(tmp_path, key, strategy, benchmark, expected):
    assert _difference(tmp_path, key, strategy, benchmark) == expected


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def _traces(html: str) -> dict[str, dict]:
    """The trace list plotly embeds, by name -- the run-level locks' parser."""
    start = html.index("[", html.index("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    return {trace["name"]: trace for trace in traces}


def _layout(html: str) -> dict:
    """The layout object plotly embeds: the JSON value after the trace array."""
    start = html.index("[", html.index("Plotly.newPlot("))
    _, end = json.JSONDecoder().raw_decode(html, start)
    layout, _ = json.JSONDecoder().raw_decode(html, html.index("{", end))
    return layout


def test_every_trace_carries_a_name(tmp_path):
    """A trace without `name` raises KeyError in the persisted-report locks."""
    html = _write(tmp_path, value=_recovering(), returns=_returns())

    start = html.index("[", html.index("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    assert traces and all("name" in trace for trace in traces)


def test_the_equity_y_stays_the_raw_value_and_the_multiple_is_customdata(tmp_path):
    """The page must not disagree with `equity.zarr`."""
    value = _value()
    html = _write(tmp_path, init_cash=1_000_000.0)

    equity = _traces(html)["equity"]
    np.testing.assert_allclose(equity["y"], value.values, rtol=1e-12)
    np.testing.assert_allclose(
        equity["customdata"], value.values / 1_000_000.0, rtol=1e-12
    )
    assert "customdata" in equity["hovertemplate"]


def test_the_log_linear_toggle_defaults_to_linear(tmp_path):
    html = _write(tmp_path)

    assert '"yaxis.type":"linear"' in html
    assert '"yaxis.type":"log"' in html
    # Default: the figure's own axis is not log.
    assert '"yaxis":{"type":"log"' not in html


def test_monthly_returns_are_grouped_by_calendar_month(tmp_path):
    """Compounded per calendar month, via `to_period`, not a resample alias."""
    index = pd.to_datetime(
        ["2024-01-10", "2024-01-20", "2024-02-05", "2024-03-01", "2024-03-20"]
    )
    returns = xr.DataArray(
        np.array([0.1, 0.1, -0.5, 0.2, 0.2]),
        dims=("timestamp",),
        coords={"timestamp": index},
    )
    html = _write(tmp_path, returns=returns)

    monthly = _traces(html)["monthly_return"]
    assert len(monthly["y"]) == 3
    np.testing.assert_allclose(
        monthly["y"], [1.1 * 1.1 - 1.0, -0.5, 1.2 * 1.2 - 1.0], atol=1e-12
    )


def test_a_one_month_run_is_a_single_bar(tmp_path):
    """A run whose whole P&L lands in one month is legible, not an error."""
    html = _write(tmp_path, returns=_returns())
    assert len(_traces(html)["monthly_return"]["y"]) == 1


def test_returns_without_a_number_mean_no_monthly_trace(tmp_path):
    assert "monthly_return" not in _traces(_write(tmp_path, returns=_nan_returns()))


# ---------------------------------------------------------------------------
# The year-by-month heatmap: a SECOND plotly div (03.8 D-04)
# ---------------------------------------------------------------------------
#
# The heatmap is its own figure rendered as a second div, never a fourth
# subplot row (RESEARCH Pitfall 6). `_traces` and `_layout` above anchor on the
# FIRST `Plotly.newPlot(` by design -- that is what keeps the exactly-five-
# traces and pixel-budget locks meaningful -- so the heatmap is invisible to
# them. These tests reach it through `_second_figure_traces`.

HEATMAP = "monthly_return_heatmap"
MONTH_LABELS = [f"{month:02d}" for month in range(1, 13)]


def _second_figure(html: str) -> tuple[list[dict], dict]:
    """Trace list and layout of the SECOND `Plotly.newPlot(` on the page."""
    first = html.index("Plotly.newPlot(")
    second = html.index("Plotly.newPlot(", first + 1)
    start = html.index("[", second)
    traces, end = json.JSONDecoder().raw_decode(html, start)
    layout, _ = json.JSONDecoder().raw_decode(html, html.index("{", end))
    return traces, layout


def _second_figure_traces(html: str) -> dict[str, dict]:
    """Traces of the SECOND `Plotly.newPlot(` on the page, by name.

    The existing parsers read the FIRST one by design -- that is what keeps
    the three-row figure's exact-trace-set and pixel-budget locks meaningful.
    The heatmap lives in its own div after it, so without this parser it
    would ship with no lock at all.
    """
    traces, _ = _second_figure(html)
    return {trace["name"]: trace for trace in traces}


def _dated_returns(dates: list[str], values: list[float]) -> xr.DataArray:
    """Returns on the given calendar dates, in `_returns`' shape."""
    return xr.DataArray(
        np.array(values, dtype=float),
        dims=("timestamp",),
        coords={"timestamp": pd.DatetimeIndex(pd.to_datetime(dates))},
    )


def _multi_year_returns() -> xr.DataArray:
    """One return on the 15th of every month, November 2023 to February 2025.

    Starts late in its first year and ends early in its last, so both partial
    rows are exercised. Values are distinct per month so a cell that lands in
    the wrong column cannot match by accident.
    """
    months = pd.period_range("2023-11", "2025-02", freq="M")
    dates = [(period.to_timestamp() + pd.Timedelta(days=14)).strftime("%Y-%m-%d") for period in months]
    values = [0.01 * (i + 1) for i in range(len(months))]
    return _dated_returns(dates, values)


def _rgb(colour: str) -> tuple[int, int, int]:
    """`#rrggbb` or `rgb(r, g, b)` -> `(r, g, b)`.

    Both forms are accepted because plotly serializes a list colorscale as
    written (hex here) but EXPANDS a named scale into `rgb(...)` strings, so a
    parse that knew only hex would fail on the named scale with a parse error
    instead of on the colour assertion that actually matters.
    """
    colour = colour.strip()
    hex_match = re.fullmatch(r"#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})", colour)
    if hex_match:
        return tuple(int(part, 16) for part in hex_match.groups())
    rgb_match = re.fullmatch(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", colour)
    assert rgb_match, f"unparseable colour {colour!r}"
    return tuple(int(part) for part in rgb_match.groups())


def _non_null_cells(heatmap: dict) -> list[tuple[str, str, float]]:
    """Every non-null cell as `(year label, month label, value)`."""
    return [
        (heatmap["y"][row], heatmap["x"][col], value)
        for row, cells in enumerate(heatmap["z"])
        for col, value in enumerate(cells)
        if value is not None
    ]


def test_the_heatmap_puts_a_march_return_in_the_march_column(tmp_path):
    """RESEARCH Pitfall 7: the month off-by-one is otherwise invisible.

    ONE return in ONE known month, and the assertion is on the x LABEL of the
    single non-null cell. Using `period.month` instead of `period.month - 1`
    would put it at `04` with nothing else differing -- the grid's shape, row
    count and non-null count all survive that bug, so asserting shape would
    stay green.
    """
    html = _write(tmp_path, returns=_dated_returns(["2024-03-12"], [0.25]))
    heatmap = _second_figure_traces(html)[HEATMAP]

    assert heatmap["x"] == MONTH_LABELS
    cells = _non_null_cells(heatmap)
    assert len(cells) == 1
    year, month, value = cells[0]
    assert (year, month) == ("2024", "03")
    assert value == pytest.approx(0.25)


def test_the_heatmap_puts_the_same_month_of_two_years_in_two_rows(tmp_path):
    """Year alignment: March 2023 and March 2024 land in their own rows."""
    html = _write(
        tmp_path, returns=_dated_returns(["2023-03-10", "2024-03-10"], [0.1, -0.2])
    )
    heatmap = _second_figure_traces(html)[HEATMAP]

    assert heatmap["y"] == ["2023", "2024"]
    assert sorted(_non_null_cells(heatmap)) == [
        ("2023", "03", pytest.approx(0.1)),
        ("2024", "03", pytest.approx(-0.2)),
    ]


def test_the_heatmap_leaves_the_months_before_a_november_start_null(tmp_path):
    """Uncovered months are JSON null, never a fabricated 0.0.

    A 0.0 cell would read as a flat month the run actually traded.
    """
    html = _write(tmp_path, returns=_multi_year_returns())
    heatmap = _second_figure_traces(html)[HEATMAP]

    assert heatmap["y"] == ["2023", "2024", "2025"]
    first_year = heatmap["z"][0]
    assert len(first_year) == 12
    assert first_year[:10] == [None] * 10
    assert first_year[10:] == pytest.approx([0.01, 0.02])


def test_the_heatmap_leaves_the_months_after_a_february_end_null(tmp_path):
    """The partial LAST year: ten trailing nulls, again never zeros."""
    html = _write(tmp_path, returns=_multi_year_returns())
    heatmap = _second_figure_traces(html)[HEATMAP]

    last_year = heatmap["z"][2]
    assert len(last_year) == 12
    assert last_year[2:] == [None] * 10
    assert last_year[:2] == pytest.approx([0.15, 0.16])
    # The full middle year is covered end to end.
    assert None not in heatmap["z"][1]


@pytest.mark.parametrize(
    "returns",
    [
        _dated_returns(["2024-01-02", "2024-02-02"], [float("nan"), float("nan")]),
        _dated_returns([], []),
    ],
    ids=["all_nan", "empty"],
)
def test_no_heatmap_div_without_returns_and_the_page_is_still_written(tmp_path, returns):
    """T-03.8-03-01: the report is written inside the run's staging directory.

    An exception here would delete the whole run, so nothing-to-draw is an
    early return: no second `Plotly.newPlot(`, no stray caption, and the page
    is still a complete document.
    """
    html = _write(tmp_path, returns=returns)

    assert html.startswith("<!DOCTYPE html>")
    assert html.count("Plotly.newPlot(") == 1
    assert "Monthly returns by year" not in html
    assert '"name":"equity"' in html


def test_the_heatmap_cells_equal_the_monthly_bars(tmp_path):
    """Both panels come from one helper, so the same months carry the same numbers."""
    html = _write(tmp_path, returns=_multi_year_returns())
    bars = _traces(html)["monthly_return"]
    heatmap = _second_figure_traces(html)[HEATMAP]

    from_bars = {
        (label[:4], label[5:7]): value for label, value in zip(bars["x"], bars["y"])
    }
    from_heatmap = {(year, month): value for year, month, value in _non_null_cells(heatmap)}
    assert from_heatmap.keys() == from_bars.keys()
    for key, value in from_bars.items():
        assert from_heatmap[key] == pytest.approx(value, abs=1e-12), key


def test_the_heatmap_colours_losses_red_gains_green_and_zero_grey(tmp_path):
    """G-03.8-1: red for a loss month, green for a gain month, grey at zero.

    The user chose the Western convention (green up, red down) and asked to
    keep the zero-centred diverging scale and the blank uncovered months.

    The hue checks assert the DIRECTION of the scale -- red at the low end,
    green at the high end, a light grey in the middle -- not particular hex
    values, so a later re-tune of the shades stays green while a swapped pair
    or a return to plotly's red-blue scale goes red. `zmid` pins zero to the
    grey midpoint, and `connectgaps` must stay off or the null cells would be
    interpolated over.
    """
    html = _write(tmp_path, returns=_multi_year_returns())
    heatmap = _second_figure_traces(html)[HEATMAP]

    stops = heatmap["colorscale"]
    # The two ends first: their hue is the user-visible claim, so a wrong
    # palette fails here, naming the offending stop, before any shape check.
    red, green, blue = _rgb(stops[0][1])
    assert red > green and red > blue, f"the 0.0 (loss) stop is not red: {stops[0]}"
    red, green, blue = _rgb(stops[-1][1])
    assert green > red and green > blue, f"the 1.0 (gain) stop is not green: {stops[-1]}"

    assert [float(position) for position, _ in stops] == [0.0, 0.5, 1.0]
    middle = _rgb(stops[1][1])
    assert max(middle) - min(middle) <= 16, f"the 0.5 stop is not grey: {stops[1]}"
    assert min(middle) >= 200, f"the 0.5 stop is not light: {stops[1]}"

    assert heatmap["zmid"] == 0
    assert not heatmap.get("connectgaps")


def _sign_colours_of_the_heatmap(html: str) -> tuple[str, str, str]:
    """`(gain, loss, neutral)` colours, read from the heatmap's OWN stops.

    Read from the page rather than imported from the module: comparing the
    bars against the other figure is what locks the two panels together.
    """
    stops = _second_figure_traces(html)[HEATMAP]["colorscale"]
    by_position = {float(position): colour for position, colour in stops}
    return by_position.get(1.0), by_position.get(0.0), by_position.get(0.5)


def test_each_monthly_bar_takes_the_heatmap_colour_of_its_sign(tmp_path):
    """G-03.8-1: a gain bar is the heatmap's gain colour, a loss bar its loss colour.

    A month compounding to exactly 0.0 lands on the heatmap's grey midpoint
    under `zmid=0`, so its bar takes the same grey.
    """
    html = _write(
        tmp_path,
        returns=_dated_returns(
            ["2024-01-10", "2024-02-12", "2024-03-11"], [0.05, -0.03, 0.0]
        ),
    )
    gain, loss, neutral = _sign_colours_of_the_heatmap(html)
    bars = _traces(html)["monthly_return"]

    assert bars.get("marker", {}).get("color") == [gain, loss, neutral]


def test_every_monthly_bar_colour_follows_the_sign_of_its_value(tmp_path):
    """A longer mixed-sign run: every bar's colour matches its own value's sign."""
    months = pd.period_range("2023-01", "2024-12", freq="M")
    dates = [(period.to_timestamp() + pd.Timedelta(days=9)).strftime("%Y-%m-%d") for period in months]
    values = [(-1) ** i * 0.01 * (i + 1) for i in range(len(months))]
    values[5] = 0.0
    values[17] = 0.0
    html = _write(tmp_path, returns=_dated_returns(dates, values))
    gain, loss, neutral = _sign_colours_of_the_heatmap(html)
    bars = _traces(html)["monthly_return"]

    colours = bars.get("marker", {}).get("color")
    assert isinstance(colours, list)
    assert len(colours) == len(bars["y"]) == len(months)
    for value, colour in zip(bars["y"], colours):
        expected = gain if value > 0 else loss if value < 0 else neutral
        assert colour == expected, (value, colour)


def test_the_heatmap_trace_carries_a_name(tmp_path):
    """Every trace on the page is parsed by name; a nameless one raises KeyError."""
    traces, _ = _second_figure(_write(tmp_path, returns=_multi_year_returns()))

    assert len(traces) == 1
    assert traces[0].get("name") == HEATMAP
    assert traces[0]["type"] == "heatmap"


def test_the_heatmap_puts_the_earliest_year_on_top(tmp_path):
    """Reading order: the y axis is reversed, and both axes are categorical."""
    _, layout = _second_figure(_write(tmp_path, returns=_multi_year_returns()))

    assert layout["yaxis"]["autorange"] == "reversed"
    assert layout["yaxis"]["type"] == "category"
    assert layout["xaxis"]["type"] == "category"


def test_the_heatmap_height_grows_with_the_number_of_years(tmp_path):
    """A six-year run must not be squashed into the height of a one-year run."""
    one_year = _second_figure(
        _write(tmp_path, returns=_dated_returns(["2024-03-12"], [0.25]))
    )[1]["height"]
    six_years = _second_figure(
        _write(
            tmp_path,
            returns=_dated_returns(
                [f"{year}-03-12" for year in range(2019, 2025)], [0.1] * 6
            ),
        )
    )[1]["height"]

    assert six_years > one_year


def test_the_heatmap_div_sits_under_the_performance_figure_before_the_notes(tmp_path):
    """Picture, heatmap, the strategy table beside them, then the notes -- and no second plotly.js."""
    html = _write(
        tmp_path, returns=_multi_year_returns(), metrics={"whole": {"Total Return [%]": 1.0}}
    )

    first = html.index("Plotly.newPlot(")
    second = html.index("Plotly.newPlot(", first + 1)
    assert first < html.index("Monthly returns by year") < second < html.index("<h2>Strategy</h2>")
    assert html.index("<h2>Strategy</h2>") < html.index("<h2>Notes</h2>")
    assert html.count("cdn.plot.ly") == 1


# Quick 260916-hro deleted three tests here -- the liquidation markers, the
# off-axis liquidation and the no-liquidation case -- together with
# `test_the_span_markers_are_not_the_liquidation_colour` below. All four
# exercised the `liquidations` parameter, which no longer exists, so they can
# no longer be WRITTEN rather than merely being redundant. What replaces them
# is the exact-set lock below, which goes red if a marker trace reappears.


def test_the_figure_draws_exactly_these_five_traces(tmp_path):
    """The whole trace set, pinned by name (quick 260916-hro).

    An exact set rather than a bare `not in`: it catches a liquidation trace
    coming back AND any other trace arriving unnoticed. The two
    `deepest_drawdown_*` traces are here because the value has a drawdown,
    and `monthly_return` because returns are.
    """
    traces = _traces(_write(tmp_path, value=_recovering(), returns=_returns()))

    assert set(traces) == {
        "equity",
        "drawdown",
        "monthly_return",
        "deepest_drawdown_valley",
        "deepest_drawdown_end",
    }


def test_the_layout_gives_every_axis_title_room_to_render(tmp_path):
    """D-03: an explicit height, short titles, and a per-row pixel budget.

    The overlap this locks was VERTICAL. A y-axis title is rotated 90
    degrees, so its rendered length is measured against its own axis height.
    Without an explicit `height` the div falls back to plotly's 450px
    default, which leaves rows 2 and 3 about 44px tall -- shorter than the
    titles they carry, so all three collided.

    The height alone is not asserted, because a later `row_heights` or margin
    change could re-create the collision at any height. The per-row budget is
    what actually encodes the rule, and it goes red at the old 450px default.
    """
    html = _write(tmp_path, value=_recovering(), returns=_returns())
    layout = _layout(html)

    assert layout.get("height") is not None, "an inherited 450px default is the bug"
    titles = (("yaxis", "value"), ("yaxis2", "drawdown"), ("yaxis3", "monthly return"))
    for axis, title in titles:
        assert layout[axis]["title"]["text"] == title, axis

    # The plotting area is the figure height less its margins. plotly's own
    # defaults are t=100 / b=80; only `b` is overridden here, so the top
    # default is what the figure really uses.
    plot_area = layout["height"] - layout["margin"].get("t", 100) - layout["margin"]["b"]

    # About 6.5px per character at the default font size, the title being
    # rotated onto the vertical axis. Derived from the measured figures in
    # the task's F-5: roughly 33 / 52 / 91px of text against 328 / 140 / 140px
    # of row -- rather than a bare pixel constant with no way to re-derive it.
    for axis, title in titles:
        domain = layout[axis]["domain"]
        row_px = (domain[1] - domain[0]) * plot_area
        assert row_px >= len(title) * 6.5, (axis, row_px, title)


# ---------------------------------------------------------------------------
# The deepest drawdown's span (quick task 260915-v6i)
# ---------------------------------------------------------------------------
#
# The page marks the deepest drawdown of the account's value
# (`backtest_stats.drawdown_span`, locked in tests/test_backtest_stats.py):
# these tests cover the drawing and the wording -- that the two endpoints
# land on the equity curve with the right symbols, that the span is stated
# in TRADING DAYS rather than as a calendar duration, and that a drawdown
# which never recovered is not described as having ended.
#
# Quick 260916-hro moved the up triangle from the bar the drawdown STARTED to
# its VALLEY, so the payload key is `valley` and the trace is
# `deepest_drawdown_valley`. Since #279 the span is derived from the value,
# so a span with an endpoint off the equity axis or missing its keys can no
# longer reach the page, and the tests that handed it one are gone.


def _recovering() -> xr.DataArray:
    """A value curve whose deepest drawdown has its valley on BARS[3] (a
    Thursday, -20%) and recovers on BARS[8], the next Thursday: 5 bars."""
    return series(
        1_000.0 * np.array([1.0, 0.95, 0.9, 0.8, 0.85, 0.88, 0.92, 0.96, 1.01, 1.02, 1.03, 1.04]),
        BARS,
    )


def _rising() -> xr.DataArray:
    """A value curve without a drawdown."""
    return series(1_000.0 * np.linspace(1.0, 1.5, N_BARS), BARS)


def test_the_span_draws_one_triangle_at_each_end_on_the_equity_curve(tmp_path):
    """Up triangle at the VALLEY bar, down triangle at the recovery bar."""
    traces = _traces(_write(tmp_path, value=_recovering()))

    valley = traces["deepest_drawdown_valley"]
    end = traces["deepest_drawdown_end"]
    assert valley["marker"]["symbol"] == "triangle-up"
    assert end["marker"]["symbol"] == "triangle-down"
    assert valley["mode"] == "markers" and end["mode"] == "markers"
    # One point each, sitting exactly on the plotted equity values.
    assert len(valley["y"]) == 1 and len(end["y"]) == 1
    np.testing.assert_allclose(valley["y"], [_recovering().values[3]], rtol=1e-12)
    np.testing.assert_allclose(end["y"], [_recovering().values[8]], rtol=1e-12)
    # Row 1 is the equity row; the drawdown row is x2.
    assert valley["xaxis"] == "x" and end["xaxis"] == "x"
    # Folded in from `test_the_span_markers_are_not_the_liquidation_colour`,
    # deleted in quick 260916-hro: its other half compared against the
    # liquidation marker, which no longer exists on the page. The surviving
    # half is that the two ends share ONE colour, because they are the two
    # ends of a single measurement and are told apart by shape.
    assert valley["marker"]["color"] == end["marker"]["color"]


def test_the_end_marker_states_the_span_in_trading_days_not_calendar_days(tmp_path):
    """D-2: the number beside the span is a BAR COUNT, worded as such.

    The span runs from a Thursday to the following Thursday -- 5 trading days
    but 7 calendar days -- so a page that rendered a timedelta would say
    something different from what vectorbt's own duration measures. The
    `days` count assertion is what catches a `Timedelta` leaking in: its repr
    (`7 days 00:00:00`) carries a `days` that is not part of `trading days`.
    """
    traces = _traces(_write(tmp_path, value=_recovering()))
    hover = traces["deepest_drawdown_end"]["hovertemplate"]

    assert "5" in hover
    assert "trading days" in hover
    assert hover.count("days") == hover.count("trading days"), hover
    assert "7" not in hover, "the calendar span must not appear anywhere"
    # The depth is stated as a percentage, so the marker is self-describing.
    assert "-20.00%" in hover
    # T-hro-03: since 260916-hro the count runs from the VALLEY, so it is not
    # that metric for two independent reasons (a possibly different episode,
    # and a different starting bar). The hover must not claim otherwise.
    assert "Max Drawdown Duration" not in hover


def test_a_never_recovered_span_says_so_and_never_claims_it_ended(tmp_path):
    """A drawdown still open at the last bar must not be described as recovered."""
    traces = _traces(_write(tmp_path))  # `_value` falls to its last bar
    hover = traces["deepest_drawdown_end"]["hovertemplate"]

    assert "not recovered" in hover
    assert "recovers" not in hover
    # Still a bar count, still in trading days.
    assert "trading days" in hover
    assert hover.count("days") == hover.count("trading days"), hover


def test_a_recovered_span_says_it_recovered(tmp_path):
    """The other branch: the marked episode really did end at that bar."""
    hover = _traces(_write(tmp_path, value=_recovering()))["deepest_drawdown_end"]["hovertemplate"]

    assert "not recovered" not in hover
    assert "recovers" in hover


def test_no_drawdown_means_no_marker_traces(tmp_path):
    """A value that never falls has no drawdown to mark."""
    traces = _traces(_write(tmp_path, value=_rising()))

    assert "deepest_drawdown_valley" not in traces
    assert "deepest_drawdown_end" not in traces


@pytest.mark.parametrize("n_bars", [1, 2])
def test_a_very_short_window_still_renders(tmp_path, n_bars):
    """A one- or two-bar window is legitimate, not an error."""
    html = _write(
        tmp_path,
        value=_value(n_bars),
        notes=(),
        title="short",
        summary={"Backtest window": f"2024-01-01 .. 2024-01-02 ({n_bars} bars)"},
    )
    assert '"name":"equity"' in html
