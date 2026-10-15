"""The backtest report page: KPI cards, grouped comparison tables, a sidebar of chart sections.

What is locked here, and what turns it red:

- The metric tables show only the slices that carry information: a fully
  out-of-sample run has one "Strategy" column (plus the benchmark and the
  difference), no empty in-sample or delta column; a run with an in-sample
  part adds an in-sample vs out-of-sample table with their difference.
- Values are formatted by unit: percent with two decimals and ``pp``
  differences, ratios with two decimals, money with thousands separators,
  durations in days, counts as integers; never ``e+06`` or a raw timedelta.
- Drawdowns are negative in every table and card.
- Every known metric, KPI card and chart carries a plain explanation, shown
  on hover; a metric the page does not know,
  from the strategy, the benchmark or the relative block, renders in an
  "Other" group, and nothing raises.
- KPI cards, the "Excess" section and the relative tables appear with a
  benchmark and are left out without one; the "Rolling" section then shows return,
  volatility and Sharpe instead of excess return, IR and beta.
- The cumulative excess figure carries a log and an arithmetic curve with a
  toggle between them; exp(final log) - 1 is the geometric excess, and a
  value that reaches zero leaves a gap instead of raising.
- The windows timeline draws the backtest row and one row per trained model
  (one for a model backtest, one per walk-forward fold), with every window's
  dates as a tooltip, and names a CV run sliding or expanding.
- The "Portfolio" tab draws the turnover series it is given, and the holdings
  and gross exposure of the target weights on the rebalance bars, a symbol a
  row leaves NaN counted at its last target.
"""

import json
import re

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs.backtest_report import write_backtest_report

N = 300
BARS = pd.bdate_range("2023-01-02", periods=N)
SYMBOLS = ["AAA", "BBB", "CCC"]


def _series(values) -> xr.DataArray:
    """``values`` on the first bars of ``BARS``."""
    return xr.DataArray(np.asarray(values, float), dims=("timestamp",), coords={"timestamp": BARS[: len(values)]})


def _paths(seed=0):
    """Per-bar returns and values of a strategy and a benchmark it tracks with beta 1.5."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0004, 0.01, N)
    r = 1.5 * b + rng.normal(0.0002, 0.005, N)
    r[0] = b[0] = 0.0
    value = 1_000_000.0 * np.cumprod(1 + r)
    bvalue = 1_000_000.0 * np.cumprod(1 + b)
    return r, b, value, bvalue


def _metrics(*, benchmark=True, in_sample=False, extra=None):
    """A ``metrics.json``-shaped mapping; ``extra`` adds strategy keys to every slice."""
    strategy = {
        "Start Value": 1_000_000.0,
        "End Value": 2_467_452.03,
        "Total Return [%]": 146.7452,
        "Annualized Return [%]": 19.8324,
        "Annualized Volatility [%]": 29.5888,
        "Max Drawdown [%]": 34.3157,
        "Max Drawdown Duration": "375 days 00:00:00",
        "Sharpe Ratio": 0.759829,
        "Total Fees Paid": 205560.4,
        "Total Orders": 3503,
        "Annualized Turnover [%]": 3974.19,
        "Avg Winning Trade Duration": "12 days 00:14:55.336787564",
        "Rebalance Win Rate [%]": 60.0,
        "Monthly Win Rate [%]": 58.3333,
        **(extra or {}),
    }
    out = {
        "whole": dict(strategy, **{"Total Return [%]": 80.0}) if in_sample else strategy,
        "in_sample": dict(strategy, **{"Total Return [%]": 10.0}) if in_sample else None,
        "out_of_sample": strategy,
        "execution": {"rejected_order_count": 7, "max_target_deviation": 0.001},
        "portfolio_construction": {"failed_bar_count": 0, "failed_bars": []},
        "notes": ["a note"],
    }
    if benchmark:
        bench = {"Total Return [%]": 147.239, "Annualized Return [%]": 19.8804, "Max Drawdown [%]": 35.1228,
                 "Sharpe Ratio": 0.836216, "Annualized Volatility [%]": 25.6341,
                 "Max Drawdown Duration": "493 days 00:00:00"}
        rel = {"Excess Return [%]": -0.19976, "Annualized Excess Return [%]": -0.0400476,
               "Total Return Difference [%]": -0.493885, "Excess Max Drawdown [%]": -35.9943,
               "Tracking Error [%]": 14.8122, "Information Ratio": 0.07067, "Beta": 0.999231,
               "Correlation": 0.865678, "CAPM Alpha [%]": 1.06326, "Win Rate vs Benchmark [%]": 50.159, "Bars": N,
               "Rebalance Win Rate vs Benchmark [%]": 55.3398, "Monthly Win Rate vs Benchmark [%]": 48.3333}
        out["benchmark"] = {"whole": bench, "in_sample": bench if in_sample else None, "out_of_sample": bench}
        out["relative"] = {"whole": rel, "in_sample": rel if in_sample else None, "out_of_sample": rel}
    return out


def _weights():
    """Target weights every fifth bar, alternating between two pairs of symbols; NaN between."""
    w = np.full((N, len(SYMBOLS)), np.nan)
    for t in range(0, N - 1, 5):
        w[t] = [0.5, 0.5, 0.0] if (t // 5) % 2 == 0 else [0.0, 0.5, 0.5]
    return xr.DataArray(w, dims=("timestamp", "symbol"), coords={"timestamp": BARS, "symbol": SYMBOLS})


def _rebalances():
    """The decided rebalances of ``_weights``: every fifth bar but the last."""
    return BARS[0 : N - 1 : 5].values


def _turnover():
    """A turnover series on the fill bars (one bar after each rebalance)."""
    fills = BARS[1 : N : 5]
    return xr.DataArray(np.linspace(0.1, 0.5, len(fills)), dims=("timestamp",), coords={"timestamp": fills})


def _page(tmp_path, *, benchmark=True, in_sample=False, extra=None, portfolio=True):
    """Write a report and return its HTML with the paths it was drawn from."""
    r, b, value, bvalue = _paths()
    kwargs = dict(
        in_sample_range=(str(BARS[0].date()), str(BARS[99].date())) if in_sample else None,
        notes=["a note"],
        title="run",
        metrics=_metrics(benchmark=benchmark, in_sample=in_sample, extra=extra),
        returns=_series(r),
        init_cash=1_000_000.0,
        bars_per_year=252,
    )
    if benchmark:
        kwargs.update(benchmark_value=_series(bvalue), benchmark_returns=_series(b), benchmark_name="QQQ")
    if portfolio:
        kwargs.update(weights=_weights(), turnover=_turnover(), rebalances=_rebalances())
    path = tmp_path / "report.html"
    write_backtest_report(_series(value), path, **kwargs)
    return path.read_text(encoding="utf-8"), (r, b, value, bvalue)


def _table(html, caption):
    """Rows of the table under ``<h2>{caption}</h2>`` as ``[[cell, ...], ...]``, header row first."""
    start = html.index(f"<h2>{caption}</h2>")
    block = html[start : html.index("</table>", start)]
    return [
        [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", block, re.S)
    ]


def _row(table, label):
    """The row of ``table`` whose first cell is ``label``."""
    return next(row for row in table if row and row[0] == label)


def _figures(html):
    """Every embedded figure as ``(traces by name, layout)``, in page order."""
    out, pos = [], 0
    while (i := html.find("Plotly.newPlot(", pos)) >= 0:
        start = html.index("[", i)
        traces, end = json.JSONDecoder().raw_decode(html, start)
        layout, pos = json.JSONDecoder().raw_decode(html, html.index("{", end))
        out.append(({t["name"]: t for t in traces}, layout))
    return out


def _figure_with(html, name):
    """The first figure holding a trace called ``name``, as ``(traces by name, layout)``."""
    return next((traces, layout) for traces, layout in _figures(html) if name in traces)


def _y(trace):
    """A trace's y, decoding plotly's binary ``{"dtype", "bdata"}`` form."""
    import plotly.io as pio

    return np.asarray(pio.from_json(json.dumps({"data": [trace]})).data[0].y, dtype=float)


# --------------------------------------------------------------------- tables


def test_an_out_of_sample_run_shows_strategy_benchmark_and_difference_only(tmp_path):
    html, _ = _page(tmp_path)

    table = _table(html, "Strategy vs QQQ")
    assert table[0] == ["", "Strategy", "QQQ", "Difference"]
    assert "in_sample" not in html and "out_of_sample - in_sample" not in html
    assert "<h2>In-sample vs out-of-sample</h2>" not in html


def test_values_are_formatted_by_unit(tmp_path):
    html, _ = _page(tmp_path)
    table = _table(html, "Strategy vs QQQ")

    assert _row(table, "Total return")[1:] == ["146.75%", "147.24%", "-0.49 pp"]
    assert _row(table, "Sharpe ratio")[1:] == ["0.76", "0.84", "-0.08"]
    assert _row(table, "Longest drawdown")[1:] == ["375 d", "493 d", "-118 d"]
    assert _row(table, "End value")[1] == "2,467,452"
    trading = _table(html, "Trading")
    assert _row(trading, "Fees paid")[1] == "205,560"
    assert _row(trading, "Orders filled")[1] == "3,503"
    assert _row(trading, "Orders rejected")[1] == "7"
    assert _row(trading, "Avg winning round-trip duration")[1] == "12.0 d"
    assert "e+0" not in html.split("Plotly.newPlot")[0]
    assert "days 00:" not in html.split("Plotly.newPlot")[0]


def test_drawdowns_are_negative_everywhere(tmp_path):
    html, _ = _page(tmp_path)

    assert _row(_table(html, "Strategy vs QQQ"), "Max drawdown")[1:3] == ["-34.32%", "-35.12%"]
    assert _row(_table(html, f"Relative to QQQ"), "Excess max drawdown")[1] == "-35.99%"
    assert re.search(r'class="kl">Max drawdown</div><div class="kv[^"]*">-34.32%', html)


def test_every_known_metric_has_a_definition(tmp_path):
    html, _ = _page(tmp_path)

    names = re.findall(r'<tr><th title="([^"]*)">([^<]+)<', html)
    assert names and all(title.strip() for title, _ in names)


def test_every_kpi_card_and_chart_explains_itself_on_hover(tmp_path):
    html, _ = _page(tmp_path)

    cards = re.findall(r'<div class="kpi"( title="[^"]+")?><div class="kl">([^<]+)</div>', html)
    assert cards and all(title for title, _ in cards), [label for title, label in cards if not title]
    charts = re.findall(r'<h3 title="([^"]*)">([^<]+) <span class="i">', html)
    assert [label for _, label in charts] == ["Performance", "Excess", "Rolling", "Portfolio"]
    assert all(title.strip() for title, _ in charts)
    # The page script shows every title as a styled tooltip.
    assert 'id="tip"' in html and "dataset.tip" in html


def test_an_unknown_metric_renders_in_the_other_group_and_nothing_raises(tmp_path):
    html, _ = _page(tmp_path, extra={"Mystery Score": 1234567.891, "Odd [%]": 3.14159, "Blob": {"a": 1}})

    other = _table(html, "Other")
    assert _row(other, "Mystery Score")[1] == "1,234,568"
    assert _row(other, "Odd [%]")[1] == "3.14%"
    assert _row(other, "Blob.a")[1] == "1"


def test_unknown_benchmark_and_relative_metrics_render_in_the_other_group(tmp_path):
    metrics = _metrics()
    metrics["benchmark"]["whole"]["Bench Oddity"] = 2.5
    metrics["relative"]["whole"]["Relative Oddity [%]"] = 1.25
    r, b, value, bvalue = _paths()
    path = tmp_path / "report.html"
    write_backtest_report(_series(value), path, in_sample_range=None, notes=[], title="t", metrics=metrics,
                          benchmark_value=_series(bvalue), benchmark_returns=_series(b), benchmark_name="QQQ")

    other = _table(path.read_text(encoding="utf-8"), "Other")
    assert _row(other, "QQQ: Bench Oddity")[1] == "2.5"
    assert _row(other, "relative: Relative Oddity [%]")[1] == "1.25%"


def test_a_run_with_an_in_sample_part_adds_the_split_table(tmp_path):
    html, _ = _page(tmp_path, in_sample=True)

    table = _table(html, "In-sample vs out-of-sample")
    assert table[0] == ["", "In-sample", "Out-of-sample", "Difference", "Whole"]
    assert _row(table, "Total return")[1:] == ["10.00%", "146.75%", "+136.75 pp", "80.00%"]


def test_with_an_in_sample_part_the_headline_numbers_are_out_of_sample(tmp_path):
    html, _ = _page(tmp_path, in_sample=True)

    table = _table(html, "Strategy vs QQQ (out-of-sample)")
    assert _row(table, "Total return")[1] == "146.75%"
    assert re.search(r'class="kl">Total return</div><div class="kv[^"]*">146.75%', html)
    assert "<h2>Relative to QQQ (out-of-sample)</h2>" in html


def test_the_end_value_card_note_comes_from_the_whole_window(tmp_path):
    metrics = _metrics(benchmark=False, in_sample=True)
    del metrics["out_of_sample"]["End Value"]
    r, b, value, bvalue = _paths()
    path = tmp_path / "report.html"
    write_backtest_report(_series(value), path, in_sample_range=(str(BARS[0].date()), str(BARS[99].date())),
                          notes=[], title="t", metrics=metrics)

    assert re.search(r'class="kl">Total return</div>.*?class="ks">end value 2,467,452<', path.read_text(), re.S)


def test_every_chart_shades_the_in_sample_range(tmp_path):
    html, _ = _page(tmp_path, in_sample=True)

    for name in ("equity", "cumulative_excess_log", "rolling_beta", "turnover"):
        _, layout = _figure_with(html, name)
        assert any(shape.get("type") == "rect" for shape in layout.get("shapes", [])), name


# ---------------------------------------------------------------- KPI cards


def test_the_kpi_cards_with_a_benchmark(tmp_path):
    html, _ = _page(tmp_path)

    labels = re.findall(r'<div class="kl">([^<]+)</div>', html)
    assert labels == ["Total return", "Excess return", "Information ratio", "Win rate", "Sharpe ratio",
                      "Max drawdown", "Beta", "Turnover / year"]
    assert re.search(r'class="kl">Excess return</div><div class="kv neg">-0.20%', html)
    assert re.search(r'class="kl">Win rate</div><div class="kv">55.3%</div><div class="ks">monthly 48.3%', html)


def test_the_win_rates_are_in_the_relative_table_with_a_benchmark(tmp_path):
    html, _ = _page(tmp_path)

    relative = _table(html, "Relative to QQQ")
    assert _row(relative, "Rebalances beating the benchmark")[1] == "55.3%"
    assert _row(relative, "Months beating the benchmark")[1] == "48.3%"
    assert "Rebalances with a gain" not in html


def test_without_a_benchmark_the_win_rates_are_in_the_strategy_table(tmp_path):
    html, _ = _page(tmp_path, benchmark=False)

    strategy = _table(html, "Strategy")
    assert _row(strategy, "Rebalances with a gain")[1] == "60.0%"
    assert _row(strategy, "Months with a gain")[1] == "58.3%"
    assert re.search(r'class="kl">Win rate</div><div class="kv">60.0%</div><div class="ks">monthly 58.3%', html)


def test_without_a_benchmark_the_relative_cards_tables_and_tab_are_left_out(tmp_path):
    html, _ = _page(tmp_path, benchmark=False)

    labels = re.findall(r'<div class="kl">([^<]+)</div>', html)
    assert labels == ["Total return", "Annualised return", "Win rate", "Sharpe ratio", "Max drawdown",
                      "Volatility", "Turnover / year"]
    assert _table(html, "Strategy")[0] == ["", "Strategy"]
    assert "Relative to" not in html
    tabs = re.findall(r'<button class="tab[^"]*" data-tab="[^"]+">([^<]+)</button>', html)
    assert tabs == ["Overview", "Rolling", "Portfolio", "Setup &amp; notes"]


# --------------------------------------------------------------------- charts


def test_the_tabs_with_a_benchmark(tmp_path):
    html, _ = _page(tmp_path)

    tabs = re.findall(r'<button class="tab[^"]*" data-tab="[^"]+">([^<]+)</button>', html)
    assert tabs == ["Overview", "Excess", "Rolling", "Portfolio", "Setup &amp; notes"]


def test_the_cumulative_excess_has_a_log_and_an_arithmetic_curve_and_a_toggle(tmp_path):
    html, (r, b, value, bvalue) = _page(tmp_path)

    traces, layout = _figure_with(html, "cumulative_excess_log")
    log, arith = _y(traces["cumulative_excess_log"]), _y(traces["cumulative_excess_arithmetic"])
    np.testing.assert_allclose(log, np.cumsum(np.log((1 + r) / (1 + b))), atol=1e-12)
    np.testing.assert_allclose(arith, np.cumsum(r - b), atol=1e-12)
    np.testing.assert_allclose(np.exp(log[-1]) - 1, value[-1] / bvalue[-1] - 1, rtol=1e-9)
    assert traces["cumulative_excess_log"].get("visible", True) is True
    assert traces["cumulative_excess_arithmetic"]["visible"] is False
    buttons = [b["label"] for menu in layout["updatemenus"] for b in menu["buttons"]]
    assert buttons == ["Log", "Arithmetic"]
    assert "excess_drawdown" in traces


def test_a_value_that_reaches_zero_leaves_a_gap_in_the_log_excess_instead_of_raising(tmp_path):
    r, b, value, bvalue = _paths()
    value = value.copy()
    value[200:] = 0.0  # wiped out: the ratio (1 + r) / (1 + b) hits zero at bar 200
    path = tmp_path / "report.html"
    write_backtest_report(_series(value), path, in_sample_range=None, notes=[], title="t",
                          benchmark_value=_series(bvalue), benchmark_returns=_series(b), benchmark_name="QQQ")

    traces, _ = _figure_with(path.read_text(encoding="utf-8"), "cumulative_excess_log")
    log = _y(traces["cumulative_excess_log"])
    assert np.isfinite(log[:200]).all() and np.isnan(log[200])


def test_the_rolling_panels_with_and_without_a_benchmark(tmp_path):
    html, (r, b, *_) = _page(tmp_path)
    traces, _ = _figure_with(html, "rolling_beta")
    assert set(traces) == {"rolling_excess_return", "rolling_information_ratio", "rolling_beta"}
    beta = _y(traces["rolling_beta"])
    window = slice(N - 252, N)
    expected = np.cov(r[window], b[window])[0, 1] / np.var(b[window], ddof=1)
    np.testing.assert_allclose(beta[-1], expected, rtol=1e-9)
    assert np.isnan(beta[:251]).all()

    html, _ = _page(tmp_path, benchmark=False)
    traces, _ = _figure_with(html, "rolling_sharpe")
    assert set(traces) == {"rolling_return", "rolling_volatility", "rolling_sharpe"}


def test_the_portfolio_tab_draws_the_given_turnover_holdings_and_gross_exposure(tmp_path):
    html, _ = _page(tmp_path)

    traces, layout = _figure_with(html, "turnover")
    turnover = traces["turnover"]
    # A filled step line: a bar a day wide vanishes on a multi-year axis.
    assert turnover["type"] == "scatter" and turnover["fill"] == "tozeroy"
    np.testing.assert_allclose(_y(turnover), _turnover().values)
    assert "turnover_1y_mean" in traces
    assert set(_y(traces["holdings"])) == {2.0}
    np.testing.assert_allclose(_y(traces["effective_holdings"]), 2.0)
    np.testing.assert_allclose(_y(traces["gross_exposure"]), 1.0)


def test_a_symbol_left_nan_in_a_rebalance_row_counts_at_its_last_target(tmp_path):
    """NaN keeps the holding (a locked position), so the row is not dropped."""
    r, b, value, bvalue = _paths()
    w = _weights().values.copy()
    rows = np.flatnonzero(np.isfinite(w).all(axis=1))
    w[rows[1]] = [np.nan, 0.25, 0.25]  # AAA kept at the previous row's 0.5
    weights = xr.DataArray(w, dims=("timestamp", "symbol"), coords={"timestamp": BARS, "symbol": SYMBOLS})
    path = tmp_path / "report.html"
    write_backtest_report(_series(value), path, in_sample_range=None, notes=[], title="t",
                          weights=weights, turnover=_turnover(), rebalances=_rebalances())

    traces, _ = _figure_with(path.read_text(encoding="utf-8"), "holdings")
    assert len(_y(traces["holdings"])) == len(rows)
    assert _y(traces["holdings"])[1] == 3.0
    np.testing.assert_allclose(_y(traces["gross_exposure"])[1], 1.0)


def test_weights_below_a_tenth_of_a_percent_are_not_counted_as_holdings(tmp_path):
    r, b, value, bvalue = _paths()
    w = _weights().values.copy()
    rows = np.flatnonzero(np.isfinite(w).all(axis=1))
    w[rows] = [0.4995, 0.5, 0.0005]  # the optimiser's residue on CCC
    weights = xr.DataArray(w, dims=("timestamp", "symbol"), coords={"timestamp": BARS, "symbol": SYMBOLS})
    path = tmp_path / "report.html"
    write_backtest_report(_series(value), path, in_sample_range=None, notes=[], title="t",
                          weights=weights, turnover=_turnover(), rebalances=_rebalances())

    traces, _ = _figure_with(path.read_text(encoding="utf-8"), "holdings")
    assert set(_y(traces["holdings"])) == {2.0}
    assert (_y(traces["effective_holdings"]) < 2.01).all()


def test_the_exposure_axis_always_spans_zero_to_one(tmp_path):
    """Round-off in a fully invested book (1 +- 1e-16) must not be magnified."""
    html, _ = _page(tmp_path)

    traces, layout = _figure_with(html, "gross_exposure")
    axis = next(v for k, v in layout.items() if k.startswith("yaxis") and v.get("title", {}).get("text") == "exposure")
    low, high = axis["range"]
    assert low <= 0.0 and high >= 1.0


def test_the_holdings_axis_ticks_are_whole_numbers(tmp_path):
    html, _ = _page(tmp_path)

    _, layout = _figure_with(html, "holdings")
    axis = next(v for k, v in layout.items() if k.startswith("yaxis") and v.get("title", {}).get("text") == "holdings")
    assert axis["dtick"] >= 1 and float(axis["dtick"]).is_integer()


def test_without_weights_or_turnover_the_portfolio_tab_is_left_out(tmp_path):
    html, _ = _page(tmp_path, portfolio=False)

    tabs = re.findall(r'<button class="tab[^"]*" data-tab="[^"]+">([^<]+)</button>', html)
    assert tabs == ["Overview", "Excess", "Rolling", "Setup &amp; notes"]


def test_the_performance_figure_and_the_heatmap_stay_the_first_two_figures(tmp_path):
    html, _ = _page(tmp_path)

    figures = _figures(html)
    assert "equity" in figures[0][0] and "benchmark_equity" in figures[0][0]
    assert "excess_return" not in figures[0][0] and "excess_drawdown" not in figures[0][0]
    assert "monthly_return_heatmap" in figures[1][0]


# ------------------------------------------------------------ better side


def _better(html, caption, label):
    """Which cells of a row carry the "better" class: [strategy, benchmark]."""
    start = html.index(f"<h2>{caption}</h2>")
    block = html[start : html.index("</table>", start)]
    row = next(r for r in re.findall(r"<tr>(.*?)</tr>", block, re.S) if f">{label}</th>" in r)
    cells = re.findall(r"<td([^>]*)>", row)
    return ['class="better"' in c for c in cells[:2]]


def test_the_better_side_of_each_comparison_is_marked(tmp_path):
    html, _ = _page(tmp_path)

    caption = "Strategy vs QQQ"
    assert _better(html, caption, "Total return") == [False, True]  # 146.75% < 147.24%
    assert _better(html, caption, "Annualised volatility") == [False, True]  # lower is better
    assert _better(html, caption, "Max drawdown") == [True, False]  # -34.32% beats -35.12%
    assert _better(html, caption, "Longest drawdown") == [True, False]  # 375 d beats 493 d
    assert _better(html, caption, "Sharpe ratio") == [False, True]


def test_rows_without_a_direction_or_with_a_tie_mark_nothing(tmp_path):
    html, _ = _page(tmp_path, extra={"Skew": 0.5})
    # Skew has no better side; a missing benchmark value marks nothing either.
    assert "better" not in re.search(r"<tr>(?:(?!</tr>).)*>Skew</th>.*?</tr>", html, re.S).group(0)
    assert "better" not in re.search(r"<tr>(?:(?!</tr>).)*>End value</th>.*?</tr>", html, re.S).group(0)


# ------------------------------------------------------------------ timeline


def _windows(n_folds, *, expanding=False, in_sample=True):
    """Windows of a walk-forward run: 60-bar training windows, 20-bar test segments."""
    BARS = pd.bdate_range("2020-01-01", periods=60 + 20 * n_folds + 2)  # noqa: N806
    folds = []
    for i in range(n_folds):
        start = 0 if expanding else 20 * i
        test = 60 + 20 * i
        folds.append({
            "label": f"fold {i}",
            "training": [str(BARS[start].date()), str(BARS[test + 1].date())],
            "traded": [str(BARS[test].date()), str(BARS[test + 19].date())],
            "in_sample": [str(BARS[test].date()), str(BARS[test + 1].date())] if in_sample else None,
        })
    return {
        "backtest": [folds[0]["traded"][0], folds[-1]["traded"][1]],
        "bars": 20 * n_folds,
        "in_sample": [fold["in_sample"] for fold in folds if fold["in_sample"]],
        "out_of_sample": [[str(BARS[62 + 20 * i].date()), str(BARS[79 + 20 * i].date())] for i in range(n_folds)],
        "folds": folds,
    }


def _timeline_page(tmp_path, windows):
    """Write a report carrying ``windows`` and return its HTML."""
    path = tmp_path / "report.html"
    write_backtest_report(_series(_paths()[2]), path, in_sample_range=None, notes=[], title="t", windows=windows)
    return path.read_text(encoding="utf-8")


def _tooltips(html):
    """Every tooltip of the windows timeline."""
    start = html.index("<h2>Windows</h2>")
    return re.findall(r"<title>([^<]+)</title>", html[start : html.index("</svg>", start)])


def test_a_walk_forward_run_draws_one_timeline_row_per_fold(tmp_path):
    windows = _windows(10)
    tips = _tooltips(_timeline_page(tmp_path, windows))

    for fold in windows["folds"]:
        assert f"{fold['label']} training {fold['training'][0]} .. {fold['training'][1]}" in tips
        assert f"{fold['label']} traded {fold['traded'][0]} .. {fold['traded'][1]}" in tips
        assert f"{fold['label']} in-sample {fold['in_sample'][0]} .. {fold['in_sample'][1]}" in tips
    assert sum(tip.startswith("out-of-sample ") for tip in tips) == 10
    assert sum(tip.startswith("in-sample ") for tip in tips) == 10


def test_the_timeline_caption_names_the_window_and_the_kind_of_cv(tmp_path):
    sliding = _timeline_page(tmp_path, _windows(10))
    expanding = _timeline_page(tmp_path, _windows(10, expanding=True))

    assert "(200 bars); 10 folds, sliding training window" in sliding
    assert "10 folds, expanding training window" in expanding


def test_a_model_backtest_is_one_row_of_training_and_traded_bars(tmp_path):
    windows = _windows(1)
    windows["folds"][0]["label"] = "model"
    html = _timeline_page(tmp_path, windows)

    tips = _tooltips(html)
    assert [" ".join(tip.split(" ")[:2]) for tip in tips] == ["model training", "model out-of-sample", "model in-sample"]
    block = html[html.index("<h2>Windows</h2>") : html.index("</svg>")]
    assert "folds" not in block and ">backtest</text>" not in block


def test_a_run_without_a_model_is_one_row_of_traded_bars_and_a_one_entry_legend(tmp_path):
    html = _timeline_page(tmp_path, {"backtest": ["2020-01-02", "2020-06-30"], "bars": 125, "folds": []})

    assert _tooltips(html) == ["traded 2020-01-02 .. 2020-06-30"]
    legend = html[html.index('<div class="legend">') : html.index("</div></div>", html.index('<div class="legend">'))]
    assert re.findall(r"</span>([^<]+)</span>", legend) == ["traded"]


def test_the_legend_lists_only_the_kinds_drawn(tmp_path):
    html = _timeline_page(tmp_path, _windows(3, in_sample=False))

    legend = html[html.index('<div class="legend">') : html.index("</div></div>", html.index('<div class="legend">'))]
    assert re.findall(r"</span>([^<]+)</span>", legend) == ["training", "traded, out-of-sample"]


def test_labels_with_a_utc_offset_share_the_axis_and_nothing_raises(tmp_path):
    windows = _windows(3)
    windows["backtest"] = [f"{windows['backtest'][0]}T14:30:00+00:00", f"{windows['backtest'][1]}T21:00:00+00:00"]
    windows["folds"][0]["training"] = [f"{d}T14:30:00-05:00" for d in windows["folds"][0]["training"]]
    tips = _tooltips(_timeline_page(tmp_path, windows))

    assert any(tip.startswith("fold 0 training") and "-05:00" in tip for tip in tips)


def test_many_folds_shrink_the_rows_and_name_every_fifth(tmp_path):
    html = _timeline_page(tmp_path, _windows(30))

    block = html[html.index("<h2>Windows</h2>") : html.index("</svg>")]
    names = re.findall(r">(fold \d+)</text>", block)
    assert names == [f"fold {i}" for i in range(0, 30, 5)]
    heights = {float(h) for h in re.findall(r'<rect [^>]*height="([\d.]+)"', block)}
    assert max(heights) < 12


def test_a_window_that_does_not_parse_is_left_out_and_nothing_raises(tmp_path):
    windows = _windows(3)
    windows["folds"][1]["training"] = ["not a date", None]
    windows["folds"][2]["traded"] = "2024"
    tips = _tooltips(_timeline_page(tmp_path, windows))

    assert not any(tip.startswith("fold 1 training") for tip in tips)
    assert not any(tip.startswith("fold 2 traded") for tip in tips)
    assert any(tip.startswith("fold 0 training") for tip in tips)


def test_without_windows_there_is_no_timeline(tmp_path):
    html, _ = _page(tmp_path)
    assert "<h2>Windows</h2>" not in html


def test_the_timeline_axis_labels_every_other_year_over_thirteen_years(tmp_path):
    windows = {
        "backtest": ["2020-01-02", "2024-12-31"],
        "folds": [{"label": "model", "training": ["2012-01-03", "2019-12-31"], "traded": ["2020-01-02", "2024-12-31"]}],
    }
    html = _timeline_page(tmp_path, windows)

    block = html[html.index("<h2>Windows</h2>") : html.index("</svg>")]
    assert re.findall(r'text-anchor="middle">([^<]+)</text>', block) == ["2014", "2016", "2018", "2020", "2022", "2024"]
