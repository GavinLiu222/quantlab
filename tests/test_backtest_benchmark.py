"""Benchmark comparison in the backtester.

`BacktestConfig.benchmark_dataset` takes a market dataset holding exactly one
symbol, a `(timestamp, symbol)` panel like the price dataset (an index ETF
such as QQQ, in a store of its own). When it is set, every run also buys and
holds that symbol on the strategy's own bars and reports the portfolio
against it. What is locked here:

- **Loading.** The benchmark is read over the window, reindexed onto the
  strategy's price bars, and a strategy bar it lacks carries its previous
  price forward with one warning. A panel with more than one symbol, a missing
  price column, or a benchmark that starts after the window are refused.
- **Simulation.** The benchmark follows the strategy's execution: all-in at
  the second bar's fill price, from the same initial cash, so the two value
  curves are comparable bar for bar.
- **Metrics.** `metrics["benchmark"]` carries the benchmark's own return
  statistics per slice, and `metrics["relative"]` the excess statistics. The
  whole-window `Excess Return [%]` is exactly `value / benchmark_value - 1` at
  the last bar, in percent, and `Excess Max Drawdown [%]` is the deepest fall
  of that ratio.
- **Persistence and report.** Read through `BacktestRun`, the run's equity
  curve carries both curves, its data fingerprint the benchmark's data, and
  `rebuild_backtester()` rebuilds the benchmark; the HTML report
  (`BacktestRun.report()`) draws the benchmark NAV with the
  portfolio's and adds the excess-return and excess-drawdown rows and tables.
- **run_cv.** The stitched curve and every fold are compared with the
  benchmark over their own bars.

Everything is synthetic, CPU-only and offline.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.runs.backtest_report import write_backtest_report
from quantlab.runs.backtest_stats import win_rates
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.run_metrics import BenchmarkRecord, run_metrics
from quantlab.utils.jsonable import to_jsonable
from tests.backtest_fixtures import (
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
WINDOW_START = 30
WINDOW_END = 55


@pytest.fixture
def warning_messages():
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _bars(n: int = N_BARS) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _benchmark_store(tmp_path, **kwargs):
    kwargs.setdefault("n_bars", N_BARS)
    return write_price_store(tmp_path / "benchmark", symbols=["QQQ"], seed=7, **kwargs)


def _config(tmp_path, benchmark_config=None, **overrides) -> CrossSectionBacktestConfig:
    """Load-mode config over a checkpoint trained on bars 0-24, window bars 30-55."""
    bars = _bars()
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    model_dates = dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **model_dates)
    )
    if benchmark_config is None:
        benchmark_config = _benchmark_store(tmp_path)
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(tmp_path / "backtest", dataset_config, **model_dates),
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(bars[WINDOW_START]),
        end_date=_day(bars[WINDOW_END]),
        output_dir=str(tmp_path / "runs"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        fees=0.0,
        slippage=0.0,
        benchmark_dataset=make_stock_dataset(benchmark_config),
    )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


@pytest.fixture
def benchmark_run(tmp_path):
    backtester = USEquityCrossectionSelectStockVectorBt(_config(tmp_path))
    return backtester, backtester.run()


# --------------------------------------------------------------------------
# Loading and simulation
# --------------------------------------------------------------------------


def test_benchmark_is_simulated_on_the_strategy_bars_as_buy_and_hold(benchmark_run):
    backtester, result = benchmark_run
    benchmark = result.benchmark
    assert benchmark is not None
    np.testing.assert_array_equal(
        benchmark.value.timestamp.values, result.simulation.value.timestamp.values
    )

    store = xr.open_zarr(backtester.config.benchmark_dataset.config.zarr_file_path)
    window = store.sel(timestamp=benchmark.value.timestamp.values).sel(symbol="QQQ")
    init_cash = backtester.config.init_cash
    shares = init_cash / float(window["adjOpen"].values[1])
    expected = np.concatenate(([init_cash], shares * window["adjClose"].values[1:]))
    np.testing.assert_allclose(benchmark.value.values, expected, rtol=1e-10)


def test_a_multi_symbol_benchmark_is_refused(tmp_path):
    config = _config(
        tmp_path,
        benchmark_config=write_price_store(
            tmp_path / "two", symbols=["QQQ", "SPY"], n_bars=N_BARS
        ),
    )
    with pytest.raises(ValueError, match="exactly one symbol"):
        USEquityCrossectionSelectStockVectorBt(config).run()


def test_a_benchmark_that_starts_after_the_window_is_refused(tmp_path):
    config = _config(tmp_path, benchmark_config=_benchmark_store(tmp_path, list_at={"QQQ": 40}))
    with pytest.raises(ValueError, match="must cover the whole backtest window"):
        USEquityCrossectionSelectStockVectorBt(config).run()


def test_missing_benchmark_bars_carry_the_previous_price_with_one_warning(
    tmp_path, warning_messages
):
    benchmark_config = _benchmark_store(tmp_path)
    # Knock out two strategy bars inside the window.
    store = xr.open_zarr(benchmark_config.zarr_file_path).load()
    holes = store.timestamp.values[[WINDOW_START + 5, WINDOW_START + 6]]
    store.drop_sel(timestamp=holes).to_zarr(benchmark_config.zarr_file_path, mode="w")

    result = USEquityCrossectionSelectStockVectorBt(
        _config(tmp_path, benchmark_config=benchmark_config)
    ).run()

    value = result.benchmark.value.to_pandas()
    assert value.loc[holes[0]] == value.loc[holes[1]]
    gap_warnings = [m for m in warning_messages if "carried forward" in m]
    assert len(gap_warnings) == 1 and "2 of the" in gap_warnings[0]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_carry_benchmark_and_relative_blocks(benchmark_run):
    _, result = benchmark_run
    metrics = result.metrics
    assert metrics["benchmark"]["symbol"] == "QQQ"
    assert metrics["benchmark"]["axis_symbol"] == "QQQ"
    for block in ("benchmark", "relative"):
        assert set(metrics[block]) >= {"whole", "in_sample", "out_of_sample"}
    # The window starts after training ends: nothing is in-sample.
    assert metrics["benchmark"]["in_sample"] is None
    assert metrics["relative"]["in_sample"] is None
    assert "Total Return [%]" in metrics["benchmark"]["whole"]


def test_a_runs_blocks_are_run_metrics_of_its_account_record(benchmark_run):
    """The backtester assembles no block itself: they are `run_metrics` of `account(simulation)`.

    Only the engine's own rows (vectorbt's statistics, the rows of `whole`
    before the turnover rows) and the benchmark's display name come from the
    backtester; every other row and block, and their order, is the module's.
    """
    backtester, result = benchmark_run
    metrics = result.metrics
    whole = list(metrics["whole"])
    engine_rows = {key: metrics["whole"][key] for key in whole[: whole.index("Turnover per Rebalance [%]")]}
    split_keys = ("training_window", "in_sample_range", "out_of_sample_ranges")

    expected = run_metrics(
        backtester.account(result.simulation),
        engine_rows=engine_rows,
        split={key: metrics[key] for key in split_keys},
        benchmark=BenchmarkRecord(
            result.benchmark.returns,
            symbol=metrics["benchmark"]["symbol"],
            axis_symbol=metrics["benchmark"]["axis_symbol"],
        ),
    )

    assert list(metrics)[: len(expected)] == list(expected)
    for block, value in expected.items():
        assert json.dumps(to_jsonable(metrics[block])) == json.dumps(to_jsonable(value)), block


def test_the_benchmark_is_named_by_its_dataset_ticker_lookup(tmp_path, monkeypatch):
    """#226: the benchmark's name comes from the lookup its dataset names, as
    of the window's last bar; the axis label is kept beside it."""
    from quantlab.dataset.base import SymbolName, TickerLookup
    from quantlab.dataset.stock import StockDataset

    asked = []

    class Named(TickerLookup):
        def names(self, symbols, day):
            asked.append(day)
            return [SymbolName(f"{s}-NAMED", "Fund") for s in symbols]

    monkeypatch.setattr(StockDataset, "ticker_lookup", lambda self: Named())

    result = USEquityCrossectionSelectStockVectorBt(_config(tmp_path)).run()

    assert result.metrics["benchmark"]["symbol"] == "QQQ-NAMED"
    assert result.metrics["benchmark"]["axis_symbol"] == "QQQ"
    assert asked[-1] == _bars()[WINDOW_END].date()


def test_whole_excess_return_and_drawdown_match_the_value_ratio(benchmark_run):
    _, result = benchmark_run
    value = result.simulation.value.values
    reference = result.benchmark.value.values
    relative = value / reference
    whole = result.metrics["relative"]["whole"]

    # Every `[%]` row is in percent, like vectorbt's `Total Return [%]`.
    assert whole["Excess Return [%]"] == pytest.approx(
        (relative[-1] - 1.0) * 100.0, rel=1e-9
    )
    peak = np.maximum.accumulate(np.maximum(relative, 1.0))
    assert whole["Excess Max Drawdown [%]"] == pytest.approx(
        float((relative / peak - 1.0).min()) * 100.0, abs=1e-10
    )
    assert whole["Strategy Total Return [%]"] == pytest.approx(
        (value[-1] / value[0] - 1.0) * 100.0, rel=1e-9
    )
    assert whole["Benchmark Total Return [%]"] == pytest.approx(
        (reference[-1] / reference[0] - 1.0) * 100.0, rel=1e-9
    )
    assert whole["Total Return Difference [%]"] == pytest.approx(
        whole["Strategy Total Return [%]"] - whole["Benchmark Total Return [%]"],
        rel=1e-9,
    )
    assert whole["Bars"] == value.size


def test_relative_stats_by_hand(benchmark_run):
    backtester, result = benchmark_run
    r = result.simulation.returns.values
    b = result.benchmark.returns.values
    bars_per_year = 252.0
    active = r - b
    whole = result.metrics["relative"]["whole"]

    tracking = np.std(active, ddof=1) * np.sqrt(bars_per_year)
    beta = np.cov(r, b, ddof=1)[0, 1] / np.var(b, ddof=1)
    assert whole["Tracking Error [%]"] == pytest.approx(tracking * 100.0, rel=1e-9)
    assert whole["Information Ratio"] == pytest.approx(
        np.mean(active) * bars_per_year / tracking, rel=1e-9
    )
    assert whole["Beta"] == pytest.approx(beta, rel=1e-9)
    assert whole["CAPM Alpha [%]"] == pytest.approx(
        (np.mean(r) - beta * np.mean(b)) * bars_per_year * 100.0, rel=1e-9
    )
    assert whole["Correlation"] == pytest.approx(np.corrcoef(r, b)[0, 1], rel=1e-9)
    assert whole["Win Rate vs Benchmark [%]"] == pytest.approx(np.mean(r > b) * 100.0)


def test_a_run_without_a_benchmark_is_unchanged(tmp_path):
    config = _config(tmp_path, benchmark_dataset=None)
    result = USEquityCrossectionSelectStockVectorBt(config).run()
    assert result.benchmark is None
    assert "benchmark" not in result.metrics and "relative" not in result.metrics
    # A model run adds its attribution curves; nothing of the benchmark's.
    assert set(BacktestRun.open(result.run_dir).equity().data_vars) == {
        "value", "returns", "universe_value", "gross_value", "group_value"
    }


# --------------------------------------------------------------------------
# Persistence, rebuild and report
# --------------------------------------------------------------------------


def test_run_directory_carries_the_benchmark(benchmark_run):
    backtester, result = benchmark_run
    run = BacktestRun.open(result.run_dir)
    equity = run.equity()
    assert set(equity.data_vars) == {
        "value",
        "returns",
        "benchmark_value",
        "benchmark_returns",
        "universe_value",
        "gross_value",
        "group_value",
    }
    np.testing.assert_allclose(
        equity["benchmark_value"].values, result.benchmark.value.values
    )
    assert "benchmark_dataset" in run.data_fingerprint
    assert run.benchmark_source == backtester.config.benchmark_dataset.config.zarr_file_path
    metrics = run.metrics()
    assert metrics["relative"]["whole"]["Excess Return [%]"] == pytest.approx(
        result.metrics["relative"]["whole"]["Excess Return [%]"]
    )


def test_the_run_rebuilds_the_benchmark(benchmark_run):
    backtester, result = benchmark_run
    rebuilt = BacktestRun.open(result.run_dir).rebuild_backtester()
    assert (
        rebuilt.config.benchmark_dataset.config.zarr_file_path
        == backtester.config.benchmark_dataset.config.zarr_file_path
    )
    again = rebuilt.run()
    np.testing.assert_allclose(again.benchmark.value.values, result.benchmark.value.values)


def test_report_draws_benchmark_nav_and_the_excess_tab(benchmark_run):
    _, result = benchmark_run
    page = BacktestRun.open(result.run_dir).report()
    for name in (
        "equity",
        "benchmark_equity",
        "benchmark_drawdown",
        "benchmark_monthly_return",
        "cumulative_excess_log",
        "cumulative_excess_arithmetic",
        "excess_drawdown",
        "turnover",
    ):
        assert f'"name":"{name}"' in page, name
    assert re.search(r"<h2>Strategy vs [^<]+</h2>", page)
    assert re.search(r"<h2>Relative to [^<]+</h2>", page)
    # The excess is stated once, in the relative table.
    assert ">Excess return (geometric)</th>" in page
    assert "Excess return vs benchmark" not in page


def _page_traces(page: str) -> dict:
    """Every trace of every figure on the page, by name."""
    traces, pos = {}, 0
    while (i := page.find("Plotly.newPlot(", pos)) >= 0:
        batch, pos = json.JSONDecoder().raw_decode(page, page.index("[", i))
        traces.update({trace["name"]: trace for trace in batch})
    return traces


def test_report_excess_curves_are_the_value_ratio(tmp_path):
    ts = pd.bdate_range("2024-01-01", periods=5)
    value = xr.DataArray([100.0, 110.0, 99.0, 120.0, 118.0], dims=("timestamp",), coords={"timestamp": ts})
    bench = xr.DataArray([100.0, 100.0, 100.0, 100.0, 110.0], dims=("timestamp",), coords={"timestamp": ts})
    path = tmp_path / "report.html"
    write_backtest_report(
        value,
        path,
        in_sample_range=None,
        notes=[],
        title="t",
        benchmark_value=bench,
        benchmark_name="QQQ",
    )
    page = path.read_text(encoding="utf-8")
    import plotly.io as pio

    traces = _page_traces(page)
    log = pio.from_json(json.dumps({"data": [traces["cumulative_excess_log"]]})).data[0].y
    excess_dd = pio.from_json(json.dumps({"data": [traces["excess_drawdown"]]})).data[0].y
    np.testing.assert_allclose(np.exp(log) - 1, [0.0, 0.1, -0.01, 0.2, 118 / 110 - 1], atol=1e-12)
    np.testing.assert_allclose(
        excess_dd, [0.0, 0.0, 0.99 / 1.1 - 1, 0.0, (118 / 110) / 1.2 - 1]
    )
    first_figure = page[: page.index("Plotly.newPlot(", page.index("Plotly.newPlot(") + 1)]
    assert '"name":"excess_drawdown"' not in first_figure


def test_report_without_benchmark_keeps_three_rows(tmp_path):
    ts = pd.bdate_range("2024-01-01", periods=5)
    value = xr.DataArray([100.0, 110.0, 99.0, 120.0, 118.0], dims=("timestamp",), coords={"timestamp": ts})
    path = tmp_path / "report.html"
    write_backtest_report(value, path, in_sample_range=None, notes=[], title="t")
    page = path.read_text(encoding="utf-8")
    assert '"yaxis3"' in page and '"yaxis4"' not in page
    assert "excess_return" not in page


# --------------------------------------------------------------------------
# run_cv
# --------------------------------------------------------------------------


@pytest.mark.parametrize("expanding", [False, True], ids=["sliding", "expanding"])
def test_run_cv_compares_the_stitched_curve_and_every_fold(tmp_path, expanding):
    n_bars, train_periods = 80, 30
    dataset_config = write_price_store(tmp_path / "store", n_bars=n_bars)
    bars = _bars(n_bars)
    dates = dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[-1]),
        train_start=_day(bars[0]),
        train_end=_day(bars[train_periods - 1]),
        test_start=_day(bars[train_periods]),
        test_end=_day(bars[-1]),
    )
    model = make_model(tmp_path / "train", dataset_config, **dates)
    model.collect()
    cv = model.train_cv(train_periods=train_periods, expanding=expanding)

    config = CrossSectionBacktestConfig(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(tmp_path / "backtest", dataset_config, **dates),
        model_mode="load",
        cv_project_dir=str(cv.path),
        start_date=_day(bars[train_periods]),
        end_date=_day(bars[-1]),
        output_dir=str(tmp_path / "runs"),
        rebalance_periods=2,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        benchmark_dataset=make_stock_dataset(_benchmark_store(tmp_path, n_bars=n_bars)),
    )
    cv = USEquityCrossectionSelectStockVectorBt(config).run_cv()

    assert cv.benchmark is not None
    stitched = cv.metrics["stitched"]
    relative = cv.simulation.value.values / cv.benchmark.value.values
    assert stitched["relative"]["whole"]["Excess Return [%]"] == pytest.approx(
        (relative[-1] - 1.0) * 100.0, rel=1e-9
    )
    for record in cv.folds:
        assert "relative" in record["metrics"]
    run = BacktestRun.open(cv.run_dir)
    assert [fold.index for fold in run.folds] == [record["fold"] for record in cv.folds]
    for fold in run.folds:
        assert "benchmark_value" in fold.equity().data_vars
    assert "benchmark_dataset" in run.data_fingerprint
    page = BacktestRun.open(cv.run_dir).report()
    assert '"name":"excess_drawdown"' in page
    # The windows timeline draws one row per fold, each with the training
    # window and the traded bars metrics.json records for it.
    assert "<h2>Windows</h2>" in page
    for record in cv.folds:
        training = record["metrics"]["training_window"]
        assert f"fold {record['fold']} training {training[0]} .. {training[1]}</title>" in page
        assert f"fold {record['fold']} traded {record['test_start']} .. {record['test_end']}</title>" in page
    kind = "expanding" if expanding else "sliding"
    assert f"{len(cv.folds)} folds, {kind} training window" in page


# --------------------------------------------------------------------------
# win rates: per holding period and per calendar month
# --------------------------------------------------------------------------


def _expected_win_rates(result, *, against_benchmark: bool) -> tuple[float, float]:
    """Share of holding periods and of months whose compounded return beats
    the benchmark's (or zero). A holding period runs from a fill bar up to
    the bar before the next fill; bars before the first fill hold nothing."""
    ts = pd.DatetimeIndex(result.simulation.returns.timestamp.values)
    r = pd.Series(result.simulation.returns.values, ts)
    b = pd.Series(result.benchmark.returns.values, ts) if against_benchmark else pd.Series(0.0, ts)
    fills = np.unique(result.simulation.orders["timestamp"].values.astype("datetime64[ns]"))
    period = np.searchsorted(fills, ts.values.astype("datetime64[ns]"), side="right") - 1
    ok = (period >= 0) & r.notna().values & b.notna().values

    def share(keys):
        frame = pd.DataFrame({"r": r.values[ok], "b": b.values[ok], "k": keys[ok]})
        grouped = frame.groupby("k")
        rp = grouped["r"].apply(lambda x: np.prod(1 + x) - 1)
        bp = grouped["b"].apply(lambda x: np.prod(1 + x) - 1)
        return float((rp > bp).mean() * 100)

    return share(period), share(np.asarray(ts.to_period("M").astype(str)))


def test_win_rates_per_holding_period_and_per_month(benchmark_run):
    _, result = benchmark_run

    rebalance, monthly = _expected_win_rates(result, against_benchmark=True)
    relative = result.metrics["relative"]["whole"]
    assert relative["Rebalance Win Rate vs Benchmark [%]"] == pytest.approx(rebalance)
    assert relative["Monthly Win Rate vs Benchmark [%]"] == pytest.approx(monthly)

    rebalance, monthly = _expected_win_rates(result, against_benchmark=False)
    whole = result.metrics["whole"]
    assert whole["Rebalance Win Rate [%]"] == pytest.approx(rebalance)
    assert whole["Monthly Win Rate [%]"] == pytest.approx(monthly)
    for name in ("in_sample", "out_of_sample"):
        block = result.metrics.get(name)
        if block:
            assert "Rebalance Win Rate [%]" in block and "Monthly Win Rate [%]" in block


def test_a_bar_at_or_below_minus_100_percent_is_left_out_of_the_win_rates():
    """A value that reached zero has no log return: the bar is dropped, not counted as a loss."""
    bars = pd.date_range("2024-01-01", periods=6, freq="D")
    returns = xr.DataArray([0.0, 0.1, -1.0, 0.05, np.nan, 0.02], dims=("timestamp",), coords={"timestamp": bars})

    rates = win_rates(returns, bars[[1, 3]].values, ranges=[("2024-01-01", "2024-01-06")])

    # Two holding periods: bar 1 (+10%) with bar 2 dropped, and bars 3-5 (+5%, NaN, +2%).
    assert rates["Rebalance Win Rate [%]"] == pytest.approx(100.0)
    assert rates["Monthly Win Rate [%]"] == pytest.approx(100.0)
