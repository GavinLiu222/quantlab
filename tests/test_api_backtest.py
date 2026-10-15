"""`quantlab.api.backtest` and `BacktestReport`: a caller's frames in, a report out.

Tested only through the public function and the report it returns (ADR 0011). Numerical
correctness is parity with the library path: the same weights backtested through the API
on a frame and through `USEquityCrossectionSelectStockVectorBt.run_weights` on the
equivalent Zarr store give the same equity, orders and metrics, and `scores` + `top_n`
equals selecting with `TopNConstructor` first and backtesting the weights. The
weight-frame rules (long or wide, a symbol missing on a given bar is 0, a bar missing
entirely is a hold) are tested here too, since the conversion has no tests of its own.
A kept run is read through `BacktestRun`; its file names appear only in the directory
listing (`RUN_DIR_ARTIFACTS`), whose subject is the layout.

Everything is synthetic, CPU-only and offline.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest
import xarray as xr
from loguru import logger

import quantlab.api as qa
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import BacktestConfig, CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.dataset.stock import StockDataset
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from quantlab.execution.rules import ExecutionSettings
from tests.backtest_fixtures import SYMBOLS, write_price_store

N_BARS = 40
ADJUSTED = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
CANONICAL = ("open", "high", "low", "close", "volume")
RUN_DIR_ARTIFACTS = [
    "config.json",
    "equity.zarr",
    "holdings.zarr",
    "inputs",
    "metrics.json",
    "report.html",
    "run.json",
    "settlements.json",
    "weights.zarr",
]


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    """Fail the test if anything starts a W&B run."""
    import wandb

    def _refuse(*args, **kwargs):
        raise AssertionError("the API must never start a W&B run")

    monkeypatch.setattr(wandb, "init", _refuse)


@pytest.fixture
def stores(tmp_path):
    """A price store and a one-symbol benchmark store, as Zarr and as frames."""
    price_config = write_price_store(tmp_path / "prices", n_bars=N_BARS)
    benchmark_config = write_price_store(
        tmp_path / "benchmark", symbols=["QQQ"], seed=7, n_bars=N_BARS
    )
    return dict(
        root=tmp_path,
        prices=price_config,
        benchmark=benchmark_config,
        frame=_canonical_frame(price_config.zarr_file_path),
        benchmark_frame=_canonical_frame(benchmark_config.zarr_file_path),
    )


def _canonical_frame(zarr_path: str) -> pd.DataFrame:
    """The adjusted columns of a store as a long frame under the canonical names."""
    store = xr.open_zarr(zarr_path).load()
    return (
        store[list(ADJUSTED)]
        .rename(dict(zip(ADJUSTED, CANONICAL)))
        .to_dataframe()
        .reset_index()
    )


def _bars() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=N_BARS)


def _library_run(stores, weights: xr.Dataset, *, benchmark: bool = False):
    """Backtest ``weights`` on the Zarr store through the library backtester."""
    bars = _bars()
    config = CrossSectionBacktestConfig(
        price_dataset=StockDataset(stores["prices"]),
        start_date=bars[0].strftime("%Y-%m-%d"),
        end_date=bars[-1].strftime("%Y-%m-%d"),
        output_dir=None,
        rebalance_periods=1,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        benchmark_dataset=StockDataset(stores["benchmark"]) if benchmark else None,
    )
    return USEquityCrossectionSelectStockVectorBt(config).run_weights(weights)


def _scores(seed: int = 3) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(N_BARS, len(SYMBOLS))),
        dims=("timestamp", "symbol"),
        coords={"timestamp": _bars(), "symbol": SYMBOLS},
    )


def _selected_weights(stores, *, direction="long_only", top_n=2, periods=5) -> xr.Dataset:
    """Top-N weights selected by the library's top-n rule on the store's own prices,
    as the library backtester drives it, the holdings replayed with the default costs."""
    scores = _scores()
    return DecisionInputs(
        StockDataset(stores["prices"]),
        TopNConstructor(TopNConfig(direction=direction, top_n=top_n)),
        fill_column="adjOpen",
        valuation_column="adjClose",
        rebalance_periods=periods,
        anchor=scores.timestamp.values[0],
        execution=ExecutionSettings(fees=BacktestConfig.fees, slippage=BacktestConfig.slippage),
    ).weights(scores.to_dataset(name="score"))[["weight"]]


def _long(weights: xr.Dataset) -> pd.DataFrame:
    return weights.to_dataframe().reset_index()


def _assert_same_block(got: dict, want: dict) -> None:
    assert sorted(got) == sorted(want)
    for key, value in want.items():
        if isinstance(value, float) and np.isnan(value):
            assert np.isnan(got[key]), key
        else:
            assert got[key] == value, key


# --------------------------------------------------------------------------- parity


def test_weights_through_the_api_match_the_zarr_backed_backtester(stores):
    weights = _selected_weights(stores)
    expected = _library_run(stores, weights)

    report = qa.backtest(stores["frame"], weights=_long(weights))

    np.testing.assert_array_equal(report.equity["value"].to_numpy(), expected.simulation.value.values)
    np.testing.assert_array_equal(
        report.equity["timestamp"].to_numpy(), expected.simulation.value.timestamp.values
    )
    np.testing.assert_array_equal(
        report.returns["returns"].to_numpy(), expected.simulation.returns.values
    )
    _assert_same_block(report.metrics["whole"], expected.metrics["whole"])
    assert report.metrics["notes"] == expected.metrics["notes"]
    assert sorted(report.metrics) == ["execution", "notes", "whole"]
    assert len(report.orders) == expected.simulation.orders.sizes["order"]
    np.testing.assert_array_equal(
        report.orders["size"].to_numpy(), expected.simulation.orders["size"].values
    )
    xr.testing.assert_identical(report.raw.weights, expected.weights)


def test_a_sparse_long_frame_means_zero_for_absent_symbols_and_hold_for_absent_bars(stores):
    weights = _selected_weights(stores)
    expected = _library_run(stores, weights)
    frame = _long(weights)
    # Keep only the held names on rebalance bars: every other bar is absent (hold) and
    # every unselected symbol is absent on a rebalance bar (weight 0).
    sparse = frame[frame["weight"].fillna(0.0) != 0.0].reset_index(drop=True)
    assert len(sparse) < len(frame)

    report = qa.backtest(stores["frame"], weights=sparse)

    np.testing.assert_array_equal(report.equity["value"].to_numpy(), expected.simulation.value.values)
    xr.testing.assert_identical(report.raw.weights, expected.weights)


def test_wide_weight_frames_match_the_long_frame(stores):
    weights = _selected_weights(stores)
    long_report = qa.backtest(stores["frame"], weights=_long(weights))
    wide_pandas = weights["weight"].to_pandas()
    wide_polars = pl.from_pandas(wide_pandas.reset_index())

    for wide in (wide_pandas, wide_polars):
        report = qa.backtest(stores["frame"], weights=wide)
        np.testing.assert_array_equal(
            report.equity["value"].to_numpy(), long_report.equity["value"].to_numpy()
        )


@pytest.mark.parametrize("index_name", [None, "date", "timestamp"])
def test_any_datetime_index_holds_the_timestamps_of_a_wide_frame(stores, index_name):
    weights = _selected_weights(stores)
    expected = qa.backtest(stores["frame"], weights=_long(weights))
    wide = weights["weight"].to_pandas().rename_axis(index=index_name, columns=None)

    report = qa.backtest(stores["frame"], weights=wide)

    np.testing.assert_array_equal(report.equity["value"], expected.equity["value"])


def test_a_sparse_long_frame_and_its_pivot_give_the_same_backtest(stores):
    frame = _long(_selected_weights(stores))
    sparse = frame[frame["weight"].fillna(0.0) != 0.0].reset_index(drop=True)
    # The pivot leaves NaN where the long frame has no row: on a bar with weights that
    # is 0, and a bar the long frame lacks is not in the pivot at all (a hold).
    wide = sparse.pivot(index="timestamp", columns="symbol", values="weight")
    assert wide.isna().any().any()

    long_report = qa.backtest(stores["frame"], weights=sparse)
    wide_report = qa.backtest(stores["frame"], weights=wide)
    polars_report = qa.backtest(stores["frame"], weights=pl.from_pandas(wide.reset_index()))

    xr.testing.assert_identical(wide_report.raw.weights, long_report.raw.weights)
    xr.testing.assert_identical(polars_report.raw.weights, long_report.raw.weights)
    np.testing.assert_array_equal(wide_report.equity["value"], long_report.equity["value"])


def test_an_all_nan_wide_row_is_a_hold(stores):
    weights = _selected_weights(stores, periods=1)
    wide = weights["weight"].to_pandas()
    wide.iloc[3] = np.nan

    report = qa.backtest(stores["frame"], weights=wide)

    assert report.raw.weights["weight"].isel(timestamp=3).isnull().all()
    assert report.raw.weights["weight"].isel(timestamp=2).notnull().all()


@pytest.mark.parametrize(
    "prices_zone, weights_zone, hint",
    [
        (None, "America/New_York", r"prices are naive, taken as UTC; weights were America/New_York"),
        ("Asia/Tokyo", None, r"prices were Asia/Tokyo; weights are naive, taken as UTC"),
    ],
)
def test_misaligned_bars_in_another_time_zone_hint_at_the_zone(
    stores, prices_zone, weights_zone, hint
):
    prices = stores["frame"].copy()
    weights = _long(_selected_weights(stores))
    if prices_zone:
        prices["timestamp"] = prices["timestamp"].dt.tz_localize(prices_zone)
    if weights_zone:
        weights["timestamp"] = weights["timestamp"].dt.tz_localize(weights_zone)

    with pytest.raises(ValueError, match=hint):
        qa.backtest(prices, weights=weights)


def test_misaligned_scores_and_benchmark_hint_at_the_zone(stores):
    scores = _scores().rename("score").to_dataframe().reset_index()
    scores["timestamp"] = scores["timestamp"].dt.tz_localize("Europe/Berlin")
    with pytest.raises(ValueError, match=r"scores were Europe/Berlin"):
        qa.backtest(stores["frame"], scores=scores, top_n=2)

    benchmark = stores["benchmark_frame"].copy()
    benchmark["timestamp"] = benchmark["timestamp"].dt.tz_localize("Europe/Berlin")
    with pytest.raises(ValueError, match=r"benchmark were Europe/Berlin"):
        qa.backtest(
            stores["frame"], weights=_long(_selected_weights(stores)), benchmark=benchmark
        )


def test_misaligned_bars_in_the_same_zone_give_no_zone_hint(stores):
    weights = _long(_selected_weights(stores))
    weights.loc[weights["timestamp"] == _bars()[-1], "timestamp"] = pd.Timestamp("2030-01-01")

    with pytest.raises(ValueError) as caught:
        qa.backtest(stores["frame"], weights=weights)
    assert "naive" not in str(caught.value)


@pytest.mark.parametrize("direction", ["long_only", "long_short"])
def test_scores_with_top_n_equal_selecting_first_then_backtesting_the_weights(stores, direction):
    weights = _selected_weights(stores, direction=direction, top_n=2, periods=3)
    via_weights = qa.backtest(stores["frame"], weights=_long(weights), rebalance_periods=3)
    scores = _scores().rename("score").to_dataframe().reset_index()

    report = qa.backtest(
        stores["frame"], scores=scores, top_n=2, direction=direction, rebalance_periods=3
    )

    xr.testing.assert_identical(report.raw.weights, via_weights.raw.weights)
    np.testing.assert_array_equal(
        report.equity["value"].to_numpy(), via_weights.equity["value"].to_numpy()
    )
    _assert_same_block(report.metrics["whole"], via_weights.metrics["whole"])


def test_a_symbol_without_scores_is_never_selected(stores):
    scores = _scores().rename("score").to_dataframe().reset_index()
    scores = scores[scores["symbol"] != "AAA"]

    report = qa.backtest(stores["frame"], scores=scores, top_n=5)

    held = report.weights[report.weights["symbol"] == "AAA"]["weight"]
    assert (held.dropna() == 0.0).all()


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, r"exactly one of weights= and scores="),
        ({"weights": "W", "scores": "S"}, r"exactly one of weights= and scores="),
        ({"scores": "S"}, r"scores= needs top_n="),
        ({"weights": "W", "top_n": 2}, r"top_n=2 applies only with scores="),
        (
            {"weights": "W", "direction": "long_short"},
            r"direction='long_short' applies only with scores=",
        ),
    ],
)
def test_the_signal_arguments_are_checked(stores, kwargs, message):
    frames = {
        "W": _long(_selected_weights(stores)),
        "S": _scores().rename("score").to_dataframe().reset_index(),
    }
    given = {key: frames.get(value, value) for key, value in kwargs.items()}

    with pytest.raises(ValueError, match=message):
        qa.backtest(stores["frame"], **given)


def test_gross_exposure_above_one_raises_naming_the_bar(stores):
    frame = _long(_selected_weights(stores))
    bar = _bars()[5]
    frame.loc[frame["timestamp"] == bar, "weight"] = 0.3

    with pytest.raises(ValueError, match=rf"weight row at {bar:%Y-%m-%d} has gross exposure"):
        qa.backtest(stores["frame"], weights=frame)


def test_an_explicit_nan_beside_finite_weights_keeps_that_holding(stores):
    frame = _long(_selected_weights(stores))
    bar = _bars()[5]
    on_bar = frame["timestamp"] == bar
    frame.loc[on_bar, "weight"] = 0.1
    frame.loc[on_bar & (frame["symbol"] == "AAA"), "weight"] = np.nan

    report = qa.backtest(stores["frame"], weights=frame)

    orders = report.raw.simulation.orders
    fill_bar = np.datetime64(_bars()[6])
    traded = {str(s) for s, t in zip(orders["symbol"].values, orders["timestamp"].values) if t == fill_bar}
    assert "AAA" not in traded and traded


def test_weights_on_a_symbol_without_prices_raise(stores):
    frame = _long(_selected_weights(stores))
    frame.loc[frame["symbol"] == "AAA", "symbol"] = "ZZZ"

    with pytest.raises(
        ValueError, match=r"weights name 1 symbol\(s\) the prices do not have.*'ZZZ'"
    ):
        qa.backtest(stores["frame"], weights=frame)


def test_weights_on_a_bar_without_prices_raise(stores):
    frame = _long(_selected_weights(stores))
    frame.loc[frame["timestamp"] == _bars()[-1], "timestamp"] = pd.Timestamp("2030-01-01")

    with pytest.raises(
        ValueError, match=r"weights name 1 bar\(s\) the prices do not have.*2030-01-01"
    ):
        qa.backtest(stores["frame"], weights=frame)


def test_a_weight_frame_with_two_value_columns_raises(stores):
    frame = _long(_selected_weights(stores)).assign(other=1.0)

    with pytest.raises(ValueError, match=r"one value column.*'weight', 'other'"):
        qa.backtest(stores["frame"], weights=frame)


def test_an_unknown_market_raises_listing_the_valid_ones(stores):
    with pytest.raises(ValueError, match=r"'equity', 'crypto'"):
        qa.backtest(stores["frame"], weights=_long(_selected_weights(stores)), market="fx")


def test_a_missing_fill_column_raises_naming_it(stores):
    with pytest.raises(ValueError, match=r"'vwap'"):
        qa.backtest(stores["frame"], weights=_long(_selected_weights(stores)), fill="vwap")


# --------------------------------------------------------------------------- market


def test_fill_and_valuation_choose_the_price_columns(stores):
    weights = _long(_selected_weights(stores))
    frame = stores["frame"]
    renamed = frame.rename(columns={"open": "o", "close": "c"})
    swapped = frame.assign(open=frame["close"], close=frame["open"])

    default = qa.backtest(frame, weights=weights)
    chosen = qa.backtest(renamed, weights=weights, fill="o", valuation="c")
    reversed_ = qa.backtest(swapped, weights=weights, fill="close", valuation="open")

    np.testing.assert_array_equal(chosen.equity["value"], default.equity["value"])
    np.testing.assert_array_equal(reversed_.equity["value"], default.equity["value"])
    other = qa.backtest(frame, weights=weights, fill="close", valuation="close")
    assert not np.array_equal(other.equity["value"], default.equity["value"])


def test_market_sets_the_annualization(stores):
    weights = _long(_selected_weights(stores))

    equity = qa.backtest(stores["frame"], weights=weights, market="equity")
    crypto = qa.backtest(stores["frame"], weights=weights, market="crypto")
    overridden = qa.backtest(
        stores["frame"], weights=weights, market="equity", trading_days_per_year=365
    )

    # Daily bars: the annualized ratios scale with the square root of bars per year,
    # while the total return does not depend on the annualization.
    whole_e, whole_c = equity.metrics["whole"], crypto.metrics["whole"]
    assert whole_c["Total Return [%]"] == whole_e["Total Return [%]"]
    assert whole_c["Sharpe Ratio"] == pytest.approx(
        whole_e["Sharpe Ratio"] * np.sqrt(365 / 252)
    )
    _assert_same_block(overridden.metrics["whole"], whole_c)


def test_session_minutes_annualize_intraday_bars(stores):
    frame = stores["frame"].copy()
    days = {day: i for i, day in enumerate(_bars())}
    frame["timestamp"] = pd.Timestamp("2024-01-01 09:30") + pd.to_timedelta(
        frame["timestamp"].map(days) * 5, unit="min"
    )
    weights = _long(_selected_weights(stores))
    weights["timestamp"] = pd.Timestamp("2024-01-01 09:30") + pd.to_timedelta(
        weights["timestamp"].map(days) * 5, unit="min"
    )

    short = qa.backtest(frame, weights=weights, session_minutes_per_day=390)
    long_ = qa.backtest(frame, weights=weights, session_minutes_per_day=1440)

    assert short.raw.simulation.bar_interval == np.timedelta64(5, "m")
    assert long_.metrics["whole"]["Sharpe Ratio"] == pytest.approx(
        short.metrics["whole"]["Sharpe Ratio"] * np.sqrt(1440 / 390)
    )


# --------------------------------------------------------------------------- benchmark


def test_a_benchmark_frame_gives_the_library_excess_metrics(stores):
    weights = _selected_weights(stores)
    expected = _library_run(stores, weights, benchmark=True)

    report = qa.backtest(
        stores["frame"], weights=_long(weights), benchmark=stores["benchmark_frame"]
    )

    assert sorted(report.metrics) == ["benchmark", "execution", "notes", "relative", "whole"]
    assert report.metrics["benchmark"]["symbol"] == "QQQ"
    _assert_same_block(report.metrics["relative"]["whole"], expected.metrics["relative"]["whole"])
    _assert_same_block(
        report.metrics["benchmark"]["whole"], expected.metrics["benchmark"]["whole"]
    )
    assert list(report.benchmark.columns) == ["timestamp", "value", "returns"]
    np.testing.assert_array_equal(report.benchmark["value"], expected.benchmark.value.values)


def test_without_a_benchmark_the_report_has_none(stores):
    report = qa.backtest(stores["frame"], weights=_long(_selected_weights(stores)))
    assert report.benchmark is None


def test_a_settlement_on_a_frame_names_the_symbol_without_a_sidecar_warning(tmp_path):
    price_config = write_price_store(tmp_path, n_bars=N_BARS, delist_at={"BBB": 12})
    frame = _canonical_frame(price_config.zarr_file_path)
    bars = _bars()
    weights = pd.DataFrame(
        # Bought on bar 3; its last price is bar 11, so it is settled on bar 12.
        {"timestamp": [bars[3], bars[11]], "symbol": ["BBB", "BBB"], "weight": [1.0, 1.0]}
    )
    messages = []
    handler = logger.add(messages.append, level="WARNING")
    try:
        report = qa.backtest(frame, weights=weights)
    finally:
        logger.remove(handler)

    assert [r["symbol"] for r in report.raw.simulation.settlements] == ["BBB"]
    assert not [m for m in messages if "sidecar" in m]


# --------------------------------------------------------------------------- files


def _files_under(root: Path) -> list[str]:
    return sorted(
        str(Path(dirpath, name).relative_to(root))
        for dirpath, _, names in os.walk(root)
        for name in names
    )


def test_nothing_is_written_by_default(stores, monkeypatch):
    workdir = stores["root"] / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    before = _files_under(stores["root"])

    report = qa.backtest(
        stores["frame"], weights=_long(_selected_weights(stores)), benchmark=stores["benchmark_frame"]
    )

    assert report.raw.run_dir is None
    assert _files_under(stores["root"]) == before


def test_save_writes_the_run_directory_of_the_same_run(stores):
    report = qa.backtest(
        stores["frame"], weights=_long(_selected_weights(stores)), benchmark=stores["benchmark_frame"]
    )

    run_dir = report.save(stores["root"] / "saved")

    assert run_dir.parent == stores["root"] / "saved"
    assert sorted(p.name for p in run_dir.iterdir()) == RUN_DIR_ARTIFACTS
    run = BacktestRun.open(run_dir)
    assert run.kind == "run_weights"
    np.testing.assert_array_equal(run.equity()["value"].values, report.equity["value"].to_numpy())
    np.testing.assert_array_equal(run.weights()["weight"].values, report.raw.weights["weight"].values)
    assert report.raw.run_dir is None


def test_output_dir_writes_the_run_directory_at_once(stores):
    report = qa.backtest(
        stores["frame"],
        weights=_long(_selected_weights(stores)),
        output_dir=stores["root"] / "runs",
    )

    assert report.raw.run_dir.parent == stores["root"] / "runs"
    assert sorted(p.name for p in report.raw.run_dir.iterdir()) == RUN_DIR_ARTIFACTS


def test_plot_returns_the_report_figure(stores):
    report = qa.backtest(
        stores["frame"], weights=_long(_selected_weights(stores)), benchmark=stores["benchmark_frame"]
    )

    figure = report.plot()

    assert isinstance(figure, go.Figure)
    names = {trace.name for trace in figure.data}
    assert {"equity", "drawdown", "benchmark_equity", "benchmark_drawdown"} <= names
    assert "excess_return" not in names and "excess_drawdown" not in names


# --------------------------------------------------------------------------- libraries


def test_every_report_frame_round_trips_between_pandas_and_polars(stores):
    weights = _long(_selected_weights(stores))
    pandas_report = qa.backtest(
        stores["frame"], weights=weights, benchmark=stores["benchmark_frame"]
    )
    polars_report = qa.backtest(
        pl.from_pandas(stores["frame"]),
        weights=pl.from_pandas(weights),
        benchmark=pl.from_pandas(stores["benchmark_frame"]),
    )

    for name in ("equity", "returns", "weights", "orders", "trades", "benchmark"):
        pandas_frame = getattr(pandas_report, name)
        polars_frame = getattr(polars_report, name)
        assert isinstance(pandas_frame, pd.DataFrame), name
        assert isinstance(polars_frame, pl.DataFrame), name
        pd.testing.assert_frame_equal(polars_frame.to_pandas(), pandas_frame, check_dtype=False)
    assert list(pandas_report.weights.columns) == ["timestamp", "symbol", "weight"]
    assert list(pandas_report.orders.columns) == [
        "timestamp", "symbol", "size", "price", "fees", "side"
    ]
    assert list(pandas_report.trades.columns) == [
        "symbol", "entry_timestamp", "exit_timestamp", "pnl", "return", "status"
    ]


def test_an_xarray_panel_of_prices_gives_xarray_members(stores):
    weights = _long(_selected_weights(stores))
    expected = qa.backtest(stores["frame"], weights=weights)
    panel = stores["frame"].set_index(["timestamp", "symbol"]).to_xarray()

    report = qa.backtest(panel, weights=weights)

    assert isinstance(report.equity, xr.Dataset)
    np.testing.assert_array_equal(report.equity["value"].values, expected.equity["value"])
    xr.testing.assert_identical(report.weights, report.raw.weights[["weight"]])


def test_a_run_without_trades_gives_an_empty_trades_frame(stores):
    bars = _bars()
    hold = pd.DataFrame({"timestamp": [bars[0]], "symbol": ["AAA"], "weight": [0.0]})

    report = qa.backtest(stores["frame"], weights=hold)

    assert len(report.trades) == 0
    assert list(report.trades.columns) == [
        "symbol", "entry_timestamp", "exit_timestamp", "pnl", "return", "status"
    ]
    assert len(report.orders) == 0


def test_columns_maps_the_price_and_weight_frames(stores):
    weights = _long(_selected_weights(stores))
    expected = qa.backtest(stores["frame"], weights=weights)
    mapping = {"date": "timestamp", "ticker": "symbol", "Open": "open", "Close": "close"}
    reverse = {value: key for key, value in mapping.items()}

    report = qa.backtest(
        stores["frame"].rename(columns=reverse),
        weights=weights.rename(columns={"timestamp": "date", "symbol": "ticker"}),
        columns=mapping,
    )

    np.testing.assert_array_equal(report.equity["value"], expected.equity["value"])


def test_scores_through_the_api_match_the_library_rule_through_a_halt(stores):
    """Every symbol halts on bars 9-10 in turn with the ones after it; a held one
    stays locked at its weight there, in the API exactly as in the library."""
    path = stores["prices"].zarr_file_path
    store = xr.open_zarr(path).load()
    for name in store.data_vars:
        values = store[name].transpose("timestamp", "symbol").values.copy()
        values[9:11, :3] = np.nan
        store[name] = (("timestamp", "symbol"), values)
    store.to_zarr(path, mode="w")
    frame = _canonical_frame(path)
    weights = _selected_weights(stores, top_n=2, periods=3)
    scores = _scores().rename("score").to_dataframe().reset_index()

    report = qa.backtest(frame, scores=scores, top_n=2, rebalance_periods=3)

    xr.testing.assert_equal(report.raw.weights[["weight"]], weights)
    np.testing.assert_array_equal(
        report.equity["value"].to_numpy(), _library_run(stores, weights).simulation.value.values
    )
    at_9 = weights["weight"].isel(timestamp=9).values
    held_before = weights["weight"].isel(timestamp=6).values
    assert (held_before[:3] != 0).any(), "a halted symbol must have been held"
    assert (at_9[:3][held_before[:3] != 0] != 0).all()
