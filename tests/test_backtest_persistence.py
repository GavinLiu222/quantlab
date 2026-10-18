"""Locks for a backtest run's persistent record (phase 03.7, plan 09).

D-24: every run writes its own directory `output_dir/{class}_{timestamp}/`
holding config.json, weights.zarr, equity.zarr (value, returns), predictions.zarr,
settlements.json, metrics.json, report.html and run.json, which records the data
fingerprints and the market (read through `BacktestRun`, #133). An existing
directory is never overwritten, and every JSON artifact is strict JSON (NaN and
inf persisted as null, timestamps as ISO strings).

The run's results are read back through `BacktestRun` (#134). File names stay
literal only where the on-disk layout is the subject: the directory listing of
D-24, the strict-JSON check over every JSON file, the existing directory that is
never overwritten, and the recipe holding no run record (its `config.json` read
directly). The report's HTML is read with `BacktestRun.report()`.

D-27: the run records a data fingerprint for every dataset it read -- the price
dataset over the fill and valuation columns, each factor's dataset over the
columns it consumes, including the warm-up bars -- and a rebuild that supplies
`expected_fingerprint` warns (and still completes) when a digest, range or
count differs. Datasets get appended to, and Tiingo re-bases adjusted prices
after new dividends, so a re-run must detect that its data changed rather than
silently disagree with the stored run.

Why the fingerprint must be NaN-canonical: NaN has many bit patterns. Two reads
of one store, or two code paths producing "missing", can carry NaNs with
different payload bits that compare as the same missing value but hash
differently byte for byte. Without canonicalizing NaN (and -0.0 vs 0.0) before
hashing, an unchanged store could report a changed digest, and a warning that
fires on unchanged data is a warning people learn to ignore.

D-23 / D-21 / D-08: report.html is one self-contained page around a plotly
figure. It states its dates as TEXT -- the window and bar count, the training
window and the in-sample/out-of-sample ranges, every string byte-identical to
the same run's metrics.json -- carries the whole/in-sample/out-of-sample
metrics as an HTML table, and draws named traces on three rows of a shared time
axis: "equity" on x, "drawdown" on x2, and "monthly_return". Forced
delisting settlements are NOT drawn on the chart (quick 260916-hro) even for a
run that settled one, while settlements.json still records them in full; that pairing
is asserted in one place below. The in-sample range is shaded when the
window overlaps training, there is no benchmark trace, and the note that
short-side returns are optimistic because no borrow cost is modelled still
appears. metrics.json carries the same note and no benchmark key.

Phase 03.8 (D-02): a second note states that the trade metrics are the
position-level view -- one entry-to-flat round trip per symbol, so a partial
trim is not counted as its own closed trade -- and that `whole` carries
`Total Orders`, the number of fills. This file locks that at the artifact
level: the note verbatim in the page and in metrics.json, the top-level trade
metrics strict-JSON clean, and no nested `whole.positions` block left behind.
The proof that the reported view really IS the positions view (rather than a
copy of the exit-trades one) needs a run that trims without closing, and lives
in tests/test_backtest_engine.py.

The metric table is rendered from whatever keys the blocks carry at render
time, never from a list written into the report module: the metric set is
moving to vectorbt's own, and the report is written inside the staging
directory of a run, so a report that hardcoded metric names would raise on the
day that lands and take the entire run directory with it. That property is
proved at the leaf level in tests/test_backtest_report.py.

Tracking (the config's `tracker`, ADR 0015) is covered in
tests/test_backtest_tracking.py.

Everything is synthetic, CPU-only and offline.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun, Market
from tests.backtest_fixtures import (
    ADJUSTED_COLUMNS,
    RAW_COLUMNS,
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)
TRAIN_END_BAR = 24
MARKET = USEquityCrossectionSelectStockVectorBt.MARKET

# The shared overlapping run: window bars 20..40 overlap the training window
# (bars 0..22 fitted after the purge, plus a 2-bar label lookahead), and the symbol picked at the first
# rebalance delists at bar 23, so the run carries a real forced liquidation.
OVERLAP_START_BAR = 20
OVERLAP_END_BAR = 40
DELIST_BAR = OVERLAP_START_BAR + 3

D24_ARTIFACTS = [
    "config.json",
    "equity.zarr",
    "holdings.zarr",
    "metrics.json",
    "predictions.zarr",
    "report.html",
    "run.json",
    "settlements.json",
    "weights.zarr",
]


@pytest.fixture
def warnings_sink():
    """Loguru WARNING-and-above messages emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _strict_json(path: Path):
    def _reject(token):
        raise ValueError(f"non-standard JSON constant {token!r} in {path}")

    return json.loads(path.read_text(), parse_constant=_reject)


def _model_dates() -> dict:
    return dict(
        start_date=_day(BARS[0]),
        end_date=_day(BARS[29]),
        train_start=_day(BARS[0]),
        train_end=_day(BARS[TRAIN_END_BAR]),
        test_start=_day(BARS[TRAIN_END_BAR + 1]),
        test_end=_day(BARS[29]),
    )


def _trained_store(root: Path, **store_kwargs):
    """A seeded price store plus a checkpoint trained on bars 0..TRAIN_END_BAR."""
    dataset_config = write_price_store(root / "store", n_bars=N_BARS, **store_kwargs)
    checkpoint = train_checkpoint(
        make_model(root / "train", dataset_config, **_model_dates())
    )
    return dataset_config, checkpoint


def _backtester(
    root: Path,
    dataset_config,
    checkpoint: Path,
    *,
    tag: str,
    window_start_bar: int,
    window_end_bar: int,
    **overrides,
) -> USEquityCrossectionSelectStockVectorBt:
    """A fresh backtester (own datasets and model objects) in load mode."""
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / f"backtest_{tag}", dataset_config, **_model_dates()),
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(BARS[window_start_bar]),
        end_date=_day(BARS[window_end_bar]),
        output_dir=str(root / "runs"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
    )
    kwargs.update(overrides)
    return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(**kwargs))


def _picked_at(root: Path, start_bar: int) -> str:
    """The symbol the fixture model ranks first at `start_bar`.

    The store is seeded, so a probe store with the same seed tells which symbol
    the model picks (the fixture factor scores by past return on adjClose, the
    factor's own input column). Same recipe as
    tests/test_backtest_engine.py::test_end_to_end_delisting_run_records_the_liquidation.
    """
    probe = xr.open_zarr(
        write_price_store(root / "probe", n_bars=N_BARS).zarr_file_path
    ).load()
    close = probe["adjClose"].transpose("timestamp", "symbol").values
    past_return = close[start_bar] / close[start_bar - 1] - 1.0
    return SYMBOLS[int(np.argmax(past_return))]


@pytest.fixture(scope="module")
def overlap_run(tmp_path_factory):
    """One overlapping, liquidating run shared by the read-only artifact locks."""
    root = tmp_path_factory.mktemp("overlap")
    picked = _picked_at(root, OVERLAP_START_BAR)
    dataset_config, checkpoint = _trained_store(
        root, delist_at={picked: DELIST_BAR}
    )
    backtester = _backtester(
        root,
        dataset_config,
        checkpoint,
        tag="overlap",
        window_start_bar=OVERLAP_START_BAR,
        window_end_bar=OVERLAP_END_BAR,
    )
    config_before_run = backtester.get_config()
    result = backtester.run()
    return {
        "backtester": backtester,
        "result": result,
        "config_before_run": config_before_run,
        "dataset_config": dataset_config,
    }


# --------------------------------------------------------------------------
# D-24: the run directory
# --------------------------------------------------------------------------


def test_run_directory_holds_every_d24_artifact(overlap_run):
    run_dir = overlap_run["result"].run_dir
    assert sorted(p.name for p in run_dir.iterdir()) == D24_ARTIFACTS
    assert not list(run_dir.rglob("*.tmp"))


def test_the_weights_round_trip_identically(overlap_run):
    result = overlap_run["result"]
    persisted = BacktestRun.open(result.run_dir).weights()
    xr.testing.assert_identical(persisted, result.weights)


def test_the_equity_curve_has_value_and_returns_on_timestamp(overlap_run):
    result = overlap_run["result"]
    equity = BacktestRun.open(result.run_dir).equity()
    # Plus the attribution curves a model run records (test_backtest_attribution.py).
    assert set(equity.data_vars) == {"value", "returns", "universe_value", "gross_value", "group_value"}
    for name in ("value", "returns", "universe_value", "gross_value"):
        assert equity[name].dims == ("timestamp",)
    np.testing.assert_array_equal(
        equity["value"].values, result.simulation.value.values
    )
    np.testing.assert_array_equal(
        equity["returns"].values, result.simulation.returns.values
    )
    np.testing.assert_array_equal(
        equity["timestamp"].values, result.simulation.value.timestamp.values
    )


def test_every_json_artifact_is_strict_json(overlap_run):
    result = overlap_run["result"]
    run_dir = result.run_dir
    for name in ("config.json", "metrics.json", "settlements.json", "run.json"):
        _strict_json(run_dir / name)

    # Not vacuous: the run really settled a delisting, and each record carries
    # the pd.Timestamp fields that plain json.dump cannot serialize.
    settlements = _strict_json(run_dir / "settlements.json")
    assert result.simulation.settlements, "the fixture run must settle a delisting"
    assert len(settlements) == len(result.simulation.settlements)
    first = settlements[0]
    # 03.11-09: the record carries BOTH halves of a security's name -- the
    # human `symbol` (the period-correct ticker when a `.crsp_tickers.json`
    # sits beside the price store) and `axis_symbol`, the panel label itself.
    # This fixture store has NO sidecar, which is the control arm: the two are
    # equal and the file reads exactly as it did before the sidecar existed.
    assert set(first) == {
        "symbol",
        "axis_symbol",
        "delisting_timestamp",
        "settlement_timestamp",
        "price",
    }
    assert first["symbol"] == result.simulation.settlements[0]["symbol"]
    assert first["axis_symbol"] == first["symbol"]
    assert pd.Timestamp(first["settlement_timestamp"]) == pd.Timestamp(
        result.simulation.settlements[0]["settlement_timestamp"]
    )


def test_existing_run_directory_is_never_overwritten(tmp_path, monkeypatch):
    dataset_config, checkpoint = _trained_store(tmp_path)
    backtester = _backtester(
        tmp_path,
        dataset_config,
        checkpoint,
        tag="fixed",
        window_start_bar=30,
        window_end_bar=50,
    )
    existing = tmp_path / "runs" / "FixedRunName"
    existing.mkdir(parents=True)
    (existing / "metrics.json").write_text('{"kept": true}')
    monkeypatch.setattr(backtester, "_run_dir_name", lambda: "FixedRunName")

    with pytest.raises(RuntimeError, match="FixedRunName"):
        backtester.run()

    assert sorted(p.name for p in existing.iterdir()) == ["metrics.json"]
    assert (existing / "metrics.json").read_text() == '{"kept": true}'


def test_an_interrupted_persist_leaves_no_run_directory(tmp_path, monkeypatch):
    """Code review WR-08: a run directory exists only once every D-24 artifact is written.

    The old `_report_and_persist` created `output_dir/{class}_{ts}/` and wrote
    config.json (and the zarr stores) before metrics, report and run.json.
    An interruption there (Ctrl-C is simulated while report.html is written)
    left a directory with a valid config.json and no metrics. It looked
    finished, and a rebuild would "reproduce" it. After the interruption
    `output_dir` must hold nothing, not even the staging directory. Once the
    report works again, the same backtester writes one complete directory. Red
    on the old code: the half-written directory remains.
    """
    import quantlab.backtest.base as backtest_module

    dataset_config, checkpoint = _trained_store(tmp_path)
    backtester = _backtester(
        tmp_path,
        dataset_config,
        checkpoint,
        tag="interrupted",
        window_start_bar=30,
        window_end_bar=50,
    )
    real_report = backtest_module.write_backtest_report

    def _interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(backtest_module, "write_backtest_report", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        backtester.run()

    runs = tmp_path / "runs"
    assert (sorted(p.name for p in runs.iterdir()) if runs.exists() else []) == []

    monkeypatch.setattr(backtest_module, "write_backtest_report", real_report)
    result = backtester.run()

    assert sorted(p.name for p in runs.iterdir()) == [result.run_dir.name]
    assert sorted(p.name for p in result.run_dir.iterdir()) == D24_ARTIFACTS


# --------------------------------------------------------------------------
# D-27: the data fingerprint
# --------------------------------------------------------------------------


def _panel(values: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"adjClose": (["timestamp", "symbol"], values)},
        coords={
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "symbol": ["AAA", "BBB"],
        },
    )


def test_fingerprint_is_stable_and_nan_canonical(tmp_path):
    from quantlab.runs.record import _dataset_fingerprint as dataset_fingerprint

    zarr_path = write_price_store(tmp_path / "store", n_bars=20).zarr_file_path
    columns = [MARKET.fill_price_column, MARKET.valuation_price_column]
    first = dataset_fingerprint(xr.open_zarr(zarr_path), columns)
    second = dataset_fingerprint(xr.open_zarr(zarr_path), columns)
    assert first == second
    assert first["algorithm"] == "sha256"
    assert first["variables"] == sorted(columns)
    assert (first["n_timestamps"], first["n_symbols"]) == (20, len(SYMBOLS))

    payload_nan = np.frombuffer(
        np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64
    )[0]
    canonical = np.array([[1.0, np.nan], [0.0, 2.0]])
    other_bits = np.array([[1.0, payload_nan], [-0.0, 2.0]])
    # Control: the two arrays really differ byte for byte.
    assert canonical.tobytes() != other_bits.tobytes()
    assert (
        dataset_fingerprint(_panel(canonical), ["adjClose"])["digest"]
        == dataset_fingerprint(_panel(other_bits), ["adjClose"])["digest"]
    )

    changed = canonical.copy()
    changed[1, 1] = 2.5
    assert (
        dataset_fingerprint(_panel(changed), ["adjClose"])["digest"]
        != dataset_fingerprint(_panel(canonical), ["adjClose"])["digest"]
    )

    with pytest.raises(ValueError, match="notAColumn"):
        dataset_fingerprint(_panel(canonical), ["adjClose", "notAColumn"])


def test_the_run_records_fingerprints_of_the_price_and_factor_datasets(overlap_run):
    """The run records what it read, by component path, one entry per request."""
    result = overlap_run["result"]
    from quantlab.runs.record import _dataset_fingerprint as dataset_fingerprint

    fingerprints = BacktestRun.open(result.run_dir).data_fingerprint
    # Load mode: no training data, and the label dataset is read for training only.
    assert set(fingerprints) == {"price_dataset", "model.factors.0.dataset"}

    entry_keys = {
        "request",
        "algorithm",
        "digest",
        "variables",
        "variable_digests",
        "variable_dtypes",
        "start",
        "end",
        "n_timestamps",
        "n_symbols",
    }
    for entries in fingerprints.values():
        for entry in entries:
            assert set(entry) == entry_keys
            assert entry["algorithm"] == "sha256"

    columns = sorted([MARKET.fill_price_column, MARKET.valuation_price_column])
    window = [
        entry for entry in fingerprints["price_dataset"]
        if pd.Timestamp(entry["request"]["start"]) == pd.Timestamp(BARS[OVERLAP_START_BAR])
    ]
    (price,) = window
    # Only the fill and valuation columns are read for the window.
    assert price["request"]["variables"] == columns == price["variables"]
    assert price["n_timestamps"] == OVERLAP_END_BAR - OVERLAP_START_BAR + 1
    store = xr.open_zarr(overlap_run["dataset_config"].zarr_file_path).sel(
        timestamp=slice(_day(BARS[OVERLAP_START_BAR]), _day(BARS[OVERLAP_END_BAR]))
    )
    assert price["digest"] == dataset_fingerprint(store, columns)["digest"]
    assert price["n_symbols"] == len(SYMBOLS)
    # The delisting check reads the valuation column after the window, of the
    # symbols that stop inside it only.
    (after,) = [
        entry for entry in fingerprints["price_dataset"]
        if entry["variables"] == [MARKET.valuation_price_column]
        and pd.Timestamp(entry["start"]) > pd.Timestamp(price["end"])
    ]
    assert 0 < after["n_symbols"] < len(SYMBOLS)

    (factor,) = fingerprints["model.factors.0.dataset"]
    assert factor["n_symbols"] == len(SYMBOLS)
    # A Polars factor consumes the whole lazyframe: every data variable counts.
    assert factor["variables"] == sorted(ADJUSTED_COLUMNS + RAW_COLUMNS)
    # The factor range includes the warm-up bars before the window start (D-27).
    assert pd.Timestamp(factor["start"]) < pd.Timestamp(price["start"])
    assert pd.Timestamp(factor["end"]) == pd.Timestamp(price["end"])


def test_one_dataset_read_by_two_consumers_is_recorded_once(tmp_path):
    """The price dataset is also the factor's dataset: one key, the first path."""
    dataset_config, checkpoint = _trained_store(tmp_path)
    backtester = _backtester(
        tmp_path, dataset_config, checkpoint, tag="shared", window_start_bar=30,
        window_end_bar=50,
    )
    shared = backtester.config.model.config.factors[0].config.dataset
    backtester.config.price_dataset = shared  # one object, two consumers

    backtester.run()

    assert set(backtester.data_fingerprint) == {"price_dataset"}
    variables = [e["request"]["variables"] for e in backtester.data_fingerprint["price_dataset"]]
    assert None in variables  # the factor's whole-frame read
    assert sorted([MARKET.fill_price_column, MARKET.valuation_price_column]) in variables


def test_the_run_records_the_market_price_columns(overlap_run):
    """#106: an executor learns the fill and valuation columns from the run.

    quantlab-ibkr reads a run directory without importing the backtester
    class (which loads vectorbt), so the columns the run filled and valued at
    are recorded in its run.json, read as `BacktestRun.market` (#133).
    """
    run = BacktestRun.open(overlap_run["result"].run_dir)

    assert run.market == Market(fill_price_column="adjOpen", valuation_price_column="adjClose")
    assert "market" not in _strict_json(overlap_run["result"].run_dir / "config.json")


def test_the_config_stays_the_recipe_after_a_run(overlap_run):
    """#133: the data fingerprint is a record of the run, never part of the config."""
    config = overlap_run["backtester"].get_config()
    assert config == overlap_run["config_before_run"]
    assert "data_fingerprint" not in config
    assert "data_fingerprint" not in _strict_json(overlap_run["result"].run_dir / "config.json")
    assert BacktestRun.open(overlap_run["result"].run_dir).data_fingerprint


def _fingerprint_warnings(messages: list[str]) -> list[str]:
    # The warning's own words, not "fingerprint": a test's temporary
    # directory may be named after it and show in other warnings' paths.
    return [m for m in messages if "data fingerprint mismatch" in m]


def test_matching_expected_fingerprint_logs_no_warning(tmp_path, warnings_sink):
    dataset_config, checkpoint = _trained_store(tmp_path)
    first = _backtester(
        tmp_path, dataset_config, checkpoint, tag="a", window_start_bar=30, window_end_bar=50
    ).run()
    expected = BacktestRun.open(first.run_dir).data_fingerprint

    rebuild = _backtester(
        tmp_path, dataset_config, checkpoint, tag="b", window_start_bar=30, window_end_bar=50
    )
    rebuild.expected_fingerprint = expected
    warnings_sink.clear()
    rebuild.run()
    assert _fingerprint_warnings(warnings_sink) == []


def test_changed_store_logs_a_fingerprint_warning_and_completes(tmp_path, warnings_sink):
    dataset_config, checkpoint = _trained_store(tmp_path)
    first = _backtester(
        tmp_path, dataset_config, checkpoint, tag="a", window_start_bar=30, window_end_bar=50
    ).run()
    expected = BacktestRun.open(first.run_dir).data_fingerprint

    # A retroactive re-base: one adjusted close inside the window changes.
    store = xr.open_zarr(dataset_config.zarr_file_path).load()
    for variable in store.variables.values():
        variable.encoding = {}
    column = MARKET.valuation_price_column
    store[column][35, 0] = float(store[column][35, 0]) * 1.01
    store.to_zarr(dataset_config.zarr_file_path, mode="w")

    rebuild = _backtester(
        tmp_path, dataset_config, checkpoint, tag="b", window_start_bar=30, window_end_bar=50
    )
    rebuild.expected_fingerprint = expected
    warnings_sink.clear()
    result = rebuild.run()

    assert result is not None and result.run_dir.exists()
    price_warnings = [
        m for m in _fingerprint_warnings(warnings_sink) if "'price_dataset'" in m
    ]
    # The window read and the rule's price-history read both cover bar 35.
    assert price_warnings, warnings_sink
    assert all("digest differs" in m for m in price_warnings)


# --------------------------------------------------------------------------
# D-27 / code review WR-05: fingerprint what predictions and training really read
# --------------------------------------------------------------------------


def _read_strategy_backtester(root, dataset_config, checkpoint, factor_store, *, tag):
    """Load mode whose model reads its features from a saved factor STORE."""
    from quantlab.model.config import ModelConfig
    from quantlab.factor.config import PolarsFactorConfig
    from tests.backtest_fixtures import (
        FirstFeatureHead,
        ForwardReturnLabel,
        PastReturnFactor,
    )

    factor = PastReturnFactor(
        PolarsFactorConfig(
            warmup_bars=5,
            dataset=make_stock_dataset(dataset_config),
            file_path=str(factor_store),
            kwargs={"n": 1},
        )
    )
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            warmup_bars=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": 1},
        )
    )
    model = FirstFeatureHead(
        ModelConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(root / f"backtest_{tag}" / "models"),
            factor_data_strategy="read",
            label_data_strategy="cal",
            val_size=0.0,
            **_model_dates(),
        )
    )
    return _backtester(
        root, dataset_config, checkpoint, tag=tag,
        window_start_bar=30, window_end_bar=50, model=model,
    )


def _shift_value(zarr_path, variable: str, timestamp, symbol: str, shift: float) -> None:
    """Rewrite one value of a Zarr store, as a factor recompute or a re-base would."""
    ds = xr.open_zarr(zarr_path).load()
    for item in ds.variables.values():
        item.encoding = {}
    point = dict(timestamp=timestamp, symbol=symbol)
    before = float(ds[variable].loc[point])
    assert np.isfinite(before), (variable, point)
    ds[variable].loc[point] = before + shift
    ds.to_zarr(zarr_path, mode="w")


def test_read_strategy_fingerprints_the_factor_store_predictions_came_from(
    tmp_path, warnings_sink
):
    """Under `factor_data_strategy="read"` the factor STORE is fingerprinted.

    Read-strategy features come from the saved factor store, not from the raw
    dataset, so a recomputed or edited factor store changes the predictions
    and weights. The store read is recorded under the factor's own component
    path, `model.factors.0`, and only it warns after one stored value changes.
    """
    from quantlab.factor.config import PolarsFactorConfig
    from tests.backtest_fixtures import PastReturnFactor

    dataset_config, checkpoint = _trained_store(tmp_path)
    factor_store = tmp_path / "factors" / "past_ret.zarr"
    PastReturnFactor(
        PolarsFactorConfig(
            warmup_bars=5,
            dataset=make_stock_dataset(dataset_config),
            file_path=str(factor_store),
            kwargs={"n": 1},
        )
    ).build(_day(BARS[0]), _day(BARS[-1]))

    first = _read_strategy_backtester(
        tmp_path, dataset_config, checkpoint, factor_store, tag="a"
    ).run()
    expected = BacktestRun.open(first.run_dir).data_fingerprint
    assert set(expected) == {"price_dataset", "model.factors.0"}, sorted(expected)

    _shift_value(factor_store, "past_ret_1", BARS[35], SYMBOLS[0], shift=0.01)
    rebuild = _read_strategy_backtester(
        tmp_path, dataset_config, checkpoint, factor_store, tag="b"
    )
    rebuild.expected_fingerprint = expected
    warnings_sink.clear()
    result = rebuild.run()

    assert result.run_dir.exists()
    changed = _fingerprint_warnings(warnings_sink)
    assert len(changed) == 1, warnings_sink
    assert "'model.factors.0'" in changed[0] and "digest differs" in changed[0]


def test_train_mode_records_no_training_data(tmp_path, warnings_sink):
    """A backtest records only what its window reads; training is the trained unit's.

    The model trains on bars 0..29 and the backtest window 30..50 warms up
    from bar 25. A re-base at bar 10 lies inside training only, so the
    backtest's own record does not change and the rebuild is silent about it
    (the trained unit's own record is what notices it).
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)

    def _train_mode(tag: str) -> USEquityCrossectionSelectStockVectorBt:
        return USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=make_stock_dataset(dataset_config),
                model=make_model(
                    tmp_path / f"backtest_{tag}", dataset_config, **_model_dates()
                ),
                model_mode="train",
                start_date=_day(BARS[30]),
                end_date=_day(BARS[50]),
                output_dir=str(tmp_path / "runs"),
                rebalance_periods=5,
                constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
            )
        )

    first = _train_mode("a").run()
    expected = BacktestRun.open(first.run_dir).data_fingerprint
    assert set(expected) == {"price_dataset", "model.factors.0.dataset"}, sorted(expected)
    (factor,) = expected["model.factors.0.dataset"]
    assert pd.Timestamp(factor["start"]) >= pd.Timestamp(BARS[25])

    _shift_value(
        dataset_config.zarr_file_path,
        MARKET.valuation_price_column,
        BARS[10],
        SYMBOLS[0],
        shift=0.5,
    )
    rebuild = _train_mode("b")
    rebuild.expected_fingerprint = expected
    warnings_sink.clear()
    rebuild.run()

    assert _fingerprint_warnings(warnings_sink) == []


# --------------------------------------------------------------------------
# D-23 / D-21 / D-08: the report and the short-side note
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def disjoint_run(tmp_path_factory):
    """One run whose window (bars 30..50) lies after the training window."""
    root = tmp_path_factory.mktemp("disjoint")
    dataset_config, checkpoint = _trained_store(root)
    backtester = _backtester(
        root,
        dataset_config,
        checkpoint,
        tag="disjoint",
        window_start_bar=30,
        window_end_bar=50,
    )
    result = backtester.run()
    return {"backtester": backtester, "result": result}


def _report_html(run: dict) -> str:
    return BacktestRun.open(run["result"].run_dir).report()


def _report_traces(html: str) -> dict[str, dict]:
    """The trace list plotly embeds as `Plotly.newPlot(id, [traces], layout, ...)`, by name.

    plotly 5.x serializes numpy arrays as plain JSON lists, so the persisted y
    values can be decoded straight from the page.
    """
    start = html.index("[", html.index("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    return {trace["name"]: trace for trace in traces}


def test_report_has_equity_and_drawdown_and_shades_the_in_sample_range(overlap_run):
    html = _report_html(overlap_run)
    metrics = BacktestRun.open(overlap_run["result"].run_dir).metrics()
    in_sample_range = metrics["in_sample_range"]
    assert in_sample_range is not None, "the fixture window must overlap training"

    assert '"name":"equity"' in html
    assert '"name":"drawdown"' in html
    # The curves are the persisted run's values: drawdown = value / running max - 1.
    traces = _report_traces(html)
    # Quick 260915-sxx grew the page from two panels to three. Everything this
    # test proves is unchanged: the equity y values are still the persisted
    # ones, drawdown is still value over running max minus one, the axes are
    # still x/x2, the band is still the persisted in-sample range, and the
    # notes still appear.
    # Quick 260915-v6i added the deepest-drawdown triangles. This fixture run
    # draws down (asserted below), so both markers are always present here.
    # Quick 260916-hro removed the "liquidation" markers from the CHART. This
    # fixture run really does liquidate, so their absence here is the point;
    # the surviving JSON half of that split is asserted in its own test below.
    assert set(traces) == {
        "equity",
        "drawdown",
        "monthly_return",
        "deepest_drawdown_valley",
        "deepest_drawdown_end",
    }
    value = BacktestRun.open(overlap_run["result"].run_dir).equity()["value"].values
    np.testing.assert_allclose(traces["equity"]["y"], value, rtol=1e-12)
    expected_drawdown = value / np.maximum.accumulate(value) - 1.0
    assert expected_drawdown.min() < 0, "the fixture run must draw down"
    np.testing.assert_allclose(traces["drawdown"]["y"], expected_drawdown, atol=1e-12)
    assert traces["drawdown"]["xaxis"] == "x2" and traces["equity"]["xaxis"] == "x"
    assert '"type":"rect"' in html
    # The shaded band is the persisted in-sample range, not an assumed one.
    assert f'"x0":"{in_sample_range[0]}"' in html
    assert f'"x1":"{in_sample_range[1]}"' in html
    notes = overlap_run["backtester"]._report_notes()
    assert notes
    for note in notes:
        assert note in html


def _second_figure_traces(html: str) -> dict[str, dict]:
    """Traces of the SECOND `Plotly.newPlot(` on the page, by name.

    `_report_traces` reads the FIRST one by design -- that is what keeps the
    three-row figure's exact-trace-set lock above meaningful. The monthly
    heatmap (03.8 D-04) lives in its own div after it, so without this parser
    it would be invisible to every persisted-report lock. Mirrors the parser
    in tests/test_backtest_report.py; the two test files carry their own
    parsers by convention rather than importing across test modules.
    """
    first = html.index("Plotly.newPlot(")
    second = html.index("Plotly.newPlot(", first + 1)
    start = html.index("[", second)
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    return {trace["name"]: trace for trace in traces}


def test_report_carries_the_monthly_heatmap_in_a_second_div(overlap_run):
    """03.8 D-04: a real run's page carries the year-by-month heatmap.

    It is a SECOND plotly div, not a trace of the main figure (whose exact
    set is locked above), with 12 month columns and one row per year the run
    spans. It lives INSIDE report.html: the run directory still holds exactly
    the D-24 artifacts, so no sibling file appeared for it.
    """
    run_dir = overlap_run["result"].run_dir
    html = _report_html(overlap_run)

    # The Performance figure, then the heatmap; the other tabs' figures follow.
    assert html.count("Plotly.newPlot(") >= 2
    heatmap = _second_figure_traces(html)["monthly_return_heatmap"]
    assert heatmap["type"] == "heatmap"
    assert heatmap["x"] == [f"{month:02d}" for month in range(1, 13)]

    timestamps = pd.DatetimeIndex(BacktestRun.open(run_dir).equity()["timestamp"].values)
    years = sorted({str(year) for year in timestamps.year})
    assert heatmap["y"] == years
    assert len(heatmap["z"]) == len(years)
    assert all(len(row) == 12 for row in heatmap["z"])
    # The months the bar row shows are exactly the heatmap's non-null cells.
    bars = _report_traces(html)["monthly_return"]
    bar_months = {(label[:4], label[5:7]) for label in bars["x"]}
    cell_months = {
        (heatmap["y"][row], heatmap["x"][col])
        for row, cells in enumerate(heatmap["z"])
        for col, value in enumerate(cells)
        if value is not None
    }
    assert cell_months == bar_months

    assert sorted(p.name for p in run_dir.iterdir()) == D24_ARTIFACTS


def _report_rows(html: str) -> dict[str, list[str]]:
    """Every metric row of the page as `label -> [cell, ...]`."""
    rows = re.findall(r'<tr><th title="[^"]*">([^<]+)</th>((?:<td[^>]*>[^<]*</td>)+)</tr>', html)
    return {name: re.findall(r"<td[^>]*>([^<]*)</td>", cells) for name, cells in rows}


def test_report_carries_the_out_of_sample_numbers_and_the_axis_toggle(overlap_run):
    """With an in-sample part the headline table is the out-of-sample slice.

    The persisted numbers reached the page: each catalogued percent and ratio of
    `metrics.json`'s `out_of_sample` block is the headline table's cell, as the
    page formats it (two decimals). The log button is what makes a curve that
    compounded by orders of magnitude readable.
    """
    html = _report_html(overlap_run)
    metrics = BacktestRun.open(overlap_run["result"].run_dir).metrics()
    oos = metrics["out_of_sample"]

    assert "<h2>Strategy (out-of-sample)</h2>" in html
    start = html.index("<h2>Strategy (out-of-sample)</h2>")
    rows = _report_rows(html[start : html.index("</table>", start)])
    for key, label, fmt in (
        ("Total Return [%]", "Total return", "{:,.2f}%"),
        ("Annualized Volatility [%]", "Annualised volatility", "{:,.2f}%"),
        ("Sharpe Ratio", "Sharpe ratio", "{:,.2f}"),
        ("Calmar Ratio", "Calmar ratio", "{:,.2f}"),
    ):
        assert rows[label][0] == fmt.format(oos[key]), (label, rows[label], oos[key])
    assert rows["Max drawdown"][0] == "{:,.2f}%".format(-abs(oos["Max Drawdown [%]"]))
    assert "Total turnover" in _report_rows(html)
    assert '"yaxis.type":"log"' in html and '"yaxis.type":"linear"' in html


def test_report_split_table_carries_both_slices(overlap_run):
    """A run with an in-sample part shows every key row for both slices."""
    html = _report_html(overlap_run)
    metrics = BacktestRun.open(overlap_run["result"].run_dir).metrics()

    assert "<h2>In-sample vs out-of-sample</h2>" in html
    start = html.index("<h2>In-sample vs out-of-sample</h2>")
    block = html[start : html.index("</table>", start)]
    table = _report_rows(block)
    for label, key in (("Total return", "Total Return [%]"), ("Sharpe ratio", "Sharpe Ratio"),
                       ("Max drawdown", "Max Drawdown [%]")):
        in_sample, out_of_sample, _difference, _whole = table[label]
        assert in_sample != "—" and out_of_sample != "—", (label, table[label])
    assert table["Total return"][0] == "{:,.2f}%".format(metrics["in_sample"]["Total Return [%]"])


def test_report_states_the_window_and_split_dates_as_text(overlap_run):
    """Quick 260915-sxx: the page states its dates in words, not only as a band.

    Before this task the report carried no date anywhere: the reader saw a
    shaded region and had to open metrics.json separately to learn which
    window it covered, where training ended, and which bars were in-sample.
    Every date on the page is the string the run's OWN metrics.json carries --
    the summary reads the persisted bar labels instead of reformatting
    timestamps, so the page and the metrics cannot drift apart.
    """
    html = _report_html(overlap_run)
    metrics = BacktestRun.open(overlap_run["result"].run_dir).metrics()
    value = BacktestRun.open(overlap_run["result"].run_dir).equity()["value"]

    n_bars = value.sizes["timestamp"]
    first = _day(value.timestamp.values[0])
    last = _day(value.timestamp.values[-1])
    assert f"{first} .. {last} ({n_bars} bars)" in html

    training_window = metrics["training_window"]
    assert training_window is not None, "the fixture model must record train dates"
    assert f"{training_window[0]} .. {training_window[1]}" in html

    in_sample_range = metrics["in_sample_range"]
    assert in_sample_range is not None, "the fixture window must overlap training"
    assert f"{in_sample_range[0]} .. {in_sample_range[1]}" in html

    out_of_sample_ranges = metrics["out_of_sample_ranges"]
    assert out_of_sample_ranges, "the fixture window must have out-of-sample bars"
    for start, end in out_of_sample_ranges:
        assert f"{start} .. {end}" in html

    # The setup those numbers were produced under is on the page too.
    config = overlap_run["backtester"].config
    assert f"<td>{config.model_mode}</td>" in html
    assert "TopNConstructor(" in html
    assert config.constructor.config.direction in html


def test_report_without_in_sample_overlap_has_no_shaded_range(disjoint_run):
    html = _report_html(disjoint_run)
    assert disjoint_run["result"].metrics["in_sample_range"] is None
    # Control: the page really carries the curves.
    assert '"name":"equity"' in html
    assert '"type":"rect"' not in html


def test_report_marks_the_deepest_drawdown_on_the_persisted_equity_curve(overlap_run):
    """Quick 260916-hro: the triangles land on the curve the run persisted.

    The leaf tests prove the markers are drawn where they are told; this
    proves a REAL run tells them the right place -- both endpoints are values
    of `equity.zarr`, not of some re-derived curve.

    The up triangle is additionally checked against an INDEPENDENTLY derived
    valley: the drawdown curve is recomputed here straight from `equity.zarr`
    and its argmin taken, without asking the engine anything. That is what
    goes red if the marker ever slides back to the bar the drawdown started.
    """
    html = _report_html(overlap_run)
    traces = _report_traces(html)
    equity = BacktestRun.open(overlap_run["result"].run_dir).equity()
    value = equity["value"].values

    valley = traces["deepest_drawdown_valley"]
    end = traces["deepest_drawdown_end"]
    assert valley["marker"]["symbol"] == "triangle-up"
    assert end["marker"]["symbol"] == "triangle-down"
    for marker in (valley, end):
        assert len(marker["y"]) == 1
        assert np.isclose(marker["y"][0], value, rtol=1e-12).any(), marker["y"]

    # The deepest bar of the deepest drawdown is the global argmin of
    # `value / running max - 1`, so it can be recovered from the persisted
    # equity alone.
    drawdown = value / np.maximum.accumulate(value) - 1.0
    assert drawdown.min() < 0, "the fixture run must draw down"
    valley_bar = int(np.argmin(drawdown))
    assert str(valley["x"][0])[:10] == _day(equity["timestamp"].values[valley_bar])
    # Non-vacuity: the valley is not the first bar of the window, so this is a
    # real localisation rather than a marker parked at the start by accident.
    assert valley_bar > 0

    # The span is stated in trading days, never as a calendar timedelta: a
    # `Timedelta` repr (`7 days 00:00:00`) would add a `days` that is not
    # part of `trading days`.
    hover = end["hovertemplate"]
    assert "trading days" in hover
    assert hover.count("days") == hover.count("trading days"), hover


def test_the_page_states_the_deepest_drawdown_span_in_words(overlap_run):
    """The picture and the text agree: same bar labels, same units."""
    html = _report_html(overlap_run)
    row = re.search(
        r"<tr><th>Deepest drawdown \(valley to recovery\)</th><td>([^<]*)</td></tr>",
        html,
    )
    assert row is not None, "the dates-and-setup block must state the span"

    text = row.group(1)
    assert "trading days" in text
    assert text.count("days") == text.count("trading days"), text

    # The same two bars the triangles sit on.
    traces = _report_traces(html)
    for name in ("deepest_drawdown_valley", "deepest_drawdown_end"):
        label = str(traces[name]["x"][0])[:10]
        assert label in text, (name, label, text)


def test_the_settling_run_draws_no_chart_markers_and_keeps_the_json(overlap_run):
    """D-02 in ONE test: the picture loses the markers, the data survives.

    Split across two tests, either half could be deleted while the other kept
    passing -- and that is precisely the regression worth guarding, because
    the chart trace and the records file were fed from the SAME
    `simulation.settlements` and only the chart was meant to lose it.
    """
    result = overlap_run["result"]
    traces = _report_traces(_report_html(overlap_run))

    # Non-vacuity: this run really did settle a delisting, so an absent
    # marker cannot be explained away by there being nothing to draw.
    assert result.simulation.settlements, "the fixture run must settle a delisting"
    assert "liquidation" not in traces

    # ...and the artifact still carries every record, field for field.
    persisted = BacktestRun.open(result.run_dir).settlements()
    assert len(persisted) == len(result.simulation.settlements)
    for stored, live in zip(persisted, result.simulation.settlements):
        assert set(stored) == {
            "symbol",
            "axis_symbol",
            "delisting_timestamp",
            "settlement_timestamp",
            "price",
        }
        assert stored["symbol"] == live["symbol"]
        for field in ("settlement_timestamp", "delisting_timestamp"):
            assert pd.Timestamp(stored[field]) == pd.Timestamp(live[field]), field
        assert stored["price"] == pytest.approx(live["price"], rel=1e-12)

    # The run directory is untouched: still exactly the seven D-24 entries.
    assert sorted(p.name for p in result.run_dir.iterdir()) == D24_ARTIFACTS


def test_report_and_metrics_carry_no_benchmark(overlap_run):
    html = _report_html(overlap_run)
    assert '"name":"equity"' in html
    assert '"name":"benchmark"' not in html
    assert "benchmark" not in BacktestRun.open(overlap_run["result"].run_dir).metrics()
    assert "benchmark" not in overlap_run["result"].metrics


def test_metrics_json_carries_the_short_side_note(overlap_run):
    metrics = BacktestRun.open(overlap_run["result"].run_dir).metrics()
    assert "notes" in metrics, sorted(metrics)
    notes = metrics["notes"]
    assert notes == overlap_run["backtester"]._report_notes()
    text = " ".join(notes).lower()
    assert "short" in text and "borrow" in text and "optimistic" in text


# --------------------------------------------------------------------------
# Phase 03.8 (D-02): the one-trade-view note and the absent nested block
# --------------------------------------------------------------------------

#: The two TOP-LEVEL trade metrics vectorbt reports as a duration;
#: `to_jsonable` renders a Timedelta as a string and NaT as null, so these two
#: are the one legitimately non-numeric pair among the trade metrics.
POSITION_DURATION_KEYS = {"Avg Winning Trade Duration", "Avg Losing Trade Duration"}

#: vectorbt's display names for the trade-derived metrics, written out here
#: rather than read off the engine: a test that asked the implementation what
#: it should contain would agree with any answer. This mirrors the same literal
#: in tests/test_backtest_engine.py deliberately -- each file states its own
#: expectation instead of importing the other's.
TRADE_METRIC_NAMES = (
    "Total Trades",
    "Total Closed Trades",
    "Total Open Trades",
    "Open Trade PnL",
    "Win Rate [%]",
    "Best Trade [%]",
    "Worst Trade [%]",
    "Avg Winning Trade [%]",
    "Avg Losing Trade [%]",
    "Avg Winning Trade Duration",
    "Avg Losing Trade Duration",
    "Profit Factor",
    "Expectancy",
)


def test_report_and_metrics_carry_the_one_trade_view_note(overlap_run):
    """Both notes reach both artifacts, and the new one survives HTML escaping.

    Every note is rendered through `html.escape`, so a note containing any of
    the five characters escaping rewrites would appear in the page in escaped
    form and NOT verbatim. That is asserted as a property of the note text
    rather than left to whoever edits it next.
    """
    html = _report_html(overlap_run)
    metrics = BacktestRun.open(overlap_run["result"].run_dir).metrics()
    notes = overlap_run["backtester"]._report_notes()

    assert len(notes) >= 2, notes
    assert metrics["notes"] == notes
    for note in notes:
        assert note in html, note
        assert not set(note) & set("<>&\"'"), note

    text = " ".join(notes).lower()
    # The base short-side note survives alongside the new one.
    assert "short" in text and "borrow" in text and "optimistic" in text
    # And the new note says which view the trade metrics are, and names the
    # execution-activity count that replaced the lot-level set.
    for mark in ("position level", "round trip", "partial trim", "total orders"):
        assert mark in text, mark


def test_metrics_json_carries_one_trade_view_and_no_nested_positions_block(
    overlap_run,
):
    """The top-level trade metrics strict-parse, and the nested block is gone.

    `test_report_shows_the_positions_rows_beside_the_lot_level_rows` was
    deleted alongside the nested block: there are no `positions.`-prefixed rows
    on the page any more, and the surviving metric-table test already covers
    the top-level rows these numbers are now rendered as.
    """
    metrics = BacktestRun.open(overlap_run["result"].run_dir).metrics()
    whole = metrics["whole"]

    assert "positions" not in whole, sorted(whole)
    # The companion change (CONTEXT item 3): dropping the lot-level set would
    # otherwise leave the whole-window block with no execution-activity count.
    assert isinstance(whole["Total Orders"], int) and not isinstance(
        whole["Total Orders"], bool
    )
    assert whole["Total Orders"] > 0

    for key in TRADE_METRIC_NAMES:
        assert key in whole, key
        value = whole[key]
        if key in POSITION_DURATION_KEYS:
            assert value is None or isinstance(value, str), (key, value)
            continue
        assert value is None or (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ), (key, value)
        if isinstance(value, float):
            assert np.isfinite(value), key
