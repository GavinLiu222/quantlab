"""A backtest records the data it reads at the dataset seam.

The backtester opens a ``DataRecorder`` around each run, each ``run_cv`` fold
and the stitched pass. Whatever a component reads is recorded, keyed by its
component path, without the backtester enumerating it: the price columns, the
delisting look-ahead, the rule's price history before the window, factor
inputs and a ``MergedDataset``'s leaves. Factor inputs are read once by the
run, and once more when the recorder hashes them; nothing else re-reads them.

Everything is synthetic, CPU-only and offline.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.stock import StockDataset
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.record import _active_recorder as active_recorder
from tests.backtest_fixtures import (
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
N_BARS = 90
WINDOW = (40, 85)
LOOKBACK = 20
PRICE_COLUMNS = ["adjClose", "adjOpen"]


def _day(ts) -> str:
    """Return ``ts`` as an ISO day."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _model_dates(bars) -> dict:
    """Training on bars 0..34, testing 35..39: the window starts after both."""
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[39]),
        train_start=_day(bars[0]),
        train_end=_day(bars[34]),
        test_start=_day(bars[35]),
        test_end=_day(bars[39]),
    )


def _backtester(price_dataset, model=None, constructor=None, **overrides):
    """A US-equity backtester over bars ``WINDOW``, kept in memory by default."""
    bars = price_dataset.calendar("2000-01-01", "2100-01-01")
    config = dict(
        price_dataset=price_dataset,
        model=model,
        model_mode=None if model is None else "train",
        start_date=_day(bars[WINDOW[0]]),
        end_date=_day(bars[WINDOW[1]]),
        output_dir=None,
        rebalance_periods=5,
        constructor=constructor or TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
    )
    config.update(overrides)
    return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(**config))


def test_the_rules_price_history_before_the_window_is_recorded(tmp_path):
    """A mean-variance rule reads ``history_bars`` of prices before the window."""
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS, seed=11)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    optimizer = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="fwd_ret_1",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
            risk_aversion=5.0,
        )
    )
    backtester = _backtester(
        make_stock_dataset(dataset_config),
        make_model(tmp_path / "model", dataset_config, **_model_dates(bars)),
        optimizer,
    )

    backtester.run()  # kept in memory: output_dir=None

    entries = backtester.data_fingerprint["price_dataset"]
    window_start = pd.Timestamp(bars[WINDOW[0]])
    history = [e for e in entries if pd.Timestamp(e["start"]) < window_start]
    assert history, entries
    assert all(e["variables"] == PRICE_COLUMNS for e in history)
    first = pd.Timestamp(bars[WINDOW[0] - (optimizer.declared_inputs().history_bars - 1)])
    assert min(pd.Timestamp(e["start"]) for e in history) == first


def test_run_weights_records_each_leaf_of_a_merged_price_dataset(tmp_path):
    """A merge records nothing itself; its inputs are keyed under its path."""
    first = make_stock_dataset(write_price_store(tmp_path / "a", symbols=SYMBOLS[:3], n_bars=N_BARS))
    second = make_stock_dataset(
        write_price_store(tmp_path / "b", symbols=SYMBOLS[3:], n_bars=N_BARS, seed=1)
    )
    merged = MergedDataset([first, second])
    backtester = _backtester(merged)
    every = merged.calendar("2000-01-01", "2100-01-01")
    bars = every[WINDOW[0] : WINDOW[1] + 1]
    weights = xr.Dataset(
        {"weight": (("timestamp", "symbol"), np.full((len(bars), len(SYMBOLS)), 0.1))},
        coords={"timestamp": bars, "symbol": SYMBOLS},
    )

    backtester.run_weights(weights)

    record = backtester.data_fingerprint
    assert set(record) == {"price_dataset.datasets.0", "price_dataset.datasets.1"}
    for key, held in (("price_dataset.datasets.0", 3), ("price_dataset.datasets.1", 3)):
        window = [e for e in record[key] if e["request"]["variables"] == PRICE_COLUMNS]
        assert window and all(e["n_symbols"] == held for e in window)


def test_factor_inputs_are_read_once_per_run(tmp_path, monkeypatch):
    """The factor reads its inputs once inside the run; the recorder once at close."""
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **_model_dates(bars)))
    backtester = _backtester(
        make_stock_dataset(dataset_config),
        make_model(tmp_path / "model", dataset_config, **_model_dates(bars)),
        model_mode="load",
        checkpoint=str(checkpoint),
    )
    factor_dataset = backtester.config.model.config.factors[0].config.dataset
    reads = {"run": 0, "hash": 0}
    panel = StockDataset.panel

    def counting(self, *args, **kwargs):
        if self is factor_dataset:
            reads["run" if active_recorder() is not None else "hash"] += 1
        return panel(self, *args, **kwargs)

    monkeypatch.setattr(StockDataset, "panel", counting)

    backtester.run()

    assert reads == {"run": 1, "hash": 1}
    assert len(backtester.data_fingerprint["model.factors.0.dataset"]) == 1


def _names_used(path: Path) -> set[str]:
    """Every attribute name and bare name ``path`` mentions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    return names | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def test_no_code_outside_the_data_and_fingerprint_modules_knows_what_a_factor_reads():
    """The fingerprint methods are gone and nobody else asks a factor its read range."""
    owners = {
        "_input_range": {"quantlab/factor/base.py"},
        "_later_end": {"quantlab/label/forward.py"},
        "_feature_start": {"quantlab/model/base.py"},
        "_dataset_fingerprint": {"quantlab/runs/record.py"},
    }
    gone = {"fingerprint_inputs", "training_fingerprint_inputs"}
    for path in sorted((REPO_ROOT / "quantlab").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        used = _names_used(path)
        assert not used & gone, relative
        for name, allowed in owners.items():
            assert name not in used or relative in allowed, (relative, name)


def test_replaying_a_runs_weights_without_its_model_compares_only_what_it_keeps(tmp_path):
    """An override replaces a component: its records are left out on both sides."""
    from loguru import logger

    from quantlab.runs.backtest_run import BacktestRun

    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **_model_dates(bars)))
    first = _backtester(
        make_stock_dataset(dataset_config),
        make_model(tmp_path / "model", dataset_config, **_model_dates(bars)),
        model_mode="load", checkpoint=str(checkpoint), output_dir=str(tmp_path / "runs"),
    ).run()
    run = BacktestRun.open(first.run_dir)
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        replay = run.rebuild_backtester(model=None, model_mode=None, checkpoint=None)
        replay.run_weights(run.weights())
    finally:
        logger.remove(handler)

    assert set(replay.expected_fingerprint) == {"price_dataset"}
    assert [m for m in messages if "mismatch" in m] == []
