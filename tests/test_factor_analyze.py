"""Tests for ``Factor.analyze`` and ``quantlab.analysis.factor_report``.

The panels are synthetic: the ``spot_kline_zarr`` store from ``conftest.py``
feeds ``Momentum`` and two test-local Polars factors. ``_ForwardReturn`` is a
``Forward`` label (span 1, delay 0) over the trailing close-to-close return,
so its value at t is the next bar's close-to-close return, and
``_Oracle`` is a factor that is a strictly increasing function of that same
forward return, so it orders symbols exactly as the label does. Every
analysis covers ``ALL``, the whole synthetic store; the factors take no
warm-up, so nothing is short of history.
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from quantlab.analysis.factor_report import FactorAnalyzer, Fret
from quantlab.backend.zarr import XrBackend
from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.label.config import ForwardConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.factor.polars import FactorPolars
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.predefined.momentum import Momentum
from quantlab.label.forward import Forward
from quantlab.core.component import rebuild

#: Every bar of the synthetic stores, which start on 2024-01-01.
ALL = ("2024-01-01", "2024-12-31")

EXPECTED_FILES = [
    "config.json",
    "ic.csv",
    "monthly_ic.csv",
    "quantile_returns.csv",
    "summary.csv",
    "summary.json",
    "turnover.csv",
]


def _forward_return() -> pl.Expr:
    """Next bar's close-to-close return per symbol (rows sorted by symbol, time)."""
    close = pl.col("Close")
    return close.shift(-1).over("symbol") / close - 1.0


class _TrailingReturn(FactorPolars):
    """Trailing one-bar close-to-close return, named ``fwd_1`` for the label."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        close = pl.col("Close")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((close / close.shift(1).over("symbol") - 1.0).alias("fwd_1"))
            .select(["timestamp", "symbol", "fwd_1"])
        )


class _ForwardReturn(Forward):
    """One-bar forward return label, ``fwd_1``: the trailing return one bar later."""

    def __init__(self, factor_config: PolarsFactorConfig):
        super().__init__(
            ForwardConfig(factor=_TrailingReturn(factor_config), span=1, delay=0)
        )


class _Oracle(FactorPolars):
    """A factor that is the cube of the forward return: a perfect ranker."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((_forward_return() ** 3).alias("oracle"))
            .select(["timestamp", "symbol", "oracle"])
        )


class _TwoOracles(_Oracle):
    """The oracle and its negation, as two factor variables."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        oracle = super()._get_factor_lazyframe(lf)
        return oracle.with_columns((-pl.col("oracle")).alias("oracle_neg"))


class _StaticLabel:
    """A duck-typed fret that hands back a fixed panel for any range."""

    def __init__(self, panel: xr.Dataset):
        self.panel = panel

    def compute(self, start, end) -> xr.Dataset:
        return self.panel

    def span_bars(self) -> int:
        return 1

    def lookahead_bars(self) -> int:
        return 1

    def get_config(self) -> dict:
        return {"name": "static"}


def _polars_config(dataset_config: DatasetConfig, tmp_path: Path, name: str, **kwargs):
    return PolarsFactorConfig(
        warmup_bars=0,
        dataset=SpotKlineDataset(dataset_config),
        file_path=str(tmp_path / "factors" / f"{name}.zarr"),
        **kwargs,
    )


@pytest.fixture
def oracle_and_label(spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path):
    """An ``_Oracle`` factor and a ``_ForwardReturn`` label, 20 symbols, 90 bars."""
    dataset_config = spot_kline_zarr(
        symbols=[f"S{i}USDT" for i in range(20)], periods=90
    )
    factor = _Oracle(_polars_config(dataset_config, tmp_path, "oracle"))
    label = _ForwardReturn(_polars_config(dataset_config, tmp_path, "fwd"))
    return factor, label


def test_a_perfect_predictor_has_ic_one_and_monotone_quantile_returns(oracle_and_label):
    factor, label = oracle_and_label

    result = factor.analyze(*ALL, frets=[label])

    assert list(result.pairs) == ["oracle__fwd_1"]
    pair = result.pairs["oracle__fwd_1"]
    ic = pair.ic.dropna()
    # 90 bars; the last has no forward return.
    assert len(ic) == 89
    np.testing.assert_allclose(ic.to_numpy(), 1.0, atol=1e-12)
    assert pair.summary["ic_mean"] == pytest.approx(1.0)
    assert pair.summary["ic_positive_ratio"] == 1.0

    means = pair.mean_quantile_returns.to_numpy()
    assert len(means) == 5
    assert np.all(np.diff(means) > 0)
    # Per period too: every bucket beats the one below it.
    per_period = pair.quantile_returns.dropna().to_numpy()
    assert np.all(np.diff(per_period, axis=1) > 0)
    assert (pair.spread.dropna() > 0).all()
    assert pair.summary["cumulative_long_short"] > 0
    turnover = pair.turnover.dropna().to_numpy()
    assert ((turnover >= 0) & (turnover <= 1)).all()
    assert result.figures["oracle__fwd_1"] is not None


def test_frequency_mismatch_raises(spot_kline_zarr, tmp_path):
    factor = Momentum(_polars_config(spot_kline_zarr(), tmp_path, "mom", kwargs={"n": 5}))
    daily = factor.compute(*ALL)["momentum_5"]
    weekly = daily.isel(timestamp=slice(None, None, 7)).rename("ret").to_dataset()

    with pytest.raises(ValueError, match=r"1 days.*7 days"):
        factor.analyze(*ALL, frets=[_StaticLabel(weekly)])


def test_panels_are_inner_joined_before_computing(oracle_and_label):
    factor, label = oracle_and_label
    labels = label.compute(*ALL)
    narrowed = labels.isel(timestamp=slice(10, 40), symbol=slice(0, 12))

    pair = factor.analyze(*ALL, frets=[_StaticLabel(narrowed)]).pairs["oracle__fwd_1"]

    assert len(pair.ic) == 30
    assert pair.summary["n_symbols"] == 12
    assert pair.start == pd.Timestamp(labels["timestamp"].values[10])


def test_output_dir_writes_tables_figure_and_rebuildable_config(
    spot_kline_zarr, tmp_path
):
    dataset_config = spot_kline_zarr(symbols=[f"S{i}USDT" for i in range(10)])
    factor = Momentum(_polars_config(dataset_config, tmp_path, "mom", kwargs={"n": 5}))
    label = _ForwardReturn(_polars_config(dataset_config, tmp_path, "fwd"))
    out = tmp_path / "report" / "momentum"

    result = factor.analyze(*ALL, frets=[label], output_dir=str(out))

    assert sorted(p.name for p in out.iterdir()) == sorted(
        EXPECTED_FILES + ["momentum_5__fwd_1.png"]
    )
    summary = json.loads((out / "summary.json").read_text())
    assert [row["factor"] for row in summary["pairs"]] == ["momentum_5"]
    assert summary["pairs"][0]["ic_mean"] == pytest.approx(
        result.pairs["momentum_5__fwd_1"].summary["ic_mean"]
    )
    assert (out / "momentum_5__fwd_1.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    ic = pd.read_csv(out / "ic.csv")
    assert list(ic.columns) == ["timestamp", "factor", "fret", "ic", "pearson_ic"]
    quantile_returns = pd.read_csv(out / "quantile_returns.csv")
    assert sorted(quantile_returns["quantile"].unique()) == [1, 2, 3, 4, 5]

    config = json.loads((out / "config.json").read_text())
    assert list(config) == ["factor", "frets"]
    rebuilt = rebuild(config["factor"])
    assert type(rebuilt) is Momentum
    assert list(rebuilt.get_factor_names()) == ["momentum_5"]
    assert config["frets"][0]["name"].endswith("_ForwardReturn")


def test_output_dir_none_writes_nothing(oracle_and_label, tmp_path):
    factor, label = oracle_and_label
    before = sorted(str(p) for p in tmp_path.rglob("*"))

    result = factor.analyze(*ALL, frets=[label])

    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    figure = result.figures["oracle__fwd_1"]
    assert type(figure).__name__ == "Figure"
    assert "oracle" in figure.get_suptitle() and "fwd_1" in figure.get_suptitle()


def test_bad_arguments_raise(oracle_and_label):
    factor, label = oracle_and_label

    with pytest.raises(ValueError, match="at least one forward-return"):
        factor.analyze(*ALL, frets=[])
    with pytest.raises(ValueError, match="not in _Oracle's panel"):
        factor.analyze(*ALL, factor_names=["nope"], frets=[label])
    with pytest.raises(ValueError, match="data_strategy"):
        factor.analyze(*ALL, frets=[label], data_strategy="load")


def test_the_read_strategy_analyzes_the_stored_panels(oracle_and_label):
    factor, label = oracle_and_label
    factor.build(*ALL)
    label.build(*ALL)

    read = factor.analyze(*ALL, frets=[label], data_strategy="read")
    computed = factor.analyze(*ALL, frets=[label])
    narrow = factor.analyze("2024-01-20", "2024-03-10", frets=[label], data_strategy="read")

    pd.testing.assert_series_equal(
        read.pairs["oracle__fwd_1"].ic, computed.pairs["oracle__fwd_1"].ic
    )
    assert narrow.pairs["oracle__fwd_1"].start == pd.Timestamp("2024-01-20")
    assert len(narrow.pairs["oracle__fwd_1"].ic) == 51


def test_the_read_strategy_refuses_a_range_the_stores_do_not_cover(oracle_and_label):
    factor, label = oracle_and_label
    factor.build("2024-01-01", "2024-02-29")
    label.build("2024-01-01", "2024-02-29")

    with pytest.raises(ValueError, match="does not contain"):
        factor.analyze(*ALL, frets=[label], data_strategy="read")


def test_a_frozen_ranking_has_zero_turnover_and_unit_autocorrelation():
    timestamps = pd.date_range("2024-01-01", periods=6)
    symbols = [f"S{i}" for i in range(10)]
    coords = {"timestamp": timestamps, "symbol": symbols}
    factor = xr.DataArray(
        np.tile(np.arange(10.0), (6, 1)), coords=coords, dims=("timestamp", "symbol")
    )
    fret = xr.DataArray(
        np.random.default_rng(1).normal(size=(6, 10)),
        coords=coords,
        dims=("timestamp", "symbol"),
    )

    pair = FactorAnalyzer(quantiles=5).analyze_pair(factor, fret, "f", "r")

    assert pair.turnover.iloc[0].isna().all()
    assert (pair.turnover.iloc[1:] == 0.0).all().all()
    np.testing.assert_allclose(pair.rank_autocorrelation.iloc[1:], 1.0)
    # Bucket 1 holds the two lowest factor values, S0 and S1.
    expected = fret.isel(symbol=[0, 1]).mean("symbol").to_numpy()
    np.testing.assert_allclose(pair.quantile_returns[1].to_numpy(), expected)


def test_a_multi_bar_fret_compounds_the_per_bar_rate():
    timestamps = pd.date_range("2024-01-01", periods=3)
    coords = {"timestamp": timestamps, "symbol": ["A", "B"]}
    factor = xr.DataArray([[0.0, 1.0]] * 3, coords=coords, dims=("timestamp", "symbol"))
    fret = xr.DataArray([[0.0, 0.21]] * 3, coords=coords, dims=("timestamp", "symbol"))

    pair = FactorAnalyzer(quantiles=2).analyze_pair(factor, fret, "f", "r", horizon=2)

    # 21% over two bars is 10% per bar, compounded over three bars.
    assert pair.cumulative_quantile_returns[2].iloc[-1] == pytest.approx(1.1**3 - 1)


def test_importing_the_factor_layer_does_not_import_matplotlib():
    code = "import sys, quantlab.factor.base; print('matplotlib' in sys.modules)"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert output.strip() == "False"


def test_analyze_against_the_kunquant_return_label(tmp_path):
    """End to end with ``quantlab.label.predefined.fret.Return`` and its horizon."""
    from quantlab.dataset.stock import StockDataset
    from quantlab.label.predefined.fret import Return

    rng = np.random.default_rng(0)
    symbols = [f"T{i}" for i in range(8)]
    timestamps = pd.date_range("2024-01-01", periods=40)
    price = 50 + np.cumsum(rng.normal(size=(40, 8)), axis=0)
    store = tmp_path / "stock.zarr"
    XrBackend().to_internal(
        xr.Dataset(
            {"adjOpen": (["timestamp", "symbol"], price), "Close": (["timestamp", "symbol"], price)},
            coords={"timestamp": timestamps, "symbol": symbols},
        )
    ).write(str(store))

    def dataset():
        return StockDataset(
            DatasetConfig(
                raw_data_dir_path=str(tmp_path / "raw"),
                zarr_file_path=str(store),
                market="us_equity",
                frequency="1d",
            )
        )

    factor = Momentum(
        PolarsFactorConfig(
            warmup_bars=0,
            dataset=dataset(),
            file_path=str(tmp_path / "mom.zarr"),
            kwargs={"n": 3},
        )
    )
    label = Return(
        FactorConfig(
            warmup_bars=0,
            mode="batch",
            data_columns=["adjOpen"],
            kwargs={"n_forward_periods": 2},
            dataset=dataset(),
            file_path=str(tmp_path / "ret.zarr"),
            njobs=2,
        )
    )

    result = factor.analyze(*ALL, frets=[label], quantiles=4)

    pair = result.pairs["momentum_3__ret_2"]
    assert pair.horizon == 2
    # Momentum needs 3 bars of history; the label loses its last 3 bars.
    assert pair.ic.notna().sum() == 40 - 3 - 3
    assert set(result.summary_table()["fret"]) == {"ret_2"}


# -- the polars metrics agree with a plain per-period reference ------------------


def _reference_pair(f: np.ndarray, r: np.ndarray, quantiles: int) -> dict:
    """Per-period metrics computed the slow, obvious way with pandas and scipy."""
    from scipy import stats

    T = f.shape[0]
    ic = np.full(T, np.nan)
    rac = np.full(T, np.nan)
    qret = np.full((T, quantiles), np.nan)
    turnover = np.full((T, quantiles), np.nan)
    previous_bucket = None
    for t in range(T):
        both = np.isfinite(f[t]) & np.isfinite(r[t])
        if both.sum() >= 2:
            ic[t] = stats.spearmanr(f[t][both], r[t][both]).statistic
        if t > 0:
            lag_both = np.isfinite(f[t]) & np.isfinite(f[t - 1])
            if lag_both.sum() >= 2:
                rac[t] = stats.spearmanr(f[t][lag_both], f[t - 1][lag_both]).statistic
        bucket = np.full(f.shape[1], np.nan)
        if both.sum() >= quantiles:
            order = pd.Series(np.where(both, f[t], np.nan)).rank(method="first")
            bucket = np.floor((order - 1) / both.sum() * quantiles).to_numpy() + 1
            for q in range(1, quantiles + 1):
                in_q = bucket == q
                qret[t, q - 1] = r[t][in_q].mean()
                if previous_bucket is not None and np.isfinite(previous_bucket).any():
                    turnover[t, q - 1] = (in_q & (previous_bucket != q)).sum() / in_q.sum()
        previous_bucket = bucket
    return {"ic": ic, "rank_autocorrelation": rac, "quantile_returns": qret, "turnover": turnover}


def test_polars_metrics_match_a_per_period_reference():
    rng = np.random.default_rng(7)
    T, S, Q = 60, 24, 4
    f = rng.normal(size=(T, S))
    r = 0.3 * f + rng.normal(size=(T, S))
    f[rng.random((T, S)) < 0.15] = np.nan
    r[rng.random((T, S)) < 0.15] = np.nan
    f[3] = np.nan          # no IC, no buckets
    f[10, :22] = np.nan    # fewer usable symbols than buckets
    coords = {"timestamp": pd.date_range("2024-01-01", periods=T), "symbol": [f"S{i}" for i in range(S)]}
    factor = xr.DataArray(f, dims=("timestamp", "symbol"), coords=coords)
    fret = xr.DataArray(r, dims=("timestamp", "symbol"), coords=coords)

    pair = FactorAnalyzer(quantiles=Q, plot=False).analyze_pair(factor, fret, "f", "r", horizon=2)
    reference = _reference_pair(f, r, Q)

    np.testing.assert_allclose(pair.ic.to_numpy(), reference["ic"], rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        pair.rank_autocorrelation.to_numpy(), reference["rank_autocorrelation"], rtol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        pair.quantile_returns.to_numpy(), reference["quantile_returns"], rtol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        pair.turnover.to_numpy(), reference["turnover"], rtol=1e-12, equal_nan=True
    )
    assert np.isnan(pair.ic.iloc[3]) and np.isnan(pair.quantile_returns.iloc[10]).all()
    assert pair.cumulative_ic.iloc[-1] == pytest.approx(np.nansum(reference["ic"]))
    assert pair.summary["n_symbols"] == S


def test_many_columns_give_the_same_metrics_as_one_at_a_time():
    rng = np.random.default_rng(11)
    T, S = 40, 16
    coords = {"timestamp": pd.date_range("2024-01-01", periods=T), "symbol": [f"S{i}" for i in range(S)]}
    r = rng.normal(size=(T, S))
    features = xr.Dataset(
        {f"a{i}": (("timestamp", "symbol"), 0.2 * r + rng.normal(size=(T, S))) for i in range(5)},
        coords=coords,
    )
    labels = xr.Dataset({"ret_1": (("timestamp", "symbol"), r)}, coords=coords)

    analyzer = FactorAnalyzer(quantiles=3, plot=False, chunk_size=2)
    together = {p.key: p for p in analyzer.analyze_many(features, labels)}
    for name in features.data_vars:
        alone = analyzer.analyze_pair(features[name], labels["ret_1"], name, "ret_1")
        pd.testing.assert_series_equal(together[alone.key].ic, alone.ic)
        pd.testing.assert_frame_equal(together[alone.key].turnover, alone.turnover)
        assert together[alone.key].summary == alone.summary


def test_output_dir_with_several_pairs_draws_every_figure_in_parallel(oracle_and_label, tmp_path):
    oracle, label = oracle_and_label
    factor = _TwoOracles(dataclasses.replace(oracle.config, factor_names=None))

    result = factor.analyze(*ALL, frets=[label], output_dir=str(tmp_path / "report"))

    assert result.figures == {}
    assert sorted(p.name for p in (tmp_path / "report").glob("*.png")) == [
        "factor_correlation.png", "oracle__fwd_1.png", "oracle_neg__fwd_1.png",
    ]
    assert result.correlation.mean.loc["oracle", "oracle_neg"] == pytest.approx(-1.0)
    assert result.pairs["oracle_neg__fwd_1"].summary["ic_mean"] == pytest.approx(-1.0)


def test_output_dir_without_plot_writes_the_tables_and_no_figure(tmp_path):
    rng = np.random.default_rng(3)
    T, S = 40, 16
    coords = {"timestamp": pd.date_range("2024-01-01", periods=T), "symbol": [f"S{i}" for i in range(S)]}
    r = rng.normal(size=(T, S))
    features = xr.Dataset(
        {name: (("timestamp", "symbol"), 0.2 * r + rng.normal(size=(T, S))) for name in ("a", "b")},
        coords=coords,
    )
    labels = xr.Dataset({"ret_1": (("timestamp", "symbol"), r)}, coords=coords)
    out = tmp_path / "tables"

    analysis = FactorAnalyzer(quantiles=3, plot=False).analyze_panels(
        features, [Fret(labels)], output_dir=out
    )

    assert analysis.figures == {}
    assert list(out.glob("*.png")) == []
    assert {"summary.json", "summary.csv", "ic.csv", "factor_correlation.csv"} <= {p.name for p in out.iterdir()}
    drawn = analysis.save(tmp_path / "drawn")
    assert sorted(p.name for p in drawn.glob("*.png")) == ["a__ret_1.png", "b__ret_1.png", "factor_correlation.png"]
