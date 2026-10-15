"""Every shipped factor built in monthly chunks equals the same factor built whole.

A chunked build (``ChunkedBuilder``) computes a factor one calendar period
at a time, each chunk warmed up with the factor's own ``warmup_bars``. That only
reproduces the whole build when every operator of the factor looks back at
most ``warmup_bars`` bars: an operator with unbounded memory (a running
total, an exponential average, an expanding statistic) gives a different
value after a chunk boundary. Here every factor class under
``quantlab/factor/predefined/`` (except the views, which have no
store) is built twice on two store paths, once whole and once through
``ChunkedBuilder(factor, granularity="month")``, with
ALL its outputs and the warm-up its docstring asks for, and the two stores
are compared under the dtype tolerance of ``tests/test_chunked_builder.py``.

Each panel has several months after the warm-up, so the chunked build is
at least three chunks; the panels carry a late listing, a delisting and a
missing bar so NaN handling across a boundary is exercised too.
"""

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from KunQuant.passes.InferWindow import infer_window

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.base import Factor
from quantlab.factor.config import (
    BenchmarkBetaConfig,
    FactorConfig,
    MarketFeatureConfig,
    NeutralizedConfig,
    PolarsFactorConfig,
)
from quantlab.factor.predefined.alpha101 import Alpha101SpotKline, Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158SpotKline, Alpha158Stock
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters
from quantlab.factor.predefined.benchmark_beta import BenchmarkBeta
from quantlab.factor.chunking import ChunkedBuilder
from quantlab.factor.predefined.literature_alpha import (
    LiteratureAlpha,
    LiteratureAlphaParameters,
)
from quantlab.factor.predefined.market import MarketFeatures
from quantlab.factor.predefined.momentum import Momentum
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.factor.predefined.residual_momentum import ResidualMomentumFF3

import tests.test_barra_style as barra_fixture
import tests.test_literature_alpha as literature_fixture
import tests.test_residual_momentum as resmom_fixture

_N_SYMBOLS = 8
_N_BARS = 420
_SPOT_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
_STOCK_COLUMNS = ["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
_NJOBS = 2

# --- comparison, copied from tests/test_chunked_builder.py --------------------

# A rolling sum is kept as a running total from the first bar computed, so a
# chunk that starts later differs in the last bits of its dtype: about one
# float32 step of the inputs' scale for KunQuant, which then survives the
# cancellation in a ratio minus one.
_TOLERANCE = {
    np.dtype("float32"): {"rtol": 1e-5, "atol": 1e-6},
    np.dtype("float64"): {"rtol": 1e-10, "atol": 1e-12},
}


def _assert_same(chunked: xr.Dataset, whole: xr.Dataset) -> None:
    assert list(chunked.data_vars) == list(whole.data_vars)
    xr.testing.assert_equal(chunked["timestamp"], whole["timestamp"])
    xr.testing.assert_equal(chunked["symbol"], whole["symbol"])
    for name in whole.data_vars:
        got, want = chunked[name].values, whole[name].values
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_allclose(got, want, equal_nan=True, **_TOLERANCE[got.dtype])


def _differences(chunked: xr.Dataset, whole: xr.Dataset) -> list[str]:
    """One line per output that ``_assert_same`` would refuse: what differs and from when."""
    lines = []
    times = pd.DatetimeIndex(whole["timestamp"].values)
    for name in whole.data_vars:
        got, want = chunked[name].values, whole[name].values
        tol = _TOLERANCE[got.dtype]
        nan_diff = np.isnan(got) != np.isnan(want)
        both = ~np.isnan(got) & ~np.isnan(want)
        abs_diff = np.where(both, np.abs(got.astype(np.float64) - want), 0.0)
        bad = nan_diff | (abs_diff > tol["atol"] + tol["rtol"] * np.abs(np.where(both, want, 0.0)))
        if not bad.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(both & (want != 0), abs_diff / np.abs(want), 0.0)
        rows = np.flatnonzero(bad.any(axis=1))
        lines.append(
            f"{name}: {int(bad.sum())} cells, {int(nan_diff.sum())} NaN-mask, "
            f"max abs {abs_diff.max():.3g}, max rel {rel.max():.3g}, "
            f"first {times[rows[0]].date()}, last {times[rows[-1]].date()}"
        )
    return lines


# --- panels -----------------------------------------------------------------


def _ohlcv(seed: int, n_bars: int) -> dict[str, np.ndarray]:
    """A seeded OHLCV random walk with a late listing, a delisting and a missing bar."""
    rng = np.random.default_rng(seed)
    shape = (n_bars, _N_SYMBOLS)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=shape), axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    open_ = prev * np.exp(rng.normal(0.0, 0.01, size=shape))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.01, size=shape)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.01, size=shape)))
    volume = rng.lognormal(13.0, 0.5, size=shape)
    values = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    for v in values.values():
        v[:100, 1] = np.nan  # listed at bar 100, inside every warm-up
        v[n_bars - 40:, 5] = np.nan  # delisted 40 bars before the end
        v[n_bars - 90, 2] = np.nan  # one missing bar inside the range
    return values


def _write(path: Path, timestamps, symbols, variables: dict[str, np.ndarray]) -> Path:
    xr.Dataset(
        {k: (("timestamp", "symbol"), v) for k, v in variables.items()},
        coords={"timestamp": timestamps, "symbol": symbols},
    ).to_zarr(path, mode="w")
    return path


def _stock_dataset(path: Path) -> StockDataset:
    return StockDataset(DatasetConfig(
        raw_data_dir_path=str(path.parent / "raw"), zarr_file_path=str(path),
        market="us_equity", frequency="1d",
    ))


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("chunked_predefined")


@pytest.fixture(scope="module")
def spot(root) -> SpotKlineDataset:
    """420 calendar days of eight symbols, Binance-named as ``SpotKlineDataset`` stores them."""
    ohlcv = _ohlcv(1, _N_BARS)
    path = _write(
        root / "spot.zarr",
        pd.date_range("2022-01-01", periods=_N_BARS, freq="D"),
        [f"S{i}USDT" for i in range(_N_SYMBOLS)],
        {
            "Open": ohlcv["open"], "High": ohlcv["high"], "Low": ohlcv["low"],
            "Close": ohlcv["close"], "Volume": ohlcv["volume"],
            # The traded value at a price inside the bar, so VWAP is not the close.
            "Quote asset volume": ohlcv["volume"] * (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3,
        },
    )
    return SpotKlineDataset(DatasetConfig(
        raw_data_dir_path=str(root / "raw"), zarr_file_path=str(path),
        market="crypto_spot", frequency="1d",
    ))


_STOCK_TIMES = pd.bdate_range("2022-01-03", periods=_N_BARS)
# Not sorted: a whole build and a chunked one must still lay the store out alike (#247).
_STOCK_SYMBOLS = ["MSFT", "AAPL", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "AVGO"]


@pytest.fixture(scope="module")
def stock(root) -> StockDataset:
    """420 business days of eight symbols with raw and adjusted price columns."""
    ohlcv = _ohlcv(2, _N_BARS)
    variables = {**ohlcv, **{f"adj{k.capitalize()}": v for k, v in ohlcv.items()}}
    return _stock_dataset(_write(root / "stock.zarr", _STOCK_TIMES, _STOCK_SYMBOLS, variables))


@pytest.fixture(scope="module")
def benchmark(root) -> StockDataset:
    """One index series on the stock calendar, with an adjusted close and volume."""
    rng = np.random.default_rng(3)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, _N_BARS)))
    volume = rng.uniform(1e6, 5e6, _N_BARS)
    path = _write(root / "spy.zarr", _STOCK_TIMES, ["SPY"], {
        "adjClose": close[:, None], "adjVolume": volume[:, None],
        "close": close[:, None], "volume": volume[:, None],
    })
    return _stock_dataset(path)


@pytest.fixture(scope="module")
def exposures(root) -> StockDataset:
    """Market cap and industry of the stock symbols on the stock calendar."""
    rng = np.random.default_rng(4)
    cap = np.tile(rng.lognormal(22.0, 1.0, _N_SYMBOLS), (_N_BARS, 1))
    cap *= rng.lognormal(0.0, 0.05, size=cap.shape)
    industry = np.tile(rng.integers(0, 3, _N_SYMBOLS).astype(float), (_N_BARS, 1))
    path = _write(root / "exposures.zarr", _STOCK_TIMES, _STOCK_SYMBOLS,
                  {"marketcap": cap, "industry": industry})
    return _stock_dataset(path)


# --- cases ------------------------------------------------------------------
#
# A case is ``(make, start, end)``. ``make(store)`` returns the factor with
# its store at ``store``; it is called twice, once per build.


def _graph_lookback(cls, dataset, columns: list[str], **config) -> int:
    """KunQuant's own window inference: the longest lookback over the graph's outputs."""
    probe = cls(FactorConfig(warmup_bars=0, dataset=dataset, mode="batch",
                             data_columns=columns, njobs=_NJOBS, **config))
    return max(infer_window(probe._get_factor_func()).values())


def _kunquant(cls, dataset, columns, warmup):
    def make(store: Path) -> Factor:
        return cls(FactorConfig(
            warmup_bars=warmup, dataset=dataset, mode="batch", data_columns=columns,
            file_path=str(store), njobs=_NJOBS,
        ))
    return make


_SPOT_RANGE = ("2022-10-03", "2023-02-24")  # bar 275 of 420
_STOCK_RANGE = ("2023-01-03", "2023-08-11")  # bar 261 of 420


def _alpha101_spot(request):
    spot = request.getfixturevalue("spot")
    # "warmup_bars must cover the longest alpha lookback plus zscore_window - 1".
    warmup = _graph_lookback(Alpha101SpotKline, spot, _SPOT_COLUMNS)
    return _kunquant(Alpha101SpotKline, spot, _SPOT_COLUMNS, warmup), *_SPOT_RANGE


def _alpha158_spot(request):
    spot = request.getfixturevalue("spot")
    # "the longest feature window (60 bars for the full set) plus zscore_window - 1".
    return _kunquant(Alpha158SpotKline, spot, _SPOT_COLUMNS, 60 + 20 - 1), *_SPOT_RANGE


def _alpha101_stock(request):
    stock = request.getfixturevalue("stock")
    warmup = _graph_lookback(Alpha101Stock, stock, _STOCK_COLUMNS)
    return _kunquant(Alpha101Stock, stock, _STOCK_COLUMNS, warmup), *_STOCK_RANGE


def _alpha158_stock(request):
    stock = request.getfixturevalue("stock")
    # The longest feature window; the z-score is cross-sectional and adds none.
    return _kunquant(Alpha158Stock, stock, _STOCK_COLUMNS, 60), *_STOCK_RANGE


def _momentum(request):
    spot = request.getfixturevalue("spot")

    def make(store: Path) -> Factor:
        return Momentum(PolarsFactorConfig(
            warmup_bars=20, dataset=spot, kwargs={"n": 20}, file_path=str(store),
        ))
    return make, *_SPOT_RANGE


def _benchmark_beta(request):
    stock, spy = request.getfixturevalue("stock"), request.getfixturevalue("benchmark")

    def make(store: Path) -> Factor:
        return BenchmarkBeta(BenchmarkBetaConfig(
            warmup_bars=60, dataset=stock, benchmark=spy, lookback_bars=60, min_bars=20,
            file_path=str(store),
        ))
    return make, *_STOCK_RANGE


def _market_features(request):
    stock, spy = request.getfixturevalue("stock"), request.getfixturevalue("benchmark")

    def make(store: Path) -> Factor:
        # warmup_bars defaults to 60, the longest window.
        return MarketFeatures(MarketFeatureConfig(
            dataset=stock, series={"spy": spy}, file_path=str(store),
        ))
    return make, *_STOCK_RANGE


_NEUTRALIZED_INNER = ("KMID", "ROC60", "STD20", "CORR60", "CORD60", "RSV10", "VSUMP30")


def _neutralized(request):
    stock, exposures = request.getfixturevalue("stock"), request.getfixturevalue("exposures")
    warmup = _graph_lookback(Alpha158Stock, stock, _STOCK_COLUMNS,
                             factor_names=_NEUTRALIZED_INNER)

    def make(store: Path) -> Factor:
        # The wrapper's own warm-up must be 0; the wrapped factor warms itself.
        inner = Alpha158Stock(FactorConfig(
            warmup_bars=warmup, dataset=stock, mode="batch", data_columns=_STOCK_COLUMNS,
            factor_names=_NEUTRALIZED_INNER, njobs=_NJOBS,
        ))
        return NeutralizedFactor(NeutralizedConfig(
            factor=inner, dataset=exposures, file_path=str(store), njobs=_NJOBS,
        ))
    return make, *_STOCK_RANGE


def _literature_alpha(request):
    root = request.getfixturevalue("root") / "literature"
    root.mkdir(exist_ok=True)
    panel, _, _ = literature_fixture._synthetic_panel(periods=220, symbols=_N_SYMBOLS)
    dataset = literature_fixture._dataset(root, panel)
    names = LiteratureAlpha._CORE_FACTOR_NAMES + LiteratureAlpha._DIAGNOSTIC_FACTOR_NAMES
    windows = dict(high_52week_window=60, short_reversal_window=21, max_return_window=21,
                   idio_vol_window=21, amihud_window=21)
    columns = LiteratureAlphaParameters(**windows).required_panel_columns(names)
    times = pd.DatetimeIndex(panel["timestamp"].values)

    def make(store: Path) -> Factor:
        # The longest window; every other input is an event value carried forward.
        return LiteratureAlpha(literature_fixture._config(
            dataset, root, warmup_bars=max(windows.values()), data_columns=columns,
            factor_names=names, file_path=str(store),
            kwargs={**windows, "emit_diagnostics": True},
        ))
    return make, str(times[65].date()), str(times[-1].date())


def _residual_momentum(csv: bool):
    def case(request):
        root = request.getfixturevalue("root") / f"resmom_{'csv' if csv else 'panel'}"
        root.mkdir(exist_ok=True)
        ret_only, full, csv_path = resmom_fixture._daily_fixture(
            root, periods=220, symbols=_N_SYMBOLS)
        params = resmom_fixture.DAILY
        times = pd.DatetimeIndex(xr.open_zarr(full.config.zarr_file_path)["timestamp"].values)

        def make(store: Path) -> Factor:
            # "warmup_bars ... covers the regression window"; diagnostics on by default.
            return ResidualMomentumFF3(resmom_fixture._factor_config(
                ret_only if csv else full, root,
                warmup_bars=params["regression_window"],
                data_columns=("ret",) if csv else ("ret", "risk_free", "mkt_rf", "smb", "hml"),
                factor_names=None, file_path=str(store),
                kwargs={**params, **({"fama_french_csv": str(csv_path)} if csv else {})},
            ))
        return make, str(times[65].date()), str(times[-1].date())
    return case


_BARRA_T = 200


def _barra(request):
    root = request.getfixturevalue("root") / "barra"
    root.mkdir(exist_ok=True)
    warmup = BarraStyleParameters(**barra_fixture._KWARGS).warmup_bars
    with pytest.MonkeyPatch.context() as patch:
        # The fixture's hard-coded events all fall before bar 100; only its length changes.
        patch.setattr(barra_fixture, "_T", _BARRA_T)
        config = barra_fixture._config(root, warmup_bars=warmup)
    times = pd.bdate_range("2021-01-04", periods=_BARRA_T)

    def make(store: Path) -> Factor:
        return BarraStyle(dataclasses.replace(config, file_path=str(store)))
    return make, str(times[warmup + 4].date()), str(times[-1].date())


_CASES: dict[str, Callable] = {
    "Alpha101SpotKline": _alpha101_spot,
    "Alpha158SpotKline": _alpha158_spot,
    "Alpha101Stock": _alpha101_stock,
    "Alpha158Stock": _alpha158_stock,
    "LiteratureAlpha": _literature_alpha,
    "ResidualMomentumFF3-panel": _residual_momentum(csv=False),
    "ResidualMomentumFF3-csv": _residual_momentum(csv=True),
    "BarraStyle": _barra,
    "BenchmarkBeta": _benchmark_beta,
    "MarketFeatures": _market_features,
    "NeutralizedFactor": _neutralized,
    "Momentum": _momentum,
}


def _stored(factor: Factor) -> xr.Dataset:
    return xr.open_zarr(factor.store_path).load()


# Every KunQuant factor here passes only because its batch graph compiles with
# ``quantlab.factor.kunquant.BATCH_OPTIONS`` (``opt_reduce=False``): KunQuant's
# running-total windowed sum carries rounding residue from the first bar
# computed, which a later chunk start changes, by O(1) where a correlation or
# z-score divides by a variance that is truly 0 (Alpha101, Alpha158 CORR10).
@pytest.mark.parametrize("case", list(_CASES))
def test_a_monthly_chunked_build_equals_the_whole_build(case, request, root):
    make, start, end = _CASES[case](request)
    whole = make(root / case / "whole.zarr")
    inner = make(root / case / "chunked.zarr")
    assert whole.config.factor_names == inner.config.factor_names

    whole.build(start, end)
    calls = []
    build, extend = inner.build, inner.extend
    inner.build = lambda s, e: calls.append("build") or build(s, e)
    inner.extend = lambda e: calls.append("extend") or extend(e)
    ChunkedBuilder(inner, granularity="month").build(start, end)

    assert calls[0] == "build" and len(calls) >= 3, calls
    assert inner.store_range() == whole.store_range()
    chunked_store, whole_store = _stored(inner), _stored(whole)
    assert set(whole_store.data_vars) == set(whole.get_factor_names())
    differences = _differences(chunked_store, whole_store)
    assert not differences, "\n".join(differences)
    _assert_same(chunked_store, whole_store)
