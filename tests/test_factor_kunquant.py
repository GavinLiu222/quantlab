"""End-to-end batch KunQuant factor-computation tests (Phase 3 Plan 01,
FACTOR-01 / ROADMAP Phase 3 Success Criterion 1).

Before 03-01, BOTH tests in this file raised
`TypeError: WindowedZScore.decompose() takes 1 positional argument but 2 were
given`: `factor/kunquant_ops.py` (now `kunquant_ts.py`) declared `decompose(self)` while the installed
KunQuant 0.1.11 declares the contract as
`CompositiveOp.decompose(self, options: dict)` (`KunQuant/Op.py:292`) and
invokes it positionally (`KunQuant/passes/Decompose.py:15`). Because both
`Alpha101SpotKline` and `Alpha158SpotKline` wrap every `Output(...)` in
`WindowedZScore(...)`, every batch computation in the repository was dead. These
tests are the regression lock on that fix.

Cost control: each test passes an explicit 1-3 element `factor_names` list so
the compiled graph stays tiny (~1.5 s per compilation) and `njobs=4` so the
KunQuant executor does not spawn `FactorConfig`'s default 128 threads.
"""

import inspect
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import xarray as xr
from KunQuant.Op import Input, Output

from conftest import compute_all

from quantlab.factor.config import FactorConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.base import MarketDataset
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.predefined.alpha101 import Alpha101SpotKline, Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158SpotKline, Alpha158Stock


_ADJUSTED_STOCK_COLUMNS = ["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]


def _factor_config(
    dataset_config: DatasetConfig,
    factor_names: list[str],
    data_columns: list[str],
    tmp_path: Path,
    warmup_bars: int = 10,
    dataset_cls: type[MarketDataset] = SpotKlineDataset,
) -> FactorConfig:
    """Build a `FactorConfig` over the given synthetic Zarr store.

    `factor_names` is always passed explicitly: it both bypasses the
    `FactorKunQuant.config` setter's `_get_factor_names()` call (which would
    enumerate all 169 Alpha158 / 82 Alpha101 names) and restricts the compiled
    graph to only the reachable `Output(...)` nodes.

    `dataset_cls` selects the `Dataset` subclass wrapped around
    `dataset_config`. It defaults to `SpotKlineDataset` so every 03-01 caller
    is unaffected; 03-03's US-equity tests pass `StockDataset` rather than
    duplicating this helper.
    """
    return FactorConfig(
        warmup_bars=warmup_bars,
        dataset=dataset_cls(dataset_config),
        mode="batch",
        data_columns=data_columns,
        factor_names=factor_names,
        file_path=str(tmp_path / "factors" / "out.zarr"),
        njobs=4,
    )


def test_alpha158_spot_batch_compute_returns_xarray_dataset(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """FACTOR-01 / ROADMAP Phase 3 Success Criterion 1: computing the Alpha158
    factor set in batch mode over crypto spot data returns an `xr.Dataset`
    indexed by `[timestamp, symbol]` carrying real, finite factor values.
    """
    dataset_config = spot_kline_zarr(periods=60, seed=0)
    factor = Alpha158SpotKline(
        _factor_config(
            dataset_config,
            factor_names=["KMID", "VOLUME0", "STD5"],
            data_columns=["open", "close", "volume"],
            tmp_path=tmp_path,
        )
    )

    result = compute_all(factor)

    assert isinstance(result, xr.Dataset)
    assert dict(result.sizes) == {"timestamp": 60, "symbol": 8}
    assert sorted(result.data_vars) == ["KMID", "STD5", "VOLUME0"]
    assert np.isfinite(result["KMID"].to_numpy()).sum() > 0


def test_alpha101_spot_batch_compute_returns_xarray_dataset(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """FACTOR-01: the same batch path works for the Alpha101 family, proving
    the shared `WindowedZScore` normalization op decomposes correctly for both
    alpha families rather than only the one that happened to be exercised.
    """
    dataset_config = spot_kline_zarr(periods=60, seed=0)
    factor = Alpha101SpotKline(
        _factor_config(
            dataset_config,
            factor_names=["alpha001"],
            data_columns=["close"],
            tmp_path=tmp_path,
        )
    )

    result = compute_all(factor)

    assert isinstance(result, xr.Dataset)
    assert "alpha001" in result.data_vars
    assert np.isfinite(result["alpha001"].to_numpy()).sum() > 0


# ---------------------------------------------------------------------------
# Phase 3 Plan 03 -- US equities (D-01, NORM-01/D-09)
#
# No US-equity store carries a dollar-volume (`amount`) column, and
# `StockDataset._to_kunquant()` exports the requested columns as they are.
# `Alpha101Stock` and `Alpha158Stock` therefore build `vwap` inside the graph
# from the adjusted typical price instead of KunQuant's `amount / volume`.
#
# `stock_zarr` defaults to 2 symbols, but KunQuant's compiled TS layout
# requires the symbol axis to be a multiple of its SIMD block width (8) or
# `kr.runGraph` raises `RuntimeError: Bad shape at open`. Every test that
# COMPILES a graph therefore passes `_STOCK_SYMBOLS` (8 tickers).
# ---------------------------------------------------------------------------

_STOCK_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
]


def test_alpha101_stock_bugfix_batch_compute_returns_xarray_dataset(
    stock_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """D-02 regression lock: before 03-03 this exact construction raised
    `RuntimeError: Bad inputs, given <class 'NoneType'>` inside
    `Alpha101.AllData.__init__` -- `Alpha101Stock._get_factor_func()` never
    passed `amount`, yet `AllData` builds `vwap` from it unless one is given.
    The class now reads the adjusted columns and passes the adjusted typical
    price as `vwap` (as `Alpha158Stock` does), so the US-equity Alpha101 batch
    path computes real values, including the vwap-based `alpha041`.

    The normalization of the stock classes (a cross-sectional z-score per
    bar, no time-series z-score) is locked below by the normalization matrix
    test, not here.
    """
    dataset_config = stock_zarr(symbols=_STOCK_SYMBOLS, periods=60, seed=0)
    factor = Alpha101Stock(
        _factor_config(
            dataset_config,
            factor_names=["alpha001", "alpha041"],
            data_columns=_ADJUSTED_STOCK_COLUMNS,
            tmp_path=tmp_path,
            dataset_cls=StockDataset,
        )
    )

    result = compute_all(factor)

    assert isinstance(result, xr.Dataset)
    assert "alpha001" in result.data_vars
    assert np.isfinite(result["alpha001"].to_numpy()).sum() > 0
    # alpha041 = sqrt(high * low) - vwap, so it exercises the vwap input.
    assert np.isfinite(result["alpha041"].to_numpy()).sum() > 0


def test_alpha158_stock_batch_compute_returns_xarray_dataset(
    stock_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """D-01 / FACTOR-01 across both markets: the Alpha158 factor set computes
    in batch mode against US equities, not only crypto spot.

    REVIEW WR-02: no stock store carries `amount`, so the old `Input("amount")`
    wiring raised `KeyError: 'amount'` and the class never computed (hidden
    inside the 56-failure baseline). `vwap` is now the adjusted typical price
    `(adjHigh + adjLow + adjClose) / 3`, built inside the graph from the same
    adjusted series as every other input, and every output is z-scored across
    symbols per bar (`CrossSectionalZScore`, ddof=1). The store's `adjHigh`
    is widened by a different factor per symbol so the typical-price ratio
    `vwap / close` varies across the cross-section: the expected `VWAP0` is
    then the per-bar z-score of that ratio, which pins both the typical-price
    construction (a `close * volume` proxy would z-score the volume instead)
    and the cross-sectional normalization. The raw columns are rescaled so
    any leak of unadjusted data would show.
    """
    dataset_config = stock_zarr(symbols=_STOCK_SYMBOLS, periods=60, seed=0)
    store = xr.open_zarr(dataset_config.zarr_file_path).load()
    widen = xr.DataArray(
        1.05 + 0.02 * np.arange(store.sizes["symbol"]),
        dims=("symbol",),
        coords={"symbol": store["symbol"]},
    )
    store["adjHigh"] = store["adjClose"] * widen
    for col in ("open", "high", "low", "close"):
        store[col] = store[col] * 4.0
    store["volume"] = store["volume"] / 4.0
    store.to_zarr(dataset_config.zarr_file_path, mode="w")

    factor = Alpha158Stock(
        _factor_config(
            dataset_config,
            factor_names=["KMID", "VOLUME0", "STD5", "VWAP0"],
            data_columns=list(_ADJUSTED_STOCK_COLUMNS),
            tmp_path=tmp_path,
            dataset_cls=StockDataset,
        )
    )

    result = compute_all(factor)

    assert isinstance(result, xr.Dataset)
    assert dict(result.sizes) == {"timestamp": 60, "symbol": 8}
    assert sorted(result.data_vars) == ["KMID", "STD5", "VOLUME0", "VWAP0"]
    assert np.isfinite(result["KMID"].to_numpy()).sum() > 0

    typical = (store["adjHigh"] + store["adjLow"] + store["adjClose"]) / 3.0
    ratio = (typical / store["adjClose"]).sel(
        timestamp=result["timestamp"], symbol=result["symbol"]
    )
    expected = (ratio - ratio.mean("symbol")) / ratio.std("symbol", ddof=1)
    actual = result["VWAP0"].to_numpy()
    finite = np.isfinite(actual)
    assert finite.sum() > 0
    np.testing.assert_allclose(
        actual[finite], expected.to_numpy()[finite], rtol=1e-4, atol=1e-5
    )
    # The z-score of a constant would be NaN everywhere; the widened
    # cross-section keeps the assertion above from passing vacuously.
    assert np.abs(actual[finite]).max() > 0.5


def test_alpha158_stock_graph_reads_only_adjusted_inputs(
    stock_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """REVIEW WR-02: the compiled Alpha158Stock graph declares exactly the five
    adjusted columns -- no `amount` (which no stock store has) and no raw
    `open/high/low/close/volume` dead inputs suggesting unadjusted data is
    consumed. `Function` prunes inputs no requested output reaches, so the
    factors are chosen to touch all five: KMID (open, close), VWAP0 (high,
    low, close) and VOLUME0 (volume)."""
    dataset_config = stock_zarr(symbols=_STOCK_SYMBOLS, periods=60, seed=0)
    factor = Alpha158Stock(
        _factor_config(
            dataset_config,
            factor_names=["KMID", "VWAP0", "VOLUME0"],
            data_columns=list(_ADJUSTED_STOCK_COLUMNS),
            tmp_path=tmp_path,
            dataset_cls=StockDataset,
        )
    )

    func = factor._get_factor_func()

    names = {op.attrs["name"] for op in func.ops if isinstance(op, Input)}
    assert names == set(_ADJUSTED_STOCK_COLUMNS)


def test_graph_returns_one_output_per_factor_name_without_reading_data(
    tmp_path: Path,
) -> None:
    dataset = StockDataset(DatasetConfig(
        zarr_file_path=str(tmp_path / "missing.zarr"),
        raw_data_dir_path=str(tmp_path), market="us_equity", frequency="1d",
    ))
    factor = Alpha158Stock(FactorConfig(
        warmup_bars=60, dataset=dataset, mode="batch",
        data_columns=list(_ADJUSTED_STOCK_COLUMNS), factor_names=["KMID", "STD5"],
    ))

    outputs = [op for op in factor.graph().ops if isinstance(op, Output)]

    assert [op.attrs["name"] for op in outputs] == ["KMID", "STD5"]
    assert factor.graph() is not factor.graph()


# The method that actually emits each class's `Output(...)` calls. For the
# Alpha101 classes that is `_get_factor_func`; the Alpha158 classes delegate
# `_get_factor_func` to `_get_func_stream`, so the emission lives there.
_EMITTING_METHOD = {
    Alpha101SpotKline: "_get_factor_func",
    Alpha101Stock: "_get_factor_func",
    Alpha158SpotKline: "_get_func_stream",
    Alpha158Stock: "_get_func_stream",
}

# NORM-01 / D-09: time-series z-score on both crypto-spot classes,
# cross-sectional z-score on both US-equity classes. The value names the op
# that wraps every `Output(...)` in that class.
_EXPECTED_NORMALIZATION_MATRIX = {
    Alpha101SpotKline: "WindowedZScore",
    Alpha101Stock: "CrossSectionalZScore",
    Alpha158SpotKline: "WindowedZScore",
    Alpha158Stock: "CrossSectionalZScore",
}

_NORMALIZATION_MISMATCH_MESSAGE = """
NORM-01 / D-09 violation: the normalization applied by one or more factor
classes no longer matches the locked matrix.

That matrix is NOT an implementation detail. `WindowedZScore` is a 时序 /
time-series normalization (each symbol against its own rolling window). Crypto
spot is traded with time-series strategies and gets it; US equities are traded
with 截面 / cross-sectional strategies and get `CrossSectionalZScore`
(normalize across symbols at each timestamp). The split is per market, not
per factor family.

So a mismatch here almost always means someone "aligned" a US-equity class with
its crypto-spot sibling (or the reverse), silently imposing the wrong
normalization on a factor set. If the change really is intended, it is a
change to a USER DECISION: update the four class docstrings and
`docs/user-guide/factors.md` (both languages) first, and only then this literal.
""".strip()


def test_normalization_matrix_matches_recorded_strategy_types() -> None:
    """NORM-01 / D-09: lock the four-class normalization matrix -- rolling
    time-series z-score on `Alpha101SpotKline`/`Alpha158SpotKline` (crypto
    spot, 时序 strategies), cross-sectional z-score on
    `Alpha101Stock`/`Alpha158Stock` (US equities, 截面 strategies).

    Each emitting method must name exactly one of the two ops: a class that
    names both, or neither, fails as loudly as one that names the wrong one.
    """
    ops = ("WindowedZScore", "CrossSectionalZScore")
    actual = {}
    for cls, method_name in _EMITTING_METHOD.items():
        source = inspect.getsource(getattr(cls, method_name))
        found = [op for op in ops if op in source]
        actual[cls] = found[0] if len(found) == 1 else found

    assert actual == _EXPECTED_NORMALIZATION_MATRIX, (
        f"{_NORMALIZATION_MISMATCH_MESSAGE}\n\n"
        f"expected: { {c.__name__: v for c, v in _EXPECTED_NORMALIZATION_MATRIX.items()} }\n"
        f"actual:   { {c.__name__: v for c, v in actual.items()} }"
    )


# -- SIMD padding of the symbol axis (macOS only) --------------------------------


def test_batch_compute_pads_the_symbol_axis_on_macos(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path, monkeypatch
) -> None:
    """On macOS a symbol count that is not a SIMD block multiple still runs:
    ``FactorKunQuant.cal`` appends all-NaN dummy symbols and cuts them back,
    so the stored panel has exactly the real symbols. Elsewhere nothing is
    padded, by decision."""
    import sys

    from quantlab.factor.kunquant import FactorKunQuant

    monkeypatch.setattr(sys, "platform", "linux")
    assert FactorKunQuant._symbol_padding(5) == 0
    monkeypatch.setattr(sys, "platform", "darwin")
    assert FactorKunQuant._symbol_padding(5) == 3
    assert FactorKunQuant._symbol_padding(16) == 0
    monkeypatch.undo()

    if sys.platform != "darwin":
        pytest.skip("padding is a macOS-only behaviour")
    dataset_config = spot_kline_zarr(symbols=[f"S{i}USDT" for i in range(5)], periods=40)
    factor = Alpha158SpotKline(
        _factor_config(
            dataset_config,
            factor_names=["KMID", "VOLUME0"],
            data_columns=["open", "close", "volume"],
            tmp_path=tmp_path,
        )
    )
    result = compute_all(factor)
    assert dict(result.sizes) == {"timestamp": 40, "symbol": 5}
    assert list(result["symbol"].values) == [f"S{i}USDT" for i in range(5)]
    assert np.isfinite(result["KMID"].to_numpy()).sum() > 0
