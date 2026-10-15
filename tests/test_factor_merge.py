"""A factor given a list of datasets merges them into one panel.

Each input is renamed to the shared variable names with its own column
mapping, then the inputs are outer-joined on timestamp and symbol, NaN where
an input has no value. A cell holding a value in two inputs, or inputs with
different bar spacing, is an error. The merged view is itself a dataset, and
a factor's ``config.json`` records the list and rebuilds from it.
"""

import json
import warnings
from pathlib import Path

import KunQuant.ops as op
import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.polars import FactorPolars
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.label.predefined.fret import Return
from quantlab.core.component import rebuild

SPOT_TO_SHARED = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Quote asset volume": "amount",
}


class MaDeviation(FactorKunQuant):
    """Close over its 5-bar average, minus one."""

    def _get_factor_names(self):
        return ("ma_dev_5",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev_5")
        return Function(builder.ops)


class Spread(FactorPolars):
    """High minus low over close, on the shared lowercase names."""

    def _get_factor_lazyframe(self, lf):
        return lf.with_columns(
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("spread")
        ).select(["timestamp", "symbol", "spread"])


def _write(panel: xr.Dataset, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_zarr(path, mode="w")
    return str(path)


def _spot(path: str) -> SpotKlineDataset:
    return SpotKlineDataset(
        DatasetConfig(
            raw_data_dir_path=str(Path(path).parent / "raw"),
            zarr_file_path=path,
            market="crypto_spot",
            frequency="1d",
        )
    )


def _stock(path: str) -> StockDataset:
    return StockDataset(
        DatasetConfig(
            raw_data_dir_path=str(Path(path).parent / "raw"),
            zarr_file_path=path,
            market="us_equity",
            frequency="1d",
        )
    )


@pytest.fixture
def source(spot_kline_zarr) -> xr.Dataset:
    """60 daily Binance-named bars over eight symbols, loaded."""
    return xr.open_zarr(spot_kline_zarr(periods=60).zarr_file_path).load()


@pytest.fixture
def symbol_halves(source, tmp_path):
    """Two spot stores with the same variables over disjoint symbols."""
    symbols = source["symbol"].values.tolist()
    left = _spot(_write(source.sel(symbol=symbols[:4]), tmp_path / "left.zarr"))
    right = _spot(_write(source.sel(symbol=symbols[4:]), tmp_path / "right.zarr"))
    return left, right


@pytest.fixture
def cross_class(source, tmp_path):
    """Half the symbols in a spot store, half in a stock store, and the hand merge.

    The stock store holds the shared lowercase names already; the spot store
    holds Binance's names, which its column mapping renames.
    """
    symbols = source["symbol"].values.tolist()
    lowercase = source.rename(SPOT_TO_SHARED)
    spot = _spot(_write(source.sel(symbol=symbols[:4]), tmp_path / "spot.zarr"))
    stock = _stock(_write(lowercase.sel(symbol=symbols[4:]), tmp_path / "stock.zarr"))
    whole = _stock(_write(lowercase, tmp_path / "whole.zarr"))
    return spot, stock, whole


def _ma_dev(dataset, tmp_path: Path, mode: str = "batch") -> MaDeviation:
    return MaDeviation(
        FactorConfig(
            warmup_bars=5,
            dataset=dataset,
            mode=mode,
            data_columns=("close",),
            file_path=str(tmp_path / "factors" / "ma_dev.zarr"),
            njobs=2,
        )
    )


def _quiet_compute(factor, start="2024-01-01", end="2024-12-31") -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return factor.compute(start, end).load()


# -- config ------------------------------------------------------------------------


def test_a_factor_config_accepts_one_dataset_or_a_list(symbol_halves, tmp_path):
    left, right = symbol_halves

    single = _ma_dev(left, tmp_path)
    merged = _ma_dev([left, right], tmp_path)

    assert single.config.dataset == left
    assert isinstance(merged.config.dataset, MergedDataset)
    assert merged.config.dataset.config.datasets == (left, right)


def test_config_json_records_the_list_and_rebuilds_an_equal_factor(
    symbol_halves, tmp_path
):
    left, right = symbol_halves
    factor = _ma_dev([left, right], tmp_path)

    saved = json.loads(json.dumps(factor.get_config()))
    rebuilt = rebuild(saved)

    assert [d["zarr_file_path"] for d in saved["dataset"]["datasets"]] == [
        left.config.zarr_file_path,
        right.config.zarr_file_path,
    ]
    assert rebuilt == factor
    xr.testing.assert_allclose(_quiet_compute(rebuilt), _quiet_compute(factor))


# -- merging -----------------------------------------------------------------------


def test_same_variables_over_disjoint_symbols_merge_into_one_symbol_axis(
    symbol_halves, source
):
    merged = MergedDataset([*symbol_halves])

    panel = merged.panel("2024-01-01", "2024-02-29")

    assert sorted(panel["symbol"].values.tolist()) == sorted(
        source["symbol"].values.tolist()
    )
    assert sorted(panel.data_vars) == sorted(SPOT_TO_SHARED.values())
    expected = source.rename(SPOT_TO_SHARED)["close"].sel(symbol=panel["symbol"])
    np.testing.assert_allclose(panel["close"].values, expected.values)


def test_a_merged_dataset_counts_bars_after_a_date_on_the_union_calendar(
    symbol_halves,
):
    merged = MergedDataset([*symbol_halves])

    assert merged.bar_after("2024-01-10", 3) == pd.Timestamp("2024-01-13")
    assert merged.bar_after("2024-02-27", 5) == pd.Timestamp("2024-02-29")


def test_same_symbols_with_different_variables_merge_into_one_set_of_variables(
    source, tmp_path
):
    prices = _spot(_write(source[["Open", "High", "Low", "Close"]], tmp_path / "p.zarr"))
    quotes = _spot(
        _write(source[["Volume", "Quote asset volume"]], tmp_path / "q.zarr")
    )

    panel = MergedDataset([prices, quotes]).panel("2024-01-01", "2024-02-29")

    assert sorted(panel.data_vars) == sorted(SPOT_TO_SHARED.values())
    assert panel.sizes == {"timestamp": 60, "symbol": 8}
    np.testing.assert_allclose(
        panel["amount"].values, source["Quote asset volume"].values
    )


def test_inputs_of_different_classes_are_renamed_to_the_shared_names(cross_class):
    spot, stock, whole = cross_class

    panel = MergedDataset([spot, stock]).panel("2024-01-01", "2024-02-29")
    expected = whole.panel("2024-01-01", "2024-02-29")

    assert sorted(panel.data_vars) == sorted(expected.data_vars)
    xr.testing.assert_allclose(
        panel.sel(symbol=expected["symbol"]).load(), expected.load()
    )


def test_missing_cells_are_nan_over_the_union_of_timestamps_and_symbols(
    source, tmp_path
):
    symbols = source["symbol"].values.tolist()
    early = source.sel(symbol=symbols[:4], timestamp=slice("2024-01-01", "2024-01-20"))
    late = source.sel(symbol=symbols[4:], timestamp=slice("2024-01-11", "2024-01-31"))
    merged = MergedDataset(
        [_spot(_write(early, tmp_path / "e.zarr")), _spot(_write(late, tmp_path / "l.zarr"))]
    )

    panel = merged.panel("2024-01-01", "2024-01-31")

    assert panel.sizes == {"timestamp": 31, "symbol": 8}
    close = panel["close"]
    assert close.sel(symbol=symbols[:4], timestamp=slice("2024-01-21", None)).isnull().all()
    assert close.sel(symbol=symbols[4:], timestamp=slice(None, "2024-01-10")).isnull().all()
    assert close.sel(symbol=symbols[:4], timestamp=slice(None, "2024-01-20")).notnull().all()


def test_a_cell_with_a_value_in_two_inputs_is_an_error(source, tmp_path):
    symbols = source["symbol"].values.tolist()
    first = _spot(_write(source.sel(symbol=symbols[:5]), tmp_path / "first.zarr"))
    second = _spot(_write(source.sel(symbol=symbols[4:]), tmp_path / "second.zarr"))

    with pytest.raises(ValueError, match="close") as raised:
        MergedDataset([first, second]).panel("2024-01-01", "2024-02-29")

    assert "first.zarr" in str(raised.value) and "second.zarr" in str(raised.value)


def test_inputs_with_different_bar_spacing_are_an_error(source, tmp_path):
    symbols = source["symbol"].values.tolist()
    daily = _spot(_write(source.sel(symbol=symbols[:4]), tmp_path / "daily.zarr"))
    hourly_panel = source.sel(symbol=symbols[4:]).isel(timestamp=slice(0, 48))
    hourly_panel = hourly_panel.assign_coords(
        timestamp=np.arange(
            np.datetime64("2024-01-01T00"), np.datetime64("2024-01-03T00"), np.timedelta64(1, "h")
        )
    )
    hourly = _spot(_write(hourly_panel, tmp_path / "hourly.zarr"))

    with pytest.raises(ValueError, match="bar spacing"):
        MergedDataset([daily, hourly]).panel("2024-01-01", "2024-01-31")


# -- factors over a merged input -----------------------------------------------------


def test_compute_on_a_merged_factor_matches_a_hand_merged_store(cross_class, tmp_path):
    spot, stock, whole = cross_class

    merged = _quiet_compute(_ma_dev([spot, stock], tmp_path / "m"))
    single = _quiet_compute(_ma_dev(whole, tmp_path / "s"))

    xr.testing.assert_allclose(merged.sel(symbol=single["symbol"]), single)


def test_warm_up_counts_bars_on_the_merged_calendar(cross_class, tmp_path):
    spot, stock, whole = cross_class
    factor = _ma_dev([spot, stock], tmp_path)

    panel = factor.compute("2024-01-20", "2024-01-31")

    assert panel.sizes["timestamp"] == 12
    assert not np.isnan(panel["ma_dev_5"].values).any()


def test_polars_factors_accept_a_merged_input(cross_class, tmp_path):
    spot, stock, whole = cross_class

    def spread(dataset, name):
        return Spread(
            PolarsFactorConfig(
                warmup_bars=0,
                dataset=dataset,
                file_path=str(tmp_path / name / "spread.zarr"),
            )
        )

    merged = spread([spot, stock], "m").compute("2024-01-01", "2024-02-29").load()
    single = spread(whole, "s").compute("2024-01-01", "2024-02-29").load()

    assert merged.sizes == {"timestamp": 60, "symbol": 8}
    xr.testing.assert_allclose(
        merged.sel(symbol=single["symbol"]), single
    )


def test_stream_mode_refuses_a_merged_input(symbol_halves, tmp_path):
    with pytest.raises(ValueError, match="stream"):
        _ma_dev(list(symbol_halves), tmp_path, mode="stream")


# -- the merged view is a dataset --------------------------------------------------


def test_the_merged_view_can_be_used_wherever_a_dataset_is_expected(
    source, tmp_path
):
    symbols = source["symbol"].values.tolist()
    shared = source.rename(SPOT_TO_SHARED)
    shared = shared.assign(adjOpen=shared["open"])
    left = _stock(_write(shared.sel(symbol=symbols[:4]), tmp_path / "left.zarr"))
    right = _stock(_write(shared.sel(symbol=symbols[4:]), tmp_path / "right.zarr"))
    whole = _stock(_write(shared, tmp_path / "whole.zarr"))
    merged = MergedDataset([left, right])

    rebuilt = rebuild(json.loads(json.dumps(merged.get_config())))
    assert rebuilt == merged
    assert merged.bar_before("2024-01-10", 3) == whole.bar_before("2024-01-10", 3)

    def fret(dataset, name):
        return Return(
            FactorConfig(
                warmup_bars=0,
                dataset=dataset,
                mode="batch",
                data_columns=("adjOpen",),
                kwargs={"n_forward_periods": 1},
                file_path=str(tmp_path / name / "ret.zarr"),
                njobs=2,
            )
        )

    labels = fret(merged, "m").compute("2024-01-01", "2024-02-29").load()
    expected = fret(whole, "s").compute("2024-01-01", "2024-02-29").load()
    xr.testing.assert_allclose(labels.sel(symbol=expected["symbol"]), expected)


def test_a_merged_factor_resamples_onto_the_bars_its_inputs_cut(source, tmp_path):
    symbols = source["symbol"].values.tolist()
    hourly = source.assign_coords(
        timestamp=pd.date_range("2024-01-01", periods=60, freq="h")
    ).rename(SPOT_TO_SHARED)
    left = _stock(_write(hourly.sel(symbol=symbols[:4]), tmp_path / "l.zarr"))
    right = _stock(_write(hourly.sel(symbol=symbols[4:]), tmp_path / "r.zarr"))
    whole = _stock(_write(hourly, tmp_path / "w.zarr"))

    merged = _ma_dev([left, right], tmp_path / "m").resample("1d", "last")
    single = _ma_dev(whole, tmp_path / "s").resample("1d", "last")

    expected = single.compute("2024-01-02", "2024-01-03")
    actual = merged.compute("2024-01-02", "2024-01-03").sel(symbol=expected["symbol"])
    assert expected.sizes["timestamp"] == 2
    xr.testing.assert_allclose(actual, expected)


def test_a_merged_dataset_stores_the_union_of_its_inputs_symbols(symbol_halves):
    merged = MergedDataset([*symbol_halves])

    assert merged.stored_symbols() == merged.panel("2024-01-01", "2024-02-29")["symbol"].values.tolist()


def test_a_factor_on_a_merged_dataset_is_planned_in_chunks(symbol_halves, tmp_path):
    from quantlab.factor.chunking import ChunkedBuilder

    factor = _ma_dev(MergedDataset([*symbol_halves]), tmp_path)

    granularity, chunks = ChunkedBuilder(factor).plan("2024-01-10", "2024-02-29")

    assert chunks
