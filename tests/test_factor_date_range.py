"""Factors answer ``read(start, end)`` and ``compute(start, end)`` by date range.

``compute`` reads the factor's inputs from ``warmup_bars`` bars before
``start``, counted on the input's own calendar, and returns only the
requested range. ``build(start, end)`` writes a store and records the range
it covers beside it; ``extend(end)`` appends later bars and moves the
recorded end; ``read(start, end)`` returns a covered range from the store.
None of these calls changes the factor's config or its dataset's config.
"""

import copy
import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.predefined.momentum import Momentum


class MaDeviation(FactorKunQuant):
    """Close over its 5-bar average, minus one: warm after 4 earlier bars."""

    def _get_factor_names(self):
        return ("ma_dev_5",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev_5")
        return Function(builder.ops)


def _kunquant(dataset_config: DatasetConfig, tmp_path: Path) -> MaDeviation:
    return MaDeviation(
        FactorConfig(
            warmup_bars=5,
            dataset=SpotKlineDataset(dataset_config),
            mode="batch",
            data_columns=("close",),
            file_path=str(tmp_path / "factors" / "ma_dev.zarr"),
            njobs=2,
        )
    )


def _momentum(dataset_config: DatasetConfig, tmp_path: Path) -> Momentum:
    return _momentum_on(SpotKlineDataset(dataset_config), tmp_path)


def _momentum_on(dataset, tmp_path: Path, warmup_bars: int = 5) -> Momentum:
    return Momentum(
        PolarsFactorConfig(
            warmup_bars=warmup_bars,
            dataset=dataset,
            file_path=str(tmp_path / "factors" / "momentum.zarr"),
            kwargs={"n": 5},
        )
    )


@pytest.fixture(params=["kunquant", "polars"])
def factor(request, spot_kline_zarr, tmp_path):
    """A 5-bar factor of either backend over 60 daily bars from 2024-01-01."""
    build = _kunquant if request.param == "kunquant" else _momentum
    return build(spot_kline_zarr(periods=60), tmp_path)


def _days(panel: xr.Dataset) -> list[str]:
    return [str(d) for d in panel["timestamp"].values.astype("datetime64[D]")]


def _full_history(factor) -> xr.Dataset:
    """The factor computed from the first bar of its store, with no warm-up."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return factor.compute("2024-01-01", "2024-12-31").load()


def _assert_same_values(actual: xr.Dataset, expected: xr.Dataset) -> None:
    """Equal up to float32 rounding.

    KunQuant keeps rolling sums in float32, so a window's sum carries the
    rounding of the bars that left it and differs in the last digits with
    where the run started.
    """
    assert list(actual.data_vars) == list(expected.data_vars)
    for name in expected.data_vars:
        np.testing.assert_allclose(
            actual[name].values, expected[name].values, atol=1e-6, equal_nan=True
        )


def _configs(factor) -> tuple[dict, object]:
    """The factor config without its dataset object, and the dataset config."""
    fields = {k: v for k, v in factor.config.to_dict().items() if k != "dataset"}
    return copy.deepcopy(fields), copy.deepcopy(factor.config.dataset.config)


# -- config ------------------------------------------------------------------------


def test_the_factor_config_describes_what_is_computed_not_when():
    fields = {f.name for f in dataclasses.fields(PolarsFactorConfig)}

    assert "warmup_bars" in fields
    assert not fields & {"start_date", "end_date", "symbols", "window"}


def test_constructing_a_factor_leaves_its_dataset_config_unchanged(
    spot_kline_zarr, tmp_path
):
    dataset = SpotKlineDataset(spot_kline_zarr(periods=60))
    before = copy.deepcopy(dataset.config)

    _momentum_on(dataset, tmp_path)

    assert dataset.config == before


def test_one_dataset_object_feeds_two_factors_with_different_warm_ups(
    spot_kline_zarr, tmp_path
):
    dataset = SpotKlineDataset(spot_kline_zarr(periods=60))
    short = _momentum_on(dataset, tmp_path / "short", warmup_bars=5)
    long = _momentum_on(dataset, tmp_path / "long", warmup_bars=20)

    long_panel = long.compute("2024-02-01", "2024-02-10")
    short_panel = short.compute("2024-02-01", "2024-02-10")

    assert _days(long_panel) == _days(short_panel)
    _assert_same_values(long_panel, short_panel)


# -- compute -----------------------------------------------------------------------


def test_compute_is_warm_on_the_first_requested_bar(factor):
    full = _full_history(factor)

    panel = factor.compute("2024-01-20", "2024-01-31")

    first = panel.isel(timestamp=0)
    assert not np.isnan(first[factor.get_factor_names()[0]].values).any()
    _assert_same_values(panel, full.sel(timestamp=slice("2024-01-20", "2024-01-31")))


def test_compute_returns_only_the_requested_range(factor):
    panel = factor.compute("2024-01-20", "2024-01-31")

    assert tuple(panel.dims) == ("timestamp", "symbol")
    assert _days(panel) == [f"2024-01-{d}" for d in range(20, 32)]


def test_warm_up_is_counted_in_bars_on_the_input_calendar(spot_kline_zarr, tmp_path):
    """Five bars back from a Monday reach the previous Monday on weekdays only.

    Five calendar days would reach Wednesday and leave the first bar cold.
    """
    config = spot_kline_zarr(periods=60)
    source = xr.open_zarr(config.zarr_file_path).load()
    weekdays = source.sel(timestamp=source["timestamp"].dt.dayofweek < 5)
    store = tmp_path / "weekdays.zarr"
    weekdays.to_zarr(store, mode="w")
    factor = _kunquant(
        dataclasses.replace(config, zarr_file_path=str(store)), tmp_path
    )

    panel = factor.compute("2024-01-15", "2024-01-19")  # Monday .. Friday

    _assert_same_values(
        panel, _full_history(factor).sel(timestamp=slice("2024-01-15", "2024-01-19"))
    )
    assert not np.isnan(panel["ma_dev_5"].values).any()


def test_too_little_history_warns_with_the_shortfall_in_bars(factor):
    with pytest.warns(UserWarning, match=r"short by 3 bar"):
        panel = factor.compute("2024-01-03", "2024-01-10")

    assert _days(panel)[0] == "2024-01-03"


def test_compute_leaves_both_configs_unchanged(factor):
    before = _configs(factor)

    factor.compute("2024-01-20", "2024-01-31")

    assert _configs(factor) == before


def test_compute_refuses_a_start_after_the_end(factor):
    with pytest.raises(ValueError, match="after end"):
        factor.compute("2024-01-31", "2024-01-20")


# -- build, read, extend ------------------------------------------------------------


def test_build_records_the_range_beside_the_store(factor):
    factor.build("2024-01-10", "2024-01-31")

    assert factor.store_range() == ("2024-01-10", "2024-01-31")
    assert Path(f"{factor.store_path}.range.json").is_file()


def test_read_returns_a_covered_range_from_the_store(factor):
    factor.build("2024-01-10", "2024-01-31")

    panel = factor.read("2024-01-15", "2024-01-20")

    assert _days(panel) == [f"2024-01-{d}" for d in range(15, 21)]
    _assert_same_values(panel, factor.compute("2024-01-15", "2024-01-20"))


def test_read_refuses_a_range_the_store_does_not_cover(factor):
    factor.build("2024-01-10", "2024-01-31")

    with pytest.raises(ValueError, match=r"covers 2024-01-10 to 2024-01-31"):
        factor.read("2024-01-15", "2024-02-05")
    with pytest.raises(ValueError, match=r"covers 2024-01-10 to 2024-01-31"):
        factor.read("2024-01-05", "2024-01-20")


def test_read_refuses_a_store_without_a_recorded_range(factor):
    # A store written some other way than build(), so no range is recorded.
    factor.compute("2024-01-10", "2024-01-31").to_zarr(factor.store_path, mode="w")

    with pytest.raises(ValueError, match="no recorded range"):
        factor.read("2024-01-15", "2024-01-20")


def test_extend_appends_later_bars_and_moves_the_recorded_end(factor):
    factor.build("2024-01-10", "2024-01-31")

    factor.extend("2024-02-15")

    assert factor.store_range() == ("2024-01-10", "2024-02-15")
    panel = factor.read("2024-01-10", "2024-02-15")
    _assert_same_values(panel, factor.compute("2024-01-10", "2024-02-15"))


def test_extend_refuses_an_end_the_store_already_covers(factor):
    factor.build("2024-01-10", "2024-01-31")

    with pytest.raises(ValueError, match="already covers"):
        factor.extend("2024-01-20")


def test_build_read_and_extend_leave_both_configs_unchanged(factor):
    before = _configs(factor)

    factor.build("2024-01-10", "2024-01-31")
    factor.read("2024-01-15", "2024-01-20")
    factor.extend("2024-02-15")

    assert _configs(factor) == before


# -- resampled factors -------------------------------------------------------------


@pytest.fixture
def minute_momentum(tmp_path: Path) -> Momentum:
    """Momentum over six days of four minute bars; ``Close`` rises by one a bar."""
    timestamps = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"2024-01-0{d} 00:00", periods=4, freq="min").values
                for d in range(1, 7)
            ]
        )
    )
    close = np.arange(1.0, len(timestamps) + 1.0)[:, None]
    store = tmp_path / "klines.zarr"
    xr.Dataset(
        {"Close": (["timestamp", "symbol"], close)},
        coords={"timestamp": timestamps, "symbol": ["AUSDT"]},
    ).to_zarr(store, mode="w")
    dataset = SpotKlineDataset(
        DatasetConfig(
            raw_data_dir_path=str(tmp_path / "raw"),
            zarr_file_path=str(store),
            market="crypto_spot",
            frequency="1m",
        )
    )
    return Momentum(
        PolarsFactorConfig(
            warmup_bars=2,
            dataset=dataset,
            file_path=str(tmp_path / "factors" / "momentum.zarr"),
            kwargs={"n": 2},
        )
    )


def test_resampled_compute_is_on_resampled_bars(minute_momentum):
    daily = minute_momentum.resample("1d", "last")
    full = _full_history(daily)

    panel = daily.compute("2024-01-03", "2024-01-05")

    assert _days(panel) == ["2024-01-03", "2024-01-04", "2024-01-05"]
    # The last bar of each day, against the bar two minutes earlier.
    assert panel["momentum_2"].values[:, 0].tolist() == pytest.approx(
        [12 / 10 - 1, 16 / 14 - 1, 20 / 18 - 1]
    )
    _assert_same_values(panel, full.sel(timestamp=slice("2024-01-03", "2024-01-05")))


def test_resampled_build_and_read_use_the_sibling_store(minute_momentum):
    daily = minute_momentum.resample("1d", "last")

    daily.build("2024-01-02", "2024-01-05")

    assert Path(daily.store_path).is_dir()
    assert not Path(minute_momentum.store_path).exists()
    assert daily.store_range() == ("2024-01-02", "2024-01-05")
    _assert_same_values(
        daily.read("2024-01-03", "2024-01-04"), daily.compute("2024-01-03", "2024-01-04")
    )


def test_resampled_read_resamples_the_source_store_without_a_sibling(minute_momentum):
    minute_momentum.build("2024-01-01", "2024-01-06")
    daily = minute_momentum.resample("1d", "last")

    panel = daily.read("2024-01-03", "2024-01-05")

    assert _days(panel) == ["2024-01-03", "2024-01-04", "2024-01-05"]
    _assert_same_values(panel, daily.compute("2024-01-03", "2024-01-05"))


def test_resampled_store_built_in_pieces_equals_one_built_whole(minute_momentum):
    daily = minute_momentum.resample("1d", "last")
    whole = daily.compute("2024-01-02", "2024-01-06")
    daily.build("2024-01-02", "2024-01-04")

    daily.extend("2024-01-06")

    assert daily.store_range() == ("2024-01-02", "2024-01-06")
    _assert_same_values(daily.read("2024-01-02", "2024-01-06"), whole)


def test_resampled_extend_mid_day_does_not_repeat_a_bar(minute_momentum):
    daily = minute_momentum.resample("1d", "last")
    daily.build("2024-01-02", "2024-01-03 12:00")

    daily.extend("2024-01-05")

    panel = daily.read("2024-01-02", "2024-01-05")
    assert _days(panel) == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    _assert_same_values(panel, daily.compute("2024-01-02", "2024-01-05"))


# -- a shared store: its owner and a view pinned to some of its outputs --------------


class ThreeOutputs(FactorKunQuant):
    """Three outputs off the close, ``a``, ``b`` and ``c``, warm on the first bar."""

    def _get_factor_names(self):
        return ("a", "b", "c")

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            for offset, name in enumerate(("a", "b", "c"), start=1):
                Output(op.SubConst(close, float(offset)), name)
        return Function(builder.ops)


def _three_outputs(dataset_config, store: Path, factor_names=None) -> ThreeOutputs:
    return ThreeOutputs(
        FactorConfig(
            warmup_bars=0,
            dataset=SpotKlineDataset(dataset_config),
            mode="batch",
            data_columns=("close",),
            file_path=str(store),
            factor_names=factor_names,
            njobs=2,
        )
    )


@pytest.fixture
def shared_store(spot_kline_zarr, tmp_path):
    """``(owner, view)``: the owner built a store of ``a, b, c``; the view pins ``b``."""
    dataset_config = spot_kline_zarr(periods=20)
    store = tmp_path / "factors" / "three.zarr"
    owner = _three_outputs(dataset_config, store).build("2024-01-01", "2024-01-20")
    return owner, _three_outputs(dataset_config, store, factor_names=("b",))


def test_read_returns_only_the_factors_own_outputs(shared_store):
    owner, view = shared_store

    panel = view.read("2024-01-05", "2024-01-10")

    assert list(panel.data_vars) == ["b"]
    np.testing.assert_array_equal(
        panel["b"].values, owner.read("2024-01-05", "2024-01-10")["b"].values
    )


def test_compute_returns_only_the_factors_own_outputs(shared_store):
    _, view = shared_store

    assert list(view.compute("2024-01-05", "2024-01-10").data_vars) == ["b"]

def test_read_refuses_a_store_missing_one_of_the_factors_outputs(spot_kline_zarr, tmp_path):
    dataset_config = spot_kline_zarr(periods=20)
    store = tmp_path / "factors" / "three.zarr"
    _three_outputs(dataset_config, store, factor_names=("a",)).build("2024-01-01", "2024-01-20")
    wider = _three_outputs(dataset_config, store, factor_names=("a", "c"))

    with pytest.raises(ValueError, match=r"lacks the factor's output\(s\) \['c'\]"):
        wider.read("2024-01-05", "2024-01-10")


def test_read_returns_only_the_requested_symbols(shared_store):
    owner, view = shared_store

    panel = view.read("2024-01-05", "2024-01-10", symbols=["S3USDT", "S1USDT"])

    assert list(panel["symbol"].values) == ["S3USDT", "S1USDT"]
    full = owner.read("2024-01-05", "2024-01-10")
    np.testing.assert_array_equal(
        panel["b"].values, full["b"].sel(symbol=["S3USDT", "S1USDT"]).values
    )


def test_read_refuses_a_symbol_the_store_lacks(shared_store):
    _, view = shared_store

    with pytest.raises(ValueError, match=r"lacks symbol\(s\) \['NOPE'\]"):
        view.read("2024-01-05", "2024-01-10", symbols=["S1USDT", "NOPE"])


def test_a_view_cannot_build_over_the_store_it_shares(shared_store):
    owner, view = shared_store

    with pytest.raises(ValueError, match=r"holds \['a', 'b', 'c'\].*only the factor"):
        view.build("2024-01-01", "2024-01-20")

    assert list(owner.read("2024-01-01", "2024-01-20").data_vars) == ["a", "b", "c"]


def test_a_view_cannot_extend_the_store_it_shares(shared_store):
    owner, view = shared_store
    owner.build("2024-01-01", "2024-01-10")

    with pytest.raises(ValueError, match=r"holds \['a', 'b', 'c'\].*only the factor"):
        view.extend("2024-01-20")

    assert owner.store_range() == ("2024-01-01", "2024-01-10")


def test_a_factor_pinned_to_some_outputs_owns_a_store_of_just_those(spot_kline_zarr, tmp_path):
    dataset_config = spot_kline_zarr(periods=20)
    pinned = _three_outputs(dataset_config, tmp_path / "b.zarr", factor_names=("b",))

    pinned.build("2024-01-01", "2024-01-10").extend("2024-01-20").build("2024-01-01", "2024-01-20")

    assert list(pinned.read("2024-01-01", "2024-01-20").data_vars) == ["b"]


def test_stored_symbols_reads_the_store_axis(shared_store):
    _, view = shared_store

    assert view.stored_symbols() == [f"S{i}USDT" for i in range(8)]


def test_a_resampled_factor_without_its_own_store_reads_the_source_axis(minute_momentum):
    minute_momentum.build("2024-01-02", "2024-01-03")

    daily = minute_momentum.resample("1d", "last")

    assert daily.stored_symbols() == minute_momentum.stored_symbols()

