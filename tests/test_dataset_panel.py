"""Datasets answer panel requests by date range, and count bars before a date.

``panel(start, end, symbols)`` opens the store lazily on every call and holds
nothing afterwards, so one dataset object can feed several consumers with
different ranges. ``bar_before(date, n)`` answers which bar lies ``n`` bars
before ``date`` on the dataset's own calendar.
"""

import copy
import dataclasses
import inspect
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset

MARKET_DATASETS = [
    pytest.param((SpotKlineDataset, "spot_kline_zarr"), id="spot"),
    pytest.param((StockDataset, "stock_zarr"), id="stock"),
]


@pytest.fixture(params=MARKET_DATASETS)
def dataset(request):
    """A dataset over a 60-day synthetic store starting 2024-01-01."""
    cls, fixture = request.param
    return cls(request.getfixturevalue(fixture)())


def _days(panel: xr.Dataset) -> list[str]:
    """Return the panel's timestamps as ISO day strings."""
    return [str(d) for d in panel["timestamp"].values.astype("datetime64[D]")]


def test_panel_covers_exactly_the_inclusive_range(dataset):
    panel = dataset.panel("2024-01-03", "2024-01-07")

    assert tuple(panel.dims) == ("timestamp", "symbol")
    assert _days(panel) == [f"2024-01-0{d}" for d in range(3, 8)]
    assert panel.sizes["symbol"] == dataset.panel("2024-01-01", "2024-03-01").sizes["symbol"]


def test_panel_restricts_to_the_requested_symbols(dataset):
    every = dataset.panel("2024-01-03", "2024-01-07")["symbol"].values.tolist()
    wanted = [every[-1], every[0]]

    panel = dataset.panel("2024-01-03", "2024-01-07", symbols=wanted)

    assert panel["symbol"].values.tolist() == wanted


def test_panel_is_lazy(dataset):
    panel = dataset.panel("2024-01-03", "2024-01-07")

    assert not any(v.variable._in_memory for v in panel.data_vars.values())


def test_repeated_requests_return_their_own_range_in_either_order(dataset):
    narrow_first = dataset.panel("2024-01-10", "2024-01-12")
    wide_after = dataset.panel("2024-01-01", "2024-01-31")
    narrow_again = dataset.panel("2024-01-10", "2024-01-12")

    assert _days(narrow_first) == _days(narrow_again)
    assert len(_days(narrow_first)) == 3
    assert len(_days(wide_after)) == 31
    xr.testing.assert_identical(narrow_first.load(), narrow_again.load())


def test_dataset_holds_no_panel_after_a_request(dataset):
    dataset.panel("2024-01-03", "2024-01-07")

    with pytest.raises(AttributeError):
        dataset.get_xarray_dataset()


def test_the_panel_request_is_the_only_read_path(dataset):
    """No read narrows a dataset to its config dates any more; a range is an argument."""
    assert not hasattr(dataset, "read")


def test_a_rewritten_store_is_seen_by_the_next_request(dataset):
    before = dataset.panel("2024-01-01", "2024-03-01").load()
    shorter = before.isel(timestamp=slice(0, 10))
    shorter.to_zarr(dataset.config.zarr_file_path, mode="w")

    after = dataset.panel("2024-01-01", "2024-03-01")

    assert after.sizes["timestamp"] == 10


def test_the_zarr_backend_opens_the_store_on_every_read(tmp_path):
    store = str(tmp_path / "panel.zarr")
    xr.Dataset({"v": ("timestamp", [1.0, 2.0])}, coords={"timestamp": [0, 1]}).to_zarr(store)
    backend = XrBackend().read(store)
    xr.Dataset({"v": ("timestamp", [5.0])}, coords={"timestamp": [0]}).to_zarr(store, mode="w")

    assert backend.read(store).data["v"].values.tolist() == [5.0]
    assert "overwrite" not in inspect.signature(XrBackend.read).parameters


def test_a_request_leaves_the_config_unchanged(dataset):
    before = copy.deepcopy(dataset.config)

    dataset.panel("2024-01-03", "2024-01-07", symbols=None)

    assert dataset.config == before


def test_panel_refuses_a_start_after_the_end(dataset):
    with pytest.raises(ValueError, match="after"):
        dataset.panel("2024-01-07", "2024-01-03")


def test_panel_refuses_an_unknown_symbol(dataset):
    with pytest.raises(KeyError):
        dataset.panel("2024-01-03", "2024-01-07", symbols=["NOPE"])


# -- bar arithmetic ---------------------------------------------------------------


def test_bar_before_counts_bars_on_the_dataset_calendar(dataset):
    assert dataset.bar_before("2024-01-10", 3) == pd.Timestamp("2024-01-07")
    assert dataset.bar_before("2024-01-10", 9) == pd.Timestamp("2024-01-01")


def test_bar_before_zero_is_the_date_itself(dataset):
    assert dataset.bar_before("2024-01-10", 0) == pd.Timestamp("2024-01-10")


def test_bar_before_skips_calendar_gaps(tmp_path):
    days = pd.bdate_range("2024-01-01", periods=10)  # Mon 1st .. Fri 12th
    store = tmp_path / "klines.zarr"
    xr.Dataset(
        {"Close": (["timestamp", "symbol"], np.ones((len(days), 1)))},
        coords={"timestamp": days, "symbol": ["AUSDT"]},
    ).to_zarr(store, mode="w")
    ds = SpotKlineDataset(
        DatasetConfig(
            raw_data_dir_path=str(tmp_path / "raw"),
            zarr_file_path=str(store),
            market="crypto_spot",
            frequency="1d",
        )
    )

    # Monday 8th: one bar back is Friday 5th, not Sunday 7th.
    assert ds.bar_before("2024-01-08", 1) == pd.Timestamp("2024-01-05")


def test_bar_before_reports_too_few_bars(dataset):
    with pytest.raises(ValueError, match=r"only 9 bar"):
        dataset.bar_before("2024-01-10", 10)


def test_bar_before_refuses_a_negative_count(dataset):
    with pytest.raises(ValueError, match="non-negative"):
        dataset.bar_before("2024-01-10", -1)


def test_bar_after_counts_bars_on_the_dataset_calendar(dataset):
    assert dataset.bar_after("2024-01-10", 1) == pd.Timestamp("2024-01-11")
    assert dataset.bar_after("2024-01-10", 3) == pd.Timestamp("2024-01-13")


def test_bar_after_zero_is_the_date_itself(dataset):
    assert dataset.bar_after("2024-01-10", 0) == pd.Timestamp("2024-01-10")


def test_bar_after_stops_at_the_last_bar(dataset):
    # The store's last bar is 2024-02-29; five bars past the 27th do not exist.
    assert dataset.bar_after("2024-02-27", 5) == pd.Timestamp("2024-02-29")


def test_bar_after_the_last_bar_is_the_date_itself(dataset):
    assert dataset.bar_after("2024-03-05", 2) == pd.Timestamp("2024-03-05")


def test_bar_after_refuses_a_negative_count(dataset):
    with pytest.raises(ValueError, match="non-negative"):
        dataset.bar_after("2024-01-10", -1)


def test_bar_after_skips_calendar_gaps(tmp_path):
    days = pd.bdate_range("2024-01-01", periods=10)  # Mon 1st .. Fri 12th
    store = tmp_path / "klines.zarr"
    xr.Dataset(
        {"Close": (["timestamp", "symbol"], np.ones((len(days), 1)))},
        coords={"timestamp": days, "symbol": ["AUSDT"]},
    ).to_zarr(store, mode="w")
    ds = SpotKlineDataset(
        DatasetConfig(
            raw_data_dir_path=str(tmp_path / "raw"),
            zarr_file_path=str(store),
            market="crypto_spot",
            frequency="1d",
        )
    )

    # Friday 5th: one bar on is Monday 8th, not Saturday 6th.
    assert ds.bar_after("2024-01-05", 1) == pd.Timestamp("2024-01-08")


# -- minute bars ------------------------------------------------------------------

#: The minute store's variables.
VARIABLES = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
    "Quote asset volume": "sum",
}


@pytest.fixture
def minute_dataset(tmp_path: Path) -> SpotKlineDataset:
    """Five days of four minute bars each; ``Close`` counts bars from 1."""
    timestamps = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"2024-01-0{d} 00:00", periods=4, freq="min").values
                for d in range(1, 6)
            ]
        )
    )
    close = np.arange(1.0, len(timestamps) + 1.0)[:, None]
    store = tmp_path / "klines.zarr"
    xr.Dataset(
        {name: (["timestamp", "symbol"], close) for name in VARIABLES},
        coords={"timestamp": timestamps, "symbol": ["AUSDT"]},
    ).to_zarr(store, mode="w")
    return SpotKlineDataset(
        DatasetConfig(
            raw_data_dir_path=str(tmp_path / "raw"),
            zarr_file_path=str(store),
            market="crypto_spot",
            frequency="1m",
        )
    )


def test_intraday_start_inside_a_date_only_end_day_is_accepted(minute_dataset):
    panel = minute_dataset.panel("2024-01-04 00:02", "2024-01-04")

    assert panel["Close"].values[:, 0].tolist() == [15.0, 16.0]


def test_panel_selects_integer_symbol_labels(tmp_path):
    """CRSP and NBBO stores key symbols by integer PERMNO."""
    days = pd.date_range("2024-01-01", periods=5, freq="D")
    store = tmp_path / "permno.zarr"
    xr.Dataset(
        {"Close": (["timestamp", "symbol"], np.arange(10.0).reshape(5, 2))},
        coords={"timestamp": days, "symbol": np.array([10107, 14593], dtype="int64")},
    ).to_zarr(store, mode="w")
    ds = SpotKlineDataset(
        DatasetConfig(
            raw_data_dir_path=str(tmp_path / "raw"),
            zarr_file_path=str(store),
            market="us_equity",
            frequency="1d",
        )
    )

    panel = ds.panel("2024-01-02", "2024-01-03", symbols=[14593])

    assert panel["symbol"].values.tolist() == [14593]
    assert panel["Close"].values[:, 0].tolist() == [3.0, 5.0]


# -- the read path goes through the dataset's own backend --------------------------


class PickleBackend(XrBackend):
    """Keeps a panel in one pickle file, which the Zarr backend cannot open."""

    def read(self, path: str, **kwargs):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        return self

    def write(self, path: str, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.data.load(), f)
        return self


class PickleStockDataset(StockDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_backend = PickleBackend()


@pytest.fixture
def pickled(stock_zarr, tmp_path):
    """A dataset over a pickle store holding the 60-day synthetic stock panel."""
    config = stock_zarr()
    store = tmp_path / "stock.pkl"
    PickleBackend().to_internal(xr.open_zarr(config.zarr_file_path).load()).write(str(store))
    return PickleStockDataset(dataclasses.replace(config, zarr_file_path=str(store)))


def test_panel_and_bar_before_read_through_the_dataset_backend(pickled):
    panel = pickled.panel("2024-01-03", "2024-01-07")

    assert _days(panel) == [f"2024-01-0{d}" for d in range(3, 8)]
    assert pickled.bar_before("2024-01-10", 2) == pd.Timestamp("2024-01-08")
    with pytest.raises(AttributeError):
        pickled.get_xarray_dataset()
