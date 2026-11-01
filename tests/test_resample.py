"""Resampling a factor panel onto coarser bars.

A *resample* aggregates a ``(timestamp, symbol)`` panel onto a coarser
regular time grid, one aggregation method per variable. ``resample()`` on a
factor returns a copy whose config carries ``resample_freq`` and
``resample_how``; the copy shares no memory with its source, reads its own
store beside the source store when one was saved, and otherwise resamples the
source on read. The backend does the grouping; the dataset decides how bars
are cut (``resample_labels``), and a factor cuts them the way its dataset
does. A dataset is not resampled.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from conftest import compute_all
from quantlab.backend.parquet import PlBackend
from quantlab.backend.zarr import XrBackend
from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.predefined.alpha158 import Alpha158SpotKline
from quantlab.factor.predefined.momentum import Momentum
from quantlab.core.component import rebuild
from quantlab.utils.resample import session_labels

SYMBOLS = ["AAAUSDT", "BBBUSDT"]
BARS_PER_DAY = 6
DAYS = 3
#: The dates the synthetic minute store covers.
STORE_RANGE = ("2024-01-01", "2024-01-03")


def _minute_panel(symbols: list[str] = SYMBOLS) -> xr.Dataset:
    """Three days of six minute bars each, labelled at their open time.

    ``Close`` counts bars from 1.0 so every aggregate is easy to predict;
    ``Volume`` is 1.0 everywhere so a daily ``sum`` is the bar count.
    """
    timestamps = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"2024-01-0{day} 00:00", periods=BARS_PER_DAY, freq="min").values
                for day in range(1, DAYS + 1)
            ]
        )
    )
    n = len(timestamps)
    scale = 10.0 ** np.arange(len(symbols))
    close = np.arange(1.0, n + 1.0)[:, None] * scale[None, :]
    volume = np.ones((n, len(symbols)))
    return xr.Dataset(
        {
            "Open": (["timestamp", "symbol"], close - 0.5),
            "High": (["timestamp", "symbol"], close + 1.0),
            "Low": (["timestamp", "symbol"], close - 1.0),
            "Close": (["timestamp", "symbol"], close),
            "Volume": (["timestamp", "symbol"], volume),
            "Quote asset volume": (["timestamp", "symbol"], volume * close),
        },
        coords={"timestamp": timestamps, "symbol": symbols},
    )


OHLCV_HOW = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
    "Quote asset volume": "sum",
}


def _write_minute_store(tmp_path: Path, symbols: list[str]) -> DatasetConfig:
    spot_dir = tmp_path / "spot"
    spot_dir.mkdir(exist_ok=True)
    zarr_path = spot_dir / "klines.zarr"
    _minute_panel(symbols).to_zarr(zarr_path, mode="w")
    return DatasetConfig(
        raw_data_dir_path=str(spot_dir / "raw"),
        zarr_file_path=str(zarr_path),
        market="crypto_spot",
        frequency="1m",
    )


@pytest.fixture
def minute_config(tmp_path: Path) -> DatasetConfig:
    """A ``SpotKlineDataset`` config over a two-symbol synthetic minute store."""
    return _write_minute_store(tmp_path, SYMBOLS)


@pytest.fixture
def minute_config_8(tmp_path: Path) -> DatasetConfig:
    """The same store with eight symbols, the lane count KunQuant needs."""
    return _write_minute_store(tmp_path, [f"S{i}USDT" for i in range(8)])


# -- backend --------------------------------------------------------------------


def _labels(timestamps) -> pd.Series:
    index = pd.DatetimeIndex(timestamps)
    return pd.Series(index.floor("D").values, index=index)


def test_xr_backend_resample_applies_one_method_per_variable():
    panel = _minute_panel()
    panel["Close"][BARS_PER_DAY - 1, 0] = np.nan  # last bar of day one, first symbol
    backend = XrBackend().to_internal(panel)

    backend.resample(_labels(panel["timestamp"].values), OHLCV_HOW)
    daily = backend.data

    assert tuple(daily["Close"].dims) == ("timestamp", "symbol")
    assert daily.sizes == {"timestamp": DAYS, "symbol": 2}
    assert daily["timestamp"].dtype == np.dtype("datetime64[ns]")
    assert daily["timestamp"].values[0] == np.datetime64("2024-01-01")
    # `last` skips the NaN and takes bar 5; the second symbol keeps bar 6.
    assert daily["Close"].values[0].tolist() == [5.0, 60.0]
    assert daily["Open"].values[0].tolist() == [0.5, 9.5]
    assert daily["High"].values[0].tolist() == [7.0, 61.0]
    assert daily["Low"].values[0].tolist() == [0.0, 9.0]
    assert daily["Volume"].values.tolist() == [[6.0, 6.0]] * DAYS


def test_xr_backend_resample_count_and_mean():
    panel = _minute_panel()[["Close"]]
    panel["Close"][0, 0] = np.nan
    backend = XrBackend().to_internal(panel)
    backend.resample(_labels(panel["timestamp"].values), {"Close": "count"})
    assert backend.data["Close"].values[0].tolist() == [5, 6]

    backend = XrBackend().to_internal(_minute_panel()[["Close"]])
    backend.resample(_labels(panel["timestamp"].values), {"Close": "mean"})
    assert backend.data["Close"].values[0].tolist() == [3.5, 35.0]


def test_xr_backend_resample_an_all_nan_bar():
    panel = _minute_panel()[["Close"]]
    panel["Close"][:BARS_PER_DAY, 0] = np.nan  # every bar of day one, first symbol
    labels = _labels(panel["timestamp"].values)

    for method in ("first", "last", "max", "min", "mean"):
        daily = XrBackend().to_internal(panel).resample(labels, {"Close": method}).data
        assert np.isnan(daily["Close"].values[0, 0]), method
        assert not np.isnan(daily["Close"].values[0, 1]), method
    daily = XrBackend().to_internal(panel).resample(labels, {"Close": "sum"}).data
    assert daily["Close"].values[0].tolist() == [0.0, 210.0]
    daily = XrBackend().to_internal(panel).resample(labels, {"Close": "count"}).data
    assert daily["Close"].values[0].tolist() == [0, 6]
    assert daily["Close"].dtype == np.dtype("int64")


def test_xr_backend_resample_keeps_layout_attrs_and_untimed_variables():
    panel = _minute_panel()[["Close"]]
    panel["Close"].attrs["unit"] = "usd"
    panel["Weight"] = ("symbol", [0.25, 0.75])
    panel["Close_T"] = panel["Close"].transpose("symbol", "timestamp")
    panel.attrs["source"] = "demo"
    labels = _labels(panel["timestamp"].values)

    daily = XrBackend().to_internal(panel).resample(
        labels, {"Close": "last", "Weight": "last", "Close_T": "last"}
    ).data

    assert daily["Close"].attrs == {"unit": "usd"}
    assert daily.attrs == {"source": "demo"}
    assert daily["Weight"].values.tolist() == [0.25, 0.75]
    assert tuple(daily["Close_T"].dims) == ("timestamp", "symbol")
    assert daily["Close_T"].values[:, 1].tolist() == [60.0, 120.0, 180.0]
    assert daily["symbol"].values.tolist() == SYMBOLS

    untimed = panel[["Weight"]].assign_coords(timestamp=panel["timestamp"])
    daily = XrBackend().to_internal(untimed).resample(labels, {"Weight": "last"}).data
    assert daily.sizes == {"symbol": 2, "timestamp": DAYS}


def test_xr_backend_resample_refuses_a_missing_label_or_method():
    panel = _minute_panel()[["Close"]]
    backend = XrBackend().to_internal(panel)
    labels = _labels(panel["timestamp"].values).iloc[:-1]
    with pytest.raises(ValueError, match="has no label"):
        backend.resample(labels, {"Close": "last"})
    with pytest.raises(ValueError, match="no method for"):
        backend.resample(_labels(panel["timestamp"].values), {})


def test_pl_backend_resample_matches_the_xarray_result():
    panel = _minute_panel()
    labels = _labels(panel["timestamp"].values)
    expected = XrBackend().to_internal(panel).resample(labels, OHLCV_HOW).data

    table = PlBackend().to_internal(
        pl.from_pandas(panel.to_dataframe().reset_index()).lazy()
    )
    got = table.resample(labels, OHLCV_HOW).get_xarray_dataset(["timestamp", "symbol"])

    for name in OHLCV_HOW:
        np.testing.assert_allclose(got[name].values, expected[name].values)


# -- factor ---------------------------------------------------------------------


def _momentum(minute_config: DatasetConfig, tmp_path: Path) -> Momentum:
    return Momentum(
        PolarsFactorConfig(
            warmup_bars=1,
            dataset=SpotKlineDataset(minute_config),
            file_path=str(tmp_path / "factors" / "momentum.zarr"),
            kwargs={"n": 1},
        )
    )


def test_factor_resample_aggregates_the_computed_panel(
    minute_config: DatasetConfig, tmp_path: Path
):
    minute = _momentum(minute_config, tmp_path)
    minute_panel = compute_all(minute)

    daily = minute.resample("1d", "last")

    assert daily.config.dataset is not minute.config.dataset
    assert compute_all(minute).sizes["timestamp"] == DAYS * BARS_PER_DAY
    panel = compute_all(daily)
    assert dict(panel.sizes) == {"timestamp": DAYS, "symbol": 2}
    last_bar_of_each_day = minute_panel["momentum_1"].values[BARS_PER_DAY - 1 :: BARS_PER_DAY]
    np.testing.assert_allclose(panel["momentum_1"].values, last_bar_of_each_day)


def test_factor_compute_on_a_resampled_copy_resamples_its_output(
    minute_config: DatasetConfig, tmp_path: Path
):
    daily = _momentum(minute_config, tmp_path).resample("1d", {"momentum_1": "mean"})
    panel = compute_all(daily)
    assert panel.sizes["timestamp"] == DAYS
    assert np.isfinite(panel["momentum_1"].values[1:]).all()


def test_factor_resample_saves_beside_the_source_and_reads_it_back(
    minute_config: DatasetConfig, tmp_path: Path
):
    minute = _momentum(minute_config, tmp_path).build(*STORE_RANGE)
    daily = minute.resample("1d", "last")
    assert daily.store_path == str(tmp_path / "factors" / "momentum_resample_1d.zarr")

    daily.build(*STORE_RANGE)
    assert xr.open_zarr(daily.store_path).sizes["timestamp"] == DAYS
    assert xr.open_zarr(minute.store_path).sizes["timestamp"] == DAYS * BARS_PER_DAY

    # Without a built store the resampled copy reads the source and resamples.
    fresh = _momentum(minute_config, tmp_path).resample("1d", "last")
    Path(daily.store_path).rename(tmp_path / "aside.zarr")
    assert fresh.read(*STORE_RANGE).sizes["timestamp"] == DAYS
    # With one, it reads the built store.
    (tmp_path / "aside.zarr").rename(daily.store_path)
    assert _momentum(minute_config, tmp_path).resample("1d", "last").read(
        *STORE_RANGE
    ).sizes["timestamp"] == DAYS

    # The sibling store extends like any factor store: built in two pieces it
    # holds what one build over the whole range writes.
    pieces = _momentum(minute_config, tmp_path / "pieces").resample("1d", "last")
    assert pieces.store_path != daily.store_path
    pieces.build(STORE_RANGE[0], "2024-01-02")
    pieces.extend(STORE_RANGE[1])
    xr.testing.assert_allclose(xr.open_zarr(pieces.store_path).load(), xr.open_zarr(daily.store_path).load())


def test_factor_resample_config_round_trips(minute_config: DatasetConfig, tmp_path: Path):
    daily = _momentum(minute_config, tmp_path).resample("1d", "last")
    saved = json.loads(json.dumps(daily.get_config()))
    assert saved["resample_freq"] == "1d"

    rebuilt = rebuild(saved)
    assert rebuilt.config.resample_freq == "1d"
    assert compute_all(rebuilt).sizes["timestamp"] == DAYS


def test_kunquant_factor_resample_and_stream_refusal(
    minute_config_8: DatasetConfig, tmp_path: Path
):
    factor = Alpha158SpotKline(
        FactorConfig(
            warmup_bars=1,
            dataset=SpotKlineDataset(minute_config_8),
            mode="batch",
            data_columns=["open", "close", "volume"],
            factor_names=["KMID", "VOLUME0"],
            file_path=str(tmp_path / "factors" / "alpha158.zarr"),
            njobs=2,
        )
    )
    daily = factor.resample("1d", {"KMID": "mean", "VOLUME0": "last"})
    assert daily._lib is None
    assert dict(compute_all(daily).sizes) == {"timestamp": DAYS, "symbol": 8}
    assert compute_all(factor).sizes["timestamp"] == DAYS * BARS_PER_DAY
    with pytest.raises(ValueError, match="init_stream"):
        daily.init_stream()


# -- session-based labels -------------------------------------------------------


def test_session_labels_cut_by_session_and_label_daily_bars_at_midnight():
    sessions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-24").date(), pd.Timestamp("2024-01-25").date()],
            "open": pd.to_datetime(["2024-01-24 14:30", "2024-01-25 14:30"]),
            "close": pd.to_datetime(["2024-01-24 21:00", "2024-01-25 21:00"]),
        }
    )
    bars = pd.to_datetime(
        ["2024-01-24 14:31", "2024-01-24 15:00", "2024-01-24 21:00", "2024-01-25 14:31"]
    ).values

    daily = session_labels(bars, "1d", sessions, "Demo")
    assert daily.astype("datetime64[D]").astype(str).tolist() == [
        "2024-01-24", "2024-01-24", "2024-01-24", "2024-01-25",
    ]

    hourly = session_labels(bars, "1h", sessions, "Demo")
    assert pd.DatetimeIndex(hourly).strftime("%Y-%m-%d %H:%M").tolist() == [
        "2024-01-24 15:30", "2024-01-24 15:30", "2024-01-24 21:00", "2024-01-25 15:30",
    ]

    with pytest.raises(ValueError, match="falls in no trading session"):
        session_labels(pd.to_datetime(["2024-01-24 14:30"]).values, "1d", sessions, "Demo")


def test_nbbo_dataset_labels_bars_by_xnys_session(tmp_path: Path):
    from quantlab.dataset.config import NbboDatasetConfig
    from quantlab.dataset.nbbo import NbboPanelDataset

    config = NbboDatasetConfig(
        raw_data_dir_path=str(tmp_path / "downloads/us_equity/tick/wrds_taq/wrds"),
        zarr_file_path=str(tmp_path / "nbbo_1m.zarr"),
        reference_dir=str(tmp_path / "_reference"),
        session_start="04:00",
        session_end="20:00",
    )
    ds = NbboPanelDataset(config)
    # Friday 2024-01-26 with an extended close at 20:00 ET = 01:00 UTC Saturday.
    bars = pd.to_datetime(["2024-01-26 09:01", "2024-01-27 01:00"]).values
    labels = ds.resample_labels(bars, "1d")
    assert labels.astype("datetime64[D]").astype(str).tolist() == ["2024-01-26", "2024-01-26"]
