"""A resampled factor reads on resample like a resampled dataset (ADR 0002, #300).

A resampled factor whose own ``<stem>_resample_<freq>.zarr`` store is not built
reads the source factor's store and resamples the part the request needs, through
the same resample-on-read helper ``BaseDataset.panel`` uses:

- the requested symbols are cut from the source bars before they are aggregated;
- inside a ``DataRecorder`` the read is recorded over the source bars it read (the
  resampled panel is computed, not stored), as a resampled dataset's is.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.config import PolarsFactorConfig
from quantlab.factor.predefined.momentum import Momentum
from quantlab.runs.record import DataRecorder

SYMBOLS = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
BARS_PER_DAY = 6
DAYS = 3
STORE_RANGE = ("2024-01-01", "2024-01-03")


def _minute_momentum(tmp_path: Path) -> Momentum:
    """A one-bar momentum over three days of six minute bars of three symbols, built."""
    timestamps = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"2024-01-0{day} 00:00", periods=BARS_PER_DAY, freq="min").values
        for day in range(1, DAYS + 1)
    ]))
    close = np.arange(1.0, len(timestamps) + 1.0)[:, None] * (10.0 ** np.arange(len(SYMBOLS)))[None, :]
    store = tmp_path / "klines.zarr"
    xr.Dataset(
        {name: (["timestamp", "symbol"], close) for name in ("Open", "High", "Low", "Close", "Volume")},
        coords={"timestamp": timestamps, "symbol": SYMBOLS},
    ).to_zarr(store, mode="w")
    dataset = SpotKlineDataset(DatasetConfig(
        raw_data_dir_path=str(tmp_path / "raw"), zarr_file_path=str(store), market="crypto_spot", frequency="1m",
    ))
    factor = Momentum(PolarsFactorConfig(
        warmup_bars=1, dataset=dataset, file_path=str(tmp_path / "momentum.zarr"), kwargs={"n": 1},
    ))
    return factor.build(*STORE_RANGE)


def test_an_unbuilt_resampled_factor_records_the_source_bars_it_read(tmp_path):
    minute = _minute_momentum(tmp_path)
    daily = minute.resample("1d", "last")
    with DataRecorder(keys=[(minute, "source")]) as expected:
        minute.read(*STORE_RANGE, symbols=["CCCUSDT", "AAAUSDT"])

    with DataRecorder(keys=[(daily, "daily")]) as recorder:
        panel = daily.read(*STORE_RANGE, symbols=["CCCUSDT", "AAAUSDT"])

    assert dict(panel.sizes) == {"timestamp": DAYS, "symbol": 2}
    (entry,) = recorder.records["daily"]
    (source,) = expected.records["source"]
    assert entry["n_timestamps"] == DAYS * BARS_PER_DAY
    assert entry["n_symbols"] == 2
    assert entry["request"]["symbols"] == ["CCCUSDT", "AAAUSDT"]
    assert entry["digest"] == source["digest"]


def test_an_unrecorded_unmapped_resampled_read_is_keyed_by_the_source_store(tmp_path):
    daily = _minute_momentum(tmp_path).resample("1d", "last")
    with DataRecorder() as recorder:
        daily.read(*STORE_RANGE)

    assert list(recorder.records) == [f"Momentum:{tmp_path / 'momentum.zarr'}"]


def test_a_symbol_cut_read_equals_the_whole_read_cut(tmp_path):
    daily = _minute_momentum(tmp_path).resample("1d", "last")

    cut = daily.read(*STORE_RANGE, symbols=["CCCUSDT", "AAAUSDT"])

    xr.testing.assert_identical(cut, daily.read(*STORE_RANGE).sel(symbol=["CCCUSDT", "AAAUSDT"]))
    # The last minute of each day, over the minute before it.
    last = np.arange(BARS_PER_DAY, DAYS * BARS_PER_DAY + 1, BARS_PER_DAY, dtype=float)
    np.testing.assert_allclose(cut["momentum_1"].values[:, 1], last / (last - 1) - 1)
