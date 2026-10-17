"""The Day aggregate store on the Massive ticker (#352): day_aggs files to ``massive_day_bars_1d``.

Seam 2 of spec #345: day-aggregate ``csv.gz`` files written into the raw tier
with tests/massive_fixtures.py, converted into a store, and assertions on the
panel ``read()`` returns. What is locked here:

- every ticker of each day's file is a column, spelled as Massive spells it,
  with the file's open, high, low, close, volume and ``transactions``
  (``n_trades``) exactly as written, one bar per trading day labelled with
  its date;
- a ticker absent from a day's file has NaN prices and zero volume and
  ``n_trades`` that day;
- a ticker first met on a later day is appended at the end of the symbol
  axis (ADR 0032);
- day-by-day ``update()`` equals a whole build, and the config rebuilds.

Every value is SYNTHETIC.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.massive_fixtures import aggregate_line, write_aggregates

DAY_1 = date(2024, 11, 26)
DAY_2 = date(2024, 11, 27)
DAY_3 = date(2024, 11, 29)  # a half day: still one bar


def _start(day: date) -> datetime:
    """The day aggregate's window start: midnight New York, naive UTC."""
    return datetime.fromisoformat(f"{day.isoformat()}T05:00")


def _raw(root: Path) -> Path:
    massive = root / "downloads" / "massive"
    write_aggregates(massive, DAY_1, [
        aggregate_line("AAA", _start(DAY_1), 10.0, 10.5, 9.75, 10.25, 1000, 12),  # SYNTHETIC
        aggregate_line("BRK.A", _start(DAY_1), 600000.0, 601000.0, 599000.0, 600500.0, 3, 3),  # SYNTHETIC
    ], data_type="day_aggs")
    write_aggregates(massive, DAY_2, [
        aggregate_line("AAA", _start(DAY_2), 10.25, 11.0, 10.0, 10.75, 1500.5, 20),  # SYNTHETIC
        aggregate_line("ZNEW", _start(DAY_2), 5.0, 5.5, 4.5, 5.25, 700, 7),  # SYNTHETIC
    ], data_type="day_aggs")
    write_aggregates(massive, DAY_3, [
        aggregate_line("AApB", _start(DAY_3), 25.0, 25.0, 25.0, 25.0, 100, 1),  # SYNTHETIC
        aggregate_line("AAA", _start(DAY_3), 10.75, 10.8, 10.6, 10.7, 400, 4),  # SYNTHETIC
        aggregate_line("BRK.A", _start(DAY_3), 601000.0, 602000.0, 600000.0, 601500.0, 2, 2),  # SYNTHETIC
    ], data_type="day_aggs")
    return massive


def _dataset(root: Path, start: date = DAY_1, end: date | None = DAY_3):
    from quantlab.dataset.config import MassiveDayBarsDatasetConfig
    from quantlab.dataset.massive.day_bars import MassiveDayBarDataset

    massive = _raw(root)
    return MassiveDayBarDataset(MassiveDayBarsDatasetConfig(
        zarr_file_path=str(root / "zarrs" / "massive_day_bars_1d.zarr"),
        raw_data_dir_path=str(massive),
        start_date=start.isoformat(),
        end_date=None if end is None else end.isoformat(),
    ))


def _read(ds) -> xr.Dataset:
    return ds.panel("2024-11-26", "2024-11-30").load()


@pytest.fixture
def built(tmp_path):
    ds = _dataset(tmp_path).update(granularity="day")
    return ds, _read(ds)


def _cell(panel, symbol: str, day: date) -> dict:
    return {name: float(panel[name].sel(timestamp=pd.Timestamp(day), symbol=symbol)) for name in panel.data_vars}


def test_each_tickers_day_reads_exactly_as_the_file_has_it(built):
    _, panel = built
    assert set(panel.data_vars) == {"open", "high", "low", "close", "volume", "n_trades"}
    assert all(panel[name].dtype == np.float64 for name in panel.data_vars)
    assert pd.DatetimeIndex(panel.timestamp.values).tolist() == [pd.Timestamp(d) for d in (DAY_1, DAY_2, DAY_3)]
    assert _cell(panel, "AAA", DAY_2) == {
        "open": 10.25, "high": 11.0, "low": 10.0, "close": 10.75, "volume": 1500.5, "n_trades": 20.0}
    assert _cell(panel, "BRK.A", DAY_3) == {
        "open": 601000.0, "high": 602000.0, "low": 600000.0, "close": 601500.0, "volume": 2.0, "n_trades": 2.0}
    assert _cell(panel, "AApB", DAY_3)["close"] == 25.0


def test_a_ticker_absent_from_a_days_file_has_no_bar(built):
    _, panel = built
    missing = _cell(panel, "BRK.A", DAY_2)
    assert all(np.isnan(missing[k]) for k in ("open", "high", "low", "close"))
    assert (missing["volume"], missing["n_trades"]) == (0.0, 0.0)
    assert _cell(panel, "ZNEW", DAY_1)["volume"] == 0.0


def test_a_new_ticker_is_appended_at_the_end_of_the_axis(tmp_path):
    ds = _dataset(tmp_path, end=DAY_1).update(granularity="day")
    assert _read(ds).symbol.values.tolist() == ["AAA", "BRK.A"]
    ds = _dataset(tmp_path, end=DAY_2).update(granularity="day")
    assert _read(ds).symbol.values.tolist() == ["AAA", "BRK.A", "ZNEW"]
    ds = _dataset(tmp_path, end=DAY_3).update(granularity="day")
    assert _read(ds).symbol.values.tolist() == ["AAA", "BRK.A", "ZNEW", "AApB"]


def test_a_day_at_a_time_equals_at_once(tmp_path):
    whole = _read(_dataset(tmp_path / "whole").update(granularity="day"))
    # The daily update calls update() as the store's component.json rebuilds it: year windows.
    for end in (DAY_1, DAY_2, DAY_3):
        daily = _dataset(tmp_path / "daily", end=end).update()
    daily = _read(daily)
    # A whole build sorts its axis; the daily store appends: same cells either way.
    xr.testing.assert_identical(daily.sel(symbol=sorted(daily.symbol.values.tolist())),
                                whole.sel(symbol=sorted(whole.symbol.values.tolist())))
    # Updating again with nothing new changes nothing.
    again = _read(_dataset(tmp_path / "daily", end=DAY_3).update())
    xr.testing.assert_identical(again, daily)


def test_an_open_ended_config_takes_every_raw_day(tmp_path):
    ds = _dataset(tmp_path, end=None).update(granularity="day")
    assert len(_read(ds).timestamp) == 3


def test_the_config_rebuilds_and_its_store_folder_reads_the_same(built, tmp_path):
    from quantlab.core.component import rebuild
    from quantlab.core.store_folder import load_component, save_component

    ds, panel = built
    assert rebuild(ds.get_config()) == ds
    folder = tmp_path / "market" / "massive" / "massive_day_bars_1d"
    save_component(ds, folder, readme="# massive_day_bars_1d\n")
    assert (folder / "README.md").exists() and (folder / "component.json").exists()
    xr.testing.assert_identical(_read(load_component(folder)), panel)


def test_symbols_are_refused(tmp_path):
    from dataclasses import replace

    from quantlab.dataset.massive.day_bars import MassiveDayBarDataset

    ds = _dataset(tmp_path)
    with pytest.raises(ValueError, match="Massive ticker"):
        MassiveDayBarDataset(replace(ds.config, symbols=("AAA",)))


def test_a_file_of_a_day_without_a_session_is_refused(tmp_path):
    ds = _dataset(tmp_path)
    write_aggregates(Path(ds.config.raw_data_dir_path), date(2024, 11, 28),  # Thanksgiving
                     [aggregate_line("AAA", _start(date(2024, 11, 28)), 1.0, 1.0, 1.0, 1.0, 1, 1)],  # SYNTHETIC
                     data_type="day_aggs")
    with pytest.raises(ValueError, match="2024-11-28 is not of an XNYS session"):
        ds.update(granularity="day")
