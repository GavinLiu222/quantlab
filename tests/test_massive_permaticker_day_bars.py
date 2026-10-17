"""The Day aggregate store on the permaticker (#353): ``massive_permaticker_day_bars_1d``.

Built on the fixture of ``test_massive_permaticker_bars``: the same raw tier holds the day
aggregates of two days, from which ``MassiveDayBarDataset`` builds its Massive-ticker store,
and the Ticker map over them. ``PermatickerDayBarDataset`` reads that store on permatickers
through the map, and ``MappedDayBarDataset`` keeps the mapping in a store. What is locked here:

- the store reads as the view, a security that changed ticker is one column, a reused ticker
  goes to the permaticker that held it that day;
- a ticker the map cannot place on a day is counted in the sidecar, never guessed (for the
  mapped Trade bars too, since both share one mapping);
- a changed map is refused, then mapped again by ``update()``;
- day-by-day ``update()`` equals a whole build, the config rebuilds and its store folder reads
  the same.

Every value is SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr

from quantlab.dataset.massive.ticker_map import TickerMap
from tests.test_massive_permaticker_bars import FULL_DAY, HALF_DAY, built  # noqa: F401  (fixture)

RANGE = ("2024-11-27", "2024-11-30")


@pytest.fixture
def view(built, tmp_path):
    from quantlab.dataset.config import MassiveDayBarsDatasetConfig
    from quantlab.dataset.massive.day_bars import MassiveDayBarDataset
    from quantlab.dataset.massive.permaticker_bars import PermatickerDayBarDataset

    trade_bars, ticker_map, _ = built
    day_bars = MassiveDayBarDataset(MassiveDayBarsDatasetConfig(
        zarr_file_path=str(tmp_path / "market" / "massive_day_bars_1d.zarr"),
        raw_data_dir_path=trade_bars.config.raw_data_dir_path,
        start_date=FULL_DAY.isoformat(),
    )).update(granularity="day")
    return PermatickerDayBarDataset(day_bars, ticker_map)


def _mapped(view, path):
    from quantlab.dataset.config import MappedDayBarsConfig
    from quantlab.dataset.massive.mapped_bars import MappedDayBarDataset

    return MappedDayBarDataset(MappedDayBarsConfig(source=view, zarr_file_path=str(path)))


def _assert_same(stored: xr.Dataset, view: xr.Dataset) -> None:
    view = view.reindex(symbol=stored["symbol"].values)
    assert stored["timestamp"].values.tolist() == view["timestamp"].values.tolist()
    assert list(stored.data_vars) == list(view.data_vars)
    for name in view.data_vars:
        np.testing.assert_array_equal(stored[name].values, view[name].values, err_msg=name)


def test_the_view_maps_each_day_onto_permatickers(view):
    panel = view.panel(*RANGE)

    assert panel["symbol"].dtype == np.int64
    assert panel["symbol"].values.tolist() == [101, 202, 303, 404, 505]
    assert sorted(panel.data_vars) == ["close", "high", "low", "n_trades", "open", "volume"]
    close = panel["close"].to_pandas()
    full, half = np.datetime64(FULL_DAY), np.datetime64(HALF_DAY)
    assert close.loc[full].tolist()[:3] == [11.0, 20.0, 30.0]  # AAA, OLD, RRR
    assert close.loc[half].tolist() == [13.0, 21.0, 31.0, 40.0, 50.0]  # AAA, NEW, RRS, RRR (404), FFF
    assert np.isnan(close.loc[full, 404]) and float(panel["volume"].sel(timestamp=full, symbol=404)) == 0.0


def test_the_store_reads_as_the_view(view, tmp_path):
    mapped = _mapped(view, tmp_path / "mapped.zarr").update()

    assert mapped.stored_days() == [FULL_DAY, HALF_DAY]
    panel = mapped.panel(*RANGE)
    assert panel["symbol"].dtype == np.int64
    _assert_same(panel, view.panel(*RANGE))


def test_a_ticker_the_map_cannot_place_is_counted_not_guessed(view, tmp_path):
    mapped = _mapped(view, tmp_path / "mapped.zarr").update()

    # ZZZ, a test symbol no security ever held, has a bar on both days and is in no column.
    assert mapped.unplaced() == {FULL_DAY: 1, HALF_DAY: 1}
    assert float(mapped.panel(*RANGE)["volume"].sum()) == float(view.panel(*RANGE)["volume"].sum())


def test_the_mapped_trade_bars_count_their_unplaced_tickers_too(built, tmp_path):
    from quantlab.dataset.config import MappedTradeBarsConfig
    from quantlab.dataset.massive.mapped_bars import MappedTradeBarDataset

    mapped = MappedTradeBarDataset(MappedTradeBarsConfig(source=built[2], zarr_file_path=str(tmp_path / "m.zarr")))

    assert mapped.update().unplaced() == {FULL_DAY: 1, HALF_DAY: 1}


def test_a_changed_map_is_refused_then_remapped_by_update(view, tmp_path):
    mapped = _mapped(view, tmp_path / "mapped.zarr").update()
    day_file = TickerMap.day_file(view.ticker_map.path, FULL_DAY)
    rows = pl.read_parquet(day_file)
    # The map now gives RRR on the full day to 404 instead of 303.
    rows.with_columns(
        pl.when(pl.col("ticker") == "RRR").then(404).otherwise(pl.col("permaticker")).alias("permaticker")
    ).write_parquet(day_file)

    with pytest.raises(ValueError, match="Ticker map changed on 1 stored day"):
        mapped.panel(*RANGE)

    mapped.update()

    panel = mapped.panel(*RANGE)
    _assert_same(panel, view.panel(*RANGE))
    assert float(panel["close"].sel(timestamp=np.datetime64(FULL_DAY), symbol=404)) == 30.0


def test_a_day_at_a_time_equals_a_whole_build(view, tmp_path):
    daily = _mapped(view, tmp_path / "daily.zarr").update(FULL_DAY)
    assert daily.stored_symbols() == [101, 202, 303]
    daily.update()

    whole = _mapped(view, tmp_path / "whole.zarr").update()

    assert daily.stored_symbols() == [101, 202, 303, 404, 505]
    _assert_same(daily.panel(*RANGE), whole.panel(*RANGE))
    assert daily.unplaced() == whole.unplaced()
    assert daily.update().stored_days() == [FULL_DAY, HALF_DAY]  # nothing left to do


def test_the_config_rebuilds_and_its_store_folder_reads_the_same(view, tmp_path):
    from quantlab.core.component import rebuild
    from quantlab.core.store_folder import load_component, save_component

    mapped = _mapped(view, tmp_path / "massive_permaticker_day_bars_1d.zarr")
    assert rebuild(mapped.get_config()) == mapped
    assert rebuild(view.get_config()) == view

    mapped.update()
    folder = tmp_path / "market" / "massive" / "massive_permaticker_day_bars_1d"
    save_component(mapped, folder, readme="# massive_permaticker_day_bars_1d\n")
    assert (folder / "README.md").exists() and (folder / "component.json").exists()
    xr.testing.assert_identical(load_component(folder).panel(*RANGE).load(), mapped.panel(*RANGE).load())
