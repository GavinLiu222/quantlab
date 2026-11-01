"""Trade bars mapped onto permatickers and kept in a store (MappedTradeBarDataset).

Built on the fixture of ``test_massive_permaticker_bars``: two days of SYNTHETIC trades, a
Ticker map over them and the permaticker view. The store must read as the view does, grow a
day at a time, and follow a map that changes under it.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import xarray as xr

from quantlab.dataset.config import MappedTradeBarsConfig
from quantlab.dataset.massive.mapped_bars import MAP_SIDECAR, MappedTradeBarDataset
from quantlab.dataset.massive.ticker_map import TickerMap
from tests.test_massive_permaticker_bars import FULL_DAY, HALF_DAY, built  # noqa: F401  (fixture)

RANGE = ("2024-11-27", "2024-11-30")


def _mapped(built, tmp_path) -> MappedTradeBarDataset:
    return MappedTradeBarDataset(MappedTradeBarsConfig(source=built[2], zarr_file_path=str(tmp_path / "mapped.zarr")))


def _assert_same(stored: xr.Dataset, view: xr.Dataset) -> None:
    view = view.reindex(symbol=stored["symbol"].values)
    assert stored["timestamp"].values.tolist() == view["timestamp"].values.tolist()
    for name in view.data_vars:
        np.testing.assert_array_equal(stored[name].values, view[name].values, err_msg=name)


def test_the_store_reads_as_the_view(built, tmp_path):
    mapped = _mapped(built, tmp_path).update()

    assert mapped.stored_days() == [FULL_DAY, HALF_DAY]
    panel = mapped.panel(*RANGE)
    assert panel["symbol"].dtype == np.int64
    assert sorted(panel["symbol"].values.tolist()) == [101, 202, 303, 404, 505]
    _assert_same(panel, built[2].panel(*RANGE))


def test_a_day_at_a_time_equals_at_once_new_permatickers_at_the_end(built, tmp_path):
    mapped = _mapped(built, tmp_path).update(FULL_DAY)
    assert mapped.stored_symbols() == [101, 202, 303]

    mapped.update()

    assert mapped.stored_symbols() == [101, 202, 303, 404, 505]
    _assert_same(mapped.panel(*RANGE), built[2].panel(*RANGE))
    assert mapped.update().stored_days() == [FULL_DAY, HALF_DAY]  # nothing left to do


def test_a_changed_map_is_refused_then_remapped_by_update(built, tmp_path):
    mapped = _mapped(built, tmp_path).update()
    ticker_map: TickerMap = built[1]
    day_file = TickerMap.day_file(ticker_map.path, FULL_DAY)
    rows = pl.read_parquet(day_file)
    # The map now gives RRR on the full day to 404 instead of 303.
    rows.with_columns(
        pl.when(pl.col("ticker") == "RRR").then(404).otherwise(pl.col("permaticker")).alias("permaticker")
    ).write_parquet(day_file)

    with pytest.raises(ValueError, match="Ticker map changed on 1 stored day"):
        mapped.panel(*RANGE)
    mapped.panel("2024-11-29", "2024-11-30")  # an unchanged day still reads

    mapped.update()

    view = built[2].panel(*RANGE)
    _assert_same(mapped.panel(*RANGE), view)
    assert float(view["volume"].sel(symbol=404).sum()) > 0  # 404 now trades on the full day


def test_a_store_without_its_sidecar_is_refused(built, tmp_path):
    mapped = _mapped(built, tmp_path).update(FULL_DAY)
    mapped.sidecar_path.unlink()

    with pytest.raises(ValueError, match=MAP_SIDECAR):
        mapped.update()


def test_the_sidecar_records_each_day(built, tmp_path):
    mapped = _mapped(built, tmp_path).update()

    payload = json.loads(mapped.sidecar_path.read_text())
    assert sorted(payload["days"]) == [FULL_DAY.isoformat(), HALF_DAY.isoformat()]
    assert all(len(digest) == 64 for digest in payload["days"].values())


def test_the_config_rebuilds(built, tmp_path):
    from quantlab.core.component import rebuild

    mapped = _mapped(built, tmp_path)
    assert rebuild(mapped.get_config()) == mapped


def test_workers_map_ahead_and_write_in_order(built, tmp_path, monkeypatch):
    monkeypatch.setattr(MappedTradeBarDataset, "BATCH_DAYS", 1)
    mapped = _mapped(built, tmp_path).update(workers=2)

    assert mapped.stored_days() == [FULL_DAY, HALF_DAY]
    _assert_same(mapped.panel(*RANGE), built[2].panel(*RANGE))
