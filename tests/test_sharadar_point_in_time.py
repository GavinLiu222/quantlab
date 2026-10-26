"""The point-in-time rule of Sharadar release panels, through `quantlab.dataset.sharadar.point_in_time`.

Every Sharadar panel built from releases (fundamentals, fiscal years, 13F
holdings, filings) places a row at its available date, the first SEP trading
day on or after the day it became public, and never shows it earlier. This
suite pins that rule once, on hand-built rows and calendars (every value is
`# SYNTHETIC`): placement, the as-of choice, staleness, the symbol axis, and
a release dataset that supplies only its rows and a spec.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from quantlab.dataset.sharadar.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
    as_of,
    on_grid,
    shown_symbols,
    to_panel,
)

#: Weekdays of 2024-01-02..2024-01-19 but Monday 2024-01-15 (a holiday).
DAYS = [d for d in pd.bdate_range("2024-01-02", "2024-01-19") if d != pd.Timestamp("2024-01-15")]


def _days(days=DAYS) -> pl.Series:
    return pl.Series("timestamp", list(days), dtype=pl.Datetime("ns"))


def _calendar(days=DAYS, start="2024-01-01", end="2024-12-31") -> TradingCalendar:
    series = _days(days)
    return TradingCalendar(series, through=series.max().date(), start=start, end=end)


def _released(rows: list[tuple[int, str, float]], calendar=None) -> pl.DataFrame:
    """``(symbol, release date, value)`` rows placed on ``calendar``, as a release table hands them over."""
    frame = pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "date": [date.fromisoformat(r[1]) for r in rows],
            "value": [r[2] for r in rows],
        },
        schema={"symbol": pl.Int64, "date": pl.Date, "value": pl.Float64},
    )
    return (
        (calendar or _calendar())
        .place(frame)
        .with_columns(pl.col("date").cast(pl.Datetime("ns")).alias("release_date"))
        .sort("available")
    )


def _shown(panel, symbol=101) -> dict[str, float]:
    column = panel["value"].sel(symbol=symbol)
    stamps = pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d")
    return {day: value for day, value in zip(stamps, column.values.tolist()) if not np.isnan(value)}


SPEC = ReleaseSpec(variables={"value": {"unit": "USD"}, "release_date": {}}, dates=("release_date",))


# -- placement ----------------------------------------------------------------


def test_a_row_released_on_a_trading_day_is_available_that_day():
    rows = _released([(101, "2024-01-04", 1.0)])  # SYNTHETIC
    assert rows.get_column("available").dt.date().to_list() == [date(2024, 1, 4)]


@pytest.mark.parametrize(
    ("released", "available"),
    [("2024-01-06", date(2024, 1, 8)), ("2024-01-13", date(2024, 1, 16)), ("2024-01-15", date(2024, 1, 16))],
)
def test_a_weekend_or_holiday_release_is_available_on_the_next_trading_day(released, available):
    rows = _released([(101, released, 1.0)])  # SYNTHETIC
    assert rows.get_column("available").dt.date().to_list() == [available]


def test_a_row_the_calendar_does_not_reach_yet_is_not_available():
    rows = _released([(101, "2024-01-20", 1.0), (101, "2024-01-19", 2.0)])  # SYNTHETIC
    assert rows.get_column("value").to_list() == [2.0]


def test_a_row_released_before_the_calendar_rolls_onto_its_first_day_unless_dropped():
    calendar = _calendar()
    frame = pl.DataFrame({"date": [date(2023, 12, 1)]})  # SYNTHETIC
    assert calendar.place(frame).get_column("available").dt.date().to_list() == [date(2024, 1, 2)]
    assert calendar.place(frame, drop_earlier=True).height == 0


def test_a_lag_is_added_before_placing():
    frame = pl.DataFrame({"quarter_end": [date(2023, 11, 30)]})  # SYNTHETIC
    placed = _calendar().place(frame, "quarter_end", lag=timedelta(days=45))
    # 2023-11-30 + 45 days = Sunday 2024-01-14; Monday is a holiday.
    assert placed.get_column("available").dt.date().to_list() == [date(2024, 1, 16)]


def test_days_in_stays_inside_the_configured_window():
    calendar = _calendar(start="2024-01-05", end="2024-01-10")
    days = calendar.days_in("2024-01-01", "2024-01-31").dt.date().to_list()
    assert days == [date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]


# -- the as-of choice ---------------------------------------------------------


def test_a_row_is_shown_from_its_available_date_and_not_a_day_earlier():
    rows = _released([(101, "2024-01-06", 1.0), (101, "2024-01-10", 2.0)])  # SYNTHETIC
    shown = _shown(as_of(rows, _days(), [101], SPEC))
    assert min(shown) == "2024-01-08"
    assert shown["2024-01-09"] == 1.0
    assert shown["2024-01-10"] == 2.0
    assert shown["2024-01-19"] == 2.0


def test_every_cell_shows_the_latest_row_available_by_its_day():
    # The reference: for each day, the last row (in available order) with
    # available <= day. Random rows of three symbols, some released on
    # weekends and some past the calendar.
    rnd = random.Random(7)  # SYNTHETIC
    stamps = pd.date_range("2023-12-25", "2024-01-25").strftime("%Y-%m-%d").tolist()
    rows = []
    for symbol in (101, 202, 303):
        for released in sorted(rnd.sample(stamps, 6)):
            rows.append((symbol, released, float(rnd.randint(1, 99))))
    released = _released(rows)
    # One row per (symbol, available): the last released, as a dataset keeps.
    released = released.sort("symbol", "date").unique(
        subset=["symbol", "available"], keep="last", maintain_order=True
    ).sort("available")
    panel = as_of(released, _days(), [101, 202, 303], SPEC)
    for symbol in (101, 202, 303):
        mine = released.filter(pl.col("symbol") == symbol).sort("available")
        for day, value in zip(DAYS, panel["value"].sel(symbol=symbol).values.tolist()):
            known = mine.filter(pl.col("available") <= day)
            expected = known.get_column("value")[-1] if known.height else np.nan
            assert value == expected or (np.isnan(value) and np.isnan(expected)), (symbol, day)
            if known.height:
                assert known.get_column("available")[-1] <= day


# -- staleness ------------------------------------------------------------------


def test_a_row_not_replaced_stops_being_shown_stale_after_days_after_its_release():
    rows = _released([(101, "2024-01-03", 1.0)])  # SYNTHETIC
    spec = ReleaseSpec(variables=SPEC.variables, stale_after_days=6)
    shown = _shown(as_of(rows, _days(), [101], spec))
    # 2024-01-03 + 6 days = 2024-01-09, the last day shown.
    assert sorted(shown) == ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]


def test_staleness_counts_from_the_release_date_not_the_available_date():
    rows = _released([(101, "2024-01-06", 1.0)])  # SYNTHETIC: a Saturday release
    spec = ReleaseSpec(variables=SPEC.variables, stale_after_days=3)
    assert sorted(_shown(as_of(rows, _days(), [101], spec))) == ["2024-01-08", "2024-01-09"]


def test_a_newer_row_is_fresh_again():
    rows = _released([(101, "2024-01-02", 1.0), (101, "2024-01-12", 2.0)])  # SYNTHETIC
    spec = ReleaseSpec(variables=SPEC.variables, stale_after_days=2)
    shown = _shown(as_of(rows, _days(), [101], spec))
    assert shown == {"2024-01-02": 1.0, "2024-01-03": 1.0, "2024-01-04": 1.0, "2024-01-12": 2.0}


def test_a_stale_cell_hides_every_variable_its_date_included():
    rows = _released([(101, "2024-01-02", 1.0)])  # SYNTHETIC
    spec = ReleaseSpec(variables=SPEC.variables, stale_after_days=1)
    panel = as_of(rows, _days(), [101], spec)
    assert np.isnat(panel["release_date"].sel(symbol=101, timestamp="2024-01-04").values)


# -- the symbol axis and the shaping -------------------------------------------


def test_a_symbol_first_available_after_the_window_is_not_on_its_axis():
    rows = _released([(202, "2024-01-03", 1.0), (101, "2024-01-17", 2.0)])  # SYNTHETIC
    assert shown_symbols(rows, _days(DAYS[:5])) == [202]
    assert shown_symbols(rows, _days()) == [101, 202]


def test_cells_are_shaped_day_by_day_symbol_by_symbol():
    days = _days(DAYS[:2])
    cells = on_grid(days, [202, 101]).with_columns(value=pl.Series([1.0, 2.0, 3.0, 4.0]))
    panel = to_panel(cells, days, [202, 101], {"value": {"unit": "USD"}})
    assert panel["value"].values.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert panel["symbol"].values.tolist() == [202, 101]
    assert panel["value"].attrs == {"unit": "USD"}


# -- a release table is rows and a spec --------------------------------------------


class _Releases(ReleasePanelDataset):
    """A release table of hand-written rows on a hand-written calendar."""

    from quantlab.dataset.config import SharadarDatasetConfig as config_cls

    ROWS: list[tuple[int, str, float]] = []
    CALENDAR_DAYS: list = DAYS
    SPEC: ReleaseSpec = SPEC

    def _trading_calendar(self) -> TradingCalendar:
        return _calendar(self.CALENDAR_DAYS, self.config.start_date, self.config.end_date)

    def _released(self) -> pl.DataFrame:
        return _released(self.ROWS, self._trading_calendar())

    def _spec(self) -> ReleaseSpec:
        return self.SPEC

    def _nothing_shown(self) -> str:
        return "no row is shown"


def _releases(tmp_path, rows, days=DAYS, spec=SPEC):
    from quantlab.dataset.config import SharadarDatasetConfig

    cls = type("Releases", (_Releases,), {"ROWS": rows, "CALENDAR_DAYS": days, "SPEC": spec})
    return cls(
        SharadarDatasetConfig(
            zarr_file_path=str(tmp_path / "releases.zarr"),
            raw_data_dir_path=str(tmp_path / "raw"),
            permatickers=(101, 202),
        )
    )


ROWS = [(101, "2024-01-03", 1.0), (101, "2024-01-10", 2.0), (202, "2024-01-13", 7.0)]  # SYNTHETIC


def test_a_release_dataset_builds_the_as_of_panel_from_its_rows_and_spec(tmp_path):
    _releases(tmp_path, ROWS).update()
    panel = _releases(tmp_path, ROWS).panel("2024-01-01", "2024-12-31").load()

    assert _shown(panel) == {d.strftime("%Y-%m-%d"): (1.0 if d < pd.Timestamp("2024-01-10") else 2.0) for d in DAYS[1:]}
    assert min(_shown(panel, 202)) == "2024-01-16"
    assert panel["value"].attrs["unit"] == "USD"


def test_an_update_appends_the_days_a_rebuild_would_hold(tmp_path):
    # Built on the first week, updated once the calendar reaches the third:
    # 202 files only in the second week, so the update widens the store.
    _releases(tmp_path / "updated", ROWS, DAYS[:4]).update()
    _releases(tmp_path / "updated", ROWS).update()
    updated = _releases(tmp_path / "updated", ROWS).panel("2024-01-01", "2024-12-31").load()
    _releases(tmp_path / "rebuilt", ROWS).update()
    rebuilt = _releases(tmp_path / "rebuilt", ROWS).panel("2024-01-01", "2024-12-31").load()

    assert updated["symbol"].values.tolist() == [101, 202]
    np.testing.assert_array_equal(updated["value"].values, rebuilt["value"].values)
    np.testing.assert_array_equal(updated["release_date"].values, rebuilt["release_date"].values)


def test_an_update_widens_a_store_whose_staleness_column_is_not_a_variable(tmp_path):
    # The 10-K changes panel's shape: staleness is measured from release_date,
    # which the panel does not show. 202's first row comes in the update.
    spec = ReleaseSpec(variables={"value": {"unit": "USD"}})
    _releases(tmp_path / "updated", ROWS, DAYS[:4], spec).update()
    _releases(tmp_path / "updated", ROWS, spec=spec).update()
    updated = _releases(tmp_path / "updated", ROWS, spec=spec).panel("2024-01-01", "2024-12-31").load()
    _releases(tmp_path / "rebuilt", ROWS, spec=spec).update()
    rebuilt = _releases(tmp_path / "rebuilt", ROWS, spec=spec).panel("2024-01-01", "2024-12-31").load()

    assert list(updated.data_vars) == ["value"]
    assert updated["symbol"].values.tolist() == [101, 202]
    np.testing.assert_array_equal(updated["value"].values, rebuilt["value"].values)


def test_an_empty_window_is_refused_with_the_datasets_words(tmp_path):
    with pytest.raises(ValueError, match=r"Releases: no row is shown in \[.*\] for the configured universe"):
        _releases(tmp_path, [(101, "2024-02-01", 1.0)]).update()  # SYNTHETIC
