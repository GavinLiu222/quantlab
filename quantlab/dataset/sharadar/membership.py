"""Point-in-time S&P 500 membership from Sharadar's SP500 table, keyed by permaticker.

The SP500 table records each change to the index: an ``added`` or ``removed``
row whose ``date`` is the *effective* membership date (not the announcement),
plus a ``historical`` snapshot of every member at each calendar quarter end
and a ``current`` snapshot dated on the vendor's last refresh. Tickers are
Sharadar's current identifiers, so a renamed company's history sits under
its present ticker and maps to one permaticker through TICKERS.

A stock is a member from its ``added`` date and is no longer one on its
``removed`` date, so a spell ends the day before the removal. The changes go
back to the index's launch in 1957, but Sharadar's prices (and so its
permatickers) start on ``PIT_COVERAGE_START``, 1997-12-31: a spell that ended
before then is dropped, whether or not the company kept trading, and the
panel answers membership from that date. So a roster (``roster_universe``)
is the same whether its window starts before or on that date. A former member Sharadar never priced is dropped
with a warning naming it; a member to the table's end without a permaticker
is refused, because it means TICKERS is older than SP500. A member whose first recorded change is a removal
is taken as a member since ``PIT_COVERAGE_START``; a member with no change at
all was in the index throughout. Every spell ends by the table's last date, so a panel
built from the same raw file is the same whenever it is built.

Examples
--------
>>> config = FileConstituentDatasetConfig(
...     zarr_file_path="/data/universe/sharadar/sharadar_sp500_membership/sharadar_sp500_membership.zarr",
...     cache_dir="/data/downloads/sharadar",
...     start_date="2015-01-01",
... )
>>> panel = SharadarSP500ConstituentDataset(config).from_raw_data().get_xarray_dataset()
>>> panel["symbol"].dtype
dtype('int64')
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
from loguru import logger

from quantlab.dataset.config import FileConstituentDatasetConfig
from quantlab.dataset.raw import IndexConstituentDataset
from quantlab.dataset.sharadar.tables import (
    map_permatickers,
    permaticker_mapping,
    scan_raw_table,
)

#: The first day of Sharadar's prices (live bulk pull, 2026-10-05). The SP500
#: changes go back to 1957, but a member that left before this day has no
#: permaticker, so membership on the permaticker axis is answered from here.
#: A former member Sharadar never priced is dropped with a warning.
PIT_COVERAGE_START = "1997-12-31"

#: The universes ``SharadarDatasetConfig.roster_universe`` accepts.
ROSTER_UNIVERSES: tuple[str, ...] = ("sp500",)

_ONE_DAY = timedelta(days=1)


def sp500_intervals(vendor_root: str | Path) -> pl.DataFrame:
    """Return every S&P 500 membership spell in the raw SP500 table.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``, holding the ``sp500`` and ``tickers``
        raw tables.

    Returns
    -------
    pl.DataFrame
        ``permaticker`` (Int64), ``start_date`` and ``end_date`` (Date, both
        inclusive), one row per spell; touching spells of one permaticker
        are merged.

    Raises
    ------
    ValueError
        If the table is empty, a member to the table's end has no
        permaticker in TICKERS, or a member's ticker has several.

    Examples
    --------
    >>> sp500_intervals("/data/downloads/sharadar").columns
    ['permaticker', 'start_date', 'end_date']
    """
    rows = scan_raw_table(vendor_root, "sp500").select("date", "action", "ticker").collect()
    if rows.height == 0:
        raise ValueError(f"the SP500 table under {vendor_root} is empty")
    horizon: date = rows.get_column("date").max()
    coverage = date.fromisoformat(PIT_COVERAGE_START)
    spells: list[tuple[str, date, date]] = []
    for (ticker,), group in rows.sort("date").group_by(["ticker"], maintain_order=True):
        spells.extend(_ticker_spells(ticker, group, coverage, horizon))
    frame = pl.DataFrame(
        spells,
        schema={"ticker": pl.String, "start_date": pl.Date, "end_date": pl.Date},
        orient="row",
    )
    # A spell that ended before the prices begin is outside every bar a
    # store can hold, whether or not the company kept trading (and so has a
    # permaticker): without this, a roster built from an open start would
    # keep it and one built from the first priced day would not.
    early = frame.filter(pl.col("end_date") < coverage)
    frame = frame.filter(pl.col("end_date") >= coverage)
    mapping = permaticker_mapping(vendor_root, "sep")
    # A former member without a permaticker has no Sharadar prices (a few
    # that left soon after they begin). Only a member to the table's end is
    # left for `map_permatickers` to refuse, since a missing permaticker
    # there means TICKERS is stale.
    unmapped = frame.join(mapping.select("ticker"), on="ticker", how="anti")
    last_end = unmapped.group_by("ticker").agg(pl.col("end_date").max())
    gone = last_end.filter(pl.col("end_date") < horizon)
    if early.height:
        logger.info(
            f"SharadarSP500ConstituentDataset: {early.height} membership "
            f"spell(s) ended before {PIT_COVERAGE_START}, when Sharadar's "
            f"prices begin, and are dropped."
        )
    if gone.height:
        logger.warning(
            f"SharadarSP500ConstituentDataset: {gone.height} former "
            f"member(s) have no Sharadar prices and are dropped: "
            f"{gone.sort('ticker')['ticker'].to_list()[:20]}."
        )
        frame = frame.join(gone.select("ticker"), on="ticker", how="anti")
    frame = map_permatickers(
        frame,
        mapping,
        owner="SharadarSP500ConstituentDataset",
        code="sp500",
    )
    return _merge(frame.select("permaticker", "start_date", "end_date"))


def _ticker_spells(
    ticker: str, group: pl.DataFrame, coverage: date, horizon: date
) -> list[tuple[str, date, date]]:
    """Return one ticker's spells from its rows, sorted by date.

    ``removed`` rows sort before ``added`` rows of the same date, so a
    removal and re-addition on one day leaves no gap.
    """
    changes = (
        group.filter(pl.col("action").is_in(["added", "removed"]))
        .with_columns((pl.col("action") == "added").alias("_added"))
        .sort("date", "_added")
    )
    spells: list[tuple[str, date, date]] = []
    opened: date | None = None
    seen_change = False
    for day, action in changes.select("date", "action").iter_rows():
        if action == "added":
            if opened is None:
                opened = day
        else:
            # A removal with no earlier change: a member since before the table.
            start = opened if opened is not None else (None if seen_change else coverage)
            if start is not None and day - _ONE_DAY >= start:
                spells.append((ticker, start, day - _ONE_DAY))
            opened = None
        seen_change = True
    if opened is not None:
        spells.append((ticker, opened, horizon))
    elif not seen_change:
        # Only snapshots: a member from the table's start to its last snapshot.
        spells.append((ticker, coverage, group.get_column("date").max()))
    return spells


def _merge(frame: pl.DataFrame) -> pl.DataFrame:
    """Merge one permaticker's overlapping or touching spells into one."""
    merged: list[tuple[int, date, date]] = []
    for permaticker, start, end in frame.sort("permaticker", "start_date").iter_rows():
        if merged and merged[-1][0] == permaticker and start <= merged[-1][2] + _ONE_DAY:
            last = merged[-1]
            merged[-1] = (permaticker, last[1], max(last[2], end))
        else:
            merged.append((permaticker, start, end))
    return pl.DataFrame(
        merged,
        schema={"permaticker": pl.Int64, "start_date": pl.Date, "end_date": pl.Date},
        orient="row",
    )


class SharadarSP500ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time S&P 500 membership panel on the permaticker axis.

    The ``symbol`` axis is the int64 permaticker, the same id the Sharadar
    price panel uses, so the mask lines up with it without ticker matching.
    The panel ends on the SP500 table's last date, never today.

    Parameters
    ----------
    config : FileConstituentDatasetConfig
        ``cache_dir`` is ``<download-dir>/sharadar``, holding the ``sp500``
        and ``tickers`` raw tables.

    Examples
    --------
    >>> ds = SharadarSP500ConstituentDataset(config)
    >>> ds.from_raw_data().save()
    >>> ds.panel("2024-01-02", "2024-01-05")["is_member"].dtype
    dtype('bool')
    """

    #: The config class a saved ``config.json`` is rebuilt with.
    config_cls = FileConstituentDatasetConfig

    def _pit_coverage_start(self, config) -> str:
        """Return ``PIT_COVERAGE_START``."""
        return PIT_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        """Return the permaticker-keyed spells, as ``symbol``/``start_date``/``end_date``."""
        return sp500_intervals(self.config.cache_dir).rename({"permaticker": "symbol"})
