"""Massive's own splits and dividends as the events of its Day aggregates on permatickers (ADR 0048).

The permaticker Day aggregate store (``MappedDayBarDataset``,
``quantlab.dataset.massive.mapped_bars``) chains its adjusted prices from
Massive's ``/stocks/v1`` splits and dividends, as the REST pull keeps them
in the raw tier, never from Sharadar's ACTIONS and never from the vendor's
``historical_adjustment_factor``. ``day_bar_events`` turns those records
into the chain's two event variables on the store's own bars:

- ``divCash``: the ``cash_amount`` per share of every USD dividend going ex
  that day, summed (the amount as declared, which is cash per share held on
  the ex-date);
- ``splitFactor``: ``split_to / split_from`` of every split executed that
  day, multiplied.

A vendor date that is not one of the store's sessions rolls onto the next
one. Each record is placed on the permaticker the Ticker map gives its
ticker on that session (``quantlab.dataset.massive.placement.TickerPlacement``),
so a reused ticker sends each event to the security that held it that day.
Where two tickers map to one permaticker on a day, the column holds the
busier one's bar (``PermatickerDataset.columns``), and only that ticker's
events are the column's. A record left out (unplaced, another currency, no
positive split ratio, another ticker's bar) is counted in
``<store>.unplaced.json``, never guessed.

Examples
--------
With ``view`` the permaticker view of the Day aggregate store and
``sessions`` the store's days::

    events = day_bar_events(view, sessions, owner="MappedDayBarDataset", report_path=store_path)
    events.columns   # ['timestamp', 'symbol', 'divCash', 'splitFactor']
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from quantlab.dataset.massive.dividends import DIVIDEND_SCHEMA, SPLIT_SCHEMA, usd_dividends
from quantlab.dataset.massive.permaticker_bars import PermatickerDataset
from quantlab.dataset.massive.placement import TickerPlacement
from quantlab.dataset.massive.raw import rest_frame

__all__ = ["ANOTHER_TICKER", "EVENT_ENDPOINTS", "day_bar_events"]

#: The REST endpoints the events come from.
EVENT_ENDPOINTS = ("dividends", "splits")

#: Why an event is left out when another ticker holds the permaticker's bar that day.
ANOTHER_TICKER = "another ticker holds the permaticker's bar that day"

_EVENT_SCHEMA = {
    "timestamp": pl.Datetime("ns"),
    "symbol": pl.Int64,
    "divCash": pl.Float64,
    "splitFactor": pl.Float64,
}


def _on_sessions(rows: pl.DataFrame, on: str, sessions: list[date]) -> pl.DataFrame:
    """Return ``rows`` with ``available``, the first session on or after ``on``; rows past the last dropped."""
    days = np.array(sessions, dtype="datetime64[D]")
    vendor = rows.get_column(on).to_numpy().astype("datetime64[D]")
    at = np.searchsorted(days, vendor, side="left")
    inside = at < len(days)
    available = np.full(len(rows), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    available[inside] = days[at[inside]]
    return (
        rows.with_columns(pl.Series("available", available).cast(pl.Date))
        .filter(pl.col("available").is_not_null() & (pl.col(on) >= pl.lit(sessions[0])))
    )


def _the_columns(view: PermatickerDataset, placed: pl.DataFrame, placement: TickerPlacement, source: str) -> pl.DataFrame:
    """Return the placed events whose ticker holds its permaticker's column that day; count the others."""
    if not placed.height:
        return placed
    pairs = placed.select("available", "ticker", "permaticker").unique()
    mapped = view.ticker_map.read_days(pairs.get_column("available").unique().to_list()).filter(
        pl.col("permaticker").is_not_null()
        & pl.col("permaticker").is_in(pairs.get_column("permaticker").unique().to_list())
        & pl.col("ticker").is_in(view.ticker_bars.stored_symbols())
    )
    clashes = (
        mapped.group_by("date", "permaticker").len().filter(pl.col("len") > 1)
        .select(pl.col("date").alias("available"), pl.col("permaticker").cast(pl.Int64))
    )
    contested = placed.join(clashes, on=["available", "permaticker"], how="semi")
    if not contested.height:
        return placed
    held = pl.concat(
        [view.columns(day, day) for day in sorted(contested.get_column("available").unique().to_list())]
    ).select(pl.col("date").alias("available"), "ticker", "permaticker")
    others = contested.join(held, on=["available", "ticker", "permaticker"], how="anti")
    placement.leave_out(others, source=source, reason=ANOTHER_TICKER, day="available")
    return placed.join(others.select("available", "ticker", "permaticker").unique(),
                       on=["available", "ticker", "permaticker"], how="anti")


def day_bar_events(
    view: PermatickerDataset, sessions: list[date], *, owner: str, report_path: str | None = None
) -> pl.DataFrame:
    """Return Massive's split and dividend events on the permaticker columns of ``sessions``.

    Parameters
    ----------
    view : PermatickerDataset
        The permaticker view of the Day aggregate store; its Day aggregate
        dataset's raw tier holds the REST pulls.
    sessions : list of datetime.date
        The store's days, in order; events are placed on them only.
    owner : str
        Named in the report and in log lines.
    report_path : str, optional
        The store whose ``<store>.unplaced.json`` lists the records left
        out (written, or removed when none is); ``None`` writes nothing.

    Returns
    -------
    pl.DataFrame
        ``timestamp`` (the session), ``symbol`` (the permaticker),
        ``divCash`` and ``splitFactor``, one row per pair with an event.

    Examples
    --------
    With ``view`` a ``PermatickerDayBarDataset`` over a raw tier on disk::

        day_bar_events(view, [date(2024, 3, 6)], owner="demo").row(0)
        # (datetime.datetime(2024, 3, 6, 0, 0), 101, 0.0, 2.0)
    """
    placement = TickerPlacement(view.ticker_map, owner=owner)
    vendor_root = view.ticker_bars.config.raw_data_dir_path
    parts = []
    if sessions:
        dividends = usd_dividends(rest_frame(vendor_root, "dividends", DIVIDEND_SCHEMA), placement)
        rows = _on_sessions(dividends, "ex_dividend_date", sessions)
        rows = _the_columns(view, placement.place(rows, source="dividends"), placement, "dividends")
        parts.append(rows.group_by("permaticker", "available").agg(
            pl.col("cash_amount").sum().alias("divCash"), pl.lit(1.0).alias("splitFactor")
        ))
        splits = rest_frame(vendor_root, "splits", SPLIT_SCHEMA).filter(pl.col("execution_date").is_not_null())
        valid = (pl.col("split_from") > 0) & (pl.col("split_to") > 0)
        placement.leave_out(splits.filter(~valid.fill_null(False)), source="splits",
                            reason="no positive split_from and split_to", day="execution_date")
        rows = _on_sessions(splits.filter(valid), "execution_date", sessions)
        rows = _the_columns(view, placement.place(rows, source="splits"), placement, "splits")
        parts.append(rows.group_by("permaticker", "available").agg(
            pl.lit(0.0).alias("divCash"), (pl.col("split_to") / pl.col("split_from")).product().alias("splitFactor")
        ))
    if report_path is not None:
        placement.write_report(report_path)
    if not parts:
        return pl.DataFrame(schema=_EVENT_SCHEMA)
    return (
        pl.concat(parts, how="diagonal_relaxed")
        .group_by("permaticker", "available")
        .agg(pl.col("divCash").sum(), pl.col("splitFactor").product())
        .select(
            pl.col("available").cast(pl.Datetime("ns")).alias("timestamp"),
            pl.col("permaticker").cast(pl.Int64).alias("symbol"),
            pl.col("divCash").cast(pl.Float64),
            pl.col("splitFactor").cast(pl.Float64),
        )
        .sort("symbol", "timestamp")
    )


def session_days(timestamps) -> list[date]:
    """Return the dates of a store's bar timestamps, in order.

    Examples
    --------
    >>> session_days(np.array(["2024-03-04"], dtype="datetime64[ns]"))
    [datetime.date(2024, 3, 4)]
    """
    return [stamp.date() for stamp in pd.DatetimeIndex(timestamps)]
