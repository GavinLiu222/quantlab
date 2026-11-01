"""Massive's dividends and splits as a daily panel on the permaticker axis.

Massive's ``/stocks/v1/dividends`` and ``/stocks/v1/splits`` records, as
``MassiveClient.pull`` keeps them in the REST raw tier
(``quantlab.dataset.massive.raw.read_rest_records``), become four variables
on SEP's trading calendar, one column per permaticker:

- ``dividend_declared``: from the trading day of a dividend's
  ``declaration_date`` through its ``ex_dividend_date``, the declared
  ``cash_amount`` per share. A security with several declared dividends
  pending shows the one going ex soonest (summed when several go ex that
  day).
- ``dividend_days_to_ex``: the trading days from the day to that ex-date,
  0 on the ex-date itself; counted on the NYSE calendar, so an ex-date past
  the store's last day counts too.
- ``dividend_cash``: on the ex-date, the ``cash_amount`` per share of every
  dividend going ex that day, summed.
- ``split_ratio``: on the ``execution_date``, ``split_to / split_from`` of
  every split that day, multiplied (2.0 a two-for-one split, 0.1 a
  one-for-ten reverse split).

A cell without one is NaN. ``cash_amount`` is the amount as declared, never
the vendor's split-adjusted amount, and ``historical_adjustment_factor`` is
never read: a later event never rewrites the panel's history. A vendor date
that is not a trading day rolls onto the next one, and a dividend declared
before the store's first day is shown from that day.

Each record is placed on the permaticker the Ticker map gives its ticker on
the record's own placement day (the ex-date, the execution date, the first
day a declared dividend is shown), through
``quantlab.dataset.massive.placement.TickerPlacement``; a record it cannot
place, and a dividend not paid in USD or without an amount, is left out
and counted in ``<store>.unplaced.json``. Only the universe's permatickers
(by default SEP's domestic common stock) are columns.

The store is built through the last day SEP, both REST pulls and the Ticker
map all reach, so ``update()`` appends a day only once each of them holds
it.

Examples
--------
Build the store and read a declared dividend::

    config = MassiveDividendsConfig(
        zarr_file_path="/data/quantlab/market/massive/massive_dividends_1d/massive_dividends_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
        massive_dir="/data/quantlab/downloads/massive",
        ticker_map_path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
        start_date="2016-10-10",
    )
    panel = MassiveDividendDataset(config).update().panel("2024-01-02", "2024-12-31")
    panel["dividend_days_to_ex"].sel(symbol=199059)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.config import MassiveDividendsConfig
from quantlab.dataset.massive.placement import TickerPlacement
from quantlab.dataset.massive.raw import rest_frame
from quantlab.dataset.massive.rest_panel import MassiveRestEventPanel
from quantlab.dataset.sharadar.universe import universe
from quantlab.utils.timer import Timer

__all__ = ["DIVIDEND_SCHEMA", "SPLIT_SCHEMA", "VARIABLES", "MassiveDividendDataset", "usd_dividends"]

#: The panel's variables and their attributes, in panel order.
VARIABLES: dict[str, dict] = {
    "dividend_declared": {
        "unit": "USD per share",
        "description": "Declared cash dividend, from its declaration date through its ex-date.",
    },
    "dividend_days_to_ex": {
        "unit": "trading days",
        "description": "Trading days from the day to the declared dividend's ex-date; 0 on the ex-date.",
    },
    "dividend_cash": {
        "unit": "USD per share",
        "description": "Cash dividends going ex on the day, summed.",
    },
    "split_ratio": {
        "unit": "new shares per old share",
        "description": "split_to / split_from of the splits executed on the day, multiplied.",
    },
}

#: The REST endpoints read (``quantlab.acquisition.massive.rest.ENDPOINTS``).
ENDPOINTS = ("dividends", "splits")

#: The fields of a dividend record read; ``historical_adjustment_factor`` is not one.
DIVIDEND_SCHEMA = {
    "ticker": pl.String,
    "cash_amount": pl.Float64,
    "currency": pl.String,
    "declaration_date": pl.String,
    "ex_dividend_date": pl.String,
}

#: The fields of a split record read.
SPLIT_SCHEMA = {
    "ticker": pl.String,
    "execution_date": pl.String,
    "split_from": pl.Float64,
    "split_to": pl.Float64,
}


def usd_dividends(dividends: pl.DataFrame, placement: TickerPlacement) -> pl.DataFrame:
    """Return the dividends with a USD cash amount and an ex-date; count the others as left out.

    Shared with the adjusted Day aggregates (``quantlab.dataset.massive.day_bar_events``).

    Examples
    --------
    With ``dividends`` read by ``rest_frame(..., DIVIDEND_SCHEMA)``::

        usd_dividends(dividends, placement)["currency"].unique().to_list()   # ['USD']
    """
    unusable = dividends.filter(
        pl.col("cash_amount").is_null() | (pl.col("currency").fill_null("") != "USD")
        | pl.col("ex_dividend_date").is_null()
    )
    for (currency,), rows in unusable.group_by(["currency"]):
        reason = (
            "no cash amount" if currency == "USD"
            else "no currency" if currency is None
            else f"currency {currency}, not USD"
        )
        placement.leave_out(rows.filter(pl.col("ex_dividend_date").is_not_null()), source="dividends",
                            reason=reason, day="ex_dividend_date")
    return dividends.filter(
        pl.col("cash_amount").is_not_null() & (pl.col("currency") == "USD") & pl.col("ex_dividend_date").is_not_null()
    )


class MassiveDividendDataset(MassiveRestEventPanel):
    """Daily panel of Massive's declared and paid dividends and splits, by permaticker.

    See the module docstring for the variables and the placement.

    Parameters
    ----------
    config : MassiveDividendsConfig
        Sharadar's raw tier (calendar, universe), Massive's raw tier and the
        Ticker map.

    Examples
    --------
    With ``config`` as in the module example::

        ds = MassiveDividendDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['dividend_declared', 'dividend_days_to_ex', 'dividend_cash', 'split_ratio']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MassiveDividendsConfig

    #: The REST endpoints read (``MassiveRestPanel``).
    REST_ENDPOINTS = ENDPOINTS

    #: The latest first day of the endpoints read (``MassiveRestPanel.EARLIEST``).
    EARLIEST = date(2000, 1, 1)

    #: The store folder under ``market/massive`` (``MassiveRestPanel.from_data_root``).
    STORE_NAME = "massive_dividends_1d"

    #: The store folder's ``README.md``.
    STORE_README = """# massive_dividends_1d

Massive's dividends and splits (its REST `/stocks/v1/dividends` and `/stocks/v1/splits`) as a
daily panel on Sharadar's permaticker axis and SEP's trading calendar, the whole market's
domestic common stock (SEP's default category filter). Variables, NaN on a day without one:
`dividend_declared` (declared cash per share, from the declaration date through the ex-date),
`dividend_days_to_ex` (trading days left to that ex-date, 0 on it), `dividend_cash` (cash per
share going ex that day, summed) and `split_ratio` (`split_to / split_from` executed that day,
multiplied). Amounts are as declared; Massive's `historical_adjustment_factor` is never read.

Each record is placed on the permaticker the Ticker map gives its ticker on the record's own
day; a record it cannot place, or a dividend not in USD, is left out and counted in
`massive_dividends_1d.zarr.unplaced.json`. Built by
`quantlab.dataset.massive.dividends.MassiveDividendDataset` from
`downloads/massive/rest/dividends`, `downloads/massive/rest/splits`,
`market/massive/massive_ticker_map` and Sharadar's SEP and TICKERS, from the Ticker map's first
day; first built by `scripts/massive/dividends.py`, then appended day by day by the daily data
update. docs/user-guide/massive.md, "Dividends and splits". Licensed data: personal use.
"""

    #: The module's ``VARIABLES``.
    VARIABLES = VARIABLES

    # -- the rows ---------------------------------------------------------------

    def _released(self) -> pl.DataFrame:
        """Return every security's values per day with one, placed and combined.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` and the four variables (null where a
            day has none of one), one row per pair, sorted by both.
        """
        calendar = self._trading_calendar()
        window = calendar.days_in(self.config.start_date, self.config.end_date)
        placement = TickerPlacement(self.ticker_map, owner=self.class_name)
        with Timer(f"{self.class_name}: place dividends and splits"):
            parts = []
            if window.len():
                first = window.min()
                dividends = rest_frame(self.config.massive_dir, "dividends", DIVIDEND_SCHEMA)
                dividends = usd_dividends(dividends, placement)
                parts = [
                    self._declared(dividends, calendar, window, placement),
                    self._ex_cash(dividends, calendar, first, placement),
                    self._splits(calendar, first, placement),
                ]
            placement.write_report(self.config.zarr_file_path)
        keep = [int(s) for s in universe(self.config)]
        schema = {"symbol": pl.Int64, "available": pl.Datetime("ns"), **{n: pl.Float64 for n in VARIABLES}}
        # Each part has one row per security and day; a day of several parts takes each one's values.
        frame = (
            pl.concat([pl.DataFrame(schema=schema), *parts], how="diagonal_relaxed")
            .group_by("symbol", "available")
            .agg(pl.col(name).drop_nulls().first() for name in VARIABLES)
        )
        return frame.filter(pl.col("symbol").is_in(keep)).select(list(schema)).sort("symbol", "available")

    def _ex_cash(self, dividends, calendar, first, placement) -> pl.DataFrame:
        """Return ``dividend_cash`` per security and ex-date."""
        rows = self._on_trading_days(dividends, calendar, "ex_dividend_date", first)
        rows = placement.place(rows, source="dividends:ex")
        return rows.group_by(pl.col("permaticker").alias("symbol"), "available").agg(
            pl.col("cash_amount").sum().alias("dividend_cash")
        )

    def _splits(self, calendar, first, placement) -> pl.DataFrame:
        """Return ``split_ratio`` per security and execution date."""
        splits = rest_frame(self.config.massive_dir, "splits", SPLIT_SCHEMA).filter(
            pl.col("execution_date").is_not_null()
        )
        valid = (pl.col("split_from") > 0) & (pl.col("split_to") > 0)
        placement.leave_out(splits.filter(~valid.fill_null(False)), source="splits",
                            reason="no positive split_from and split_to", day="execution_date")
        rows = self._on_trading_days(splits.filter(valid), calendar, "execution_date", first)
        rows = placement.place(rows, source="splits")
        return rows.group_by(pl.col("permaticker").alias("symbol"), "available").agg(
            (pl.col("split_to") / pl.col("split_from")).product().alias("split_ratio")
        )

    def _declared(self, dividends, calendar, window: pl.Series, placement) -> pl.DataFrame:
        """Return ``dividend_declared`` and ``dividend_days_to_ex`` on every day of each declared spell."""
        empty = pl.DataFrame(schema={"symbol": pl.Int64, "available": pl.Datetime("ns"),
                                     "dividend_declared": pl.Float64, "dividend_days_to_ex": pl.Float64})
        declared = dividends.filter(
            pl.col("declaration_date").is_not_null() & (pl.col("declaration_date") <= pl.col("ex_dividend_date"))
        )
        if declared.height == 0:
            return empty
        sessions = _Sessions()
        # The first day shown: the declaration's trading day, or the store's first day.
        declared = calendar.place(declared, on="declaration_date").with_columns(
            pl.max_horizontal("available", pl.lit(window.min())).alias("available")
        )
        declared = declared.with_columns(
            pl.Series("_ex", sessions.on_or_after(declared.get_column("ex_dividend_date")), dtype=pl.Date)
        ).filter(pl.col("available").cast(pl.Date) <= pl.col("_ex"))
        rows = placement.place(declared, source="dividends:declared")
        if rows.height == 0:
            return empty
        # Every window day from the first shown through the ex-date (or the window's end).
        days = window.to_numpy()
        start = np.searchsorted(days, rows.get_column("available").to_numpy(), side="left")
        stop = np.searchsorted(days, rows.get_column("_ex").cast(pl.Datetime("ns")).to_numpy(), side="right")
        calendar_rows = pl.DataFrame({"_index": np.arange(len(days), dtype=np.int64), "available": window})
        spells = (
            rows.select("permaticker", "cash_amount", "_ex")
            .with_columns(pl.int_ranges(pl.Series(start, dtype=pl.Int64), pl.Series(stop, dtype=pl.Int64))
                          .alias("_index"))
            .explode("_index", empty_as_null=True)
            .drop_nulls("_index")
            .join(calendar_rows, on="_index")
        )
        nearest = (
            spells.group_by("permaticker", "available")
            .agg(pl.col("_ex").min().alias("_nearest"))
        )
        shown = (
            spells.join(nearest, on=["permaticker", "available"])
            .filter(pl.col("_ex") == pl.col("_nearest"))
            .group_by(pl.col("permaticker").alias("symbol"), "available")
            .agg(pl.col("cash_amount").sum().alias("dividend_declared"), pl.col("_ex").first())
        )
        return shown.with_columns(
            pl.Series(
                "dividend_days_to_ex",
                sessions.between(shown.get_column("available").cast(pl.Date), shown.get_column("_ex")),
                dtype=pl.Float64,
            )
        ).select(empty.columns)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no Massive dividend or split is placed"


class _Sessions:
    """NYSE sessions, to count trading days up to an ex-date the store has not reached."""

    def __init__(self) -> None:
        sessions = XnysSessionCalendar().calendar.sessions
        self.days = np.asarray(sessions.tz_localize(None) if sessions.tz is not None else sessions,
                               dtype="datetime64[D]")

    def _position(self, days: pl.Series) -> np.ndarray:
        """Each day's position among the sessions, a day past the calendar's end counted in weekdays."""
        values = days.cast(pl.Date).to_numpy().astype("datetime64[D]")
        position = np.searchsorted(self.days, values, side="left")
        last = self.days[-1]
        beyond = values > last
        if beyond.any():
            position[beyond] = len(self.days) - 1 + np.busday_count(last, values[beyond])
        return position

    def on_or_after(self, days: pl.Series) -> list[date]:
        """Return each day's first session on or after it."""
        position = self._position(days)
        values = days.cast(pl.Date).to_numpy().astype("datetime64[D]")
        inside = position < len(self.days)
        out = values.copy()
        out[inside] = self.days[position[inside]]
        return [pd.Timestamp(day).date() for day in out]

    def between(self, first: pl.Series, last: pl.Series) -> np.ndarray:
        """Return the sessions after ``first`` up to ``last``, both sessions."""
        return (self._position(last) - self._position(first)).astype(np.float64)
