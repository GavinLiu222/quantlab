"""Massive's IPOs as a daily panel on the permaticker axis.

Massive's ``/vX/reference/ipos`` records, as ``MassiveClient.pull`` keeps
them in the REST raw tier (``quantlab.dataset.massive.raw.read_rest_records``),
become three variables on SEP's trading calendar, one column per
permaticker, each on the IPO's listing day only:

- ``ipo_offer_price``: the final offer price per share
  (``final_issue_price``).
- ``ipo_offer_size``: the total amount the offer raised
  (``total_offer_size``).
- ``ipo_days_announced``: the trading days from the announcement
  (``announced_date``, rolled onto the next NYSE session when it is not
  one) to the listing day, 0 when the IPO lists on the day it is announced.

A cell without one is NaN. Only an IPO that listed is shown: its
``ipo_status`` is ``new``, ``history`` or ``direct_listing_process`` and it
has a ``listing_date``. A pending, rumoured, postponed or withdrawn IPO
stays in the raw tier only and is not counted as left out. The records
have no id and a revision changes their fields (a pending IPO becomes new,
then history), so each IPO, a ticker and its announcement date, is read
from its newest copy (the latest ``last_updated``, then the latest pull).
A listing date that is not a trading day rolls onto the next one; an IPO
listed before the store's first day is not shown.

Each listed IPO is placed on the permaticker the Ticker map gives its
ticker on the listing day, through
``quantlab.dataset.massive.placement.TickerPlacement``; an IPO it cannot
place, and one not in USD, is left out and counted in
``<store>.unplaced.json``. Only the universe's permatickers (by default
SEP's domestic common stock) are columns.

The store is built through the last day SEP, the REST pull and the Ticker
map all reach, so ``update()`` appends a day only once each of them holds
it.

Examples
--------
Build the store and read the IPOs of a year::

    config = MassiveIposConfig(
        zarr_file_path="/data/quantlab/market/massive/massive_ipos_1d/massive_ipos_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
        massive_dir="/data/quantlab/downloads/massive",
        ticker_map_path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
        start_date="2016-10-10",
    )
    panel = MassiveIpoDataset(config).update().panel("2024-01-02", "2024-12-31")
    panel["ipo_offer_price"].max("timestamp").dropna("symbol")
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.config import MassiveIposConfig
from quantlab.dataset.massive.placement import TickerPlacement
from quantlab.dataset.massive.raw import rest_frame
from quantlab.dataset.massive.rest_panel import MassiveRestEventPanel
from quantlab.dataset.sharadar.point_in_time import TradingCalendar
from quantlab.dataset.sharadar.universe import universe
from quantlab.utils.timer import Timer

__all__ = ["LISTED", "VARIABLES", "MassiveIpoDataset"]

#: The panel's variables and their attributes, in panel order.
VARIABLES: dict[str, dict] = {
    "ipo_offer_price": {
        "unit": "USD per share",
        "description": "Final offer price of the IPO listing on the day (final_issue_price).",
    },
    "ipo_offer_size": {
        "unit": "USD",
        "description": "Total amount raised by the IPO listing on the day (total_offer_size).",
    },
    "ipo_days_announced": {
        "unit": "trading days",
        "description": "Trading days from the IPO's announcement to its listing on the day; 0 on the same day.",
    },
}

#: The REST endpoint read (``quantlab.acquisition.massive.rest.ENDPOINTS``).
ENDPOINT = "ipos"

#: The ``ipo_status`` values of an IPO that listed; every other one stays raw only.
LISTED = ("new", "history", "direct_listing_process")

#: The fields of an IPO record read.
SCHEMA = {
    "ticker": pl.String,
    "ipo_status": pl.String,
    "currency_code": pl.String,
    "final_issue_price": pl.Float64,
    "total_offer_size": pl.Float64,
    "listing_date": pl.String,
    "announced_date": pl.String,
    "last_updated": pl.String,
}


def _newest_copies(vendor_root) -> pl.DataFrame:
    """Return each IPO's newest copy, dates as ``pl.Date``.

    An IPO is a ticker and its announcement date; its newest copy has the
    latest ``last_updated``, then comes from the latest pull.
    """
    return (
        rest_frame(vendor_root, ENDPOINT, SCHEMA, key=None)
        .with_columns(pl.col("last_updated").str.slice(0, 10).str.to_date(strict=False))
        .with_row_index("_order")
        .sort("last_updated", "_order", nulls_last=False)
        .unique(["ticker", "announced_date"], keep="last", maintain_order=True)
        .drop("_order")
    )


class MassiveIpoDataset(MassiveRestEventPanel):
    """Daily panel of Massive's listed IPOs on their listing day, by permaticker.

    See the module docstring for the variables and the placement.

    Parameters
    ----------
    config : MassiveIposConfig
        Sharadar's raw tier (calendar, universe), Massive's raw tier and the
        Ticker map.

    Examples
    --------
    With ``config`` as in the module example::

        ds = MassiveIpoDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['ipo_offer_price', 'ipo_offer_size', 'ipo_days_announced']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MassiveIposConfig

    #: The REST endpoints read (``MassiveRestPanel``).
    REST_ENDPOINTS = (ENDPOINT,)

    #: The latest first day of the endpoints read (``MassiveRestPanel.EARLIEST``).
    EARLIEST = date(2008, 12, 1)

    #: The store folder under ``market/massive`` (``MassiveRestPanel.from_data_root``).
    STORE_NAME = "massive_ipos_1d"

    #: The store folder's ``README.md``.
    STORE_README = """# massive_ipos_1d

Massive's IPOs (its REST `/vX/reference/ipos`) as a daily panel on Sharadar's permaticker axis
and SEP's trading calendar, the whole market's domestic common stock (SEP's default category
filter). On an IPO's listing day only, NaN otherwise: `ipo_offer_price` (final offer price per
share, `final_issue_price`), `ipo_offer_size` (total amount raised, `total_offer_size`) and
`ipo_days_announced` (trading days from the announcement to the listing, 0 on the same day).
Only IPOs that listed (`ipo_status` new, history or direct_listing_process) are shown; pending,
rumoured, postponed and withdrawn IPOs stay in the raw tier. Each IPO is read from its newest
copy.

Each IPO is placed on the permaticker the Ticker map gives its ticker on the listing day; one it
cannot place, or not in USD, is left out and counted in `massive_ipos_1d.zarr.unplaced.json`.
Built by `quantlab.dataset.massive.ipos.MassiveIpoDataset` from `downloads/massive/rest/ipos`,
`market/massive/massive_ticker_map` and Sharadar's SEP and TICKERS, from the Ticker map's first
day; first built by `scripts/massive/ipos.py`, then appended day by day by the daily data
update. docs/user-guide/massive.md, "IPOs". Licensed data: personal use.
"""

    #: The module's ``VARIABLES``.
    VARIABLES = VARIABLES

    # -- the rows ---------------------------------------------------------------

    def _released(self) -> pl.DataFrame:
        """Return every listed IPO's values on its listing day, placed.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` and the three variables, one row per
            pair, sorted by both.
        """
        calendar = self._trading_calendar()
        window = calendar.days_in(self.config.start_date, self.config.end_date)
        placement = TickerPlacement(self.ticker_map, owner=self.class_name)
        schema = {"symbol": pl.Int64, "available": pl.Datetime("ns"), **{n: pl.Float64 for n in VARIABLES}}
        frame = pl.DataFrame(schema=schema)
        with Timer(f"{self.class_name}: place IPOs"):
            if window.len():
                frame = self._listed(calendar, window.min(), placement)
            placement.write_report(self.config.zarr_file_path)
        keep = [int(s) for s in universe(self.config)]
        return frame.filter(pl.col("symbol").is_in(keep)).select(list(schema)).sort("symbol", "available")

    def _listed(self, calendar: TradingCalendar, first, placement: TickerPlacement) -> pl.DataFrame:
        """Return the listed IPOs in the window, placed, one row per security and listing day."""
        ipos = _newest_copies(self.config.massive_dir).filter(
            pl.col("ipo_status").is_in(LISTED) & pl.col("listing_date").is_not_null()
        )
        rows = self._on_trading_days(ipos, calendar, "listing_date", first)
        usd = pl.col("currency_code").fill_null("") == "USD"
        for (currency,), unusable in rows.filter(~usd).group_by(["currency_code"]):
            reason = "no currency" if currency is None else f"currency {currency}, not USD"
            placement.leave_out(unusable, source=ENDPOINT, reason=reason, day="available")
        rows = placement.place(rows.filter(usd), source=ENDPOINT)
        rows = rows.with_columns(
            pl.Series("ipo_days_announced", _sessions_between(rows.get_column("announced_date"),
                                                              rows.get_column("available")), dtype=pl.Float64)
        )
        return (
            rows.sort("last_updated", nulls_last=False)
            .group_by(pl.col("permaticker").alias("symbol"), "available")
            .agg(
                pl.col("final_issue_price").last().alias("ipo_offer_price"),
                pl.col("total_offer_size").last().alias("ipo_offer_size"),
                pl.col("ipo_days_announced").last(),
            )
        )

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no listed Massive IPO is placed"



def _sessions_between(announced: pl.Series, listed: pl.Series) -> np.ndarray:
    """Return the NYSE sessions from each announcement (rolled forward) to its listing day.

    NaN without an announcement date, for one before the session calendar's
    first day, and for one after the listing.
    """
    sessions = XnysSessionCalendar().calendar.sessions
    days = np.asarray(sessions.tz_localize(None) if sessions.tz is not None else sessions, dtype="datetime64[D]")
    out = np.full(len(listed), np.nan)
    if not len(listed):
        return out
    start = announced.cast(pl.Date).to_numpy().astype("datetime64[D]")
    stop = listed.cast(pl.Date).to_numpy().astype("datetime64[D]")
    known = ~np.isnat(start) & (start >= days[0])
    count = (np.searchsorted(days, stop, side="left") - np.searchsorted(days, start, side="left")).astype(np.float64)
    valid = known & (count >= 0)
    out[valid] = count[valid]
    return out
