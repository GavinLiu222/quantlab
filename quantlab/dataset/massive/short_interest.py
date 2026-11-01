"""Massive's short interest as a daily release panel on the permaticker axis.

Massive's ``/stocks/v1/short-interest`` records (FINRA's short interest,
settled twice a month), as ``MassiveClient.pull`` keeps them in the REST raw
tier (``quantlab.dataset.massive.raw.read_rest_records``), become three
variables on SEP's trading calendar, one column per permaticker:

- ``short_interest``: the shares sold short and not yet covered at the
  settlement date;
- ``short_avg_daily_volume``: the vendor's average daily volume over the
  settlement period (``avg_daily_volume``);
- ``days_to_cover``: the vendor's ``short_interest / avg_daily_volume``.

Each settlement date is a *release*. FINRA publishes it about seven business
days after settlement, so a release becomes visible on the 8th trading day
after its settlement date (``release_lag_bars``) and is shown until the next
release becomes visible. A security the next release does not carry shows
nothing from then on: a cell shows only the release visible that day, never
an older one.

Each record is placed on the permaticker the Ticker map gives its ticker on
the settlement date (the day the figure is about), through
``quantlab.dataset.massive.placement.TickerPlacement``; a record it cannot
place is left out and counted in ``<store>.unplaced.json``. A record pulled
again with other values (a revision) replaces the earlier one. Only the
universe's permatickers (by default SEP's domestic common stock) are columns.

The store is built through the last day SEP, the REST pull and the Ticker
map all reach, so ``update()`` appends a day only once each of them holds it.

Examples
--------
Build the store and read one security's short interest::

    config = MassiveShortInterestConfig(
        zarr_file_path="/data/quantlab/market/massive/massive_short_interest_1d/massive_short_interest_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
        massive_dir="/data/quantlab/downloads/massive",
        ticker_map_path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
        start_date="2016-10-10",
    )
    panel = MassiveShortInterestDataset(config).update().panel("2024-01-02", "2024-12-31")
    panel["days_to_cover"].sel(symbol=199059)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import xarray as xr

from quantlab.dataset.config import DatasetConfig, MassiveShortInterestConfig
from quantlab.dataset.massive.placement import TickerPlacement
from quantlab.dataset.massive.raw import rest_frame
from quantlab.dataset.massive.rest_panel import MassiveRestEventPanel
from quantlab.dataset.sharadar.point_in_time import TradingCalendar, on_grid, to_panel
from quantlab.dataset.sharadar.universe import universe
from quantlab.utils.timer import Timer

__all__ = ["VARIABLES", "MassiveShortInterestDataset"]

#: The panel's variables and their attributes, in panel order.
VARIABLES: dict[str, dict] = {
    "short_interest": {
        "unit": "shares",
        "description": "Shares sold short and not yet covered at the settlement date of the visible release.",
    },
    "short_avg_daily_volume": {
        "unit": "shares",
        "description": "Massive's average daily volume over the visible release's settlement period.",
    },
    "days_to_cover": {
        "unit": "days",
        "description": "short_interest / avg_daily_volume of the visible release, as Massive gives it.",
    },
}

#: The REST endpoint read (``quantlab.acquisition.massive.rest.ENDPOINTS``).
ENDPOINT = "short_interest"

#: The fields of a record read, by the vendor's names.
SCHEMA = {
    "ticker": pl.String,
    "settlement_date": pl.String,
    "short_interest": pl.Float64,
    "avg_daily_volume": pl.Float64,
    "days_to_cover": pl.Float64,
}

#: The vendor's field of each variable.
FIELDS = {"short_interest": "short_interest", "short_avg_daily_volume": "avg_daily_volume",
          "days_to_cover": "days_to_cover"}


class MassiveShortInterestDataset(MassiveRestEventPanel):
    """Daily release panel of Massive's short interest, by permaticker.

    See the module docstring for the variables and the release rule.

    Parameters
    ----------
    config : MassiveShortInterestConfig
        Sharadar's raw tier (calendar, universe), Massive's raw tier and the
        Ticker map.

    Examples
    --------
    With ``config`` as in the module example::

        ds = MassiveShortInterestDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['short_interest', 'short_avg_daily_volume', 'days_to_cover']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MassiveShortInterestConfig

    #: The REST endpoints read (``MassiveRestPanel``).
    REST_ENDPOINTS = (ENDPOINT,)

    #: The latest first day of the endpoints read (``MassiveRestPanel.EARLIEST``).
    EARLIEST = date(2017, 12, 1)

    #: The store folder under ``market/massive`` (``MassiveRestPanel.from_data_root``).
    STORE_NAME = "massive_short_interest_1d"

    #: The store folder's ``README.md``.
    STORE_README = """# massive_short_interest_1d

Massive's short interest (its REST `/stocks/v1/short-interest`, FINRA's figures settled twice a
month from 2017-12) as a daily release panel on Sharadar's permaticker axis and SEP's trading
calendar, the whole market's domestic common stock (SEP's default category filter). Variables:
`short_interest` (shares short at the settlement date), `short_avg_daily_volume` (Massive's
`avg_daily_volume`) and `days_to_cover` (Massive's own ratio). A release is visible from the 8th
trading day after its settlement date and shown until the next release is visible; a security
the visible release does not carry is NaN.

Each record is placed on the permaticker the Ticker map gives its ticker on the settlement date;
a record it cannot place is left out and counted in
`massive_short_interest_1d.zarr.unplaced.json`. Built by
`quantlab.dataset.massive.short_interest.MassiveShortInterestDataset` from
`downloads/massive/rest/short_interest`, `market/massive/massive_ticker_map` and Sharadar's SEP
and TICKERS, from the Ticker map's first day; first built by
`scripts/massive/short_interest.py`, then appended day by day by the daily data update.
docs/user-guide/massive.md, "Short interest". Licensed data: personal use.
"""

    #: The module's ``VARIABLES``.
    VARIABLES = VARIABLES

    def _normalize_config(self, config: DatasetConfig) -> MassiveShortInterestConfig:
        """Normalise as the base class does, then check the table, the lag and the universe fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``MassiveShortInterestConfig``.
        ValueError
            If ``table`` is not ``"sep"``, ``release_lag_bars`` is negative or
            a universe field is invalid.
        """
        config = super()._normalize_config(config)
        if config.release_lag_bars < 0:
            raise ValueError(f"{self.class_name}: release_lag_bars must be >= 0, got {config.release_lag_bars}.")
        return config

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._schedule_cache: pl.DataFrame | None = None

    # -- the rows ---------------------------------------------------------------

    def _records(self) -> pl.DataFrame:
        """Return the records, the newest copy of each ticker and settlement date, dates as ``pl.Date``."""
        return (
            rest_frame(self.config.massive_dir, ENDPOINT, SCHEMA, key=None)
            .filter(pl.col("ticker").is_not_null() & pl.col("settlement_date").is_not_null())
            # Oldest pull first: the last copy is the newest.
            .unique(["ticker", "settlement_date"], keep="last", maintain_order=True)
        )

    def _visible(self, records: pl.DataFrame, calendar: TradingCalendar) -> pl.DataFrame:
        """Return the records the calendar shows, with ``_settled`` and ``available``.

        ``_settled`` is the settlement date's trading day (the placement day),
        ``available`` the trading day ``release_lag_bars`` after it; a record
        whose release is not visible by the calendar's last day is dropped,
        and so is one settled before the calendar's first day.
        """
        settled = calendar.place(records, on="settlement_date", drop_earlier=True).rename({"available": "_settled"})
        days = calendar.days.to_numpy()
        position = np.searchsorted(days, settled.get_column("_settled").to_numpy(), side="left")
        position = position + self.config.release_lag_bars
        visible = position < len(days)
        return settled.filter(pl.Series(visible)).with_columns(
            pl.Series("available", days[position[visible]], dtype=pl.Datetime("ns"))
        )

    def _released(self) -> pl.DataFrame:
        """Return every placed record at its release's available day.

        Also sets the release schedule (each settlement date and the day it
        becomes visible) the window reads.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available``, ``settlement_date`` and the three
            variables, one row per security and release, sorted by
            ``available``.
        """
        calendar = self._trading_calendar()
        window = calendar.days_in(self.config.start_date, self.config.end_date)
        placement = TickerPlacement(self.ticker_map, owner=self.class_name)
        schema = {"symbol": pl.Int64, "available": pl.Datetime("ns"), "settlement_date": pl.Date,
                  **{name: pl.Float64 for name in VARIABLES}}
        frame = pl.DataFrame(schema=schema)
        with Timer(f"{self.class_name}: place short interest"):
            records = self._visible(self._records(), calendar)
            self._schedule_cache = (
                records.select("settlement_date", "available").unique().sort("available", "settlement_date")
            )
            if window.len():
                # A release replaced before the store's first day is never shown.
                before = self._schedule_cache.filter(pl.col("available") <= window.min())
                if before.height:
                    records = records.filter(pl.col("settlement_date") >= before.get_column("settlement_date").max())
                placed = placement.place(records, source=ENDPOINT, day="_settled")
                frame = (
                    placed.sort("permaticker", "available", "ticker")
                    .group_by(pl.col("permaticker").alias("symbol"), "available", "settlement_date",
                              maintain_order=True)
                    .agg(pl.col(field).first().alias(name) for name, field in FIELDS.items())
                    .select(list(schema))
                )
            placement.write_report(self.config.zarr_file_path)
        keep = [int(s) for s in universe(self.config)]
        return frame.filter(pl.col("symbol").is_in(keep)).sort("available", "symbol")

    def _schedule(self) -> pl.DataFrame:
        """Return each release's settlement date and the trading day it becomes visible."""
        self._released_rows()  # builds the schedule with the rows
        return self._schedule_cache

    # -- the panel: the release visible each day ----------------------------------

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no Massive short interest release is visible"

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list[int] | None = None) -> xr.Dataset:
        """Return one window: each security's row of the release visible that day, NaN elsewhere."""
        days, symbols = self._window_axes(start_date, end_date, symbols)
        rows = self._released_rows().filter(pl.col("symbol").is_in(symbols))
        current = days.to_frame().join_asof(
            self._schedule().rename({"settlement_date": "_current"}),
            left_on="timestamp", right_on="available", strategy="backward",
        ).select("timestamp", "_current")
        cells = (
            on_grid(days, symbols)
            .join_asof(rows, left_on="timestamp", right_on="available", by="symbol", strategy="backward",
                       # Both sides are sorted by time; polars cannot check that per group.
                       check_sortedness=False)
            .join(current, on="timestamp", how="left")
            .sort("_day", "_column")
        )
        shown = pl.col("settlement_date") == pl.col("_current")
        cells = cells.with_columns(
            pl.when(shown.fill_null(False)).then(pl.col(name)).otherwise(None)
            .cast(pl.Float64).fill_null(float("nan")).alias(name)
            for name in VARIABLES
        )
        return to_panel(cells, days, symbols, VARIABLES)

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count the days each added permaticker shows a release between ``start`` and ``end``."""
        if not added:
            return {}
        window = self._raw_data_to_xr_window(start, end, symbols=[int(s) for s in added])
        counts = window["short_interest"].notnull().sum("timestamp")
        return {
            str(symbol): int(count)
            for symbol, count in zip(window["symbol"].values.tolist(), counts.values.tolist())
            if count
        }
