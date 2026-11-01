"""Massive's daily short volume as a panel on the permaticker axis.

Massive's ``/stocks/v1/short-volume`` records (FINRA's daily short sale
volume, one record per ticker and day, from 2024-02-06), as
``MassiveClient.pull`` keeps them in the REST raw tier, become four
variables on SEP's trading calendar, one column per permaticker, each on
the record's own trading day:

- ``short_volume``: the shares sold short across all venues (the vendor's
  ``short_volume``);
- ``short_total_volume``: the total volume reported across all venues
  (``total_volume``);
- ``short_exempt_volume``: the short volume marked exempt from Regulation
  SHO (``exempt_volume``);
- ``short_volume_ratio``: the percent of the total volume sold short
  (``short_volume_ratio``, ``100 * short_volume / total_volume``).

A cell without a record is NaN. The per-venue columns (ADF, Nasdaq
Carteret, Nasdaq Chicago, NYSE) stay in the raw tier. A record re-read by a
later pull (FINRA's revision, the pull's overlap days) replaces the earlier
copy of its ticker and day; a revision of a day already in the store
reaches it only when the store is rebuilt, as the store only appends days.
When two tickers place on one permaticker on one day their volumes are
summed and the ratio is recomputed from the sums.

Each record is placed on the permaticker the Ticker map gives its ticker on
the record's day, through
``quantlab.dataset.massive.placement.TickerPlacement``; a record it cannot
place is left out and counted in ``<store>.unplaced.json``. Only the
universe's permatickers (by default SEP's domestic common stock) are
columns.

The store is built through the last day SEP, the REST pull and the Ticker
map all reach, so ``update()`` appends a day only once each of them holds
it.

Examples
--------
Build the store and read one security's short volume ratio::

    config = MassiveShortVolumeConfig(
        zarr_file_path="/data/quantlab/market/massive/massive_short_volume_1d/massive_short_volume_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
        massive_dir="/data/quantlab/downloads/massive",
        ticker_map_path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
        start_date="2024-02-06",
    )
    panel = MassiveShortVolumeDataset(config).update().panel("2024-03-01", "2024-03-29")
    panel["short_volume_ratio"].sel(symbol=199059)
"""

from __future__ import annotations

from datetime import date

import polars as pl

from quantlab.dataset.config import MassiveShortVolumeConfig
from quantlab.dataset.massive.placement import TickerPlacement
from quantlab.dataset.massive.raw import rest_page_files
from quantlab.dataset.massive.rest_panel import MassiveRestEventPanel
from quantlab.dataset.sharadar.universe import universe
from quantlab.utils.timer import Timer

__all__ = ["VARIABLES", "MassiveShortVolumeDataset"]

#: The panel's variables and their attributes, in panel order.
VARIABLES: dict[str, dict] = {
    "short_volume": {
        "unit": "shares",
        "description": "Shares sold short across all venues on the day (Massive's short_volume).",
    },
    "short_total_volume": {
        "unit": "shares",
        "description": "Total volume reported across all venues on the day (Massive's total_volume).",
    },
    "short_exempt_volume": {
        "unit": "shares",
        "description": "Short volume marked exempt from Regulation SHO (Massive's exempt_volume).",
    },
    "short_volume_ratio": {
        "unit": "percent",
        "description": "100 * short_volume / short_total_volume (Massive's short_volume_ratio).",
    },
}

#: The REST endpoint read (``quantlab.acquisition.massive.rest.ENDPOINTS``).
ENDPOINT = "short_volume"

#: The fields of a record read; the per-venue columns are not among them.
SCHEMA = {
    "ticker": pl.String,
    "date": pl.String,
    "short_volume": pl.Float64,
    "total_volume": pl.Float64,
    "exempt_volume": pl.Float64,
}


def _records(vendor_root) -> pl.DataFrame:
    """Return every (ticker, date)'s newest record, ``date`` as ``pl.Date``.

    Pages are read oldest pull first, so the last copy of a ticker and day
    is the newest.
    """
    pages = [
        pl.read_ndjson(path, schema=SCHEMA).with_columns(pl.lit(index, pl.Int64).alias("_page"))
        for index, path in enumerate(rest_page_files(vendor_root, ENDPOINT))
    ]
    frame = pl.concat([pl.DataFrame(schema={**SCHEMA, "_page": pl.Int64}), *pages])
    return (
        frame.with_columns(pl.col("date").str.slice(0, 10).str.to_date(strict=False))
        .filter(pl.col("ticker").is_not_null() & pl.col("date").is_not_null())
        .with_row_index("_row")
        .sort("_page", "_row")
        .unique(["ticker", "date"], keep="last", maintain_order=True)
        .drop("_page", "_row")
    )


class MassiveShortVolumeDataset(MassiveRestEventPanel):
    """Daily panel of Massive's short volume totals, by permaticker.

    See the module docstring for the variables and the placement.

    Parameters
    ----------
    config : MassiveShortVolumeConfig
        Sharadar's raw tier (calendar, universe), Massive's raw tier and the
        Ticker map.

    Examples
    --------
    With ``config`` as in the module example::

        ds = MassiveShortVolumeDataset(config).update()
        list(ds.panel("2024-03-01", "2024-03-05").data_vars)
        # ['short_volume', 'short_total_volume', 'short_exempt_volume', 'short_volume_ratio']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MassiveShortVolumeConfig

    #: The REST endpoints read (``MassiveRestPanel``).
    REST_ENDPOINTS = (ENDPOINT,)

    #: The latest first day of the endpoints read (``MassiveRestPanel.EARLIEST``).
    EARLIEST = date(2024, 2, 6)

    #: The store folder under ``market/massive`` (``MassiveRestPanel.from_data_root``).
    STORE_NAME = "massive_short_volume_1d"

    #: The store folder's ``README.md``.
    STORE_README = """# massive_short_volume_1d

Massive's daily short volume (its REST `/stocks/v1/short-volume`, FINRA's daily short sale
volume) as a daily panel on Sharadar's permaticker axis and SEP's trading calendar, the whole
market's domestic common stock (SEP's default category filter), from 2024-02-06. Variables, each
on the record's own trading day, NaN on a day without one: `short_volume` (shares sold short
across all venues), `short_total_volume` (total volume across all venues), `short_exempt_volume`
(short volume exempt from Regulation SHO) and `short_volume_ratio` (percent of the total volume
sold short). The per-venue columns stay in the raw tier; a record re-read by a later pull
replaces the earlier copy, but a day already stored changes only on a rebuild.

Each record is placed on the permaticker the Ticker map gives its ticker on its day; a record it
cannot place is left out and counted in `massive_short_volume_1d.zarr.unplaced.json`. Built by
`quantlab.dataset.massive.short_volume.MassiveShortVolumeDataset` from
`downloads/massive/rest/short_volume`, `market/massive/massive_ticker_map` and Sharadar's SEP
and TICKERS; first built by `scripts/massive/short_volume.py`, then appended day by day by the
daily data update. docs/user-guide/massive.md, "Short volume". Licensed data: personal use.
"""

    #: The module's ``VARIABLES``.
    VARIABLES = VARIABLES

    # -- the rows ---------------------------------------------------------------

    def _released(self) -> pl.DataFrame:
        """Return every security's totals per day with a record, placed and summed.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` and the four variables, one row per
            pair, sorted by both.
        """
        calendar = self._trading_calendar()
        window = calendar.days_in(self.config.start_date, self.config.end_date)
        placement = TickerPlacement(self.ticker_map, owner=self.class_name)
        schema = {"symbol": pl.Int64, "available": pl.Datetime("ns"), **{n: pl.Float64 for n in VARIABLES}}
        frame = pl.DataFrame(schema=schema)
        with Timer(f"{self.class_name}: place short volume"):
            if window.len():
                rows = self._on_trading_days(_records(self.config.massive_dir), calendar, "date", window.min()).filter(
                    pl.col("available") <= window.max()
                )
                rows = placement.place(rows, source=ENDPOINT)
                frame = (
                    rows.group_by(pl.col("permaticker").alias("symbol"), "available")
                    .agg(
                        pl.col("short_volume").sum(),
                        pl.col("total_volume").sum().alias("short_total_volume"),
                        pl.col("exempt_volume").sum().alias("short_exempt_volume"),
                    )
                    .with_columns(
                        pl.when(pl.col("short_total_volume") > 0)
                        .then(100 * pl.col("short_volume") / pl.col("short_total_volume"))
                        .alias("short_volume_ratio")
                    )
                )
            placement.write_report(self.config.zarr_file_path)
        keep = [int(s) for s in universe(self.config)]
        return (
            pl.concat([pl.DataFrame(schema=schema), frame], how="diagonal_relaxed")
            .filter(pl.col("symbol").is_in(keep))
            .select(list(schema))
            .sort("symbol", "available")
        )

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no Massive short volume record is placed"
