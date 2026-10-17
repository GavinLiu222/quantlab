"""Massive's news as a daily count and sentiment panel on the permaticker axis (#357).

Massive's ``/v2/reference/news`` articles, as ``MassiveClient.pull`` keeps
them in the REST raw tier (``quantlab.dataset.massive.raw.rest_page_files``),
become four variables on SEP's trading calendar, one column per permaticker:

- ``news_count``: the articles naming the security that count on the day;
- ``news_positive``, ``news_negative``, ``news_neutral``: those of them
  whose per-ticker ``insights`` sentiment for the security is that label.
  An article without an insight for the ticker is counted in
  ``news_count`` only.

A cell without an article is 0. An article counts on the trading day of
its ``published_utc`` in New York time when published before 16:00 there,
and on the next trading day when published at or after 16:00 or on a day
without trading, so news after the close never reaches the day's bar. The
cut is 16:00 on every day, an early close included.

An article's tickers are its ``tickers`` and the tickers of its
``insights``; each is placed on the permaticker the Ticker map gives it on
the article's trading day, through
``quantlab.dataset.massive.placement.TickerPlacement``, and the article
counts once for each permaticker placed (two tickers placed on one
security count once). A ticker the map cannot place is left out and
counted in ``<store>.unplaced.json``. Only the universe's permatickers (by
default SEP's domestic common stock) are columns.

The store is built through the last day SEP, the news pull and the Ticker
map all reach, so ``update()`` appends a day only once each of them holds
it. The pull's watermark is a day in UTC; every article counting on a
trading day was published by 16:00 New York time that day, before the UTC
day ends, so a watermark through the day holds them all.

Examples
--------
Build the store and read a security's daily counts::

    config = MassiveNewsConfig(
        zarr_file_path="/data/quantlab/market/massive/massive_news_1d/massive_news_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
        massive_dir="/data/quantlab/downloads/massive",
        ticker_map_path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
        start_date="2016-10-10",
    )
    panel = MassiveNewsDataset(config).update().panel("2024-01-02", "2024-12-31")
    panel["news_count"].sel(symbol=199059)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.config import DatasetConfig, MassiveNewsConfig, TickerMapConfig
from quantlab.dataset.massive.placement import UNPLACED_SUFFIX, TickerPlacement
from quantlab.dataset.massive.raw import read_rest_watermark, rest_page_files
from quantlab.dataset.massive.ticker_map import TickerMap
from quantlab.dataset.sharadar.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
    on_grid,
    to_panel,
)
from quantlab.dataset.sharadar.tables import raw_through, trading_days
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.timer import Timer

__all__ = ["VARIABLES", "MassiveNewsDataset"]

#: The panel's variables and their attributes, in panel order.
VARIABLES: dict[str, dict] = {
    "news_count": {
        "unit": "articles",
        "description": "News articles naming the security that count on the day.",
    },
    "news_positive": {
        "unit": "articles",
        "description": "Of them, those whose insights sentiment for the security is positive.",
    },
    "news_negative": {
        "unit": "articles",
        "description": "Of them, those whose insights sentiment for the security is negative.",
    },
    "news_neutral": {
        "unit": "articles",
        "description": "Of them, those whose insights sentiment for the security is neutral.",
    },
}

#: The REST endpoint read (``quantlab.acquisition.massive.rest.ENDPOINTS``).
ENDPOINT = "news"

#: The time zone, and the hour there from which an article counts on the next trading day.
NEW_YORK = "America/New_York"
CLOSE_HOUR = 16

#: The sentiment labels counted, each its variable.
SENTIMENTS = {"positive": "news_positive", "negative": "news_negative", "neutral": "news_neutral"}


def _article_tickers(vendor_root) -> pl.DataFrame:
    """Return one row per article and ticker: ``id``, ``published``, ``ticker``, ``sentiment``.

    Reads each page line by line keeping only those fields, the newest copy
    of an article (by ``id``) winning, as ``raw.read_rest_records`` does.
    """
    articles: dict[str, tuple] = {}
    for path in rest_page_files(vendor_root, ENDPOINT):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                sentiment = {}
                for insight in record.get("insights") or ():
                    if insight.get("ticker"):
                        sentiment.setdefault(insight["ticker"], insight.get("sentiment"))
                tickers = dict.fromkeys([*(record.get("tickers") or ()), *sentiment])
                identity = record["id"]
                articles.pop(identity, None)
                articles[identity] = (record.get("published_utc"), tuple((t, sentiment.get(t)) for t in tickers))
    rows = [
        (identity, published, ticker, label)
        for identity, (published, pairs) in articles.items()
        for ticker, label in pairs
    ]
    schema = {"id": pl.String, "published_utc": pl.String, "ticker": pl.String, "sentiment": pl.String}
    frame = pl.DataFrame(rows, schema=schema, orient="row")
    return frame.with_columns(
        pl.col("published_utc").str.to_datetime(time_zone="UTC", strict=False).alias("published")
    ).drop("published_utc")


class MassiveNewsDataset(ReleasePanelDataset):
    """Daily panel of Massive's news counts and sentiment labels, by permaticker.

    See the module docstring for the variables and the placement.

    Parameters
    ----------
    dataset_config : MassiveNewsConfig
        Sharadar's raw tier (calendar, universe), Massive's raw tier and the
        Ticker map.

    Examples
    --------
    With ``config`` as in the module example::

        ds = MassiveNewsDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['news_count', 'news_positive', 'news_negative', 'news_neutral']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MassiveNewsConfig

    def _normalize_config(self, config: DatasetConfig) -> MassiveNewsConfig:
        """Normalise as the base class does, then check the table and the universe fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``MassiveNewsConfig``.
        ValueError
            If ``table`` is not ``"sep"`` or a universe field is invalid.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, MassiveNewsConfig):
            raise TypeError(f"{self.class_name} needs a MassiveNewsConfig, got {type(config).__name__}.")
        if config.table != "sep":
            raise ValueError(f"{self.class_name}: table must be 'sep', got {config.table!r}.")
        return normalize_universe(config, self.class_name)

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._ticker_map: TickerMap | None = None

    @property
    def ticker_map(self) -> TickerMap:
        """The Ticker map the articles' tickers are placed through.

        Examples
        --------
        >>> MassiveNewsDataset(config).ticker_map.path.name
        'massive_ticker_map'
        """
        if self._ticker_map is None:
            self._ticker_map = TickerMap(TickerMapConfig(
                path=self.config.ticker_map_path,
                massive_dir=self.config.massive_dir,
                sharadar_dir=self.config.raw_data_dir_path,
            ))
        return self._ticker_map

    @property
    def unplaced_path(self) -> str:
        """The report of the tickers left out (``<store>.unplaced.json``).

        Examples
        --------
        >>> MassiveNewsDataset(config).unplaced_path.endswith("massive_news_1d.zarr.unplaced.json")
        True
        """
        return f"{self.config.zarr_file_path}{UNPLACED_SUFFIX}"

    # -- the calendar: the last day every input reaches -----------------------

    def _trading_calendar(self) -> TradingCalendar:
        """Return SEP's trading days through the last day SEP, the news pull and the Ticker map reach.

        Raises
        ------
        ValueError
            If the news endpoint was never pulled or the Ticker map is empty.
        """
        if self._calendar_cache is None:
            root = self.config.raw_data_dir_path
            pulled = read_rest_watermark(self.config.massive_dir, ENDPOINT)
            if pulled is None:
                raise ValueError(
                    f"{self.class_name}: Massive's {ENDPOINT!r} endpoint was never pulled into "
                    f"{self.config.massive_dir}; run scripts/massive/rest_pull.py first."
                )
            days = self.ticker_map.days()
            if not days:
                raise ValueError(
                    f"{self.class_name}: the Ticker map at {self.config.ticker_map_path} maps no day; "
                    f"run scripts/massive/ticker_map.py first."
                )
            through = min(raw_through(root, ("sep",)), pulled, days[-1])
            self._calendar_cache = TradingCalendar(
                trading_days(root, through), through=through, start=self.config.start_date, end=self.config.end_date
            )
        return self._calendar_cache

    # -- the rows ---------------------------------------------------------------

    def _released(self) -> pl.DataFrame:
        """Return every security's counts per day with an article, placed.

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
        with Timer(f"{self.class_name}: place news"):
            if window.len():
                rows = _article_tickers(self.config.massive_dir).drop_nulls(["published", "ticker"])
                local = pl.col("published").dt.convert_time_zone(NEW_YORK)
                after_close = local.dt.hour() >= CLOSE_HOUR
                rows = rows.with_columns(
                    pl.when(after_close).then(local.dt.date() + pl.duration(days=1)).otherwise(local.dt.date())
                    .alias("_day")
                )
                rows = calendar.place(rows, on="_day").filter(pl.col("available") >= window.min())
                placed = placement.place(rows, source="news")
                frame = (
                    placed.sort("id", "ticker")
                    .unique(["id", "permaticker"], keep="first", maintain_order=True)
                    .group_by(pl.col("permaticker").alias("symbol"), "available")
                    .agg(
                        pl.len().cast(pl.Float64).alias("news_count"),
                        *((pl.col("sentiment") == label).sum().cast(pl.Float64).alias(name)
                          for label, name in SENTIMENTS.items()),
                    )
                    .select(list(schema))
                    .cast(schema)
                )
            placement.write_report(self.config.zarr_file_path)
        keep = [int(s) for s in universe(self.config)]
        return frame.filter(pl.col("symbol").is_in(keep)).sort("symbol", "available")

    # -- the panel: a count on its own day ---------------------------------------

    def _spec(self) -> ReleaseSpec:
        """Return the four variables and their attributes; no staleness."""
        return ReleaseSpec(variables=VARIABLES, stamp=None)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no Massive news article is placed"

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list[int] | None = None) -> xr.Dataset:
        """Return one window: each day's counts on that day, 0 elsewhere."""
        days, symbols = self._window_axes(start_date, end_date, symbols)
        cells = (
            on_grid(days, symbols)
            .join(self._released_rows(), left_on=["timestamp", "symbol"], right_on=["available", "symbol"],
                  how="left")
            .sort("_day", "_column")
            .with_columns(pl.col(name).cast(pl.Float64).fill_null(0.0) for name in VARIABLES)
        )
        return to_panel(cells, days, symbols, VARIABLES)

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count each added permaticker's days with an article between ``start`` and ``end``."""
        if not added:
            return {}
        counts = (
            self._released_rows()
            .filter(
                pl.col("symbol").is_in([int(s) for s in added])
                & pl.col("available").is_between(
                    pl.lit(pd.Timestamp(start).to_datetime64()), pl.lit(pd.Timestamp(end).to_datetime64())
                )
            )
            .group_by("symbol")
            .len()
        )
        return {str(symbol): int(rows) for symbol, rows in counts.iter_rows()}

    def _widen_fill_values(self) -> dict:
        """Fill a new permaticker's stored days with 0: no article counted then."""
        return {name: np.float64(0.0) for name in VARIABLES}
