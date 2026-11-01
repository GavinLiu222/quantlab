"""Massive's news as a daily count and sentiment panel on the permaticker axis.

Massive's ``/v2/reference/news`` articles, as ``MassiveClient.pull`` keeps
them in the REST raw tier (``quantlab.dataset.massive.raw.rest_page_files``),
become four variables on SEP's trading calendar, one column per permaticker:

- ``news_count``: the articles naming the security that count on the day;
- ``news_positive``, ``news_negative``, ``news_neutral``: those of them
  whose per-ticker ``insights`` sentiment for the security is that label.
  An article without an insight for the ticker is counted in
  ``news_count`` only.

A cell without an article is 0. An article counts on the trading day of
its ``published_utc`` in New York time when published before that
session's close, and on the next trading day when published at or after the
close or on a day without trading, so news after the close never reaches the
day's bar. The close is the session's actual one on the XNYS calendar
(``XnysSessionCalendar``): 16:00, and 13:00 on an early close.

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
trading day was published by its close in New York that day, before the UTC
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
from datetime import date, datetime, time

import pandas as pd
import polars as pl

from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.config import MassiveNewsConfig
from quantlab.dataset.massive.placement import TickerPlacement
from quantlab.dataset.massive.raw import rest_page_files
from quantlab.dataset.massive.rest_panel import MassiveRestEventPanel
from quantlab.dataset.sharadar.universe import universe
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

#: The time zone of the sessions.
NEW_YORK = "America/New_York"

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


def _session_closes(days: list[date]) -> pl.DataFrame:
    """Return each NYSE session's close among ``days`` as a UTC time: ``date``, ``_close``.

    The close is the session's actual one (13:00 New York time on an early
    close). A day past the session calendar's last one closes at 16:00 when
    it is a weekday; a day without a session has no row.
    """
    sessions = XnysSessionCalendar()
    known = set(sessions.calendar.sessions.date)
    last = sessions.calendar.last_session.date()
    inside = sorted(day for day in days if day in known)
    closes = sessions.session_bounds(inside).select("date", pl.col("close").alias("_close"))
    beyond = sorted(day for day in days if day > last and day.weekday() < 5)
    if beyond:
        regular = [
            pd.Timestamp(datetime.combine(day, time(16))).tz_localize(NEW_YORK).tz_convert("UTC").tz_localize(None)
            for day in beyond
        ]
        closes = pl.concat([closes, pl.DataFrame({"date": beyond, "_close": regular},
                                                 schema={"date": pl.Date, "_close": pl.Datetime("ns")})])
    return closes.with_columns(pl.col("_close").dt.replace_time_zone("UTC"))


def _trading_day(rows: pl.DataFrame) -> pl.DataFrame:
    """Return ``rows`` with ``_day``, the day each article counts on before rolling onto SEP's calendar.

    An article published before its New York day's session close counts on
    that day; one published at or after the close, or on a day without a
    session, on the next day (which ``TradingCalendar.place`` rolls onto the
    next trading day).

    Examples
    --------
    NYSE closed at 13:00 on 2024-07-03:

    >>> rows = pl.DataFrame({"published": ["2024-07-03T16:59:00Z", "2024-07-03T18:00:00Z"]}).with_columns(
    ...     pl.col("published").str.to_datetime(time_zone="UTC"))
    >>> _trading_day(rows).get_column("_day").to_list()
    [datetime.date(2024, 7, 3), datetime.date(2024, 7, 4)]
    """
    rows = rows.with_columns(pl.col("published").dt.convert_time_zone(NEW_YORK).dt.date().alias("_local_day"))
    closes = _session_closes(rows.get_column("_local_day").drop_nulls().unique().to_list())
    before_close = pl.col("published") < pl.col("_close").cast(rows.schema["published"])
    return (
        rows.join(closes, left_on="_local_day", right_on="date", how="left")
        .with_columns(
            pl.when(before_close.fill_null(False))
            .then(pl.col("_local_day"))
            .otherwise(pl.col("_local_day") + pl.duration(days=1))
            .alias("_day")
        )
        .drop("_local_day", "_close")
    )


class MassiveNewsDataset(MassiveRestEventPanel):
    """Daily panel of Massive's news counts and sentiment labels, by permaticker.

    See the module docstring for the variables and the placement.

    Parameters
    ----------
    config : MassiveNewsConfig
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

    #: The REST endpoints read (``MassiveRestPanel``).
    REST_ENDPOINTS = (ENDPOINT,)

    #: The latest first day of the endpoints read (``MassiveRestPanel.EARLIEST``).
    EARLIEST = date(2016, 6, 1)

    #: The store folder under ``market/massive`` (``MassiveRestPanel.from_data_root``).
    STORE_NAME = "massive_news_1d"

    #: The store folder's ``README.md``.
    STORE_README = """# massive_news_1d

Massive's news articles (its REST `/v2/reference/news`) counted per security and trading day, on
Sharadar's permaticker axis and SEP's trading calendar, the whole market's domestic common stock
(SEP's default category filter). Variables, 0 on a day without an article: `news_count`
(articles naming the security), and `news_positive`, `news_negative`, `news_neutral` (those
whose per-ticker `insights` sentiment for the security is that label). An article counts on the
trading day it was published before that session's close in New York (16:00, 13:00 on an early
close), otherwise on the next trading day (after the close, on a weekend or holiday).

Each of an article's tickers is placed on the permaticker the Ticker map gives it on the
article's trading day, and the article counts once for each security placed; a ticker the map
cannot place is left out and counted in `massive_news_1d.zarr.unplaced.json`. The article text
(title, description, keywords, insights) stays raw under `downloads/massive/rest/news`. Built by
`quantlab.dataset.massive.news.MassiveNewsDataset` from `downloads/massive/rest/news`,
`market/massive/massive_ticker_map` and Sharadar's SEP and TICKERS, from the Ticker map's first
day; first built by `scripts/massive/news.py`, then appended day by day by the daily data
update. docs/user-guide/massive.md, "News". Licensed data: personal use.
"""

    #: The module's ``VARIABLES``.
    VARIABLES = VARIABLES
    #: A day without an article counts 0.
    FILL_VALUE = 0.0

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
                rows = self._on_trading_days(_trading_day(rows), calendar, "_day", window.min())
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

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no Massive news article is placed"
