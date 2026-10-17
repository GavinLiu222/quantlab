"""Rows keyed by a Massive ticker placed on Sharadar's permatickers through the Ticker map.

Massive's REST endpoints (dividends, splits, short interest, short volume,
news, 8-K disclosures, IPOs) name a security by its ticker, and a ticker is
not a security: one ticker belongs to different securities on different
days. A panel on the permaticker axis places each row on the permaticker
the Ticker map (``quantlab.dataset.massive.ticker_map``) gives the row's
ticker on the row's own *placement day* (an ex-date, an execution date, a
settlement date plus its lag, a filing or publication day), so a reused
ticker sends each row to the security that held it that day.

A row the map cannot place is never guessed onto a security. It is left out
and counted, with the reason, in the store's sidecar
``<store>.unplaced.json`` (``UNPLACED_SUFFIX``):

- the map row of the ticker and day is unmapped: ``"unmapped: <the map's
  reason>"`` (``quantlab.dataset.massive.ticker_map.UNMAPPED_REASONS``);
- the ticker is not in Massive's day aggregates of that day, so the map has
  no row of it (``NOT_TRADED``);
- the map does not hold the day at all (``NOT_MAPPED``).

A panel may also leave rows out for reasons of its own (no amount, another
currency) through ``TickerPlacement.leave_out``: they are counted in the same
report. A row naming several tickers (a news article, an 8-K) is exploded
into one row per ticker before it is placed, and counts once for each placed
ticker.

Examples
--------
Place one endpoint's rows, already on trading days, and write the report::

    placement = TickerPlacement(ticker_map, owner="MassiveDividendDataset")
    placed = placement.place(rows, source="dividends:ex", day="available")
    placed.columns[-1]  # 'permaticker'
    placement.write_report("/data/quantlab/market/massive/massive_dividends_1d/massive_dividends_1d.zarr")
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from quantlab.dataset.massive.ticker_map import TickerMap
from quantlab.utils.atomic import write_json_atomically

__all__ = ["NOT_MAPPED", "NOT_TRADED", "UNPLACED_SUFFIX", "TickerPlacement"]

#: The suffix of a store's report of the rows left out (``<store>.unplaced.json``).
UNPLACED_SUFFIX = ".unplaced.json"

#: Why a row is left out when the Ticker map does not hold its day.
NOT_MAPPED = "the Ticker map does not hold the day"

#: Why a row is left out when its ticker is not in the day's aggregates.
NOT_TRADED = "the ticker is not in Massive's day aggregates that day"

#: How many tickers a log line names.
_SAMPLE = 10

_LEFT_OUT_SCHEMA = {"source": pl.String, "ticker": pl.String, "date": pl.Date, "reason": pl.String}


class TickerPlacement:
    """Place rows on permatickers by the Ticker map of their own day, counting what it cannot place.

    One instance serves one build of one store: each ``place`` (and
    ``leave_out``) call adds to the rows left out, and ``write_report``
    writes them all into the store's sidecar.

    Parameters
    ----------
    ticker_map : TickerMap
        The map read.
    owner : str
        Named in the report and in log lines.

    Examples
    --------
    With ``rows`` holding ``ticker`` and ``available`` columns::

        placement = TickerPlacement(ticker_map, owner="demo")
        placement.place(rows, source="splits", day="available").get_column("permaticker")
    """

    def __init__(self, ticker_map: TickerMap, *, owner: str) -> None:
        self.ticker_map = ticker_map
        self.owner = owner
        self._left_out: list[pl.DataFrame] = []
        self._days: set | None = None

    def place(self, rows: pl.DataFrame, *, source: str, day: str = "available", ticker: str = "ticker") -> pl.DataFrame:
        """Return the rows the Ticker map places, each with its ``permaticker``.

        Parameters
        ----------
        rows : pl.DataFrame
            Rows with a Massive ticker column ``ticker`` and a placement day
            column ``day`` (a date or a datetime at midnight). The day must
            be a trading day: roll a vendor date onto the calendar first
            (``TradingCalendar.place``).
        source : str
            Names the rows in the report (``"dividends:ex"``).
        day : str, default "available"
            The column of the placement day.
        ticker : str, default "ticker"
            The column of the Massive ticker.

        Returns
        -------
        pl.DataFrame
            The placed rows, with an int64 ``permaticker`` column added; the
            rest are counted in the report.

        Examples
        --------
        A ticker of the map's day is placed; one the day's aggregates lack
        is counted::

            placed = placement.place(rows, source="dividends:ex")
            placement.left_out()["reason"].to_list()
            # ["the ticker is not in Massive's day aggregates that day"]
        """
        if "permaticker" in rows.columns:
            raise ValueError(f"{self.owner}: rows to place already have a 'permaticker' column.")
        keyed = rows.with_columns(
            pl.col(ticker).cast(pl.String).alias("_ticker"), pl.col(day).cast(pl.Date).alias("_date")
        )
        wanted = keyed.select("_ticker", "_date").unique().drop_nulls()
        if self._days is None:
            self._days = set(self.ticker_map.days())
        lookup = self.ticker_map.read_days(
            wanted.get_column("_date").unique().to_list(),
            tickers=wanted.get_column("_ticker").unique().to_list(),
        ).select(
            pl.col("ticker").alias("_ticker"),
            pl.col("date").alias("_date"),
            "permaticker",
            pl.col("evidence").alias("_evidence"),
            pl.col("reason").alias("_reason"),
        )
        joined = keyed.join(lookup, on=["_ticker", "_date"], how="left")
        reason = (
            pl.when(~pl.col("_date").is_in(sorted(self._days)))
            .then(pl.lit(NOT_MAPPED))
            .when(pl.col("_evidence").is_null())
            .then(pl.lit(NOT_TRADED))
            .otherwise(pl.lit("unmapped: ") + pl.col("_reason").fill_null("no reason recorded"))
        )
        missing = joined.filter(pl.col("permaticker").is_null())
        if missing.height:
            self._left_out.append(
                missing.select(
                    pl.lit(source).alias("source"),
                    pl.col("_ticker").alias("ticker"),
                    pl.col("_date").alias("date"),
                    reason.alias("reason"),
                )
            )
        return (
            joined.filter(pl.col("permaticker").is_not_null())
            .with_columns(pl.col("permaticker").cast(pl.Int64))
            .drop("_ticker", "_date", "_evidence", "_reason")
        )

    def leave_out(self, rows: pl.DataFrame, *, source: str, reason: str, day: str, ticker: str = "ticker") -> None:
        """Count rows a panel leaves out for a reason of its own (no amount, another currency).

        Examples
        --------
        Count the dividends paid in another currency::

            placement.leave_out(rows, source="dividends", reason="currency CAD, not USD", day="ex_dividend_date")
        """
        if rows.height:
            self._left_out.append(
                rows.select(
                    pl.lit(source).alias("source"),
                    pl.col(ticker).cast(pl.String).alias("ticker"),
                    pl.col(day).cast(pl.Date).alias("date"),
                    pl.lit(reason).alias("reason"),
                )
            )

    def left_out(self) -> pl.DataFrame:
        """Return every row left out so far: ``source``, ``ticker``, ``date``, ``reason``.

        Examples
        --------
        >>> placement.left_out().columns
        ['source', 'ticker', 'date', 'reason']
        """
        if not self._left_out:
            return pl.DataFrame(schema=_LEFT_OUT_SCHEMA)
        return pl.concat(self._left_out).select(list(_LEFT_OUT_SCHEMA))

    def write_report(self, store_path: str | Path) -> Path | None:
        """Write the rows left out into ``<store_path>.unplaced.json``, or remove a stale report.

        The report is ``{"owner", "rows", "unplaced": [{"source", "ticker",
        "first_date", "last_date", "rows", "reason"}, ...]}``, one entry per
        source, ticker and reason, written atomically. With nothing left
        out an existing report is deleted, so its presence means rows are
        missing from the store.

        Returns
        -------
        Path or None
            The report written, or ``None`` when nothing was left out.

        Examples
        --------
        >>> placement.write_report("/data/market/massive/massive_dividends_1d/massive_dividends_1d.zarr").name
        'massive_dividends_1d.zarr.unplaced.json'
        """
        path = Path(f"{store_path}{UNPLACED_SUFFIX}")
        rows = self.left_out()
        if rows.height == 0:
            path.unlink(missing_ok=True)
            return None
        summary = (
            rows.group_by("source", "ticker", "reason")
            .agg(
                pl.col("date").min().alias("first_date"),
                pl.col("date").max().alias("last_date"),
                pl.len().alias("rows"),
            )
            .sort("source", "ticker", "reason")
        )
        entries = [
            {
                "source": source,
                "ticker": ticker,
                "first_date": None if first is None else first.isoformat(),
                "last_date": None if last is None else last.isoformat(),
                "rows": int(count),
                "reason": reason,
            }
            for source, ticker, reason, first, last, count in summary.select(
                "source", "ticker", "reason", "first_date", "last_date", "rows"
            ).iter_rows()
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomically(
            path, {"owner": self.owner, "rows": rows.height, "unplaced": entries}, indent=2, sort_keys=True
        )
        tickers = sorted(rows.get_column("ticker").unique().drop_nulls().to_list())
        logger.warning(
            f"{self.owner}: {rows.height} row(s) of {len(tickers)} ticker(s) are not placed on a "
            f"permaticker and are left out; listed in {path}, first {tickers[:_SAMPLE]}."
        )
        return path
