"""The Ticker map: each day's Massive ticker to the permaticker that traded under it.

The Trade bar store's symbol axis is the Massive ticker, the ticker a
security traded under that day (ADR 0031). A ticker is not a security: one
ticker can belong to different securities on different days. The Ticker map
says, for each day and Massive ticker, which Sharadar permaticker traded
under it, or none, with the evidence for the choice. It is the only way
from the Trade bar store to Sharadar's securities.

**The rules, version 1** (``RULES_VERSION``). For a day D and a ticker t in
Massive's day aggregates of D:

1. The *candidates* are every Sharadar holder of t, its span dates ignored
   (``PermatickerResolver.ticker_holders``), from SEP's and SFP's TICKERS
   rows: stocks and funds.
2. A candidate *matches* when its Sharadar unadjusted close of D
   (``closeunadj``) is within ``max(1% of it, 0.0051)`` of Massive's day
   close; the absolute floor admits a low-priced stock Sharadar rounds to
   two decimals.
3. One candidate matches: it is chosen (``"price"``). Several match: the
   one whose span covers D (``"price+span"``), and with none or several
   such, the ticker is unmapped.
4. None matches: a candidate whose span covers D and whose close is off by
   a split ratio is chosen (``"split_corrected"``). Sharadar's "unadjusted"
   close is sometimes adjusted for a later split; the ratios tried are the
   cumulative products of the candidate's ACTIONS splits dated after D, in
   either direction, within 1%.
5. Otherwise the ticker is unmapped (``"unmapped"``) with a reason
   (``UNMAPPED_REASONS``).

The resolver's two fallbacks for Sharadar's own files ("a single holder
maps whatever the dates", "with none on the day, the one that ended last")
are not used: they gave trades of securities Sharadar lacks to companies
delisted years before.

**Storage.** The map is a directory holding one Parquet file of rows per
mapped day, ``<path>/<YYYY>/<YYYY-MM-DD>.parquet``, with the columns of
``ROW_SCHEMA``. ``append`` maps one more day; ``rebuild`` maps every day of
the raw tier again (after a change of rules or a Sharadar restatement) into
a sidecar and swaps it in. A row records the version of the rules that
built it.

Examples
--------
Needs Massive's day aggregates and Sharadar's TICKERS, ACTIONS, SEP and SFP
in the raw tier::

    ticker_map = TickerMap(TickerMapConfig(
        path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
        massive_dir="/data/quantlab/downloads/massive",
        sharadar_dir="/data/quantlab/downloads/sharadar",
    ))
    ticker_map.append(date(2020, 3, 2))
    rows = ticker_map.read("2020-03-02", "2020-03-02")
    rows.filter(pl.col("ticker") == "VIAC")["evidence"].item()  # 'unmapped'
"""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import polars as pl
from loguru import logger

from quantlab.core.component import Component
from quantlab.dataset.config import TickerMapConfig
from quantlab.dataset.massive.raw import raw_days, raw_file, read_aggregates
from quantlab.dataset.sharadar.permatickers import HolderSpan, PermatickerResolver
from quantlab.dataset.sharadar.tables import scan_raw_table

__all__ = [
    "EVIDENCE",
    "ROW_SCHEMA",
    "RULES_VERSION",
    "UNMAPPED_REASONS",
    "TickerMap",
]

#: The version of the rules; every row records the version that built it.
RULES_VERSION = 1

#: The relative tolerance of a close match.
PRICE_TOLERANCE = 0.01

#: The absolute floor of a close match, for a low price rounded to cents.
PRICE_FLOOR = 0.0051

#: The Sharadar price tables whose holders are candidates: stocks, then funds.
CANDIDATE_TABLES = ("sep", "sfp")

#: How a row was decided.
EVIDENCE = ("price", "price+span", "split_corrected", "unmapped")

#: Why a ticker is unmapped.
UNMAPPED_REASONS = (
    "no Massive close",
    "no Sharadar holder",
    "no candidate priced that day",
    "several candidates match its close",
    "no candidate's close matches",
)

#: The columns of a map row.
ROW_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "ticker": pl.String,
    "permaticker": pl.Int64,
    "evidence": pl.String,
    "reason": pl.String,
    "massive_close": pl.Float64,
    "sharadar_close": pl.Float64,
    "rules_version": pl.Int64,
}

#: Days mapped together by ``rebuild``: Sharadar's closes are read once per batch.
REBUILD_BATCH_DAYS = 250

#: Suffixes of ``rebuild``'s sidecars beside the map's directory.
REBUILDING_SUFFIX = ".rebuilding.tmp"
SUPERSEDED_SUFFIX = ".superseded.tmp"


def _matches(massive_close: float, sharadar_close: float) -> bool:
    """Return whether two closes agree within ``max(1%, 0.0051)``."""
    return abs(massive_close - sharadar_close) <= max(
        PRICE_TOLERANCE * abs(sharadar_close), PRICE_FLOOR
    )


def _split_corrected(massive_close: float, sharadar_close: float, splits: list[tuple[date, float]], day: date) -> bool:
    """Return whether the closes differ by a cumulative split ratio of the splits after ``day``."""
    if sharadar_close == 0:
        return False
    ratio = massive_close / sharadar_close
    factor = 1.0
    for when, value in sorted(splits):
        if when <= day or not value:
            continue
        factor *= value
        for target in (factor, 1.0 / factor):
            if abs(ratio - target) <= PRICE_TOLERANCE * target:
                return True
    return False


def decide(
    ticker: str,
    day: date,
    massive_close: float,
    spans: list[HolderSpan],
    closes: dict[tuple[int, date], float],
    splits: dict[int, list[tuple[date, float]]],
) -> dict:
    """Map one ticker on one day by the rules of version 1 (see the module docstring).

    Parameters
    ----------
    ticker : str
        The Massive ticker.
    day : date
        The trading day.
    massive_close : float
        Massive's day close of the ticker.
    spans : list of HolderSpan
        Every Sharadar holder of the ticker.
    closes : dict
        ``{(permaticker, day): unadjusted close}`` of Sharadar.
    splits : dict
        ``{permaticker: [(date, split value), ...]}`` from ACTIONS.

    Returns
    -------
    dict
        One row of ``ROW_SCHEMA``.

    Examples
    --------
    >>> from quantlab.dataset.sharadar.permatickers import HolderSpan
    >>> decide("AAA", date(2020, 3, 2), 10.05, [HolderSpan(101, None, None)],
    ...        {(101, date(2020, 3, 2)): 10.0}, {})["evidence"]
    'price'
    """
    row = {
        "date": day,
        "ticker": ticker,
        "permaticker": None,
        "evidence": "unmapped",
        "reason": None,
        "massive_close": massive_close,
        "sharadar_close": None,
        "rules_version": RULES_VERSION,
    }

    def chosen(span: HolderSpan, close: float, evidence: str) -> dict:
        return {**row, "permaticker": span.permaticker, "evidence": evidence, "sharadar_close": close}

    if not spans:
        return {**row, "reason": "no Sharadar holder"}
    priced = [(span, closes[(span.permaticker, day)]) for span in spans if (span.permaticker, day) in closes]
    if not priced:
        return {**row, "reason": "no candidate priced that day"}
    matching = [(span, close) for span, close in priced if _matches(massive_close, close)]
    if len(matching) == 1:
        return chosen(*matching[0], "price")
    if matching:
        covering = [(span, close) for span, close in matching if span.covers(day)]
        if len(covering) == 1:
            return chosen(*covering[0], "price+span")
        return {**row, "reason": "several candidates match its close"}
    corrected = [
        (span, close)
        for span, close in priced
        if span.covers(day) and _split_corrected(massive_close, close, splits.get(span.permaticker, []), day)
    ]
    if len(corrected) == 1:
        return chosen(*corrected[0], "split_corrected")
    nearest = min(priced, key=lambda pair: abs(math.log(max(massive_close, 1e-12) / max(pair[1], 1e-12))))
    return {**row, "reason": "no candidate's close matches", "sharadar_close": nearest[1]}


class TickerMap(Component):
    """The Ticker map of one directory: append a day, rebuild all, read rows.

    See the module docstring for the rules and the layout. Sharadar's
    holders and splits are read once per instance; its closes once per
    batch of days mapped.

    Parameters
    ----------
    config : TickerMapConfig
        The map's directory and the two raw tiers it reads.

    Examples
    --------
    Needs the raw tiers::

        ticker_map = TickerMap(config)
        ticker_map.append(date(2020, 3, 2))
        ticker_map.days()  # [datetime.date(2020, 3, 2)]
    """

    config_cls = TickerMapConfig

    def __init__(self, config: TickerMapConfig) -> None:
        """Hold ``config``; nothing is read until a day is mapped."""
        self.config = config
        self._resolvers: dict[str, PermatickerResolver] = {}
        self._holders: dict[str, list[HolderSpan]] | None = None
        self._splits: dict[int, list[tuple[date, float]]] | None = None

    def __eq__(self, other: object) -> bool:
        """Two maps are equal when their configs are."""
        return isinstance(other, TickerMap) and other.config == self.config

    __hash__ = None  # type: ignore[assignment]

    # -- the map's files --------------------------------------------------------

    @property
    def path(self) -> Path:
        """The map's directory.

        Examples
        --------
        >>> ticker_map.path.name
        'massive_ticker_map'
        """
        return Path(self.config.path)

    def _recover(self) -> None:
        """Put back a map a ``rebuild`` killed between its two renames left aside."""
        superseded = Path(f"{self.path}{SUPERSEDED_SUFFIX}")
        if superseded.exists() and not self.path.exists():
            logger.warning(f"TickerMap: restoring {self.path} from {superseded}, left by a killed rebuild.")
            os.replace(superseded, self.path)

    @staticmethod
    def day_file(root: str | Path, day: date) -> Path:
        """Return the file of ``day``'s rows under the map directory ``root``.

        Examples
        --------
        >>> TickerMap.day_file("/m", date(2020, 3, 2))
        PosixPath('/m/2020/2020-03-02.parquet')
        """
        return Path(root) / f"{day.year:04d}" / f"{day.isoformat()}.parquet"

    def days(self) -> list[date]:
        """Return the mapped days, in order.

        Examples
        --------
        >>> ticker_map.days()
        [datetime.date(2020, 3, 2), datetime.date(2020, 3, 3)]
        """
        self._recover()
        if not self.path.exists():
            return []
        return sorted(date.fromisoformat(p.stem) for p in self.path.glob("*/*.parquet"))

    def read(self, start: str | date | None = None, end: str | date | None = None) -> pl.DataFrame:
        """Return the rows of the mapped days from ``start`` to ``end``, both inclusive.

        Parameters
        ----------
        start, end : str or date, optional
            The first and last day; ``None`` leaves that side open.

        Returns
        -------
        pl.DataFrame
            Rows of ``ROW_SCHEMA``, by date and ticker.

        Examples
        --------
        >>> ticker_map.read("2020-03-02", "2020-03-02").columns
        ['date', 'ticker', 'permaticker', 'evidence', 'reason', 'massive_close', 'sharadar_close', 'rules_version']
        """
        first = None if start is None else _as_date(start)
        last = None if end is None else _as_date(end)
        files = [
            self.day_file(self.path, day)
            for day in self.days()
            if (first is None or day >= first) and (last is None or day <= last)
        ]
        if not files:
            return pl.DataFrame(schema=ROW_SCHEMA)
        return pl.read_parquet(files).select(list(ROW_SCHEMA)).sort("date", "ticker")

    # -- writing ----------------------------------------------------------------

    def append(self, day: str | date) -> pl.DataFrame:
        """Map ``day`` and add its rows to the map.

        Parameters
        ----------
        day : str or date
            A day whose Massive day aggregates are in the raw tier.

        Returns
        -------
        pl.DataFrame
            The day's rows.

        Raises
        ------
        ValueError
            If the map already holds ``day``; ``rebuild`` maps it again.
        FileNotFoundError
            If the raw tier has no day aggregates of ``day``.

        Examples
        --------
        >>> ticker_map.append("2020-03-02").height
        8954
        """
        day = _as_date(day)
        if self.day_file(self.path, day).exists():
            raise ValueError(
                f"TickerMap: {self.path} already maps {day}; rebuild() maps every day again."
            )
        rows = self.map_days([day])
        self._write_day(self.path, day, rows)
        self._log(day, rows)
        return rows

    def rebuild(self, days: Iterable[str | date] | None = None) -> list[date]:
        """Map every day again and replace the map with the result.

        The days are mapped into a sidecar directory beside the map, which
        then replaces it by two renames: a run that fails before them leaves
        the map as it was, and one killed between them leaves it at
        ``<path>.superseded.tmp``, put back by the next call. Do not append
        while a rebuild runs: the swap would drop the appended day.

        Parameters
        ----------
        days : iterable of str or date, optional
            The days to map; ``None`` maps every day whose day aggregates
            are in the raw tier.

        Returns
        -------
        list of date
            The days mapped.

        Raises
        ------
        ValueError
            If there is no day to map: an empty map would replace this one.

        Examples
        --------
        >>> ticker_map.rebuild()
        [datetime.date(2020, 3, 2), datetime.date(2020, 3, 3)]
        """
        wanted = (
            raw_days(self.config.massive_dir, "day_aggs")
            if days is None
            else sorted({_as_date(day) for day in days})
        )
        if not wanted:
            raise ValueError(f"TickerMap: no day to map into {self.path}; refusing to replace it with an empty map.")
        self._recover()
        rebuilding = Path(f"{self.path}{REBUILDING_SUFFIX}")
        superseded = Path(f"{self.path}{SUPERSEDED_SUFFIX}")
        shutil.rmtree(rebuilding, ignore_errors=True)
        try:
            for low in range(0, len(wanted), REBUILD_BATCH_DAYS):
                batch = wanted[low : low + REBUILD_BATCH_DAYS]
                rows = self.map_days(batch)
                for day in batch:
                    self._write_day(rebuilding, day, rows.filter(pl.col("date") == day))
                logger.info(f"TickerMap: mapped {batch[0]}..{batch[-1]} ({len(batch)} day(s)).")
            rebuilding.mkdir(parents=True, exist_ok=True)
        except BaseException:
            shutil.rmtree(rebuilding, ignore_errors=True)
            raise
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.rmtree(superseded, ignore_errors=True)
            os.replace(self.path, superseded)
        os.replace(rebuilding, self.path)
        shutil.rmtree(superseded, ignore_errors=True)
        return wanted

    @staticmethod
    def _write_day(root: Path, day: date, rows: pl.DataFrame) -> None:
        """Write one day's rows under ``root``, atomically."""
        target = TickerMap.day_file(root, day)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".parquet.tmp")
        rows.select(list(ROW_SCHEMA)).sort("ticker").write_parquet(partial)
        os.replace(partial, target)

    @staticmethod
    def _log(day: date, rows: pl.DataFrame) -> None:
        mapped = rows.filter(pl.col("permaticker").is_not_null()).height
        logger.info(f"TickerMap: {day}: {mapped} of {rows.height} ticker(s) mapped.")

    # -- mapping ----------------------------------------------------------------

    def map_days(self, days: list[date]) -> pl.DataFrame:
        """Return the rows of ``days`` without writing them.

        Parameters
        ----------
        days : list of date
            Days whose day aggregates are in the raw tier.

        Returns
        -------
        pl.DataFrame
            Rows of ``ROW_SCHEMA``, one per day and ticker of the day
            aggregates.

        Raises
        ------
        FileNotFoundError
            If a day's aggregates are missing.

        Examples
        --------
        >>> ticker_map.map_days([date(2020, 3, 2)]).filter(pl.col("ticker") == "AAPL")["evidence"].item()
        'price'
        """
        holders = self._ticker_holders()
        splits = self._split_values()
        closes = self._closes(days)
        rows = []
        for day in days:
            path = raw_file(self.config.massive_dir, "day_aggs", day)
            if not path.exists():
                raise FileNotFoundError(f"TickerMap: no Massive day aggregates of {day} at {path}.")
            aggregates = read_aggregates(path).select("ticker", "close").drop_nulls("ticker")
            for ticker, close in aggregates.iter_rows():
                if close is None or close != close:
                    rows.append(decide(ticker, day, float("nan"), [], closes, splits)
                                | {"reason": "no Massive close"})
                    continue
                rows.append(decide(ticker, day, float(close), holders.get(ticker, []), closes, splits))
        return pl.DataFrame(rows, schema=ROW_SCHEMA, orient="row") if rows else pl.DataFrame(schema=ROW_SCHEMA)

    def _resolver(self, code: str) -> PermatickerResolver:
        if code not in self._resolvers:
            self._resolvers[code] = PermatickerResolver(self.config.sharadar_dir, code)
        return self._resolvers[code]

    def _ticker_holders(self) -> dict[str, list[HolderSpan]]:
        """Return every ticker's holders among SEP's and SFP's securities, read once."""
        if self._holders is None:
            holders: dict[str, list[HolderSpan]] = {}
            for code in CANDIDATE_TABLES:
                for ticker, spans in self._resolver(code).ticker_holders().items():
                    known = {span.permaticker for span in holders.get(ticker, [])}
                    holders.setdefault(ticker, []).extend(s for s in spans if s.permaticker not in known)
            self._holders = holders
        return self._holders

    def _split_values(self) -> dict[int, list[tuple[date, float]]]:
        """Return each permaticker's ACTIONS splits, ``(date, value)``, read once."""
        if self._splits is None:
            splits: dict[int, set[tuple[date, float]]] = {}
            for code in CANDIDATE_TABLES:
                try:
                    frame = (
                        scan_raw_table(self.config.sharadar_dir, "actions", annotate=self._resolver(code).annotate)
                        .filter((pl.col("action") == "split") & pl.col("permaticker").is_not_null())
                        .select("permaticker", "date", "value")
                        .collect()
                    )
                except FileNotFoundError:
                    continue
                for permaticker, when, value in frame.iter_rows():
                    if value is not None:
                        splits.setdefault(int(permaticker), set()).add((when, float(value)))
            self._splits = {permaticker: sorted(rows) for permaticker, rows in splits.items()}
        return self._splits

    def _closes(self, days: list[date]) -> dict[tuple[int, date], float]:
        """Return Sharadar's unadjusted close of every security priced on one of ``days``."""
        closes: dict[tuple[int, date], float] = {}
        for code in CANDIDATE_TABLES:
            frame = (
                scan_raw_table(self.config.sharadar_dir, code, annotate=self._resolver(code).annotate)
                .filter(
                    pl.col("date").is_in(days)
                    & pl.col("permaticker").is_not_null()
                    & pl.col("closeunadj").is_not_null()
                )
                .select("permaticker", "date", "closeunadj")
                .unique(["permaticker", "date"], keep="first", maintain_order=True)
                .collect()
            )
            for permaticker, when, close in frame.iter_rows():
                closes.setdefault((int(permaticker), when), float(close))
        return closes


def _as_date(value) -> date:
    """Return a day given as text, a date, a datetime or a timestamp."""
    if isinstance(value, datetime):
        return value.date()
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
