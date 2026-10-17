"""Map the tickers of Sharadar's raw files to permatickers, as of each file's pull.

Sharadar keys every table but TICKERS by ticker, and renames a security's
whole history when its ticker changes (LESL became LESLQ: every LESL row
reads LESLQ from then on). Each raw file therefore names its securities by
the tickers *of its own pull*: a bulk file pulled before a change holds the
old ticker, a window pulled after it the new one. TICKERS, pulled whole every
run, holds only each permaticker's current ticker, so the old ticker of a
file pulled before a change is not in it.

``PermatickerResolver`` maps one file at a time (``annotate``, the hook of
``scan_raw_table``), each ticker in this order:

1. the TICKERS snapshot of the file's own run (``tables.snapshot_for``),
   which knows exactly the tickers of that pull. A ticker the snapshot gives
   to two permatickers is refused, as TICKERS always was.
2. otherwise the ticker's *holders*, the securities known to have used it:
   the permaticker current TICKERS gives it to (a ticker current TICKERS
   gives to two permatickers is refused); the securities the ACTIONS
   ``tickerchangefrom`` chain says used it, each from its change into the
   ticker (or its first price) until its change away
   (``ticker_changes``); the security TICKERS renamed to the ticker plus a
   number when another took the ticker over (NSTR became NSTR1), often with
   no ACTIONS row: its ``relatedtickers`` lists the old ticker, held from its
   first to its last price (``related_spans``); and, for a store with one,
   the securities its ticker sidecar (``<store>.sharadar_tickers.json``)
   names by it, the sidecar having been written from an earlier TICKERS. A ticker with a single
   holder maps to it, whatever the dates (so a vendor slow to rename a
   security still maps). Several holders (the ticker was reused) are told
   apart by the file's pull date: the holder whose span covers it, the
   latest-starting one if several do, or the latest to have used it if none
   does. A change dated on the pull's own day counts from the next day, as
   the vendor applies it in its evening update.

   The TICKERS history (``quantlab.acquisition.sharadar.raw_tier.fold_tickers_history``) adds holders too:
   each security a folded snapshot showed under the ticker, until the day
   before the history first shows it renamed or the ticker reused (a rename
   TICKERS made before ACTIONS has a row of it). A holder the history shows
   under another ticker on or after the file's pull, and under this one
   only later, is not one of the file's, whatever date ACTIONS gives its
   change: the vendor renames rows when TICKERS shows the new ticker.

   A holder is not taken when another security, under any table, held the
   ticker at the file's pull: the old ticker of a stock that a fund (SFP)
   has taken over names the fund in a file of that pull (ACTIONS and SF3A
   hold funds; ACTIONS names the FB fund's dividends FB, not Meta's, #235).
   Who held it is the file's own TICKERS snapshot, or without one current
   TICKERS, each security counted from the day after it took the ticker (its
   last ACTIONS ticker change, else its first price). A security first priced
   after the file's latest row of the ticker does not count: it owns none of
   them (a fund listed under a dead stock's old ticker after the file's rows).

A row whose ticker none of them maps is left out rather than refused:
``left_out`` drops it, logs a warning and writes the report
``<store>.unmapped.json`` (``UNMAPPED_SUFFIX``) naming each ticker's first
and last date, row count and raw files, so a daily update completes. So does
a ticker whose holders tie (two securities starting on the same day).

One security can then have rows under two tickers for one key: SF3A, pulled
whole, keeps some quarters of a security under its current ticker and under
a former one. ``settle`` keeps one row per security and key (the security's
own ticker as of the file's pull, else the ticker it used on the row's date,
else one of rows identical but for their ticker) and leaves out, as
unmapped, the rows it cannot choose between.

Examples
--------
>>> resolver = PermatickerResolver("/data/downloads/sharadar", "sep")
>>> prices = scan_raw_table("/data/downloads/sharadar", "sep", annotate=resolver.annotate).collect()
>>> prices = resolver.left_out(prices, owner="demo", report_path="/data/market/sharadar/sep/sep.zarr.unmapped.json")
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
from loguru import logger

from quantlab.dataset.sharadar.tables import (
    permaticker_mapping,
    pull_time,
    scan_raw_table,
    snapshot_for,
    table,
    tickers_history,
)
from quantlab.utils.atomic import write_json_atomically

__all__ = [
    "TICKER_CHANGE",
    "UNMAPPED_SUFFIX",
    "HolderSpan",
    "PermatickerResolver",
    "SeenSpan",
    "map_raw_table",
    "related_spans",
    "ticker_changes",
]

#: The ACTIONS type recording a ticker change: ``contraticker`` became ``ticker``.
TICKER_CHANGE = "tickerchangefrom"

#: Suffix of the report of raw rows left out for want of a permaticker.
UNMAPPED_SUFFIX = ".unmapped.json"

#: How the vendor fills an unused contra column.
_NOT_APPLICABLE = "N/A"

#: Number of offending keys a message lists.
_SAMPLE = 5


def _text_or_none(value) -> str | None:
    """Return ``value`` as text, or ``None`` for a null, blank or ``N/A``."""
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text == _NOT_APPLICABLE else text


def ticker_changes(
    vendor_root: str | Path, code: str | None
) -> dict[int, list[tuple[str, str, str | None]]]:
    """Return each permaticker's ticker changes, ``(date, old ticker, old company)``, by date.

    A ``tickerchangefrom`` row of ACTIONS says that on its ``date`` the
    security trading as ``contraticker`` (company ``contraname``) became
    ``ticker``. If a security changed away from that ticker later (``AAX``
    -> ``BBX`` -> ``BBB``: the change into ``BBX``), the row is that
    security's, the one whose change away from it comes first after the
    row's date, even when another company trades under the ticker now.
    Otherwise the row is the security the TICKERS rows of ``code`` (of every
    table for ``None``) give the ticker to. A row that maps to nothing, or to
    several permatickers, is left out.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``, holding TICKERS and ACTIONS.
    code : str or None
        The table whose TICKERS rows own the tickers (``"sep"``); ``None``
        for the rows of every table (funds included).

    Returns
    -------
    dict
        ``{permaticker: [(date as YYYY-MM-DD, old ticker, old company), ...]}``.

    Examples
    --------
    >>> ticker_changes("/data/downloads/sharadar", "sep")[194817]
    [('2022-06-09', 'FB', 'FACEBOOK INC')]
    >>> ticker_changes("/data/downloads/sharadar", None)[117736]  # a fund (SFP)
    [('2026-04-09', 'BJK', None)]
    """
    rows = (
        scan_raw_table(vendor_root, "actions")
        .filter(pl.col("action") == TICKER_CHANGE)
        .select("date", "ticker", "contraticker", "contraname")
        .collect()
        .sort("date", descending=True)
    )
    owners: dict[str, set[int]] = {}
    if code is None:
        mapping = scan_raw_table(vendor_root, "tickers").select("ticker", "permaticker").unique().collect()
    else:
        mapping = permaticker_mapping(vendor_root, code)
    for ticker, permaticker in mapping.drop_nulls().iter_rows():
        owners.setdefault(str(ticker), set()).add(int(permaticker))
    # Latest first, so a later change has resolved the ticker an earlier one
    # changed into before that earlier one is looked at.
    resolved: dict[int, list[tuple[str, str, str | None]]] = {}
    later: dict[str, list[tuple[str, int]]] = {}
    for day, ticker, old, old_company in rows.iter_rows():
        old = _text_or_none(old)
        if old is None or ticker is None:
            continue
        day = str(day)[:10]
        left = [(since, perm) for since, perm in later.get(str(ticker), []) if since > day]
        if left:
            permaticker = min(left)[1]
        else:
            candidates = owners.get(str(ticker), set())
            if len(candidates) != 1:
                continue
            (permaticker,) = candidates
        resolved.setdefault(permaticker, []).append((day, old, _text_or_none(old_company)))
        later.setdefault(old, []).append((day, permaticker))
    return {perm: sorted(changes) for perm, changes in resolved.items()}


def related_spans(vendor_root: str | Path, code: str) -> dict[tuple[str, int], tuple[date | None, date | None]]:
    """Return the former tickers TICKERS' ``relatedtickers`` shows, with the security's price span.

    ``relatedtickers`` lists, space-separated, a security's former tickers
    and also its related securities (other share classes, a SPAC's units,
    predecessors), most of them other securities' tickers. Only a token the
    security's current ticker is followed by a number (NSTR of NSTR1) counts:
    TICKERS appends one when it renames a security whose ticker another
    took over, so the token was this security's. It is dated from the
    security's first to the day after its last price (open-ended without
    one), as the rename is not dated.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The table whose TICKERS rows are read (``"sep"``).

    Returns
    -------
    dict
        ``{(former ticker, permaticker): (start, end)}``.

    Examples
    --------
    >>> related_spans("/data/downloads/sharadar", "sep")[("NSTR", 173730)]
    (datetime.date(2006, 5, 8), datetime.date(2009, 7, 21))
    """
    rows = (
        scan_raw_table(vendor_root, "tickers")
        .filter(
            pl.col("table").is_in(table(code).mapping_labels)
            & pl.col("relatedtickers").is_not_null()
        )
        .select("ticker", "permaticker", "relatedtickers", "firstpricedate", "lastpricedate")
        .unique()
        .collect()
    )
    spans: dict[tuple[str, int], tuple[date | None, date | None]] = {}
    for ticker, permaticker, related, first, last in rows.iter_rows():
        if ticker is None or permaticker is None:
            continue
        for token in str(related).split():
            if re.fullmatch(re.escape(token) + r"\d+", str(ticker)):
                end = None if last is None else _day(last) + timedelta(days=1)
                spans[(token, int(permaticker))] = (_day(first), end)
    return spans


@dataclass(frozen=True)
class HolderSpan:
    """One security's use of one ticker: from ``start`` (``None``: always) to ``end`` (``None``: still)."""

    permaticker: int
    start: date | None
    end: date | None

    def covers(self, day: date) -> bool:
        """Return whether the span holds ``day``."""
        return (self.start is None or self.start <= day) and (self.end is None or day < self.end)

    def held_at_pull(self, day: date) -> bool:
        """Return whether a file pulled on ``day`` names the security by the ticker.

        A change dated ``day`` shows only in the vendor's update that
        evening, so a file pulled that day still has the ticker's earlier
        holder: the span starts the day after its start and ends the day
        after its end.
        """
        return (self.start is None or self.start < day) and (self.end is None or day <= self.end)


@dataclass(frozen=True)
class SeenSpan:
    """One security under one ticker in the TICKERS history (``raw_tier.fold_tickers_history``).

    ``first`` and ``last`` are the pull times (UTC) of the first and last
    snapshots holding the pair; ``end`` is the day before a snapshot after
    ``last`` first shows the security under another ticker or the ticker
    under another security (``None``: none has).
    """

    first: datetime
    last: datetime
    end: date | None


def _day(value) -> date | None:
    """Return a ``YYYY-MM-DD`` text, a date or ``None`` as a date or ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


class PermatickerResolver:
    """Map each raw file's tickers to permatickers as of the file's pull.

    See the module docstring for the order. One resolver serves one
    mapping table (``code``: the TICKERS rows of ``tables.SharadarTable.mapping_labels``),
    and may annotate the files of several raw tables mapped through it (a
    price table and ACTIONS).

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The table whose TICKERS rows map the tickers (``"sep"``, ``"sf1"``).
    sidecar_path : str or Path, optional
        The store's ticker sidecar (``<store>.sharadar_tickers.json``): the
        securities it names keep their tickers as of when it was written,
        including the former ones it carries. Missing or unreadable, it is
        ignored.

    Examples
    --------
    >>> resolver = PermatickerResolver("/data/downloads/sharadar", "sep")
    >>> frame = scan_raw_table("/data/downloads/sharadar", "sep", annotate=resolver.annotate)
    >>> "permaticker" in frame.collect_schema().names()
    True
    """

    def __init__(
        self, vendor_root: str | Path, code: str, *, sidecar_path: str | Path | None = None
    ) -> None:
        """Initialize without reading anything; see the class docstring."""
        self.vendor_root = Path(vendor_root)
        self.code = table(code).code
        self.sidecar_path = None if sidecar_path is None else Path(sidecar_path)
        self._holders: dict[str, list[HolderSpan]] | None = None
        self._current: dict[str, set[int]] | None = None
        self._snapshots: dict[Path, dict[str, set[int]]] = {}
        self._snapshots_all: dict[Path, dict[str, set[int]]] = {}
        self._since: dict[str, dict[int, date | None]] | None = None
        self._seen: dict[int, dict[str, SeenSpan]] | None = None
        self._first_price: dict[int, date] | None = None
        #: Per raw file, the tickers it names that were not mapped, and why.
        self.unresolved: dict[Path, dict[str, str]] = {}
        #: ``{ticker: permatickers}`` mapped as the security's own ticker of a
        #: pull: by the file's TICKERS snapshot, or by current TICKERS for a
        #: file without one.
        self.own: dict[str, set[int]] = {}
        #: Tickers whose rows ``settle`` left out on some key, and why.
        self.ambiguous: dict[str, str] = {}

    # -- the sources ----------------------------------------------------------

    def _pairs(self, frame: pl.LazyFrame) -> dict[str, set[int]]:
        """Return ``{ticker: permatickers}`` from TICKERS-shaped rows of this table."""
        pairs: dict[str, set[int]] = {}
        rows = (
            frame.filter(pl.col("table").is_in(table(self.code).mapping_labels))
            .select("ticker", "permaticker")
            .unique()
            .collect()
        )
        for ticker, permaticker in rows.iter_rows():
            if ticker is not None and permaticker is not None:
                pairs.setdefault(str(ticker), set()).add(int(permaticker))
        return pairs

    def _snapshot(self, path: Path) -> dict[str, set[int]]:
        """Return the ``{ticker: permatickers}`` of one TICKERS snapshot, read once."""
        if path not in self._snapshots:
            self._snapshots[path] = self._pairs(pl.scan_parquet(path))
        return self._snapshots[path]

    def _snapshot_any_table(self, path: Path) -> dict[str, set[int]]:
        """Return the ``{ticker: permatickers}`` of every table's rows of one TICKERS snapshot, read once."""
        if path not in self._snapshots_all:
            pairs: dict[str, set[int]] = {}
            rows = pl.scan_parquet(path).select("ticker", "permaticker").unique().collect()
            for ticker, permaticker in rows.iter_rows():
                if ticker is not None and permaticker is not None:
                    pairs.setdefault(str(ticker), set()).add(int(permaticker))
            self._snapshots_all[path] = pairs
        return self._snapshots_all[path]

    def _current_since(self) -> dict[str, dict[int, date | None]]:
        """Return ``{ticker: {permaticker: since}}`` of current TICKERS, every table's rows, read once.

        ``since`` is the day the security took its ticker: its last ACTIONS
        ticker change, else its first price (``None`` without either).
        """
        if self._since is None:
            rows = (
                scan_raw_table(self.vendor_root, "tickers")
                .group_by("ticker", "permaticker")
                .agg(pl.col("firstpricedate").min())
                .collect()
            )
            try:
                changes = ticker_changes(self.vendor_root, None)
            except FileNotFoundError:
                changes = {}
            since: dict[str, dict[int, date | None]] = {}
            for ticker, permaticker, first in rows.iter_rows():
                if ticker is None or permaticker is None:
                    continue
                changed = changes.get(int(permaticker))
                start = _day(changed[-1][0]) if changed else _day(first)
                since.setdefault(str(ticker), {})[int(permaticker)] = start
            self._since = since
        return self._since

    def _holding_on(self, day: date) -> dict[str, set[int]]:
        """Return ``{ticker: permatickers}`` current TICKERS (every table) shows holding it on ``day``.

        A security counts from the day after it took the ticker
        (``_current_since``): a change shows in the vendor's update that
        evening, so a file pulled on the day of the change, or before it,
        names the ticker's earlier holder.
        """
        return {
            ticker: held
            for ticker, starts in self._current_since().items()
            if (held := {p for p, start in starts.items() if start is None or start < day})
        }

    def _current_pairs(self) -> dict[str, set[int]]:
        """Return the ``{ticker: permatickers}`` of current TICKERS, read once."""
        if self._current is None:
            self._current = self._pairs(scan_raw_table(self.vendor_root, "tickers"))
        return self._current

    def _history_spans(self) -> dict[int, dict[str, SeenSpan]]:
        """Return ``{permaticker: {ticker: SeenSpan}}`` of the TICKERS history, read once.

        Over this table's TICKERS rows of every snapshot folded
        (``quantlab.acquisition.sharadar.raw_tier.fold_tickers_history``). A span ends only on a pair seen
        after its last sighting: a pair merely first seen late (the vendor
        dropped its row from earlier snapshots) overlaps it and ends nothing.
        """
        if self._seen is None:
            history = tickers_history(self.vendor_root)
            pairs: list[tuple[str, int, datetime, datetime]] = []
            if history is not None and history.height:
                pairs = [
                    (str(t), int(p), first, last)
                    for t, p, first, last in history.filter(
                        pl.col("table").is_in(table(self.code).mapping_labels)
                    )
                    .group_by("ticker", "permaticker")
                    .agg(pl.col("first_seen").min(), pl.col("last_seen").max())
                    .iter_rows()
                ]
            firsts_of_ticker: dict[str, list[datetime]] = {}
            firsts_of_permaticker: dict[int, list[datetime]] = {}
            for ticker, permaticker, first, _ in pairs:
                firsts_of_ticker.setdefault(ticker, []).append(first)
                firsts_of_permaticker.setdefault(permaticker, []).append(first)
            seen: dict[int, dict[str, SeenSpan]] = {}
            for ticker, permaticker, first, last in pairs:
                # The pair's own first sighting is not after its last one.
                after = [
                    f
                    for f in firsts_of_ticker[ticker] + firsts_of_permaticker[permaticker]
                    if f > last
                ]
                end = min(after).date() - timedelta(days=1) if after else None
                seen.setdefault(permaticker, {})[ticker] = SeenSpan(first, last, end)
            self._seen = seen
        return self._seen

    def _taken_after(self, ticker: str, permaticker: int, day: date) -> bool:
        """Return whether the TICKERS history shows the security taking ``ticker`` only after ``day``.

        True when a snapshot on or after ``day`` shows the security under
        another ticker, before any shows it under ``ticker``: a file pulled
        on ``day`` cannot name the security by ``ticker``, whatever date
        ACTIONS gives the change (the vendor renames rows when TICKERS shows
        it). A pair merely first seen late (the vendor dropped the row from
        earlier snapshots) is not evidence.
        """
        tickers = self._history_spans().get(permaticker, {})
        span = tickers.get(ticker)
        if span is None:
            return False
        return any(
            other != ticker and seen.last.date() >= day and seen.last < span.first
            for other, seen in tickers.items()
        )

    def _listed_after(self, permaticker: int, day: date) -> bool:
        """Return whether the security's first price (TICKERS, any table) is after ``day``.

        Such a security, a later listing that took a ticker, owns no row of
        the ticker dated on or before ``day``. Without a first price, False.
        """
        if self._first_price is None:
            self._first_price = {
                int(permaticker): _day(first)
                for permaticker, first in scan_raw_table(self.vendor_root, "tickers")
                .group_by("permaticker")
                .agg(pl.col("firstpricedate").min())
                .collect()
                .iter_rows()
                if permaticker is not None and first is not None
            }
        first = self._first_price.get(permaticker)
        return first is not None and first > day

    def _sidecar_spans(self) -> list[tuple[str, int, date | None, date | None]]:
        """Return ``(ticker, permaticker, start, end)`` of every span the store's sidecar names.

        A span runs to the next span's start, the last one open-ended; a
        former ticker (``former``) runs from its recorded start, open-ended.
        """
        if self.sidecar_path is None or not self.sidecar_path.exists():
            return []
        try:
            payload = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
            spans = []
            for permaticker, entries in payload.get("intervals", {}).items():
                starts = [_day(entry.get("start")) for entry in entries]
                for k, entry in enumerate(entries):
                    end = starts[k + 1] if k + 1 < len(entries) else None
                    spans.append((str(entry["ticker"]), int(permaticker), starts[k], end))
            for permaticker, entries in payload.get("former", {}).items():
                for entry in entries:
                    spans.append((str(entry["ticker"]), int(permaticker), _day(entry.get("start")), None))
            return spans
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning(
                f"{type(self).__name__}: the ticker sidecar {str(self.sidecar_path)!r} is "
                f"unusable ({type(exc).__name__}: {exc}); tickers are mapped without it."
            )
            return []

    def ticker_holders(self) -> dict[str, list[HolderSpan]]:
        """Return each ticker's holders, one span per permaticker, built once.

        Every security known to have used the ticker is a holder, whatever
        its dates; ``resolve`` then chooses among them, and the Massive
        Ticker map (``quantlab.dataset.massive.ticker_map``) checks each
        against the day's close instead.

        A span's start is the security's change into the ticker (ACTIONS),
        else the sidecar's start, else its first price date (TICKERS); its
        end the change away (ACTIONS, an effective date: kept to the TICKERS
        history's last sighting of the pair when that is later, as the vendor
        renames rows only when it publishes the change), else none for its
        current ticker,
        else the day before the TICKERS history first shows it renamed or
        the ticker reused (``_history_spans``), else the sidecar's next
        span, else the day after its last price for a ``relatedtickers``
        former ticker (``related_spans``).

        Returns
        -------
        dict
            ``{ticker: [HolderSpan, ...]}``.

        Examples
        --------
        Needs the Sharadar raw tier::

            spans = PermatickerResolver("/data/downloads/sharadar", "sep").ticker_holders()["TWO"]
            [(span.permaticker, span.covers(date(2020, 3, 2))) for span in spans]
        """
        if self._holders is not None:
            return self._holders
        first_price = {
            int(permaticker): first
            for permaticker, first in scan_raw_table(self.vendor_root, "tickers")
            .filter(pl.col("table").is_in(table(self.code).mapping_labels))
            .group_by("permaticker")
            .agg(pl.col("firstpricedate").min())
            .collect()
            .iter_rows()
        }
        chain: dict[tuple[str, int], tuple[date | None, date | None]] = {}
        current_of = {perm: ticker for ticker, perms in self._current_pairs().items() for perm in perms}
        try:
            changes_by_permaticker = ticker_changes(self.vendor_root, self.code)
        except FileNotFoundError:
            # No ACTIONS pulled (a raw tier of prices and holdings only).
            changes_by_permaticker = {}
        for permaticker, changes in changes_by_permaticker.items():
            start = None
            for day, old, _ in changes:
                chain[(old, permaticker)] = (start, _day(day))
                start = _day(day)
            current = current_of.get(permaticker)
            if current is not None and start is not None:
                chain[(current, permaticker)] = (start, None)
        sidecar = {}
        for ticker, permaticker, start, end in self._sidecar_spans():
            sidecar.setdefault((ticker, permaticker), (start, end))
        related = related_spans(self.vendor_root, self.code)
        seen = {
            (ticker, permaticker): span
            for permaticker, tickers in self._history_spans().items()
            for ticker, span in tickers.items()
        }
        keys = set(chain) | set(sidecar) | set(related) | set(seen)
        keys |= {(t, p) for t, perms in self._current_pairs().items() for p in perms}
        holders: dict[str, list[HolderSpan]] = {}
        for ticker, permaticker in keys:
            chain_start, chain_end = chain.get((ticker, permaticker), (None, None))
            side_start, side_end = sidecar.get((ticker, permaticker), (None, None))
            rel_start, rel_end = related.get((ticker, permaticker), (None, None))
            start = chain_start or side_start or rel_start or first_price.get(permaticker)
            span = seen.get((ticker, permaticker))
            if (ticker, permaticker) in chain:
                end = chain_end
                # ACTIONS dates a change when it takes effect; the vendor renames
                # rows when it publishes it, which TICKERS shows, often weeks later.
                if end is not None and span is not None:
                    end = max(end, span.end or span.last.date())
            elif ticker in self._current_pairs() and permaticker in self._current_pairs()[ticker]:
                end = None
            elif span is not None:
                end = span.end
            elif (ticker, permaticker) in sidecar:
                end = side_end
            else:
                end = rel_end
            holders.setdefault(ticker, []).append(HolderSpan(permaticker, start, end))
        self._holders = holders
        return holders

    # -- mapping --------------------------------------------------------------

    def _refuse_ambiguous(self, pairs: dict[str, set[int]], tickers: Iterable[str], where: str) -> None:
        """Raise if one of ``tickers`` maps to several permatickers in ``pairs``."""
        ambiguous = {t: sorted(pairs[t]) for t in sorted(tickers) if len(pairs.get(t, ())) > 1}
        if ambiguous:
            sample = dict(list(ambiguous.items())[:_SAMPLE])
            raise ValueError(
                f"{len(ambiguous)} {self.code!r} ticker(s) map to several "
                f"permatickers in {where}, first {sample}. Refusing rather than "
                f"guessing which company a row belongs to; re-pull TICKERS."
            )

    def several_permatickers(self, path: str | Path, tickers: Iterable[str]) -> dict[str, list[int]]:
        """Return those of ``tickers`` that ``resolve`` would refuse for ``path``, each with its permatickers.

        A ticker is refused when the TICKERS snapshot of the file's pull, or
        current TICKERS for a ticker the snapshot does not hold, gives it to
        several permatickers. A caller that can leave such a ticker unmapped
        (an ACTIONS counterparty) asks this first and resolves the rest.

        Returns
        -------
        dict of str to list of int
            ``{ticker: [permaticker, ...]}``, sorted.

        Examples
        --------
        Needs the Sharadar raw tier::

            resolver.several_permatickers("/data/downloads/sharadar/actions/actions.parquet", ["DUP", "AAPL"])
            # {'DUP': [501, 502]}
        """
        left = {str(t) for t in tickers if t is not None}
        several: dict[str, list[int]] = {}
        snapshot = snapshot_for(self.vendor_root, pull_time(Path(path)))
        if snapshot is not None:
            pairs = self._snapshot(snapshot)
            known = left & set(pairs)
            several.update({t: sorted(pairs[t]) for t in known if len(pairs[t]) > 1})
            left -= known
        current = self._current_pairs()
        several.update({t: sorted(current[t]) for t in left if len(current.get(t, ())) > 1})
        return dict(sorted(several.items()))

    def resolve(
        self, path: str | Path, tickers: Iterable[str], *, latest: dict[str, date] | None = None
    ) -> dict[str, int]:
        """Return the permaticker of each of ``tickers`` as named in the raw file ``path``.

        Tickers left unmapped are recorded in ``unresolved`` under ``path``;
        those mapped as the security's own ticker of the pull in ``own``.

        Parameters
        ----------
        path : str or Path
            The raw file; its pull time decides the TICKERS snapshot and the
            holders (see the module docstring).
        tickers : iterable of str
            The tickers the file names.
        latest : dict of str to date, optional
            Each ticker's latest row date in the file. A security first
            priced after it owns none of the ticker's rows, so it does not
            stop them mapping to an earlier holder. ``annotate`` passes it
            for a table with a ``date`` column.

        Returns
        -------
        dict of str to int
            ``{ticker: permaticker}`` of the tickers mapped.

        Raises
        ------
        ValueError
            If the file's TICKERS snapshot, or current TICKERS, gives one of
            the tickers to several permatickers.

        Examples
        --------
        >>> resolver.resolve("/data/downloads/sharadar/sep/sep.parquet", ["LESL"])
        {'LESL': 632479}
        """
        path = Path(path)
        mapped, unresolved = self._resolve(pull_time(path), tickers, latest)
        if unresolved:
            self.unresolved.setdefault(path, {}).update(unresolved)
        return mapped

    def resolve_on(self, day: date, tickers: Iterable[str]) -> tuple[dict[str, int], dict[str, str]]:
        """Return the permaticker of each of ``tickers`` as traded on ``day``, and why the rest are not mapped.

        For a vendor that names each security by the ticker it traded under
        that day (Massive's trade files), rather than by the tickers of a
        later pull. A ticker change dated ``day`` is in force on ``day``, so
        the tickers are mapped as a file pulled at the start of the next day
        would name them; the order is the module docstring's.

        Returns
        -------
        mapped : dict of str to int
            ``{ticker: permaticker}``.
        unresolved : dict of str to str
            ``{ticker: reason}`` for the tickers left unmapped. Nothing is
            recorded in ``unresolved``, which is keyed by raw file.

        Raises
        ------
        ValueError
            If TICKERS (or the TICKERS snapshot of that pull) gives one of
            the tickers to several permatickers.

        Examples
        --------
        >>> resolver.resolve_on(date(2016, 11, 25), ["AAPL", "ZVZZT"])
        ({'AAPL': 199059}, {'ZVZZT': 'no permaticker'})
        """
        pulled = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        return self._resolve(pulled, tickers)

    def _resolve(
        self, pulled: datetime, tickers: Iterable[str], latest: dict[str, date] | None = None
    ) -> tuple[dict[str, int], dict[str, str]]:
        """Map ``tickers`` as named by a pull at ``pulled``; return the mapped and the unresolved.

        ``latest`` is ``resolve``'s: each ticker's latest row date in the file.
        """
        latest = latest or {}
        left = {str(t) for t in tickers if t is not None}
        mapped: dict[str, int] = {}
        snapshot = snapshot_for(self.vendor_root, pulled)
        if snapshot is not None:
            pairs = self._snapshot(snapshot)
            known = left & set(pairs)
            self._refuse_ambiguous(pairs, known, f"the TICKERS snapshot {snapshot.name}")
            mapped.update({t: next(iter(pairs[t])) for t in known})
            left -= known
        if left:
            self._refuse_ambiguous(self._current_pairs(), left, "TICKERS")
        holders = self.ticker_holders() if left else {}
        day = pulled.date()
        if snapshot is not None:
            others = self._snapshot_any_table(snapshot)
        else:
            others = self._holding_on(day) if left else {}
        unresolved: dict[str, str] = {}
        for ticker in left:
            spans = [s for s in holders.get(ticker, []) if not self._taken_after(ticker, s.permaticker, day)]
            held = set(others.get(ticker, ()))
            if snapshot is None:
                held = {p for p in held if not self._taken_after(ticker, p, day)}
            if ticker in latest:
                held = {p for p in held if not self._listed_after(p, latest[ticker])}
            permatickers = {span.permaticker for span in spans}
            if not spans:
                unresolved[ticker] = "no permaticker"
                continue
            chosen = spans[0].permaticker if len(permatickers) == 1 else self._holder_on(spans, day)
            if chosen is None:
                unresolved[ticker] = f"several securities used it ({sorted(permatickers)})"
            elif held and chosen not in held:
                where = "the TICKERS snapshot of its pull" if snapshot is not None else "TICKERS"
                unresolved[ticker] = (
                    f"{where} gives it to another security as of its pull ({sorted(held)})"
                )
            else:
                mapped[ticker] = chosen
        for ticker, permaticker in mapped.items():
            if ticker not in left or (
                snapshot is None and permaticker in self._current_pairs().get(ticker, ())
            ):
                self.own.setdefault(ticker, set()).add(permaticker)
        return mapped, unresolved

    @staticmethod
    def _holder_on(spans: list[HolderSpan], day: date) -> int | None:
        """Return the holder of a reused ticker on ``day``, or ``None`` on a tie.

        The span held at a pull on ``day`` (``HolderSpan.held_at_pull``), the
        latest-starting one if several are; with none, the one that ended
        last before it.
        """
        covering = [s for s in spans if s.held_at_pull(day)]
        if covering:
            best = max((s.start or date.min) for s in covering)
            chosen = {s.permaticker for s in covering if (s.start or date.min) == best}
        else:
            ended = [s for s in spans if s.end is not None and s.end < day]
            if not ended:
                return None
            best = max(s.end for s in ended)
            chosen = {s.permaticker for s in ended if s.end == best}
        return next(iter(chosen)) if len(chosen) == 1 else None

    def _used_on(self, ticker: str, permaticker: int, day) -> bool:
        """Return whether a span of the security's use of ``ticker`` holds ``day``."""
        if isinstance(day, datetime):
            day = day.date()
        day = _day(day)
        if day is None:
            return False
        spans = self.ticker_holders().get(ticker, [])
        return any(s.permaticker == permaticker and s.covers(day) for s in spans)

    def settle(
        self, frame: pl.DataFrame, key: Iterable[str], *, date_column: str = "date"
    ) -> pl.DataFrame:
        """Keep one row per security and key where several of its tickers give one.

        Rows of one permaticker sharing ``key`` under *different* tickers
        (several rows of one ticker are the caller's to judge) are settled,
        in order:

        1. the row of the security's own ticker as of a pull (``own``);
        2. else the row of the ticker the security used on the row's
           ``date_column`` (its holder span, see the module docstring);
        3. else, rows equal in every column but ``ticker`` and ``name``
           keep one.

        The other rows of the key are dropped (another name for the row
        kept). Rows none of these chooses between keep no permaticker:
        ``left_out`` then leaves them out and reports them, with the reason
        recorded in ``ambiguous``.

        Parameters
        ----------
        frame : pl.DataFrame
            Annotated rows (``annotate``) with ``ticker``, ``permaticker``,
            ``key`` and ``date_column``.
        key : iterable of str
            The columns that, with the permaticker, name one row
            (``("date",)`` for a price table).
        date_column : str, default "date"
            The date the row is of, checked against the tickers' spans.

        Returns
        -------
        pl.DataFrame
            ``frame`` with the settled rows dropped and the ambiguous ones
            without a permaticker, in its order.

        Examples
        --------
        >>> rows = resolver.settle(rows, ("date",))
        >>> rows = resolver.left_out(rows, owner="demo")
        """
        key = list(key)
        identity = ["permaticker", *key]
        indexed = frame.with_row_index("_settle_row")
        clashes = (
            indexed.filter(pl.col("permaticker").is_not_null())
            .filter(pl.col("ticker").n_unique().over(identity) > 1)
            .sort("_settle_row")
        )
        if clashes.height == 0:
            return frame
        compared = [c for c in frame.columns if c not in ("ticker", "name", "permaticker")]
        drop: list[int] = []
        unmap: list[int] = []
        for _, group in clashes.group_by(identity, maintain_order=True):
            permaticker = int(group["permaticker"][0])
            day = group[date_column][0]
            tickers = sorted(set(group["ticker"].to_list()))
            pool = [t for t in tickers if permaticker in self.own.get(t, ())] or tickers
            if len(pool) > 1:
                pool = [t for t in pool if self._used_on(t, permaticker, day)] or pool
            kept = group.filter(pl.col("ticker").is_in(pool))
            if len(pool) > 1 and kept.select(compared).unique().height == 1:
                pool = pool[:1]
                kept = kept.filter(pl.col("ticker") == pool[0])
            rows = group["_settle_row"].to_list()
            if len(pool) == 1:
                keep = set(kept["_settle_row"].to_list())
                drop.extend(r for r in rows if r not in keep)
            else:
                unmap.extend(rows)
                reason = (
                    f"several tickers {tickers} give security {permaticker} one "
                    f"{'/'.join(key)}, none its own nor the one it used then"
                )
                for ticker in tickers:
                    self.ambiguous.setdefault(ticker, reason)
        if drop:
            logger.info(
                f"{type(self).__name__}: {len(drop)} {self.code!r} row(s) name a security "
                f"by a second ticker on a key it has a row of under its own; dropped."
            )
        return (
            indexed.filter(~pl.col("_settle_row").is_in(drop))
            .with_columns(
                pl.when(pl.col("_settle_row").is_in(unmap))
                .then(None)
                .otherwise(pl.col("permaticker"))
                .alias("permaticker")
            )
            .drop("_settle_row")
        )

    def annotate(self, path: Path, frame: pl.LazyFrame) -> pl.LazyFrame:
        """Return one raw file's scan with a ``permaticker`` column, null where none maps.

        The hook ``scan_raw_table(..., annotate=resolver.annotate)`` calls for
        every file; it reads the file's distinct tickers and, for a table
        with a ``date`` column (SF3A's is text), each one's latest date.
        """
        schema = frame.collect_schema()
        if "date" in schema.names():
            # SF3A writes its date as text; a malformed one is no date, not an error.
            day = (
                pl.col("date").str.to_date("%Y-%m-%d", strict=False)
                if schema["date"] == pl.String
                else pl.col("date").cast(pl.Date)
            )
            dated = frame.group_by("ticker").agg(day.max()).collect()
            tickers = dated.get_column("ticker").to_list()
            latest = {str(t): d for t, d in dated.iter_rows() if t is not None and d is not None}
        else:
            tickers = frame.select(pl.col("ticker").unique()).collect().get_column("ticker").to_list()
            latest = {}
        mapped = self.resolve(path, tickers, latest=latest)
        pairs = pl.DataFrame(
            {"ticker": list(mapped), "permaticker": list(mapped.values())},
            schema={"ticker": pl.String, "permaticker": pl.Int64},
        )
        return frame.join(pairs.lazy(), on="ticker", how="left")

    def left_out(
        self,
        frame: pl.DataFrame,
        *,
        owner: str,
        report_path: str | Path | None = None,
        date_column: str = "date",
        quiet: bool = False,
        report_unmapped: bool = True,
    ) -> pl.DataFrame:
        """Drop the rows no permaticker was found for, report them, and return the rest.

        Parameters
        ----------
        frame : pl.DataFrame
            Annotated rows (``annotate``), with ``ticker`` and ``date_column``.
        owner : str
            Named in the warning.
        report_path : str or Path, optional
            Where to write the report (``<store>.unmapped.json``), atomically:
            ``{"table", "owner", "unmapped": [{"ticker", "first_date",
            "last_date", "rows", "raw_files", "reason"}, ...]}``. An existing
            report is deleted when nothing is left out, so its presence
            means bars are missing. ``None`` writes nothing.
        date_column : str, default "date"
            The column whose first and last values the report gives.
        quiet : bool, default False
            Log at ``info`` instead of ``warning``, for a table where rows
            of securities outside the mapping are expected (SF3A holds funds).
        report_unmapped : bool, default True
            List the rows of tickers that map to nothing in the report.
            ``False`` lists only the rows ``settle`` could not choose
            between, for such a table.

        Returns
        -------
        pl.DataFrame
            ``frame`` without the rows whose ``permaticker`` is null.

        Examples
        --------
        >>> kept = resolver.left_out(frame, owner="demo", report_path="sep.zarr.unmapped.json")
        """
        missing = frame.filter(pl.col("permaticker").is_null())
        reported = missing if report_unmapped else missing.filter(
            pl.col("ticker").is_in(list(self.ambiguous))
        )
        if reported.height == 0 and report_path is not None:
            Path(report_path).unlink(missing_ok=True)
        if missing.height == 0:
            if report_path is not None:
                Path(report_path).unlink(missing_ok=True)
            return frame
        summary = (
            reported.group_by("ticker")
            .agg(
                pl.col(date_column).min().alias("first_date"),
                pl.col(date_column).max().alias("last_date"),
                pl.len().alias("rows"),
            )
            .sort("ticker")
        )
        entries = []
        for ticker, first, last, rows in summary.iter_rows():
            files = sorted(p.name for p, names in self.unresolved.items() if ticker in names)
            reasons = sorted({names[ticker] for names in self.unresolved.values() if ticker in names})
            if ticker in self.ambiguous:
                reasons.append(self.ambiguous[ticker])
            entries.append(
                {
                    "ticker": ticker,
                    "first_date": None if first is None else str(first)[:10],
                    "last_date": None if last is None else str(last)[:10],
                    "rows": int(rows),
                    "raw_files": files,
                    "reason": "; ".join(reasons) or "no permaticker",
                }
            )
        if report_path is not None and entries:
            write_json_atomically(
                report_path,
                {"table": self.code, "owner": owner, "unmapped": entries},
                indent=2,
                sort_keys=True,
            )
        log = logger.info if quiet else logger.warning
        where = f"; listed in {report_path}" if report_path is not None and entries else ""
        tickers = missing.get_column("ticker").unique().sort().to_list()
        log(
            f"{owner}: {missing.height} raw row(s) of {len(tickers)} ticker(s) have "
            f"no permaticker (as of their own pull, through TICKERS, the ACTIONS "
            f"ticker changes and the store's ticker sidecar, or several tickers "
            f"give one security a row) and are left out{where}, first "
            f"{tickers[:_SAMPLE]}."
        )
        if ambiguous_rows := missing.filter(pl.col("ticker").is_in(list(self.ambiguous))).height:
            logger.warning(
                f"{owner}: {ambiguous_rows} raw row(s) are left out because several "
                f"tickers give one security a row on one key and none could be "
                f"chosen{where}."
            )
        return frame.filter(pl.col("permaticker").is_not_null())


def map_raw_table(
    vendor_root: str | Path,
    code: str,
    *,
    owner: str,
    query=None,
    store_path: str | Path | None = None,
    sidecar_path: str | Path | None = None,
    date_column: str = "date",
    quiet: bool = False,
    key: Iterable[str] | None = None,
    report_unmapped: bool = True,
) -> pl.DataFrame:
    """Read one raw table with each row's permaticker, leaving out and reporting the unmapped rows.

    The rows are mapped through the TICKERS rows of ``code``'s mapping
    labels (``tables.SharadarTable.mapping_labels``), each raw file as of its
    own pull (``PermatickerResolver``).

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The raw table to read (``"sf1"``, ``"daily"``, ``"events"``, ...).
    owner : str
        Named in messages.
    query : callable, optional
        Applied to the annotated scan before it is collected (filters, a
        column selection keeping ``ticker``, ``permaticker`` and
        ``date_column``).
    store_path : str or Path, optional
        The store the rows feed: the report goes to
        ``<store_path>.unmapped.json`` (``UNMAPPED_SUFFIX``). ``None`` writes
        no report.
    sidecar_path : str or Path, optional
        A ticker sidecar to map old tickers with (``PermatickerResolver``).
    date_column : str, default "date"
        The column the report dates rows by.
    quiet : bool, default False
        Log the left-out rows at ``info`` (``PermatickerResolver.left_out``).
    key : iterable of str, optional
        The columns (after ``query``) that, with the permaticker, name one
        row: rows of one security under several tickers on one key are
        settled (``PermatickerResolver.settle``). ``None`` settles nothing.
    report_unmapped : bool, default True
        ``PermatickerResolver.left_out``.

    Returns
    -------
    pl.DataFrame
        The rows with a ``permaticker`` column, unmapped ones left out.

    Raises
    ------
    ValueError
        If a ticker maps to several permatickers in TICKERS or in its file's
        TICKERS snapshot.

    Examples
    --------
    >>> rows = map_raw_table("/data/downloads/sharadar", "daily", owner="demo",
    ...                      store_path="/data/market/sharadar/sharadar_daily_1d/sharadar_daily_1d.zarr")
    >>> rows.columns[-1]
    'permaticker'
    """
    resolver = PermatickerResolver(vendor_root, code, sidecar_path=sidecar_path)
    frame = scan_raw_table(vendor_root, code, annotate=resolver.annotate)
    if query is not None:
        frame = query(frame)
    report = None if store_path is None else f"{store_path}{UNMAPPED_SUFFIX}"
    rows = frame.collect()
    if key is not None:
        rows = resolver.settle(rows, key, date_column=date_column)
    return resolver.left_out(
        rows,
        owner=owner,
        report_path=report,
        date_column=date_column,
        quiet=quiet,
        report_unmapped=report_unmapped,
    )
