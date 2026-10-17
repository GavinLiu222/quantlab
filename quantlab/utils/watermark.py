"""The watermark record: how far a raw tier reaches, read and written by one module.

A *watermark* is a small JSON file beside raw vendor data saying the day
through which that data is complete. Every raw tier keeps one at its own
granularity:

``"symbol"``
    One sidecar per symbol (``<watermark_path>/<SYMBOL>.json``), written by
    ``quantlab.acquisition.base.Acquisition`` and read by
    ``CoverageLedger``: ``last_date``, optionally ``start_date`` (the covered
    start) and ``no_data`` (the vendor returned nothing). Compact JSON, keys
    in that order.
``"table"``
    One per Sharadar table (``<sharadar>/<table>/_watermark.json``):
    ``table`` and ``through``.
``"daily_file"``
    One per Massive day-file type (``<massive>/<type>/_watermark.json``):
    ``data_type`` and ``through``.
``"endpoint"``
    One per Massive REST endpoint (``<massive>/rest/<endpoint>/_watermark.json``):
    ``endpoint``, ``field`` (the date field the pull advances on, or
    ``"snapshot"`` for a snapshot endpoint, whose watermark is the day of its
    last snapshot) and ``through``.

The last three are written with ``indent=2`` and sorted keys. These are the
shapes the raw tiers on disk already hold; this module reads and writes them
unchanged, so no raw tier is migrated. One difference is kept on purpose: a
corrupt or non-object per-symbol sidecar reads as no watermark (a refresh
must never crash on one; it costs at most a wider re-fetch), while a corrupt
file of the other three raises, as their readers always have.

Where each file lives is the vendor's raw-tier layout (``quantlab.dataset``
modules, ``CoverageLedger``); the readers there and the writers in
``quantlab.acquisition`` both go through ``read_watermark`` and
``write_watermark``. ``resume_from`` and ``holds`` are the two rules every
day-indexed pull shares: resume the day after the watermark (less an
overlap), and skip what the watermark already holds.

Examples
--------
>>> import tempfile
>>> from pathlib import Path
>>> path = Path(tempfile.mkdtemp()) / "_watermark.json"
>>> write_watermark(path, Watermark("table", "2024-01-11", name="sep"))
>>> read_watermark(path, "table").day
datetime.date(2024, 1, 11)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from quantlab.utils.atomic import write_json_atomically

#: Name of the watermark file of a table, day-file type or REST endpoint, beside its data.
WATERMARK_FILE = "_watermark.json"

#: The key naming what a watermark is of, by granularity (``None``: the file's name says it).
NAME_KEYS = {"symbol": None, "table": "table", "daily_file": "data_type", "endpoint": "endpoint"}

#: The granularities a watermark is kept at.
GRANULARITIES = tuple(NAME_KEYS)


def _check(granularity: str) -> str:
    """Return ``granularity``, or raise if it is not one of ``GRANULARITIES``."""
    if granularity not in NAME_KEYS:
        raise ValueError(f"watermark granularity {granularity!r} is not one of {GRANULARITIES}.")
    return granularity


@dataclass(frozen=True)
class Watermark:
    """How far one raw tier reaches.

    Parameters
    ----------
    granularity : str
        One of ``GRANULARITIES``.
    through : str or None
        The day (``YYYY-MM-DD``, as recorded) the data is complete through;
        ``None`` only for a per-symbol sidecar recording none.
    name : str or None, default None
        The table, day-file type or endpoint; ``None`` for a per-symbol
        sidecar, which its file's name identifies.
    field : str or None, default None
        For an endpoint, the date field the pull advances on (``"snapshot"``
        for a snapshot endpoint).
    start : str or None, default None
        For a per-symbol sidecar, the covered start, when recorded.
    no_data : bool, default False
        For a per-symbol sidecar, whether the vendor returned no rows.

    Examples
    --------
    >>> Watermark("endpoint", "2024-03-14", name="splits", field="execution_date").payload()
    {'endpoint': 'splits', 'field': 'execution_date', 'through': '2024-03-14'}
    >>> Watermark("symbol", "2024-01-31", start="2024-01-01").payload()
    {'last_date': '2024-01-31', 'start_date': '2024-01-01'}
    """

    granularity: str
    through: str | None
    name: str | None = None
    field: str | None = None
    start: str | None = None
    no_data: bool = False

    def __post_init__(self) -> None:
        """Validate the granularity."""
        _check(self.granularity)

    @property
    def day(self) -> date | None:
        """Return ``through`` as a date, or ``None`` without one.

        Examples
        --------
        >>> Watermark("daily_file", "2024-11-29", name="trades").day
        datetime.date(2024, 11, 29)
        """
        return None if self.through is None else date.fromisoformat(self.through)

    def payload(self) -> dict:
        """Return the record as its file holds it.

        A per-symbol sidecar leaves out an unknown start (so it cannot be
        mistaken for a recorded one) and ``no_data`` when false (an older
        sidecar without the key reads as "had data").

        Examples
        --------
        >>> Watermark("table", "2024-01-11", name="sep").payload()
        {'table': 'sep', 'through': '2024-01-11'}
        """
        if self.granularity == "symbol":
            payload: dict[str, str | bool] = {}
            if self.through is not None:
                payload["last_date"] = self.through
            if self.start is not None:
                payload["start_date"] = self.start
            if self.no_data:
                payload["no_data"] = True
            return payload
        payload = {NAME_KEYS[self.granularity]: self.name}
        if self.granularity == "endpoint":
            payload["field"] = self.field
        payload["through"] = self.through
        return payload

    @classmethod
    def from_payload(cls, granularity: str, payload: dict) -> Watermark:
        """Return the record a file's JSON object holds.

        Raises
        ------
        KeyError
            If a table, day-file or endpoint watermark has no ``through``.

        Examples
        --------
        >>> Watermark.from_payload("daily_file", {"data_type": "trades", "through": "2024-11-29"})
        Watermark(granularity='daily_file', through='2024-11-29', name='trades', field=None, start=None, no_data=False)
        """
        if _check(granularity) == "symbol":
            return cls(
                granularity,
                payload.get("last_date"),
                start=payload.get("start_date"),
                no_data=bool(payload.get("no_data", False)),
            )
        return cls(
            granularity,
            payload["through"],
            name=payload.get(NAME_KEYS[granularity]),
            field=payload.get("field"),
        )


def read_watermark(path: str | Path, granularity: str) -> Watermark | None:
    """Return the watermark in ``path``, or ``None`` when there is none.

    Parameters
    ----------
    path : str or Path
        The watermark file.
    granularity : str
        One of ``GRANULARITIES``: what the file is the watermark of.

    Returns
    -------
    Watermark or None
        ``None`` when the file is absent, or for a per-symbol sidecar that
        is corrupt or not a JSON object.

    Raises
    ------
    ValueError
        If ``granularity`` is unknown, or a table, day-file or endpoint
        watermark is not valid JSON.
    KeyError
        If such a watermark has no ``through``.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> read_watermark(Path(tempfile.mkdtemp()) / "_watermark.json", "endpoint") is None
    True
    """
    _check(granularity)
    path = Path(path)
    if not path.exists():
        return None
    if granularity != "symbol":
        return Watermark.from_payload(granularity, json.loads(path.read_text(encoding="utf-8")))
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        # A corrupt or half-written sidecar must never crash a refresh;
        # treating it as absent costs at most a wider re-fetch. Old
        # non-atomic writes may have left such files, so do not raise.
        return None
    return Watermark.from_payload(granularity, payload) if isinstance(payload, dict) else None


def write_watermark(path: str | Path, watermark: Watermark) -> None:
    """Write ``watermark`` to ``path`` atomically, in its granularity's shape.

    A per-symbol sidecar is compact JSON, so the sidecars already on disk stay
    byte-comparable with new ones; the others are ``indent=2`` with sorted
    keys. The file is written to a temporary file and renamed into place, so
    an interrupted write leaves the previous watermark.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> path = Path(tempfile.mkdtemp()) / "AAPL.json"
    >>> write_watermark(path, Watermark("symbol", "2024-01-31"))
    >>> path.read_text()
    '{"last_date": "2024-01-31"}'
    """
    if watermark.granularity == "symbol":
        write_json_atomically(path, watermark.payload())
    else:
        write_json_atomically(path, watermark.payload(), indent=2, sort_keys=True)


def resume_from(through: date | None, start: date, overlap_days: int = 0) -> date:
    """Return the first day a pull asks for: the day after the watermark, less an overlap.

    Never before ``start`` (the first day the pull would ask for without a
    watermark). Massive's day files and its REST endpoints resume this way;
    Sharadar's tables and the per-symbol acquisitions keep their own rules.

    Parameters
    ----------
    through : date or None
        The watermark's day, or ``None`` without one.
    start : date
        The first day to ask for without a watermark.
    overlap_days : int, default 0
        Days before the day after the watermark asked for again.

    Examples
    --------
    >>> resume_from(date(2024, 3, 14), date(2024, 1, 2))
    datetime.date(2024, 3, 15)
    >>> resume_from(date(2024, 3, 14), date(2024, 1, 2), overlap_days=3)
    datetime.date(2024, 3, 12)
    >>> resume_from(None, date(2024, 1, 2))
    datetime.date(2024, 1, 2)
    """
    if through is None:
        return start
    return max(start, through + timedelta(days=1 - overlap_days))


def holds(through: date | None, day: date) -> bool:
    """Return whether a watermark already reaches ``day``.

    A snapshot endpoint pulled once a day skips a pull whose day its
    watermark holds.

    Examples
    --------
    >>> holds(date(2026, 10, 16), date(2026, 10, 16))
    True
    >>> holds(None, date(2026, 10, 16))
    False
    """
    return through is not None and through >= day
