"""Dated watermark files: the day through which some raw data is complete.

A *dated watermark* is a small JSON file beside raw data saying the day
through which that data is complete. It holds ``through`` (``YYYY-MM-DD``)
and any labels its writer adds to say what it is the watermark of; the file
is written with ``indent=2`` and sorted keys. What the labels are and where
the file lives is the owner's layout: the reader and the writer of each raw
tier pass them. This module knows no vendor.

``resume_from`` and ``holds`` are the two rules every day-indexed pull
shares: resume the day after the watermark (less an overlap), and skip what
the watermark already holds.

Examples
--------
>>> import tempfile
>>> from pathlib import Path
>>> path = Path(tempfile.mkdtemp()) / WATERMARK_FILE
>>> write_watermark(path, date(2024, 1, 11), {"name": "prices"})
>>> print(path.read_text())
{
  "name": "prices",
  "through": "2024-01-11"
}
>>> read_watermark(path)
datetime.date(2024, 1, 11)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

from quantlab.utils.atomic import write_json_atomically

#: The usual name of a dated watermark file, beside the data it covers.
WATERMARK_FILE = "_watermark.json"

#: The key holding the day a dated watermark reaches.
THROUGH_KEY = "through"


def read_watermark(path: str | Path) -> date | None:
    """Return the day the dated watermark in ``path`` reaches, or ``None`` without a file.

    Parameters
    ----------
    path : str or Path
        The watermark file.

    Returns
    -------
    date or None
        ``None`` when the file is absent.

    Raises
    ------
    ValueError
        If the file is not valid JSON.
    KeyError
        If it has no ``through``.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> read_watermark(Path(tempfile.mkdtemp()) / WATERMARK_FILE) is None
    True
    """
    path = Path(path)
    if not path.exists():
        return None
    return date.fromisoformat(json.loads(path.read_text(encoding="utf-8"))[THROUGH_KEY])


def write_watermark(path: str | Path, through: date, labels: Mapping[str, str] | None = None) -> None:
    """Write a dated watermark reaching ``through`` to ``path`` atomically.

    The file holds ``labels`` and ``through``, ``indent=2`` with sorted keys.
    It is written to a temporary file and renamed into place, so an
    interrupted write leaves the previous watermark.

    Parameters
    ----------
    path : str or Path
        The watermark file.
    through : date
        The day the data is complete through.
    labels : Mapping[str, str] or None, default None
        What the watermark is of (for example ``{"name": "prices"}``).

    Raises
    ------
    ValueError
        If ``labels`` holds ``through``.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> path = Path(tempfile.mkdtemp()) / WATERMARK_FILE
    >>> write_watermark(path, date(2024, 3, 14), {"name": "events", "field": "updated"})
    >>> json.loads(path.read_text())
    {'field': 'updated', 'name': 'events', 'through': '2024-03-14'}
    """
    labels = dict(labels or {})
    if THROUGH_KEY in labels:
        raise ValueError(f"a watermark's labels cannot hold {THROUGH_KEY!r}; pass the day as through.")
    write_json_atomically(path, {**labels, THROUGH_KEY: through.isoformat()}, indent=2, sort_keys=True)


def resume_from(through: date | None, start: date, overlap_days: int = 0) -> date:
    """Return the first day a pull asks for: the day after the watermark, less an overlap.

    Never before ``start`` (the first day the pull would ask for without a
    watermark).

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

    A snapshot pulled once a day skips a pull whose day its watermark holds.

    Examples
    --------
    >>> holds(date(2026, 10, 16), date(2026, 10, 16))
    True
    >>> holds(None, date(2026, 10, 16))
    False
    """
    return through is not None and through >= day
