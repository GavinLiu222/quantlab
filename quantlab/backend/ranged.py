"""A Zarr store written and read by date range: ``RangedStore``.

A *ranged store* answers a request for a date range only inside the range it
was built for, which is recorded beside it in ``<store>.range.json``.
``build(start, end)`` replaces the store with the rows of the range and
records it, ``extend(end)`` appends the rows after the recorded range and
moves its end, ``rewind(end)`` drops the rows after an earlier end and moves
the end back, and a reader asks ``check_covers(start, end)`` before it opens
the store. Every row is computed by the ``compute`` function the store is
given, so a store built in one go and one built then extended are equal
whenever a row depends only on a bounded window of bars ending at its own.

A factor's store (``quantlab.factor.base.Factor``) and a factor risk model's
two stores (``quantlab.risk.base.RiskStore``) are written through one; each
supplies only its compute function, the name its messages give, and (a
factor) the rule refusing a write and the fill values of widened cells. How
a store is read (which variables and symbols, what is recorded) stays the
adapter's.

Examples
--------
>>> store = RangedStore("rows.zarr", "Rows", compute)   # compute(start, end) -> xr.Dataset
>>> store.build("2024-01-01", "2024-03-31")
>>> store.store_range()
('2024-01-01', '2024-03-31')
>>> store.extend("2024-04-30")
>>> store.store_range()
('2024-01-01', '2024-04-30')
"""

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.date_range import last_moment, range_text
from quantlab.utils.timer import Timer

__all__ = ["RangedStore"]


class RangedStore:
    """A store written by date range, the range recorded beside it.

    Parameters
    ----------
    path : str or None
        The Zarr store. ``None`` when the config names none: every method
        but ``store_range`` then refuses.
    owner : str
        What the store is called in error and timer messages: a factor's
        class name (``"Momentum"``), a risk model's ``Class.part``
        (``"Use4RiskModel.regression"``).
    compute : callable
        ``compute(start, end)`` returning the rows of the range, both
        inclusive, as an ``xr.Dataset`` on ``timestamp`` (and its other
        axes). It reads its own warm-up.
    guard : callable, optional
        ``guard(method)`` raises when the store may not be written
        (``method`` is ``"build"`` or ``"extend"``); it runs before
        anything is computed or written. A factor's owner rule.
    fill_values : mapping, optional
        Per-variable fill values for cells a widening creates (see
        ``XrBackend.widen_and_append``).

    Attributes
    ----------
    store_path : str or None
        As ``path``.
    owner : str
        As given.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> def compute(start, end):
    ...     times = pd.bdate_range(start, end)
    ...     return xr.Dataset(
    ...         {"x": (("timestamp", "symbol"), np.ones((len(times), 2)))},
    ...         coords={"timestamp": times, "symbol": ["a", "b"]},
    ...     )
    >>> store = RangedStore("rows.zarr", "Rows", compute)
    >>> store.build("2024-01-01", "2024-01-10")
    >>> store.store_range()
    ('2024-01-01', '2024-01-10')
    >>> store.check_covers("2024-01-02", "2024-01-11")
    Traceback (most recent call last):
    ValueError: Rows.read(): the store at rows.zarr covers 2024-01-01 to 2024-01-10, which does not contain 2024-01-02 to 2024-01-11. Extend it with extend(end) or rebuild it with build(start, end).
    """

    #: Appended to the store's path to name the file recording its range.
    RANGE_SUFFIX = ".range.json"

    def __init__(
        self,
        path: str | None,
        owner: str,
        compute: Callable[[object, object], xr.Dataset],
        *,
        guard: Callable[[str], None] | None = None,
        fill_values: Mapping[str, object] | None = None,
    ):
        """Initialize the store; see the class docstring for parameters."""
        self.store_path = path
        self.owner = owner
        self._compute = compute
        self._guard = guard
        self._fill_values = fill_values

    def __repr__(self) -> str:
        """Return the owner and the path."""
        return f"RangedStore({self.owner!r}, path={self.store_path!r})"

    @property
    def range_path(self) -> Path | None:
        """The file recording the store's range, ``<path>.range.json``.

        Examples
        --------
        >>> RangedStore("rows.zarr", "Rows", compute).range_path
        PosixPath('rows.zarr.range.json')
        """
        if self.store_path is None:
            return None
        return Path(f"{self.store_path}{self.RANGE_SUFFIX}")

    def store_range(self) -> tuple[str, str] | None:
        """Return the ``(start, end)`` the store was built for; ``None`` when none is recorded.

        Examples
        --------
        >>> store.build("2024-01-01", "2024-01-10")
        >>> store.store_range()
        ('2024-01-01', '2024-01-10')
        """
        if self.range_path is None or not self.range_path.is_file():
            return None
        recorded = json.loads(self.range_path.read_text())
        return recorded["start"], recorded["end"]

    def check_covers(self, start, end) -> None:
        """Raise unless the recorded range contains ``start`` to ``end``: a read's refusals.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range a reader asks for. A date-only ``end`` includes every
            bar of that day.

        Raises
        ------
        ValueError
            If the store has no path or no recorded range, or the recorded
            range does not contain the request.

        Examples
        --------
        >>> store.check_covers("2024-01-02", "2024-01-09")   # inside: nothing happens
        >>> store.check_covers("2024-01-02", "2024-01-11")
        Traceback (most recent call last):
        ValueError: Rows.read(): the store at rows.zarr covers 2024-01-01 to 2024-01-10, ...
        """
        path = self._require_path("read")
        recorded = self.store_range()
        if recorded is None:
            raise ValueError(
                f"{self.owner}.read(): the store at {path} has no recorded range, so it "
                f"cannot answer a date-range request; write it with build(start, end)."
            )
        recorded_start, recorded_end = recorded
        if pd.Timestamp(start) < pd.Timestamp(recorded_start) or last_moment(
            end
        ) > last_moment(recorded_end):
            raise ValueError(
                f"{self.owner}.read(): the store at {path} covers {recorded_start} to "
                f"{recorded_end}, which does not contain {start} to {end}. Extend it "
                f"with extend(end) or rebuild it with build(start, end)."
            )

    def build(self, start, end) -> None:
        """Compute ``start`` to ``end`` and write it as the store, recording the range.

        The store and its recorded range are replaced.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range the store covers.

        Raises
        ------
        ValueError
            If the store has no path, or the guard refuses the write.

        Examples
        --------
        >>> store.build("2024-01-01", "2024-01-10")
        >>> store.store_range()
        ('2024-01-01', '2024-01-10')
        """
        self._require_path("build")
        self._check_guard("build")
        self.write(self._compute(start, end), start, end)

    def write(self, rows: xr.Dataset, start, end) -> None:
        """Replace the store with ``rows``, computed over ``start`` to ``end``, and record the range.

        The write half of ``build``, for rows computed elsewhere (a chunk
        computed in a worker process).

        Raises
        ------
        ValueError
            If the store has no path, or the guard refuses the write.

        Examples
        --------
        >>> store.write(compute("2024-01-01", "2024-01-05"), "2024-01-01", "2024-01-05")
        >>> store.store_range()
        ('2024-01-01', '2024-01-05')
        """
        path = self._require_path("build")
        self._check_guard("build")
        with Timer(f"{self.owner}: build"):
            self.range_path.unlink(missing_ok=True)
            XrBackend().to_internal(rows).create(path, fill_values=self._fill_values)
            self._record(start, end)

    def extendable_range(self, end) -> tuple[str, str]:
        """Return the recorded range, refusing an ``extend`` to ``end``.

        Raises
        ------
        ValueError
            If the store has no path, the guard refuses the write, the
            store has no recorded range, or the recorded range already
            reaches ``end``.

        Examples
        --------
        >>> store.extendable_range("2024-01-31")
        ('2024-01-01', '2024-01-10')
        >>> store.extendable_range("2024-01-05")
        Traceback (most recent call last):
        ValueError: Rows.extend(): the store at rows.zarr already covers 2024-01-01 to 2024-01-10; ...
        """
        path = self._require_path("extend")
        self._check_guard("extend")
        recorded_start, recorded_end = self._recorded("extend")
        if last_moment(end) <= last_moment(recorded_end):
            raise ValueError(
                f"{self.owner}.extend(): the store at {path} already covers "
                f"{recorded_start} to {recorded_end}; extend() appends only bars after "
                f"{recorded_end}, got end {end!r}."
            )
        return recorded_start, recorded_end

    def extend(self, end) -> None:
        """Append the rows after the recorded range, up to ``end``, and move its end.

        The rows are computed from the first moment after the recorded end,
        so they are warmed from the inputs' history; the store's axes widen
        to fit.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range.

        Raises
        ------
        ValueError
            As ``extendable_range``.

        Examples
        --------
        >>> store.extend("2024-01-31")
        >>> store.store_range()
        ('2024-01-01', '2024-01-31')
        """
        _, recorded_end = self.extendable_range(end)
        after = last_moment(recorded_end) + pd.Timedelta(1, "ns")
        self.append(self._compute(after, end), end)

    def append(self, rows: xr.Dataset, end) -> None:
        """Append ``rows``, the bars after the recorded range up to ``end``, and move its end.

        The write half of ``extend``, for rows computed elsewhere. Empty
        ``rows`` (no bar in the range) only move the end.

        Raises
        ------
        ValueError
            As ``extendable_range``.

        Examples
        --------
        >>> store.append(compute("2024-01-08", "2024-01-10"), "2024-01-10")
        >>> store.store_range()
        ('2024-01-01', '2024-01-10')
        """
        recorded_start, recorded_end = self.extendable_range(end)
        with Timer(f"{self.owner}: extend"):
            # Bars past the recorded end are a rewind killed before its cut.
            XrBackend().shorten_axis(self.store_path, last_moment(recorded_end))
            if rows.sizes["timestamp"]:
                XrBackend().to_internal(rows).widen_and_append(
                    self.store_path, fill_values=self._fill_values
                )
            self._record(recorded_start, end)

    def rewind(self, end) -> None:
        """Drop the bars after ``end`` and end the recorded range there: the inverse of ``extend``.

        The bars up to ``end`` stay as they are, and the symbol and variable
        axes keep every label, so the next ``extend`` computes the dropped
        bars again. The range is recorded first, then the store's time axis
        shortened in place (``XrBackend.shorten_axis``): a rewind killed
        between the two leaves bars past the recorded end, which no read
        reaches and the next ``extend`` drops before it appends.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range, inside it and before its end.

        Raises
        ------
        ValueError
            If the store has no path, the guard refuses the write, the store
            has no recorded range, ``end`` is not before the recorded end, or
            ``end`` is before the recorded start.

        Examples
        --------
        >>> store.build("2024-01-01", "2024-01-31")
        >>> store.rewind("2024-01-15")
        >>> store.store_range()
        ('2024-01-01', '2024-01-15')
        >>> store.rewind("2024-01-15")
        Traceback (most recent call last):
        ValueError: Rows.rewind(): the store at rows.zarr covers 2024-01-01 to 2024-01-15; ...
        """
        path = self._require_path("rewind")
        self._check_guard("rewind")
        recorded_start, recorded_end = self._recorded("rewind")
        if last_moment(end) >= last_moment(recorded_end):
            raise ValueError(
                f"{self.owner}.rewind(): the store at {path} covers {recorded_start} to "
                f"{recorded_end}; rewind() keeps the bars up to an earlier end, got end {end!r}."
            )
        if last_moment(end) < pd.Timestamp(recorded_start):
            raise ValueError(
                f"{self.owner}.rewind(): the store at {path} covers {recorded_start} to "
                f"{recorded_end}; rewind() cannot end before its start, got end {end!r}. "
                f"Rebuild it with build(start, end)."
            )
        with Timer(f"{self.owner}: rewind"):
            self._record(recorded_start, end)
            XrBackend().shorten_axis(path, last_moment(end))

    def _recorded(self, method: str) -> tuple[str, str]:
        """Return the recorded range, or raise naming ``method`` when none is recorded."""
        recorded = self.store_range()
        if recorded is None:
            raise ValueError(
                f"{self.owner}.{method}(): the store at {self.store_path} has no recorded range; "
                f"write it with build(start, end) first."
            )
        return recorded

    def _require_path(self, method: str) -> str:
        """Return the path, or raise if the store has none."""
        if self.store_path is None:
            raise ValueError(f"{self.owner}.{method}(): the store has no path in the config.")
        return self.store_path

    def _check_guard(self, method: str) -> None:
        """Run the guard, which raises when the store may not be written."""
        if self._guard is not None:
            self._guard(method)

    def _record(self, start, end) -> None:
        """Record ``start`` and ``end`` as the store's range."""
        write_json_atomically(
            self.range_path, {"start": range_text(start), "end": range_text(end)}
        )
