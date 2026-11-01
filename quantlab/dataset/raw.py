"""The store build path: datasets converted from raw vendor files into a store they own.

``quantlab.dataset.base.BaseDataset`` is the read interface every dataset
answers; it has no build path. The datasets that convert a vendor's raw
files (the *raw tier*, the files as downloaded) into a Zarr store of their
own derive from ``RawDataset`` here, or from ``RawMarketDataset`` for price
bars, and inherit the whole build path:

- ``from_raw_data`` and ``save`` convert the whole configured range at once
  and write it;
- ``from_raw_data_chunked`` converts one time window at a time on a *pinned
  symbol axis* (every symbol of the range, worked out once), appending each
  window and recording it in a resume ledger (``<store>.chunks.json``), so an
  interrupted run resumes; a symbol the store lacks (a *new listing*) is
  refused, widened in with NaN history, or rebuilt in from the raw files,
  with the original store set aside and put back if a rebuild fails;
- ``update`` is ``from_raw_data_chunked`` with the new-listing strategy
  chosen from raw-tier evidence.

A subclass implements ``_raw_data_to_xr`` (and, for ``RawMarketDataset``,
``_raw_data_to_xr_window`` and ``_to_kunquant``); hooks such as
``_raw_axes_in_range``, ``_added_symbols_with_raw_history``,
``_continue_store``, ``_widen_fill_values`` and ``_clean`` let it read its raw
files more cheaply or continue what is stored. ``IndexConstituentDataset`` is
the raw dataset of a point-in-time index membership built from intervals.

A view (a roster cut, a merge, a mapping) or a panel held in memory is not a
raw dataset and has none of these methods.
"""

import copy
import dataclasses
import datetime
import os
import shutil
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.backend.zarr import XrBackend
from quantlab.dataset._support.cleaning import clean_market_data, clean_membership_panel
from quantlab.dataset.base import BaseDataset, MarketDataset
from quantlab.dataset.config import ConstituentDatasetConfig, RawDatasetConfig
from quantlab.enums.constant import Date
from quantlab.utils.progress import CancelToken, ProgressEvent, ProgressReporter
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

__all__ = ["ConversionResult", "IndexConstituentDataset", "RawDataset", "RawMarketDataset"]


@dataclass(frozen=True)
class ConversionResult:
    """Summary of one chunked conversion from raw files to a Zarr store.

    ``RawDataset.from_raw_data_chunked`` stores an instance on the
    dataset's ``last_chunk_result`` attribute when it finishes, so a caller
    can report what the run did. The attributes are documented inline below.

    Examples
    --------
    >>> ds = DemoDataset(config).from_raw_data_chunked(granularity="month")
    >>> result = ds.last_chunk_result
    >>> result.windows_written, result.rows_written, result.resumed
    (1, 6, False)
    """

    #: The Zarr store that was written (``config.zarr_file_path``).
    zarr_path: str
    #: Path of the ledger file (a JSON sidecar next to the store) that records
    #: completed windows.
    ledger_path: str
    #: Window granularity the run planned on (``"year"``, ``"quarter"``, ...).
    granularity: str
    #: Number of symbols on the pinned symbol axis, the fixed list of symbols
    #: for the whole date range that every window is written on.
    pinned_symbols: int
    #: Number of windows the planner produced over the observed timestamps.
    windows_planned: int
    #: Number of windows this run built and appended.
    windows_written: int
    #: Number of windows skipped because the ledger already recorded them.
    windows_skipped: int
    #: Rows appended along the append dimension, summed over written windows.
    rows_written: int
    #: Largest in-memory size (``window.nbytes``) of any window this run built;
    #: ``None`` when no window was written.
    peak_window_bytes: int | None
    #: The caller's own estimate of the peak, passed through unchanged;
    #: usually ``None``.
    predicted_peak_bytes: int | None = None
    #: Whether at least one window was skipped, meaning this run continued an
    #: earlier one.
    resumed: bool = False
    #: Whether a ``CancelToken`` stopped the loop between windows. Without this
    #: flag a cancelled run would look like a run that had less work to do.
    cancelled: bool = False
    #: Whether an ``on_new_listing="rebuild"`` run was cancelled and rolled
    #: back, restoring the previous store and ledger. Only ``True`` together
    #: with ``cancelled``. When set, ``windows_written`` and ``rows_written``
    #: describe work that no longer exists on disk.
    rebuild_rolled_back: bool = False


class RawDataset(BaseDataset):
    """A dataset converted from raw vendor files into a store it owns.

    A subclass implements ``_raw_data_to_xr``, which reads the raw files and
    returns a *dense* panel (a full ``timestamp`` by ``symbol`` grid, with
    NaN where a symbol has no value). Everything else is inherited: the read
    interface of ``BaseDataset``, Zarr storage through ``XrBackend``, the
    cleaning hook, and chunked ingestion with its resume ledger and its
    handling of *new listings* (symbols that appear in the raw tier but not
    yet in the store). Only the build path (``from_raw_data``,
    ``from_raw_data_chunked``, ``update``) holds a panel; reading stays a
    query that holds nothing. Datasets that are not price bars, such as a
    rate or a boolean membership panel, derive from this class directly;
    price bars derive from ``RawMarketDataset``.

    Parameters
    ----------
    config : RawDatasetConfig
        The dataset config. It is not modified: the dataset holds a
        normalised copy; see the ``config`` property.

    Attributes
    ----------
    data_backend : XrBackend
        Holds the panel the build path converted, and writes the store.
        Requests read through a fresh instance of its type, so a dataset
        reads the medium it writes.
    last_chunk_result : ConversionResult or None
        Summary of the last ``from_raw_data_chunked`` or ``update`` run, or
        ``None`` if neither has completed on this object.

    Examples
    --------
    A minimal subclass and the whole storage lifecycle::

        class MembershipDataset(RawDataset):
            def _raw_data_to_xr(self) -> xr.Dataset:
                return load_membership_panel(self.config.kwargs["source"])

        ds = MembershipDataset(config)
        ds.from_raw_data().save()
        panel = MembershipDataset(config).panel("2020-01-01", "2020-12-31")

    The method examples below use the ``DemoDataset`` described on
    ``BaseDataset``.
    """

    def __init__(self, config: RawDatasetConfig):
        """Initialize the dataset; see the class docstring for parameters.

        The backend is created before the config is assigned because a
        config setter may use the backend; swapping the two steps raises
        ``AttributeError``.
        """
        # `from_raw_data_chunked()` stores its summary here; a dataset that was
        # never built reports None.
        self.last_chunk_result: "ConversionResult | None" = None

        self.data_backend = XrBackend()
        self.config = config

    def copy(self) -> Self:
        """Return a copy with its own config and an empty storage backend.

        The config is deep-copied, so the copy shares no mutable state with
        this object, not even a ``kwargs`` dict. The copy holds no panel
        until it is built.

        Examples
        --------
        >>> other = ds.copy()
        >>> other.config is ds.config, other.data_backend is ds.data_backend
        (False, False)
        """
        other = copy.copy(self)
        other.data_backend = type(self.data_backend)()
        other.last_chunk_result = None
        other.config = copy.deepcopy(self.config)
        return other

    def _normalize_config(self, config: RawDatasetConfig) -> RawDatasetConfig:
        """Return ``config`` with ``name`` set and its build dates filled and checked.

        A missing ``start_date`` or ``end_date`` falls back to
        ``Date.START_DATE`` or ``Date.END_DATE``. Both dates must be ISO
        ``YYYY-MM-DD`` strings; a ``datetime.date`` is accepted and
        converted. Later code compares dates as plain strings, so a value
        such as ``"2007-2-1"`` would silently compare wrong; it is refused
        here instead.

        Parameters
        ----------
        config : RawDatasetConfig
            The config to normalise. It is not modified.

        Returns
        -------
        RawDatasetConfig
            A new config of the same class.

        Raises
        ------
        ValueError
            If a date is not an ISO ``YYYY-MM-DD`` date.

        Examples
        --------
        >>> ds._normalize_config(dataclasses.replace(ds.config, start_date=None)).start_date
        '1900-01-01'
        >>> ds._normalize_config(
        ...     dataclasses.replace(ds.config, start_date="01/02/2024")
        ... )
        Traceback (most recent call last):
        ValueError: DemoDataset: start_date must be an ISO YYYY-MM-DD date ...
        """
        config = super()._normalize_config(config)
        start_date = Date.START_DATE if config.start_date is None else config.start_date
        end_date = Date.END_DATE if config.end_date is None else config.end_date
        return dataclasses.replace(
            config,
            start_date=self._normalize_date(start_date, "start_date"),
            end_date=self._normalize_date(end_date, "end_date"),
        )

    def _normalize_date(self, value: str, field_name: str) -> str:
        """Return ``value`` as a zero-padded ISO date string.

        ``field_name`` is used only in the error message.

        Raises
        ------
        ValueError
            If ``value`` is not an ISO ``YYYY-MM-DD`` date.
        """
        try:
            return datetime.date.fromisoformat(str(value)).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.class_name}: {field_name} must be an ISO YYYY-MM-DD "
                f"date string, got {value!r}. Dates are compared "
                f"lexicographically throughout this pipeline, so a non-ISO "
                f"value compares wrong rather than failing to match."
            ) from exc

    def _store_backend(self) -> XrBackend:
        """Return a fresh backend of ``data_backend``'s type, so reads go through the medium written."""
        return type(self.data_backend)()

    #: Accepted values of ``on_new_listing``: stop (``"refuse"``), rewrite the
    #: whole store (``"rebuild"``), or add NaN columns (``"widen"``).
    NEW_LISTING_STRATEGIES: tuple[str, ...] = ("refuse", "rebuild", "widen")

    #: Marker value that ``update()`` passes to ``from_raw_data_chunked()`` to
    #: have the new-listing strategy chosen from raw-tier evidence. It is an
    #: ``object()`` rather than a string so no command-line flag, config file
    #: or JSON value can produce it, and it is not in
    #: ``NEW_LISTING_STRATEGIES``, which is what the command line offers.
    _AUTOMATIC = object()

    #: Maximum number of symbols named in a new-listing log message.
    NEW_LISTING_REPORT_LIMIT: int = 20

    @property
    def num_symbols(self) -> int:
        """Return the number of symbols in the built panel.

        Examples
        --------
        >>> ds.num_symbols
        3
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.size

    @property
    def _progress_vendor(self) -> str:
        """Return the value for ``ProgressEvent.vendor`` during a conversion.

        A conversion has no vendor client, so the config's ``vendor`` is used
        when it has one and the class name otherwise. The field must not be
        empty.
        """
        return getattr(self.config, "vendor", None) or self.class_name

    def _emit_progress(
        self, reporter: ProgressReporter | None, event: ProgressEvent
    ) -> None:
        """Deliver one progress event to ``reporter`` without ever raising.

        A ``None`` reporter does nothing. An exception raised by the reporter
        is logged as a warning and otherwise ignored, so a broken display
        cannot end a multi-hour conversion. Conversion events carry only
        window dates, counts and the class name, so there are no credentials
        to remove from them.
        """
        if reporter is None:
            return
        try:
            reporter.emit(event)
        except Exception as exc:  # noqa: BLE001 -- a reporter must never stop the run
            logger.warning(
                f"{self.class_name}: progress reporter "
                f"{type(reporter).__name__} raised on a {event.kind!r} event "
                f"and was ignored; the conversion is unaffected. "
                f"{type(exc).__name__}: {exc}"
            )

    @property
    def symbols(self) -> list[str]:
        """Return the symbol labels of the built panel, in axis order.

        Examples
        --------
        >>> ds.symbols
        ['AAA', 'BBB', 'CCC']
        """
        return self.data_backend.get_xarray_dataset(
            ["timestamp", "symbol"]
        ).symbol.values.tolist()

    @property
    def time_interval(self) -> np.timedelta64:
        """Return the most common spacing between consecutive timestamps.

        The most common spacing (the mode) is used rather than the minimum so
        that gaps such as weekends and holidays do not distort the answer.

        Examples
        --------
        >>> ds.time_interval  # daily bars built from raw files
        np.timedelta64(86400000000000,'ns')
        """
        timestamps = self.data_backend.get_xarray_dataset(["timestamp"])[
            "timestamp"
        ]
        return (
            timestamps.diff(dim="timestamp")
            .to_series()
            .mode()
            .values[0]
        )

    def _filter(self):
        """Narrow the built panel to the config's date range and symbols."""
        self.data_backend.filter_by_date(
            "timestamp", self.config.start_date, self.config.end_date
        )
        if self.config.symbols is not None:
            self.data_backend.filter_by_symbol("symbol", self.config.symbols)

    def _get_symbols(self) -> list[str]:
        """Return the symbol labels of the built panel."""
        return self.data_backend.get_xarray_dataset(
            ["symbol", "timestamp"]
        ).symbol.values.tolist()

    def save(self, **kwargs):
        """Narrow the built panel to the config's range and write it to Zarr.

        The write replaces the whole store directory at ``store_path``, the
        config's ``zarr_file_path``.

        Parameters
        ----------
        **kwargs
            Passed to ``XrBackend.create``, which lays the store out as
            ``widen_and_append`` does (sorted symbol axis, pinned chunk grid
            and fill values) so it can grow at its end.

        Examples
        --------
        >>> DemoDataset(config).from_raw_data().save()
        >>> Path(config.zarr_file_path).is_dir()
        True
        """
        with Timer(f"{self.__class__.__name__}: save"):
            self._filter()
            self.data_backend.create(
                self.store_path, fill_values=self._widen_fill_values(), **kwargs
            )

    def get_lazyframe(self) -> pl.LazyFrame:
        """Return the built panel as a long-format polars ``LazyFrame``.

        Long format means one row per ``(timestamp, symbol)`` pair, with one
        column per variable.

        Examples
        --------
        >>> ds.get_lazyframe().collect().shape  # 4 days x 3 symbols, 8 columns
        (12, 8)
        """
        return self.data_backend.get_lazyframe()

    def get_xarray_dataset(self) -> xr.Dataset:
        """Return the built panel indexed by ``(timestamp, symbol)``.

        Examples
        --------
        >>> tuple(ds.get_xarray_dataset().dims)
        ('timestamp', 'symbol')
        """
        return self.data_backend.get_xarray_dataset(["timestamp", "symbol"])

    def from_raw_data(self) -> Self:
        """Convert the raw files for the whole configured range, in memory.

        Runs ``_raw_data_to_xr``, then ``_clean``, and gives the result to
        the backend. Nothing is written to disk until ``save`` is called.
        Every call converts again; nothing is cached.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> ds = DemoDataset(config).from_raw_data()
        >>> list(ds.get_xarray_dataset().data_vars)  # cleaning adds the flag
        ['open', 'high', 'low', 'close', 'volume', 'anomaly_flag']
        """
        data = self._raw_data_to_xr()
        data = self._clean(data)
        self.data_backend.to_internal(data)  # type: ignore
        return self

    def _raw_axes_in_range(self) -> tuple[list, "pd.DatetimeIndex"]:
        """Return ``(pinned_symbols, observed_timestamps)`` for the whole range.

        The default converts the whole range with ``_raw_data_to_xr`` and
        reads both axes from the result. ``sort_symbol_axis`` sets the symbol
        order, and labels keep the raw panel's own type so that each window's
        ``reindex`` matches them. A subclass whose raw files can be read one
        column at a time should override this to avoid building the full
        panel.
        """
        data = self._raw_data_to_xr()
        symbols = sort_symbol_axis(data["symbol"].values.tolist())
        return symbols, pd.DatetimeIndex(data["timestamp"].values)

    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset:
        """Return the dense panel for one time window.

        The default converts the whole range and slices it, so it limits the
        size of each write but not the memory used to build the panel. When
        ``symbols`` is given, the result is reindexed onto exactly those
        symbols, with missing ones as all-NaN columns.
        """
        data = self._raw_data_to_xr()
        data = data.sel(timestamp=slice(start_date, end_date))
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def update(
        self,
        granularity: str = "year",
        ledger_path: str | None = None,
        append_dim: str = "timestamp",
    ) -> Self:
        """Bring the store up to date, choosing the new-listing strategy itself.

        This is ``from_raw_data_chunked`` with ``on_new_listing`` chosen from
        the data. When the raw tier has symbols the store lacks, the raw tier
        is checked for rows of those symbols inside the store's existing date
        range. If there are none, the symbols are genuinely new listings and
        the store is widened with NaN history. If there are some, the store is
        rebuilt so that their real history is included. If a symbol in the
        store has disappeared from the raw tier, the run refuses. The choice
        is logged before it runs.

        Parameters
        ----------
        granularity : str, default "year"
            Window size, passed to ``TimeChunkPlanner``.
        ledger_path : str or None, default None
            Ledger file path; ``None`` uses ``<store>.chunks.json``.
        append_dim : str, default "timestamp"
            Dimension windows are appended along.

        Returns
        -------
        Self
            ``self``, with ``last_chunk_result`` populated.

        Examples
        --------
        >>> ds = DemoDataset(config).update(granularity="month")
        >>> ds.last_chunk_result.windows_written
        1
        >>> again = DemoDataset(config).update(granularity="month")
        >>> again.last_chunk_result.windows_skipped, again.last_chunk_result.resumed
        (1, True)
        """
        return self.from_raw_data_chunked(
            granularity=granularity,
            ledger_path=ledger_path,
            append_dim=append_dim,
            on_new_listing=self._AUTOMATIC,
        )

    def from_raw_data_chunked(
        self,
        granularity: str = "year",
        ledger_path: str | None = None,
        append_dim: str = "timestamp",
        on_new_listing: str | object = "refuse",
        *,
        reporter: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> Self:
        """Convert the raw files one time window at a time, appending to Zarr.

        The *pinned symbol axis*, the list of symbols for the whole date
        range, is worked out once before any window, and every window is
        built on exactly that axis so that all windows line up column by
        column. Completed windows are recorded in a ledger file, so a
        crashed or cancelled run resumes by skipping them. Everything up to
        the ledger's last end counts as written, so a window that straddles
        it is cut to its new bars: a daily run extends a store whose last
        window is still growing, at any granularity. Every window appended
        to a store that existed before the run passes through
        ``_continue_store`` first. Cleaning runs per
        window, so a price jump that spans two windows is not flagged.

        When the store already exists with a different symbol axis,
        ``on_new_listing`` decides what happens. ``"refuse"`` stops.
        ``"widen"`` adds the new symbols with NaN over the store's existing
        history. ``"rebuild"`` moves the store aside and rebuilds every window
        from the raw files; if the rebuild fails or is cancelled, the
        original store is put back.

        Parameters
        ----------
        granularity : str, default "year"
            Window size, passed to ``TimeChunkPlanner`` (``"year"``,
            ``"quarter"``, ``"month"``, ``"day"`` or ``"hour"``).
        ledger_path : str or None, default None
            Ledger file path; ``None`` uses ``<store>.chunks.json``.
        append_dim : str, default "timestamp"
            Dimension windows are appended along.
        on_new_listing : str, default "refuse"
            One of ``NEW_LISTING_STRATEGIES``. ``update()`` passes a private
            marker here instead, to have the strategy chosen automatically.
        reporter : ProgressReporter or None, default None
            Receives an event for each window.
        cancel : CancelToken or None, default None
            Checked before each window; when set, the run stops there.

        Returns
        -------
        Self
            ``self``, with ``last_chunk_result`` describing the run.

        Raises
        ------
        ValueError
            If ``on_new_listing`` is unknown, if a window comes back on a
            symbol axis other than the pinned one, or if the ledger and the
            store disagree (see ``ChunkLedger.assert_consistent``).

        Examples
        --------
        >>> ds = DemoDataset(config).from_raw_data_chunked(granularity="month")
        >>> ds.last_chunk_result.windows_written
        1
        """
        from quantlab.dataset._support.ledger import ChunkLedger, TimeChunkPlanner

        # The private marker is accepted by identity; the error lists only the
        # public strategies.
        if (
            on_new_listing is not self._AUTOMATIC
            and on_new_listing not in self.NEW_LISTING_STRATEGIES
        ):
            raise ValueError(
                f"{self.class_name}: unknown on_new_listing strategy "
                f"{on_new_listing!r}; accepted values are "
                f"{list(self.NEW_LISTING_STRATEGIES)}."
            )

        symbols, timestamps = self._raw_axes_in_range()
        symbols = self._pinned_symbol_axis(symbols)
        planner = TimeChunkPlanner(granularity)
        windows = planner.plan_from_timestamps(timestamps)
        resolved_ledger_path = ledger_path or ChunkLedger.default_path(
            self.config.zarr_file_path
        )
        ledger = ChunkLedger(resolved_ledger_path, append_dim=append_dim)

        # Handle a changed symbol axis before the ledger check, because widen
        # and rebuild both change what that check looks at. The full timestamp
        # count is passed so a widen sizes the store's Zarr chunks for the
        # whole range, not just for what is stored so far.
        ledger, rebuild_asides = self._reconcile_new_listings(
            symbols,
            ledger,
            append_dim,
            on_new_listing,
            append_dim_size=len(timestamps),
        )
        if rebuild_asides is not None:
            # A rebuild starts a new store, which is sorted (ADR 0032).
            symbols = sort_symbol_axis(symbols)
        # Set before the `try` so the result below can always read it.
        rebuild_rolled_back: bool = False

        try:
            # The ledger and the store record the same history; check they agree
            # before the first append, which cannot be undone.
            ledger.assert_consistent(symbols, self.config.zarr_file_path)
            # Windows are appended in time order, so the ledger covers every
            # timestamp up to its last end. A window there is written even
            # when its edges differ from the recorded ones: the last window
            # of a store grows with every new bar, and a store may be updated
            # at another granularity than it was built at.
            covered = (
                pd.Timestamp(ledger.last_end)
                if ledger.last_end is not None
                else None
            )
            # A store that existed before this run is continued, not built.
            continuing = Path(str(self.config.zarr_file_path)).exists()

            logger.info(
                f"{self.class_name}: chunked ingestion over {len(windows)} "
                f"{granularity} window(s), {len(symbols)} pinned symbol(s), "
                f"{len(timestamps)} observed timestamp(s)."
            )

            first_timestamp = timestamps.min() if len(timestamps) else None
            boundaries = 0
            # Counted inside the loop rather than read back from the ledger,
            # which also records earlier runs' windows.
            windows_written = 0
            windows_skipped = 0
            rows_written = 0
            cancelled = False
            peak_window_bytes: int | None = None
            self._emit_progress(
                reporter,
                ProgressEvent(
                    kind="conversion_started",
                    vendor=self._progress_vendor,
                    total=len(windows),
                    message=(
                        f"{self.class_name} {granularity} conversion "
                        f"to {self.config.zarr_file_path}"
                    ),
                    detail={
                        "pinned_symbols": len(symbols),
                        "granularity": granularity,
                        "zarr_path": self.config.zarr_file_path,
                    },
                ),
            )
            for start, end in windows:
                # Check for cancellation before building the next window.
                if cancel is not None and cancel.is_cancelled():
                    cancelled = True
                    resumability = (
                        "This run is a rebuild, so the windows written so far "
                        "will be discarded and the pre-rebuild store restored; "
                        "a later rebuild starts over."
                        if rebuild_asides is not None
                        else "Every recorded window stays resumable."
                    )
                    logger.warning(
                        f"{self.class_name}: cancel observed at the "
                        f"{start.date()}..{end.date()} window boundary; "
                        f"stopping with {windows_written} window(s) written "
                        f"this run. {resumability}"
                    )
                    self._emit_progress(
                        reporter,
                        ProgressEvent(
                            kind="cancelled",
                            vendor=self._progress_vendor,
                            completed=windows_written + windows_skipped,
                            total=len(windows),
                            message=(
                                f"cancelled at {start.date()}..{end.date()}"
                            ),
                            detail={"granularity": granularity},
                        ),
                    )
                    break

                if ledger.is_written(start, end) or (
                    covered is not None and end <= covered
                ):
                    windows_skipped += 1
                    logger.info(
                        f"{self.class_name}: window {start.date()}..{end.date()} "
                        f"already recorded in the ledger, skipping."
                    )
                    self._emit_progress(
                        reporter,
                        ProgressEvent(
                            kind="window_skipped",
                            vendor=self._progress_vendor,
                            completed=windows_written + windows_skipped,
                            total=len(windows),
                            message=(
                                f"skipped {start.date()}..{end.date()} "
                                f"(already in the ledger)"
                            ),
                            detail={
                                "start": str(start.date()),
                                "end": str(end.date()),
                            },
                        ),
                    )
                    continue

                if covered is not None and start <= covered:
                    # The window straddles the ledger's end: only its new
                    # bars are appended.
                    start = timestamps[timestamps > covered].min()
                window = self._raw_data_to_xr_window(start, end, symbols)
                # Compare in the axis's own type, not as text: integer labels
                # would never equal their string forms.
                actual = list(window["symbol"].values.tolist())
                if actual != list(symbols):
                    raise ValueError(
                        f"{self.class_name}: window {start.date()}..{end.date()} "
                        f"came back on a symbol axis of {len(actual)} label(s), "
                        f"but the pinned whole-range axis has {len(symbols)}. "
                        f"Every window must be built on the pinned axis; "
                        f"appending this one would silently misalign every "
                        f"column in the store."
                    )

                if continuing:
                    window = self._continue_store(window)
                window = self._clean(window)
                window = self._pin_append_dtypes(window)
                # Measure after cleaning and dtype promotion: that is what the
                # append actually holds in memory.
                window_bytes = int(window.nbytes)
                if (
                    peak_window_bytes is None
                    or window_bytes > peak_window_bytes
                ):
                    peak_window_bytes = window_bytes
                if start != first_timestamp:
                    boundaries += 1

                self.data_backend.to_internal(window)
                # widen_and_append adds any missing symbols or variables to the
                # store, then appends; if nothing is missing it is a plain
                # append. append_dim_size is passed every time because any
                # iteration may be the one that creates the store (a resume
                # skips windows, a rebuild moves the store aside); the backend
                # only uses it when creating the store.
                self.data_backend.widen_and_append(
                    self.config.zarr_file_path,
                    append_dim=append_dim,
                    fill_values=self._widen_fill_values(),
                    append_dim_size=len(timestamps),
                )
                ledger.record(
                    start, end, int(window.sizes[append_dim]), symbols
                )
                windows_written += 1
                rows_written += int(window.sizes[append_dim])
                logger.info(
                    f"{self.class_name}: appended window "
                    f"{start.date()}..{end.date()} "
                    f"({int(window.sizes[append_dim])} row(s))."
                )
                self._emit_progress(
                    reporter,
                    ProgressEvent(
                        kind="window_written",
                        vendor=self._progress_vendor,
                        completed=windows_written + windows_skipped,
                        total=len(windows),
                        message=f"appended {start.date()}..{end.date()}",
                        detail={
                            "start": str(start.date()),
                            "end": str(end.date()),
                            "rows": int(window.sizes[append_dim]),
                            "window_bytes": window_bytes,
                        },
                    ),
                )

            # Emitted whether the loop finished or was cancelled, so a reporter
            # can always release what it opened.
            self._emit_progress(
                reporter,
                ProgressEvent(
                    kind="conversion_finished",
                    vendor=self._progress_vendor,
                    completed=windows_written + windows_skipped,
                    total=len(windows),
                    message=(
                        f"{self.class_name}: {windows_written} written, "
                        f"{windows_skipped} skipped, "
                        f"{len(windows)} planned"
                    ),
                    detail={
                        "windows_written": windows_written,
                        "windows_skipped": windows_skipped,
                        "windows_planned": len(windows),
                        "rows_written": rows_written,
                        "cancelled": cancelled,
                    },
                ),
            )

            if boundaries:
                logger.warning(
                    f"{self.class_name}: cleaning ran per window, so at "
                    f"{boundaries} window-boundary timestamp(s) `flag_anomalies` "
                    f"had no previous value to compare with, and a one-step jump "
                    f"across that boundary is not flagged. This is an expected "
                    f"side effect of chunking; a finer --chunk granularity "
                    f"creates more such boundaries, not fewer."
                )
        except BaseException:
            # The originals were renamed, not deleted, so a failed rebuild is
            # undone by renaming them back.
            if rebuild_asides is not None:
                self._restore_rebuild_asides(rebuild_asides)
            raise
        else:
            if rebuild_asides is not None:
                if cancelled:
                    # A cancelled rebuild has rebuilt only some windows, so the
                    # set-aside copies are still the only complete data. Roll
                    # back rather than leave a truncated store where readers
                    # look. The flag records whether the restore succeeded.
                    rebuild_rolled_back = self._restore_rebuild_asides(
                        rebuild_asides, reason="cancelled"
                    )
                else:
                    self._discard_rebuild_asides(rebuild_asides)

        # Only reached on success; the `except` branch re-raises and must not
        # leave a result describing a partial run.
        self.last_chunk_result = ConversionResult(
            zarr_path=self.config.zarr_file_path,
            ledger_path=resolved_ledger_path,
            granularity=granularity,
            pinned_symbols=len(symbols),
            windows_planned=len(windows),
            windows_written=windows_written,
            windows_skipped=windows_skipped,
            rows_written=rows_written,
            peak_window_bytes=peak_window_bytes,
            resumed=windows_skipped > 0,
            cancelled=cancelled,
            rebuild_rolled_back=rebuild_rolled_back,
        )
        return self

    #: Suffix added to the store and ledger paths while a rebuild is running
    #: and the originals are set aside. It reuses ``XrBackend.SUPERSEDED_SUFFIX``
    #: so the backend, which refuses to widen while a set-aside copy exists,
    #: always recognises one left behind by a rebuild that was killed.
    SUPERSEDED_SUFFIX = XrBackend.SUPERSEDED_SUFFIX

    def _pinned_symbol_axis(self, symbols: list) -> list:
        """Return the axis a write pins: the stored one, then the new symbols.

        A store's symbol axis grows at its end: when every stored
        symbol is among ``symbols`` (the raw axis, sorted), the pinned axis
        is the stored axis followed by the symbols it lacks, sorted among
        themselves, so a new listing extends the store in place and a run
        with no new symbol never reorders it. With no store, or with a
        stored symbol missing from ``symbols`` (a case
        ``_reconcile_new_listings`` refuses), ``symbols`` is returned as it
        is.

        Parameters
        ----------
        symbols : list
            The raw tier's symbols over the whole range, in
            ``sort_symbol_axis`` order.

        Returns
        -------
        list
            The axis every window of this run is built on.
        """
        stored = self._stored_symbol_axis(self.config.zarr_file_path)
        if stored is None:
            return symbols
        known = set(stored)
        if not known.issubset(symbols):
            return symbols
        return stored + [symbol for symbol in symbols if symbol not in known]

    @staticmethod
    def _stored_append_extent(
        store_path: str, append_dim: str = "timestamp"
    ) -> Optional[tuple]:
        """Return the store's first and last ``append_dim`` labels as a pair.

        Returns ``None`` when there is no store, no such coordinate, or the
        axis is empty.
        """
        if not Path(store_path).exists():
            return None
        store = xr.open_zarr(store_path)
        try:
            if append_dim not in store.coords:
                return None
            values = store[append_dim].values
            if not len(values):
                return None
            return (values[0], values[-1])
        finally:
            store.close()

    def _added_symbols_with_raw_history(
        self, added: list, start, end
    ) -> dict[str, int]:
        """Count the raw rows each symbol in ``added`` has between ``start`` and ``end``.

        ``update`` uses these counts to choose between widening and
        rebuilding. The default builds the whole window and counts, which is
        correct but uses memory for every symbol; a subclass whose raw files
        can be filtered by symbol while reading should override it.

        Symbols are compared as text because the raw tier stores symbols as
        strings even when the pinned axis uses integers. Without the
        conversion no added symbol would ever match, and the resolver would
        silently choose ``widen``.

        Parameters
        ----------
        added : list
            Symbols present in the raw tier but not in the store.
        start, end : timestamp-like
            The date range to check, both ends included.

        Returns
        -------
        dict[str, int]
            ``{symbol: row count}`` for the symbols that have at least one row.
        """
        wanted = [str(symbol) for symbol in added]
        if not wanted:
            return {}

        if (
            type(self)._added_symbols_with_raw_history
            is RawDataset._added_symbols_with_raw_history
        ):
            logger.warning(
                f"{self.class_name}: _added_symbols_with_raw_history has not "
                f"been overridden, so the raw-history probe is answered by "
                f"building the whole window and counting. The answer is "
                f"correct but memory is not bounded; override this method for "
                f"a source that can filter by symbol while reading."
            )

        window = self._raw_data_to_xr_window(start, end, symbols=None)
        if "symbol" not in window.coords:
            return {}
        # Compared as text, selected by the window's own label (an int axis
        # is not indexable by the text).
        present = {str(label): label for label in window["symbol"].values.tolist()}

        counts: dict[str, int] = {}
        for symbol in wanted:
            if symbol not in present:
                continue
            column = window.sel(symbol=present[symbol])
            observed = None
            for variable in column.data_vars.values():
                notnull = variable.notnull()
                observed = notnull if observed is None else (observed | notnull)
            if observed is None:
                continue
            rows = int(np.count_nonzero(observed.values))
            if rows:
                counts[symbol] = rows
        return counts

    def _widen_fill_values(self) -> dict:
        """Return per-variable fill values used when widening the symbol axis.

        Non-float variables need an explicit fill value because NaN is not
        available for them. The default covers the boolean ``anomaly_flag``
        that cleaning adds.
        """
        return {"anomaly_flag": False}

    def _resolve_new_listing_strategy(
        self,
        added: list,
        removed: list,
        store_path: str,
        append_dim: str,
    ) -> str:
        """Choose a new-listing strategy from raw-tier evidence and log why.

        If any stored symbol was removed, the answer is ``"refuse"``, since
        neither widening nor rebuilding keeps its history safely. Otherwise
        the raw tier is checked for rows of the added symbols inside the
        store's existing date range. No rows means they are new listings, so
        ``"widen"``. Any rows means widening would replace real history with
        NaN, so ``"rebuild"``.

        Returns
        -------
        str
            ``"refuse"``, ``"widen"`` or ``"rebuild"``.
        """
        if removed:
            logger.warning(
                f"{self.class_name}: {len(removed)} symbol(s) present in the "
                f"store at {store_path} are absent from the pinned "
                f"whole-range axis ({sorted(removed)[: self.NEW_LISTING_REPORT_LIMIT]}"
                f"{', truncated' if len(removed) > self.NEW_LISTING_REPORT_LIMIT else ''}), "
                f"so this resolves to 'refuse'. Neither other strategy is safe "
                f"for a dropped symbol: 'widen' cannot remove one "
                f"(widen_symbol_axis refuses a target axis that is not a "
                f"superset of the stored one), and 'rebuild' would silently "
                f"discard that symbol's stored history. Choosing between losing "
                f"history and stopping is your call: re-run "
                f"from_raw_data_chunked() with an explicit on_new_listing once "
                f"you know which you want."
            )
            return "refuse"

        extent = self._stored_append_extent(store_path, append_dim)
        if extent is None:
            logger.info(
                f"{self.class_name}: the store at {store_path} has no "
                f"{append_dim} extent to lose (absent, or a zero-length axis), "
                f"so there is no history a NaN backfill could destroy; "
                f"resolving to 'widen' without asking raw anything."
            )
            return "widen"

        start, end = extent
        evidence = self._added_symbols_with_raw_history(added, start, end)
        if not evidence:
            logger.info(
                f"{self.class_name}: the raw tier was asked about all "
                f"{len(added)} added symbol(s) over the store's own date range "
                f"({start}..{end}) and reported no rows there for any of them. "
                f"They are genuine new listings, NaN is the correct value over "
                f"the store's history, and a rebuild would gain nothing; "
                f"resolving to 'widen'."
            )
            return "widen"

        ranked = sorted(evidence.items(), key=lambda item: (-item[1], item[0]))
        shown = ranked[: self.NEW_LISTING_REPORT_LIMIT]
        listing = ", ".join(f"{symbol}={rows}" for symbol, rows in shown)
        truncation = (
            f" (top {len(shown)} of {len(ranked)} by raw row count; the rest "
            f"are not listed)"
            if len(ranked) > len(shown)
            else ""
        )
        logger.warning(
            f"{self.class_name}: {len(ranked)} of {len(added)} added symbol(s) "
            f"already have raw rows inside the store's own date range "
            f"({start}..{end}), so this resolves to 'rebuild' rather than "
            f"'widen': a widen does not re-read the raw files, so it would "
            f"replace that history with NaN and nothing would report it. "
            f"Qualifying symbol(s) by raw row count: {listing}{truncation}. "
            f"A rebuild rewrites the whole store, not just these symbols, so "
            f"this run rebuilds every window."
        )
        return "rebuild"

    def _reconcile_new_listings(
        self,
        symbols: list,
        ledger,
        append_dim: str,
        on_new_listing: str,
        *,
        append_dim_size: Optional[int] = None,
    ) -> tuple:
        """Apply ``on_new_listing`` when the store's symbols differ from ``symbols``.

        Returns
        -------
        tuple
            ``(ledger, rebuild_asides)``. ``ledger`` is the ledger to use from
            now on. ``rebuild_asides`` is ``None`` unless a rebuild started,
            in which case it is a dict naming the original and set-aside
            paths of the store and ledger, so they can later be restored or
            deleted.
        """
        store_path = self.config.zarr_file_path
        stored = self._stored_symbol_axis(store_path)
        if stored is None or stored == list(symbols):
            return ledger, None

        # The sets are built once: a store holds tens of thousands of symbols.
        stored_set, symbol_set = set(stored), set(symbols)
        added = [symbol for symbol in symbols if symbol not in stored_set]
        removed = [symbol for symbol in stored if symbol not in symbol_set]

        # Resolved only after the early return above, so the raw tier is not
        # queried when the axes already agree.
        if on_new_listing is self._AUTOMATIC:
            on_new_listing = self._resolve_new_listing_strategy(
                added, removed, store_path, append_dim
            )

        if on_new_listing == "refuse":
            # Leave everything unchanged; `assert_consistent` raises next.
            logger.info(
                f"{self.class_name}: the store at {store_path} holds "
                f"{len(stored)} symbol(s) but the pinned whole-range axis has "
                f"{len(symbols)} ({len(added)} added, {len(removed)} removed). "
                f"on_new_listing='refuse' (the default), so this run will "
                f"halt. Pass --on-new-listing rebuild to re-densify every "
                f"window from raw, or --on-new-listing widen to keep the store "
                f"and backfill the new listing(s) with NaN."
            )
            return ledger, None

        if on_new_listing == "widen":
            logger.warning(
                f"{self.class_name}: widening {store_path} from {len(stored)} "
                f"to {len(symbols)} symbol(s); added={added}, "
                f"removed={removed}. The added symbol(s) will be NaN for all "
                f"existing history: a widen does not re-read the raw files, so "
                f"history the vendor already has is not recovered. Use "
                f"on_new_listing='rebuild' for that. The added symbol(s) are "
                f"appended at the end of the axis in place, rewriting no "
                f"stored chunk (XrBackend.extend_symbol_axis)."
            )
            if removed:
                # Refuses, naming the symbols it would drop.
                self.data_backend.widen_symbol_axis(
                    store_path,
                    list(symbols),
                    append_dim=append_dim,
                    fill_values=self._widen_fill_values(),
                    append_dim_size=append_dim_size,
                )
            self.data_backend.extend_symbol_axis(
                store_path, added, fill_values=self._widen_fill_values()
            )
            # The ledger stores a fingerprint of the symbol list, so update it
            # now or the widened store cannot be resumed.
            ledger.rebase(symbols)
            return ledger, None

        # Remaining case: on_new_listing == "rebuild".
        from quantlab.dataset._support.ledger import ChunkLedger

        logger.warning(
            f"{self.class_name}: rebuilding {store_path}: every window will "
            f"be rebuilt from the raw files on the new {len(symbols)}-symbol "
            f"axis (was {len(stored)}); added={added}, removed={removed}. "
            f"This recovers the added symbol(s)' real history instead of "
            f"filling it with NaN, at the cost of a full rebuild."
        )
        asides = {
            "store": store_path,
            "store_aside": f"{store_path}{self.SUPERSEDED_SUFFIX}",
            "ledger": ledger.path,
            "ledger_aside": f"{ledger.path}{self.SUPERSEDED_SUFFIX}",
            "ledger_existed": Path(ledger.path).exists(),
        }
        # Rename rather than delete, so a failed rebuild can be undone.
        os.replace(asides["store"], asides["store_aside"])
        if asides["ledger_existed"]:
            os.replace(asides["ledger"], asides["ledger_aside"])
        # With a fresh ledger at the original path, the rebuild proceeds like
        # an ordinary first run.
        return ChunkLedger(asides["ledger"], append_dim=append_dim), asides

    def _restore_rebuild_asides(
        self, asides: dict, *, reason: str = "failed"
    ) -> bool:
        """Put the pre-rebuild store and ledger back, deleting the partial rebuild.

        Parameters
        ----------
        asides : dict
            The paths returned by ``_reconcile_new_listings``.
        reason : str, default "failed"
            Word used in the log message (``"failed"`` or ``"cancelled"``).

        Returns
        -------
        bool
            ``True`` if the originals were restored. ``False`` if the
            filesystem refused; then nothing is deleted, and the error log
            names both copies so the user can recover by hand. The error is
            logged rather than raised so it cannot hide the failure that
            stopped the rebuild.
        """
        try:
            if Path(asides["store"]).exists():
                shutil.rmtree(asides["store"], ignore_errors=True)
            os.replace(asides["store_aside"], asides["store"])
            Path(asides["ledger"]).unlink(missing_ok=True)
            if asides["ledger_existed"]:
                os.replace(asides["ledger_aside"], asides["ledger"])
            logger.warning(
                f"{self.class_name}: the rebuild of {asides['store']} was "
                f"{reason}; the pre-rebuild store and ledger have been "
                f"restored. Any window this run rebuilt has been discarded "
                f"with the partial store, so a later rebuild starts over."
            )
            return True
        except OSError as exc:
            logger.error(
                f"{self.class_name}: the rebuild of {asides['store']} was "
                f"{reason}, but the pre-rebuild copy could not be put back: "
                f"{type(exc).__name__}: {exc}. Nothing was deleted: the "
                f"complete pre-rebuild copy is still at "
                f"{asides['store_aside']}, and the partial rebuild is at "
                f"{asides['store']}. Recover by hand: once you have dealt "
                f"with whatever refused the rename, move "
                f"{asides['store_aside']} back over {asides['store']}. This "
                f"is reported rather than raised so a cleanup error cannot "
                f"replace the failure that stopped the rebuild."
            )
            return False

    @staticmethod
    def _discard_rebuild_asides(asides: dict) -> None:
        """Delete the set-aside store and ledger after a successful rebuild."""
        shutil.rmtree(asides["store_aside"], ignore_errors=True)
        Path(asides["ledger_aside"]).unlink(missing_ok=True)

    def _continue_store(self, window: xr.Dataset) -> xr.Dataset:
        """Return a window about to be appended to a store that existed before this run.

        ``from_raw_data_chunked`` (and so ``update``) calls this for every
        window it appends after the store's existing bars, before cleaning;
        a fresh build and a rebuild never call it. The default returns the
        window unchanged. A dataset whose stored values depend on earlier
        bars (an adjusted price chained from an anchor) overrides it to
        continue from what is stored, or to report what the vendor has
        changed in the stored bars.

        Parameters
        ----------
        window : xr.Dataset
            The window on the pinned symbol axis, all of it after the
            store's last timestamp.

        Returns
        -------
        xr.Dataset
            The window to clean and append, on the same axes.
        """
        return window

    @staticmethod
    def _pin_append_dtypes(data: xr.Dataset) -> xr.Dataset:
        """Promote integer data variables to float64 before an append.

        The first window written fixes each variable's dtype in the store. An
        integer variable cannot hold NaN, so the missing cells of later
        windows would be silently cast to integers; storing float64 keeps
        them as NaN.
        """
        promoted = {
            name: variable.astype("float64")
            for name, variable in data.data_vars.items()
            if np.issubdtype(variable.dtype, np.integer)
        }
        return data.assign(**promoted) if promoted else data

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Validate and flag the converted panel before it is stored.

        The default runs ``clean_market_data``, which checks for the OHLCV
        columns (open, high, low, close, volume) and adds a boolean
        ``anomaly_flag`` variable marking suspicious jumps. Override it for
        panels whose columns are not lowercase OHLCV, or that are not market
        data at all.
        """
        return clean_market_data(data)

    @abstractmethod
    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the raw files as a dense ``(timestamp, symbol)`` panel.

        The result must cover the config's whole date range, and each
        ``(timestamp, symbol)`` pair must appear once; remove duplicates
        before converting a table with ``to_xarray``. The caller applies
        cleaning, so do not clean here.
        """


class RawMarketDataset(RawDataset, MarketDataset):
    """Market price bars converted from raw vendor files into a store they own.

    ``RawDataset``'s build path with ``MarketDataset``'s market rules and
    KunQuant export. A subclass implements ``_raw_data_to_xr``,
    ``_raw_data_to_xr_window`` and ``_to_kunquant``.

    Parameters
    ----------
    config : DatasetConfig
        The dataset config.

    Examples
    --------
    Using the ``DemoDataset`` described on ``BaseDataset``:

    >>> DemoDataset(config).from_raw_data().save()
    >>> DemoDataset(config).panel("2024-01-02", "2024-01-05").sizes["timestamp"]
    4
    """

    @abstractmethod
    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset:
        """Return the dense panel for one time window, on ``symbols`` if given.

        Declared abstract again here so that every market dataset must say
        explicitly how it builds one window for chunked conversion, rather
        than inheriting the slow default.
        """


class IndexConstituentDataset(RawDataset):
    """Daily point-in-time index-membership panel built from intervals.

    A subclass implements two hooks: ``_pit_coverage_start`` (the earliest
    date the source can answer membership for) and ``_build_intervals`` (the
    membership intervals); it may override a third, ``_pit_coverage_end``
    (the last date it has observed). Everything else is shared. Config normalisation
    raises ``start_date`` to the coverage start, ``_densify`` turns the
    intervals into the ``is_member`` grid, and ``_clean`` validates the panel
    instead of running the price-data (OHLCV) cleaner. The class derives
    from ``BaseDataset`` directly, so it has no KunQuant export.

    Two conventions matter to a consumer. The ``timestamp`` axis covers every
    calendar day, weekends and holidays included, so to combine it with a
    trading-day price panel, select it onto the price timestamps rather than
    assuming the axes line up. Intervals include both end dates: a symbol
    removed on date ``D`` reads ``True`` on ``D`` and ``False`` on ``D + 1``.

    Parameters
    ----------
    config : ConstituentDatasetConfig
        The dataset config. ``as_of`` pins the right edge of the panel when
        some membership is still open; see ``_densify``.

    Examples
    --------
    A minimal index over an in-memory interval table::

        class DemoPanel(IndexConstituentDataset):
            def _pit_coverage_start(self, config) -> str:
                return "2020-01-01"

            def _build_intervals(self) -> pl.DataFrame:
                return pl.DataFrame(
                    [
                        ("AAA", "2020-01-01", None),
                        ("BBB", "2020-01-01", "2020-01-05"),
                        ("CCC", "2020-01-04", None),
                    ],
                    schema=["symbol", "start_date", "end_date"],
                    orient="row",
                )

    >>> config = ConstituentDatasetConfig(
    ...     zarr_file_path="data/demo.zarr",
    ...     start_date="2020-01-01",
    ...     end_date="2020-01-08",
    ...     as_of="2020-01-08",
    ... )
    >>> panel = DemoPanel(config).from_raw_data().get_xarray_dataset()
    >>> panel["is_member"].to_pandas().astype(int)
    symbol      AAA  BBB  CCC
    timestamp
    2020-01-01    1    1    0
    2020-01-02    1    1    0
    2020-01-03    1    1    0
    2020-01-04    1    1    1
    2020-01-05    1    1    1
    2020-01-06    1    0    1
    2020-01-07    1    0    1
    2020-01-08    1    0    1
    """

    #: The config class a saved ``config.json`` is rebuilt with.
    config_cls = ConstituentDatasetConfig

    def _normalize_config(
        self, config: ConstituentDatasetConfig
    ) -> ConstituentDatasetConfig:
        """Normalise as the base class does, then raise ``start_date`` to the coverage start.

        The source cannot answer membership before ``_pit_coverage_start(config)``.
        Without this adjustment, the inherited default start date
        (``Date.START_DATE``) would add decades of all-False rows, which read
        as "not a member" when the truth is "unknown". A warning is logged
        only when the caller explicitly asked for an earlier date; the
        default is adjusted silently so that ordinary construction does not
        warn. The base normalisation runs first because the adjustment reads
        the resolved ``start_date``.

        Parameters
        ----------
        config : ConstituentDatasetConfig
            The config to normalise. It is not modified.

        Returns
        -------
        ConstituentDatasetConfig
            A new config whose ``start_date`` is not before the coverage
            start.

        Raises
        ------
        TypeError
            If ``config`` is not exactly the class's ``config_cls``: a source
            read from files given a config with no ``cache_dir``, or a
            membership derived from another store given one.

        Examples
        --------
        A requested date before the coverage start is raised, with a
        warning, on the dataset's own config:

        >>> ds = DemoPanel(dataclasses.replace(config, start_date="2019-06-01"))
        >>> ds.config.start_date
        '2020-01-01'
        """
        config = super()._normalize_config(config)
        # The exact class: a subclass config carries fields this class never reads, which its
        # saved config.json then cannot be rebuilt with.
        if type(config) is not self.config_cls:
            raise TypeError(
                f"{self.class_name} needs a {self.config_cls.__name__}, got "
                f"{type(config).__name__}."
            )
        requested = config.start_date
        # The config's class is checked first, so a refused config reads no store.
        coverage_start = self._pit_coverage_start(config)
        if requested >= coverage_start:
            return config

        if requested != Date.START_DATE:
            logger.warning(
                f"{self.class_name}: requested start_date {requested} is "
                f"before this index's point-in-time coverage start "
                f"{coverage_start}; the panel's left edge was clamped to "
                f"{coverage_start}. Membership before that date cannot be "
                f"answered from the source."
            )
        return dataclasses.replace(config, start_date=coverage_start)

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Validate the membership panel instead of running the OHLCV cleaner."""
        return clean_membership_panel(data)

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Build the interval table and densify it into the daily panel."""
        return self._densify(self._build_intervals())

    def _densify(self, intervals: pl.DataFrame) -> xr.Dataset:
        """Densify a membership-interval table into a daily boolean panel.

        The symbol axis holds every symbol that appears anywhere in the
        table, sorted, and is computed before any date filtering. So a symbol
        whose membership ended before the window still gets an all-False
        column, and selecting a symbol that was never a member raises. Labels
        keep the table's own type: ticker strings give a string axis in
        alphabetical order, and integer identifiers (such as CRSP PERMNOs,
        permanent security numbers) give an int64 axis in numeric order.
        ``sort_symbol_axis`` decides the order.

        The first date is ``max(config.start_date, coverage start)``. The
        last date is ``min(config.end_date, horizon)``. ``horizon`` is the
        coverage end (``_pit_coverage_end``) when the source gives one, so
        dates on which no symbol is a member stay on the panel; otherwise it
        is the latest date in the table, and if any interval is still open it
        is extended to ``config.as_of``, or to today when ``as_of`` is unset.
        Set ``as_of`` when the panel must be reproducible. Cutting back an
        explicitly requested ``end_date`` logs a warning; the inherited
        default (``Date.END_DATE``) is cut back silently. Each interval
        includes both of its end dates, and a null ``end_date`` runs to the
        last date of the panel.

        Parameters
        ----------
        intervals : pl.DataFrame
            A frame with ``symbol``, ``start_date`` and ``end_date`` columns.
            A null ``end_date`` means the symbol is still a member.

        Returns
        -------
        xr.Dataset
            A dataset with one boolean ``is_member`` variable on
            ``(timestamp, symbol)``, with one ``timestamp`` per calendar day.

        Raises
        ------
        ValueError
            If the table is empty without a coverage end, any row has a null
            ``start_date``, or the resolved date range is empty.
        """
        rows = intervals.select(
            ["symbol", "start_date", "end_date"]
        ).to_dicts()
        coverage_end = self._pit_coverage_end(self.config)
        if not rows and coverage_end is None:
            raise ValueError(
                f"{self.class_name}: the membership-interval table is empty; "
                f"there is no membership history to densify."
            )

        # Reject null start dates: pd.Timestamp(None) is NaT, every comparison
        # with NaT is False, so max() below would return an arbitrary value and
        # the fill loop would produce an all-False column. str() is only for
        # the error message; the axis keeps the table's type.
        undated = sorted(
            {str(row["symbol"]) for row in rows if row["start_date"] is None}
        )
        if undated:
            raise ValueError(
                f"{self.class_name}: interval rows with a null start_date "
                f"cannot be densified: {undated}. A membership with no start "
                f"date is not a membership -- silently treating it as one "
                f"corrupts the panel's horizon and yields an all-False column "
                f"indistinguishable from 'never a member'."
            )

        # sort_symbol_axis orders integer identifiers numerically even when
        # they arrive as digit strings; plain sorted() would put "14593"
        # before "7000".
        symbols = sort_symbol_axis({row["symbol"] for row in rows})
        column_of = {symbol: i for i, symbol in enumerate(symbols)}

        left = pd.Timestamp(
            max(self.config.start_date, self._pit_coverage_start(self.config))
        )

        if coverage_end is not None:
            horizon = pd.Timestamp(coverage_end)
        else:
            observed = max(
                pd.Timestamp(row["start_date"]) for row in rows
            )
            closed_ends = [
                pd.Timestamp(row["end_date"])
                for row in rows
                if row["end_date"] is not None
            ]
            if closed_ends:
                observed = max(observed, max(closed_ends))
            has_open_membership = any(row["end_date"] is None for row in rows)
            if has_open_membership:
                # An open membership is still current, so the last recorded change
                # is only a lower bound on the last date. config.as_of fixes it;
                # without it today's date is used, which is not reproducible.
                today = (
                    pd.Timestamp(self.config.as_of)
                    if self.config.as_of is not None
                    else pd.Timestamp.today().normalize()
                )
                horizon = max(observed, today)
            else:
                horizon = observed

        requested_right = pd.Timestamp(self.config.end_date)
        right = min(requested_right, horizon)
        if right < requested_right and self.config.end_date != Date.END_DATE:
            logger.warning(
                f"{self.class_name}: requested end_date "
                f"{self.config.end_date} is beyond what the source supports; "
                f"the panel's right edge was truncated to "
                f"{right.strftime('%Y-%m-%d')}. Membership past that date "
                f"would be fabricated, not observed."
            )

        if right < left:
            raise ValueError(
                f"{self.class_name}: empty membership window -- resolved left "
                f"edge {left.strftime('%Y-%m-%d')} is after resolved right "
                f"edge {right.strftime('%Y-%m-%d')} (index coverage starts "
                f"{self._pit_coverage_start(self.config)}). An empty panel is never a "
                f"valid answer."
            )

        timestamps = pd.date_range(left, right, freq="D")
        values = np.zeros((len(timestamps), len(symbols)), dtype=bool)

        for row in rows:
            start = pd.Timestamp(row["start_date"])
            end = (
                pd.Timestamp(row["end_date"])
                if row["end_date"] is not None
                else right
            )
            # Include both ends: the day a removal takes effect is still a
            # membership day.
            mask = (timestamps >= start) & (timestamps <= end)
            values[mask, column_of[row["symbol"]]] = True

        # An empty table keeps its symbol column's type, so the empty axis is the one asked about.
        axis = np.asarray(symbols) if symbols else intervals["symbol"].to_numpy()
        return xr.Dataset(
            {"is_member": (["timestamp", "symbol"], values)},
            coords={"timestamp": timestamps, "symbol": axis},
        )

    @abstractmethod
    def _pit_coverage_start(self, config: ConstituentDatasetConfig) -> str:
        """Return the ISO date before which the source cannot answer membership.

        The panel never starts earlier than this, whatever the config asks.
        It reads ``config``, not ``self.config``: normalisation calls it on
        a config not yet installed, and it changes nothing on ``self``.
        """

    def _pit_coverage_end(self, config: ConstituentDatasetConfig) -> str | None:
        """Return the ISO date of the last day the source has observed, or None to read it off the intervals.

        A source whose intervals end where its observations do (the last
        membership change, an open interval running to ``as_of``) keeps the
        default. One that observes days on which no symbol is a member gives
        its last observed day, so those days stay on the panel and a source
        with no member at all still has one. It reads ``config``, as
        ``_pit_coverage_start`` does.
        """
        return None

    @abstractmethod
    def _build_intervals(self) -> pl.DataFrame:
        """Return the membership intervals as a polars frame.

        The frame has ``symbol``, ``start_date`` and ``end_date`` columns; a
        null ``end_date`` means the symbol is still a member.
        """

    def is_member_at(self, timestamps, symbols, *, owner: str | None = None) -> xr.DataArray:
        """Return whether each symbol is a member on each timestamp's date.

        Each timestamp is matched to the membership panel by its calendar
        date, so the answer at bar t reads the membership of t's date and
        nothing later. A symbol missing from the panel is not a member. A
        date the panel does not cover (before its first or after its last
        day) is refused, because unknown membership is not "not a member".

        Parameters
        ----------
        timestamps : array-like of datetime64
            The bars to answer for, such as a panel's ``timestamp`` axis.
        symbols : array-like
            The symbols to answer for, on the membership panel's symbol axis.
        owner : str, optional
            Who asks, named in the error message; defaults to this class.

        Returns
        -------
        xr.DataArray
            Booleans on ``(timestamp, symbol)`` with exactly the given
            coordinates.

        Raises
        ------
        ValueError
            If a timestamp's date lies outside the membership panel, or none
            of ``symbols`` is on the membership store's symbol axis (an
            integer permaticker against a text one), which would answer
            every symbol as a non-member.

        Examples
        --------
        With ``membership`` a ``DemoPanel`` (as in the class docstring) in
        which ``AAA`` is a member from 2024-01-01 on and ``BBB`` from
        2024-01-01 to 2024-01-02, saved with ``as_of="2024-01-05"``:

        >>> bars = pd.bdate_range("2024-01-02", periods=3)
        >>> membership.is_member_at(bars, ["AAA", "BBB", "ZZZ"]).to_pandas()
        symbol       AAA    BBB    ZZZ
        timestamp
        2024-01-02  True   True  False
        2024-01-03  True  False  False
        2024-01-04  True  False  False
        """
        timestamps = pd.DatetimeIndex(np.asarray(timestamps))
        symbols = np.asarray(symbols)
        owner = owner or type(self).__name__
        stored = self.stored_symbols()
        # A panel without symbols (no member ever) answers False for any symbol, on either axis type.
        if len(symbols) and len(stored) and not pd.Index(symbols).isin(stored).any():
            raise ValueError(
                f"{owner}: none of the {len(symbols)} symbol(s) asked about is on the "
                f"symbol axis of {type(self).__name__}'s store (labels {symbols[:3].tolist()} "
                f"of type {symbols.dtype}); every one would be a non-member. Check that both "
                f"are on the same axis (an integer permaticker against a text one)."
            )
        if len(timestamps) == 0:
            return xr.DataArray(
                np.zeros((0, len(symbols)), dtype=bool),
                dims=("timestamp", "symbol"),
                coords={"timestamp": timestamps, "symbol": symbols},
            )
        days = timestamps.normalize()
        is_member = self.panel(days.min(), days.max(), variables=["is_member"])["is_member"]
        uncovered = days.unique().difference(pd.DatetimeIndex(is_member["timestamp"].values))
        if len(uncovered):
            raise ValueError(
                f"{owner}: the membership panel of "
                f"{type(self).__name__} does not cover {len(uncovered)} "
                f"date(s) ({uncovered[0].date()}..{uncovered[-1].date()}); "
                f"extend the membership store or narrow the window, since "
                f"unknown membership is not 'not a member'"
            )
        return (
            is_member.sel(timestamp=days)
            .reindex(symbol=symbols)
            .fillna(False)
            .astype(bool)
            .assign_coords(timestamp=timestamps.values)
        )

    def mask(self, panel: xr.Dataset, *, owner: str | None = None) -> xr.Dataset:
        """Return ``panel`` with every cell of a symbol that is not a member on its bar's date NaN.

        This is the membership rule every masked factor, label and
        prediction applies (``MembershipMaskedFactor``,
        ``MembershipMaskedLabel``, ``MembershipMaskedPredictor``): a cell is
        kept when ``is_member_at`` is true for its bar and symbol. A panel
        without bars is returned as it is.

        Parameters
        ----------
        panel : xr.Dataset
            A panel on ``(timestamp, symbol)``.
        owner : str, optional
            Who asks, named in the error messages; defaults to this class.

        Returns
        -------
        xr.Dataset
            ``panel``, NaN where the symbol is not a member on the bar's date.

        Raises
        ------
        ValueError
            If a bar's date lies outside the membership panel, or no symbol
            of ``panel`` is on the membership store's symbol axis (an integer
            permaticker against a text one), which would mask every cell.

        Examples
        --------
        With ``membership`` the ``DemoPanel`` of ``is_member_at``'s example
        and ``panel`` a panel of ones on ``AAA`` and ``BBB`` over its three bars:

        >>> membership.mask(panel)["x"].to_pandas()
        symbol      AAA  BBB
        timestamp
        2024-01-02  1.0  1.0
        2024-01-03  1.0  NaN
        2024-01-04  1.0  NaN
        """
        if panel.sizes.get("timestamp", 0) == 0:
            return panel
        member = self.is_member_at(panel["timestamp"].values, panel["symbol"].values, owner=owner)
        return panel.where(member)
