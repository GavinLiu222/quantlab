"""US-equity bars read from one vendor's hive-partitioned parquet tree.

A *hive-partitioned* tree stores data in directories named
``<key>=<value>`` (for example ``month=2024-01/part.pqt``), so a reader can
skip whole directories that fall outside a query. A *panel* is an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, the format every
quantlab layer exchanges.

``StockDataset`` reads the raw files written by the acquisition layer under
``downloads/{market}/{frequency}/{subdir}/{vendor}/<key>=<value>/*.pqt``. It
filters on the date window inside the parquet scan, so only the partitions it
needs are opened, and returns the panel. It is the US-equity counterpart of
``quantlab.dataset.spot``, and its window-by-window conversion keeps memory
bounded by the size of one window.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawMarketDataset
from quantlab.dataset._support.cleaning import dedup_raw_frame
from quantlab.enums.data import RAW_HIVE_KEYS
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer


class StockDataset(RawMarketDataset):
    """Read US-equity prices or convert a vendor's downloaded files to a panel.

    Use ``panel`` to read an existing Zarr store. To build one from raw files,
    set ``raw_data_dir_path`` to a single vendor's download directory, then
    call ``from_raw_data().save()``.

    Daily files are partitioned by month, minute files by trading-session
    date, and tick files by date and symbol. When a vendor supplies several
    data types, ``kwargs["data_type"]`` selects the subdirectory, such as
    ``"bars"`` or ``"quotes"``. Files from different vendors cannot be mixed
    in one directory. Tick data cannot be converted to a dense price panel.

    Parameters
    ----------
    config : DatasetConfig
        Paths, market, frequency, vendor and date range of the dataset. For
        tick data, ``kwargs["data_type"]`` must be set (``"quotes"`` or
        ``"trades"`` for Alpaca).

    Examples
    --------
    >>> config = DatasetConfig(
    ...     raw_data_dir_path="downloads/us_equity/1d/us_all/tiingo",
    ...     zarr_file_path="data/us_equity/1d/us_all.zarr",
    ...     market="us_equity",
    ...     frequency="1d",
    ...     vendor="tiingo",
    ...     start_date="2024-01-01",
    ...     end_date="2024-12-31",
    ... )
    >>> StockDataset(config).from_raw_data_chunked(granularity="quarter")
    >>> panel = StockDataset(config).panel("2024-01-01", "2024-12-31")
    """

    #: Hive key dtypes per frequency, with the same keys as ``RAW_HIVE_KEYS``.
    #: Always passed to ``pl.scan_parquet`` explicitly: polars would infer a
    #: numeric-looking value such as ``symbol=8686`` as an integer, and a
    #: string filter on it would then silently match nothing.
    HIVE_SCHEMA_BY_FREQUENCY = {
        "1d": {"month": pl.String},
        "1m": {"date": pl.Date},
        "tick": {"date": pl.Date, "symbol": pl.String},
    }

    #: Hive keys that only describe the partition and are dropped after the
    #: scan. ``symbol`` is left out on purpose: in the tick layout the
    #: directory name is the only place the symbol is stored.
    DERIVED_HIVE_KEYS = ("month", "date")

    #: Filename suffix of one raw shard. The scan glob and ``has_raw_data``
    #: both use it, so they always look at the same set of files.
    RAW_SHARD_SUFFIX = ".pqt"

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """Return the hive partition keys for this config's frequency.

        They come from ``RAW_HIVE_KEYS``, the same mapping the acquisition
        layer uses to write the tree, so reader and writer always agree.
        """
        return RAW_HIVE_KEYS[self.config.frequency]

    def _assert_vendor_root(self) -> Path:
        """Return the raw root after checking it is one vendor's directory.

        ``pl.scan_parquet`` silently combines every shard below its root, and
        the later deduplication would then collapse two vendors' rows into an
        arbitrary blend. The root must therefore end at the configured
        vendor's own directory.

        Returns
        -------
        Path
            The raw root, ``config.raw_data_dir_path``.

        Raises
        ------
        ValueError
            If ``config.vendor`` is unset or the root's basename
            differs from it.
        """
        if not self.config.vendor:
            raise ValueError(
                f"{self.__class__.__name__}: DatasetConfig.vendor is not set, "
                f"so there is no way to check that "
                f"{self.config.raw_data_dir_path!r} holds exactly one vendor's "
                f"data. Two vendors under one root would merge with no error and "
                f"no record of which rows came from which vendor, so this scan "
                f"refuses rather than guessing. Set vendor=... on the config, "
                f"or build it via "
                f"config.stock_kline_config(vendor=...)."
            )

        root = Path(self.config.raw_data_dir_path)
        if root.name != self.config.vendor:
            raise ValueError(
                f"{self.__class__.__name__}: raw_data_dir_path {str(root)!r} "
                f"has basename {root.name!r} but the configured vendor is "
                f"{self.config.vendor!r}. The raw path must terminate at the "
                f"vendor directory. A root one level higher would read every "
                f"vendor directory beneath it and merge them with no error and "
                f"no record of origin, so this is refused rather than scanned. "
                f"Expected a path ending in "
                f"/{self.config.vendor}."
            )
        return root

    def _scan_root(self) -> Path:
        """Return the directory handed to ``pl.scan_parquet``.

        The vendor root, or ``{vendor_root}/{data type}`` when the dataset
        has a data type (``_raw_data_type``), which is where the acquisition
        wrote the shards. Polars takes the scan's schema from the first file
        it finds and enforces it on every file, even files a filter would
        skip. Two data types of one vendor have different columns, so they
        can only be separated by the root directory, never by a filter.

        Returns
        -------
        Path
            The directory to scan.
        """
        root = Path(self.config.raw_data_dir_path)
        data_type = self._raw_data_type
        return root if data_type is None else root / data_type

    def _scanned_hive_schema(self) -> dict:
        """Return the ``HIVE_SCHEMA_BY_FREQUENCY`` entry for this frequency."""
        return dict(self.HIVE_SCHEMA_BY_FREQUENCY[self.config.frequency])

    def _hive_window_predicate(self, start, end) -> pl.Expr:
        """Return the filter on hive keys that skips partitions outside the window.

        For ``1d`` the key is ``month``, a ``YYYY-MM`` string compared as
        text. The filter keeps both edge partitions whole; a separate
        ``timestamp`` filter trims them to the exact window.

        Parameters
        ----------
        start, end : datetime
            Inclusive window edges.

        Returns
        -------
        pl.Expr
            A boolean polars expression over the hive key columns.

        Raises
        ------
        NotImplementedError
            For a frequency with no known window key.
        """
        keys = self._hive_keys
        if keys == ("month",):
            return (pl.col("month") >= pl.lit(start.strftime("%Y-%m"))) & (
                pl.col("month") <= pl.lit(end.strftime("%Y-%m"))
            )
        if keys in (("date",), ("date", "symbol")):
            # ``date`` is the only window key for minute and tick data. A scan
            # reads every symbol, so ``symbol`` gets no filter.
            return self._session_date_window_predicate(start, end)
        raise NotImplementedError(
            f"{self.__class__.__name__}: no hive window predicate for "
            f"frequency {self.config.frequency!r} (scanned keys {keys})."
        )

    @property
    def _raw_data_type(self) -> str | None:
        """Return the data type whose directory the scan starts in, or None.

        It comes from ``config.kwargs["data_type"]``; a dataset class bound
        to one vendor product (CRSP, NBBO) overrides this with a constant.
        ``None`` means the vendor keeps one shape of data at this frequency
        under the vendor root itself (Tiingo). Tick data always needs one:
        quotes and trades share one vendor root but have different columns.

        Raises
        ------
        ValueError
            If the frequency is ``tick`` and the config does not say which
            data type to read.
        """
        data_type = (self.config.kwargs or {}).get("data_type")
        if data_type:
            return str(data_type)
        if self.config.frequency == "tick":
            raise ValueError(
                f"{self.__class__.__name__}: frequency "
                f"{self.config.frequency!r} needs kwargs['data_type'] set "
                f"(for Alpaca, 'quotes' or 'trades'); got {data_type!r}. Each "
                f"data type is stored in its own directory under the vendor "
                f"root and they have different columns. Scanning the vendor "
                f"root itself would raise a schema error instead of returning "
                f"a mixed frame. Say which one you want."
            )
        return None

    #: How far the intraday ``date`` filter widens the window at each edge.
    #: The ``date=`` key is a session date in the exchange's time zone, while
    #: window edges are naive UTC datetimes, and the two can differ by up to a
    #: day. One day of slack reads at most two extra partitions, which the
    #: ``timestamp`` filter then trims exactly. Without it, the end of the
    #: window's first session could be silently lost.
    SESSION_DATE_SLACK = timedelta(days=1)

    def _session_date_window_predicate(self, start, end) -> pl.Expr:
        """Return the ``date`` filter for the window, widened by ``SESSION_DATE_SLACK`` on each side."""
        return (
            pl.col("date") >= pl.lit((start - self.SESSION_DATE_SLACK).date())
        ) & (pl.col("date") <= pl.lit((end + self.SESSION_DATE_SLACK).date()))

    def _assert_single_vendor_and_drop(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Check the scanned ``vendor`` column holds one value, then drop it.

        This runs before ``dedup_raw_frame``, which would otherwise collapse
        two vendors' overlapping rows and hide the fact that they were mixed.

        Parameters
        ----------
        data : pl.LazyFrame
            The raw scan, still carrying the ``vendor`` column.

        Returns
        -------
        pl.LazyFrame
            The same scan without the ``vendor`` column.

        Raises
        ------
        ValueError
            If more than one vendor is present, or the one
            present is not the configured vendor.
        """
        vendors = (
            data.select(pl.col("vendor").unique())
            .collect()
            .get_column("vendor")
            .to_list()
        )
        if len(vendors) > 1:
            raise ValueError(
                f"{self.__class__.__name__}: the raw tree under "
                f"{self.config.raw_data_dir_path!r} holds rows from "
                f"{len(vendors)} vendors ({sorted(map(str, vendors))}) but is "
                f"configured for {self.config.vendor!r} alone. Merging two "
                f"vendors' bars produces a blended price series that cannot "
                f"be traced; deduplication on (timestamp, symbol) would then "
                f"collapse the overlaps arbitrarily. Put each vendor in its "
                f"own sibling directory rather than relaxing this check."
            )
        if vendors and str(vendors[0]) != self.config.vendor:
            raise ValueError(
                f"{self.__class__.__name__}: the raw tree under "
                f"{self.config.raw_data_dir_path!r} holds rows written by "
                f"vendor {str(vendors[0])!r} but the config says "
                f"{self.config.vendor!r}. The path and the data disagree; "
                f"refusing to scan rather than mislabelling the provenance of "
                f"everything downstream."
            )
        return data.drop("vendor")

    def has_raw_data(self) -> bool:
        """Return whether the raw tree holds at least one shard to convert.

        This is the one place that checks whether raw data exists.
        ``_scan_raw`` uses it to tell a missing root apart from a window that
        simply has no rows, and the ingest scripts call it to refuse a
        conversion before it starts. It looks inside the data type's
        directory (``_scan_root``), which a plain check of
        ``config.raw_data_dir_path`` would miss. It only lists files and
        opens no parquet file.

        Returns
        -------
        bool
            True if at least one ``.pqt`` shard exists below the scan root.

        Examples
        --------
        >>> ds = StockDataset(config)
        >>> ds.has_raw_data()  # the vendor root exists but holds no shard
        False
        >>> ds.has_raw_data()  # once a month=2024-01/part.pqt shard lands
        True
        """
        root = self._scan_root()
        return root.exists() and any(root.rglob(f"*{self.RAW_SHARD_SUFFIX}"))

    def _scan_raw(self, start_date=None, end_date=None) -> pl.LazyFrame:
        """Return the lazy (not yet collected) scan of the raw tree for a window.

        The scan skips partitions outside the window using the hive keys,
        trims the exact edges with a ``timestamp`` filter, checks that all
        rows come from the configured vendor, drops the hive metadata
        columns, and sorts. Except for tick data, it then removes duplicate
        ``(timestamp, symbol)`` rows, keeping the last.

        Parameters
        ----------
        start_date, end_date : date-like, optional
            Inclusive window edges. ``None`` means the config's own edge.

        Returns
        -------
        pl.LazyFrame
            The filtered, sorted scan.

        Raises
        ------
        ValueError
            If the vendor is unset or does not match the root, if the raw
            tree is missing or empty, or if the rows come from another vendor.
        """
        self._assert_vendor_root()
        root = self._scan_root()

        # A missing root would fail inside polars with an unhelpful schema
        # error. A window with no rows is fine and yields an empty frame.
        if not self.has_raw_data():
            raise ValueError(
                f"{self.__class__.__name__}: no raw data for vendor "
                f"{self.config.vendor!r} at frequency "
                f"{self.config.frequency!r} under {str(root)!r}. Fetch it "
                f"first with the vendor's ingest script before "
                f"converting. This means the raw root is missing; a window "
                f"that merely has zero rows returns an empty frame instead."
            )

        # Glob for the shard suffix instead of passing the bare directory:
        # polars refuses a directory with mixed file types, and one stray
        # `.DS_Store` is enough. Partition skipping still works with a glob.
        # `hive_schema` stops numeric-looking keys being read as integers.
        # Polars' default of raising on extra or missing columns is kept, so a
        # shard with unexpected columns is an error.
        data = pl.scan_parquet(
            str(root / "**" / f"*{self.RAW_SHARD_SUFFIX}"),
            hive_partitioning=True,
            hive_schema=self._scanned_hive_schema(),
        )

        start = self._as_datetime(
            self.config.start_date if start_date is None else start_date
        )
        end = self._as_datetime(
            self.config.end_date if end_date is None else end_date
        )

        # Both filters are needed. The hive filter skips whole partitions
        # before reading (a `timestamp` filter cannot), and the timestamp
        # filter trims the two partly covered edge partitions exactly.
        data = data.filter(self._hive_window_predicate(start, end))
        data = data.filter(
            pl.col("timestamp") >= pl.lit(start),
            pl.col("timestamp") <= pl.lit(end),
        )

        data = self._assert_single_vendor_and_drop(data)
        # Drop only the metadata keys; `symbol` is real data that the tick
        # layout stores as a directory name.
        data = data.drop(
            [key for key in self._hive_keys if key in self.DERIVED_HIVE_KEYS]
        )

        data = data.sort(by=["timestamp", "symbol"])
        if self.config.frequency == "tick":
            # Many genuine quotes or trades share one (timestamp, symbol), and
            # tick data never becomes a dense panel, so it is not deduplicated.
            return data
        return dedup_raw_frame(data, keep="last")

    @staticmethod
    def _as_datetime(value) -> datetime:
        """Convert a window edge to a naive ``datetime``.

        An ISO date string becomes midnight of that date, so an inclusive
        ``<=`` end edge still keeps that day's daily bar. A ``pd.Timestamp``
        passed by the chunked conversion keeps its time of day.

        Parameters
        ----------
        value : str, date, datetime or pd.Timestamp
            The window edge.

        Returns
        -------
        datetime
            The same instant as a Python ``datetime``.
        """
        return pd.Timestamp(value).to_pydatetime()

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the symbol axis and the observed timestamps of the raw data.

        Only the two axis columns are collected, so no dense panel is built.
        Symbols are ordered by ``sort_symbol_axis`` and keep the dtype of the
        raw ``symbol`` column.

        Returns
        -------
        symbols : list
            Every symbol in the configured range, sorted.
        timestamps : pd.DatetimeIndex
            Every distinct timestamp in the configured range, sorted.
        """
        scan = self._scan_raw()
        symbols = sort_symbol_axis(
            scan.select("symbol").unique().collect()["symbol"].to_list()
        )
        timestamps = (
            scan.select("timestamp").unique().collect()["timestamp"].to_list()
        )
        return symbols, pd.DatetimeIndex(sorted(timestamps))

    def _added_symbols_with_raw_history(
        self, added: list, start, end
    ) -> dict[str, int]:
        """Count the raw rows each symbol in ``added`` has inside a window.

        This overrides the base-class check with a cheaper one: the symbol
        filter runs inside the parquet scan and only a per-symbol count is
        collected. Symbols are compared as text because the raw ``symbol``
        column holds strings even when the stored symbol axis is integer.
        Without that conversion no rows would match, and ``update`` would
        wrongly treat the symbols as new listings with no history.

        Parameters
        ----------
        added : list
            Symbols that are new relative to the stored panel.
        start, end : date-like
            Inclusive window edges.

        Returns
        -------
        dict of str to int
            Row count per symbol that has at least one row.
        """
        wanted = [str(symbol) for symbol in added]
        if not wanted:
            return {}

        counts = (
            self._scan_raw(start, end)
            .filter(pl.col("symbol").is_in(wanted))
            .group_by("symbol")
            .agg(pl.len().alias("rows"))
            .collect()
        )
        return {
            str(record["symbol"]): int(record["rows"])
            for record in counts.to_dicts()
        }

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: Optional[list[str]] = None
    ) -> xr.Dataset:
        """Return the dense panel for one window, reindexed onto ``symbols``.

        Only the window's partitions are scanned, so memory use depends on
        the window size, not on the whole configured range.

        Parameters
        ----------
        start_date, end_date : date-like
            Inclusive window edges.
        symbols : list of str, optional
            If given, the result has exactly these symbols, in this order;
            a symbol with no rows becomes an all-NaN column.

        Returns
        -------
        xr.Dataset
            The panel for the window.
        """
        data = self._scan_raw(start_date, end_date)
        data = data.collect().to_pandas().set_index(["timestamp", "symbol"])
        data = data.to_xarray()
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the dense panel for the whole configured date range."""
        with Timer(f" {self.__class__.__name__}: from pqt"):
            return self._raw_data_to_xr_window(
                self.config.start_date, self.config.end_date, symbols=None
            )

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the requested columns as float32 arrays for KunQuant.

        Parameters
        ----------
        data : xr.Dataset
            The panel to export.
        data_columns : tuple of str
            Variables to export.

        Returns
        -------
        input_dict : dict of str to np.ndarray
            One C-contiguous float32 array of shape ``[time, symbol]`` per
            column.
        symbols : np.ndarray
            Symbol labels of the second axis.
        timestamp : np.ndarray
            Timestamps of the first axis.
        """
        with Timer(f"{self.__class__.__name__}: to kunquant"):
            return self._kunquant_arrays(self.to_shared_names(data), data_columns)
