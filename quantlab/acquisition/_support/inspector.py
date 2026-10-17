"""Read-only inspection of the data a source has already downloaded.

Each data source keeps two kinds of files on disk, called *tiers*. The raw
tier is the vendor's rows as downloaded, stored as parquet files (*shards*)
in a directory tree. The Zarr tier is the converted *panel*: an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, saved in the Zarr
format. Beside the raw tier sit small per-symbol JSON files (*watermark
sidecars*) recording which dates each symbol has been downloaded for, and a
failure manifest listing symbols whose downloads failed.

``SourceInspector`` answers questions about these files alone: which
requested symbols are already downloaded, which failed in earlier runs, how
many shards and sidecars exist, and a row-level view of either tier limited
to chosen symbols and dates. It imports no vendor client and never creates an
``SymbolBatchAcquisition``, so it needs no API key and makes no network request,
whatever is called in whatever order. Listing the sources themselves is left
to ``quantlab.acquisition.registry``, whose import loads every vendor module.

Nothing is cached on the instance. Every method builds its own coverage
ledger or opens its own store and closes it before returning, so one query
cannot restrict what a later query on the same inspector sees.
"""

import os
from pathlib import Path
from typing import Sequence

import polars as pl
import xarray as xr

from quantlab.acquisition.config import AcquisitionConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.acquisition._support.coverage import CoverageLedger, validate_symbols
from quantlab.dataset.stock import StockDataset


class _RawTierReader(StockDataset):
    """A ``StockDataset`` that reads nothing when constructed.

    The base config setter calls ``_reset_symbols`` whenever ``symbols`` is
    assigned. That opens the Zarr store, or builds the whole panel from the
    raw tier when the store is missing. A browse only needs a lazy view of
    the raw tier, so the method is overridden to do nothing. A fresh reader
    is built per query, which lets ``browse_raw`` reuse ``_scan_raw`` (with
    its directory, schema and column checks) instead of a second copy.
    """

    def _reset_symbols(self) -> None:
        """Do nothing: the inspector never needs a resolved symbol axis."""
        return None


class SourceInspector:
    """Answer read-only questions about locally downloaded data.

    Every method takes the config that locates the data and answers from the
    file system. The inspector holds no state, so one instance can be kept
    and reused, or a new one built per call. The constructor takes no
    arguments.

    Examples
    --------
    ``acq_cfg`` is an ``AcquisitionConfig`` and ``ds_cfg`` a
    ``DatasetConfig`` for the same source::

        from quantlab.acquisition._support.inspector import SourceInspector

        inspector = SourceInspector()
        print(inspector.coverage(acq_cfg)["pending"])
        print(inspector.inventory(acq_cfg, ds_cfg)["raw"]["shards"])
    """

    def __repr__(self) -> str:
        """Return ``SourceInspector()``; the instance carries no state."""
        return f"{self.__class__.__name__}()"

    # -- coverage -----------------------------------------------------------

    def coverage(
        self,
        config: AcquisitionConfig,
        symbols: Sequence[str] | None = None,
    ) -> dict:
        """Report which symbols are already downloaded for the config's window.

        This runs the same check as ``SymbolBatchAcquisition.coverage_report()``,
        through the same ``CoverageLedger`` class, so the two always agree.
        Symbols are validated before any path is built, because each symbol
        becomes a sidecar file name.

        Parameters
        ----------
        config : AcquisitionConfig
            Locates the watermark sidecars and supplies the window.
        symbols : Sequence[str] | None, default None
            The symbols to classify; ``config.symbols`` when ``None``.

        Returns
        -------
        dict
            Symbol counts under ``requested``, ``pending`` (need a download),
            ``skipped`` (already downloaded), and the per-state counts
            ``covered``, ``widened`` (recorded start is later than the
            requested start), ``legacy`` (no recorded start) and ``no_data``
            (the vendor had nothing).

        Examples
        --------
        ::

            report = inspector.coverage(acq_cfg, ["AAPL", "MSFT"])
            print(report["pending"], "of", report["requested"], "still needed")
        """
        ledger = CoverageLedger.for_config(config)
        requested = ledger.validate_symbols(
            list(symbols if symbols is not None else config.symbols)
        )
        pending, counts = ledger.partition_by_coverage(
            requested, from_watermark=False
        )
        return {
            "requested": len(requested),
            "pending": len(pending),
            "skipped": len(requested) - len(pending),
            **counts,
        }

    # -- failures -----------------------------------------------------------

    def failures(self, config: AcquisitionConfig) -> dict[str, str]:
        """Return every symbol the failure manifest records, across runs.

        The manifest (``_failures.json`` beside the watermark sidecars) is
        written atomically, so it survives a crash. It collects entries
        across runs, so it may name symbols the most recent run never
        requested. The reasons already have credentials removed. A missing
        or unreadable manifest gives ``{}``. Resuming a download does not
        read the manifest; it only checks which sidecars exist.

        Parameters
        ----------
        config : AcquisitionConfig
            Locates the watermark directory holding the manifest.

        Returns
        -------
        dict[str, str]
            ``{symbol: reason}``, or ``{}``.
        """
        return CoverageLedger.for_config(config).read_failure_manifest()

    # -- inventory ----------------------------------------------------------

    def inventory(
        self,
        config: AcquisitionConfig,
        dataset_config: DatasetConfig | None = None,
    ) -> dict:
        """Report what exists on disk, for the raw tier and Zarr tier separately.

        The two tiers are different things: the raw tier holds the vendor's
        rows and only grows, while the Zarr store is the converted panel and
        may not exist yet. Reading the sidecars is the slow part, so every
        figure comes from one directory walk and one pass over the sidecars.

        Parameters
        ----------
        config : AcquisitionConfig
            Locates the raw tier and its sidecars.
        dataset_config : DatasetConfig | None, default None
            Locates the Zarr store. When omitted, ``"zarr"``
            is ``None``, meaning "not asked" rather than "absent".

        Returns
        -------
        dict
            ``{"raw": {...}, "zarr": {...} | None}``. The raw dict carries the
            root, shard count, byte total, sidecar count, coverage span and
            failure count; the zarr dict carries the path, byte total,
            dimension sizes, variable names and timestamp span.

        Examples
        --------
        ::

            report = inspector.inventory(acq_cfg, ds_cfg)
            print(report["raw"]["shards"], "shards,",
                  report["raw"]["symbols_with_watermark"], "symbols")
            if not report["zarr"]["exists"]:
                print("not converted yet")
        """
        ledger = CoverageLedger.for_config(config)
        return {
            "raw": self._raw_inventory(config, ledger),
            "zarr": (
                None
                if dataset_config is None
                else self._zarr_inventory(dataset_config)
            ),
        }

    @staticmethod
    def _raw_root(config: AcquisitionConfig, ledger: CoverageLedger) -> Path:
        """Return the directory this config's raw shards are written under.

        A run with a data type writes under ``<vendor root>/<data type>``,
        so two data types of one vendor are never counted together.
        """
        root = Path(config.raw_data_dir_path)
        if ledger.data_type is not None:
            root = root / str(ledger.data_type)
        return root

    def _raw_inventory(
        self, config: AcquisitionConfig, ledger: CoverageLedger
    ) -> dict:
        """Compute the raw tier's figures from one walk and one sidecar pass.

        See ``inventory`` for the keys of the returned dict.
        """
        root = self._raw_root(config, ledger)

        shards = 0
        total_bytes = 0
        if root.exists():
            # One `os.walk` for every figure, rather than an `rglob` per figure.
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    if not filename.endswith(".pqt"):
                        continue
                    shards += 1
                    try:
                        total_bytes += (Path(dirpath) / filename).stat().st_size
                    except OSError:
                        # Deleted between listing and stat; just skip it.
                        continue

        # One pass over the sidecars gives every figure below.
        symbols_with_watermark = 0
        no_data = 0
        earliest_start: str | None = None
        latest_last_date: str | None = None
        for symbol in ledger.iter_watermark_symbols():
            coverage = ledger.read_coverage(symbol)
            if coverage is None:
                continue
            symbols_with_watermark += 1
            if coverage["no_data"]:
                no_data += 1
            start = coverage["start_date"]
            if start is not None and (
                earliest_start is None or start < earliest_start
            ):
                earliest_start = start
            last = coverage["last_date"]
            if last is not None and (
                latest_last_date is None or last > latest_last_date
            ):
                latest_last_date = last

        return {
            "vendor": config.vendor,
            "market": config.market,
            "frequency": config.frequency,
            "data_type": ledger.data_type,
            "root": str(root),
            "exists": root.exists(),
            "shards": shards,
            "bytes": total_bytes,
            "watermark_root": str(ledger.watermark_root),
            "symbols_with_watermark": symbols_with_watermark,
            # ISO date strings sort correctly as text, so no parsing.
            "coverage_start": earliest_start,
            "coverage_last_date": latest_last_date,
            "no_data": no_data,
            "failures": len(self.failures(config)),
        }

    @staticmethod
    def _zarr_inventory(dataset_config: DatasetConfig) -> dict:
        """Compute the Zarr tier's figures from metadata, then close the store.

        ``xr.open_zarr`` reads only metadata and coordinates, so the sizes,
        variable names and timestamp span need no data reads. Only plain
        Python values are returned, never an open store.
        """
        path = Path(dataset_config.zarr_file_path)
        if not path.exists():
            return {
                "path": str(path),
                "exists": False,
                "bytes": 0,
                "dims": {},
                "data_vars": [],
                "timestamp_start": None,
                "timestamp_end": None,
            }

        total_bytes = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                try:
                    total_bytes += (Path(dirpath) / filename).stat().st_size
                except OSError:
                    continue

        dataset = xr.open_zarr(path)
        try:
            span: tuple[str | None, str | None] = (None, None)
            timestamps = (
                dataset["timestamp"] if "timestamp" in dataset.coords else None
            )
            if timestamps is not None and timestamps.size:
                values = timestamps.values
                span = (str(values[0]), str(values[-1]))
            return {
                "path": str(path),
                "exists": True,
                "bytes": total_bytes,
                "dims": {name: int(size) for name, size in dataset.sizes.items()},
                "data_vars": sorted(str(name) for name in dataset.data_vars),
                "timestamp_start": span[0],
                "timestamp_end": span[1],
            }
        finally:
            dataset.close()

    # -- row-level browsing -------------------------------------------------

    #: Error message for an empty ``symbols`` in ``browse_raw`` and
    #: ``browse_zarr``. Both require the symbols and the date window so the
    #: view they return is already limited. An empty list could be mistaken
    #: for "everything", so it is refused rather than answered with zero rows.
    EMPTY_SYMBOLS_MESSAGE = (
        "symbols must be a non-empty sequence. Symbols and the date window are "
        "required arguments so that the lazy view is already limited when it "
        "is returned (us_all is ~15.4k symbols x ~5.2k trading days, ~30M "
        "rows). An empty list could be read as 'everything', so it is refused "
        "rather than answered with zero rows."
    )

    def _require_symbols(self, symbols: Sequence[str], caller: str) -> list[str]:
        """Return ``symbols`` as a list, raising ``ValueError`` if it is empty.

        ``caller`` names the public method in the error message.
        """
        listed = list(symbols)
        if not listed:
            raise ValueError(f"{caller}: {self.EMPTY_SYMBOLS_MESSAGE}")
        return listed

    def browse_raw(
        self,
        dataset_config: DatasetConfig,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pl.LazyFrame:
        """Return a lazy view of the raw parquet tier for chosen symbols and dates.

        All four arguments are required so the view is already limited when
        it is returned; reading a whole tier means passing every symbol and
        the full window on purpose. Nothing is read until the caller calls
        ``collect()``.

        The scan itself is ``StockDataset._scan_raw``, which checks that the
        root is a vendor directory, enters the tick data-type subdirectory,
        fixes the partition column types and raises on unexpected or missing
        columns. This method only adds the symbol filter and a sort. Symbols
        are checked against the shared ticker pattern first. For daily and
        minute data ``symbol`` is a column, so the symbol filter skips parts
        of files while the date window skips whole directories; for tick data
        ``symbol`` is a directory level, so the filter skips directories.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Locates the vendor's raw root.
        symbols : Sequence[str]
            Non-empty sequence of tickers to keep.
        start_date : str
            Inclusive ISO start of the window.
        end_date : str
            Inclusive ISO end of the window.

        Returns
        -------
        polars.LazyFrame
            Rows sorted by ``(timestamp, symbol)``, so repeated collection
            gives the same row order.

        Raises
        ------
        ValueError
            If ``symbols`` is empty or contains an invalid ticker.

        Examples
        --------
        ::

            frame = inspector.browse_raw(
                ds_cfg, ["AAPL"], "2024-02-01", "2024-03-31"
            )
            rows = frame.collect()
        """
        listed = self._require_symbols(symbols, "browse_raw")
        listed = validate_symbols(
            listed,
            owner_label=f"{self.__class__.__name__}.browse_raw",
            raw_root=dataset_config.raw_data_dir_path,
        )
        # A fresh reader per query, so this query's filter cannot affect a
        # later one.
        reader = _RawTierReader(dataset_config)
        return (
            reader._scan_raw(start_date, end_date)
            .filter(pl.col("symbol").is_in(listed))
            .sort(["timestamp", "symbol"])
        )

    def browse_zarr(
        self,
        dataset_config: DatasetConfig,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> xr.Dataset:
        """Return the Zarr store's data for chosen symbols and dates, unloaded.

        The store is opened fresh on every call, so no selection carries over
        to the next. Arrays are not loaded until used. A symbol the store
        does not hold raises rather than being filled in as a column of NaN,
        because such a column would look exactly like a genuinely empty
        history. Some stores come from CRSP (the Center for Research in
        Security Prices), whose ``symbol`` axis holds integer PERMNOs,
        CRSP's permanent security identifiers, instead of tickers. For such a
        store the message says so and points at the ticker file beside the
        store, since a ticker can never match that axis.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Locates the Zarr store.
        symbols : Sequence[str]
            Non-empty sequence of symbol labels to select.
        start_date : str
            Inclusive ISO start of the window.
        end_date : str
            Inclusive ISO end of the window.

        Returns
        -------
        xarray.Dataset
            The selected data, with the store's arrays still unloaded.

        Raises
        ------
        ValueError
            If ``symbols`` is empty, or if the store does not hold every
            requested symbol. The message names the store, the missing
            symbols and how many symbols the store holds.

        Examples
        --------
        ::

            view = inspector.browse_zarr(
                ds_cfg, ["AAPL"], "2024-01-10", "2024-01-12"
            )
            print(dict(view.sizes))
        """
        listed = self._require_symbols(symbols, "browse_zarr")
        # No ticker-pattern check here: a symbol is only looked up in an
        # index, never used as a path, and an unknown one raises below.
        path = Path(dataset_config.zarr_file_path)
        dataset = xr.open_zarr(path)
        try:
            return dataset.sel(
                symbol=listed, timestamp=slice(start_date, end_date)
            )
        except KeyError as exc:
            carried = int(dataset.sizes.get("symbol", 0))
            known = set()
            axis_dtype = None
            if "symbol" in dataset.coords:
                known = {str(value) for value in dataset["symbol"].values}
                axis_dtype = dataset["symbol"].dtype
            missing = sorted(symbol for symbol in listed if symbol not in known)
            dataset.close()
            # On an integer (PERMNO) axis, "does not carry AAPL" alone would
            # mislead: the store may hold that company under its PERMNO.
            axis_note = ""
            if axis_dtype is not None and axis_dtype.kind in "iu":
                # Imported here because importing the CRSP package loads its
                # whole converter, which other vendors do not need.
                from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX

                axis_note = (
                    f" This store's symbol axis is integer "
                    f"(dtype {axis_dtype}), i.e. CRSP PERMNOs, not tickers, "
                    f"because tickers get reused by different companies over "
                    f"time. A ticker can never match it, whether or not "
                    f"the security is present. The tickers live in "
                    f"'{path.name}{TICKER_SIDECAR_SUFFIX}' beside the store "
                    f"and are queried as-of through "
                    f"quantlab/dataset/crsp/tickers.py:CrspTickerLookup; ask "
                    f"for the PERMNO it gives you."
                )
            raise ValueError(
                f"{self.__class__.__name__}.browse_zarr: the Zarr store at "
                f"{str(path)!r} does not carry {missing} (requested "
                f"{sorted(listed)}). The store carries {carried} symbol(s)."
                f"{axis_note} "
                f"This is refused rather than reindexed on purpose: a "
                f"NaN-filled column for a symbol the store has never heard of "
                f"is indistinguishable from a symbol with a genuinely empty "
                f"history, and an operator cannot tell those two apart from "
                f"the data. Convert the raw tier for these symbols, or ask for "
                f"the ones the store has."
            ) from exc
        except BaseException:
            dataset.close()
            raise
