"""Point-in-time panel of Sharadar fundamentals (SF1), keyed by permaticker.

SF1 holds one row per company, *dimension* and filing. Only the as-reported
dimensions are point-in-time: ARQ (each fiscal quarter) and ART (trailing
twelve months). Their ``date`` is the SEC filing date, the row's *release
date*, and a later filing that restates a period is a new row rather than a
rewrite of the old one. The most-recent dimensions (MRQ, MRY, MRT) are dated
at the period end and rewritten on restatement, so they would leak a value
one to three months early; they stay in the raw tier and never enter a store.

``SharadarFundamentalsDataset`` places each row of one as-reported dimension
at its *available date*, the first SEP trading day on or after its release
date, and is *point-in-time* (as the Compustat panel of ADR 0003 is): on
every trading day it shows, per security, the row with the latest fiscal
period among the rows available by that day, and within that period the
latest filing. So:

- a restatement of a period is shown from its own release date, never
  earlier;
- a later filing of an older period never replaces a newer period's row;
- a row no newer row has replaced stops being shown ``stale_after_days``
  after its release date (365 by default, as for Compustat).

The panel's calendar is SEP's trading days, so it lines up with the price
panels bar for bar, and its ``symbol`` axis is the int64 permaticker, as
theirs is. Each of SF1's 105 indicators is a float64 variable whose
``unit`` attribute is the vendor's unit type from INDICATORS (``currency``
is the reporting currency, ``USD``, ``ratio``, ``units``, per share);
``release_date`` and ``reportperiod`` (the fiscal period end) say which row
is shown. The store grows with ``update()``: new trading days are appended
and a stored day is never rewritten, so a value the vendor changes later
(SF1 rows are refreshed by ``lastupdated``, ``SharadarClient.updated_table``)
reaches only the days after the update.

Examples
--------
Build the quarterly store from a downloaded raw tier and read a quarter::

    config = SharadarFundamentalsConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_sf1_arq/sharadar_sf1_arq.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarFundamentalsDataset(config).update()
    panel = SharadarFundamentalsDataset(config).panel("2024-01-02", "2024-03-28")
    panel["revenue"].attrs["unit"]  # 'currency'

SF1 also holds DAILY's valuation indicators, as of each filing. Merged with
the DAILY panel (``MergedDataset``), they are renamed ``sf1_marketcap``,
``sf1_ev``, ``sf1_pe`` and so on (``COLUMN_MAP``), so ``marketcap`` is
DAILY's daily value.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawDataset
from quantlab.dataset.config import DatasetConfig, SharadarFundamentalsConfig
from quantlab.dataset.sharadar.permatickers import map_raw_table
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.dataset.sharadar.tables import (
    DAILY_INDICATORS,
    SF1_INDICATORS,
    raw_through,
    scan_raw_table,
    table,
    trading_days,
)
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: The dimensions a store may hold: as reported, quarterly or trailing twelve months.
AS_REPORTED: tuple[str, ...] = ("ARQ", "ART")

#: The date variables saying which row a cell shows.
DATE_VARIABLES: tuple[str, ...] = ("release_date", "reportperiod")

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


class SharadarFundamentalsDataset(RawDataset):
    """Daily point-in-time panel of one as-reported SF1 dimension, keyed by permaticker.

    Parameters
    ----------
    dataset_config : SharadarFundamentalsConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``sf1``, ``sep`` (the calendar), ``tickers`` and ``indicators`` raw
        tables; ``dimension`` is ``"ARQ"`` or ``"ART"``.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarFundamentalsDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)[:3]
        # ['accoci', 'assets', 'assetsavg']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarFundamentalsConfig

    #: SF1's valuation indicators that DAILY also holds (``marketcap``,
    #: ``ev``, ``pe``, ...) go by ``sf1_<name>`` when merged or exported, so
    #: a merge with the DAILY panel keeps DAILY's daily values under the
    #: plain names. The store and ``panel()`` keep SF1's own names.
    COLUMN_MAP = {
        name: f"sf1_{name}" for name in DAILY_INDICATORS if name in SF1_INDICATORS
    }

    def _normalize_config(self, config: DatasetConfig) -> SharadarFundamentalsConfig:
        """Normalise as the base class does, then check the SF1 fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarFundamentalsConfig``.
        ValueError
            If ``table`` is not ``"sf1"``, ``dimension`` is not as reported,
            ``stale_after_days`` is below 1, or a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``).

        Examples
        --------
        A most-recent dimension is refused::

            SharadarFundamentalsDataset(dataclasses.replace(config, dimension="MRQ"))
            # ValueError: SharadarFundamentalsDataset: dimension 'MRQ' is not as reported ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarFundamentalsConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarFundamentalsConfig, got "
                f"{type(config).__name__}."
            )
        if config.table != "sf1":
            raise ValueError(f"{self.class_name}: table must be 'sf1', got {config.table!r}.")
        if config.dimension not in AS_REPORTED:
            raise ValueError(
                f"{self.class_name}: dimension {config.dimension!r} is not as "
                f"reported; a store holds {' or '.join(AS_REPORTED)}. The "
                f"most-recent dimensions are dated at the period end, not the "
                f"release date, and would leak values early."
            )
        if config.stale_after_days is not None and config.stale_after_days < 1:
            raise ValueError(
                f"{self.class_name}: stale_after_days must be at least 1 or "
                f"None, got {config.stale_after_days}."
            )
        return normalize_universe(config, self.class_name)

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        self._shown_cache: pl.DataFrame | None = None
        self._days_cache: pl.Series | None = None

    # -- inputs ---------------------------------------------------------------

    def _raw_through(self) -> date:
        """Return the last day both SF1 and the SEP calendar are complete through.

        See ``quantlab.dataset.sharadar.tables.raw_through``.
        """
        return raw_through(self.config.raw_data_dir_path, ("sf1", "sep"))

    def _trading_days(self) -> pl.Series:
        """Return SEP's trading days up to ``_raw_through()``, sorted, as ns datetimes (cached)."""
        if self._days_cache is None:
            self._days_cache = trading_days(self.config.raw_data_dir_path, self._raw_through())
        return self._days_cache

    def _shown(self) -> pl.DataFrame:
        """Return the rows the panel ever shows, each at its available date (cached).

        A row is kept when its fiscal period is the latest of its
        permaticker's rows released by then, so a later filing of an older
        period is dropped. Of several rows reaching one available date, the
        last filed (and so the latest period) is kept.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` (a trading day), the ``DATE_VARIABLES``
            and ``SF1_INDICATORS``, sorted by ``available``.

        Raises
        ------
        ValueError
            If a raw ticker has two permatickers, or two rows of one ticker
            and permaticker share a release date and fiscal period (rows
            under two tickers are settled, ``PermatickerResolver.settle``).
        """
        if self._shown_cache is not None:
            return self._shown_cache
        config = self.config
        root = config.raw_data_dir_path
        with Timer(f"{self.class_name}: shown rows"):
            frame = map_raw_table(
                root,
                "sf1",
                owner=self.class_name,
                store_path=config.zarr_file_path,
                query=lambda scan: scan.filter(pl.col("dimension") == config.dimension).filter(
                    pl.col("date") <= pl.lit(self._raw_through())
                ),
                key=("dimension", "date", "reportperiod"),
            )
            frame = frame.filter(pl.col("permaticker").is_in(universe(config)))
            self._assert_unique_keys(frame)
            frame = frame.sort("permaticker", "date", "reportperiod").filter(
                pl.col("reportperiod")
                == pl.col("reportperiod").cum_max().over("permaticker")
            )
            available = (
                frame.select(pl.col("date").unique().sort())
                .join_asof(
                    self._trading_days().to_frame("available").with_columns(
                        pl.col("available").cast(pl.Date).alias("date")
                    ),
                    on="date",
                    strategy="forward",
                )
            )
            frame = (
                frame.join(available, on="date", how="inner")
                .filter(pl.col("available").is_not_null())
                # A join does not keep row order; the last filing must be last.
                .sort("permaticker", "date", "reportperiod")
                .unique(subset=["permaticker", "available"], keep="last", maintain_order=True)
                .select(
                    pl.col("permaticker").alias("symbol"),
                    "available",
                    pl.col("date").cast(pl.Datetime("ns")).alias("release_date"),
                    pl.col("reportperiod").cast(pl.Datetime("ns")),
                    *(pl.col(name).cast(pl.Float64) for name in SF1_INDICATORS),
                )
                .sort("available")
            )
        self._shown_cache = frame
        return frame

    def _assert_unique_keys(self, frame: pl.DataFrame) -> None:
        """Raise if two rows of one permaticker share a release date and fiscal period."""
        duplicates = (
            frame.group_by("permaticker", "date", "reportperiod")
            .agg(pl.col("ticker").sort())
            .filter(pl.col("ticker").list.len() > 1)
            .sort("permaticker", "date")
        )
        if duplicates.height:
            sample = [
                f"{row['permaticker']}@{row['date']}/{row['reportperiod']}:{row['ticker']}"
                for row in duplicates.head(_ERROR_SAMPLE).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {duplicates.height} (permaticker, release "
                f"date, fiscal period) key(s) have several raw rows, first "
                f"{sample}; refusing rather than choosing one."
            )

    def _units(self) -> dict[str, str]:
        """Return each SF1 indicator's unit type from the INDICATORS table."""
        rows = (
            scan_raw_table(self.config.raw_data_dir_path, "indicators")
            .filter(pl.col("table").is_in(table("sf1").tickers_labels))
            .select("indicator", "unittype")
            .collect()
        )
        return dict(rows.iter_rows())

    # -- axes and windows -------------------------------------------------------

    def _days_in(self, start_date, end_date) -> pl.Series:
        """Return the trading days in the configured window and ``start_date``..``end_date``."""
        first = max(pd.Timestamp(start_date), pd.Timestamp(self.config.start_date))
        last = min(pd.Timestamp(end_date), pd.Timestamp(self.config.end_date))
        days = self._trading_days()
        return days.filter(
            days.is_between(pl.lit(first.to_datetime64()), pl.lit(last.to_datetime64()))
        )

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the permatickers shown by the window's end, and its trading days.

        Raises
        ------
        ValueError
            If no row is shown in the configured window.
        """
        days = self._days_in(self.config.start_date, self.config.end_date)
        shown = self._shown()
        if days.len():
            shown = shown.filter(pl.col("available") <= days.max())
        symbols = shown.get_column("symbol").unique().to_list()
        if not days.len() or not symbols:
            raise ValueError(
                f"{self.class_name}: no {self.config.dimension} row is shown in "
                f"[{self.config.start_date}, {self.config.end_date}] for the "
                f"configured universe."
            )
        return sort_symbol_axis(symbols), pd.DatetimeIndex(days.to_list())

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Return one window of the panel: every trading day by every symbol.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list of int, optional
            Permatickers to build; ``None`` is every one shown by the
            window's end.
        """
        days = self._days_in(start_date, end_date)
        shown = self._shown()
        if symbols is None:
            last = days.max() if days.len() else None
            visible = shown if last is None else shown.filter(pl.col("available") <= last)
            symbols = sort_symbol_axis(visible.get_column("symbol").unique().to_list())
        symbols = [int(s) for s in symbols]
        grid = days.to_frame().with_row_index("_day").join(
            pl.DataFrame({"symbol": symbols}, schema={"symbol": pl.Int64}).with_row_index("_column"),
            how="cross",
        ).sort("_day", "_column")
        cells = grid.join_asof(
            shown.filter(pl.col("symbol").is_in(symbols)),
            left_on="timestamp",
            right_on="available",
            by="symbol",
            strategy="backward",
            # Both sides are sorted by time; polars cannot check that per group.
            check_sortedness=False,
        ).sort("_day", "_column")
        if self.config.stale_after_days is not None:
            limit = timedelta(days=self.config.stale_after_days)
            fresh = (pl.col("timestamp") - pl.col("release_date")) <= pl.lit(limit)
            cells = cells.with_columns(
                pl.when(fresh).then(pl.col(name)).otherwise(None).alias(name)
                for name in (*DATE_VARIABLES, *SF1_INDICATORS)
            )
        shape = (days.len(), len(symbols))
        units = self._units()
        variables = {}
        for name in (*SF1_INDICATORS, *DATE_VARIABLES):
            values = cells.get_column(name).to_numpy().reshape(shape)
            attrs = {"unit": units[name]} if name in units else {}
            variables[name] = (("timestamp", "symbol"), values, attrs)
        variables["release_date"][2]["description"] = "SEC filing date of the row shown"
        variables["reportperiod"][2]["description"] = "fiscal period end of the row shown"
        return xr.Dataset(
            variables,
            coords={
                "timestamp": pd.DatetimeIndex(days.to_list()),
                "symbol": np.asarray(symbols, dtype=np.int64),
            },
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the panel for the whole configured window."""
        symbols, _ = self._raw_axes_in_range()
        return self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=symbols
        )

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count the cells each added permaticker shows between ``start`` and ``end``.

        A permaticker whose first row is released after the store's last
        day shows nothing in it, so it is a new filer and widens the store.
        """
        if not added:
            return {}
        window = self._raw_data_to_xr_window(start, end, symbols=[int(s) for s in added])
        counts = window["release_date"].notnull().sum("timestamp")
        return {
            str(symbol): int(count)
            for symbol, count in zip(window["symbol"].values.tolist(), counts.values.tolist())
            if count
        }

    def _widen_fill_values(self) -> dict:
        """Fill a new permaticker's stored days of the date variables with NaT."""
        return {name: np.datetime64("NaT", "ns") for name in DATE_VARIABLES}

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel unchanged: fundamentals are not OHLCV market data."""
        return data
