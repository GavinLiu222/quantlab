"""The shared shape of a Sharadar panel of SEC filings placed at their filing date.

Some Sharadar tables are events, not states: an 8-K (EVENTS) or an insider's
Form 4 (SF2) is filed on one day and says something about that day only. A
panel of them has a value on the day a filing became public and nothing
in between. ``FilingPanelDataset`` is that shape; a subclass says which raw
rows count and how one day's filings of one security combine.

Each filing is placed at its *available date*, the first SEP trading day on
or after its filing date, so a weekend or holiday filing lands on the next
trading day and the panel lines up with the price panels bar for bar. Its
``symbol`` axis is the int64 permaticker. A day with no filing holds the
subclass's empty value (``False``, ``0.0``), never NaN: nothing was filed.
A filing made after the close lands on its filing date's bar, so a signal
formed at a bar's close should read the panel one bar back.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawDataset
from quantlab.dataset.config import DatasetConfig, SharadarDatasetConfig
from quantlab.dataset.sharadar.tables import (
    raw_through,
    trading_days,
)
from quantlab.dataset.sharadar.permatickers import map_raw_table
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer


class FilingPanelDataset(RawDataset):
    """Daily panel of one Sharadar filing table, each filing at its available date.

    A subclass sets ``config_cls`` and ``TABLE`` and implements
    ``_empty`` (each variable's value on a day without a filing),
    ``_filings`` (the raw rows that count, with their values) and
    ``_combine`` (how one day's rows of one security add up).

    Parameters
    ----------
    dataset_config : SharadarDatasetConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        filing table, ``sep`` (the calendar) and ``tickers``.
    """

    #: Code of the filing table (a key of ``quantlab.dataset.sharadar.tables.TABLES``).
    TABLE: str

    #: The columns naming one row of a security, whose rows under two of its
    #: tickers are settled (``PermatickerResolver.settle``); ``None`` when a
    #: security has several rows a day (SF2's filing lines).
    KEY: tuple[str, ...] | None = None

    def _normalize_config(self, config: DatasetConfig) -> SharadarDatasetConfig:
        """Normalise as the base class does, then check the table and the universe fields.

        Raises
        ------
        TypeError
            If ``config`` is not this dataset's ``config_cls``.
        ValueError
            If ``table`` is not ``TABLE`` or a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``).
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} needs a {self.config_cls.__name__}, got "
                f"{type(config).__name__}."
            )
        if config.table != self.TABLE:
            raise ValueError(
                f"{self.class_name}: table must be {self.TABLE!r}, got {config.table!r}."
            )
        return normalize_universe(config, self.class_name)

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        self._placed_cache: pl.DataFrame | None = None
        self._days_cache: pl.Series | None = None

    # -- the subclass's part --------------------------------------------------

    def _empty(self) -> dict[str, object]:
        """Return each variable's value on a day without a filing, in variable order."""
        raise NotImplementedError

    def _filings(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Return the raw rows that count, with their values.

        Parameters
        ----------
        raw : pl.DataFrame
            The filing table's raw rows in the window, with a
            ``permaticker`` column added.

        Returns
        -------
        pl.DataFrame
            ``permaticker``, ``date`` (the filing date) and one column per
            variable, any number of rows per pair.
        """
        raise NotImplementedError

    def _combine(self) -> list[pl.Expr]:
        """Return one aggregation per variable, combining a security's filings of one day."""
        raise NotImplementedError

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's attributes; none by default."""
        return {}

    # -- inputs ---------------------------------------------------------------

    def _raw_through(self) -> date:
        """Return the last day the filing table and the SEP calendar are complete through."""
        return raw_through(self.config.raw_data_dir_path, (self.TABLE, "sep"))

    def _trading_days(self) -> pl.Series:
        """Return SEP's trading days up to ``_raw_through()``, sorted, as ns datetimes (cached)."""
        if self._days_cache is None:
            self._days_cache = trading_days(self.config.raw_data_dir_path, self._raw_through())
        return self._days_cache

    def _days_in(self, start_date, end_date) -> pl.Series:
        """Return the trading days in the configured window and ``start_date``..``end_date``."""
        first = max(pd.Timestamp(start_date), pd.Timestamp(self.config.start_date))
        last = min(pd.Timestamp(end_date), pd.Timestamp(self.config.end_date))
        days = self._trading_days()
        return days.filter(
            days.is_between(pl.lit(first.to_datetime64()), pl.lit(last.to_datetime64()))
        )

    def _placed(self) -> pl.DataFrame:
        """Return every day with a filing, per security, combined (cached).

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``timestamp`` (the available date) and one column
            per variable, one row per pair, sorted by both.

        Raises
        ------
        ValueError
            If a raw ticker has two permatickers (``map_raw_table``); a row
            with none is left out and reported in ``<store>.unmapped.json``.
        """
        if self._placed_cache is not None:
            return self._placed_cache
        root = self.config.raw_data_dir_path
        days = self._trading_days()
        with Timer(f"{self.class_name}: place filings"):
            frame = map_raw_table(
                root,
                self.TABLE,
                owner=self.class_name,
                store_path=self.config.zarr_file_path,
                query=lambda scan: scan.filter(pl.col("date") <= pl.lit(self._raw_through())),
                key=self.KEY,
            )
            frame = frame.filter(pl.col("permaticker").is_in(universe(self.config)))
            filings = self._filings(frame)
            if days.len():
                # A filing before the calendar's first day has no trading
                # day of its own to land on; rolling it onto the first would
                # misdate it.
                filings = filings.filter(pl.col("date") >= days.min().date())
            calendar = days.to_frame().with_columns(pl.col("timestamp").cast(pl.Date).alias("date"))
            available = (
                filings.select(pl.col("date").unique().sort())
                .join_asof(calendar, on="date", strategy="forward")
                .filter(pl.col("timestamp").is_not_null())
            )
            placed = (
                filings.join(available, on="date", how="inner")
                .group_by(pl.col("permaticker").alias("symbol"), "timestamp")
                .agg(self._combine())
                .sort("symbol", "timestamp")
            )
        self._placed_cache = placed
        return placed

    # -- axes and windows -----------------------------------------------------

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the permatickers with a filing by the window's end, and its trading days.

        Raises
        ------
        ValueError
            If the window holds no trading day or no security has a filing by its end.
        """
        days = self._days_in(self.config.start_date, self.config.end_date)
        placed = self._placed()
        if days.len():
            placed = placed.filter(pl.col("timestamp") <= days.max())
        symbols = placed.get_column("symbol").unique().to_list()
        if not days.len() or not symbols:
            raise ValueError(
                f"{self.class_name}: no {self.TABLE!r} filing in "
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
            Permatickers to build; ``None`` is every one with a filing by
            the window's end.
        """
        days = self._days_in(start_date, end_date)
        placed = self._placed()
        if symbols is None:
            visible = placed.filter(pl.col("timestamp") <= days.max()) if days.len() else placed
            symbols = sort_symbol_axis(visible.get_column("symbol").unique().to_list())
        symbols = [int(s) for s in symbols]
        grid = days.to_frame().with_row_index("_day").join(
            pl.DataFrame({"symbol": symbols}, schema={"symbol": pl.Int64}).with_row_index("_column"),
            how="cross",
        )
        cells = (
            grid.join(placed, on=["timestamp", "symbol"], how="left")
            .sort("_day", "_column")
            .with_columns(pl.col(name).fill_null(value) for name, value in self._empty().items())
        )
        shape = (days.len(), len(symbols))
        attrs = self._attrs()
        return xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    cells.get_column(name).to_numpy().reshape(shape),
                    dict(attrs.get(name, {})),
                )
                for name in self._empty()
            },
            coords={
                "timestamp": pd.DatetimeIndex(days.to_list()),
                "symbol": pd.Index(symbols, dtype="int64"),
            },
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the panel for the whole configured window."""
        symbols, _ = self._raw_axes_in_range()
        return self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=symbols
        )

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count each added permaticker's filing days between ``start`` and ``end``.

        A security whose first filing is after the store's last day has
        none in it, so it widens the store.
        """
        if not added:
            return {}
        counts = (
            self._placed()
            .filter(
                pl.col("symbol").is_in([int(s) for s in added])
                & pl.col("timestamp").is_between(
                    pl.lit(pd.Timestamp(start).to_datetime64()),
                    pl.lit(pd.Timestamp(end).to_datetime64()),
                )
            )
            .group_by("symbol")
            .len()
        )
        return {str(symbol): int(rows) for symbol, rows in counts.iter_rows()}

    def _widen_fill_values(self) -> dict:
        """Fill a new permaticker's stored days with the empty values: it filed nothing then."""
        return self._empty()

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel unchanged: filings are not OHLCV market data."""
        return data
