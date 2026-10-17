"""Daily panels of per-security values that hold over spells of days, on SEP's calendar.

A *spell* is a span of days over which one security holds one value: an
industry code from one SIC change to the next, or the security whose
filings carry a share class's firm values. ``SpellPanelDataset`` turns a
table of spells into a ``(timestamp, symbol)`` panel of float variables
(one by default) on SEP's trading days, NaN outside every spell and outside the security's
first and last price dates, and gives it the store machinery of a dataset:
the axes of a window, the symbols an update adds, and no OHLCV cleaning.
A subclass says what its spells are (``_build_spells``) and what the
variable's attributes are (``_variable_attrs``); a panel of several
variables names them in ``VARIABLES`` and gives each one's attributes in
``_attrs``.

The point-in-time industry (``quantlab.dataset.sharadar.industry``), the
share-class firm (``quantlab.dataset.sharadar.share_class``) and the listing
panel (``quantlab.dataset.sharadar.listing``) are built on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawDataset
from quantlab.dataset.config import DatasetConfig, SharadarDatasetConfig
from quantlab.dataset.sharadar.point_in_time import TradingCalendar
from quantlab.dataset.sharadar.universe import normalize_universe
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: Number of offending keys an error message lists.
ERROR_SAMPLE = 5

#: Stand-ins for an open end of a spell, as numpy days.
_EARLIEST = np.datetime64("0001-01-01", "D")
_LATEST = np.datetime64("9999-12-31", "D")
_ONE_DAY = np.timedelta64(1, "D")


class SpellPanelDataset(RawDataset):
    """Daily panel of per-security values held over spells of days.

    A subclass sets ``config_cls`` (a ``SharadarDatasetConfig`` whose
    ``table`` is ``"sep"``), ``VARIABLE`` (the panel's one variable) or
    ``VARIABLES`` (its several),
    ``RAW_TABLES`` (the raw tables whose watermarks bound the calendar) and
    implements ``_build_spells`` and ``_variable_attrs``.

    Examples
    --------
    ``SharadarIndustryDataset`` is one::

        SharadarIndustryDataset(config).panel("2024-01-02", "2024-01-05")["industry"]
    """

    #: Name of the panel's one variable.
    VARIABLE: str = ""

    #: Names of the panel's variables when it has several; empty means ``(VARIABLE,)``.
    VARIABLES: tuple[str, ...] = ()

    #: Raw tables that must be complete through a day for it to be built;
    #: ``"sep"`` is always among them, being the calendar.
    RAW_TABLES: tuple[str, ...] = ("sep",)

    def _normalize_config(self, config: DatasetConfig) -> SharadarDatasetConfig:
        """Normalise as the base class does, then check the config class, table and universe.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of ``config_cls``.
        ValueError
            If ``table`` is not ``"sep"`` or a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``).
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} needs a {self.config_cls.__name__}, got "
                f"{type(config).__name__}."
            )
        if config.table != "sep":
            raise ValueError(f"{self.class_name}: table must be 'sep', got {config.table!r}.")
        return normalize_universe(config, self.class_name)

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        self._spells_cache: pl.DataFrame | None = None
        self._calendar_cache: TradingCalendar | None = None

    # -- what a subclass provides -------------------------------------------

    def _build_spells(self) -> pl.DataFrame:
        """Return every security's spells.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the permaticker), ``start`` (a date, inclusive, null
            when open), ``end`` (exclusive, null when open), ``first`` and
            ``last`` (the security's first and last price dates, null when
            unknown) and one float column per variable. A row whose value of
            a variable is null or NaN leaves that variable empty on its
            days, so one row can carry some variables and leave the others
            to other rows. Spells of one symbol must not overlap where a
            variable has a value.
        """
        raise NotImplementedError

    def _variable_attrs(self) -> dict:
        """Return the attributes of the panel's variable."""
        raise NotImplementedError

    def _variables(self) -> tuple[str, ...]:
        """Return the panel's variables: ``VARIABLES``, or ``(VARIABLE,)``."""
        return self.VARIABLES or (self.VARIABLE,)

    def _attrs(self, variable: str) -> dict:
        """Return one variable's attributes; ``_variable_attrs()`` unless overridden."""
        return self._variable_attrs()

    # -- inputs ---------------------------------------------------------------

    def _trading_calendar(self) -> TradingCalendar:
        """Return SEP's trading days through the day ``RAW_TABLES`` are complete through (cached)."""
        if self._calendar_cache is None:
            self._calendar_cache = TradingCalendar.of_sep(
                self.config.raw_data_dir_path,
                self.RAW_TABLES,
                start=self.config.start_date,
                end=self.config.end_date,
            )
        return self._calendar_cache

    def _days_in(self, start_date, end_date) -> pl.Series:
        """Return the trading days in the configured window and ``start_date``..``end_date``."""
        return self._trading_calendar().days_in(start_date, end_date)

    def _spells(self) -> pl.DataFrame:
        """Return ``_build_spells()``, built once per config (cached)."""
        if self._spells_cache is None:
            with Timer(f"{self.class_name}: build {', '.join(self._variables())} spells"):
                self._spells_cache = self._build_spells()
        return self._spells_cache

    # -- axes and windows -----------------------------------------------------

    def _listed_in(self, days: pl.Series) -> list[int]:
        """Return the permatickers priced at some point in ``days``' span, sorted."""
        securities = self._spells().select("symbol", "first", "last").unique()
        if days.len():
            securities = securities.filter(
                (pl.col("first").is_null() | (pl.col("first") <= days.max().date()))
                & (pl.col("last").is_null() | (pl.col("last") >= days.min().date()))
            )
        return sort_symbol_axis(securities.get_column("symbol").unique().to_list())

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the permatickers priced in the window, and its trading days.

        Raises
        ------
        ValueError
            If the window holds no trading day or no security is priced in it.
        """
        days = self._days_in(self.config.start_date, self.config.end_date)
        symbols = self._listed_in(days) if days.len() else []
        if not symbols:
            raise ValueError(
                f"{self.class_name}: no security priced in "
                f"[{self.config.start_date}, {self.config.end_date}] for the "
                f"configured universe."
            )
        return symbols, pd.DatetimeIndex(days.to_list())

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Return one window of the panel: every trading day by every symbol.

        A spell's value fills the days from its ``start`` (and the
        security's first price date) up to its ``end`` (and through its
        last price date).

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list of int, optional
            Permatickers to build; ``None`` is every one priced in the window.
        """
        days = self._days_in(start_date, end_date)
        if symbols is None:
            symbols = self._listed_in(days)
        symbols = [int(s) for s in symbols]
        stamps = days.to_numpy().astype("datetime64[D]")
        spells = self._spells().filter(pl.col("symbol").is_in(symbols))
        panel = {v: np.full((len(stamps), len(symbols)), np.nan) for v in self._variables()}
        if len(stamps) and spells.height:
            column = {symbol: index for index, symbol in enumerate(symbols)}
            low = np.maximum(
                _days(spells.get_column("start"), _EARLIEST),
                _days(spells.get_column("first"), _EARLIEST),
            )
            # ``last`` is a priced day, so the spell runs through it.
            high = np.minimum(
                _days(spells.get_column("end"), _LATEST),
                _days(spells.get_column("last"), _LATEST - _ONE_DAY) + _ONE_DAY,
            )
            rows_from = np.searchsorted(stamps, low, side="left")
            rows_to = np.searchsorted(stamps, high, side="left")
            columns = [column[symbol] for symbol in spells.get_column("symbol").to_list()]
            for variable, values in panel.items():
                spell_values = (
                    spells.get_column(variable).cast(pl.Float64).fill_null(np.nan).to_numpy()
                )
                for at, first, stop, value in zip(
                    columns, rows_from, rows_to, spell_values, strict=True
                ):
                    if first < stop and np.isfinite(value):
                        values[first:stop, at] = value
        return xr.Dataset(
            {
                variable: (("timestamp", "symbol"), values, self._attrs(variable))
                for variable, values in panel.items()
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
        """Count each added permaticker's days with a value between ``start`` and ``end``.

        A security first priced after the store's last day has none, so it
        widens the store.
        """
        if not added:
            return {}
        window = self._raw_data_to_xr_window(start, end, symbols=[int(s) for s in added])
        counts = np.logical_or.reduce(
            [np.isfinite(window[v].values) for v in self._variables()]
        ).sum(axis=0)
        return {
            str(symbol): int(count)
            for symbol, count in zip(window["symbol"].values.tolist(), counts, strict=True)
            if count
        }

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel unchanged: a spell value is not OHLCV market data."""
        return data


def _days(column: pl.Series, missing: np.datetime64) -> np.ndarray:
    """Return a date column as numpy days, ``missing`` where it is null."""
    days = column.cast(pl.Date).to_numpy().astype("datetime64[D]")
    return np.where(np.isnat(days), missing, days)

