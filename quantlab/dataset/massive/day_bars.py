"""The Day aggregate panel built from Massive's daily ``day_aggs`` files, on the Massive ticker axis.

``MassiveDayBarDataset`` reads the Day aggregates the Massive client keeps
in the raw tier (``quantlab.dataset.massive.raw``, data type ``day_aggs``),
one file per trading day, and returns the dense ``(timestamp, symbol)``
panel its store ``massive_day_bars_1d`` holds (#352; the two-store pattern
of the Trade bars).

- **The variables are the file's**, unchanged: ``open``, ``high``, ``low``,
  ``close``, ``volume`` and ``n_trades`` (the file's ``transactions``), as
  float64 (``DAY_BAR_VARIABLES``). Prices are raw: no split or dividend
  adjustment.
- **One bar per trading day**, labelled with its date (midnight, like
  SEP's bars). The days are those of the raw tier's files; a file whose day
  is not an XNYS session is refused.
- **The symbol axis is the Massive ticker**, exactly as the files spell it
  (``BRK.A``, ``AApB``), every ticker kept, exchange test symbols included.
  A column is a ticker, not a security: reading the store as securities
  goes through the Ticker map (``quantlab.dataset.massive.ticker_map``).
  No Sharadar data is read.
- **A ticker absent from a day's file** has NaN prices and zero
  ``volume`` and ``n_trades`` that day.
- **The store grows at its end.** ``update()`` appends the raw days after
  the store's last; a ticker first met on one of them is appended at the
  end of the symbol axis in place, with no bar over the days
  before, so a backfill and a daily update are the same operation. With
  ``end_date=None`` the range is every raw day, so the daily update extends
  the store from its own ``component.json``.

Examples
--------
Needs day-aggregate files under ``<raw_data_dir_path>/day_aggs``::

    config = MassiveDayBarsDatasetConfig(
        zarr_file_path="/data/quantlab/market/massive/massive_day_bars_1d/massive_day_bars_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/massive",
        start_date="2016-10-10",
    )
    ds = MassiveDayBarDataset(config).update(granularity="year")
    panel = ds.panel("2024-11-26", "2024-11-29")
"""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.config import DatasetConfig, MassiveDayBarsDatasetConfig
from quantlab.dataset.massive.raw import raw_days, raw_file, read_aggregates
from quantlab.dataset.stock import StockDataset
from quantlab.utils.symbol_axis import sort_symbol_axis

__all__ = ["DAY_BAR_VARIABLES", "MassiveDayBarDataset"]

#: The panel's variables, from the file's columns of the same name
#: (``n_trades`` from ``transactions``).
DAY_BAR_VARIABLES = ("open", "high", "low", "close", "volume", "n_trades")

#: The variables a ticker without a bar reads as zero; prices stay NaN.
_ZERO_WITHOUT_BAR = ("volume", "n_trades")

class MassiveDayBarDataset(StockDataset):
    """Dense Day aggregate panel on the Massive ticker axis; see the module docstring.

    Parameters
    ----------
    config : MassiveDayBarsDatasetConfig
        The raw tier, the store and the date range.

    Examples
    --------
    With ``config`` a ``MassiveDayBarsDatasetConfig`` of a raw tier on disk::

        ds = MassiveDayBarDataset(config).update(granularity="day")
        list(ds.panel("2024-11-26", "2024-11-29").data_vars)
        # ['open', 'high', 'low', 'close', 'volume', 'n_trades']
    """

    #: The config class used to rebuild the dataset from a saved ``config.json``.
    config_cls = MassiveDayBarsDatasetConfig

    #: The data type the panel is built from.
    DATA_TYPE = "day_aggs"

    #: No ``"rebuild"``: a ticker a new day brings has no bar before it, so
    #: the store is only refused or extended.
    NEW_LISTING_STRATEGIES = ("refuse", "widen")

    def _normalize_config(self, config: DatasetConfig) -> MassiveDayBarsDatasetConfig:
        """Normalise as the base class does, then refuse ``symbols``.

        Raises
        ------
        TypeError
            If ``config`` is not a ``MassiveDayBarsDatasetConfig``.
        ValueError
            If ``symbols`` is set.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.config_cls):
            raise TypeError(f"{self.class_name} needs a {self.config_cls.__name__}, got {type(config).__name__}.")
        if config.symbols is not None:
            raise ValueError(
                f"{self.class_name}: config.symbols is not selectable; got {config.symbols!r}. "
                f"The panel's symbol axis is the Massive ticker, every ticker of the files kept."
            )
        return dataclasses.replace(config)

    def _on_config_installed(self) -> None:
        """Create the session calendar and clear the per-day cache."""
        self._sessions = XnysSessionCalendar()
        self._bars: dict[date, pl.DataFrame] = {}

    @property
    def _raw_data_type(self) -> str:
        """Return ``DATA_TYPE``."""
        return self.DATA_TYPE

    def has_raw_data(self) -> bool:
        """Return whether the raw tier holds at least one day-aggregate file.

        Examples
        --------
        With ``config`` a ``MassiveDayBarsDatasetConfig`` of a raw tier on disk::

            MassiveDayBarDataset(config).has_raw_data()
            # True
        """
        return bool(raw_days(self.config.raw_data_dir_path, self.DATA_TYPE))

    # -- days -------------------------------------------------------------------

    def _days_in_range(self) -> list[date]:
        """Return the raw days inside the configured range, ascending."""
        start = None if self.config.start_date is None else date.fromisoformat(self.config.start_date[:10])
        end = None if self.config.end_date is None else date.fromisoformat(self.config.end_date[:10])
        return [
            day
            for day in raw_days(self.config.raw_data_dir_path, self.DATA_TYPE)
            if (start is None or start <= day) and (end is None or day <= end)
        ]

    def _day_bars(self, day: date) -> pl.DataFrame:
        """Return one day's file as ``symbol`` and ``DAY_BAR_VARIABLES``, read once per instance.

        Raises
        ------
        ValueError
            If ``day`` is not an XNYS session, or the file names a ticker twice.
        """
        if day in self._bars:
            return self._bars[day]
        if not self._sessions.is_session(day):
            raise ValueError(
                f"{self.class_name}: the day-aggregate file of {day} is not of an XNYS session; "
                f"refusing to place it on the trading calendar."
            )
        bars = read_aggregates(raw_file(self.config.raw_data_dir_path, self.DATA_TYPE, day)).select(
            pl.col("ticker").alias("symbol"),
            *(pl.col(name) for name in ("open", "high", "low", "close", "volume")),
            pl.col("transactions").cast(pl.Float64).alias("n_trades"),
        )
        repeated = bars.filter(pl.col("symbol").is_duplicated())["symbol"].unique().sort().to_list()
        if repeated:
            raise ValueError(f"{self.class_name}: the day-aggregate file of {day} has several rows for {repeated[:5]}.")
        self._bars[day] = bars
        return bars

    def stored_through(self) -> date | None:
        """Return the last day the store holds, from its chunk ledger, or ``None`` without one.

        Examples
        --------
        With ``config`` a ``MassiveDayBarsDatasetConfig`` of a raw tier on disk::

            MassiveDayBarDataset(config).update(granularity="day").stored_through()
            # datetime.date(2024, 11, 29)
        """
        from quantlab.dataset._support.ledger import ChunkLedger

        if not Path(self.config.zarr_file_path).exists():
            return None
        last_end = ChunkLedger(ChunkLedger.default_path(self.config.zarr_file_path)).last_end
        return None if last_end is None else pd.Timestamp(last_end).date()

    # -- the panel -------------------------------------------------------------

    def _raw_axes_in_range(self) -> tuple[list[str], pd.DatetimeIndex]:
        """Return the tickers and days of the configured range.

        Only the days after the store's last are read for their tickers: a
        stored day's tickers are on the stored axis already.

        Raises
        ------
        ValueError
            If the range has no raw day.
        """
        days = self._days_in_range()
        if not days:
            raise ValueError(
                f"{self.class_name}: no Massive day-aggregate file under "
                f"{str(Path(self.config.raw_data_dir_path) / self.DATA_TYPE)!r} in "
                f"[{self.config.start_date}, {self.config.end_date}]. Download it first."
            )
        stored = self._stored_symbol_axis(self.config.zarr_file_path)
        through = self.stored_through() if stored is not None else None
        symbols: set[str] = {str(symbol) for symbol in stored} if stored is not None else set()
        for day in days:
            if through is None or day > through:
                symbols.update(self._day_bars(day)["symbol"].to_list())
        return sort_symbol_axis(symbols), pd.DatetimeIndex([pd.Timestamp(day) for day in days])

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Return no history for the tickers an append adds.

        A ticker the store lacks is in none of its days' files: the NaN
        prices and zero volume an extension gives it there are its real
        values, so ``update`` extends the axis.
        """
        return {}

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list[str] | None = None) -> xr.Dataset:
        """Build the days of the window densely on ``symbols``.

        Parameters
        ----------
        start_date, end_date : date-like
            First and last day to include, inclusive.
        symbols : list of str, optional
            The ticker axis, in its order; ``None`` uses ``_raw_axes_in_range``'s.

        Returns
        -------
        xr.Dataset
            ``DAY_BAR_VARIABLES`` as float64 on ``(timestamp, symbol)``.
        """
        if symbols is None:
            symbols, _ = self._raw_axes_in_range()
        symbols = [str(symbol) for symbol in symbols]
        start, end = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
        days = [day for day in self._days_in_range() if start <= day <= end]
        position = {symbol: index for index, symbol in enumerate(symbols)}
        shape = (len(days), len(symbols))
        values = {
            name: np.zeros(shape) if name in _ZERO_WITHOUT_BAR else np.full(shape, np.nan)
            for name in DAY_BAR_VARIABLES
        }
        for row, day in enumerate(days):
            bars = self._day_bars(day).filter(pl.col("symbol").is_in(symbols))
            columns = np.array([position[symbol] for symbol in bars["symbol"].to_list()], dtype=np.int64)
            for name in DAY_BAR_VARIABLES:
                values[name][row, columns] = bars[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
        return xr.Dataset(
            {name: (("timestamp", "symbol"), array) for name, array in values.items()},
            coords={
                "timestamp": pd.DatetimeIndex([pd.Timestamp(day) for day in days]).values,
                "symbol": np.array(symbols, dtype=object),
            },
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Build the whole configured range as one window."""
        symbols, timestamps = self._raw_axes_in_range()
        return self._raw_data_to_xr_window(timestamps[0], timestamps[-1], symbols=symbols)

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel as the files have it: nothing is cleaned."""
        return data

    def _widen_fill_values(self) -> dict:
        """Return 0 for the volume and trade count of a ticker added to an existing store."""
        return {name: 0.0 for name in _ZERO_WITHOUT_BAR}
