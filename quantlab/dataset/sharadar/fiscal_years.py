"""Point-in-time history of each company's latest fiscal years (SF1 ARY), keyed by permaticker.

SF1's ARY dimension holds one row per company, fiscal year and filing, each
dated at its SEC filing date (its *release date*); a restatement of a year
is a new row with a later date, never a rewrite. The quarterly fundamentals
panel (``quantlab.dataset.sharadar.fundamentals``) shows only the latest
period's row, so it cannot say what a company earned three years ago as
known today. ``SharadarFiscalYearsDataset`` shows that history: on every
trading day, per security, it holds the latest ``years`` fiscal years whose
rows are available by that day, newest first, as ``<indicator>_fy0`` (the
latest fiscal year) .. ``<indicator>_fy<years - 1>``.

A row is placed at its *available date*, the first SEP trading day on or
after its release date, the same rule as the fundamentals panel. So:

- a fiscal year appears from its own release date, never earlier, and its
  arrival shifts the older years one slot back;
- each slot shows its fiscal year's latest filing available by that day, so
  a restatement of any year, the newest or an older one, is shown from its
  own release date;
- a slot whose year is not yet known (fewer than ``years`` are released) is
  NaN, and its ``reportperiod_fy<k>`` is NaT;
- a security shows nothing once the row shown in ``fy0`` (the newest
  fiscal year's latest filing) was released more than ``stale_after_days``
  ago.

A fiscal year is identified by its ``reportperiod``: a company that moves
its fiscal year end has two slots a few months apart.

The calendar is SEP's trading days and the ``symbol`` axis the int64
permaticker, as for the other Sharadar panels. Besides the indicator
variables (float64, ``unit`` from INDICATORS), ``reportperiod_fy<k>`` holds
each slot's fiscal year end and ``release_date`` the filing date of the row
shown in ``fy0``. The store grows with ``update()``: new trading days are
appended and a stored day is never rewritten.

Examples
--------
Build the store from a downloaded raw tier and read a quarter::

    config = SharadarFiscalYearsConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_sf1_fiscal_years/sharadar_sf1_fiscal_years.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarFiscalYearsDataset(config).update()
    panel = SharadarFiscalYearsDataset(config).panel("2024-01-02", "2024-03-28")
    panel["eps_fy0"].attrs["unit"]  # 'USD/share'
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawDataset
from quantlab.dataset.config import DatasetConfig, SharadarFiscalYearsConfig
from quantlab.dataset.sharadar.permatickers import map_raw_table
from quantlab.dataset.sharadar.tables import (
    SF1_INDICATORS,
    raw_through,
    scan_raw_table,
    table,
    trading_days,
)
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: The SF1 dimension the panel is built from: annual, as reported.
DIMENSION = "ARY"

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


def slot_name(name: str, slot: int) -> str:
    """Return the variable name of ``name`` in fiscal-year slot ``slot``.

    Examples
    --------
    >>> slot_name("eps", 0)
    'eps_fy0'
    """
    return f"{name}_fy{slot}"


class SharadarFiscalYearsDataset(RawDataset):
    """Daily point-in-time panel of each company's latest fiscal years, keyed by permaticker.

    Parameters
    ----------
    dataset_config : SharadarFiscalYearsConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``sf1``, ``sep`` (the calendar), ``tickers`` and ``indicators`` raw
        tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarFiscalYearsDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)[:3]
        # ['eps_fy0', 'eps_fy1', 'eps_fy2']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarFiscalYearsConfig

    def _normalize_config(self, config: DatasetConfig) -> SharadarFiscalYearsConfig:
        """Normalise as the base class does, then check the fiscal-year fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarFiscalYearsConfig``.
        ValueError
            If ``table`` is not ``"sf1"``, ``indicators`` is empty, repeats a
            name or names one SF1 does not have, ``years`` or
            ``stale_after_days`` is below 1, or a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``).

        Examples
        --------
        An unknown indicator is refused::

            SharadarFiscalYearsDataset(dataclasses.replace(config, indicators=("epss",)))
            # ValueError: SharadarFiscalYearsDataset: indicators ['epss'] are not SF1 indicators.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarFiscalYearsConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarFiscalYearsConfig, got "
                f"{type(config).__name__}."
            )
        if config.table != "sf1":
            raise ValueError(f"{self.class_name}: table must be 'sf1', got {config.table!r}.")
        indicators = tuple(config.indicators)
        if not indicators or len(set(indicators)) != len(indicators):
            raise ValueError(
                f"{self.class_name}: indicators must name at least one SF1 "
                f"indicator, each once; got {indicators!r}."
            )
        unknown = [name for name in indicators if name not in SF1_INDICATORS]
        if unknown:
            raise ValueError(f"{self.class_name}: indicators {unknown} are not SF1 indicators.")
        if config.years < 1:
            raise ValueError(f"{self.class_name}: years must be at least 1, got {config.years}.")
        if config.stale_after_days is not None and config.stale_after_days < 1:
            raise ValueError(
                f"{self.class_name}: stale_after_days must be at least 1 or "
                f"None, got {config.stale_after_days}."
            )
        return normalize_universe(config, self.class_name)

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        self._states_cache: pl.DataFrame | None = None
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

    def _rows(self) -> pl.DataFrame:
        """Return the universe's ARY rows, each with its available date.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` (a trading day), ``release_date``,
            ``reportperiod`` and the configured indicators; rows released
            after the calendar's last day are dropped.

        Raises
        ------
        ValueError
            If a raw ticker has two permatickers, or two rows of one ticker
            and permaticker share a release date and fiscal year (rows under
            two tickers are settled, ``PermatickerResolver.settle``).
        """
        config = self.config
        root = config.raw_data_dir_path
        frame = map_raw_table(
            root,
            "sf1",
            owner=self.class_name,
            store_path=config.zarr_file_path,
            query=lambda scan: scan.filter(pl.col("dimension") == DIMENSION)
            .filter(pl.col("date") <= pl.lit(self._raw_through()))
            .select("ticker", "date", "reportperiod", *config.indicators, "permaticker"),
            key=("date", "reportperiod"),
        )
        frame = frame.filter(pl.col("permaticker").is_in(universe(config)))
        self._assert_unique_keys(frame)
        available = frame.select(pl.col("date").unique().sort()).join_asof(
            self._trading_days().to_frame("available").with_columns(
                pl.col("available").cast(pl.Date).alias("date")
            ),
            on="date",
            strategy="forward",
        )
        return (
            frame.join(available, on="date", how="inner")
            .filter(pl.col("available").is_not_null())
            .select(
                pl.col("permaticker").alias("symbol"),
                "available",
                pl.col("date").cast(pl.Datetime("ns")).alias("release_date"),
                pl.col("reportperiod").cast(pl.Datetime("ns")),
                *(pl.col(name).cast(pl.Float64) for name in config.indicators),
            )
        )

    def _states(self) -> pl.DataFrame:
        """Return each security's fiscal-year slots at every day they change (cached).

        A security's state changes on every available date of one of its
        rows. The state on such a day takes, per fiscal year, the latest
        filing available by then, and keeps the ``years`` latest fiscal
        years, newest in slot 0.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available``, ``release_date`` (of slot 0's row) and
            per slot ``k`` the ``reportperiod_fy<k>`` and indicator
            ``<name>_fy<k>`` columns, sorted by ``available``.
        """
        if self._states_cache is not None:
            return self._states_cache
        config = self.config
        with Timer(f"{self.class_name}: fiscal-year states"):
            rows = self._rows()
            events = rows.select("symbol", pl.col("available").alias("as_of")).unique()
            candidates = (
                events.join(rows, on="symbol", how="inner")
                .filter(pl.col("available") <= pl.col("as_of"))
                .sort("symbol", "as_of", "reportperiod", "release_date")
                .unique(subset=["symbol", "as_of", "reportperiod"], keep="last", maintain_order=True)
                .sort(
                    ["symbol", "as_of", "reportperiod"], descending=[False, False, True]
                )
                .with_columns(slot=pl.int_range(pl.len()).over("symbol", "as_of"))
                .filter(pl.col("slot") < config.years)
            )
            states = events.sort("symbol", "as_of").join(
                candidates.filter(pl.col("slot") == 0).select("symbol", "as_of", "release_date"),
                on=["symbol", "as_of"],
                how="left",
            )
            for slot in range(config.years):
                states = states.join(
                    candidates.filter(pl.col("slot") == slot).select(
                        "symbol",
                        "as_of",
                        *(
                            pl.col(name).alias(slot_name(name, slot))
                            for name in ("reportperiod", *config.indicators)
                        ),
                    ),
                    on=["symbol", "as_of"],
                    how="left",
                )
            states = states.rename({"as_of": "available"}).sort("available")
        self._states_cache = states
        return states

    def _assert_unique_keys(self, frame: pl.DataFrame) -> None:
        """Raise if two rows of one permaticker share a release date and fiscal year."""
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
                f"date, fiscal year) key(s) have several raw rows, first "
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

    def _date_variables(self) -> tuple[str, ...]:
        """Return the names of the panel's datetime variables."""
        return (
            "release_date",
            *(slot_name("reportperiod", slot) for slot in range(self.config.years)),
        )

    def _value_variables(self) -> tuple[str, ...]:
        """Return the names of the panel's indicator variables, slot by slot."""
        return tuple(
            slot_name(name, slot)
            for name in self.config.indicators
            for slot in range(self.config.years)
        )

    # -- axes and windows -------------------------------------------------------

    def _days_in(self, start_date, end_date) -> pl.Series:
        """Return the trading days in the configured window and ``start_date``..``end_date``."""
        first = max(pd.Timestamp(start_date), pd.Timestamp(self.config.start_date))
        last = min(pd.Timestamp(end_date), pd.Timestamp(self.config.end_date))
        days = self._trading_days()
        return days.filter(
            days.is_between(pl.lit(first.to_datetime64()), pl.lit(last.to_datetime64()))
        )

    def _shown_by(self, days: pl.Series) -> list[int]:
        """Return the permatickers with a row available by the last of ``days``."""
        states = self._states()
        if days.len():
            states = states.filter(pl.col("available") <= days.max())
        return sort_symbol_axis(states.get_column("symbol").unique().to_list())

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the permatickers shown by the window's end, and its trading days.

        Raises
        ------
        ValueError
            If no ARY row is available in the configured window.
        """
        days = self._days_in(self.config.start_date, self.config.end_date)
        symbols = self._shown_by(days)
        if not days.len() or not symbols:
            raise ValueError(
                f"{self.class_name}: no {DIMENSION} row is shown in "
                f"[{self.config.start_date}, {self.config.end_date}] for the "
                f"configured universe."
            )
        return symbols, pd.DatetimeIndex(days.to_list())

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
        if symbols is None:
            symbols = self._shown_by(days)
        symbols = [int(s) for s in symbols]
        grid = days.to_frame().with_row_index("_day").join(
            pl.DataFrame({"symbol": symbols}, schema={"symbol": pl.Int64}).with_row_index("_column"),
            how="cross",
        ).sort("_day", "_column")
        cells = grid.join_asof(
            self._states().filter(pl.col("symbol").is_in(symbols)),
            left_on="timestamp",
            right_on="available",
            by="symbol",
            strategy="backward",
            # Both sides are sorted by time; polars cannot check that per group.
            check_sortedness=False,
        ).sort("_day", "_column")
        names = (*self._value_variables(), *self._date_variables())
        if self.config.stale_after_days is not None:
            limit = timedelta(days=self.config.stale_after_days)
            fresh = (pl.col("timestamp") - pl.col("release_date")) <= pl.lit(limit)
            cells = cells.with_columns(
                pl.when(fresh).then(pl.col(name)).otherwise(None).alias(name)
                for name in names
            )
        shape = (days.len(), len(symbols))
        units = self._units()
        variables = {}
        for name in self.config.indicators:
            for slot in range(self.config.years):
                attrs = {"description": f"{name} of fiscal-year slot {slot} (0 is the latest)"}
                if name in units:
                    attrs["unit"] = units[name]
                variables[slot_name(name, slot)] = attrs
        for slot in range(self.config.years):
            variables[slot_name("reportperiod", slot)] = {
                "description": f"fiscal year end of slot {slot}"
            }
        variables["release_date"] = {
            "description": "SEC filing date of the row shown in slot 0"
        }
        variables = {
            name: (
                ("timestamp", "symbol"),
                cells.get_column(name).to_numpy().reshape(shape),
                attrs,
            )
            for name, attrs in variables.items()
        }
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
        return {name: np.datetime64("NaT", "ns") for name in self._date_variables()}

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel unchanged: fundamentals are not OHLCV market data."""
        return data
