"""Sharadar 13F institutional ownership (SF3A) as a daily panel, keyed by permaticker.

Institutions file their 13F holdings up to 45 days after each calendar
quarter's end. SF3 holds every holding (security, investor, security type,
quarter); SF3A is the vendor's sum of SF3 by security and quarter, and is
what this panel reads: ``shrholders`` (institutions holding the common
stock) and ``shrunits`` (their shares, in thousands). On the 2026-10-05
pull, SF3A equals SF3 summed by hand up to rounding.

SF3 has no filing date, and a quarter's rows fill in as the filings arrive:
40 days after a quarter end, SF3A's newest quarter can hold a few dozen
holders of a stock that has six thousand. ``SharadarHoldingsDataset``
therefore shows a quarter from its *available date*, the first SEP trading
day on or after quarter end + 45 days (``AVAILABILITY_LAG``), until the
next quarter's available date. So a partial quarter is never shown, and a
security with no row in the shown quarter shows nothing (NaN) rather than
an older quarter's count. Variables:

- ``holders``: institutions holding the stock;
- ``shares_held``: the shares they hold (``shrunits`` times 1,000);
- ``quarter_end``: the quarter shown.

The ``symbol`` axis is the int64 permaticker. SF3 keys holdings by the
security's ticker, which TICKERS lists only under the price tables (its
``SF3B`` rows are investors), so tickers are mapped through SEP's rows; a
ticker SEP does not know (a fund, or a CUSIP Sharadar has no prices for)
is left out and counted in the log. SF3A is small and has no
``lastupdated``: ``update.py`` pulls it whole every run, and ``update()``
appends the new trading days.

A rebuilt store shows each quarter as the vendor holds it now, filings made
after the 45 days included; an updated store shows it as it stood on the
update. Both show a quarter only from its available date.

Examples
--------
Build the store and read the holders of a stock::

    config = SharadarHoldingsConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_holdings_1d/sharadar_holdings_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarHoldingsDataset(config).update()
    panel = SharadarHoldingsDataset(config).panel("2024-01-02", "2024-12-31")
    panel["holders"].sel(symbol=199059)  # AAPL
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawDataset
from quantlab.dataset.config import DatasetConfig, SharadarHoldingsConfig
from quantlab.dataset.sharadar.permatickers import map_raw_table
from quantlab.dataset.sharadar.tables import (
    raw_through,
    trading_days,
)
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: Days after a quarter's end until its 13F filings are all due.
AVAILABILITY_LAG = timedelta(days=45)

#: The panel's variables and their attributes.
VARIABLES: dict[str, dict] = {
    "holders": {"unit": "institutions", "description": "13F filers holding the common stock"},
    "shares_held": {"unit": "shares", "description": "common shares 13F filers hold"},
    "quarter_end": {"description": "calendar quarter end of the 13F holdings shown"},
}

#: SF3A writes shares in thousands.
UNITS_SCALE = 1_000.0

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


class SharadarHoldingsDataset(RawDataset):
    """Daily panel of 13F holders and shares held, each quarter from quarter end + 45 days.

    Parameters
    ----------
    dataset_config : SharadarHoldingsConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``sf3a``, ``sep`` (the calendar) and ``tickers`` raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarHoldingsDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['holders', 'quarter_end', 'shares_held']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarHoldingsConfig

    def _normalize_config(self, config: DatasetConfig) -> SharadarHoldingsConfig:
        """Normalise as the base class does, then check the table and the universe fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarHoldingsConfig``.
        ValueError
            If ``table`` is not ``"sf3a"`` or a universe field is invalid.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarHoldingsConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarHoldingsConfig, got "
                f"{type(config).__name__}."
            )
        if config.table != "sf3a":
            raise ValueError(f"{self.class_name}: table must be 'sf3a', got {config.table!r}.")
        return normalize_universe(config, self.class_name)

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        self._quarters_cache: pl.DataFrame | None = None
        self._days_cache: pl.Series | None = None

    # -- inputs ---------------------------------------------------------------

    def _raw_through(self) -> date:
        """Return the last day SF3A and the SEP calendar are complete through."""
        return raw_through(self.config.raw_data_dir_path, ("sf3a", "sep"))

    def _trading_days(self) -> pl.Series:
        """Return SEP's trading days up to ``_raw_through()``, sorted, as ns datetimes (cached)."""
        if self._days_cache is None:
            self._days_cache = trading_days(self.config.raw_data_dir_path, self._raw_through())
        return self._days_cache

    def _quarters(self) -> pl.DataFrame:
        """Return every security's holdings per quarter, with the quarter's available date (cached).

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``quarter_end``, ``available`` (a trading day, or
            null past the calendar), ``holders`` and ``shares_held``.

        Raises
        ------
        ValueError
            If two rows of one ticker and permaticker share a quarter; rows
            of two tickers of one security are settled first
            (``PermatickerResolver.settle``).
        """
        if self._quarters_cache is not None:
            return self._quarters_cache
        root = self.config.raw_data_dir_path
        with Timer(f"{self.class_name}: quarters"):
            # SF3A holds funds and securities without Sharadar prices: their
            # rows map to nothing and are left out quietly, unreported. SF3A
            # can hold a quarter of one security under its current and a
            # former ticker: the rows are settled (``key``), and those that
            # cannot be are left out and listed in ``<store>.unmapped.json``.
            # A ticker that maps to two permatickers is still refused.
            frame = map_raw_table(
                root,
                "sf3a",
                owner=self.class_name,
                query=lambda scan: scan.select(
                    "ticker",
                    pl.col("date").str.to_date().alias("quarter_end"),
                    "shrholders",
                    "shrunits",
                    "permaticker",
                ),
                store_path=self.config.zarr_file_path,
                date_column="quarter_end",
                key=("quarter_end",),
                quiet=True,
                report_unmapped=False,
            )
            frame = frame.filter(pl.col("permaticker").is_in(universe(self.config)))
            self._assert_unique_keys(frame)
            calendar = self._trading_days().to_frame("available").with_columns(
                pl.col("available").cast(pl.Date).alias("_due")
            )
            due = (
                frame.select(pl.col("quarter_end").unique().sort())
                .with_columns((pl.col("quarter_end") + AVAILABILITY_LAG).alias("_due"))
                .sort("_due")
                .join_asof(calendar, on="_due", strategy="forward")
                .drop("_due")
            )
            quarters = frame.join(due, on="quarter_end").select(
                pl.col("permaticker").alias("symbol"),
                pl.col("quarter_end").cast(pl.Datetime("ns")),
                "available",
                pl.col("shrholders").cast(pl.Float64).alias("holders"),
                (pl.col("shrunits") * UNITS_SCALE).alias("shares_held"),
            )
        self._quarters_cache = quarters
        return quarters

    def _assert_unique_keys(self, frame: pl.DataFrame) -> None:
        """Raise if two rows of one permaticker share a quarter."""
        duplicates = (
            frame.group_by("permaticker", "quarter_end")
            .agg(pl.col("ticker").sort())
            .filter(pl.col("ticker").list.len() > 1)
            .sort("permaticker", "quarter_end")
        )
        if duplicates.height:
            sample = [
                f"{row['permaticker']}@{row['quarter_end']}:{row['ticker']}"
                for row in duplicates.head(_ERROR_SAMPLE).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {duplicates.height} (permaticker, quarter) "
                f"pair(s) have several SF3A rows, first {sample}; refusing "
                f"rather than choosing one."
            )

    # -- axes and windows -----------------------------------------------------

    def _days_in(self, start_date, end_date) -> pl.Series:
        """Return the trading days in the configured window and ``start_date``..``end_date``."""
        first = max(pd.Timestamp(start_date), pd.Timestamp(self.config.start_date))
        last = min(pd.Timestamp(end_date), pd.Timestamp(self.config.end_date))
        days = self._trading_days()
        return days.filter(
            days.is_between(pl.lit(first.to_datetime64()), pl.lit(last.to_datetime64()))
        )

    def _shown_by(self, last) -> pl.DataFrame:
        """Return the quarters' rows available on or before ``last``."""
        quarters = self._quarters().filter(pl.col("available").is_not_null())
        return quarters.filter(pl.col("available") <= last) if last is not None else quarters

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the permatickers shown by the window's end, and its trading days.

        Raises
        ------
        ValueError
            If the window holds no trading day or no quarter is shown by its end.
        """
        days = self._days_in(self.config.start_date, self.config.end_date)
        shown = self._shown_by(days.max() if days.len() else None)
        symbols = shown.get_column("symbol").unique().to_list()
        if not days.len() or not symbols:
            raise ValueError(
                f"{self.class_name}: no 13F quarter is shown in "
                f"[{self.config.start_date}, {self.config.end_date}] for the "
                f"configured universe."
            )
        return sort_symbol_axis(symbols), pd.DatetimeIndex(days.to_list())

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Return one window of the panel: each day shows the latest quarter available by then.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list of int, optional
            Permatickers to build; ``None`` is every one shown by the
            window's end.
        """
        days = self._days_in(start_date, end_date)
        quarters = self._shown_by(days.max() if days.len() else None)
        if symbols is None:
            symbols = sort_symbol_axis(quarters.get_column("symbol").unique().to_list())
        symbols = [int(s) for s in symbols]
        # One quarter per day, the same for every security: the latest available.
        current = days.to_frame().join_asof(
            quarters.select("available", "quarter_end").unique().sort("available"),
            left_on="timestamp",
            right_on="available",
            strategy="backward",
        ).select("timestamp", "quarter_end")
        grid = (
            current.with_row_index("_day")
            .join(
                pl.DataFrame({"symbol": symbols}, schema={"symbol": pl.Int64}).with_row_index("_column"),
                how="cross",
            )
        )
        cells = grid.join(
            quarters.select("symbol", "quarter_end", "holders", "shares_held"),
            on=["symbol", "quarter_end"],
            how="left",
        ).sort("_day", "_column")
        shown = cells.get_column("holders").is_not_null()
        cells = cells.with_columns(
            pl.when(shown).then(pl.col("quarter_end")).alias("quarter_end")
        )
        shape = (days.len(), len(symbols))
        return xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    cells.get_column(name).to_numpy().reshape(shape),
                    dict(attrs),
                )
                for name, attrs in VARIABLES.items()
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
        """Count, per added permaticker, its quarters shown between ``start`` and ``end``.

        A quarter shown from before ``start`` counts too: it is still shown on ``start``.
        """
        if not added:
            return {}
        quarters = self._shown_by(pd.Timestamp(end).to_datetime64())
        starts = quarters.select("available").unique().sort("available").get_column("available")
        # The quarter shown on ``start`` is the latest available by then.
        # polars compares a Datetime series with a Python datetime, not a numpy datetime64.
        first = starts.filter(starts <= pd.Timestamp(start).to_pydatetime()).max()
        counts = (
            quarters.filter(
                pl.col("symbol").is_in([int(s) for s in added])
                & (pl.col("available") >= (first if first is not None else starts.min()))
            )
            .group_by("symbol")
            .len()
        )
        return {str(symbol): int(rows) for symbol, rows in counts.iter_rows()}

    def _widen_fill_values(self) -> dict:
        """Fill a new permaticker's stored days of ``quarter_end`` with NaT."""
        return {"quarter_end": np.datetime64("NaT", "ns")}

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel unchanged: holdings are not OHLCV market data."""
        return data
