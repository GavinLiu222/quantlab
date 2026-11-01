"""Sharadar DAILY valuation panel (market cap, EV, PE, PB, PS), keyed by permaticker.

DAILY holds one row per company and trading day: ``marketcap``, ``ev`` and
the valuation ratios ``evebit``, ``evebitda``, ``pb``, ``pe`` and ``ps``. The
vendor computes them from the price of the day and the most recent SEC
filing (as reported), so a row uses nothing later than its date.

``SharadarDailyDataset`` converts the raw tier into a dense
``(timestamp, symbol)`` Zarr panel on DAILY's own dates whose ``symbol``
coordinate is the int64 permaticker, as the price and fundamentals panels'
is. TICKERS has no rows of DAILY, which covers SF1's filers, so its tickers
are mapped through the SF1 rows. The conversion refuses a raw ticker TICKERS
does not know, a ticker TICKERS maps to two permatickers, and two rows of one
permaticker on one date.

The vendor writes ``marketcap`` and ``ev`` in USD millions, but SF1 writes
them in USD; the panel holds them in USD (``MILLIONS``), so merging the two
never mixes units. Each variable records its unit in its ``unit`` attribute
(``USD`` or ``ratio``), from INDICATORS. The store grows with ``update()``:
new days are appended and a stored day is never rewritten, so a value the
vendor changes later (DAILY rows are refreshed by ``lastupdated``,
``SharadarClient.updated_table``) never reaches the store.

Examples
--------
Build the store from a downloaded raw tier and read a week::

    config = SharadarDailyConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_daily_1d/sharadar_daily_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarDailyDataset(config).update()
    panel = SharadarDailyDataset(config).panel("2024-01-02", "2024-01-08")
    panel["marketcap"].attrs["unit"]  # 'USD'
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawDataset
from quantlab.dataset.config import DatasetConfig, SharadarDailyConfig
from quantlab.dataset.sharadar.tables import (
    DAILY_INDICATORS,
    raw_through,
    scan_raw_table,
    table,
)
from quantlab.dataset.sharadar.permatickers import map_raw_table
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: The variables the vendor writes in USD millions; the panel holds them in USD.
MILLIONS: tuple[str, ...] = ("ev", "marketcap")

#: The vendor's unit type of ``MILLIONS`` in INDICATORS.
VENDOR_MILLIONS_UNIT = "USD millions"

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


class SharadarDailyDataset(RawDataset):
    """Dense daily panel of Sharadar's DAILY valuations, keyed by permaticker.

    Parameters
    ----------
    config : SharadarDailyConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``daily``, ``tickers`` and ``indicators`` raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarDailyDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['ev', 'evebit', 'evebitda', 'marketcap', 'pb', 'pe', 'ps']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarDailyConfig

    def _normalize_config(self, config: DatasetConfig) -> SharadarDailyConfig:
        """Normalise as the base class does, then check the DAILY fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarDailyConfig``.
        ValueError
            If ``table`` is not ``"daily"`` or a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``).

        Examples
        --------
        A config of another table is refused::

            SharadarDailyDataset(dataclasses.replace(config, table="sep"))
            # ValueError: SharadarDailyDataset: table must be 'daily', got 'sep'.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarDailyConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarDailyConfig, got "
                f"{type(config).__name__}."
            )
        if config.table != "daily":
            raise ValueError(f"{self.class_name}: table must be 'daily', got {config.table!r}.")
        return normalize_universe(config, self.class_name)

    def _on_config_installed(self) -> None:
        """Drop the derivation cached for the previous config."""
        self._derivation_cache: pl.DataFrame | None = None

    # -- inputs ---------------------------------------------------------------

    def _units(self) -> dict[str, str]:
        """Return each variable's unit in the panel, ``MILLIONS`` converted to USD.

        Raises
        ------
        ValueError
            If INDICATORS does not give a ``MILLIONS`` variable in
            ``VENDOR_MILLIONS_UNIT``: the conversion would then be wrong.
        """
        rows = (
            scan_raw_table(self.config.raw_data_dir_path, "indicators")
            .filter(pl.col("table").is_in(table("daily").tickers_labels))
            .select("indicator", "unittype")
            .collect()
        )
        units = dict(rows.iter_rows())
        wrong = {name: units.get(name) for name in MILLIONS if units.get(name) != VENDOR_MILLIONS_UNIT}
        if wrong:
            raise ValueError(
                f"{self.class_name}: INDICATORS gives {wrong} for DAILY, not "
                f"{VENDOR_MILLIONS_UNIT!r}; refusing to convert to USD with "
                f"the wrong factor."
            )
        return {name: "USD" if name in MILLIONS else units[name] for name in DAILY_INDICATORS if name in units}

    def _derivation(self) -> pl.DataFrame:
        """Return the panel's rows for the whole configured window, cached.

        The window ends at ``config.end_date`` or at the day DAILY is complete through (``raw_through``),
        whichever is earlier.

        Returns
        -------
        pl.DataFrame
            Columns ``timestamp``, ``symbol`` (the permaticker) and
            ``DAILY_INDICATORS`` (``MILLIONS`` in USD), one row per pair.

        Raises
        ------
        ValueError
            If a ticker has two permatickers, or two rows of one ticker share
            a permaticker and date (rows under two tickers are settled,
            ``PermatickerResolver.settle``).
        """
        if self._derivation_cache is not None:
            return self._derivation_cache
        root = self.config.raw_data_dir_path
        start = datetime.fromisoformat(self.config.start_date).date()
        end = min(
            datetime.fromisoformat(self.config.end_date).date(), raw_through(root, ("daily",))
        )
        with Timer(f"{self.class_name}: derive"):
            frame = map_raw_table(
                root,
                "daily",
                owner=self.class_name,
                store_path=self.config.zarr_file_path,
                query=lambda scan: scan.filter(pl.col("date").is_between(pl.lit(start), pl.lit(end))),
                key=("date",),
            )
            frame = frame.filter(pl.col("permaticker").is_in(universe(self.config)))
            self._assert_unique_keys(frame)
            frame = frame.select(
                pl.col("date").cast(pl.Datetime("ns")).alias("timestamp"),
                pl.col("permaticker").alias("symbol"),
                *(
                    (pl.col(name).cast(pl.Float64) * (1e6 if name in MILLIONS else 1.0)).alias(name)
                    for name in DAILY_INDICATORS
                ),
            ).sort("timestamp", "symbol")
        self._derivation_cache = frame
        return frame

    def _assert_unique_keys(self, frame: pl.DataFrame) -> None:
        """Raise if two rows share a permaticker and a date."""
        duplicates = (
            frame.group_by("permaticker", "date")
            .agg(pl.col("ticker").sort())
            .filter(pl.col("ticker").list.len() > 1)
            .sort("permaticker", "date")
        )
        if duplicates.height:
            sample = [
                f"{row['permaticker']}@{row['date']}:{row['ticker']}"
                for row in duplicates.head(_ERROR_SAMPLE).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {duplicates.height} (permaticker, date) "
                f"pair(s) have several raw rows, first {sample}; refusing "
                f"rather than choosing one."
            )

    # -- axes and windows -------------------------------------------------------

    def _rows_to_build(self) -> pl.DataFrame:
        """Return the derivation, refusing an empty one: a build never writes an empty store."""
        derivation = self._derivation()
        if derivation.height == 0:
            raise ValueError(
                f"{self.class_name}: no DAILY row in [{self.config.start_date}, "
                f"{self.config.end_date}] for the configured universe."
            )
        return derivation

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the permatickers (sorted) and the dates of the derivation."""
        derivation = self._rows_to_build()
        symbols = sort_symbol_axis(derivation.get_column("symbol").unique().to_list())
        timestamps = derivation.get_column("timestamp").unique().sort().to_list()
        return symbols, pd.DatetimeIndex(timestamps)

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Return one window of the derivation as a dense panel, each variable with its unit.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list of int, optional
            Permatickers to reindex onto; ``None`` keeps those in the window.
        """
        window = self._derivation().filter(
            pl.col("timestamp").is_between(
                pl.lit(pd.Timestamp(start_date).to_datetime64()),
                pl.lit(pd.Timestamp(end_date).to_datetime64()),
            )
        )
        if symbols is not None:
            symbols = [int(s) for s in symbols]
            window = window.filter(pl.col("symbol").is_in(symbols))
        data = window.to_pandas().set_index(["timestamp", "symbol"]).to_xarray()
        if symbols is None:
            symbols = sort_symbol_axis(data["symbol"].values.tolist())
        data = data.reindex(symbol=symbols).sortby("timestamp")
        for name, unit in self._units().items():
            data[name].attrs["unit"] = unit
        return data

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the dense panel for the whole configured window."""
        self._rows_to_build()
        return self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=None
        )

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count each added permaticker's raw rows between ``start`` and ``end``.

        Read from the derivation, so no dense window is built; see
        ``RawDataset._added_symbols_with_raw_history``.
        """
        if not added:
            return {}
        counts = (
            self._derivation()
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

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel unchanged: valuations are not OHLCV market data."""
        return data
