"""Binance spot klines loaded as a ``(timestamp, symbol)`` panel.

A *kline* (candlestick) is one bar of open, high, low, close and volume
(OHLCV) for a trading pair over a fixed interval. A *panel* is an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, the format every
quantlab layer exchanges.

``SpotKlineDataset`` reads the monthly, header-less CSV files that Binance
publishes for spot markets, stacks them into one panel per configured date
range, and exposes the KunQuant exit of ``MarketDataset``. It is
the crypto counterpart of ``quantlab/dataset/stock.py``.
"""

from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawMarketDataset
from quantlab.dataset._support.cleaning import dedup_raw_frame, flag_anomalies, validate_schema
from quantlab.enums.data import BinanceCSVHeaders
from quantlab.dataset._support.file import file_date_filter, get_csv_files
from quantlab.utils.timer import Timer


class SpotKlineDataset(RawMarketDataset):
    """Binance spot kline dataset built from monthly CSV files.

    The raw directory holds one header-less CSV per symbol and month, named
    ``<SYMBOL>-...csv``. The column names come from
    ``BinanceCSVHeaders.SPOT`` and the symbol comes from the file name.
    Columns keep Binance's Title-Case spelling (``Open``, ``High``, ...);
    ``COLUMN_MAP`` renames them to the shared lowercase names in
    ``to_kunquant`` and in a merge.

    Parameters
    ----------
    config : DatasetConfig
        Paths (raw CSV directory, Zarr store, Nautilus catalog), market,
        frequency and date range of the dataset.

    Examples
    --------
    >>> config = DatasetConfig(
    ...     raw_data_dir_path="downloads/crypto_spot/1d/klines",
    ...     zarr_file_path="data/crypto_spot/1d/spot.zarr",
    ...     market="crypto_spot",
    ...     frequency="1d",
    ... )
    >>> SpotKlineDataset(config).from_raw_data().save()
    >>> panel = SpotKlineDataset(config).panel("2024-01-01", "2024-12-31")
    """

    # Binance columns are Title-Case, so the schema check cannot use the
    # shared lowercase default names; ``_clean`` passes these instead.
    _RAW_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

    #: ``Quote asset volume`` (traded value in the quote currency) becomes
    #: ``amount`` and the OHLCV columns are lowercased.
    COLUMN_MAP = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Quote asset volume": "amount",
    }

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Check the Title-Case OHLCV columns exist and add ``anomaly_flag``.

        ``flag_anomalies`` only inspects lowercase price columns, so on this
        panel the flag is always ``False``. It is still added so every
        dataset carries the same set of variables.

        Parameters
        ----------
        data : xr.Dataset
            The raw panel.

        Returns
        -------
        xr.Dataset
            The same panel with an ``anomaly_flag`` variable.
        """
        data = validate_schema(data, required_columns=self._RAW_REQUIRED_COLUMNS)
        data = flag_anomalies(data)
        return data

    def _raw_data_to_xr_window(
        self,
        start_date,
        end_date,
        symbols: list[str] | None = None,
    ) -> xr.Dataset:
        """Return the dense panel for one time window.

        This is a non-memory-bounded implementation. It converts the whole
        configured range with ``_raw_data_to_xr`` and slices afterwards, so
        a chunked build limits the size of each write but not of the
        conversion, and repeats the full conversion for every window. A
        bounded version would filter the monthly file list by date with
        ``file_date_filter`` before loading, the way ``StockDataset``
        filters inside its parquet scan.

        Parameters
        ----------
        start_date, end_date : date-like
            Inclusive bounds of the window.
        symbols : list of str, optional
            If given, the result has exactly these symbols, in this order. A
            symbol with no row in the window becomes an all-NaN column
            instead of being dropped.

        Returns
        -------
        xr.Dataset
            The panel restricted to the window.
        """
        data = self._raw_data_to_xr()
        data = data.sel(timestamp=slice(start_date, end_date))
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    @staticmethod
    def _spot_kline_to_df(csv_file: Path, before_2025: bool) -> pl.LazyFrame:
        """Scan one monthly kline CSV into a ``LazyFrame`` with a ``symbol`` column.

        Binance changed the unit of ``Open time`` from milliseconds to
        microseconds in 2025, so ``before_2025`` picks the right parse. Both
        paths return microsecond timestamps. The symbol is the part of the
        file name before the first dash.

        Parameters
        ----------
        csv_file : Path
            One monthly kline CSV file.
        before_2025 : bool
            True if the file covers a month before 2025 (millisecond epochs).

        Returns
        -------
        pl.LazyFrame
            The file's rows with a parsed ``Open time`` and a ``symbol`` column.
        """
        df = pl.scan_csv(
            str(csv_file), has_header=False, new_columns=BinanceCSVHeaders.SPOT
        )

        if before_2025:
            df = df.with_columns(
                pl.from_epoch("Open time", time_unit="ms")
                .dt.cast_time_unit("us")
                .alias("Open time")
            )
        else:
            df = df.with_columns(
                pl.from_epoch("Open time", time_unit="us").alias("Open time")
            )
        df = df.with_columns(
            pl.lit(str(csv_file.name).split("-")[0]).alias("symbol"),
        )
        return df

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Stack the CSV files in the configured date range into a dense panel.

        Files are selected by the date in their name, concatenated and sorted.
        Duplicate ``(timestamp, symbol)`` rows keep the last occurrence. The
        result is then turned into a dense panel with pandas ``to_xarray``,
        so missing ``(timestamp, symbol)`` cells become NaN.

        Returns
        -------
        xr.Dataset
            The dense panel for the configured date range.

        Raises
        ------
        ValueError
            If no CSV file in the raw directory matches the range.
        """
        with Timer(f" {self.__class__.__name__}: from csv"):
            csv_files = get_csv_files(self.config.raw_data_dir_path)
            csv_files = file_date_filter(
                csv_files,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            csv_before_2025 = file_date_filter(csv_files, end_date="2024-12-31")
            csv_after_2025 = file_date_filter(
                csv_files, start_date="2025-01-01"
            )

            dfs: list[pl.LazyFrame] = []
            if csv_after_2025:
                dfs.extend(
                    self._spot_kline_to_df(csv, before_2025=False)
                    for csv in csv_after_2025
                )
            if csv_before_2025:
                dfs.extend(
                    self._spot_kline_to_df(csv, before_2025=True)
                    for csv in csv_before_2025
                )
            if not dfs:
                raise ValueError(
                    f"No CSV file matching the configured date range was found "
                    f"under {self.config.raw_data_dir_path}"
                )

            res = pl.concat(dfs)
            res = res.rename({"Open time": "timestamp"})
            res = res.sort(by=["timestamp", "symbol"])
            res = dedup_raw_frame(res, keep="last")
            res = res.collect().to_pandas().set_index(["timestamp", "symbol"])
            return res.to_xarray()

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Rename Binance columns by ``COLUMN_MAP`` and export float32 arrays.

        Parameters
        ----------
        data : xr.Dataset
            The panel to export.
        data_columns : tuple of str
            KunQuant input names to export, e.g. ``("open", "close")``.

        Returns
        -------
        input_dict : dict of str to np.ndarray
            One C-contiguous float32 array of shape ``[time, symbol]`` per
            requested column.
        symbols : np.ndarray
            Symbol labels of the second axis.
        timestamp : np.ndarray
            Timestamps of the first axis.
        """
        with Timer(f"{self.__class__.__name__}: to kunquant"):
            return self._kunquant_arrays(self.to_shared_names(data), data_columns)
