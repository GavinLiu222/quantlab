"""Single-symbol panel of one FRED interest-rate series, such as the 3-month T-bill rate.

FRED (Federal Reserve Economic Data, St. Louis Fed) publishes each series
as one value per observation date. ``FredAcquisition``
(``quantlab.acquisition.fred``) downloads a series into the usual daily raw
tier, ``month=YYYY-MM/`` parquet shards whose ``symbol`` is the series id
and whose ``value`` is the published number, NaN where FRED has none (a
holiday). ``FredRateDataset`` turns one series of that tier into a panel
with one symbol, the series id, on the dates FRED published a value for (a
holiday FRED lists without one is no bar: a factor merging the rate with
stock prices counts its warm-up on the merged calendar, and holiday bars
would shorten it):

- ``rate``: the value as published, in annualized percent;
- ``risk_free``: ``rate / 100 / days_per_year``, a decimal return per bar
  (per trading day with the default 252), which a factor subtracts from
  stock returns.

The panel has one symbol so that it is stored once; a factor that needs
the rate on every stock broadcasts it across symbols itself. The rate of
a day is published the next business day (H.15), so a factor that must
not look ahead lags it one bar; the dataset does not.

Examples
--------
Download DTB3 (no API key) and build its store::

    from quantlab.acquisition.config import AcquisitionConfig
    from quantlab.acquisition.fred import FredAcquisition

    FredAcquisition(AcquisitionConfig(
        market="us_equity", frequency="1d", vendor="fred",
        raw_data_dir_path="/data/quantlab/downloads/us_equity/1d/macro/fred",
        watermark_path="/data/quantlab/downloads/us_equity/1d/macro/_watermarks/fred",
        symbols=("DTB3",), start_date="1954-01-04",
    )).download()
    config = FredRateConfig(
        zarr_file_path="/data/quantlab/market/fred/fred_dtb3_1d/fred_dtb3_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/us_equity/1d/macro/fred",
    )
    FredRateDataset(config).update()
    FredRateDataset(config).panel("2024-01-02", "2024-12-31")["risk_free"]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.raw import RawDataset
from quantlab.dataset.config import DatasetConfig, FredRateConfig


class FredRateDataset(RawDataset):
    """Single-symbol daily panel of one FRED rate series: ``rate`` and ``risk_free``.

    Parameters
    ----------
    config : FredRateConfig
        ``raw_data_dir_path`` is the FRED raw tier; ``series`` picks the
        series, which is also the panel's one symbol.

    Examples
    --------
    With ``config`` as in the module example::

        panel = FredRateDataset(config).update().panel("2024-01-02", "2024-01-05")
        panel["symbol"].values.tolist()  # ['DTB3']
        panel["rate"].attrs["unit"]      # 'percent per annum'
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = FredRateConfig

    def _normalize_config(self, config: DatasetConfig) -> FredRateConfig:
        """Normalise as the base class does, then check the FRED fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``FredRateConfig``.
        ValueError
            If ``series`` is empty, ``days_per_year`` is below 1 or
            ``symbols`` is set (the one symbol is the series).

        Examples
        --------
        >>> FredRateDataset(dataclasses.replace(config, days_per_year=0))
        Traceback (most recent call last):
            ...
        ValueError: FredRateDataset: days_per_year must be at least 1, got 0.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, FredRateConfig):
            raise TypeError(
                f"{self.class_name} needs a FredRateConfig, got {type(config).__name__}."
            )
        if not config.series:
            raise ValueError(f"{self.class_name}: series must name a FRED series.")
        if config.days_per_year < 1:
            raise ValueError(
                f"{self.class_name}: days_per_year must be at least 1, got "
                f"{config.days_per_year}."
            )
        if config.symbols is not None:
            raise ValueError(
                f"{self.class_name}: the panel's one symbol is the series; "
                f"set series, not symbols."
            )
        return config

    def _observations(self) -> pl.DataFrame:
        """Return the series' observations, one per date.

        A refresh fetches its first day again into a new shard, so a date
        can sit in several shards; a value is kept over a missing one, and
        of two values the one in the later shard file name (any of them: the
        same day fetched twice holds the same published number).

        Raises
        ------
        ValueError
            If the raw tier holds no row of the series.
        """
        root = Path(self.config.raw_data_dir_path)
        files = sorted(root.glob("month=*/*.pqt"))
        frame = (
            pl.scan_parquet(files, include_file_paths="_file")
            .filter(pl.col("symbol") == self.config.series)
            .select("timestamp", "value", "_file")
            .collect()
            if files
            else pl.DataFrame()
        )
        if frame.is_empty():
            raise ValueError(
                f"{self.class_name}: the raw tier at {root} holds no row of "
                f"series {self.config.series!r}; download it with FredAcquisition."
            )
        return (
            frame.sort("timestamp", pl.col("value").is_not_null(), "_file")
            .unique(subset="timestamp", keep="last", maintain_order=True)
            .select(pl.col("timestamp").cast(pl.Datetime("ns")), "value")
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the panel over the configured window."""
        config = self.config
        frame = self._observations().filter(pl.col("value").is_not_null() & pl.col("value").is_not_nan())
        if config.start_date is not None:
            frame = frame.filter(pl.col("timestamp") >= pd.Timestamp(config.start_date))
        if config.end_date is not None:
            frame = frame.filter(pl.col("timestamp") <= pd.Timestamp(config.end_date))
        rate = frame.get_column("value").cast(pl.Float64).to_numpy()
        return xr.Dataset(
            {
                "rate": (
                    ("timestamp", "symbol"),
                    rate[:, None],
                    {
                        "unit": "percent per annum",
                        "description": f"FRED {config.series} as published",
                    },
                ),
                "risk_free": (
                    ("timestamp", "symbol"),
                    rate[:, None] / 100.0 / config.days_per_year,
                    {
                        "unit": "decimal return per trading day",
                        "description": f"rate / 100 / {config.days_per_year}",
                    },
                ),
            },
            coords={
                "timestamp": pd.DatetimeIndex(frame.get_column("timestamp").to_list()),
                "symbol": np.asarray([config.series], dtype=object),
            },
        )

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel unchanged: a rate series is not OHLCV market data."""
        return data
