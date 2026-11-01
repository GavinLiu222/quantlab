"""A factor store's panel read as a dataset's columns.

``FactorStoreDataset`` reads a stored factor panel, such as daily session
statistics of minute bars, by date range and exposes its outputs as panel
variables, optionally under a prefix, so a ``MergedDataset`` puts them
beside a price dataset's columns.
"""

import numpy as np
import polars as pl
import xarray as xr

from quantlab.dataset.base import MarketDataset
from quantlab.dataset.config import FactorStoreDatasetConfig


class FactorStoreDataset(MarketDataset):
    """Read a factor store's Zarr panel as panel variables.

    ``panel(start, end)`` opens the store lazily and returns its outputs
    over those dates, each named ``prefix + output``; ``symbols`` and
    ``variables`` select by the panel's (prefixed) names, as on any dataset.
    Merged with a price dataset through ``MergedDataset``, the outputs become
    columns beside the prices. The merge refuses a cell that holds a value
    in two inputs, while same-named columns over different dates or symbols
    merge into one, so a prefix is how two stores' outputs of the same name
    are kept apart.

    The store is read as it is. The dataset never rebuilds or asks the
    factor that wrote it, so it serves any panel on ``(timestamp, symbol)``
    whatever built it. A missing value stays NaN.

    **Look-ahead contract.** The dataset shifts nothing: the value stored at
    day t is read as day t's value, beside day t's prices. A store read
    through it must therefore hold, at day t, only what is known at day t's
    close. Aligning the values is the store builder's responsibility.

    Parameters
    ----------
    config : FactorStoreDatasetConfig
        The store's Zarr path and the optional prefix.

    Notes
    -----
    The config holds only the path and the prefix: a store is read as
    stored, with no date bounds or conversion in the config. The class
    changes only how the store is read: ``_open_bars`` and ``head`` put the
    prefix on every output, and ``_to_kunquant`` exports the columns under
    those names.

    Examples
    --------
    A store of two outputs, ``rvol`` and ``skew``, for ``AAA`` and ``BBB``
    over three days; BBB has no ``rvol`` on 2024-01-03:

    >>> import tempfile
    >>> from pathlib import Path
    >>> import pandas as pd
    >>> from quantlab.dataset.config import FactorStoreDatasetConfig
    >>> from quantlab.dataset.factor_store import FactorStoreDataset
    >>> from quantlab.dataset.memory import FrameDataset
    >>> from quantlab.dataset.merged import MergedDataset
    >>> days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    >>> path = str(Path(tempfile.mkdtemp()) / "stats.zarr")
    >>> _ = FrameDataset(pd.DataFrame({
    ...     "timestamp": days.repeat(2),
    ...     "symbol": ["AAA", "BBB"] * 3,
    ...     "rvol": [0.011, 0.020, 0.013, None, 0.012, 0.018],
    ...     "skew": [0.1, -0.2, 0.3, 0.0, -0.1, 0.2],
    ... })).to_zarr(path)
    >>> stats = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=path, prefix="hf_"))
    >>> panel = stats.panel("2024-01-02", "2024-01-03")
    >>> list(panel.data_vars)
    ['hf_rvol', 'hf_skew']
    >>> panel["hf_rvol"].to_pandas()
    symbol        AAA   BBB
    timestamp
    2024-01-02  0.011  0.02
    2024-01-03  0.013   NaN

    Merged with prices of the same days and symbols, the missing statistic
    leaves the price alone:

    >>> prices = FrameDataset(pd.DataFrame({
    ...     "timestamp": days.repeat(2),
    ...     "symbol": ["AAA", "BBB"] * 3,
    ...     "close": [10.0, 20.0, 11.0, 21.0, 12.0, 22.0],
    ... }))
    >>> merged = MergedDataset([prices, stats]).panel("2024-01-02", "2024-01-04")
    >>> list(merged.data_vars)
    ['close', 'hf_rvol', 'hf_skew']
    >>> merged["hf_rvol"].to_pandas()
    symbol        AAA    BBB
    timestamp
    2024-01-02  0.011  0.020
    2024-01-03  0.013    NaN
    2024-01-04  0.012  0.018
    >>> merged["close"].sel(timestamp="2024-01-03", symbol="BBB").item()
    21.0
    """

    # Narrower type annotation for readers and type checkers only.
    config: FactorStoreDatasetConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = FactorStoreDatasetConfig

    def head(self, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows of the store, its outputs under the prefix.

        Parameters
        ----------
        n : int
            Maximum number of rows.

        Returns
        -------
        polars.LazyFrame
            Up to ``n`` rows, with ``timestamp`` and ``symbol`` as columns and
            every output named ``prefix + output``.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from quantlab.dataset.config import FactorStoreDatasetConfig
        >>> from quantlab.dataset.factor_store import FactorStoreDataset
        >>> path = str(Path(tempfile.mkdtemp()) / "stats.zarr")
        >>> _ = xr.Dataset(
        ...     {"rvol": (("timestamp", "symbol"), [[0.011, 0.020]])},
        ...     coords={"timestamp": pd.to_datetime(["2024-01-02"]), "symbol": ["AAA", "BBB"]},
        ... ).to_zarr(path)
        >>> stats = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=path, prefix="hf_"))
        >>> stats.head(1).collect_schema().names()
        ['timestamp', 'symbol', 'hf_rvol']
        """
        frame = super().head(n)
        names = [c for c in frame.collect_schema().names() if c not in ("timestamp", "symbol")]
        return frame.rename(self._prefixed(names))

    def _prefixed(self, names) -> dict[str, str]:
        """Return ``{stored name: panel name}`` for the store's ``names``."""
        return {name: f"{self.config.prefix}{name}" for name in names}

    def _open_bars(self, start, end) -> xr.Dataset:
        """Return the store's bars from ``start`` to ``end`` lazily, under the prefix."""
        data = super()._open_bars(start, end)
        return data.rename(self._prefixed(data.data_vars))

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns under their panel names."""
        return self._kunquant_arrays(data, data_columns)
