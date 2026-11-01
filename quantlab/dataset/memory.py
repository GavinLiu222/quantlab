"""Use pandas, polars or xarray data in the research pipeline.

``FrameDataset`` holds market data in memory and supports the same panel
queries as stored datasets. It can supply data to factors, labels and
backtests without an initial download or conversion step.
"""

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.dataset.config import FrameDatasetConfig
from quantlab.dataset.base import MarketDataset
from quantlab.dataset._support.frame import to_panel

class FrameDataset(MarketDataset):
    """Use a DataFrame or xarray panel as an in-memory market dataset.

    Pass long-form data with ``timestamp`` and ``symbol`` keys. Each remaining
    column becomes a panel variable under the same name. The ``columns``
    argument renames input columns first; a pandas MultiIndex with the two
    key names is also accepted.

    Duplicate key pairs raise an error. Missing cells become NaN, symbols
    become strings, and timezone-aware timestamps become naive UTC.

    Construction from a frame or panel performs no disk I/O. Alternatively,
    pass ``FrameDatasetConfig`` to load an existing Zarr store into memory.
    ``to_zarr`` saves the held panel.
    Stream-mode factors are not supported. For stock factors, use the variable
    names they expect, such as ``adjOpen`` and ``adjClose``; ``quantlab.api``
    handles this renaming when using its predefined stock factor names.

    Parameters
    ----------
    data : pandas.DataFrame, polars.DataFrame, xarray.Dataset or FrameDatasetConfig
        The bars, as a long frame or a panel on ``(timestamp, symbol)``; or a config
        naming the Zarr store to read them from.
    columns : mapping of str to str, optional
        Renames the frame's columns before conversion, ``{caller_name: name}``. Not
        accepted with a config, whose store is already in the library's names.

    Raises
    ------
    TypeError
        If ``data`` is not a pandas or polars frame, an xarray panel or a
        ``FrameDatasetConfig``.
    ValueError
        If ``timestamp`` or ``symbol`` is missing after renaming, ``columns`` names an
        absent column, a ``(timestamp, symbol)`` pair repeats, or ``data`` is a config
        without ``zarr_file_path`` or comes with ``columns``.
    FileNotFoundError
        If the config's store does not exist.

    Examples
    --------
    >>> import pandas as pd
    >>> from quantlab.dataset.memory import FrameDataset
    >>> frame = pd.DataFrame({
    ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
    ...     "symbol": ["AAA", "BBB", "AAA"],
    ...     "close": [10.0, 20.0, 11.0],
    ... })
    >>> ds = FrameDataset(frame)
    >>> ds.panel("2024-01-02", "2024-01-03")["close"].values
    array([[10., 20.],
           [11., nan]])
    >>> ds.bar_before("2024-01-03", 1)
    Timestamp('2024-01-02 00:00:00')

    Read back from a store:

    >>> import tempfile
    >>> from pathlib import Path
    >>> from quantlab.dataset.config import FrameDatasetConfig
    >>> path = str(Path(tempfile.mkdtemp()) / "bars.zarr")
    >>> _ = ds.to_zarr(path)
    >>> FrameDataset(FrameDatasetConfig(zarr_file_path=path)).panel(
    ...     "2024-01-02", "2024-01-03")["close"].values
    array([[10., 20.],
           [11., nan]])
    """

    # Narrower type annotation for readers and type checkers only.
    config: FrameDatasetConfig

    #: The config class of this dataset.
    config_cls = FrameDatasetConfig

    def __init__(self, data, *, columns: Mapping[str, str] | None = None):
        """Convert and hold ``data``, or read the store its config names.

        See the class docstring for parameters.
        """
        if isinstance(data, FrameDatasetConfig):
            self._init_from_store(data, columns)
            return
        panel = to_panel(data, columns=columns, purpose="FrameDataset").load()
        super().__init__(FrameDatasetConfig())
        self._panel = XrBackend().to_internal(panel).get_xarray_dataset(["timestamp", "symbol"])

    def _init_from_store(
        self, config: FrameDatasetConfig, columns: Mapping[str, str] | None
    ) -> None:
        """Read the store ``config`` names into memory.

        Raises
        ------
        ValueError
            If ``columns`` is given or the config names no store.
        FileNotFoundError
            If the store does not exist.
        """
        if columns is not None:
            raise ValueError(
                f"{type(self).__name__}: columns= renames a frame's columns; a "
                f"FrameDatasetConfig names a store already in the library's names, so "
                f"pass no columns with it."
            )
        if config.zarr_file_path is None:
            raise ValueError(
                f"{type(self).__name__}: a FrameDatasetConfig without zarr_file_path "
                f"names no store to read; pass the frame or panel itself instead."
            )
        super().__init__(config)
        self._panel = XrBackend().read(str(config.zarr_file_path)).get_xarray_dataset(
            ["timestamp", "symbol"]
        ).load()

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a ``FrameDataset`` with an equal config and panel.

        The panel is compared too, since two frames give the same config.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> ds == FrameDataset(frame.copy()), ds == FrameDataset(frame.assign(close=0.0))
        (True, False)
        """
        if type(other) is not type(self):
            return NotImplemented
        return self.config == other.config and self._held().equals(other._held())

    def _held(self) -> xr.Dataset:
        """Return the panel this dataset holds."""
        return self._panel

    def to_zarr(self, path: "str | os.PathLike") -> Self:
        """Write the held panel to a new Zarr store and return a dataset reading it.

        The panel is written as held. This dataset is not changed. A backtest
        run directory's copies of held panels are written this way.

        Parameters
        ----------
        path : str or os.PathLike
            The store to create; it must not exist.

        Returns
        -------
        FrameDataset
            A dataset whose config names ``path``, read back from it.

        Raises
        ------
        FileExistsError
            If ``path`` exists; a store is never overwritten.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> path = Path(tempfile.mkdtemp()) / "bars.zarr"
        >>> on_disk = FrameDataset(frame).to_zarr(path)
        >>> on_disk.store_path == str(path)
        True
        >>> on_disk.panel("2024-01-03", "2024-01-03")["close"].values
        array([[11., nan]])
        >>> FrameDataset(frame).to_zarr(path)
        Traceback (most recent call last):
        FileExistsError: FrameDataset.to_zarr(): ... already exists; a store is never overwritten.
        """
        target = Path(path)
        if target.exists():
            raise FileExistsError(
                f"{self.class_name}.to_zarr(): {target} already exists; a store is "
                f"never overwritten."
            )
        XrBackend().to_internal(self._held()).write(str(target))
        return type(self)(
            dataclasses.replace(self.config, zarr_file_path=str(target))
        )

    def persist_with_run(self, run_dir: Path, store: str) -> dict:
        """Write the held panel into ``run_dir`` and return a config reading it.

        The panel belongs to no project store, so a backtest run directory keeps a
        copy at ``store``, written by ``to_zarr``. The returned config names it
        relative to the run directory, so the directory can be moved;
        ``resolve_run_config`` resolves it again when the run is rebuilt.

        Parameters
        ----------
        run_dir : Path
            The run directory being written.
        store : str
            Where the run keeps the copy, relative to ``run_dir``.

        Returns
        -------
        dict
            This dataset's config reading the copy, with its path relative to
            ``run_dir``.

        Raises
        ------
        FileExistsError
            If the copy exists already.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> run_dir = Path(tempfile.mkdtemp())
        >>> FrameDataset(frame).persist_with_run(run_dir, "copies/prices.zarr")["zarr_file_path"]
        'copies/prices.zarr'
        >>> FrameDataset.resolve_run_config(
        ...     {"zarr_file_path": "copies/prices.zarr"}, run_dir)["zarr_file_path"] == str(
        ...     run_dir / "copies" / "prices.zarr")
        True
        """
        config = self.to_zarr(Path(run_dir) / store).get_config()
        config["zarr_file_path"] = store
        return config

    @classmethod
    def resolve_run_config(cls, config: dict, run_dir: Path | None) -> dict:
        """Return ``config`` with a store named relative to ``run_dir`` made absolute.

        A relative ``zarr_file_path`` is one ``persist_with_run`` recorded; it is
        resolved against ``run_dir`` and never against the working directory. An
        absolute path, or none, is left as it is. ``config`` is not modified.

        Parameters
        ----------
        config : dict
            A recorded ``FrameDataset`` config.
        run_dir : Path or None
            The run directory the config was read from.

        Returns
        -------
        dict
            The config to construct the dataset from.

        Raises
        ------
        ValueError
            If the path is relative and ``run_dir`` is ``None``.

        Examples
        --------
        >>> from quantlab.dataset.memory import FrameDataset
        >>> FrameDataset.resolve_run_config(
        ...     {"zarr_file_path": "copies/prices.zarr"}, "/runs/WeightsVectorBt_1"
        ... )
        {'zarr_file_path': '/runs/WeightsVectorBt_1/copies/prices.zarr'}
        >>> FrameDataset.resolve_run_config(
        ...     {"zarr_file_path": "copies/prices.zarr"}, None)
        Traceback (most recent call last):
        ValueError: quantlab.dataset.memory.FrameDataset reads the store 'copies/prices.zarr', ...
        """
        path = config.get("zarr_file_path")
        if path is None or Path(path).is_absolute():
            return config
        if run_dir is None:
            name = config.get("name") or f"{cls.__module__}.{cls.__qualname__}"
            raise ValueError(
                f"{name} reads the store {path!r}, which is relative to the run "
                f"directory the config was saved in; pass run_dir= (the directory "
                f"holding config.json) to rebuild it. It is never resolved against the "
                f"working directory."
            )
        return {**config, "zarr_file_path": str(Path(run_dir) / path)}

    def _open_bars(self, start, end) -> xr.Dataset:
        """Return the held bars from ``start`` to ``end``: the read seam's bars (``BaseDataset.panel``)."""
        return self._held().sel(timestamp=slice(start, end))

    def _calendar(self) -> pd.DatetimeIndex:
        """Return the held panel's timestamps."""
        return pd.DatetimeIndex(self._held()["timestamp"].values)

    def _read_store(self) -> None:
        """Return ``None``: requests read the panel held in memory, never a store."""
        return None

    def head(self, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows of the held panel as a ``LazyFrame``.

        Parameters
        ----------
        n : int
            Maximum number of rows.

        Examples
        --------
        >>> import pandas as pd
        >>> from quantlab.dataset.memory import FrameDataset
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        ...     "symbol": ["AAA", "BBB", "AAA"], "close": [10.0, 20.0, 11.0]})
        >>> ds = FrameDataset(frame)
        >>> ds.head(2).collect().columns
        ['timestamp', 'symbol', 'close']
        """
        held = self._held()
        bounded = held.isel(timestamp=slice(0, n), symbol=slice(0, n))
        return XrBackend().to_internal(bounded).get_lazyframe().head(n)

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns under their own names."""
        return self._kunquant_arrays(data, data_columns)
