"""The Parquet storage backend: long-format tables read lazily through polars.

``PlBackend`` holds a ``polars.LazyFrame`` and persists it as Parquet. It is used
for reference tables that are tabular rather than panel shaped (the universe
catalog in ``quantlab.universe``), and implements ``DataBackend`` from
``quantlab.backend.base``; see ``docs/user-guide/datasets.md``.
"""

from pathlib import Path
from typing import Optional, Self

import pandas as pd
import polars as pl
import xarray as xr

from quantlab.backend.base import DataBackend


class PlBackend(DataBackend):
    """Parquet-backed storage for a long-format table held as a lazy frame.

    ``data`` is a ``polars.LazyFrame`` produced by ``scan_parquet``, so reads
    and filters stay lazy until ``write`` or ``get_xarray_dataset`` collects
    them. Used for reference tables that are tabular rather than panel
    shaped.

    Examples
    --------
    >>> table = PlBackend().read("universe.parquet")
    >>> frame = table.filter_by_symbol("symbol", ("AAPL",)).get_lazyframe()
    >>> frame.collect()
    """

    def read(self, path: str, **kwargs) -> Self:
        """Lazily scan the Parquet file at ``path`` into ``data``.

        Parameters
        ----------
        path : str
            Path of the Parquet file.
        **kwargs
            Accepted for interface compatibility and ignored.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> table = PlBackend().read("universe.parquet")
        >>> type(table.data).__name__
        LazyFrame
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = pl.scan_parquet(path)
        return self

    def write(self, path: str, **kwargs) -> Self:
        """Collect ``data`` and write it to ``path`` as Parquet.

        Parameters
        ----------
        path : str
            Path of the Parquet file to write.
        **kwargs
            Passed through to ``polars.DataFrame.write_parquet``.

        Examples
        --------
        >>> PlBackend().to_internal(frame.lazy()).write("universe.parquet")
        PlBackend()
        """
        self.data.collect().write_parquet(path, **kwargs)
        return self

    def to_internal(self, data: pl.LazyFrame) -> Self:
        """Adopt an in-memory ``polars.LazyFrame`` as ``data``.

        Parameters
        ----------
        data : pl.LazyFrame
            The table to hold.

        Examples
        --------
        >>> PlBackend().to_internal(frame.lazy())
        PlBackend()
        """
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        """Narrow ``data`` in place to rows whose ``col`` lies in the range.

        Parameters
        ----------
        col : str
            The date column to filter on.
        start_date : str
            First date to keep, inclusive.
        end_date : str
            Last date to keep, inclusive.

        Examples
        --------
        >>> table.filter_by_date("timestamp", "2024-01-02", "2024-01-03")
        PlBackend()
        >>> table.get_lazyframe().collect().height  # two days of two symbols
        4
        """
        self.data = self.data.filter(
            pl.col(col).is_between(
                pl.lit(pd.to_datetime(start_date)),
                pl.lit(pd.to_datetime(end_date)),
            )
        )
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        """Narrow ``data`` in place to rows whose ``col`` is in ``symbols``.

        Parameters
        ----------
        col : str
            The column to filter on.
        symbols : tuple[str, ...]
            The values to keep.

        Examples
        --------
        >>> table = PlBackend().read("universe.parquet")
        >>> table.filter_by_symbol("symbol", ("BBB",))
        PlBackend()
        >>> table.get_lazyframe().collect()["symbol"].unique().to_list()
        ['BBB']
        """
        self.data = self.data.filter(pl.col(col).is_in(symbols))
        return self

    def get_lazyframe(self) -> pl.LazyFrame:
        """Return the held lazy frame.

        Examples
        --------
        >>> table.get_lazyframe().collect().shape
        (8, 3)
        """
        return self.data

    def resample(self, labels: pd.Series, how: dict[str, str]) -> Self:
        """Aggregate the long-format frame onto the bars ``labels`` assigns.

        Rows are grouped by their label and by every non-``timestamp``
        index column the frame has (``symbol``, in a panel). ``how`` names
        a method for every other column. The result replaces ``data``
        lazily; nothing is collected here.

        Parameters
        ----------
        labels : pd.Series
            Source timestamp to target timestamp; see ``DataBackend``.
        how : dict[str, str]
            Column name to aggregation method, one entry per value column.

        Examples
        --------
        >>> table.resample(labels, {"close": "last"})
        >>> table.get_lazyframe().collect().shape
        (2, 3)
        """
        schema = self.data.collect_schema().names()
        keys = [name for name in schema if name == "symbol"]
        missing = [name for name in schema if name not in how
                   and name not in ("timestamp", *keys)]
        if missing:
            raise ValueError(f"PlBackend.resample: no method for {missing}.")

        mapping = pl.LazyFrame(
            {
                "timestamp": pd.DatetimeIndex(labels.index).values,
                "_resample_label": pd.DatetimeIndex(labels.values).values,
            }
        ).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns")),
            pl.col("_resample_label").cast(pl.Datetime("ns")),
        )
        aggregations = []
        for name, method in how.items():
            if name in ("timestamp", *keys):
                continue
            column = pl.col(name)
            aggregations.append(
                {
                    "first": column.drop_nulls().first(),
                    "last": column.drop_nulls().last(),
                    "max": column.max(),
                    "min": column.min(),
                    "sum": column.sum(),
                    "mean": column.mean(),
                    "count": column.count(),
                }[method].alias(name)
            )
        self.data = (
            self.data.with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
            .join(mapping, on="timestamp", how="left")
            .sort("timestamp")
            .group_by(["_resample_label", *keys], maintain_order=True)
            .agg(aggregations)
            .rename({"_resample_label": "timestamp"})
            .sort(["timestamp", *keys])
        )
        return self

    def head(self, path: str, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows scanned lazily from ``path``.

        Parameters
        ----------
        path : str
            Path of the Parquet file.
        n : int
            Maximum number of rows to return.

        The limit is pushed down into the Parquet reader, and a fresh frame
        is returned without touching ``data``. The existence check is done
        here because ``scan_parquet`` on a missing file only fails at
        collect time.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> PlBackend().head("universe.parquet", 3).collect().shape
        (3, 3)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        return pl.scan_parquet(path).head(n)

    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        """Collect ``data`` and convert it to a dataset indexed by ``indexes``.

        The named columns become the dataset's dimensions and every other
        column becomes a data variable.

        Parameters
        ----------
        indexes : list[str]
            The columns to index by. Required: a lazy frame has no
            dimensions to fall back on.

        Raises
        ------
        ValueError
            If ``indexes`` is ``None``.

        Examples
        --------
        >>> ds = table.get_xarray_dataset(["timestamp", "symbol"])
        >>> tuple(ds.dims), list(ds.data_vars)
        (('timestamp', 'symbol'), ['close'])
        """
        if indexes is None:
            raise ValueError(
                "PlBackend.get_xarray_dataset: `indexes` is required. A "
                "LazyFrame has no dimensions to fall back on; name the "
                "columns that should become the dataset's index, e.g. "
                '["timestamp", "symbol"].'
            )
        data = self.data.collect().to_pandas()
        data = data.set_index(indexes)
        return xr.Dataset.from_dataframe(data)
