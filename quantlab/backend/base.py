"""The abstract storage contract for data panels.

``DataBackend`` keeps "where something is stored" separate from "what it
means": it is the contract a dataset or factor object talks to when it
reads, writes, filters or converts its *panel* (an ``xarray.Dataset``
indexed by ``timestamp`` and ``symbol``). Concrete implementations live in
``quantlab.backend.zarr`` (``XrBackend``) and ``quantlab.backend.parquet``
(``PlBackend``).
The contract assumes no particular schema, so a backend can be swapped
without touching the layers above it. See ``docs/user-guide/datasets.md``.
"""

from abc import ABC, abstractmethod
from typing import Optional, Self

import pandas as pd
import polars as pl
import xarray as xr


class DataBackend(ABC):
    """Contract for a storage medium that holds one tabular or panel dataset.

    A backend owns a single in-memory object, exposed as ``data``, and knows
    how to move it to and from a path, narrow it in place, and convert it to
    the two exchange formats used across the pipeline: an ``xarray.Dataset``
    and a ``polars.LazyFrame``. Every mutating method returns ``self`` so
    calls can be chained.

    Two details of the contract are easy to get wrong when implementing it.
    ``filter_by_date`` and ``filter_by_symbol`` narrow ``data`` in place, and
    every object sharing the backend instance sees the result. ``head`` must
    do the opposite: open the store at the given path, read at most ``n``
    rows without materialising the whole store, and leave ``data`` untouched.

    Examples
    --------
    >>> backend = XrBackend().read("prices.zarr")
    >>> panel = backend.filter_by_date("timestamp", "2022-01-01",
    ...                                "2022-12-31")
    >>> ds = panel.get_xarray_dataset(["timestamp", "symbol"])
    """

    def __repr__(self) -> str:
        """Return the class name followed by empty parentheses."""
        return f"{self.__class__.__name__}()"

    @property
    def data(self):
        """The object this backend currently holds.

        Raises
        ------
        AttributeError
            If nothing has been loaded yet. Accessing an
            unpopulated backend fails immediately rather than returning
            an empty dataset that downstream code would mistake for
            "no data in this range".

        Examples
        --------
        >>> XrBackend().to_internal(panel).data is panel
        True
        >>> XrBackend().data
        Traceback (most recent call last):
        AttributeError: Please call 'read' or 'to_internal' first.
        """
        try:
            return self._data
        except AttributeError:
            raise AttributeError("Please call 'read' or 'to_internal' first.")

    @data.setter
    def data(self, data):
        """Replace the held object.

        Examples
        --------
        >>> backend = XrBackend()
        >>> backend.data = panel
        """
        self._data = data

    @abstractmethod
    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        """Return the held data as an ``xarray.Dataset``.

        Parameters
        ----------
        indexes : list[str], optional
            The dimensions the returned dataset must be indexed by,
            in order. ``None`` asks for no particular shape.

        Examples
        --------
        >>> backend = XrBackend().to_internal(panel)
        >>> tuple(backend.get_xarray_dataset(["timestamp", "symbol"]).dims)
        ('timestamp', 'symbol')
        """
        ...

    @abstractmethod
    def get_lazyframe(self) -> pl.LazyFrame:
        """Return the held data as a ``polars.LazyFrame``.

        Examples
        --------
        >>> backend.get_lazyframe().collect().columns
        ['timestamp', 'symbol', 'close']
        """
        ...

    @abstractmethod
    def resample(self, labels: pd.Series, how: dict[str, str]) -> Self:
        """Aggregate ``data`` onto the coarser time grid ``labels`` describes.

        ``labels`` maps every source timestamp (its index) to the timestamp
        of the bar it belongs to (its value); the caller works those out, so
        the backend knows nothing about clocks or trading sessions. ``how``
        names one ``ResampleMethod`` for every variable. Each variable is
        reduced over the timestamps sharing a label, NaN cells skipped, and
        the result replaces ``data`` in place, sorted by the new timestamps.

        Parameters
        ----------
        labels : pd.Series
            Source timestamp to target timestamp, covering every timestamp
            ``data`` holds.
        how : dict[str, str]
            Variable name to aggregation method.

        Examples
        --------
        >>> labels = pd.Series(minutes.floor("D"), index=minutes)
        >>> backend.resample(labels, {"close": "last", "volume": "sum"})
        >>> dict(backend.data.sizes)
        {'timestamp': 2, 'symbol': 3}
        """
        ...

    @abstractmethod
    def head(self, path: str, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows from the store at ``path``.

        Implementations must not materialise the whole store, must not touch
        ``data``, and must raise ``FileNotFoundError`` when ``path`` does not
        exist rather than deferring the failure to a later collect.

        Parameters
        ----------
        path : str
            Location of the store.
        n : int
            Maximum number of rows to return.

        Examples
        --------
        >>> XrBackend().head("prices.zarr", 2).collect().shape
        (2, 3)
        """
        ...

    @abstractmethod
    def read(self, path: str, **kwargs) -> Self:
        """Load the store at ``path`` into ``data`` and return ``self``.

        Parameters
        ----------
        path : str
            Location of the store.
        **kwargs
            Options for the underlying reader.

        Examples
        --------
        >>> dict(XrBackend().read("prices.zarr").data.sizes)
        {'timestamp': 4, 'symbol': 2}
        """
        ...

    @abstractmethod
    def write(self, path: str, **kwargs) -> Self:
        """Persist ``data`` to ``path`` and return ``self``.

        Parameters
        ----------
        path : str
            Location of the store.
        **kwargs
            Options for the underlying writer.

        Examples
        --------
        >>> XrBackend().to_internal(panel).write("prices.zarr")
        XrBackend()
        """
        ...

    @abstractmethod
    def to_internal(self, data) -> Self:
        """Adopt an in-memory object as ``data``, bypassing disk.

        Parameters
        ----------
        data : object
            The object to hold, typically an ``xarray.Dataset`` or a
            ``polars.LazyFrame``.

        Examples
        --------
        >>> XrBackend().to_internal(panel).data is panel
        True
        """
        ...

    @abstractmethod
    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        """Narrow ``data`` in place to ``start_date..end_date`` on ``col``.

        Parameters
        ----------
        col : str
            The date column or dimension to filter on.
        start_date : str
            First date to keep, inclusive.
        end_date : str
            Last date to keep, inclusive.

        Examples
        --------
        >>> backend.filter_by_date("timestamp", "2024-01-02", "2024-01-03")
        XrBackend()
        >>> backend.data["timestamp"].values.astype("datetime64[D]")
        array(['2024-01-02', '2024-01-03'], dtype='datetime64[D]')
        """
        ...

    @abstractmethod
    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        """Narrow ``data`` in place to the rows whose ``col`` is in ``symbols``.

        Parameters
        ----------
        col : str
            The symbol column or dimension to filter on.
        symbols : tuple[str, ...]
            The symbols to keep.

        Examples
        --------
        >>> backend.filter_by_symbol("symbol", ("BBB",))
        XrBackend()
        >>> backend.data["symbol"].values.tolist()
        ['BBB']
        """
        ...
