"""Dataset views: datasets that read other datasets and keep no store.

A **dataset view** reads other datasets, its inputs (``inputs()``), and is a
``MarketDataset`` itself: it drops in wherever the dataset it reads is used
and changes only what it says it changes, such as which symbols it keeps
(``RosterDataset``). ``DatasetView`` answers the read interface from the
view's input once, so a view defines only what it changes, a market rule
added to ``MarketDataset`` reaches every view over one input, and views of
views compose.

A view over several inputs (a merge) has no input to answer from: each
member of the read interface raises until the view defines it, so a new
member gets a deliberate answer there instead of a silent wrong one.

A dataset that keeps a store of its own, even one written from other
datasets, is a leaf, not a view.
"""

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.core.component import child_components
from quantlab.core.store_uses import BuiltFrom
from quantlab.dataset.base import MarketDataset, TickerLookup


class DatasetView(MarketDataset):
    """Root class of the datasets that read other datasets and keep no store.

    With one input, every member of the read interface but ``panel`` is by
    default the input's, through the view; a subclass overrides the members
    it changes (``RosterDataset`` its stored symbols, ``PermatickerDataset``
    its symbol axis, calendar and rows):

    ==============================  ========================================
    ``calendar``, ``bar_before``,   the input's calendar (``_calendar``)
    ``bar_after``
    ``resample_labels``             the input's
    ``ticker_lookup``               the input's
    ``stored_symbols``              the input's
    ``head``                        the input's
    ``own_names``,                  the input's shared-name mapping
    ``shared_name_map``
    ``tradable_bars``               the input's
    ``delisting_bars``              the input's
    ``_own_delisting_symbols``      the input's
    ``_to_kunquant``                the input's export, behind ``to_kunquant``
    ``panel``                       abstract: what the view changes
    ``store_path``                  ``None``: a view keeps no store
    ``store_uses``                  none of its own; its inputs declare the
                                    stores it reads
    ==============================  ========================================

    With several inputs there is no input to answer from: each member above
    but ``panel`` and the store rows raises ``NotImplementedError`` naming
    the member and the view, until the view defines it.

    The input answered from is ``_source()``: the only input. A view that
    reads a second dataset for something it defines itself (the roster
    whose symbols ``RosterDataset`` keeps) overrides ``_source``. A subclass
    declares ``config_cls``, its ``inputs()`` and ``panel``, and overrides
    each member it changes.

    Parameters
    ----------
    config : FrozenConfig
        The view's config, holding its inputs in component fields. Only
        ``name`` is filled in.

    Examples
    --------
    With ``prices`` a ``FrameDataset`` of daily closes written to a store
    (``to_zarr``):

    >>> from dataclasses import dataclass
    >>> from quantlab.core.component import component
    >>> from quantlab.core.config import FrozenConfig
    >>> @dataclass(kw_only=True, frozen=True)
    ... class HalvedConfig(FrozenConfig):
    ...     dataset: MarketDataset = component()
    ...     name: str | None = None
    >>> class Halved(DatasetView):
    ...     config_cls = HalvedConfig
    ...     def inputs(self):
    ...         return (self.config.dataset,)
    ...     def panel(self, start, end, symbols=None, variables=None):
    ...         return self.config.dataset.panel(start, end, symbols, variables) / 2
    >>> view = Halved(HalvedConfig(dataset=prices))
    >>> view.inputs() == (prices,)
    True
    >>> view.calendar("2024-01-01", "2024-01-31").equals(prices.calendar("2024-01-01", "2024-01-31"))
    True
    >>> view.store_path is None
    True
    """

    _is_view = True

    def _source(self) -> "MarketDataset | None":
        """Return the input the read interface is answered from: the only one, else ``None``."""
        inputs = self.inputs()
        return inputs[0] if len(inputs) == 1 else None

    def _answering(self, member: str) -> MarketDataset:
        """Return the input that answers ``member``, or raise for a view of several inputs."""
        source = self._source()
        if source is None:
            raise NotImplementedError(
                f"{self.class_name}.{member}: a view of {len(self.inputs())} inputs must "
                f"define {member}; DatasetView answers it from a single input only."
            )
        return source

    @abstractmethod
    def panel(
        self,
        start,
        end,
        symbols: "Sequence | None" = None,
        variables: "Sequence[str] | None" = None,
    ) -> xr.Dataset:
        """Return the view's panel from ``start`` to ``end``: what the view changes.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range, both inclusive.
        symbols : sequence, optional
            Symbols to keep, in the order given; ``None`` keeps every one.
        variables : sequence of str, optional
            Variables to keep, in the order given; ``None`` keeps every one.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``.

        Examples
        --------
        >>> view.panel("2024-01-02", "2024-01-03")["adjClose"].equals(
        ...     prices.panel("2024-01-02", "2024-01-03")["adjClose"] / 2)
        True
        """

    @property
    def store_path(self) -> None:
        """``None``: a view keeps no store of its own.

        Examples
        --------
        >>> view.store_path is None
        True
        """
        return None

    def store_uses(self) -> tuple:
        """Return no store use of the view's own: its inputs declare the stores it reads.

        A component the view holds that is not one of its inputs is one it
        is only built from (``BuiltFrom``).

        Examples
        --------
        >>> view.store_uses()
        ()
        >>> view.stores() == prices.stores()
        True
        """
        inputs = {id(dataset) for dataset in self.inputs()}
        return tuple(
            BuiltFrom(child) for _, child in child_components(self) if id(child) not in inputs
        )

    def _calendar(self) -> pd.DatetimeIndex:
        """Return the input's calendar."""
        return self._answering("_calendar")._calendar()

    def resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bars the input cuts ``timestamps`` into.

        Examples
        --------
        >>> stamps = prices.calendar("2024-01-01", "2024-01-31").values
        >>> bool((view.resample_labels(stamps, "1d") == prices.resample_labels(stamps, "1d")).all())
        True
        """
        return self._answering("resample_labels").resample_labels(timestamps, freq)

    def ticker_lookup(self) -> TickerLookup | None:
        """Return the input's ticker lookup.

        Examples
        --------
        >>> view.ticker_lookup() is None   # a FrameDataset names no tickers
        True
        """
        return self._answering("ticker_lookup").ticker_lookup()

    def stored_symbols(self) -> list:
        """Return the input's stored symbols.

        Examples
        --------
        >>> view.stored_symbols() == prices.stored_symbols()
        True
        """
        return self._answering("stored_symbols").stored_symbols()

    def head(self, n: int) -> pl.LazyFrame:
        """Return up to ``n`` rows of the input, for its column names and dtypes.

        Examples
        --------
        >>> view.head(0).collect_schema().names() == prices.head(0).collect_schema().names()
        True
        """
        return self._answering("head").head(n)

    def own_names(self, names) -> list[str]:
        """Return the input's own names of the shared ``names``.

        Examples
        --------
        >>> view.own_names(["adjClose"])
        ['adjClose']
        """
        return self._answering("own_names").own_names(names)

    def shared_name_map(self, names) -> dict[str, str]:
        """Return the input's renaming of ``names`` onto the shared names.

        Examples
        --------
        >>> view.shared_name_map(["adjClose"])
        {}
        """
        return self._answering("shared_name_map").shared_name_map(names)

    def tradable_bars(self, prices: xr.Dataset, fill_column: str) -> xr.DataArray:
        """Return the input's tradable bars of a panel of this view.

        Examples
        --------
        >>> panel = view.panel("2024-01-02", "2024-01-05")
        >>> view.tradable_bars(panel, "adjClose").equals(prices.tradable_bars(panel, "adjClose"))
        True
        """
        return self._answering("tradable_bars").tradable_bars(prices, fill_column)

    def delisting_bars(self, prices: xr.Dataset, valuation_column: str) -> xr.DataArray:
        """Return the input's delisting bars of a panel of this view.

        Examples
        --------
        >>> panel = view.panel("2024-01-02", "2024-01-05")
        >>> view.delisting_bars(panel, "adjClose").equals(prices.delisting_bars(panel, "adjClose"))
        True
        """
        return self._answering("delisting_bars").delisting_bars(prices, valuation_column)

    def _own_delisting_symbols(self) -> list:
        """Return the symbols the input delists by a rule of its own."""
        return self._answering("_own_delisting_symbols")._own_delisting_symbols()

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns as the input does."""
        return self._answering("_to_kunquant")._to_kunquant(data, data_columns)
