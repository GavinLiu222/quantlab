"""Roster dataset: another dataset read on a roster only.

A universe computes its factors and labels on its own **Roster**: the
undated list of symbols its panel may hold (cross-sectional ranks and
z-scores depend on who is in the panel). ``RosterDataset`` wraps the
vendor's market-wide dataset and takes the roster dataset's whole symbol
axis, so a universe reads the shared store instead of keeping a copy of it
cut to the roster. The roster is read on every call and follows its store:
a universe's membership store gains a symbol the day it first enters.

The wrapper is a drop-in ``MarketDataset``: a factor, a label or a backtest
given it in place of the wrapped dataset changes nothing else. Its calendar,
variable names, KunQuant export, tradable and delisting bars and ticker
lookup are the wrapped dataset's. It has no store of its own; the wrapped
dataset's store is updated by its own vendor update. Who is a member on a
given date is the membership a label or a predictor is masked with.

The wrapped dataset records the read in an open ``DataRecorder``, asked
only for the roster's symbols, so a run's data fingerprint covers the
roster's cells only.
"""

from typing import Self, Sequence

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.dataset.base import BaseDataset, MarketDataset, TickerLookup
from quantlab.dataset.config import RosterDatasetConfig
from quantlab.utils.symbol_axis import on_roster


class RosterDataset(MarketDataset):
    """``config.dataset`` on the symbols of ``config.roster``.

    The roster is the symbol axis of the roster dataset's store. It is cast
    to the type of the wrapped store's symbol axis (an integer permaticker
    axis against a text one), and the symbols the wrapped store lacks are
    left out, the count logged: a symbol without values is a fact of the
    data, not an error. The kept symbols are in the wrapped store's order.

    Parameters
    ----------
    config : RosterDatasetConfig or MarketDataset
        The config, or the wrapped dataset with ``roster`` beside it.
    roster : BaseDataset, optional
        The roster dataset, when ``config`` is the wrapped dataset.

    Raises
    ------
    ValueError
        If ``dataset`` is not a ``MarketDataset`` or ``roster`` not a
        dataset.

    Examples
    --------
    With ``sep`` a store on permatickers 1..5 and ``membership`` a universe's
    membership store on 2 and 4:

    >>> prices = RosterDataset(sep, membership)
    >>> prices.panel("2024-01-02", "2024-01-05")["symbol"].values.tolist()
    [2, 4]
    """

    # Narrower type annotation for readers and type checkers only.
    config: RosterDatasetConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = RosterDatasetConfig

    def __init__(
        self,
        config: "RosterDatasetConfig | MarketDataset",
        roster: "BaseDataset | None" = None,
    ):
        """Initialize the view; see the class docstring for parameters."""
        if not isinstance(config, RosterDatasetConfig):
            config = RosterDatasetConfig(dataset=config, roster=roster)
        elif roster is not None:
            raise ValueError(
                f"{type(self).__name__}: give the roster in the config, not beside it."
            )
        super().__init__(config)

    def inputs(self) -> tuple[BaseDataset, ...]:
        """Return the wrapped dataset and the roster dataset, both read by ``panel``.

        Examples
        --------
        >>> RosterDataset(sep, membership).inputs() == (sep, membership)
        True
        """
        return (self.dataset, self.config.roster)

    @property
    def dataset(self) -> MarketDataset:
        """The wrapped dataset.

        Examples
        --------
        >>> RosterDataset(sep, membership).dataset is sep
        True
        """
        return self.config.dataset

    def _normalize_config(self, config: RosterDatasetConfig) -> RosterDatasetConfig:
        """Return ``config`` with ``name`` set, refusing inputs that are not datasets."""
        if not isinstance(config.dataset, MarketDataset):
            raise ValueError(
                f"{self.class_name}: dataset must be a MarketDataset, got "
                f"{type(config.dataset).__name__}."
            )
        if not isinstance(config.roster, BaseDataset):
            raise ValueError(
                f"{self.class_name}: roster must be a dataset, got "
                f"{type(config.roster).__name__}."
            )
        return RosterDatasetConfig(
            dataset=config.dataset, roster=config.roster, name=self.import_path
        )

    def stored_symbols(self) -> list:
        """Return the roster's symbols the wrapped store holds, in its axis order.

        Only the two symbol axes are read, never a data variable.

        Examples
        --------
        >>> RosterDataset(sep, membership).stored_symbols()
        [2, 4]
        """
        axis = pd.Index(self.dataset.stored_symbols())
        listed = pd.Index(self.config.roster.stored_symbols())
        kept = on_roster(axis, listed)
        dropped = listed.nunique() - len(kept)
        if dropped:
            logger.info(
                f"{self.class_name}: {dropped} of {listed.nunique()} roster symbol(s) "
                f"are not in {self.dataset.class_name}'s store and are left out."
            )
        return kept

    def panel(
        self,
        start,
        end,
        symbols: "Sequence | None" = None,
        variables: "Sequence[str] | None" = None,
    ) -> xr.Dataset:
        """Return the wrapped dataset's panel from ``start`` to ``end`` on the roster.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.
        symbols : sequence, optional
            Narrows the roster further, in the order given; ``None`` reads
            the whole roster.
        variables : sequence of str, optional
            Variables to keep, in the order given; ``None`` keeps every one.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``.

        Raises
        ------
        KeyError
            If ``symbols`` names a symbol off the roster, or a variable is
            not in the wrapped store.
        ValueError
            If ``start`` is after ``end``.

        Examples
        --------
        >>> prices.panel("2024-01-02", "2024-01-05", symbols=[4])["symbol"].values.tolist()
        [4]
        >>> prices.panel("2024-01-02", "2024-01-05", symbols=[3])
        Traceback (most recent call last):
        KeyError: 'RosterDataset.panel(): symbol(s) [3] are not on the roster ...'
        """
        kept = self.stored_symbols()
        if symbols is not None:
            off = pd.Index(list(symbols)).difference(pd.Index(kept))
            if len(off):
                raise KeyError(
                    f"{self.class_name}.panel(): symbol(s) {off.tolist()} are not on "
                    f"the roster of {self.dataset.class_name}'s store."
                )
            kept = list(symbols)
        return self.dataset.panel(start, end, symbols=kept, variables=variables)

    def _calendar(self) -> pd.DatetimeIndex:
        """Return the wrapped dataset's calendar."""
        return self.dataset._calendar()

    def resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bars the wrapped dataset cuts ``timestamps`` into."""
        return self.dataset.resample_labels(timestamps, freq)

    def own_names(self, names) -> list[str]:
        """Return the wrapped dataset's own names of the shared ``names``.

        Examples
        --------
        >>> prices.own_names(["adjClose"])
        ['adjClose']
        """
        return self.dataset.own_names(names)

    def shared_name_map(self, names) -> dict[str, str]:
        """Return the wrapped dataset's renaming of ``names`` onto the shared names.

        Examples
        --------
        >>> prices.shared_name_map(["adjClose"])
        {}
        """
        return self.dataset.shared_name_map(names)

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns as the wrapped dataset does."""
        return self.dataset._to_kunquant(data, data_columns)

    def tradable_bars(self, prices: xr.Dataset, fill_column: str) -> xr.DataArray:
        """Return the wrapped dataset's tradable bars of a panel of this view.

        Examples
        --------
        >>> panel = prices.panel("2024-01-02", "2024-01-05")
        >>> prices.tradable_bars(panel, "adjOpen").equals(sep.tradable_bars(panel, "adjOpen"))
        True
        """
        return self.dataset.tradable_bars(prices, fill_column)

    def delisting_bars(self, prices: xr.Dataset, valuation_column: str) -> xr.DataArray:
        """Return the wrapped dataset's delisting bars of a panel of this view.

        Examples
        --------
        >>> panel = prices.panel("2024-01-02", "2024-01-05")
        >>> prices.delisting_bars(panel, "adjClose").equals(sep.delisting_bars(panel, "adjClose"))
        True
        """
        return self.dataset.delisting_bars(prices, valuation_column)

    def ticker_lookup(self) -> TickerLookup | None:
        """Return the wrapped dataset's ticker lookup.

        Examples
        --------
        With ``sep`` a Sharadar SEP dataset whose ticker sidecar names permaticker 199059 AAPL:

        >>> RosterDataset(sep, membership).ticker_lookup().label([199059], date(2024, 1, 2))
        ['AAPL']
        """
        return self.dataset.ticker_lookup()

    def head(self, n: int) -> pl.LazyFrame:
        """Return up to ``n`` rows of the wrapped dataset, for its column names and dtypes.

        Examples
        --------
        >>> prices.head(0).collect_schema().names() == sep.head(0).collect_schema().names()
        True
        """
        return self.dataset.head(n)

    def copy(self) -> Self:
        """Return a roster view over copies of the wrapped and roster datasets.

        Examples
        --------
        >>> other = prices.copy()
        >>> other == prices, other.dataset is prices.dataset
        (True, False)
        """
        return type(self)(self.dataset.copy(), self.config.roster.copy())

    def _refuse(self, method: str):
        """Raise: the roster view holds no store of its own."""
        raise ValueError(
            f"{self.class_name}.{method}(): a roster dataset is a view of "
            f"{self.dataset.class_name}'s store and holds no store of its own; "
            f"call {method}() on the wrapped dataset."
        )

    def resample(self, freq, how) -> Self:
        """Refuse: resample the wrapped dataset, then wrap it.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> prices.resample("1w", "last")
        Traceback (most recent call last):
        ValueError: RosterDataset.resample(): a roster dataset is a view of ...
        """
        self._refuse("resample")

    def save(self, **kwargs):
        """Refuse: the view has no store.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> prices.save()
        Traceback (most recent call last):
        ValueError: RosterDataset.save(): a roster dataset is a view of ...
        """
        self._refuse("save")

    def from_raw_data(self) -> Self:
        """Refuse: build the wrapped dataset instead.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> prices.from_raw_data()
        Traceback (most recent call last):
        ValueError: RosterDataset.from_raw_data(): a roster dataset is a view of ...
        """
        self._refuse("from_raw_data")

    def from_raw_data_chunked(self, *args, **kwargs):
        """Refuse: build the wrapped dataset instead.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> prices.from_raw_data_chunked()
        Traceback (most recent call last):
        ValueError: RosterDataset.from_raw_data_chunked(): a roster dataset is a view of ...
        """
        self._refuse("from_raw_data_chunked")

    def update(self, *args, **kwargs):
        """Refuse: update the wrapped dataset instead.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> prices.update()
        Traceback (most recent call last):
        ValueError: RosterDataset.update(): a roster dataset is a view of ...
        """
        self._refuse("update")

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Refuse: a roster view has no raw files."""
        self._refuse("_raw_data_to_xr")

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None) -> xr.Dataset:
        """Refuse: a roster view has no raw files."""
        self._refuse("_raw_data_to_xr_window")
