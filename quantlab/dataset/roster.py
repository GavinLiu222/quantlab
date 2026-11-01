"""Roster dataset: another dataset read on a roster only.

A universe computes its factors and labels on its own **Roster**: the
undated list of symbols its panel may hold (cross-sectional ranks and
z-scores depend on who is in the panel). ``RosterDataset`` wraps the
vendor's market-wide dataset and takes the roster dataset's whole symbol
axis, so a universe reads the shared store instead of keeping a copy of it
cut to the roster. The roster is read on every call and follows its store:
a universe's membership store gains a symbol the day it first enters.

The wrapper is a dataset view (``DatasetView``): a factor, a label or a
backtest given it in place of the wrapped dataset changes nothing else. It
defines only its panel cut and its symbols; its calendar, variable names,
KunQuant export, tradable and delisting bars and ticker lookup are the
wrapped dataset's. It has no store of its own and no build path: its inputs
are the wrapped and the roster datasets (``inputs()``), whose stores are
updated by their own vendor updates. Who is a member on a
given date is the membership a label or a predictor is masked with.

The wrapped dataset records the read in an open ``DataRecorder``, asked
only for the roster's symbols, so a run's data fingerprint covers the
roster's cells only.
"""

from typing import Self, Sequence

import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.dataset.base import BaseDataset, MarketDataset
from quantlab.dataset.config import RosterDatasetConfig
from quantlab.dataset.view import DatasetView
from quantlab.utils.symbol_axis import on_roster


class RosterDataset(DatasetView):
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

    def _source(self) -> MarketDataset:
        """Return the wrapped dataset: the roster is read for its symbols only."""
        return self.dataset

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

    def _own_delisting_symbols(self) -> list:
        """Return the roster's symbols the wrapped dataset delists by a rule of its own."""
        kept = pd.Index(self.stored_symbols())
        return kept[kept.isin(self.dataset._own_delisting_symbols())].tolist()

    def copy(self) -> Self:
        """Return a roster view over copies of the wrapped and roster datasets.

        Examples
        --------
        >>> other = prices.copy()
        >>> other == prices, other.dataset is prices.dataset
        (True, False)
        """
        return type(self)(self.dataset.copy(), self.config.roster.copy())
