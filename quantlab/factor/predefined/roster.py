"""Roster factor: another factor read on a roster only.

A model built for one universe reads a market-wide factor store (the
BarraStyle store of every listed stock) on its own **Roster**: the undated
list of symbols its panel may hold. ``RosterFactor`` wraps any factor and
takes the roster dataset's whole symbol axis; the outputs keep the wrapped
factor's names, so the wrapper replaces the factor in a model's config
without anything else changing. Who is a member on a given date is decided
elsewhere (the membership a predictor is masked with), never by the roster.

The wrapper has no store of its own: it reads the wrapped factor's store,
and only the factor owning every variable of that store writes it (ADR
0029).
"""

import dataclasses
from typing import Self

import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.factor.base import Factor
from quantlab.factor.config import RosterConfig
from quantlab.utils.symbol_axis import on_roster


class RosterFactor(Factor):
    """The outputs of ``config.factor`` on the symbols of ``config.roster``.

    The roster is the symbol axis of the roster dataset's store, read on
    every call, so it follows the store as members are added. It is cast
    to the type of the wrapped panel's symbol axis (an integer permaticker
    axis against a text one) and the symbols the wrapped panel lacks are
    dropped, the count logged: a symbol without values is a fact of the
    data, not an error. The kept symbols are in the wrapped panel's order.

    ``read`` reads the wrapped factor's store on the kept symbols, so only
    those are read and fingerprinted. ``compute`` computes the wrapped
    factor on its whole dataset and cuts afterwards: a cross-sectional
    factor standardizes over the whole market, so computing it on the
    roster alone would change its values.

    There is no store: ``store_path`` is ``None``, ``build`` and ``extend``
    are refused, and ``store_range`` is the wrapped factor's. A live day
    therefore extends the wrapped store only through its owner.

    Parameters
    ----------
    config : RosterConfig
        The wrapped ``factor`` and the ``roster`` dataset.

    Examples
    --------
    With ``styles`` a factor pinned to ``b`` of a store holding ``a, b, c``
    on symbols 1..5, and ``roster`` a dataset on symbols 2 and 4:

    >>> factor = RosterFactor(RosterConfig(factor=styles, roster=roster))
    >>> panel = factor.read("2024-01-05", "2024-01-10")
    >>> list(panel.data_vars), panel["symbol"].values.tolist()
    (['b'], [2, 4])
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = RosterConfig

    # Narrower type annotation for readers and type checkers only.
    config: RosterConfig

    @Factor.config.setter
    def config(self, config: RosterConfig):
        """Install ``config`` with ``dataset`` set to the wrapped factor's dataset."""
        Factor.config.fset(
            self, dataclasses.replace(config, dataset=config.factor.config.dataset)
        )

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the wrapped factor's output names."""
        return tuple(self.config.factor.get_factor_names())

    def _validate_config(self) -> None:
        """Refuse a warm-up, a store, resampling or names the wrapped factor lacks.

        Raises
        ------
        ValueError
            If ``warmup_bars`` is not 0, ``file_path`` or ``resample_freq``
            is set, or ``factor_names`` are not the wrapped factor's.
        """
        config = self.config
        if config.warmup_bars != 0:
            raise ValueError(
                f"{self.class_name}: warmup_bars must be 0, got {config.warmup_bars}; "
                f"the wrapped factor warms itself up."
            )
        if config.file_path is not None:
            raise ValueError(
                f"{self.class_name}: file_path must be None; the wrapper reads the "
                f"wrapped factor's store and has none of its own."
            )
        if config.resample_freq is not None:
            raise ValueError(
                f"{self.class_name}: cannot be resampled; resample the wrapped factor."
            )
        if tuple(config.factor_names) != self._get_factor_names():
            raise ValueError(
                f"{self.class_name}: factor_names {list(config.factor_names)} are not "
                f"the wrapped factor's {list(self._get_factor_names())}; pin the "
                f"wrapped factor instead."
            )

    def output_symbols(self) -> list:
        """Return the wrapped factor's output symbols on the roster, in the wrapped factor's order.

        Unlike ``read``, nothing is logged: sizing a computation is not a read.

        Examples
        --------
        >>> factor.output_symbols()
        [2, 4]
        """
        listed = pd.Index(self.config.roster.stored_symbols())
        return on_roster(pd.Index(self.config.factor.output_symbols()), listed)

    def read(self, start, end, symbols=None) -> xr.Dataset:
        """Return the wrapped factor's store from ``start`` to ``end`` on the roster.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.
        symbols : sequence, optional
            Narrows the roster further; ``None`` reads the whole roster.

        Raises
        ------
        ValueError
            If ``symbols`` names a symbol off the roster, or the wrapped
            factor's ``read`` refuses the request.

        Examples
        --------
        >>> factor.read("2024-01-05", "2024-01-10")["symbol"].values.tolist()
        [2, 4]
        """
        wrapped = self.config.factor
        kept = self._on_roster(pd.Index(wrapped.stored_symbols()))
        if symbols is not None:
            off = pd.Index(list(symbols)).difference(pd.Index(kept))
            if len(off):
                raise ValueError(
                    f"{self.class_name}.read(): symbol(s) {off.tolist()} are not "
                    f"on the roster of {wrapped.class_name}'s store."
                )
            kept = list(symbols)
        return wrapped.read(start, end, symbols=kept)

    def compute(self, start, end) -> xr.Dataset:
        """Compute the wrapped factor on its whole dataset, then keep the roster.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to compute, both inclusive.

        Examples
        --------
        >>> factor.compute("2024-01-05", "2024-01-10")["symbol"].values.tolist()
        [2, 4]
        """
        panel = self.config.factor.compute(start, end)
        return panel.sel(symbol=self._on_roster(pd.Index(panel["symbol"].values)))

    def build(self, start, end) -> Self:
        """Refused: the wrapper has no store; build the wrapped factor's owner.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-01-30")
        Traceback (most recent call last):
        ValueError: RosterFactor.build(): the wrapper has no store; ...
        """
        raise ValueError(
            f"{self.class_name}.build(): the wrapper has no store; build the "
            f"wrapped factor's store with the factor owning every variable of it."
        )

    def extend(self, end) -> Self:
        """Refused: the wrapper has no store; extend the wrapped factor's owner.

        Examples
        --------
        >>> factor.extend("2024-02-10")
        Traceback (most recent call last):
        ValueError: RosterFactor.extend(): the wrapper has no store; ...
        """
        raise ValueError(
            f"{self.class_name}.extend(): the wrapper has no store; extend the "
            f"wrapped factor's store with the factor owning every variable of it."
        )

    def store_range(self) -> tuple[str, str] | None:
        """Return the wrapped factor's recorded store range.

        Examples
        --------
        >>> factor.store_range() == factor.config.factor.store_range()
        True
        """
        return self.config.factor.store_range()

    def _on_roster(self, axis: pd.Index) -> list:
        """Return the symbols of ``axis`` on the roster, in ``axis`` order.

        The roster is cast to ``axis``'s type first; the count of roster
        symbols ``axis`` lacks is logged.
        """
        listed = pd.Index(self.config.roster.stored_symbols())
        kept = on_roster(axis, listed)
        dropped = listed.nunique() - len(kept)
        if dropped:
            logger.info(
                f"{self.class_name}: {dropped} of {listed.nunique()} roster symbol(s) "
                f"are not in {self.config.factor.class_name}'s panel and are left out."
            )
        return kept
