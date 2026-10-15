"""Roster factor: another factor read on a roster only.

A model built for one universe reads a market-wide factor store (the
BarraStyle store of every listed stock) on its own **Roster**: the undated
list of symbols its panel may hold. ``RosterFactor`` wraps any factor and
takes the roster dataset's whole symbol axis; the outputs keep the wrapped
factor's names, so the wrapper replaces the factor in a model's config
without anything else changing. Who is a member on a given date is decided
elsewhere (the membership a predictor is masked with), never by the roster.

The wrapper is a factor view (``quantlab.factor.view``): it reads the
wrapped factor's store, and only the factor owning every variable of that
store writes it (ADR 0029).
"""

import pandas as pd
from loguru import logger

from quantlab.factor.config import RosterConfig
from quantlab.factor.view import FactorView
from quantlab.utils.symbol_axis import on_roster


class RosterFactor(FactorView):
    """The outputs of ``config.factor`` on the symbols of ``config.roster``.

    The roster is the symbol axis of the roster dataset's store, read on
    every call, so it follows the store as members are added. It is cast
    to the type of the wrapped panel's symbol axis (an integer permaticker
    axis against a text one) and the symbols the wrapped panel lacks are
    dropped, the count logged on a read or a computation: a symbol without
    values is a fact of the data, not an error. The kept symbols are in the
    wrapped panel's order.

    ``read`` reads the wrapped factor's store on the kept symbols, so only
    those are read and fingerprinted. ``compute`` computes the wrapped
    factor on its whole dataset and cuts afterwards: a cross-sectional
    factor standardizes over the whole market, so computing it on the
    roster alone would change its values. The store queries are a view's
    (see ``FactorView``): ``store_path`` and ``store_range()`` are the
    wrapped factor's, ``owns_store()`` is false and ``build`` and
    ``extend`` are refused, so a live day extends the wrapped store only
    through its owner.

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
    (['b'], ['2', '4'])
    >>> factor.stored_symbols()
    ['2', '4']
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = RosterConfig

    # Narrower type annotation for readers and type checkers only.
    config: RosterConfig

    def _kept_symbols(self, axis: pd.Index, log: bool) -> list:
        """Return the symbols of ``axis`` on the roster, in ``axis`` order.

        The roster is cast to ``axis``'s type first; with ``log``, the count
        of roster symbols ``axis`` lacks is logged.
        """
        listed = pd.Index(self.config.roster.stored_symbols())
        kept = on_roster(axis, listed)
        dropped = listed.nunique() - len(kept)
        if log and dropped:
            logger.info(
                f"{self.class_name}: {dropped} of {listed.nunique()} roster symbol(s) "
                f"are not in {self.config.factor.class_name}'s panel and are left out."
            )
        return kept
