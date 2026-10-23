"""Membership-masked factor: another factor kept only where its symbol is an index member at t.

A model built for an index reads a market-wide factor store on its roster
(``RosterFactor``), and a roster is every symbol ever a member. A
cross-sectional model takes every symbol with a finite feature at bar t as
context, so on the roster alone the cross-section at t holds symbols that
are not members at t, among them those that join the index later: a choice
made with future information. ``MembershipMaskedFactor`` wraps any factor
and sets a cell to NaN wherever the symbol is not a member on the bar's own
date, the rule ``MembershipMaskedLabel`` and ``MembershipMaskedPredictor``
apply to the label and the predictions. A non-member then has no finite
feature at t and is left out of the bar's cross-section.

The mask goes on the computed factor, never on the prices: the wrapped
factor is computed on unmasked prices, so a symbol that joins at t has a
value at t from its full price history and enters the cross-section at t.
The wrapper is a factor view (``quantlab.factor.view``): it reads the
wrapped factor's store, and only the factor owning every variable of that
store writes it.
"""

import xarray as xr

from quantlab.factor.config import MembershipMaskedFactorConfig
from quantlab.factor.view import FactorView


class MembershipMaskedFactor(FactorView):
    """The outputs of ``config.factor``, NaN where the symbol is not a member at the bar.

    A cell is kept when ``config.membership.is_member_at`` is true on the
    bar's date; a symbol missing from the membership panel is not a member,
    and a bar whose date the membership panel does not cover is refused,
    warm-up bars included: a model reading it from ``start_date`` with a
    warm-up needs the membership from that many bars earlier. A panel none
    of whose symbols is on the membership store's axis is refused too,
    since it would be masked whole.
    The outputs keep the wrapped factor's names, so the wrapper replaces the
    factor in a model's config without anything else changing.

    A model's feature window over a symbol that joined less than a window
    ago holds NaN for the bars before it joined, which a live model would
    know: a small loss of information, not look-ahead.

    The store queries are a view's (see ``FactorView``): every symbol of the
    wrapped factor is kept, ``store_path`` is ``None``, ``store_uses()``
    reads the wrapped factor's store, ``store_range()`` is the wrapped
    factor's, ``owns_store()`` is false and ``build`` and
    ``extend`` are refused.

    Parameters
    ----------
    config : MembershipMaskedFactorConfig
        The wrapped ``factor`` and the ``membership`` dataset.

    Examples
    --------
    With ``alphas`` a factor on symbols ``2`` (always a member) and ``4``
    (a member from 2024-01-22 on):

    >>> factor = MembershipMaskedFactor(
    ...     MembershipMaskedFactorConfig(factor=alphas, membership=membership)
    ... )
    >>> factor.read("2024-01-18", "2024-01-23")["c"].notnull().to_pandas()
    symbol        2      4
    timestamp
    2024-01-18  True  False
    2024-01-19  True  False
    2024-01-22  True   True
    2024-01-23  True   True
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = MembershipMaskedFactorConfig

    # Narrower type annotation for readers and type checkers only.
    config: MembershipMaskedFactorConfig

    def _view_panel(self, panel: xr.Dataset) -> xr.Dataset:
        """Return ``panel`` masked by the membership (``IndexConstituentDataset.mask``)."""
        return self.config.membership.mask(panel, owner=self.class_name)
