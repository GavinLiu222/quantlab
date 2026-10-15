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
The wrapper has no store of its own: it reads the wrapped factor's store,
and only the factor owning every variable of that store writes it (ADR
0029).
"""

import dataclasses
from typing import Self

import pandas as pd
import xarray as xr

from quantlab.factor.base import Factor
from quantlab.factor.config import MembershipMaskedFactorConfig


class MembershipMaskedFactor(Factor):
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

    There is no store: ``store_path`` is ``None``, ``build`` and ``extend``
    are refused, and ``store_range`` is the wrapped factor's.

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

    @Factor.config.setter
    def config(self, config: MembershipMaskedFactorConfig):
        """Install ``config`` with ``dataset`` set to the wrapped factor's dataset."""
        Factor.config.fset(
            self, dataclasses.replace(config, dataset=config.factor.config.dataset)
        )

    def copy(self) -> Self:
        """Return a copy with its own wrapped factor and membership dataset.

        Examples
        --------
        >>> other = factor.copy()
        >>> other == factor, other.config.factor is factor.config.factor
        (True, False)
        """
        other = super().copy()
        other.config = dataclasses.replace(
            other.config,
            factor=self.config.factor.copy(),
            membership=self.config.membership.copy(),
        )
        return other

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
        """Return the wrapped factor's output symbols.

        Examples
        --------
        >>> factor.output_symbols() == factor.config.factor.output_symbols()
        True
        """
        return self.config.factor.output_symbols()

    def read(self, start, end, symbols=None) -> xr.Dataset:
        """Return the wrapped factor's store from ``start`` to ``end``, masked.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.
        symbols : sequence, optional
            The symbols to read; ``None`` reads the wrapped factor's.

        Raises
        ------
        ValueError
            If a bar's date lies outside the membership panel, or the
            wrapped factor's ``read`` refuses the request.

        Examples
        --------
        >>> factor.read("2024-01-18", "2024-01-23", symbols=["4"])["c"].notnull().values.ravel().tolist()
        [False, False, True, True]
        """
        return self.mask(self.config.factor.read(start, end, symbols=symbols))

    def compute(self, start, end) -> xr.Dataset:
        """Compute the wrapped factor on unmasked prices, then mask it.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to compute, both inclusive.

        Raises
        ------
        ValueError
            If a bar's date lies outside the membership panel.

        Examples
        --------
        >>> factor.compute("2024-01-18", "2024-01-23")["symbol"].values.tolist()
        ['2', '4']
        """
        return self.mask(self.config.factor.compute(start, end))

    def mask(self, panel: xr.Dataset) -> xr.Dataset:
        """Return ``panel`` with every cell that is not a member at its own bar set to NaN.

        Parameters
        ----------
        panel : xr.Dataset
            A factor panel on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            If a bar's date lies outside the membership panel, or no symbol
            of ``panel`` is on the membership store's symbol axis.

        Examples
        --------
        >>> panel = factor.config.factor.read("2024-01-19", "2024-01-22")
        >>> factor.mask(panel)["c"].sel(symbol="4").notnull().values.tolist()
        [False, True]
        """
        if panel.sizes.get("timestamp", 0) == 0:
            return panel
        symbols = pd.Index(panel["symbol"].values)
        membership = self.config.membership
        if len(symbols) and not symbols.isin(membership.stored_symbols()).any():
            raise ValueError(
                f"{self.class_name}: none of the {len(symbols)} symbol(s) of "
                f"{self.config.factor.class_name}'s panel is on the symbol axis of "
                f"{membership.class_name}'s store (labels {symbols[:3].tolist()} of type "
                f"{symbols.dtype}); every cell would be masked. Check that both are "
                f"on the same axis (an integer permaticker against a text one)."
            )
        member = membership.is_member_at(
            panel["timestamp"].values, panel["symbol"].values, owner=self.class_name
        )
        return panel.where(member)

    def build(self, start, end) -> Self:
        """Refused: the wrapper has no store; build the wrapped factor's owner.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-01-30")
        Traceback (most recent call last):
        ValueError: MembershipMaskedFactor.build(): the wrapper has no store; ...
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
        ValueError: MembershipMaskedFactor.extend(): the wrapper has no store; ...
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
