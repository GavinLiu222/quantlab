"""Mask a label with an index's point-in-time membership, judged at t only.

An index such as the S&P 500 is a *universe*: it says which samples exist.
A forward-return label at bar t reads prices up to bar t + lookahead, so a
label computed on a price panel blanked off-membership exists only when the
symbol is still a member at the label's endpoint. A stock usually leaves an
index because it fell, so that drops the losers from the training target and
from every factor report on it: survivorship bias.

``MembershipMaskedLabel`` wraps any label computed on unmasked prices and
sets a cell to NaN only where the symbol is not an index member on t's own
date, the same rule ``MembershipMaskedPredictor`` applies to predictions.
It changes nothing else, so it drops into ``ModelConfig.labels`` and
``Factor.analyze(frets=...)`` in place of the label it wraps.
"""

import dataclasses
from typing import TYPE_CHECKING, Self

import xarray as xr

from quantlab.core.component import Component, component

if TYPE_CHECKING:  # type hints only
    from quantlab.dataset.raw import IndexConstituentDataset
    from quantlab.label.forward import Forward


@dataclasses.dataclass(frozen=True)
class MembershipMaskedLabelConfig:
    """What a ``MembershipMaskedLabel`` is rebuilt from.

    Examples
    --------
    >>> sorted(MembershipMaskedLabelConfig(label=label, membership=membership).__dataclass_fields__)
    ['label', 'membership']
    """

    #: The wrapped label, computed on unmasked prices.
    label: "Forward" = component()
    #: The index-membership dataset the label is masked with.
    membership: "IndexConstituentDataset" = component()


class MembershipMaskedLabel(Component):
    """A label that is NaN where the symbol is not an index member at t.

    ``read`` and ``compute`` return the wrapped label's panel with a cell
    kept only when the membership panel's ``is_member`` is true on that
    bar's date for that symbol. The mask at bar t reads the membership of
    t's date and nothing later, so a symbol that leaves the index inside the
    label's horizon keeps its return at t, and a symbol that is not a member
    at t has no label whatever its later membership. Every other label
    method forwards to the wrapped label unchanged; the wrapper owns no
    store.

    A symbol missing from the membership panel is not a member. A bar whose
    date the membership panel does not cover (before its first or after its
    last day) is refused, because unknown membership is not "not a member".

    Parameters
    ----------
    label : Forward
        The label to mask, such as ``quantlab.label.predefined.fret.Return``
        computed on the index's unmasked prices.
    membership : IndexConstituentDataset
        The point-in-time membership, with a stored ``is_member`` panel on
        the label's symbol axis (PERMNOs for CRSP stores).

    Attributes
    ----------
    label : Forward
        The wrapped label.
    membership : IndexConstituentDataset
        The membership dataset.

    Examples
    --------
    With ``prices`` a ``CrspStockDataset`` over the S&P 500 roster's
    unmasked prices and ``membership`` the ``CrspSP500ConstituentDataset``
    over the same roster:

    >>> ret = Return(FactorConfig(dataset=prices, warmup_bars=0, data_columns=("adjOpen",),
    ...                           kwargs={"n_forward_periods": 20}))
    >>> label = MembershipMaskedLabel(ret, membership)
    >>> label.get_factor_names(), label.lookahead_bars(), label.span_bars()
    (('ret_20',), 21, 20)
    >>> model = XGBoostRegressor(ModelConfig(factors=[alpha158], labels=[label], ...))
    """

    #: The config dataclass the wrapper is serialised and rebuilt with.
    config_cls = MembershipMaskedLabelConfig

    def __init__(self, label: "Forward", membership: "IndexConstituentDataset") -> None:
        """Initialize the wrapper; see the class docstring for parameters."""
        self.label = label
        self.membership = membership

    def __repr__(self) -> str:
        """Return the class name, the wrapped label and the membership class."""
        return (
            f"{type(self).__name__}(label={self.label!r}, "
            f"membership={type(self.membership).__name__})"
        )

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` masks an equal label with an equally configured membership.

        Examples
        --------
        >>> MembershipMaskedLabel(ret, membership) == label
        True
        """
        if type(other) is not type(self):
            return NotImplemented
        return (
            self.label == other.label
            and type(self.membership) is type(other.membership)
            and self.membership.get_config() == other.membership.get_config()
        )

    __hash__ = None

    # -- the mask --------------------------------------------------------

    def read(self, start, end) -> xr.Dataset:
        """Return the wrapped label's stored panel from ``start`` to ``end``, masked.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return; see ``Forward.read``.

        Returns
        -------
        xr.Dataset
            The masked label panel on ``(timestamp, symbol)``.

        Examples
        --------
        >>> label.build("2024-01-01", "2024-02-29")
        >>> label.read("2024-02-01", "2024-02-10").equals(
        ...     membership.mask(ret.read("2024-02-01", "2024-02-10")))
        True
        """
        return self.membership.mask(self.label.read(start, end), owner=type(self).__name__)

    def compute(self, start, end) -> xr.Dataset:
        """Compute the wrapped label from ``start`` to ``end`` and mask it.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return; see ``Forward.compute``.

        Returns
        -------
        xr.Dataset
            The masked label panel on ``(timestamp, symbol)``, in memory.

        Examples
        --------
        >>> label.compute("2024-02-01", "2024-02-10").equals(
        ...     membership.mask(ret.compute("2024-02-01", "2024-02-10")))
        True
        """
        return self.membership.mask(self.label.compute(start, end), owner=type(self).__name__)

    # -- the label protocol, forwarded -----------------------------------

    @property
    def kind(self) -> str:
        """The wrapped label's ``kind``; masking changes what is measured nowhere.

        Examples
        --------
        >>> label.kind
        'return'
        """
        return getattr(self.label, "kind", "return")

    @property
    def class_name(self) -> str:
        """Bare class name, used in log and error messages.

        Examples
        --------
        >>> label.class_name
        'MembershipMaskedLabel'
        """
        return type(self).__name__

    def lookahead_bars(self) -> int:
        """The wrapped label's lookahead; masking reads no later bar.

        Examples
        --------
        >>> label.lookahead_bars() == ret.lookahead_bars()
        True
        """
        return self.label.lookahead_bars()

    def span_bars(self) -> int:
        """The wrapped label's span, the horizon ``Factor.analyze`` reports.

        Examples
        --------
        >>> label.span_bars() == ret.span_bars()
        True
        """
        return self.label.span_bars()

    def delay_bars(self) -> int:
        """The wrapped label's delay, which a model reports as its ``label_delays``.

        Examples
        --------
        >>> label.delay_bars() == ret.delay_bars()
        True
        """
        return self.label.delay_bars()

    def get_factor_names(self) -> tuple[str, ...]:
        """The wrapped label's variable names.

        Examples
        --------
        >>> label.get_factor_names() == ret.get_factor_names()
        True
        """
        return self.label.get_factor_names()

    def build(self, start, end) -> Self:
        """Build the wrapped label's store; returns the wrapper.

        The membership store is not built here; save it beforehand.

        Examples
        --------
        >>> label.build("2024-01-01", "2024-02-29") is label
        True
        """
        self.label.build(start, end)
        return self

    def extend(self, end) -> Self:
        """Extend the wrapped label's store to ``end``; returns the wrapper.

        Examples
        --------
        >>> label.extend("2024-03-29") is label
        True
        """
        self.label.extend(end)
        return self

    def store_range(self) -> tuple[str, str] | None:
        """The wrapped label's ``store_range()``.

        Examples
        --------
        >>> label.store_range() == ret.store_range()
        True
        """
        return self.label.store_range()

    # -- config ----------------------------------------------------------

    @property
    def config(self) -> MembershipMaskedLabelConfig:
        """The wrapped label and the membership dataset; ``get_config()`` serialises it.

        Examples
        --------
        >>> sorted(label.get_config())
        ['label', 'membership', 'name']
        """
        return MembershipMaskedLabelConfig(label=self.label, membership=self.membership)

    @classmethod
    def from_config(cls, config: dict, run_dir=None) -> Self:
        """Rebuild the wrapper, its label and its membership dataset from ``get_config()``.

        Both are rebuilt by the component rule, with ``run_dir`` passed down.

        Examples
        --------
        >>> MembershipMaskedLabel.from_config(label.get_config()) == label
        True
        """
        fields = cls._rebuilt_fields(config, run_dir)
        return cls(fields["label"], fields["membership"])
