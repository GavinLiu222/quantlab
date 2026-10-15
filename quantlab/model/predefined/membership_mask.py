"""Mask a predictor's predictions with an index's point-in-time membership.

An index such as the S&P 500 is a *universe*: it says which securities a
strategy may enter at bar t. That is a decision input, so it belongs on the
predictions, never on the prices. A security that leaves the index is still
quoted and can still be sold; masking its prices instead would make it
untradable on its first non-member bar and settle it like a delisting at its
last member close, a sale that never happens live.

``MembershipMaskedPredictor`` wraps any ``Predictor`` and sets its
predictions to NaN wherever the symbol is not an index member at that bar.
It changes nothing else, so it drops into ``BacktestConfig.model`` in place
of the model it wraps, and the backtester's ``predictions.zarr`` holds the
masked panel.
"""

import dataclasses
from typing import TYPE_CHECKING, Self

import xarray as xr

from quantlab.core.component import Component, component

if TYPE_CHECKING:  # type hints only
    from pathlib import Path

    from quantlab.dataset.raw import IndexConstituentDataset


@dataclasses.dataclass(frozen=True)
class MembershipMaskConfig:
    """What a ``MembershipMaskedPredictor`` is rebuilt from.

    Examples
    --------
    >>> sorted(MembershipMaskConfig(predictor=model, membership=membership).__dataclass_fields__)
    ['membership', 'predictor']
    """

    #: The wrapped predictor.
    predictor: object = component()
    #: The index-membership dataset the predictions are masked with.
    membership: "IndexConstituentDataset" = component()


class MembershipMaskedPredictor(Component):
    """A ``Predictor`` whose predictions are NaN where the symbol is not an index member.

    ``predict_window`` asks the wrapped predictor for its predictions and
    keeps a cell only when the membership panel's ``is_member`` is true on
    that bar's date for that symbol; every other protocol member forwards to
    the wrapped predictor unchanged. The mask at bar t reads the membership
    of t's date and nothing later, so it is information at t. Prices,
    tradability and delistings are untouched: a security that leaves the
    index keeps its prices, its prediction turns NaN, and the portfolio rule
    decides what happens to a holding (TopN sells it at the next rebalance,
    mean-variance holds it at mu=0).

    A symbol missing from the membership panel is not a member. A bar whose
    date the membership panel does not cover (before its first or after its
    last day) is refused, because unknown membership is not "not a member".

    Parameters
    ----------
    predictor : Predictor
        The model or ensemble whose predictions are masked.
    membership : IndexConstituentDataset
        The point-in-time membership, with a stored ``is_member`` panel on
        the symbol axis of the predictions (PERMNOs for CRSP stores).

    Attributes
    ----------
    predictor : Predictor
        The wrapped predictor.
    membership : IndexConstituentDataset
        The membership dataset.

    Examples
    --------
    With ``model`` a model over CRSP PERMNOs and ``membership`` the
    ``CrspSP500ConstituentDataset`` over the same roster::

        >>> masked = MembershipMaskedPredictor(model, membership)
        >>> backtester = USEquityCrossectionSelectStockVectorBt(
        ...     CrossSectionBacktestConfig(
        ...         price_dataset=crsp_index_dataset,  # unmasked prices
        ...         model=masked, model_mode="load", checkpoint=checkpoint,
        ...         start_date="2020-01-01", end_date="2024-12-31",
        ...         output_dir="runs", rebalance_periods=5,
        ...         constructor=TopNConstructor(TopNConfig(top_n=50)),
        ...     )
        ... )
        >>> result = backtester.run()  # predictions.zarr is masked too
    """

    #: The config dataclass the wrapper is serialised and rebuilt with.
    config_cls = MembershipMaskConfig

    def __init__(
        self, predictor, membership: "IndexConstituentDataset"
    ) -> None:
        """Initialize the wrapper; see the class docstring for parameters."""
        self.predictor = predictor
        self.membership = membership

    # -- the mask --------------------------------------------------------

    def mask(self, predictions: xr.Dataset) -> xr.Dataset:
        """Return ``predictions`` with every non-member cell set to NaN.

        Parameters
        ----------
        predictions : xr.Dataset
            Predictions on ``(timestamp, symbol)``.

        Returns
        -------
        xr.Dataset
            The same panel, NaN where ``is_member`` is false or the symbol
            has no membership column.

        Raises
        ------
        ValueError
            If a bar's date lies outside the membership panel.

        Examples
        --------
        With ``membership`` a ``DemoPanel`` (an ``IndexConstituentDataset``
        subclass, as in its docstring) in which ``AAA`` is a member from
        2024-01-01 on, ``BBB`` from 2024-01-01 to 2024-01-02 and ``CCC``
        never, saved with ``as_of="2024-01-05"``, and ``predictions`` a
        ``fwd_ret_5`` panel of 0.1 / 0.2 / 0.3 over three business days:

        >>> masked = MembershipMaskedPredictor(model, membership)
        >>> masked.mask(predictions)["fwd_ret_5"].to_pandas()
        symbol      AAA  BBB  CCC
        timestamp
        2024-01-02  0.1  0.2  NaN
        2024-01-03  0.1  NaN  NaN
        2024-01-04  0.1  NaN  NaN
        >>> later = predictions.assign_coords(
        ...     timestamp=pd.bdate_range("2024-01-04", periods=3)
        ... )
        >>> masked.mask(later)
        Traceback (most recent call last):
        ...
        ValueError: MembershipMaskedPredictor: the membership panel of DemoPanel does not cover 1 date(s) (2024-01-08..2024-01-08); extend the membership store or narrow the window, since unknown membership is not 'not a member'
        """
        if predictions.sizes.get("timestamp", 0) == 0:
            return predictions
        member = self.membership.is_member_at(
            predictions["timestamp"].values,
            predictions["symbol"].values,
            owner=type(self).__name__,
        )
        return predictions.where(member)

    # -- Predictor protocol ----------------------------------------------
    # Everything but ``predict_window`` forwards to the wrapped predictor
    # unchanged.

    @property
    def labels(self) -> list:
        """The wrapped predictor's labels.

        Examples
        --------
        >>> masked.labels is model.labels
        True
        """
        return self.predictor.labels

    @property
    def train_bounds(self) -> tuple:
        """The wrapped predictor's ``(start, end)`` training window.

        Examples
        --------
        >>> masked.train_bounds == model.train_bounds
        True
        """
        return self.predictor.train_bounds

    @property
    def test_bounds(self) -> tuple:
        """The wrapped predictor's ``(start, end)`` test window.

        Examples
        --------
        >>> masked.test_bounds == model.test_bounds
        True
        """
        return self.predictor.test_bounds

    @property
    def fitted_train_bounds(self) -> tuple:
        """The wrapped predictor's fitted ``(start, end)`` training window.

        Examples
        --------
        >>> masked.fitted_train_bounds == model.fitted_train_bounds
        True
        """
        return self.predictor.fitted_train_bounds

    @property
    def label_delays(self) -> tuple[int, ...]:
        """The wrapped predictor's label delays; masking moves no bar.

        Examples
        --------
        >>> masked.label_delays == model.label_delays
        True
        """
        return self.predictor.label_delays

    @property
    def label_scales(self) -> dict[str, str]:
        """The wrapped predictor's label scales; masking rescales nothing.

        Examples
        --------
        >>> masked.label_scales == model.label_scales
        True
        """
        return self.predictor.label_scales

    def predict_window(self, start, end) -> xr.Dataset:
        """The wrapped predictor's predictions over the window, masked; see ``mask``.

        Examples
        --------
        >>> out = masked.predict_window("2024-02-12", "2024-03-11")
        >>> bool(out.equals(masked.mask(model.predict_window("2024-02-12", "2024-03-11"))))
        True
        """
        return self.mask(self.predictor.predict_window(start, end))

    def collect(self) -> Self:
        """Collect the wrapped predictor's training data; returns the wrapper.

        Examples
        --------
        >>> masked.collect() is masked
        True
        """
        self.predictor.collect()
        return self

    def train(self) -> "Path":
        """Train the wrapped predictor on unmasked data; returns its checkpoint.

        Examples
        --------
        >>> checkpoint = masked.collect().train()  # what model.train() returns
        """
        return self.predictor.train()

    def load(self, path) -> Self:
        """Load ``path`` into the wrapped predictor; returns the wrapper.

        Examples
        --------
        >>> masked.load(checkpoint) is masked
        True
        """
        self.predictor.load(path)
        return self

    def check_checkpoint(self, path) -> None:
        """Validate ``path`` with the wrapped predictor, without loading it.

        Examples
        --------
        >>> masked.check_checkpoint(checkpoint)  # raises if it does not fit
        """
        self.predictor.check_checkpoint(path)

    @property
    def config(self) -> MembershipMaskConfig:
        """The wrapped predictor and the membership dataset; ``get_config()`` serialises it.

        Examples
        --------
        >>> sorted(masked.get_config())
        ['membership', 'name', 'predictor']
        """
        return MembershipMaskConfig(predictor=self.predictor, membership=self.membership)

    @classmethod
    def from_config(cls, config: dict, run_dir=None) -> Self:
        """Rebuild the wrapper, its predictor and its membership dataset from ``get_config()``.

        Both are rebuilt by the component rule, with ``run_dir`` passed down.

        Examples
        --------
        >>> rebuilt = MembershipMaskedPredictor.from_config(masked.get_config())
        >>> type(rebuilt.membership) is type(masked.membership)
        True
        """
        fields = cls._rebuilt_fields(config, run_dir)
        return cls(fields["predictor"], fields["membership"])
