"""An index's capitalization weights: ``MembershipCapWeight``.

The weight of a symbol at bar t is its market cap at t over the summed cap
of the index's members at t, and 0 for a symbol that is not a member, has no
cap or a non-positive one. It is the benchmark book ``ActiveRiskOptimizer``
prices active risk against (``ActiveRiskConfig.benchmark``): with the us3000
membership (BarraStyle's ``estu``, the 3,000 largest by their previous-bar
cap) the Barra styles are cap-weighted standardised over the same names, so
this book's style exposures are about 0 and an active exposure bound is close
to an absolute one.
"""

import numpy as np
import xarray as xr

from quantlab.factor.base import Factor
from quantlab.factor.config import MembershipCapWeightConfig

#: The factor's one output, the name ``ActiveRiskConfig.benchmark_weight`` reads by default.
OUTPUT = "benchmark_weight"


class MembershipCapWeight(Factor):
    """Cap weights of an index's members at each bar.

    Parameters
    ----------
    config : MembershipCapWeightConfig
        The market-cap ``dataset``, the ``membership`` and ``cap_column``.

    Examples
    --------
    With ``daily`` a dataset of market caps and ``us3000`` a membership:

    >>> weights = MembershipCapWeight(MembershipCapWeightConfig(dataset=daily, membership=us3000))
    >>> weights.compute("2026-10-01", "2026-10-02")["benchmark_weight"].sum("symbol").values  # doctest: +SKIP
    array([1., 1.])
    """

    config_cls = MembershipCapWeightConfig
    config: MembershipCapWeightConfig

    def _get_factor_names(self) -> tuple[str, ...]:
        return (OUTPUT,)

    def _input_variables(self) -> list[str]:
        return self.config.dataset.own_names([self.config.cap_column])

    def _compute_panel(self, inputs: xr.Dataset) -> xr.Dataset:
        cap = self.config.dataset.to_shared_names(inputs)[self.config.cap_column].transpose("timestamp", "symbol")
        timestamps, symbols = cap["timestamp"].values, cap["symbol"].values
        member = self.config.membership.is_member_at(timestamps, symbols, owner=self.class_name).values
        values = np.where(member & np.isfinite(cap.values) & (cap.values > 0), cap.values, 0.0)
        total = values.sum(axis=1, keepdims=True)
        weight = np.divide(values, total, out=np.zeros_like(values), where=total > 0)
        return xr.Dataset(
            {OUTPUT: (("timestamp", "symbol"), weight)},
            coords={"timestamp": timestamps, "symbol": symbols},
        )


__all__ = ["MembershipCapWeight"]
