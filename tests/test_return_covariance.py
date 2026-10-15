"""The interface of a covariance estimate, ``ReturnCovariance`` (#293, #294).

Both estimate types a covariance estimator returns, the dense
``CovarianceEstimate`` and a factor risk model's ``FactorRiskForecast``, satisfy
the protocol; they stay two types (ADR 0028). A forecast answers a factor's
exposure column over any symbols with its own coverage rule (NaN off the
symbols it covers, 0 on a factor it does not carry at the bar); a dense
estimate has no risk factors and refuses. The mean-variance optimiser asks
the estimate and probes none of its fields.
"""

import ast
import inspect

import numpy as np
import pytest

from quantlab.portfolio.base import CovarianceEstimate
from quantlab.portfolio.predefined import mean_variance
from quantlab.risk.base import FactorRiskForecast, ReturnCovariance


def _forecast() -> FactorRiskForecast:
    return FactorRiskForecast(
        symbols=np.array(["AAA", "BBB", "CCC"]),
        factor_names=("market", "value"),
        exposures=np.array([[1.0, 0.2], [0.5, -0.3], [0.8, 0.0]]),
        factor_covariance=np.diag([0.04, 0.01]),
        specific_variance=np.array([0.01, 0.02, 0.03]),
    )


def _dense() -> CovarianceEstimate:
    return CovarianceEstimate(
        symbols=np.array(["AAA", "BBB"]),
        covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
    )


@pytest.mark.parametrize("estimate", [_forecast(), _dense()], ids=["forecast", "dense"])
def test_both_estimate_types_are_a_return_covariance(estimate):
    assert isinstance(estimate, ReturnCovariance)
    assert isinstance(estimate.scaled(5), ReturnCovariance)
    assert isinstance(estimate.subset(np.array([1, 0])), ReturnCovariance)


def test_a_forecast_gives_a_factors_column_over_any_symbols():
    column = _forecast().factor_exposure("value", np.array(["CCC", "AAA"]))
    np.testing.assert_array_equal(column, [0.0, 0.2])
    assert column.dtype == np.float64


def test_a_symbol_the_forecast_does_not_cover_has_no_exposure():
    column = _forecast().factor_exposure("market", np.array(["BBB", "ZZZ"]))
    np.testing.assert_array_equal(column, [0.5, np.nan])


def test_a_factor_the_forecast_does_not_carry_is_zero_on_every_covered_symbol():
    # Dropped at the bar by covered_factors: no covered symbol is exposed to it.
    column = _forecast().factor_exposure("momentum", np.array(["AAA", "ZZZ", "CCC"]))
    np.testing.assert_array_equal(column, [0.0, np.nan, 0.0])


def test_a_dense_estimate_has_no_risk_factor_exposure():
    with pytest.raises(ValueError, match="no risk factors, so no exposure to 'market'"):
        _dense().factor_exposure("market", np.array(["AAA"]))


def test_the_optimizer_reads_no_field_of_the_estimate_beyond_the_protocol():
    # The two estimate types were told apart by probing factor_names and
    # indexing exposures; the optimiser now asks factor_exposure.
    tree = ast.parse(inspect.getsource(mean_variance))
    probed = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "estimate"
    }
    allowed = {"symbols", "covariance", "variance", "scaled", "subset", "factor_form", "factor_exposure"}
    assert probed <= allowed, probed - allowed
    getattr_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "estimate"
    ]
    assert not getattr_calls
