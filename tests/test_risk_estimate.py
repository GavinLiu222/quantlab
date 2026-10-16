"""The estimate store of a factor risk model and the reader on the portfolio side (#195).

The planted panel of ``tests/test_risk_regression.py`` gives a regression
store with a factor left out on some bars (industry 4) and specific returns
for every symbol. The estimate store built from it is checked against
truncated exponentially weighted moments written here directly from the
definition, pair by pair over the bars both factors have, with small windows
so the truncation, the half-lives and the missing returns all matter.

``FactorRiskStoreEstimator`` is checked on a hand-built context at a bar: its
estimate covers exactly the symbols with every style, a model industry and a
specific risk, in factor form, a locked position included. It declares no
factors: the decision inputs hand it the bar's exposures as its risk model
gives them (#230), in ``context.risk_exposures``.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.base import InputDeclaration, PortfolioContext
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig, MeanVarianceConfig
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.runs.prediction_panel import LabelSpec
from tests.test_risk_regression import _INDUSTRIES, _STYLES, _SYMBOLS, _T, _day, _model, _plant

WINDOWS = dict(
    volatility_half_life=4.0,
    volatility_window=8,
    correlation_half_life=8.0,
    correlation_window=12,
    specific_half_life=3.0,
    specific_window=8,
    min_observations=3,
    # The raw model: Newey-West is checked in tests/test_risk_newey_west.py.
    volatility_lags=0,
    correlation_lags=0,
    specific_lags=0,
    specific_autocorrelation_window=8,
    # The time-series specific risk alone: the refinements are checked in
    # tests/test_risk_specific.py.
    structural_model="off",
    shrinkage=0.0,
    # No eigenfactor adjustment: it is checked in tests/test_risk_eigen.py.
    eigen_simulations=0,
    # No volatility regime adjustment: it is checked in tests/test_risk_vra.py.
    vra_half_life=None,
)
START = 1  # the first bar with a regression


def _risk_model(tmp_path, name="risk", planted=None, **config):
    prices, exposures, _, _ = planted or _plant()
    return _model(
        prices, exposures,
        path=str(tmp_path / f"{name}_regression.zarr"),
        estimate_path=str(tmp_path / f"{name}_estimate.zarr"),
        **{**WINDOWS, **config},
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("estimate")
    model = _risk_model(root)
    model.regression.build(_day(START), _day(_T - 1))
    model.estimate.build(_day(START), _day(_T - 1))
    return model


def _moment(x, y, half_life, least):
    """Exponentially weighted covariance of ``x`` and ``y`` over the bars both have."""
    weights = 0.5 ** (np.arange(len(x) - 1, -1, -1) / half_life)
    both = np.isfinite(x) & np.isfinite(y)
    if both.sum() < least:
        return np.nan
    w, x, y = weights[both], x[both], y[both]
    mx, my = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    return (w * (x - mx) * (y - my)).sum() / w.sum()


def _expected_covariance(history):
    least = WINDOWS["min_observations"]
    vol = history[-WINDOWS["volatility_window"]:]
    cor = history[-WINDOWS["correlation_window"]:]
    k = history.shape[1]
    sigma = np.array([np.sqrt(_moment(vol[:, i], vol[:, i], 4.0, least)) for i in range(k)])
    expected = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if i == j:
                rho = 1.0
            else:
                pair = np.isfinite(cor[:, i]) & np.isfinite(cor[:, j])
                xi, xj = np.where(pair, cor[:, i], np.nan), np.where(pair, cor[:, j], np.nan)
                rho = _moment(xi, xj, 8.0, least) / np.sqrt(
                    _moment(xi, xi, 8.0, least) * _moment(xj, xj, 8.0, least)
                )
            expected[i, j] = rho * sigma[i] * sigma[j]
    return expected


def test_the_factor_covariance_matches_truncated_ewma(built):
    regression = built.regression.read(_day(START), _day(_T - 1))
    factor_returns = regression["factor_return"].values
    rows = built.estimate.read(_day(START), _day(_T - 1))
    assert rows["factor_covariance"].dims == ("timestamp", "factor_i", "factor_j")
    got = rows["factor_covariance"].values
    for t in range(len(factor_returns)):
        expected = _expected_covariance(factor_returns[: t + 1])
        np.testing.assert_allclose(got[t], expected, rtol=1e-9, atol=1e-15, err_msg=f"bar {t}")


def test_a_factor_with_missing_returns_uses_its_pairs(built):
    # Industry 4 has no factor return on bars 21-26. Bar 28's volatility
    # window (bars 21-28) holds two of its returns, too few; bar 29's three.
    rows = built.estimate.read(_day(28), _day(29))
    industry = rows["factor_covariance"].sel(factor_i="industry_4", factor_j="industry_4")
    assert np.isnan(industry.values[0]) and np.isfinite(industry.values[1])


def test_the_specific_risk_matches_truncated_ewma(built):
    regression = built.regression.read(_day(START), _day(_T - 1))
    specific = regression["specific_return"]
    rows = built.estimate.read(_day(START), _day(_T - 1))
    got = rows["specific_risk"].sel(symbol=specific["symbol"]).values
    values = specific.values
    least = WINDOWS["min_observations"]
    for t in range(len(values)):
        window = values[: t + 1][-WINDOWS["specific_window"]:]
        expected = [np.sqrt(_moment(u, u, 3.0, least)) for u in window.T]
        np.testing.assert_allclose(got[t], expected, rtol=1e-9, atol=1e-15)
    # The first bars have fewer than min_observations specific returns.
    assert np.isnan(got[: least - 1]).all()


def test_build_then_extend_equals_build(tmp_path):
    once = _risk_model(tmp_path, "once")
    once.regression.build(_day(START), _day(_T - 1))
    once.estimate.build(_day(15), _day(_T - 1))
    twice = _risk_model(tmp_path, "twice")
    twice.regression.build(_day(START), _day(25))
    twice.estimate.build(_day(15), _day(22))
    twice.regression.extend(_day(_T - 1))
    twice.estimate.extend(_day(30)).extend(_day(_T - 1))
    a = once.estimate.read(_day(15), _day(_T - 1)).load()
    b = twice.estimate.read(_day(15), _day(_T - 1)).load()
    xr.testing.assert_identical(a, b)
    for name in a.data_vars:
        np.testing.assert_array_equal(a[name].values, b[name].values)


def test_the_estimate_needs_the_regression_store(tmp_path):
    model = _risk_model(tmp_path)
    with pytest.raises(ValueError, match="regression store has no recorded range"):
        model.estimate.build(_day(5), _day(10))
    model.regression.build(_day(START), _day(20))
    with pytest.raises(ValueError, match="does not contain"):
        model.estimate.build(_day(5), _day(25))


def test_invalid_estimate_parameters_are_refused(built):
    with pytest.raises(ValueError, match="volatility_window"):
        type(built)(dataclasses.replace(built.config, volatility_window=1))
    with pytest.raises(ValueError, match="specific_half_life"):
        type(built)(dataclasses.replace(built.config, specific_half_life=0.0))


# --------------------------------------------------------------------------
# The reader
# --------------------------------------------------------------------------

BAR = 30


def _context(built, *, held=None, untradable=(), drop_style=None, bar=BAR):
    """A hand-built context at ``bar``: the planted exposures and the estimate store's row there."""
    _, exposures, _, _ = _plant()
    risk_exposures = exposures.isel(timestamp=bar, drop=True).copy(deep=True)
    if drop_style is not None:
        risk_exposures[_STYLES[0]].loc[{"symbol": drop_style}] = np.nan
    symbols = np.array(_SYMBOLS)
    weights = np.zeros(len(symbols))
    if held is not None:
        weights[symbols == held] = 0.1
    tradable = ~np.isin(symbols, untradable)
    on_symbol = {"dims": "symbol", "coords": {"symbol": symbols}}
    return PortfolioContext(
        timestamp=pd.Timestamp(_day(bar)),
        predictions=xr.Dataset(
            {"ret_1": ("symbol", np.linspace(-0.01, 0.01, len(symbols)))},
            coords={"symbol": symbols},
        ),
        tradable=xr.DataArray(tradable, **on_symbol),
        current_weights=xr.DataArray(weights, **on_symbol),
        risk_exposures=risk_exposures,
        risk_estimate=built.estimate.read(_day(bar), _day(bar)).isel(timestamp=0, drop=True),
    )


def test_the_reader_declares_its_risk_model_and_no_factor(built):
    reader = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=built))
    assert reader.declared_inputs() == InputDeclaration(risk_model=built)
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
        expected_return_label="ret_1", covariance=reader, expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0, turnover_penalty=0.0, weight_cap=0.5,
    ))
    assert optimizer.declared_inputs().factors == ()
    assert optimizer.declared_inputs().risk_model is built


def test_the_reader_returns_the_bar_in_factor_form(built):
    reader = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=built))
    estimate = reader.estimate(_context(built, drop_style=_SYMBOLS[3]))
    row = built.estimate.read(_day(BAR), _day(BAR)).isel(timestamp=0)
    # Covered: every style, a model industry, a specific risk. Symbol 3 lost a
    # style, the last symbol has an unknown industry.
    expected = [s for s in _SYMBOLS[:-1] if s != _SYMBOLS[3]]
    assert estimate.symbols.tolist() == expected
    exposures, covariance, specific = estimate.factor_form()
    np.testing.assert_array_equal(covariance, row["factor_covariance"].values)
    np.testing.assert_allclose(
        specific, row["specific_risk"].sel(symbol=expected).values ** 2, rtol=0
    )
    _, planted_exposures, _, _ = _plant()
    at = planted_exposures.isel(timestamp=BAR).sel(symbol=expected)
    industry = at["industry"].values
    np.testing.assert_array_equal(exposures[:, 0], 1.0)
    for j, code in enumerate(_INDUSTRIES):
        np.testing.assert_array_equal(exposures[:, 1 + j], (industry == code).astype(float))
    for k, name in enumerate(_STYLES):
        np.testing.assert_array_equal(exposures[:, 1 + len(_INDUSTRIES) + k], at[name].values)


def test_a_factor_without_a_variance_leaves_out_its_symbols(built):
    # Bar 28: industry 4 has too few returns in its volatility window.
    reader = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=built))
    context = _context(built, bar=28)
    estimate = reader.estimate(context)
    exposures, covariance, _ = estimate.factor_form()
    assert covariance.shape == (len(built.factor_names) - 1,) * 2
    assert np.isfinite(covariance).all()
    industry_4 = set(_SYMBOLS[54:60])
    assert not industry_4 & set(estimate.symbols.tolist())
    assert len(estimate.symbols) == 54


def test_a_locked_position_is_priced_for_risk(built):
    locked = _SYMBOLS[7]
    context = _context(built, held=locked, untradable=[locked])
    assert bool(context.locked.sel(symbol=locked))
    reader = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=built))
    assert locked in reader.estimate(context).symbols.tolist()
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
        expected_return_label="ret_1", covariance=reader, expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0, turnover_penalty=0.0, weight_cap=0.5,
    ))
    optimizer.bind([LabelSpec(name="ret_1", scale="raw", delay=1, span=1)])
    inputs = optimizer.problem_inputs(context)
    assert inputs.locked_symbols.tolist() == [locked]
    np.testing.assert_array_equal(inputs.risk_locked_weights, [0.1])
    # The locked symbol is in the estimate the optimiser prices risk with.
    assert inputs.estimate.symbols.tolist()[-1] == locked


def test_the_reader_reads_only_the_context(built, monkeypatch):
    context = _context(built)
    reader = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=built))
    expected = reader.estimate(context)
    # No store is opened: the forecast is the model's forecast of the context's row.
    monkeypatch.setattr(xr, "open_zarr", lambda *a, **k: pytest.fail("the estimator opened a store"))
    for x, y in zip(reader.estimate(context).factor_form(), expected.factor_form()):
        np.testing.assert_array_equal(x, y)
    # A hand-built row decides the forecast: doubled specific risks, four times the variances.
    doubled = context.risk_estimate.assign(specific_risk=2.0 * context.risk_estimate["specific_risk"])
    forecast = reader.estimate(dataclasses.replace(context, risk_estimate=doubled))
    np.testing.assert_allclose(forecast.specific_variance, 4.0 * expected.specific_variance, rtol=0)


def test_the_reader_refuses_a_context_without_an_estimate_row_and_volatilities(built):
    reader = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=built))
    context = _context(built)
    with pytest.raises(ValueError, match="no predicted volatilities"):
        reader.estimate(context, volatility=context.current_weights)
    with pytest.raises(ValueError, match="no risk estimate: the estimate store at .* covers"):
        reader.estimate(dataclasses.replace(context, risk_estimate=None))
    with pytest.raises(ValueError, match="no risk exposures"):
        reader.estimate(dataclasses.replace(context, risk_exposures=None))


def test_a_bound_named_by_a_factor_and_the_risk_model_is_refused(built):
    reader = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=built))
    with pytest.raises(ValueError, match="both an output"):
        MeanVarianceOptimizer(MeanVarianceConfig(
            expected_return_label="ret_1", covariance=reader, expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
            risk_aversion=5.0, turnover_penalty=0.0, weight_cap=0.5,
            exposure_factors=(built.config.exposures,),
            exposure_bounds={_STYLES[0]: (-0.1, 0.1)},
        ))
