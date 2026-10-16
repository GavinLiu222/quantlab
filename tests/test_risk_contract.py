"""The contract every factor risk model meets, on a model that is not USE4 (#202).

``MarketModel`` is a minimal factor risk model written here: one factor, the
equal-weighted excess return of the estimation universe, every symbol
exposed 1 to it; its estimates are plain sample variances over a short
window. Nothing of USE4 is involved, yet its stores work with the bias
statistics and the portfolio-side store estimator, which read only the
contract of ``FactorRiskModel``. A model whose rows break the contract is
refused when it computes them.

Every factor is a style by default (``factor_groups``); USE4 groups its
country factor, industries and styles (#211). Every factor is labelled by its
name by default (``factor_labels``); USE4 names its industries and drops the
``style_`` prefix (#215).
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.core.component import rebuild
from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.base import PortfolioContext
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.risk.base import FACTOR_GROUPS, FactorRiskModel
from quantlab.risk.bias import risk_model_bias_statistics
from quantlab.risk.config import FactorRiskConfig
from quantlab.risk.predefined.use4 import Use4RiskModel
from tests.test_risk_regression import (
    _SYMBOLS,
    _T,
    _day,
    _exposure_factor,
    _frame,
    _plant,
)
from tests.test_risk_regression import _model as _use4

WINDOW = 5


@dataclasses.dataclass(kw_only=True, frozen=True)
class MarketModelConfig(FactorRiskConfig):
    """``FactorRiskConfig`` and the sample window of the estimates."""

    window: int = WINDOW


class MarketModel(FactorRiskModel):
    """One market factor: the universe's mean excess return; sample-variance forecasts."""

    config_cls = MarketModelConfig

    @property
    def factor_names(self):
        return ("market",)

    @property
    def exposure_names(self):
        return ()

    def exposure_matrix(self, exposures):
        n = exposures.sizes["symbol"]
        return np.ones((n, 1)), np.ones(n, dtype=bool)

    @property
    def regression_warmup_bars(self):
        return 1

    @property
    def estimate_warmup_bars(self):
        return self.config.window - 1

    def _compute_regression(self, start, end):
        config = self.config
        first = pd.Timestamp(start)
        prices = self.prices(_day(0), end)
        estu = self.exposures(_day(0), end)[config.estu_name] == 1.0
        price = prices[config.price_column]
        excess = price / price.shift(timestamp=1) - 1.0 - prices[config.risk_free_column].shift(
            timestamp=1
        )
        market = excess.where(estu.shift(timestamp=1) == 1.0).mean("symbol")
        rows = xr.Dataset({
            "factor_return": market.expand_dims(factor=["market"], axis=1),
            "specific_return": excess - market,
        })
        return rows.sel(timestamp=slice(first, None))

    def _compute_estimate(self, start, end):
        regression = self.regression.read(*self.regression.store_range()).load()
        rows = regression.rolling(timestamp=self.config.window, min_periods=2)
        variance = rows.var()["factor_return"]
        return xr.Dataset({
            "factor_covariance": variance.rename(factor="factor_i").expand_dims(
                factor_j=["market"], axis=2
            ),
            "specific_risk": np.sqrt(rows.var()["specific_return"]),
        }).sel(timestamp=slice(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta("1D")))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    root = tmp_path_factory.mktemp("contract")
    prices, exposures, _, _ = _plant()
    model = MarketModel(MarketModelConfig(
        exposures=_exposure_factor(exposures, stored=root),
        dataset=_frame(prices, root, "prices"),
        exposure_data_strategy="cal",
        estu_name="estu",
        regression_path=str(root / "regression.zarr"),
        estimate_path=str(root / "estimate.zarr"),
    ))
    model.regression.build(_day(1), _day(_T - 1))
    model.estimate.build(_day(8), _day(_T - 1))
    return model


def test_a_minimal_model_builds_both_stores(model):
    regression = model.regression.read(_day(1), _day(_T - 1))
    assert regression["factor_return"].dims == ("timestamp", "factor")
    estimate = model.estimate.read(_day(8), _day(_T - 1))
    assert estimate["factor_covariance"].dims == ("timestamp", "factor_i", "factor_j")
    assert np.isfinite(estimate["factor_covariance"].values).all()
    assert rebuild(model.get_config()) == model


def test_the_bias_statistics_read_any_factor_risk_model(model):
    stats = risk_model_bias_statistics(
        model, _day(8), _day(_T - 1), window=8, random_portfolios=3, portfolio_size=10
    )
    assert stats["factor"]["outcome"].dims == ("timestamp", "factor")
    assert np.isfinite(stats["factor"]["bias"].values).all()
    assert np.isfinite(stats["random"]["outcome"].values).all()


def test_the_store_estimator_reads_any_factor_risk_model(model):
    _, exposures, _, _ = _plant()
    bar = 30
    symbols = np.array(_SYMBOLS)
    on_symbol = {"dims": "symbol", "coords": {"symbol": symbols}}
    context = PortfolioContext(
        timestamp=pd.Timestamp(_day(bar)),
        predictions=xr.Dataset(coords={"symbol": symbols}),
        tradable=xr.DataArray(np.ones(len(symbols), dtype=bool), **on_symbol),
        current_weights=xr.DataArray(np.zeros(len(symbols)), **on_symbol),
        risk_exposures=exposures.isel(timestamp=bar, drop=True),
        risk_estimate=model.estimate.read(_day(bar), _day(bar)).isel(timestamp=0, drop=True),
    )
    reader = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model))
    estimate = reader.estimate(context)
    row = model.estimate.read(_day(bar), _day(bar)).isel(timestamp=0)
    exposures_b, covariance, specific = estimate.factor_form()
    assert covariance.shape == (1, 1)
    np.testing.assert_array_equal(exposures_b, 1.0)
    np.testing.assert_allclose(
        specific, row["specific_risk"].sel(symbol=estimate.symbols).values ** 2
    )


class BrokenModel(MarketModel):
    """``MarketModel`` whose regression rows lack the specific returns."""

    def _compute_regression(self, start, end):
        return super()._compute_regression(start, end).drop_vars("specific_return")


class RenamedFactorModel(MarketModel):
    """``MarketModel`` whose factor axis is not its ``factor_names``."""

    def _compute_regression(self, start, end):
        rows = super()._compute_regression(start, end)
        return rows.assign_coords(factor=["other"])


@pytest.mark.parametrize(
    ("cls", "message"),
    [(BrokenModel, "'specific_return'"), (RenamedFactorModel, "factor_names")],
)
def test_rows_breaking_the_contract_are_refused(model, cls, message):
    broken = cls(model.config)
    with pytest.raises(TypeError, match=message):
        broken.regression.compute(_day(1), _day(10))


def test_the_root_class_cannot_be_constructed(model):
    with pytest.raises(TypeError, match="abstract"):
        FactorRiskModel(FactorRiskConfig(
            exposures=model.config.exposures, dataset=model.config.dataset,
            exposure_data_strategy="cal",
        ))


def test_every_factor_is_a_style_by_default(model):
    assert model.factor_groups() == {"market": "style"}


def test_use4_groups_its_country_industries_and_styles(planted_use4):
    model = planted_use4
    groups = model.factor_groups()
    assert list(groups) == list(model.factor_names)
    assert set(groups.values()) <= set(FACTOR_GROUPS)
    assert groups["country"] == "country"
    assert [name for name, group in groups.items() if group == "industry"] == [
        f"industry_{code}" for code in model.config.industries
    ]
    assert [name for name, group in groups.items() if group == "style"] == list(
        model.config.style_names
    )
    without_country = Use4RiskModel(dataclasses.replace(model.config, country=False))
    assert "country" not in without_country.factor_groups().values()


def test_every_factor_is_labelled_by_its_name_by_default(model):
    assert model.factor_labels() == {"market": "market"}


def test_use4_labels_industries_by_name_and_styles_without_their_prefix(planted_use4):
    labels = planted_use4.factor_labels()
    assert list(labels) == list(planted_use4.factor_names)
    assert labels["country"] == "Country"
    assert [labels[f"industry_{code}"] for code in planted_use4.config.industries] == [
        "Agriculture", "Food Products", "Candy & Soda", "Beer & Liquor"
    ]
    assert labels["style_a"] == "a"
    renamed = Use4RiskModel(dataclasses.replace(
        planted_use4.config, style_names=("style_residual_volatility", "style_b")
    ))
    assert renamed.factor_labels()["style_residual_volatility"] == "residual volatility"


@pytest.fixture(scope="module")
def planted_use4():
    prices, exposures, _, _ = _plant()
    return _use4(prices, exposures)


def test_an_optimiser_asking_the_store_estimator_for_a_volatility_is_refused_at_construction(model):
    from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig
    from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
    from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer

    params = dict(expected_return_label="ret_1", volatility_label="vol_1", expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.2)
    with pytest.raises(ValueError, match="volatility_label.*FactorRiskStoreEstimator"):
        MeanVarianceOptimizer(MeanVarianceConfig(
            covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model)), **params
        ))
    MeanVarianceOptimizer(MeanVarianceConfig(
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)), **params
    ))
