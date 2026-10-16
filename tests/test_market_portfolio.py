"""``MarketPortfolio``: the market portfolio's factor exposures and covered share under a factor risk model.

A planted risk model with two factors (market, style) over symbols A..F
and cap weights over A..F and G (ADR 0047, #339). Per bar:

- ``F`` has no style exposure and ``G`` is not on the exposures' axis, so
  neither is covered; ``E`` has no specific risk on ``NO_RISK_BAR``;
- ``D`` is never a member, so it weighs 0 but is covered;
- ``C`` joins the membership on ``JOIN``.

Locked:

- the market exposures are ``B' w`` and the covered share the weight left on
  covered symbols, ``w`` the cap weights of the covered symbols renormalised;
- ``covariance_with_market`` on a bar's forecast, row and cap weights is the
  dense ``Sigma w`` for every covered symbol, a covered non-member included;
- a bar without covered weight is NaN;
- the store built in one go equals one built and then extended, it is read
  only inside its range, and the component rebuilds from its config and
  declares its store;
- a rule declaring it reads the bar's row from its context, and the
  covariance priced from that context is the dense one.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.base import Factor
from quantlab.factor.config import BaseFactorConfig, MembershipCapWeightConfig
from quantlab.factor.predefined.membership_cap_weight import MembershipCapWeight
from quantlab.portfolio.base import InputDeclaration
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.risk.base import FactorRiskModel
from quantlab.risk.config import FactorRiskConfig, MarketPortfolioConfig
from quantlab.risk.market import MarketPortfolio, covariance_with_market
from quantlab.runs.prediction_panel import LabelSpec
from tests.test_membership_mask import COVERAGE_START, _membership

BARS = pd.bdate_range("2024-01-01", periods=12)
SYMBOLS = ["A", "B", "C", "D", "E", "F"]
CAP_SYMBOLS = [*SYMBOLS, "G"]
FACTORS = ("market", "style")
NO_RISK_BAR = 3  # E has no specific risk here
JOIN = BARS[6]  # C is a member from here on
_START, _END = BARS[0].strftime("%Y-%m-%d"), BARS[-1].strftime("%Y-%m-%d")


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


class Exposures(Factor):
    """The exposures: its dataset's ``style``, unchanged."""

    config_cls = BaseFactorConfig

    def _get_factor_names(self):
        return ("style",)

    def _compute_panel(self, inputs):
        return inputs[["style"]]


class RawCaps(Factor):
    """The market caps themselves as weights: they do not sum to 1."""

    config_cls = BaseFactorConfig

    def _get_factor_names(self):
        return ("benchmark_weight",)

    def _compute_panel(self, inputs):
        return inputs.rename(marketcap="benchmark_weight")


@dataclasses.dataclass(kw_only=True, frozen=True)
class StubRiskConfig(FactorRiskConfig):
    """``FactorRiskConfig`` and the Zarr store of the planted estimate rows."""

    planted_estimate: str


class StubRiskModel(FactorRiskModel):
    """A market factor and one style; its estimate rows are the planted ones."""

    config_cls = StubRiskConfig

    @property
    def factor_names(self):
        return FACTORS

    @property
    def exposure_names(self):
        return ("style",)

    def exposure_matrix(self, exposures):
        style = exposures["style"].values
        return np.column_stack([np.ones_like(style), style]), np.isfinite(style)

    @property
    def regression_warmup_bars(self):
        return 0

    @property
    def estimate_warmup_bars(self):
        return 0

    def _compute_regression(self, start, end):
        raise NotImplementedError

    def _compute_estimate(self, start, end):
        rows = xr.open_zarr(self.config.planted_estimate).load()
        return rows.sel(timestamp=slice(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta("1D")))


def plant_risk_model(root, seed: int = 339, *, bars=BARS, symbols=SYMBOLS) -> StubRiskModel:
    """The risk model over ``bars`` and ``symbols``, its estimate store built; F and E as above when present."""
    rng = np.random.default_rng(seed)
    t, n, k = len(bars), len(symbols), len(FACTORS)
    style = rng.normal(size=(t, n))
    if "F" in symbols:
        style[:, list(symbols).index("F")] = np.nan
    roots = rng.normal(0, 0.01, size=(t, k, k))
    covariance = roots @ roots.transpose(0, 2, 1) + 1e-5 * np.eye(k)
    specific_risk = rng.uniform(0.005, 0.02, size=(t, n))
    if "E" in symbols:
        specific_risk[NO_RISK_BAR, list(symbols).index("E")] = np.nan
    panel = ("timestamp", "symbol")
    coords = {"timestamp": bars.values, "symbol": list(symbols)}
    xr.Dataset(
        {
            "factor_covariance": (("timestamp", "factor_i", "factor_j"), covariance),
            "specific_risk": (panel, specific_risk),
        },
        coords={**coords, "factor_i": list(FACTORS), "factor_j": list(FACTORS)},
    ).to_zarr(root / "planted_estimate.zarr", mode="w")
    exposures = FrameDataset(xr.Dataset({"style": (panel, style)}, coords=coords)).to_zarr(
        root / "exposures.zarr"
    )
    prices = FrameDataset(xr.Dataset(
        {name: (panel, np.ones((t, n))) for name in ("adjClose", "marketcap", "risk_free")},
        coords=coords,
    )).to_zarr(root / "prices.zarr")
    model = StubRiskModel(StubRiskConfig(
        exposures=Exposures(BaseFactorConfig(warmup_bars=0, dataset=exposures)),
        dataset=prices,
        exposure_data_strategy="cal",
        estimate_path=str(root / "estimate.zarr"),
        planted_estimate=str(root / "planted_estimate.zarr"),
    ))
    model.estimate.build(_day(bars[0]), _day(bars[-1]))
    return model


def plant_cap_weight(root, seed: int = 7, *, bars=BARS, symbols=CAP_SYMBOLS, intervals=None) -> MembershipCapWeight:
    """Cap weights over ``bars`` and ``symbols``; the membership of the module docstring by default."""
    rng = np.random.default_rng(seed)
    cap = rng.uniform(1e9, 5e9, size=(len(bars), len(symbols)))
    caps = FrameDataset(xr.Dataset(
        {"marketcap": (("timestamp", "symbol"), cap)},
        coords={"timestamp": bars.values, "symbol": list(symbols)},
    )).to_zarr(root / "caps.zarr")
    membership = _membership(root, intervals or [
        *((s, COVERAGE_START, None) for s in ("A", "B", "E", "F", "G")),
        ("C", _day(JOIN), None),
    ])
    return MembershipCapWeight(MembershipCapWeightConfig(dataset=caps, membership=membership))


@pytest.fixture
def market(tmp_path) -> MarketPortfolio:
    return MarketPortfolio(MarketPortfolioConfig(
        risk_model=plant_risk_model(tmp_path),
        cap_weight=plant_cap_weight(tmp_path),
        path=str(tmp_path / "market_portfolio.zarr"),
    ))


def _bar(market: MarketPortfolio, bar: int):
    """The bar's forecast, its cap weights on the forecast's symbols, and all cap weights."""
    model = market.config.risk_model
    t = BARS[bar]
    forecast = model.forecast(
        model.estimate.read(t, t).isel(timestamp=0), model.exposures(t, t).isel(timestamp=0)
    )
    weight = market.config.cap_weight.compute(t, t)["benchmark_weight"].isel(timestamp=0).to_pandas()
    return forecast, weight[list(forecast.symbols)].values, weight


@pytest.mark.parametrize("bar", [0, NO_RISK_BAR, 6, 11])
def test_the_market_exposures_and_covariance_are_the_dense_formulas(market, bar):
    rows = market.market.compute(_START, _END)
    forecast, w, weight = _bar(market, bar)
    expected_symbols = [s for s in SYMBOLS if s != "F" and not (s == "E" and bar == NO_RISK_BAR)]
    assert forecast.symbols.tolist() == expected_symbols
    renormalised = w / w.sum()

    row = rows.isel(timestamp=bar)
    np.testing.assert_allclose(row["market_exposure"].values, forecast.exposures.T @ renormalised, rtol=1e-12)
    np.testing.assert_allclose(float(row["covered_share"]), w.sum() / weight.sum(), rtol=1e-12)
    np.testing.assert_allclose(
        covariance_with_market(forecast, row, w), forecast.covariance @ renormalised, rtol=1e-12
    )
    assert row["factor"].values.tolist() == list(FACTORS)


def test_uncovered_weight_is_dropped_from_the_covered_share(market):
    share = market.market.compute(_START, _END)["covered_share"].to_pandas()
    weight = market.config.cap_weight.compute(_START, _END)["benchmark_weight"].to_pandas()

    np.testing.assert_allclose(share, 1.0 - weight["F"] - weight["G"] - np.where(
        np.arange(len(BARS)) == NO_RISK_BAR, weight["E"], 0.0
    ))
    assert (share < 1.0).all()


def test_a_covered_non_member_gets_the_factor_part_alone(market):
    forecast, w, _ = _bar(market, 2)
    row = market.market.compute(_START, _END).isel(timestamp=2)
    d = forecast.symbols.tolist().index("D")

    assert w[d] == 0
    b, f, _ = forecast.factor_form()
    np.testing.assert_allclose(
        covariance_with_market(forecast, row, w)[d], b[d] @ f @ row["market_exposure"].values, rtol=1e-12
    )


def test_weights_not_summing_to_one_give_the_same_rows(tmp_path):
    model = plant_risk_model(tmp_path)
    shares = plant_cap_weight(tmp_path)
    everyone = MembershipCapWeight(dataclasses.replace(
        shares.config, membership=_membership(tmp_path, [(s, COVERAGE_START, None) for s in CAP_SYMBOLS]),
    ))
    raw = RawCaps(BaseFactorConfig(warmup_bars=0, dataset=shares.config.dataset))

    by_share = MarketPortfolio(MarketPortfolioConfig(risk_model=model, cap_weight=everyone))
    by_cap = MarketPortfolio(MarketPortfolioConfig(risk_model=model, cap_weight=raw))

    xr.testing.assert_allclose(by_cap.market.compute(_START, _END), by_share.market.compute(_START, _END))


def test_a_bar_without_covered_weight_is_nan(tmp_path):
    cap_weight = plant_cap_weight(tmp_path)
    cap_weight = MembershipCapWeight(dataclasses.replace(
        cap_weight.config, membership=_membership(tmp_path, [("A", _day(BARS[5]), None)]),
    ))
    market = MarketPortfolio(MarketPortfolioConfig(risk_model=plant_risk_model(tmp_path), cap_weight=cap_weight))

    rows = market.market.compute(_START, _END)

    assert rows.isel(timestamp=slice(0, 5)).to_array().isnull().all()
    assert rows.isel(timestamp=slice(5, None)).to_array().notnull().all()
    forecast, w, _ = _bar(market, 2)
    assert np.isnan(covariance_with_market(forecast, rows.isel(timestamp=2), w)).all()


def test_a_store_built_then_extended_equals_one_built_at_once(market, tmp_path):
    whole = MarketPortfolio(dataclasses.replace(market.config, path=str(tmp_path / "whole.zarr")))
    whole.market.build(_START, _END)
    market.market.build(_START, _day(BARS[5])).extend(_END)

    xr.testing.assert_allclose(market.market.read(_START, _END), whole.market.read(_START, _END))
    xr.testing.assert_allclose(market.market.read(_START, _END), market.market.compute(_START, _END))
    assert market.market.store_range() == (_START, _END)


def test_the_store_is_read_only_inside_its_range(market):
    market.market.build(_START, _day(BARS[5]))

    with pytest.raises(ValueError, match="covers"):
        market.market.read(_START, _END)


def test_it_rebuilds_from_its_config_and_declares_its_store(market):
    again = rebuild(market.get_config())

    assert again == market
    xr.testing.assert_identical(again.market.compute(_START, _END), market.market.compute(_START, _END))
    ((use),) = market.store_uses()
    assert (use.path, use.relation, use.handle.part) == (market.config.path, "owns", "market")
    assert MarketPortfolio(dataclasses.replace(market.config, path=None)).store_uses() == ()


def test_cap_weights_sharing_no_symbol_with_the_exposures_are_refused(tmp_path):
    other = plant_cap_weight(
        tmp_path, symbols=["X", "Y"], intervals=[("X", COVERAGE_START, None), ("Y", COVERAGE_START, None)]
    )
    market = MarketPortfolio(MarketPortfolioConfig(risk_model=plant_risk_model(tmp_path), cap_weight=other))

    with pytest.raises(ValueError, match="MarketPortfolio: none of the cap weights' 2 symbol"):
        market.market.compute(_START, _END)


def test_a_cap_weight_without_the_named_output_is_refused(market):
    with pytest.raises(ValueError, match="cap_weight_name 'weight'"):
        MarketPortfolio(dataclasses.replace(market.config, cap_weight_name="weight"))


class _DeclaresMarket(TopNConstructor):
    """Top-n declaring a factor risk model and its market portfolio."""

    market = None

    def declared_inputs(self):
        return InputDeclaration(risk_model=self.market.config.risk_model, market=self.market)


def test_a_rule_reads_the_market_row_and_prices_the_covariance_from_its_context(market):
    market.market.build(_START, _END)
    rule = _DeclaresMarket(TopNConfig(direction="long_only", top_n=2))
    rule.market = market
    rule.bind([LabelSpec(name="ret", scale="raw", delay=1, span=1)])
    model = market.config.risk_model
    inputs = DecisionInputs(
        model.config.dataset, rule, fill_column="adjClose", valuation_column="adjClose",
        rebalance_periods=1, anchor=BARS[0],
    )
    t = BARS[7]
    predictions = xr.Dataset({"ret": ("symbol", np.arange(6.0))}, coords={"symbol": SYMBOLS})
    flat = xr.DataArray(np.zeros(6), dims="symbol", coords={"symbol": SYMBOLS})

    context = inputs.context(t, predictions, flat)

    xr.testing.assert_identical(context.market, market.market.read(t, t).isel(timestamp=0, drop=True).load())
    forecast = model.forecast(context.risk_estimate, context.risk_exposures)
    weight = market.config.cap_weight.compute(t, t)["benchmark_weight"].isel(timestamp=0).to_pandas()
    w = weight[list(forecast.symbols)].values
    np.testing.assert_allclose(
        covariance_with_market(forecast, context.market, w), forecast.covariance @ (w / w.sum()), rtol=1e-12
    )
