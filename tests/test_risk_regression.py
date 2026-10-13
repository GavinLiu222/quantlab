"""The regression store of a factor risk model (#194).

A synthetic panel is planted with factor returns and specific returns:
on every bar t the excess return of each symbol is its exposures of t-1
times the planted factor returns of t plus its planted specific return.
Estimation-universe members carry no specific return, so the weighted
regression must give the planted factor returns back exactly; symbols
outside the universe carry one, which must come back as their specific
return. Everything goes through ``Use4RiskModel.regression``'s public
lifecycle (``compute``, ``build``, ``extend``, ``read``), the model configured
with this panel's factor set in place of ``BarraStyle``'s.

The panel: 60 symbols, four industries (codes 1-4) and two styles. Industry 4
has six members; on a few bars two of them leave the estimation universe, so
it has four members there, below the minimum of five. Eight symbols of
industry 3 are never in the universe. Symbol 1061 is priced but has an
industry code outside the model's, so it has no exposure.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.factor.base import Factor
from quantlab.factor.config import BaseFactorConfig
from quantlab.core.component import rebuild
from quantlab.risk.base import FactorRiskModel
from quantlab.risk.config import FF48_INDUSTRIES, USE4_STYLES, Use4RiskConfig
from quantlab.risk.predefined.use4 import Use4RiskModel

_T = 40
_TIMES = pd.bdate_range("2024-01-01", periods=_T).values
_SYMBOLS = [str(i) for i in range(1001, 1062)]  # 1061 has an unknown industry
_N = len(_SYMBOLS)
_STYLES = ("style_a", "style_b")
_INDUSTRIES = (1, 2, 3, 4)
_CODES = np.array([1] * 18 + [2] * 18 + [3] * 18 + [4] * 6 + [9], dtype=float)
_OUTSIDE = np.arange(40, 48)  # industry 3 symbols never in the universe
_THIN_MEMBERS = np.array([54, 55])  # industry 4 symbols that leave the universe
_THIN_EXPOSURE_BARS = np.arange(20, 26)  # bars whose universe is thin
_THIN_BARS = _THIN_EXPOSURE_BARS + 1  # regression bars using those exposures
_FACTORS = ("country", "industry_1", "industry_2", "industry_3", "industry_4", *_STYLES)


def _day(i: int) -> str:
    return str(np.datetime_as_string(_TIMES[i], unit="D"))


class PassThrough(Factor):
    """The exposures: its dataset's variables, unchanged."""

    config_cls = BaseFactorConfig

    def _get_factor_names(self):
        return (*_STYLES, "industry", "estu")

    def _compute_panel(self, inputs):
        return inputs


def _plant(
    outlier: tuple[int, int, float] | None = None,
    seed: int = 194,
    industries: bool = True,
    autocorrelation: float = 0.0,
    specific_everywhere: float = 0.0,
    jump: tuple[int, float] | None = None,
):
    """Return ``(prices, exposures, planted factor returns, planted specific returns)``.

    ``outlier`` is ``(bar, symbol position, return)``: that estimation-universe
    symbol's specific return on that bar. With ``industries=False`` no
    industry return and no specific return is planted. With
    ``autocorrelation`` the country and style factor returns follow an AR(1)
    with that coefficient. With ``specific_everywhere`` every symbol,
    estimation universe included, gets a specific return of that typical
    size times a volatility of its own (the factor returns are then no
    longer recovered exactly). ``jump`` is ``(bar, factor)``: from that bar
    on, every factor return and specific return is that many times larger.
    """
    rng = np.random.default_rng(seed)
    noise = np.random.default_rng(seed + 1)
    specific_scale = specific_everywhere * noise.lognormal(0.0, 0.5, size=_N)
    cap = rng.lognormal(22.0, 1.0, size=_N) * rng.lognormal(0.0, 0.02, size=(_T, _N))
    styles = rng.normal(size=(_T, _N, len(_STYLES)))
    estu = np.ones((_T, _N))
    estu[:, _OUTSIDE] = 0.0
    estu[np.ix_(_THIN_EXPOSURE_BARS, _THIN_MEMBERS)] = 0.0
    estu[:, -1] = 0.0
    risk_free = 1e-4 + 1e-6 * np.arange(_T)

    factor_returns = np.full((_T, len(_FACTORS)), np.nan)
    specific = np.zeros((_T, _N))
    excess = np.full((_T, _N), np.nan)
    for t in range(1, _T):
        previous = np.nan_to_num(factor_returns[t - 1])
        country = autocorrelation * previous[0] + rng.normal(0, 0.01)
        raw = rng.normal(0, 0.005, size=len(_INDUSTRIES))
        index = _CODES[:-1].astype(int) - 1
        in_fit = estu[t - 1, :-1] == 1.0
        members = np.bincount(index[in_fit], minlength=len(_INDUSTRIES))
        kept = members >= 5
        industry_cap = np.bincount(index[in_fit], weights=cap[t - 1, :-1][in_fit], minlength=4)
        mean = (industry_cap * raw)[kept].sum() / industry_cap[kept].sum()
        industry = np.where(kept, raw - mean, np.nan) if industries else np.zeros(4)
        style = autocorrelation * previous[-len(_STYLES):] + rng.normal(
            0, 0.003, size=len(_STYLES)
        )
        factor_returns[t] = np.concatenate([[country], industry, style])
        specific[t] = np.where(estu[t - 1] == 1.0, 0.0, rng.normal(0, 0.02, size=_N))
        if not industries:
            specific[t] = 0.0
        if specific_everywhere:
            specific[t] = specific_scale * noise.standard_normal(_N)
        if jump is not None and t >= jump[0]:
            factor_returns[t] *= jump[1]
            country, industry, style = country * jump[1], industry * jump[1], style * jump[1]
            specific[t] *= jump[1]
        if outlier is not None and outlier[0] == t:
            specific[t, outlier[1]] = outlier[2]
        industry_part = np.append(np.nan_to_num(industry)[index], 0.0)
        excess[t] = country + industry_part + styles[t - 1] @ style + specific[t]
    specific[0] = np.nan
    specific[:, -1] = np.nan  # no industry exposure, so no specific return

    price = np.empty((_T, _N))
    price[0] = rng.uniform(10, 100, size=_N)
    for t in range(1, _T):
        price[t] = price[t - 1] * (1.0 + excess[t] + risk_free[t - 1])

    coords = {"timestamp": _TIMES, "symbol": _SYMBOLS}
    panel = ("timestamp", "symbol")
    prices = xr.Dataset(
        {
            "adjClose": (panel, price),
            "marketcap": (panel, cap),
            "risk_free": (panel, np.repeat(risk_free[:, None], _N, axis=1)),
        },
        coords=coords,
    )
    exposures = xr.Dataset(
        {
            **{name: (panel, styles[:, :, k]) for k, name in enumerate(_STYLES)},
            "industry": (panel, np.repeat(_CODES[None, :], _T, axis=0)),
            "estu": (panel, estu),
        },
        coords=coords,
    )
    return prices, exposures, factor_returns, specific


def _frame(panel, stored=None, name=""):
    """``panel`` as a dataset: in memory, or in a Zarr store under ``stored``."""
    dataset = FrameDataset(panel)
    return dataset if stored is None else dataset.to_zarr(stored / f"{name}.zarr")


def _exposure_factor(exposures, store=None, stored=None) -> PassThrough:
    return PassThrough(BaseFactorConfig(
        warmup_bars=0, dataset=_frame(exposures, stored, "exposures"), file_path=store,
    ))


def _model(
    prices, exposures, path=None, strategy="cal", exposure_store=None, stored=None, **config
) -> Use4RiskModel:
    fields = dict(industry_name="industry", industries=_INDUSTRIES, estu_name="estu")
    return Use4RiskModel(Use4RiskConfig(
        exposures=_exposure_factor(exposures, exposure_store, stored),
        dataset=_frame(prices, stored, "prices"),
        exposure_data_strategy=strategy,
        regression_path=path,
        style_names=_STYLES,
        **{**fields, **config},
    ))


@pytest.fixture(scope="module")
def planted():
    return _plant()


@pytest.fixture(scope="module")
def rows(planted):
    prices, exposures, _, _ = planted
    return _model(prices, exposures).regression.compute(_day(0), _day(_T - 1))


def test_the_factor_axis_is_country_industries_styles(rows):
    assert tuple(rows["factor"].values.tolist()) == _FACTORS
    assert rows["industry"].values.tolist() == list(_INDUSTRIES)


def test_factor_returns_recover_the_planted_ones(planted, rows):
    _, _, factor_returns, _ = planted
    np.testing.assert_allclose(
        rows["factor_return"].values[1:], factor_returns[1:], rtol=0, atol=1e-12
    )
    # The first bar has no previous bar, so no regression.
    assert np.isnan(rows["factor_return"].values[0]).all()


def test_cap_weighted_industry_returns_sum_to_zero(planted, rows):
    prices, exposures, _, _ = planted
    cap = prices["marketcap"].values
    estu = exposures["estu"].values
    industry = rows["factor_return"].sel(factor=[f"industry_{c}" for c in _INDUSTRIES]).values
    excluded = rows["industry_excluded"].values
    for t in range(1, _T):
        in_fit = estu[t - 1, :-1] == 1.0
        weights = np.bincount(
            _CODES[:-1][in_fit].astype(int) - 1, weights=cap[t - 1, :-1][in_fit], minlength=4
        )
        kept = ~excluded[t]
        weighted = weights[kept] * industry[t, kept]
        assert abs(weighted.sum()) <= 1e-12 * np.abs(weighted).sum()
        assert np.isfinite(industry[t, kept]).all()


def test_a_thin_industry_gets_no_factor_return_and_its_members_specific_returns(planted, rows):
    _, _, _, specific = planted
    thin = rows["factor_return"].sel(factor="industry_4").values
    assert np.isnan(thin[_THIN_BARS]).all()
    others = np.setdiff1d(np.arange(1, _T), _THIN_BARS)
    assert np.isfinite(thin[others]).all()
    excluded = rows["industry_excluded"].sel(industry=4).values
    assert excluded[_THIN_BARS].all() and not excluded[others].any()
    assert (rows["industry_members"].sel(industry=4).values[_THIN_BARS] == 4).all()
    members = [_SYMBOLS[i] for i in range(54, 60)]
    got = rows["specific_return"].sel(symbol=members).values[_THIN_BARS]
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, specific[_THIN_BARS][:, 54:60], rtol=0, atol=1e-12)


def test_symbols_outside_the_estimation_universe_get_specific_returns(planted, rows):
    _, _, _, specific = planted
    outside = [_SYMBOLS[i] for i in _OUTSIDE]
    got = rows["specific_return"].sel(symbol=outside).values[1:]
    np.testing.assert_allclose(got, specific[1:, _OUTSIDE], rtol=0, atol=1e-12)
    assert np.abs(got).max() > 0.01  # planted, not zero


def test_a_symbol_without_exposures_has_no_specific_return(rows):
    assert _SYMBOLS[-1] not in rows["symbol"].values.tolist()


def test_diagnostics(rows):
    assert rows["estu_count"].values[0] == 0
    # 60 symbols with an industry, 8 always outside, 2 more on thin bars plus
    # the 4 left in their thin industry.
    expected = np.full(_T, 52)
    expected[_THIN_BARS] = 52 - 6
    assert rows["estu_count"].values[1:].tolist() == expected[1:].tolist()
    np.testing.assert_allclose(rows["r_squared"].values[1:], 1.0, atol=1e-12)
    assert rows["industry_members"].sel(industry=1).values[1:].tolist() == [18] * (_T - 1)


def test_an_extreme_return_does_not_move_the_factor_returns(planted):
    bar, position = 30, 5  # an industry 1 symbol in the universe
    prices, exposures, _, _ = _plant(outlier=(bar, position, 5.0))
    extreme = _model(prices, exposures).regression.compute(_day(0), _day(_T - 1))
    prices, exposures, _, _ = _plant(outlier=(bar, position, 50.0))
    larger = _model(prices, exposures).regression.compute(_day(0), _day(_T - 1))
    np.testing.assert_array_equal(
        extreme["factor_return"].values[bar], larger["factor_return"].values[bar]
    )
    _, _, planted_returns, _ = planted
    # Trimmed to a bound of a few percent, the outlier barely moves them.
    np.testing.assert_allclose(
        extreme["factor_return"].values[bar], planted_returns[bar], rtol=0, atol=5e-3
    )
    symbol = _SYMBOLS[position]
    assert extreme["specific_return"].sel(symbol=symbol).values[bar] > 4.5
    assert larger["specific_return"].sel(symbol=symbol).values[bar] > 49.5


def test_build_then_extend_equals_build(planted, tmp_path):
    prices, exposures, _, _ = planted
    once = _model(prices, exposures, str(tmp_path / "once.zarr")).regression
    once.build(_day(3), _day(_T - 1))
    twice = _model(prices, exposures, str(tmp_path / "twice.zarr")).regression
    twice.build(_day(3), _day(17)).extend(_day(_T - 1))
    assert once.store_range() == twice.store_range() == (_day(3), _day(_T - 1))
    a = once.read(_day(3), _day(_T - 1)).load()
    b = twice.read(_day(3), _day(_T - 1)).load()
    xr.testing.assert_identical(a, b)
    for name in a.data_vars:
        assert a[name].dtype == b[name].dtype
        np.testing.assert_array_equal(a[name].values, b[name].values)


def test_a_symbol_appearing_later_widens_the_store(planted, tmp_path):
    prices, exposures, _, _ = planted
    late = _SYMBOLS[10]
    prices = prices.copy(deep=True)
    prices["adjClose"].loc[{"timestamp": _TIMES[:25], "symbol": late}] = np.nan
    once = _model(prices, exposures, str(tmp_path / "once.zarr")).regression
    once.build(_day(3), _day(_T - 1))
    twice = _model(prices, exposures, str(tmp_path / "twice.zarr")).regression
    twice.build(_day(3), _day(17)).extend(_day(_T - 1))
    # The late symbol is appended at the end of the axis (ADR 0032): compare by label.
    xr.testing.assert_identical(
        once.read(_day(3), _day(_T - 1)).load().sortby("symbol"),
        twice.read(_day(3), _day(_T - 1)).load().sortby("symbol"),
    )


def test_exposures_read_from_their_store_match_computed_ones(planted, rows, tmp_path):
    prices, exposures, _, _ = planted
    store = str(tmp_path / "exposures.zarr")
    model = _model(prices, exposures, strategy="read", exposure_store=store)
    model.config.exposures.build(_day(0), _day(_T - 1))
    read = model.regression.compute(_day(1), _day(_T - 1))
    xr.testing.assert_identical(read, rows.isel(timestamp=slice(1, None)))


def test_reading_outside_the_recorded_range_is_refused(planted, tmp_path):
    prices, exposures, _, _ = planted
    store = _model(prices, exposures, str(tmp_path / "regression.zarr")).regression
    with pytest.raises(ValueError, match="no recorded range"):
        store.read(_day(5), _day(10))
    store.build(_day(5), _day(20))
    with pytest.raises(ValueError, match="does not contain"):
        store.read(_day(4), _day(10))
    with pytest.raises(ValueError, match="does not contain"):
        store.read(_day(10), _day(21))
    with pytest.raises(ValueError, match="already covers"):
        store.extend(_day(20))


def test_the_model_rebuilds_from_its_config(planted, tmp_path):
    prices, exposures, _, _ = planted
    model = _model(prices, exposures, stored=tmp_path)
    assert Use4RiskModel.from_config(model.get_config()) == model
    # By the class its config names, as a run's config.json is rebuilt.
    assert rebuild(model.get_config()) == model


def test_a_model_without_industries_or_universe_fits_every_symbol():
    prices, exposures, factor_returns, _ = _plant(industries=False)
    model = _model(prices, exposures, industry_name=None, industries=(), estu_name=None)
    assert model.factor_names == ("country", *_STYLES)
    rows = model.regression.compute(_day(0), _day(_T - 1))
    np.testing.assert_allclose(
        rows["factor_return"].values[1:], factor_returns[1:, [0, 5, 6]], rtol=0, atol=1e-12
    )
    # Every symbol is fitted, the one with an unknown industry code included.
    assert rows["estu_count"].values[1:].tolist() == [_N] * (_T - 1)
    assert rows.sizes["industry"] == 0


def test_without_a_country_factor_the_industries_carry_the_market(planted):
    prices, exposures, factor_returns, _ = planted
    model = _model(prices, exposures, country=False, weighting="equal")
    assert model.factor_names == _FACTORS[1:]
    rows = model.regression.compute(_day(0), _day(_T - 1))
    # No constraint: each industry's return is the country's plus its own.
    expected = factor_returns[1:, 1:5] + factor_returns[1:, :1]
    got = rows["factor_return"].values[1:, :4]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        rows["factor_return"].values[1:, 4:], factor_returns[1:, 5:], rtol=0, atol=1e-12
    )


def test_use4_is_the_model_with_barra_style_defaults(planted, tmp_path):
    prices, exposures, _, _ = planted
    model = Use4RiskModel(Use4RiskConfig(
        exposures=_exposure_factor(exposures, stored=tmp_path),
        dataset=_frame(prices, tmp_path, "prices"),
        exposure_data_strategy="cal",
    ))
    assert isinstance(model, FactorRiskModel)
    assert len(model.factor_names) == 61
    assert model.factor_names[:2] == ("country", "industry_1")
    assert model.factor_names[-12:] == USE4_STYLES
    assert model.config.industries == FF48_INDUSTRIES
    assert (model.config.industry_name, model.config.estu_name) == ("industry", "estu")
    assert model.config.weighting == "sqrt_cap"
    # Newey-West: USE4's factor lags; our choices for the factor
    # autocorrelations' half-life and no specific adjustment (#197).
    assert (model.config.volatility_lags, model.config.correlation_lags) == (5, 2)
    assert model.config.volatility_autocorrelation_half_life == 504.0
    assert model.config.specific_lags == 0
    # The 1512-bar correlation window, then the 126 bars of the volatility
    # regime adjustment's bias statistics.
    assert model.estimate_warmup_bars == 1511 + 126
    with_specific = dataclasses.replace(model.config, specific_lags=5)
    assert Use4RiskModel(with_specific).estimate_warmup_bars == 1511 + 126
    longer = dataclasses.replace(with_specific, specific_autocorrelation_window=2000)
    assert Use4RiskModel(longer).estimate_warmup_bars == 1999 + 126
    assert Use4RiskModel(dataclasses.replace(model.config, vra_half_life=None)).estimate_warmup_bars == 1511
    assert Use4RiskModel.from_config(model.get_config()) == model


def test_invalid_parameters_are_refused(planted):
    prices, exposures, _, _ = planted
    config = _model(prices, exposures).config

    with pytest.raises(ValueError, match="min_industry_members"):
        Use4RiskModel(dataclasses.replace(config, min_industry_members=0))
    with pytest.raises(ValueError, match="return_outlier_sigma"):
        Use4RiskModel(dataclasses.replace(config, return_outlier_sigma=0.0))
    with pytest.raises(ValueError, match="weighting"):
        Use4RiskModel(dataclasses.replace(config, weighting="vol"))
    with pytest.raises(ValueError, match="industry_name"):
        Use4RiskModel(dataclasses.replace(config, industry_name=None))
    with pytest.raises(TypeError):
        Use4RiskModel(object())
