"""Specific-risk refinements of the USE4 model: structural blend and Bayesian shrinkage (#199).

The planted panel of ``tests/test_risk_regression.py`` with a specific return
for every symbol, each of its own volatility, and a spike in one symbol's
specific returns (a fat-tailed history). One more symbol has exposures but
no price, so no specific return at all. The estimate store's specific risk
is checked against USE4 eqs. 5.3-5.9 written here from the definitions: the
time-series volatility, the blending coefficient, the structural regression
of log volatility on the exposures over the symbols with a coefficient of 1,
the blend, and the shrinkage toward the cap-weighted mean of a size group.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.base import PortfolioContext
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from tests.test_risk_estimate import START, WINDOWS, _moment
from tests.test_risk_regression import _N, _SYMBOLS, _T, _day, _model, _plant

SPIKE = (25, 3, 0.8)  # bar, symbol position, specific return
NEW = "1099"  # exposures, never a price
REFINE = dict(
    specific_window=20,
    specific_half_life=10.0,
    blending_min_observations=6,
    blending_ramp=6,
    structural_model="blend",
    structural_fit="blending",
    structural_bias=1.05,
    structural_history_window=None,
    shrinkage=0.0,
)
LATE = (np.arange(8, 20), 14)  # symbol positions first priced at this bar (#203)
HISTORY = 15  # the history-length regressor's window in the #203 tests
BARS = (22, 30, 38)


def _panel(late=False):
    prices, exposures, _, _ = _plant(outlier=SPIKE, specific_everywhere=0.02)
    if late:
        positions, first = LATE
        price = prices["adjClose"].values.copy()
        price[:first, positions] = np.nan
        prices = prices.assign(adjClose=(prices["adjClose"].dims, price))
    new = exposures.isel(symbol=[5]).assign_coords(symbol=[NEW])
    return prices, xr.concat([exposures, new], dim="symbol")


def _built(tmp_path, name, late=False, **config):
    prices, exposures = _panel(late)
    model = _model(
        prices, exposures,
        path=str(tmp_path / f"{name}_regression.zarr"),
        estimate_path=str(tmp_path / f"{name}_estimate.zarr"),
        **{**WINDOWS, **REFINE, **config},
    )
    model.regression.build(_day(START), _day(_T - 1))
    model.estimate.build(_day(START), _day(_T - 1))
    return model, prices, exposures


def _time_series(model, t):
    """Each symbol's time-series specific volatility at bar ``t`` (no Newey-West)."""
    regression = model.regression.read(_day(START), _day(t))
    u = regression["specific_return"].values[-REFINE["specific_window"]:]
    risk = [np.sqrt(_moment(x, x, REFINE["specific_half_life"], WINDOWS["min_observations"])) for x in u.T]
    return regression["symbol"].values, np.array(risk), u


def _gamma(u):
    """The blending coefficient of each column of ``u``, from its definition."""
    out = []
    for x in u.T:
        x = x[np.isfinite(x)]
        h = len(x)
        if h <= REFINE["blending_min_observations"]:
            out.append(0.0)
            continue
        q1, q3 = np.percentile(x, [25, 75])
        robust = (q3 - q1) / 1.35
        if not robust > 0:
            out.append(0.0)
            continue
        spread = np.std(np.clip(x, -10 * robust, 10 * robust), ddof=1)
        z = abs(spread / robust - 1)
        coverage = min(1.0, max(0.0, (h - REFINE["blending_min_observations"]) / REFINE["blending_ramp"]))
        out.append(coverage * min(1.0, max(0.0, np.exp(1 - z))))
    return np.array(out)


def _expected(
    model, prices, exposures, t, shrinkage=0.0, groups=10, structural="blend",
    fit_set="blending", bias=REFINE["structural_bias"], history=None,
):
    """The specific risk of every symbol at bar ``t`` from USE4 eqs. 5.3-5.9.

    ``fit_set`` is ``"blending"`` (USE4: the symbols with a coefficient of 1)
    or ``"series"`` (every symbol with a time series); ``bias`` a number or
    ``"smearing"``; ``history`` the window of the history-length regressor
    ``log(1 + h / 252)``, ``h`` the symbol's specific returns in it (#203).
    """
    symbols, sigma_ts, u = _time_series(model, t)
    gamma = _gamma(u)
    all_symbols = np.array(sorted(set(symbols) | {NEW}, key=int))
    sigma = pd.Series(sigma_ts, index=symbols).reindex(all_symbols).values
    gamma = pd.Series(gamma, index=symbols).reindex(all_symbols).fillna(0.0).values
    at = exposures.sel(timestamp=_day(t)).reindex(symbol=all_symbols)
    matrix, covered = model.exposure_matrix(at)
    cap = prices["marketcap"].sel(timestamp=_day(t)).reindex(symbol=all_symbols).values
    if structural != "off":
        if history is not None:
            returns = model.regression.read(_day(START), _day(t))["specific_return"]
            h = np.isfinite(returns.values[-history:]).sum(axis=0)
            h = pd.Series(h, index=returns["symbol"].values).reindex(all_symbols).fillna(0).values
            matrix = np.column_stack([matrix, np.log1p(h / 252)])
        fit = covered & np.isfinite(sigma) & (sigma > 0) & np.isfinite(cap)
        if fit_set == "blending":
            fit &= gamma >= 1
        root = np.sqrt(np.sqrt(cap[fit]))
        b = np.linalg.lstsq(matrix[fit] * root[:, None], np.log(sigma[fit]) * root, rcond=None)[0]
        if bias == "smearing":
            bias = np.mean(np.exp(np.log(sigma[fit]) - matrix[fit] @ b))
        structural_risk = np.where(covered, bias * np.exp(matrix @ b), np.nan)
        if structural == "blend":
            blended = np.where(
                gamma >= 1, sigma, gamma * np.nan_to_num(sigma) + (1 - gamma) * structural_risk
            )
        else:
            blended = np.where(np.isfinite(sigma), sigma, structural_risk)
        sigma = np.where(np.isfinite(blended), blended, sigma)
    if shrinkage:
        usable = np.flatnonzero(np.isfinite(sigma) & np.isfinite(cap))
        order = usable[np.argsort(cap[usable], kind="stable")]
        out = sigma.copy()
        for group in np.array_split(order, groups):
            w = cap[group] / cap[group].sum()
            m = (w * sigma[group]).sum()
            d = np.sqrt(((sigma[group] - m) ** 2).mean())
            nu = shrinkage * np.abs(sigma[group] - m) / (d + shrinkage * np.abs(sigma[group] - m))
            out[group] = nu * m + (1 - nu) * sigma[group]
        sigma = out
    return pd.Series(sigma, index=all_symbols), pd.Series(gamma, index=all_symbols)


@pytest.fixture(scope="module")
def blended(tmp_path_factory):
    return _built(tmp_path_factory.mktemp("specific"), "blended")


def test_the_blend_follows_the_structural_model(blended):
    model, prices, exposures = blended
    rows = model.estimate.read(_day(START), _day(_T - 1))["specific_risk"]
    for t in BARS:
        expected, gamma = _expected(model, prices, exposures, t)
        got = rows.sel(timestamp=_day(t)).to_series().reindex(expected.index)
        np.testing.assert_allclose(got.values, expected.values, rtol=1e-9, err_msg=f"bar {t}")
        # Clean histories keep their time-series value; the new symbol and the
        # spiked one take (some of) the structural value.
        assert (gamma > 0.99).sum() > _N // 2
        assert gamma[NEW] == 0.0
        if t >= SPIKE[0]:
            assert gamma[_SYMBOLS[SPIKE[1]]] < 1.0


def test_a_clean_history_keeps_its_time_series_value(blended):
    model, _, _ = blended
    t = BARS[1]
    symbols, sigma_ts, u = _time_series(model, t)
    gamma = _gamma(u)
    clean = symbols[gamma >= 1.0]
    row = model.estimate.read(_day(t), _day(t))["specific_risk"].isel(timestamp=0)
    np.testing.assert_allclose(
        row.sel(symbol=clean).values, pd.Series(sigma_ts, index=symbols)[clean].values, rtol=1e-9
    )


def test_every_symbol_with_exposures_has_a_specific_risk(blended):
    model, _, exposures = blended
    row = model.estimate.read(_day(30), _day(30))["specific_risk"].isel(timestamp=0)
    assert NEW in row["symbol"].values.tolist()
    assert np.isfinite(float(row.sel(symbol=NEW)))
    # The store estimator covers it too.
    symbols = np.array(row["symbol"].values)
    on_symbol = {"dims": "symbol", "coords": {"symbol": symbols}}
    context = PortfolioContext(
        timestamp=pd.Timestamp(_day(30)),
        predictions=xr.Dataset(coords={"symbol": symbols}),
        tradable=xr.DataArray(np.ones(len(symbols), dtype=bool), **on_symbol),
        current_weights=xr.DataArray(np.zeros(len(symbols)), **on_symbol),
        risk_exposures=exposures.sel(timestamp=_day(30), drop=True),
        risk_estimate=model.estimate.read(_day(30), _day(30)).isel(timestamp=0, drop=True),
    )
    estimator = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model))
    assert NEW in estimator.estimate(context).symbols.tolist()


def test_fill_gives_the_structural_value_only_to_symbols_without_a_time_series(tmp_path):
    model, prices, exposures = _built(tmp_path, "fill", structural_model="fill")
    rows = model.estimate.read(_day(START), _day(_T - 1))["specific_risk"]
    for t in BARS:
        expected, _ = _expected(model, prices, exposures, t, structural="fill")
        got = rows.sel(timestamp=_day(t)).to_series().reindex(expected.index)
        np.testing.assert_allclose(got.values, expected.values, rtol=1e-9, err_msg=f"bar {t}")
    # The spiked symbol keeps its time series; the new one is structural.
    symbols, sigma_ts, _ = _time_series(model, BARS[-1])
    spiked = _SYMBOLS[SPIKE[1]]
    row = rows.sel(timestamp=_day(BARS[-1]))
    assert float(row.sel(symbol=spiked)) == pytest.approx(
        pd.Series(sigma_ts, index=symbols)[spiked], rel=1e-12
    )
    assert np.isfinite(float(row.sel(symbol=NEW)))


def test_the_default_fills_and_shrinks(blended):
    model, _, _ = blended
    defaults = type(model.config)(exposures=model.config.exposures, dataset=model.config.dataset,
                                  exposure_data_strategy="cal")
    assert (defaults.structural_model, defaults.shrinkage, defaults.shrinkage_groups) == ("fill", 0.1, 10)
    assert (
        defaults.structural_fit, defaults.structural_bias, defaults.structural_history_window
    ) == ("series", "smearing", 756)
    with pytest.raises(ValueError, match="structural_model"):
        type(model)(dataclasses.replace(model.config, structural_model="yes"))


def test_shrinkage_matches_a_hand_computed_reference(tmp_path):
    model, prices, exposures = _built(
        tmp_path, "shrunk", structural_model="off", shrinkage=0.3, shrinkage_groups=4
    )
    rows = model.estimate.read(_day(START), _day(_T - 1))["specific_risk"]
    for t in BARS:
        expected, _ = _expected(
            model, prices, exposures, t, shrinkage=0.3, groups=4, structural="off"
        )
        got = rows.sel(timestamp=_day(t)).to_series().reindex(expected.index)
        np.testing.assert_allclose(got.values, expected.values, rtol=1e-9, err_msg=f"bar {t}")


def test_the_full_chain_blends_then_shrinks(tmp_path):
    model, prices, exposures = _built(tmp_path, "chain", shrinkage=0.1, shrinkage_groups=5)
    rows = model.estimate.read(_day(START), _day(_T - 1))["specific_risk"]
    for t in BARS:
        expected, _ = _expected(model, prices, exposures, t, shrinkage=0.1, groups=5)
        got = rows.sel(timestamp=_day(t)).to_series().reindex(expected.index)
        np.testing.assert_allclose(got.values, expected.values, rtol=1e-9, err_msg=f"bar {t}")


def test_the_refined_store_does_not_depend_on_njobs(tmp_path, blended):
    model, _, _ = blended
    other, _, _ = _built(tmp_path, "jobs", njobs=3)
    a = model.estimate.read(_day(START), _day(_T - 1)).load()
    b = other.estimate.read(_day(START), _day(_T - 1)).load()
    xr.testing.assert_identical(a, b)


def test_invalid_refinement_parameters_are_refused(blended):
    model, _, _ = blended
    for field, value in (
        ("structural_bias", 0.0), ("structural_bias", "mean"), ("structural_fit", "all"),
        ("structural_bias", True), ("structural_history_window", 0),
        ("structural_history_window", 2.5), ("blending_ramp", 0), ("blending_outlier_bound", 0.0),
        ("shrinkage", -0.1), ("shrinkage_groups", 0),
    ):
        with pytest.raises(ValueError, match=field):
            type(model)(dataclasses.replace(model.config, **{field: value}))


def test_build_then_extend_equals_build_with_the_refinements(tmp_path):
    refined = dict(structural_model="fill", shrinkage=0.1, shrinkage_groups=4)
    once, _, _ = _built(tmp_path, "once", **refined)
    prices, exposures = _panel()
    twice = _model(
        prices, exposures,
        path=str(tmp_path / "twice_regression.zarr"),
        estimate_path=str(tmp_path / "twice_estimate.zarr"),
        **{**WINDOWS, **REFINE, **refined},
    )
    twice.regression.build(_day(START), _day(25))
    twice.estimate.build(_day(START), _day(22))
    twice.regression.extend(_day(_T - 1))
    twice.estimate.extend(_day(31)).extend(_day(_T - 1))
    a = once.estimate.read(_day(START), _day(_T - 1)).load()
    b = twice.estimate.read(_day(START), _day(_T - 1)).load()
    xr.testing.assert_identical(a, b)


def test_shrinkage_pulls_every_forecast_toward_its_group_mean_without_crossing(tmp_path):
    plain, _, _ = _built(tmp_path, "plain", structural_model="off", shrinkage=0.0)
    shrunk, prices, _ = _built(tmp_path, "pulled", structural_model="off", shrinkage=0.5, shrinkage_groups=1)
    t = BARS[1]
    before = plain.estimate.read(_day(t), _day(t))["specific_risk"].isel(timestamp=0).to_series().dropna()
    after = shrunk.estimate.read(_day(t), _day(t))["specific_risk"].isel(timestamp=0).to_series()[before.index]
    cap = prices["marketcap"].sel(timestamp=_day(t)).to_series()[before.index]
    target = (cap * before).sum() / cap.sum()
    # One group: every forecast moves toward the cap-weighted mean, stays on its side.
    assert ((after - target).abs() <= (before - target).abs() + 1e-15).all()
    assert (np.sign(after - target) == np.sign(before - target)).all()
    assert ((after - target).abs() < (before - target).abs()).sum() > len(before) // 2


#: The #203 structural model: fitted on every symbol with a time series,
#: smearing ``E_0`` and the history-length regressor.
CALIBRATED = dict(
    structural_model="fill", structural_fit="series", structural_bias="smearing",
    structural_history_window=HISTORY,
)


def test_the_calibrated_structural_model_matches_its_definition(tmp_path):
    model, prices, exposures = _built(tmp_path, "calibrated", late=True, **CALIBRATED)
    rows = model.estimate.read(_day(START), _day(_T - 1))["specific_risk"]
    positions, first = LATE
    # Before a late listing has min_observations returns it is structural only.
    for t in (first + 1, first + 2, *BARS):
        expected, _ = _expected(
            model, prices, exposures, t, structural="fill", fit_set="series",
            bias="smearing", history=HISTORY,
        )
        got = rows.sel(timestamp=_day(t)).to_series().reindex(expected.index)
        np.testing.assert_allclose(got.values, expected.values, rtol=1e-9, err_msg=f"bar {t}")
    late = [_SYMBOLS[i] for i in positions]
    assert rows.sel(timestamp=_day(first + 1), symbol=late).notnull().all()


def test_the_history_regressor_raises_a_short_history_s_structural_value(tmp_path):
    """Late listings move more than their exposures say only through the regressor."""
    with_history, _, _ = _built(tmp_path, "with", late=True, **CALIBRATED)
    without, _, _ = _built(
        tmp_path, "without", late=True, **{**CALIBRATED, "structural_history_window": None}
    )
    _, first = LATE
    a = with_history.estimate.read(_day(first + 1), _day(first + 1))["specific_risk"]
    b = without.estimate.read(_day(first + 1), _day(first + 1))["specific_risk"]
    assert not np.allclose(a.values, b.values, equal_nan=True)


def test_build_then_extend_equals_build_with_the_calibrated_structural_model(tmp_path):
    once, _, _ = _built(tmp_path, "once", late=True, **CALIBRATED)
    prices, exposures = _panel(late=True)
    twice = _model(
        prices, exposures,
        path=str(tmp_path / "twice_regression.zarr"),
        estimate_path=str(tmp_path / "twice_estimate.zarr"),
        **{**WINDOWS, **REFINE, **CALIBRATED},
    )
    twice.regression.build(_day(START), _day(25))
    twice.estimate.build(_day(START), _day(22))
    twice.regression.extend(_day(_T - 1))
    twice.estimate.extend(_day(31)).extend(_day(_T - 1))
    a = once.estimate.read(_day(START), _day(_T - 1)).load()
    b = twice.estimate.read(_day(START), _day(_T - 1)).load()
    xr.testing.assert_identical(a, b)


def test_the_calibrated_store_does_not_depend_on_njobs(tmp_path):
    one, _, _ = _built(tmp_path, "one", late=True, **CALIBRATED)
    three, _, _ = _built(tmp_path, "three", late=True, njobs=3, **CALIBRATED)
    xr.testing.assert_identical(
        one.estimate.read(_day(START), _day(_T - 1)).load(),
        three.estimate.read(_day(START), _day(_T - 1)).load(),
    )
