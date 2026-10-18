"""``NeutralizedFactor``: a factor's outputs neutralized against industry and size (#190).

The wrapper takes another factor's outputs (already cross-sectionally
z-scored), neutralizes each against the industry and the log market cap of
its exposures dataset with ``CrossSectionalNeutralize``, and z-scores the
residual again, under the same output names. The reference here is written
from that definition with ``numpy.linalg.lstsq`` on explicit industry
dummies, independent of the KunQuant operators.

The inner factor is a small ``FactorKunQuant`` over an in-memory price panel
of 13 symbols. The exposures panel has a symbol the prices lack and lacks
one the prices have, a non-positive market cap, missing industry codes and a
missing timestamp, so alignment and missing exposures are exercised.
"""

import dataclasses
import json

import numpy as np
import pytest
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig, NeutralizedConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.kunquant_cs import CrossSectionalZScore
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.xgb import XGBoostRegressor
from tests.label_stubs import StubLabel

_T = 30
_SYMBOLS = [f"S{i:02d}" for i in range(13)]
_TIMES = np.datetime64("2024-01-01") + np.arange(_T).astype("timedelta64[D]")
_START, _END = "2024-01-01", "2024-01-30"
_NO_EXPOSURES = "S12"  # priced, but absent from the exposures panel
_GAP = 7  # a bar the exposures panel does not have


def _day(i: int) -> str:
    return str(np.datetime_as_string(_TIMES[i], unit="D"))


class TwoSignals(FactorKunQuant):
    """``sig_a`` and ``sig_b``: cross-sectional z-scores of the close and the volume."""

    def _get_factor_names(self):
        return ("sig_a", "sig_b")

    def _get_factor_func(self):
        b = Builder()
        with b:
            Output(CrossSectionalZScore(Input("adjClose")), "sig_a")
            Output(CrossSectionalZScore(Input("adjVolume")), "sig_b")
        return Function(b.ops)


def _prices() -> xr.Dataset:
    rng = np.random.default_rng(190)
    shape = (_T, len(_SYMBOLS))
    industry = rng.integers(0, 3, size=len(_SYMBOLS)).astype(float)
    cap = rng.lognormal(22.0, 1.0, size=len(_SYMBOLS))
    # Signals carrying industry and size, so neutralizing visibly changes them.
    close = 50 + 5 * industry + 3 * np.log(cap) + rng.normal(0, 1, size=shape)
    volume = rng.lognormal(10, 1, size=shape)
    close[3, 2] = np.nan
    return xr.Dataset(
        {"adjClose": (("timestamp", "symbol"), close), "adjVolume": (("timestamp", "symbol"), volume)},
        coords={"timestamp": _TIMES, "symbol": _SYMBOLS},
    ), industry, cap


def _exposures(industry: np.ndarray, cap: np.ndarray) -> xr.Dataset:
    rng = np.random.default_rng(1900)
    symbols = [s for s in _SYMBOLS if s != _NO_EXPOSURES] + ["EXTRA"]
    n = len(symbols)
    marketcap = np.tile(np.append(cap[:-1], 1e9), (_T, 1)) * rng.lognormal(0, 0.05, size=(_T, n))
    codes = np.tile(np.append(industry[:-1], 0.0), (_T, 1))
    marketcap[5, 1] = np.nan
    marketcap[6, 3] = 0.0  # log is -inf: missing
    codes[9, 4] = np.nan
    times = np.delete(_TIMES, _GAP)
    keep = np.arange(_T) != _GAP
    return xr.Dataset(
        {
            "marketcap": (("timestamp", "symbol"), marketcap[keep]),
            "industry": (("timestamp", "symbol"), codes[keep]),
        },
        coords={"timestamp": times, "symbol": symbols},
    )


@pytest.fixture(scope="module")
def panels():
    prices, industry, cap = _prices()
    return prices, _exposures(industry, cap)


def _inner(prices: xr.Dataset, **config) -> TwoSignals:
    return TwoSignals(FactorConfig(
        warmup_bars=0, dataset=FrameDataset(prices), mode="batch",
        data_columns=("adjClose", "adjVolume"), njobs=2, **config,
    ))


def _wrapper(panels, tmp_path=None, **config) -> NeutralizedFactor:
    prices, exposures = panels
    file_path = None if tmp_path is None else str(tmp_path / "neutral.zarr")
    return NeutralizedFactor(NeutralizedConfig(
        factor=_inner(prices), dataset=FrameDataset(exposures), file_path=file_path,
        njobs=2, **config,
    ))


def _reference(inner: xr.Dataset, exposures: xr.Dataset, regressors) -> dict[str, np.ndarray]:
    """Neutralize and z-score each variable of ``inner`` bar by bar with ``lstsq``."""
    aligned = exposures.reindex(timestamp=inner["timestamp"], symbol=inner["symbol"])
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.log(aligned["marketcap"].values)
    industry = aligned["industry"].values
    out = {}
    for name in inner.data_vars:
        y = inner[name].values.astype(np.float64)
        result = np.full(y.shape, np.nan)
        for t in range(y.shape[0]):
            ok = np.isfinite(y[t])
            if "size" in regressors:
                ok &= np.isfinite(size[t])
            if "industry" in regressors:
                ok &= np.isfinite(industry[t])
            if ok.sum() < 2:
                continue
            columns = []
            if "industry" in regressors:
                columns += [(industry[t, ok] == c).astype(float) for c in np.unique(industry[t, ok])]
            else:
                columns.append(np.ones(ok.sum()))
            if "size" in regressors:
                columns.append(size[t, ok])
            design = np.column_stack(columns)
            coef, *_ = np.linalg.lstsq(design, y[t, ok], rcond=None)
            residual = y[t, ok] - design @ coef
            sd = residual.std(ddof=1)
            if sd > 0:
                result[t, ok] = (residual - residual.mean()) / sd
        out[name] = result
    return out


@pytest.mark.parametrize("regressors", [("industry", "size"), ("industry",), ("size",)])
def test_outputs_are_the_z_scored_lstsq_residuals_under_the_inner_names(panels, regressors) -> None:
    factor = _wrapper(panels, regressors=regressors)
    got = factor.compute(_START, _END)
    inner = factor.config.factor.compute(_START, _END)
    assert factor.get_factor_names() == ("sig_a", "sig_b")
    assert list(got.data_vars) == ["sig_a", "sig_b"]
    assert list(got["symbol"].values) == _SYMBOLS
    want = _reference(inner, panels[1], regressors)
    for name in ("sig_a", "sig_b"):
        np.testing.assert_array_equal(np.isnan(got[name].values), np.isnan(want[name]))
        finite = np.isfinite(want[name])
        np.testing.assert_allclose(got[name].values[finite], want[name][finite], rtol=1e-4, atol=1e-4)
    assert not np.allclose(
        np.nan_to_num(got["sig_a"].values), np.nan_to_num(inner["sig_a"].values)
    ), "neutralizing a signal built from industry and size should change it"


def test_missing_exposures_give_nan(panels) -> None:
    got = _wrapper(panels).compute(_START, _END)["sig_a"]
    assert np.isnan(got.sel(symbol=_NO_EXPOSURES)).all()
    assert np.isnan(got.isel(timestamp=_GAP)).all()
    assert np.isnan(got.isel(timestamp=5, symbol=1))  # market cap missing
    assert np.isnan(got.isel(timestamp=6, symbol=3))  # market cap 0
    assert np.isnan(got.isel(timestamp=9, symbol=4))  # industry missing
    assert np.isfinite(got.isel(timestamp=10, symbol=4))


def test_industry_only_reads_no_market_cap(panels) -> None:
    prices, exposures = panels
    factor = _wrapper((prices, exposures.drop_vars("marketcap")), regressors=("industry",))
    got = factor.compute(_START, _END)["sig_a"]
    assert np.isfinite(got.isel(timestamp=6, symbol=3))  # the zero cap does not matter


def test_a_missing_exposure_column_is_named(panels) -> None:
    prices, exposures = panels
    factor = _wrapper((prices, exposures.drop_vars("industry")))
    with pytest.raises(KeyError, match="industry"):
        factor.compute(_START, _END)


def test_build_read_and_extend_answer_like_compute(panels, tmp_path) -> None:
    factor = _wrapper(panels, tmp_path)
    factor.build(_START, _day(19))
    assert factor.store_range() == (_START, _day(19))
    factor.extend(_END)
    read = factor.read(_START, _END)
    computed = factor.compute(_START, _END)
    for name in ("sig_a", "sig_b"):
        np.testing.assert_allclose(read[name].values, computed[name].values, equal_nan=True)


def test_keep_compiled_keeps_the_wrapped_factor_compiled(panels) -> None:
    factor = _wrapper(panels)
    inner = factor.config.factor
    with factor.keep_compiled():
        factor.compute(_START, _day(9))
        assert inner._lib is not None
    assert inner._lib is None


def test_pinned_factor_names_keep_only_those(panels) -> None:
    got = _wrapper(panels, factor_names=("sig_b",)).compute(_START, _END)
    assert list(got.data_vars) == ["sig_b"]


@pytest.mark.parametrize(
    "config, message",
    [
        ({"regressors": ()}, "regressors"),
        ({"regressors": ("industry", "sector")}, "regressors"),
        ({"regressors": ("size", "size")}, "regressors"),
        ({"warmup_bars": 5}, "warmup_bars"),
        ({"factor_names": ("sig_c",)}, "sig_c"),
    ],
)
def test_bad_configs_are_refused(panels, config, message) -> None:
    with pytest.raises(ValueError, match=message):
        _wrapper(panels, **config)


def test_resampling_is_refused(panels) -> None:
    with pytest.raises(ValueError, match="resample"):
        _wrapper(panels).resample("1w", "last")


def test_rebuilds_from_its_config(panels, tmp_path) -> None:
    prices, exposures = panels
    factor = NeutralizedFactor(NeutralizedConfig(
        factor=TwoSignals(FactorConfig(
            warmup_bars=0, dataset=FrameDataset(prices).to_zarr(tmp_path / "prices.zarr"),
            mode="batch", data_columns=("adjClose", "adjVolume"), njobs=2,
        )),
        dataset=FrameDataset(exposures).to_zarr(tmp_path / "exposures.zarr"),
        regressors=("size",), njobs=2,
    ))
    rebuilt = rebuild(json.loads(json.dumps(factor.get_config())))
    assert isinstance(rebuilt, NeutralizedFactor) and rebuilt == factor
    other = factor.copy()
    assert other == factor and other.config.factor is not factor.config.factor


def test_a_model_reads_it_like_any_factor(panels, tmp_path) -> None:
    factor = _wrapper(panels, tmp_path).build(_START, _END)
    label = StubLabel(_inner(panels[0]))
    model = XGBoostRegressor(ModelConfig(
        factors=[factor], labels=[label], model_save_dir=str(tmp_path / "model"),
        factor_data_strategy="read", label_data_strategy="cal",
        start_date=_START, end_date=_END, train_start=_START, train_end=_day(19),
        test_start=_day(20), test_end=_END,
        hyperparameters={"num_boost_round": 2, "nthread": 1},
    ))
    features = model._collect_all_features()
    expected = factor.read(_START, _END)
    for name in ("sig_a", "sig_b"):
        np.testing.assert_allclose(features[name].values, expected[name].values, equal_nan=True)


def test_config_field_defaults() -> None:
    fields = {f.name: f.default for f in dataclasses.fields(NeutralizedConfig)}
    assert fields["warmup_bars"] == 0 and fields["size_column"] == "marketcap"
    assert fields["industry_column"] == "industry" and fields["regressors"] == ("industry", "size")
