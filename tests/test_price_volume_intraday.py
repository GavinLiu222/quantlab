"""PriceVolumeIntradayStats and PriceVolumeIntradayAlpha against a numpy reference on a synthetic minute panel."""

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig, PriceVolumeIntradayConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.predefined.price_volume_intraday import (
    PRICE_VOLUME_INTRADAY_COLUMNS,
    PRICE_VOLUME_INTRADAY_OUTPUTS,
    PRICE_VOLUME_INTRADAY_STATS,
    PriceVolumeIntradayAlpha,
    PriceVolumeIntradayStats,
    session_amplitude_median,
)


class Identity(FactorKunQuant):
    """The close, as output ``AMP_MEAN``."""

    def _get_factor_names(self):
        return ("AMP_MEAN",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            Output(Input("close") * 1.0, "AMP_MEAN")
        return Function(builder.ops)


DAYS = ("2024-11-25", "2024-11-26", "2024-11-27", "2024-11-29")  # the 29th is a half day
SYMBOLS = 8


def _minute_panel() -> xr.Dataset:
    rng = np.random.default_rng(1)
    stamps = []
    for day in DAYS:
        bars = 210 if day == "2024-11-29" else 390
        open_at = pd.Timestamp(day) + pd.Timedelta("14h30min")
        stamps += list(open_at + pd.Timedelta("1min") * np.arange(1, bars + 1))
    stamps = pd.DatetimeIndex(stamps)
    shape = (len(stamps), SYMBOLS)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, shape), axis=0))
    close[rng.random(shape) < 0.1] = np.nan
    open_ = close * (1 + rng.normal(0, 2e-4, shape))
    spread = np.abs(rng.normal(0, 5e-4, shape))
    volume = np.where(np.isnan(close), np.nan, rng.lognormal(6, 0.8, shape).round())
    data = {
        "open": open_,
        "high": np.maximum(open_, close) * (1 + spread),
        "low": np.minimum(open_, close) * (1 - spread),
        "close": close,
        "volume": volume,
        "dollar_volume": volume * close * (1 + rng.normal(0, 1e-4, shape)),
    }
    return xr.Dataset(
        {name: (("timestamp", "symbol"), data[name]) for name in PRICE_VOLUME_INTRADAY_COLUMNS},
        coords={"timestamp": stamps, "symbol": [f"S{i}" for i in range(SYMBOLS)]},
    )


def _reference(panel: xr.Dataset, day: str, symbol: int, bars: int | None = None) -> dict:
    """Every statistic of one session and symbol, from plain numpy; with ``bars``, of its first ``bars`` bars."""
    session = pd.DatetimeIndex(panel["timestamp"].values).normalize() == pd.Timestamp(day)
    get = lambda name: panel[name].values[session, symbol][:bars]
    close = pd.Series(get("close"))
    first = close.first_valid_index()
    price = close.ffill().to_numpy()
    reference = np.r_[np.nan, price[:-1]]
    reference[first] = get("open")[first]
    r = np.log(price / reference)[first:]
    n = len(r)
    traded = np.isfinite(get("close"))
    amp = ((get("high") - get("low")) / reference)[traded]
    volume = np.nan_to_num(get("volume"))
    vwap = np.cumsum(np.nan_to_num(get("dollar_volume"))) / np.cumsum(volume)
    dev = (get("close") / vwap - 1)[traded]
    threshold = volume.mean() + volume.std(ddof=1)
    up = volume > threshold
    up_r = np.where(up, np.r_[np.full(first, np.nan), r], np.nan)
    up_r = up_r[np.isfinite(up_r)]
    w = volume[first:]
    weighted_mean = (w * r).sum() / w.sum()
    return {
        "AMP_MEAN": amp.mean(),
        "AMP_MMR": amp.mean() / np.median(amp),
        "AMP_SKEW": pd.Series(amp).skew(),
        "AMP_STD": amp.std(ddof=1),
        "AMP_STR": amp.mean() / amp.std(ddof=1),
        "PRICE_DEV_STD": dev.std(ddof=1),
        "PRICE_DEV_STR": dev.mean() / dev.std(ddof=1),
        "RET_STD": r.std(ddof=1),
        "UP_RET_STD": math.sqrt((r[r > 0] ** 2).sum() / (n - 1)),
        "DOWN_RET_STD": math.sqrt((r[r < 0] ** 2).sum() / (n - 1)),
        "VOL_WEIGHTED_RET_STD": math.sqrt((w * r * r).sum() / w.sum() - weighted_mean**2),
        "VOL_UP_RET_STD": up_r.std(ddof=1),
        "VOL_UP_RET_POS_MEAN": up_r[up_r > 0].mean(),
        "VOL_UP_RET_NEG_MEAN": up_r[up_r < 0].mean(),
        "VOL_UP_NUM": float((up & ~np.r_[False, up[:-1]]).sum()),
    }


@pytest.fixture(scope="module")
def panel() -> xr.Dataset:
    return _minute_panel()


def _stats(panel, **kwargs) -> PriceVolumeIntradayStats:
    return PriceVolumeIntradayStats(FactorConfig(
        warmup_bars=390, dataset=FrameDataset(panel), mode="batch", data_columns=PRICE_VOLUME_INTRADAY_COLUMNS, njobs=2,
        **kwargs,
    ))


@pytest.fixture(scope="module")
def daily(panel) -> xr.Dataset:
    return _stats(panel).resample("1d", "last").compute("2024-11-26", "2024-11-29")


def test_one_value_per_session(daily):
    assert list(daily.data_vars) == list(PRICE_VOLUME_INTRADAY_STATS)
    assert pd.DatetimeIndex(daily["timestamp"].values).strftime("%Y-%m-%d").tolist() == list(DAYS[1:])


@pytest.mark.parametrize("day", DAYS[1:])
@pytest.mark.parametrize("symbol", [0, 5])
def test_outputs_match_numpy(panel, daily, day, symbol):
    expected = _reference(panel, day, symbol)
    got = daily.sel(timestamp=day).isel(symbol=symbol)
    for name in PRICE_VOLUME_INTRADAY_STATS:
        assert float(got[name]) == pytest.approx(expected[name], rel=2e-3, abs=1e-6), name


@pytest.fixture(scope="module")
def minutes(panel) -> xr.Dataset:
    stamps = pd.DatetimeIndex(panel["timestamp"].values)
    return _stats(panel).compute(stamps[stamps.normalize() == pd.Timestamp(DAYS[1])][0], stamps[-1])


@pytest.mark.parametrize("day, bars", [(DAYS[1], 120), (DAYS[2], 375), (DAYS[3], 150)])
@pytest.mark.parametrize("symbol", [0, 5])
def test_every_bar_holds_the_session_so_far(panel, minutes, day, bars, symbol):
    expected = _reference(panel, day, symbol, bars=bars)
    stamps = pd.DatetimeIndex(minutes["timestamp"].values)
    got = minutes.isel(timestamp=np.flatnonzero(stamps.normalize() == pd.Timestamp(day))[bars - 1], symbol=symbol)
    for name in PRICE_VOLUME_INTRADAY_STATS:
        if name == "AMP_MMR":  # the median is on the session's last bar only
            assert np.isnan(float(got[name]))
        else:
            assert float(got[name]) == pytest.approx(expected[name], rel=2e-3, abs=1e-6), name


def test_median_is_on_the_last_bar():
    session = np.array([0, 0, 0, 1, 1])
    high = np.array([[11.0, np.nan], [12.0, np.nan], [np.nan, np.nan], [10.5, 10.0], [10.4, 10.0]])
    out = session_amplitude_median(session, high, high - np.array([1.0, 0.5]), np.full_like(high, 10.0))
    np.testing.assert_allclose(out[:, 0], [np.nan, np.nan, 0.1, np.nan, 0.1], rtol=1e-6)
    np.testing.assert_allclose(out[:, 1], [np.nan, np.nan, np.nan, np.nan, 0.05], rtol=1e-6)


def test_short_session_is_nan(panel):
    sparse = panel.copy(deep=True)
    for name in ("close", "high", "low", "volume", "dollar_volume"):
        sparse[name][:, 3] = np.where(np.arange(sparse.sizes["timestamp"]) % 10 == 0, sparse[name][:, 3], np.nan)
    out = _stats(sparse, kwargs={"min_returns": 200}).resample("1d", "last").compute("2024-11-26", "2024-11-27")
    assert np.isnan(out["RET_STD"].values[:, 3]).all()
    assert np.isfinite(out["RET_STD"].values[:, 0]).all()


def test_refuses_stream_mode_other_columns_and_short_warmup(panel):
    with pytest.raises(ValueError, match="stream mode"):
        PriceVolumeIntradayStats(FactorConfig(warmup_bars=390, dataset=FrameDataset(panel), mode="stream",
                                        data_columns=PRICE_VOLUME_INTRADAY_COLUMNS))
    with pytest.raises(ValueError, match="data_columns must be"):
        PriceVolumeIntradayStats(FactorConfig(warmup_bars=390, dataset=FrameDataset(panel), mode="batch",
                                        data_columns=("close",)))
    with pytest.raises(ValueError, match="cover one session"):
        PriceVolumeIntradayStats(FactorConfig(warmup_bars=100, dataset=FrameDataset(panel), mode="batch",
                                        data_columns=PRICE_VOLUME_INTRADAY_COLUMNS))


def test_alpha_averages_the_wrapped_store(tmp_path):
    values = np.arange(12, dtype=float).reshape(6, 2)
    values[2, 0] = np.nan
    panel = xr.Dataset(
        {"close": (("timestamp", "symbol"), values)},
        coords={"timestamp": pd.date_range("2024-01-01", periods=6), "symbol": ["A", "B"]},
    )
    source = Identity(FactorConfig(
        warmup_bars=0, dataset=FrameDataset(panel), mode="batch", data_columns=("close",),
        file_path=str(tmp_path / "source.zarr"), njobs=2,
    ))
    source.build("2024-01-01", "2024-01-06")
    alpha = PriceVolumeIntradayAlpha(PriceVolumeIntradayConfig(factor=source, windows=(1, 3, 4), njobs=2))

    assert alpha.get_factor_names() == ("AMP_MEAN_1D", "AMP_MEAN_3D", "AMP_MEAN_4D")
    out = alpha.compute("2024-01-04", "2024-01-06")
    np.testing.assert_allclose(out["AMP_MEAN_1D"].values, values[3:])
    np.testing.assert_allclose(out["AMP_MEAN_3D"].values[:, 0], [(2 + 6) / 2, (6 + 8) / 2, (6 + 8 + 10) / 3])
    np.testing.assert_allclose(out["AMP_MEAN_4D"].values[:, 1], [4, 6, 8])
    early = alpha.compute("2024-01-01", "2024-01-02")
    assert np.isnan(early["AMP_MEAN_4D"].values).all()
    np.testing.assert_allclose(early["AMP_MEAN_1D"].values, values[:2])


def test_alpha_names_are_the_list(panel):
    alpha = PriceVolumeIntradayAlpha(PriceVolumeIntradayConfig(factor=_stats(panel).resample("1d", "last")))
    assert alpha.get_factor_names() == PRICE_VOLUME_INTRADAY_OUTPUTS
    assert len(PRICE_VOLUME_INTRADAY_OUTPUTS) == 90
    assert PRICE_VOLUME_INTRADAY_OUTPUTS[:2] == ("AMP_MEAN_1D", "AMP_MEAN_5D")
    assert PRICE_VOLUME_INTRADAY_OUTPUTS[-1] == "VOL_UP_NUM_60D"


def test_alpha_refuses_bad_windows(panel):
    daily = _stats(panel).resample("1d", "last")
    for windows in ((), (5, 5), (0,)):
        with pytest.raises(ValueError, match="windows must be"):
            PriceVolumeIntradayAlpha(PriceVolumeIntradayConfig(factor=daily, windows=windows))


def test_alpha_ranked_is_the_percentile_rank(tmp_path):
    rng = np.random.default_rng(5)
    values = rng.normal(size=(8, 6))
    values[3, 2] = np.nan
    panel = xr.Dataset(
        {"close": (("timestamp", "symbol"), values)},
        coords={"timestamp": pd.date_range("2024-01-01", periods=8), "symbol": [f"S{i}" for i in range(6)]},
    )
    source = Identity(FactorConfig(
        warmup_bars=0, dataset=FrameDataset(panel), mode="batch", data_columns=("close",),
        file_path=str(tmp_path / "source.zarr"), njobs=2,
    ))
    source.build("2024-01-01", "2024-01-08")
    raw = PriceVolumeIntradayAlpha(PriceVolumeIntradayConfig(factor=source, windows=(1, 3), njobs=2))
    ranked = PriceVolumeIntradayAlpha(PriceVolumeIntradayConfig(factor=source, windows=(1, 3), ranked=True, njobs=2))
    plain, ranks = raw.compute("2024-01-03", "2024-01-08"), ranked.compute("2024-01-03", "2024-01-08")
    for name in ("AMP_MEAN_1D", "AMP_MEAN_3D"):
        expected = plain[name].to_pandas().rank(axis=1, pct=True)
        np.testing.assert_allclose(ranks[name].to_pandas().values, expected.values, rtol=1e-6, err_msg=name)
