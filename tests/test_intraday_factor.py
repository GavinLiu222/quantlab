"""IntradayStats and IntradayAverage against a numpy reference on a synthetic minute panel."""

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig, IntradayAverageConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.predefined.intraday import (
    INTRADAY_COLUMNS,
    INTRADAY_OUTPUTS,
    LAST_PRICE,
    PRIOR_CLOSE,
    REFERENCE_PRICE,
    SESSION_BARS_LEFT,
    SESSION_POSITION,
    IntradayAverage,
    IntradayStats,
    session_inputs,
)

class Identity(FactorKunQuant):
    """The close, as output ``value``."""

    def _get_factor_names(self):
        return ("value",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            Output(Input("close") * 1.0, "value")
        return Function(builder.ops)


DAYS = ("2024-11-25", "2024-11-26", "2024-11-27", "2024-11-29")  # the 29th is a half day
SYMBOLS = 8


def _minute_panel() -> xr.Dataset:
    rng = np.random.default_rng(0)
    stamps = []
    for day in DAYS:
        bars = 210 if day == "2024-11-29" else 390
        open_at = pd.Timestamp(day) + pd.Timedelta("14h30min")
        stamps += list(open_at + pd.Timedelta("1min") * np.arange(1, bars + 1))
    stamps = pd.DatetimeIndex(stamps)
    shape = (len(stamps), SYMBOLS)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, shape), axis=0))
    close[rng.random(shape) < 0.1] = np.nan
    volume = np.where(np.isnan(close), 0.0, rng.integers(100, 1000, shape).astype(float))
    buy = np.floor(volume * rng.random(shape))
    session = stamps.normalize()
    last = np.r_[session[1:] != session[:-1], True]
    auction_price = np.full(shape, np.nan)
    auction_price[last] = close[last] * 1.001
    auction_volume = np.zeros(shape)
    auction_volume[last] = 5000.0
    data = {
        "open": close * (1 + rng.normal(0, 2e-4, shape)),
        "close": close,
        "volume": volume,
        "buy_volume": buy,
        "sell_volume": volume - buy,
        "dollar_volume": volume * np.nan_to_num(close),
        "n_trades": np.ceil(volume / 100),
        "offexchange_volume": np.floor(volume * rng.random(shape)),
        "close_auction_price": auction_price,
        "close_auction_volume": auction_volume,
    }
    return xr.Dataset(
        {name: (("timestamp", "symbol"), values) for name, values in data.items()},
        coords={"timestamp": stamps, "symbol": [f"S{i}" for i in range(SYMBOLS)]},
    )


def _reference(panel: xr.Dataset, day: str, symbol: int, previous_close: float) -> dict:
    """Every output of one session and symbol, from plain numpy."""
    session = pd.DatetimeIndex(panel["timestamp"].values).normalize() == pd.Timestamp(day)
    get = lambda name: panel[name].values[session, symbol]  # noqa: E731
    close = pd.Series(get("close"))
    first = close.first_valid_index()
    price = close.ffill().to_numpy()[first:]
    reference = np.r_[get("open")[first], price[:-1]]
    r = np.log(price / reference)
    volume = get("volume")
    n = len(r)
    rv = (r**2).sum()
    left = np.arange(len(volume))[::-1][first:]
    position = np.arange(len(volume))[first:]
    continuous = volume.sum()
    full_price = close.ffill().to_numpy()
    both = np.isfinite(full_price)
    return {
        "rvol": math.sqrt(rv),
        "rskew": math.sqrt(n) * (r**3).sum() / rv**1.5,
        "rkurt": n * (r**4).sum() / rv**2,
        "rsj": ((r[r > 0] ** 2).sum() - (r[r < 0] ** 2).sum()) / rv,
        "jump": max(rv - math.pi / 2 * n / (n - 1) * (np.abs(r[1:]) * np.abs(r[:-1])).sum(), 0) / rv,
        "ret_autocorr": (r[1:] * r[:-1]).sum() / rv,
        "ret_intraday": r.sum(),
        "ret_overnight": math.log(get("open")[first] / previous_close),
        "ret_open30": r[position < 30].sum(),
        "ret_close30": r[left < 30].sum(),
        "vwap_dev": math.log(price[-1] / (get("dollar_volume").sum() / continuous)),
        "oib": (get("buy_volume").sum() - get("sell_volume").sum()) / (get("buy_volume").sum() + get("sell_volume").sum()),
        "corr_rv": np.corrcoef(r, volume[first:])[0, 1],
        "corr_pv": np.corrcoef(full_price[both], volume[both])[0, 1],
        "volume_cv": volume.std() / volume.mean(),
        "volume_close30": volume[-30:].sum() / continuous,
        "auction_share": get("close_auction_volume").sum() / (continuous + get("close_auction_volume").sum()),
        "offexchange_share": get("offexchange_volume").sum() / continuous,
        "amihud": 1e6 * np.abs(r).sum() / get("dollar_volume").sum(),
        "trade_size": math.log(continuous / get("n_trades").sum()),
    }


@pytest.fixture(scope="module")
def panel() -> xr.Dataset:
    return _minute_panel()


@pytest.fixture(scope="module")
def daily(panel) -> xr.Dataset:
    stats = IntradayStats(FactorConfig(
        warmup_bars=390, dataset=FrameDataset(panel), mode="batch",
        data_columns=INTRADAY_COLUMNS, njobs=2,
    ))
    return stats.resample("1d", "last").compute("2024-11-26", "2024-11-29")


def test_session_inputs_restart_each_session():
    session = np.array([0, 0, 0, 1, 1])
    close = np.array([[10.0], [np.nan], [11.0], [12.0], [np.nan]])
    auction = np.array([[np.nan], [np.nan], [11.5], [np.nan], [np.nan]])
    out = session_inputs(session, close - 1, close, auction)
    assert out[SESSION_POSITION][:, 0].tolist() == [0, 1, 2, 0, 1]
    assert out[SESSION_BARS_LEFT][:, 0].tolist() == [2, 1, 0, 1, 0]
    assert out[LAST_PRICE][:, 0].tolist() == [10, 10, 11, 12, 12]
    assert out[REFERENCE_PRICE][:, 0].tolist() == [9, 10, 10, 11, 12]
    np.testing.assert_array_equal(out[PRIOR_CLOSE][:, 0], [np.nan, np.nan, np.nan, 11.5, 11.5])


def test_one_value_per_session(daily):
    assert list(daily.data_vars) == list(INTRADAY_OUTPUTS)
    assert pd.DatetimeIndex(daily["timestamp"].values).strftime("%Y-%m-%d").tolist() == list(DAYS[1:])


@pytest.mark.parametrize("day", DAYS[1:])
@pytest.mark.parametrize("symbol", [0, 5])
def test_outputs_match_numpy(panel, daily, day, symbol):
    previous = DAYS[DAYS.index(day) - 1]
    before = pd.DatetimeIndex(panel["timestamp"].values).normalize() == pd.Timestamp(previous)
    previous_close = panel["close_auction_price"].values[before, symbol][-1]
    if np.isnan(previous_close):
        previous_close = pd.Series(panel["close"].values[before, symbol]).ffill().iloc[-1]
    expected = _reference(panel, day, symbol, previous_close)
    got = daily.sel(timestamp=day).isel(symbol=symbol)
    for name in INTRADAY_OUTPUTS:
        assert float(got[name]) == pytest.approx(expected[name], rel=2e-4, abs=1e-4), name


def test_short_session_is_nan(panel):
    sparse = panel.copy(deep=True)
    sparse["close"][:, 3] = np.where(np.arange(sparse.sizes["timestamp"]) % 10 == 0, sparse["close"][:, 3], np.nan)
    stats = IntradayStats(FactorConfig(
        warmup_bars=390, dataset=FrameDataset(sparse), mode="batch",
        data_columns=INTRADAY_COLUMNS, njobs=2, kwargs={"min_returns": 200},
    ))
    out = stats.resample("1d", "last").compute("2024-11-26", "2024-11-27")
    assert np.isnan(out["rvol"].values[:, 3]).all()
    assert np.isfinite(out["rvol"].values[:, 0]).all()


def test_refuses_stream_mode_and_short_warmup(panel):
    with pytest.raises(ValueError, match="stream mode"):
        IntradayStats(FactorConfig(
            warmup_bars=390, dataset=FrameDataset(panel), mode="stream", data_columns=INTRADAY_COLUMNS,
        ))
    with pytest.raises(ValueError, match="cover one session"):
        IntradayStats(FactorConfig(
            warmup_bars=100, dataset=FrameDataset(panel), mode="batch", data_columns=INTRADAY_COLUMNS,
        ))


def test_average_reads_the_wrapped_store(tmp_path):
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
    average = IntradayAverage(IntradayAverageConfig(factor=source, window=3, min_count=2, njobs=2))

    assert average.get_factor_names() == ("value_3d",)
    out = average.compute("2024-01-04", "2024-01-06")["value_3d"].values
    np.testing.assert_allclose(out[:, 0], [(2 + 6) / 2, (6 + 8) / 2, (6 + 8 + 10) / 3])
    np.testing.assert_allclose(out[:, 1], [5, 7, 9])
    assert np.isnan(average.compute("2024-01-01", "2024-01-02")["value_3d"].values).all()


def test_average_needs_a_store(panel):
    source = Identity(FactorConfig(
        warmup_bars=0, dataset=FrameDataset(panel), mode="batch", data_columns=("close",), njobs=2,
    ))
    with pytest.raises(ValueError, match="has no store"):
        IntradayAverage(IntradayAverageConfig(factor=source)).compute("2024-11-26", "2024-11-27")
