"""BarraStyle extended one bar at a time equals the same bars computed whole.

What is locked here, and what turns it red:

- A store built to some bar, then extended, holds on the new bar exactly
  what computing every bar at once gives (every output), with
  ``warmup_bars`` set to ``BarraStyleParameters.warmup_bars`` as the
  recipes set it, on a risk-free rate that changes every bar (the factor
  lags it one bar, so the first excess return reads the rate a bar earlier);
- the same with the rate a single ``risk_free_symbol`` whose calendar holds
  dates without trading (FRED's DTB3 on 2025-01-09 and Good Fridays): the
  warm-up is counted in bars with a price, so the dates only the rate has do
  not shorten it (#252: counted on the merged calendar, RSTR fell short);
- with that rate symbol the factor calendar is the bars of its output, the
  price inputs' bars without the dates only the rate has: ``calendar``,
  ``bar_before`` and ``bar_after`` skip those dates, and a chunked
  computation is on the same bars as the whole one (on the merged calendar
  it allocated a row for each, never written), also when the prices and the
  rate are merged inside one input;
- a resampled BarraStyle is refused (its inputs are daily and no coarser
  bar than a day exists).

Synthetic panels of tests/test_barra_style.py; CPU-only and offline.
"""

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.chunking import ChunkedBuilder
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters
from tests.test_barra_style import _KWARGS, _T, _config, _inputs, _store

#: RSTR's window plus lag the longest, as with the USE4 defaults (504 + 21).
KWARGS = {**_KWARGS, "momentum_window": 50, "momentum_min_observations": 25}
PARAMETERS = BarraStyleParameters(**KWARGS)
BARS = pd.bdate_range("2021-01-04", periods=_T)


#: Dates the rate alone has, inside the extended bars' warm-up: the Saturdays
#: after two of the bars.
RATE_ONLY_DAYS = pd.DatetimeIndex([BARS[i] + pd.offsets.Week(weekday=5) for i in (60, 80)])


def _varying_rate(inputs):
    inputs["risk_free"] = inputs["risk_free"] * (1.0 + np.random.default_rng(3).uniform(-0.5, 0.5, inputs["risk_free"].shape[0]))[:, None]
    return inputs


def _rate_symbol_config(tmp_path):
    """The rate on its own ``DTB3`` symbol, on the trading days and ``RATE_ONLY_DAYS``."""
    inputs = _varying_rate(_inputs())
    days = BARS.append(RATE_ONLY_DAYS).sort_values()
    rate = pd.Series(inputs["risk_free"][:, 0], index=BARS).reindex(days)
    rate[RATE_ONLY_DAYS] = 0.5  # SYNTHETIC: a rate the next trading day replaces
    store = tmp_path / "fred.zarr"
    xr.Dataset(
        {"risk_free": (("timestamp", "symbol"), rate.to_numpy()[:, None])},
        coords={"timestamp": days, "symbol": np.asarray(["DTB3"], dtype=object)},
    ).to_zarr(store, mode="w")
    fred = StockDataset(DatasetConfig(
        zarr_file_path=str(store), raw_data_dir_path=str(tmp_path / "raw"),
        market="us_equity", frequency="1d",
    ))
    kwargs = {**KWARGS, "risk_free_symbol": "DTB3"}
    datasets = list(_config(tmp_path, inputs).dataset)
    price_columns = ("adjClose", "close", "volume", "divCash", "splitFactor")
    datasets[0] = _store(tmp_path, "prices_without_rate", {k: inputs[k] for k in price_columns})
    return _config(
        tmp_path, inputs, warmup_bars=PARAMETERS.warmup_bars, kwargs=kwargs, dataset=[*datasets, fred]
    )


@pytest.mark.parametrize("rate", ["column", "symbol"])
@pytest.mark.parametrize("built_to", [_T - 2, _T - 6])
def test_an_extended_bar_equals_the_whole_computation(tmp_path, built_to, rate):
    if rate == "symbol":
        factor = BarraStyle(_rate_symbol_config(tmp_path))
    else:
        inputs = _varying_rate(_inputs())
        factor = BarraStyle(_config(tmp_path, inputs, warmup_bars=PARAMETERS.warmup_bars, kwargs=dict(KWARGS)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*warm-up bar\(s\) (with a price )?are needed", category=UserWarning)
        whole = factor.compute(BARS[0], BARS[-1])
        factor.build(BARS[0], BARS[built_to])
        factor.extend(BARS[-1])
    stored = xr.open_zarr(factor.config.file_path).sel(timestamp=slice(BARS[built_to + 1], None))
    expected = whole.sel(timestamp=stored.timestamp)
    wrong = {
        name: int((~np.isclose(stored[name].values, expected[name].values, rtol=1e-9, atol=1e-12, equal_nan=True)).sum())
        for name in stored.data_vars
    }
    assert {name: n for name, n in wrong.items() if n} == {}


@pytest.mark.parametrize("nested", [False, True])
def test_with_a_rate_symbol_the_calendar_is_the_output_bars(tmp_path, nested):
    config = _rate_symbol_config(tmp_path)
    if nested:  # the prices and the rate merged inside one input
        prices, *others, fred = config.dataset
        config = dataclasses.replace(config, dataset=[MergedDataset([prices, fred]), *others])
    factor = BarraStyle(config)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*warm-up bar\(s\) (with a price )?are needed", category=UserWarning)
        whole = factor.compute(BARS[0], BARS[-1])
        chunked = ChunkedBuilder(factor, granularity="month").compute(BARS[0], BARS[-1])
    assert factor.calendar(BARS[0], BARS[-1]).equals(pd.DatetimeIndex(whole["timestamp"].values))
    saturday = RATE_ONLY_DAYS[0]
    assert factor.bar_after(saturday - pd.Timedelta(days=1), 1) == saturday + pd.Timedelta(days=2)
    assert factor.bar_before(saturday + pd.Timedelta(days=2), 1) == saturday - pd.Timedelta(days=1)
    assert pd.DatetimeIndex(chunked["timestamp"].values).equals(pd.DatetimeIndex(whole["timestamp"].values))
    for name in whole.data_vars:
        np.testing.assert_allclose(chunked[name].values, whole[name].values, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_a_resampled_barra_style_is_refused(tmp_path):
    config = _rate_symbol_config(tmp_path)
    with pytest.raises(ValueError, match="resample"):
        BarraStyle(config).resample("1d", "last")
    with pytest.raises(ValueError, match="resample"):
        BarraStyle(dataclasses.replace(config, resample_freq="1d", resample_how="last"))
