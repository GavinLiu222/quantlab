"""``ShortHorizonAlpha``: every output against a pandas reference of its formula.

The panel is 16 symbols over 60 business days of random adjusted prices,
volumes and market caps, with a hole and a symbol that stops trading.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.short_horizon import SHORT_HORIZON_COLUMNS, ShortHorizonAlpha

DAYS = pd.bdate_range("2024-01-01", periods=60)
START, END = DAYS[0].strftime("%Y-%m-%d"), DAYS[-1].strftime("%Y-%m-%d")
SYMBOLS = [str(s) for s in range(16)]


def _panel() -> xr.Dataset:
    rng = np.random.default_rng(3)
    shape = (len(DAYS), len(SYMBOLS))
    close = 30 * np.exp(rng.normal(0, 0.02, size=shape).cumsum(axis=0))
    open_ = close * np.exp(rng.normal(0, 0.01, size=shape))
    volume = rng.lognormal(12, 0.5, size=shape)
    cap = close * rng.uniform(1e6, 1e8, size=(1, len(SYMBOLS)))
    for x in (open_, close, volume, cap):
        x[20:24, 5] = np.nan
        x[45:, 9] = np.nan
    return xr.Dataset(
        {name: (("timestamp", "symbol"), x) for name, x in zip(SHORT_HORIZON_COLUMNS, (open_, close, volume, cap))},
        coords={"timestamp": DAYS.values, "symbol": SYMBOLS},
    )


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def outputs(panel, tmp_path_factory):
    prices = FrameDataset(panel).to_zarr(tmp_path_factory.mktemp("short") / "prices.zarr")
    factor = ShortHorizonAlpha(FactorConfig(
        warmup_bars=0, dataset=prices, mode="batch", data_columns=SHORT_HORIZON_COLUMNS, njobs=2,
    ))
    return factor.compute(START, END)


def _reference(panel) -> dict[str, pd.DataFrame]:
    o, c, v, cap = (panel[n].to_pandas() for n in SHORT_HORIZON_COLUMNS)
    one = c / c.shift(1) - 1
    out = {f"reversal_{k}": -(c / c.shift(k) - 1) for k in (1, 3, 5)}
    out["low_max_20"] = -one.rolling(20).max()
    for k in (5, 20):
        turnover = (v * c).rolling(k).mean() / cap
        out[f"turnover_momentum_{k}"] = (c / c.shift(k) - 1) * (turnover.rank(axis=1, pct=True) - 0.5)
    out["abnormal_volume"] = np.log(v / v.rolling(20).mean())
    for n in (5, 20):
        out[f"overnight_{n}"] = (o / c.shift(1) - 1).rolling(n).mean()
        out[f"intraday_{n}"] = (c / o - 1).rolling(n).mean()
    return out


def test_every_output_matches_its_formula(panel, outputs):
    for name, expected in _reference(panel).items():
        got = outputs[name].to_pandas().reindex(index=expected.index, columns=expected.columns).values
        np.testing.assert_allclose(got, expected.values, rtol=2e-3, atol=2e-5, err_msg=name)


def test_names_are_the_outputs(outputs):
    assert list(outputs.data_vars) == [
        "reversal_1", "reversal_3", "reversal_5", "low_max_20", "turnover_momentum_5", "turnover_momentum_20",
        "abnormal_volume", "overnight_5", "overnight_20", "intraday_5", "intraday_20",
    ]


def test_other_data_columns_are_refused(panel, tmp_path):
    prices = FrameDataset(panel).to_zarr(tmp_path / "prices.zarr")
    with pytest.raises(ValueError, match="data_columns must be"):
        ShortHorizonAlpha(FactorConfig(warmup_bars=0, dataset=prices, mode="batch",
                                       data_columns=("adjClose",), njobs=2))
