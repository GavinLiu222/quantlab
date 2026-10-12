"""BarraStyle extended one bar at a time equals the same bars computed whole.

What is locked here, and what turns it red:

- A store built to some bar, then extended, holds on the new bar exactly
  what computing every bar at once gives (every output), with
  ``warmup_bars`` set to ``BarraStyleParameters.warmup_bars`` as the
  recipes set it.

Synthetic panels of tests/test_barra_style.py; CPU-only and offline.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters
from tests.test_barra_style import _KWARGS, _T, _config

PARAMETERS = BarraStyleParameters(**_KWARGS)
BARS = pd.bdate_range("2021-01-04", periods=_T)


@pytest.mark.parametrize("built_to", [_T - 2, _T - 6])
def test_an_extended_bar_equals_the_whole_computation(tmp_path, built_to):
    factor = BarraStyle(_config(tmp_path, warmup_bars=PARAMETERS.warmup_bars))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*warm-up bar\(s\) are needed", category=UserWarning)
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
