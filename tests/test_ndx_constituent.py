"""Tests for the Nasdaq-100 daily membership panel (DATA-05, DATA-06).

The Nasdaq-100 counterpart of `tests/test_constituent_panel.py`: that module
owns the shared densification rules (interval closedness, right-edge horizon,
calendar-day axis, delisted columns) via no-I/O fixture subclasses, so nothing
here re-tests them. This module tests only what is specific to the second
index -- its own left edge, its own store, and the structural claim that
adding it cost nothing above `dataset/`.

Every test runs OFFLINE. The only route to the network in this code path is
`quantlab.universe.requests.get`, which the shared `mock_universe_fetchers`
fixture in `tests/conftest.py` replaces with a URL-keyed fake that raises
`AssertionError` on any unexpected URL.
"""

import numpy as np
import pandas as pd
import xarray as xr

from conftest import WHOLE_STORE
from quantlab.dataset.config import FileConstituentDatasetConfig
from quantlab.dataset.constituent import Nasdaq100ConstituentDataset, SP500ConstituentDataset

_NDX_COVERAGE_START = "2007-02-01"
_SP500_COVERAGE_START = "1976-07-01"


def _make_config(tmp_path, **overrides) -> FileConstituentDatasetConfig:
    """Module-local config constructor (03.1-PATTERNS.md §6: shared mocks live
    in conftest, per-module config constructors stay module-local)."""
    params = dict(
        zarr_file_path=str(
            tmp_path / "us_equity" / "nasdaq100_constituent.zarr"
        ),
        cache_dir=str(tmp_path / "reference" / "_cache"),
    )
    params.update(overrides)
    return FileConstituentDatasetConfig(**params)  # type: ignore[arg-type]


def test_nasdaq100_panel_round_trips_through_zarr(mock_universe_fetchers, tmp_path):
    """DATA-05 / D-04: the Nasdaq-100 membership panel survives a full
    build -> Zarr write -> fresh-instance read cycle as a boolean
    `(timestamp, symbol)` grid, exactly like the S&P 500 panel does.
    """
    cfg = _make_config(tmp_path)
    Nasdaq100ConstituentDataset(cfg).from_raw_data().save()

    reloaded = (
        Nasdaq100ConstituentDataset(_make_config(tmp_path)).panel(*WHOLE_STORE)
    )

    assert isinstance(reloaded, xr.Dataset)
    assert tuple(reloaded["is_member"].dims) == ("timestamp", "symbol")
    assert set(reloaded.data_vars) == {"is_member"}
    assert reloaded["is_member"].dtype == np.dtype(bool)

    symbols = set(reloaded["symbol"].values.tolist())
    # `LOGI` is the change log's earliest added ticker (2007-02-01), removed
    # in 2018. `GONE1` is never added anywhere in the fixture -- a
    # left-censored, drop-only row -- and must still be a real column rather
    # than an absent symbol, which is the survivorship-bias guarantee.
    assert "LOGI" in symbols
    assert "GONE1" in symbols


def test_nasdaq100_panel_left_edge_is_2007_02_01(mock_universe_fetchers, tmp_path):
    """RESEARCH Finding 2: `2007-02-01` (`LOGI` added / `CMVT` removed) is the
    verified earliest row the Wikipedia Nasdaq-100 change log actually
    contains, so it is the earliest date this index's membership can be
    answered for at all.

    A panel starting earlier -- and the inherited `Date.START_DATE` default is
    1900-01-01, so that is what happens without the clamp -- would be all-False
    across a region where the true answer is UNKNOWN, and in a boolean panel
    those two are indistinguishable at read time.
    """
    dataset = Nasdaq100ConstituentDataset(
        _make_config(tmp_path, start_date=None)
    )
    panel = dataset.from_raw_data().get_xarray_dataset()

    assert dataset.config.start_date == _NDX_COVERAGE_START
    assert pd.Timestamp(
        panel["timestamp"].values[0]
    ) == pd.Timestamp(_NDX_COVERAGE_START)


def test_the_two_index_panels_keep_separate_axes_and_left_edges(
    mock_universe_fetchers, tmp_path
):
    """RESEARCH Finding 6 bullet 4: two categories with different left edges
    must NOT be silently unioned onto one axis that implies coverage neither
    has.

    The two panels are built here in the same test precisely so the separation
    is asserted rather than assumed: distinct `zarr_file_path` values, distinct
    first timestamps (1976-07-01 versus 2007-02-01, ~31 years apart), and
    neither symbol axis a superset of the other. A consumer that wants both
    opens both and joins on the intersection of their timestamp axes -- a
    deliberate, visible step.
    """
    ndx_cfg = _make_config(tmp_path, start_date=None)
    sp500_cfg = _make_config(
        tmp_path,
        zarr_file_path=str(tmp_path / "us_equity" / "sp500_constituent.zarr"),
        start_date=None,
    )
    assert ndx_cfg.zarr_file_path != sp500_cfg.zarr_file_path

    ndx = Nasdaq100ConstituentDataset(ndx_cfg).from_raw_data().get_xarray_dataset()
    sp500 = SP500ConstituentDataset(sp500_cfg).from_raw_data().get_xarray_dataset()

    ndx_left = pd.Timestamp(ndx["timestamp"].values[0])
    sp500_left = pd.Timestamp(sp500["timestamp"].values[0])
    assert ndx_left != sp500_left
    assert ndx_left == pd.Timestamp(_NDX_COVERAGE_START)
    assert sp500_left == pd.Timestamp(_SP500_COVERAGE_START)

    ndx_symbols = set(ndx["symbol"].values.tolist())
    sp500_symbols = set(sp500["symbol"].values.tolist())
    assert not ndx_symbols >= sp500_symbols
    assert not sp500_symbols >= ndx_symbols


def test_adding_an_index_overrides_only_the_two_abstract_hooks():
    """DATA-06's acceptance -- 新增一类指数不需要改动上层代码 -- in constructive
    form.

    The entire cost of a second index is two methods in `dataset/` plus one
    config factory. `Nasdaq100ConstituentDataset` overrides NONE of the shared
    machinery: not the densification, not the cleaning route, not the symbol
    seam, not the coverage clamp, not the config property. If a future index
    needs to override one of these, the abstraction drawn in 03.1-03 was wrong
    and belongs on the base -- a per-index special case in `dataset/` would
    hide that.

    This is the structural analogue of 03-02's zero-diff-on-`base/model.py`
    proof of factor-backend interchangeability.
    """
    shared_machinery = {
        "_raw_data_to_xr",
        "_densify",
        "_clean",
        "_reset_symbols",
        "config",
        "_clamp_coverage_start",
    }
    own_members = set(vars(Nasdaq100ConstituentDataset))

    assert own_members & shared_machinery == set()
    assert {"_pit_coverage_start", "_build_intervals"} <= own_members
