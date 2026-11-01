"""`get_xarray_dataset(indexes)` must actually honour `indexes` (defect E).

Quick task 260907-fl6, batch 2. `XrBackend.get_xarray_dataset`'s entire body
used to be `return self.data`: the parameter was accepted and ignored, so a
caller could pass `["timestamp"]`, `["timestamp", "symbol"]` or complete
nonsense and get the same whole panel back. `PlBackend` on the same interface
DID use it (`set_index(indexes)` then `Dataset.from_dataframe`), so the two
implementations of one ABC method meant different things.

The knock-on was `RawDataset.time_interval`, which is written as "give me
just the time axis, then diff it" and therefore did not work at all under
`XrBackend` -- two errors in a row, both measured on
`data/data/us_equity/1d/us_all.zarr`:

    TypeError: numpy boolean subtract, the `-` operator, is not supported ...
    AttributeError: 'Dataset' object has no attribute 'to_series'

**The semantics chosen** (and what these tests pin): `indexes` names the
dimensions the returned dataset is indexed by, in order -- the same meaning
`PlBackend` already gave it. Data variables laid out on any dimension outside
`indexes` are dropped, dimensions left unused are dropped with their
coordinates, and what survives is transposed onto `indexes`. `indexes=None`
means "no shape request, hand it over as-is".

That choice is deliberately backward compatible with every existing call site.
The repository only ever passes `["timestamp", "symbol"]` or nothing at all,
and on a canonical panel -- where every variable is laid out on exactly those
two dims -- steps 2 and 3 drop nothing, leaving only the transpose. The tests
below assert that non-regression directly rather than trusting the argument.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from quantlab.dataset.config import RawDatasetConfig
from quantlab.dataset.raw import RawDataset
from quantlab.backend.parquet import PlBackend
from quantlab.backend.zarr import XrBackend

TIMES = pd.date_range("2024-01-01", periods=4, freq="D")
SYMBOLS = ["AAA", "BBB"]

#: Daily, then a two-day hole (a weekend). `time_interval` must report 1 day.
GAPPY_TIMES = pd.to_datetime(
    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05", "2024-01-06"]
)


class PanelDataset(RawDataset):
    """The smallest concrete `RawDataset` that can answer `time_interval`.

    Modelled on `tests/test_dataset_hierarchy.py:PanelDataset`: one variable,
    `_clean` overridden to the identity (the inherited default runs the
    OHLCV-shaped `clean_market_data()` pipeline, which a non-market panel
    fails schema validation on), `_reset_symbols` overridden so construction
    does not touch disk.

    Unlike that one it carries a BOOL variable, because the bool is what made
    `time_interval` raise: `.diff(dim="timestamp")` on a whole panel hits
    `anomaly_flag` and numpy refuses to subtract booleans.
    """

    def __init__(self, config: RawDatasetConfig, gappy: bool = False):
        self._times = GAPPY_TIMES if gappy else TIMES
        super().__init__(config)

    def _raw_data_to_xr(self) -> xr.Dataset:
        n = len(self._times)
        return xr.Dataset(
            {
                "close": (
                    ["timestamp", "symbol"],
                    np.arange(float(n * 2)).reshape(n, 2),
                ),
                "anomaly_flag": (
                    ["timestamp", "symbol"],
                    np.zeros((n, 2), dtype=bool),
                ),
            },
            coords={"timestamp": self._times, "symbol": list(SYMBOLS)},
        )

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        return data

    def _reset_symbols(self) -> None:
        return None


def _panel(transposed: bool = False) -> xr.Dataset:
    """A canonical two-variable panel, one float and one BOOL.

    The bool `anomaly_flag` is not decoration: `quantlab/dataset/_support/cleaning.py` adds
    exactly this variable to every cleaned market panel, and it is what made
    `.diff(dim="timestamp")` raise on the whole-dataset return.
    """
    close = np.arange(8.0).reshape(4, 2)
    flag = np.array(
        [[False, False], [False, True], [False, False], [True, False]]
    )
    dims = ["symbol", "timestamp"] if transposed else ["timestamp", "symbol"]
    if transposed:
        close, flag = close.T, flag.T
    return xr.Dataset(
        {"close": (dims, close), "anomaly_flag": (dims, flag)},
        coords={"timestamp": TIMES, "symbol": SYMBOLS},
    )


# --------------------------------------------------------------------------
# The semantics
# --------------------------------------------------------------------------


def test_indexes_none_returns_the_backend_object_itself():
    """`None` is the "no shape request" case, and it must stay identity.

    Dozens of call sites do `backend.get_xarray_dataset()` and then read
    `.symbol`, `.sizes` or `.data_vars` off the result; several rely on it
    being the backend's own object rather than a copy.
    """
    panel = _panel()
    backend = XrBackend().to_internal(panel)
    assert backend.get_xarray_dataset() is panel


def test_two_dim_request_pins_the_axis_order_and_keeps_every_variable():
    """The call the whole repository makes: `["timestamp", "symbol"]`.

    The stored panel here is deliberately laid out `(symbol, timestamp)` --
    the reverse of the pipeline's convention -- so the transpose is
    observable. Nothing may be dropped: both variables live on exactly the
    requested dims.
    """
    backend = XrBackend().to_internal(_panel(transposed=True))
    result = backend.get_xarray_dataset(["timestamp", "symbol"])

    assert set(result.data_vars) == {"close", "anomaly_flag"}
    assert result["close"].dims == ("timestamp", "symbol")
    assert result["anomaly_flag"].dims == ("timestamp", "symbol")
    assert result["anomaly_flag"].dtype == np.dtype("bool")
    np.testing.assert_array_equal(
        result["close"].values, np.arange(8.0).reshape(4, 2)
    )


def test_one_dim_request_drops_the_other_axis_and_its_variables():
    """`["timestamp"]` really does narrow to the time axis.

    Before the fix this returned the whole panel, which is why
    `time_interval`'s `.diff(dim="timestamp")` hit the bool variable. The
    `timestamp` COORDINATE must survive -- dropping the variables must not
    take the index with them (`ds[[]]` does exactly that, which is why the
    implementation uses `drop_vars` + `drop_dims` instead).
    """
    backend = XrBackend().to_internal(_panel())
    result = backend.get_xarray_dataset(["timestamp"])

    assert dict(result.sizes) == {"timestamp": 4}
    assert list(result.data_vars) == []
    assert "symbol" not in result.coords
    np.testing.assert_array_equal(
        result["timestamp"].values, TIMES.values
    )


def test_variables_on_an_unrequested_dimension_are_dropped():
    """A third axis is what makes "drop what does not fit" observable.

    A canonical panel has nothing to drop for `["timestamp", "symbol"]`,
    which is exactly why the non-regression above passes. This test proves
    the dropping rule is real rather than vacuous.
    """
    panel = _panel().assign(
        depth=(
            ["timestamp", "symbol", "level"],
            np.zeros((4, 2, 3)),
        )
    )
    backend = XrBackend().to_internal(panel)
    result = backend.get_xarray_dataset(["timestamp", "symbol"])

    assert "depth" not in result.data_vars
    assert "level" not in result.dims
    assert set(result.data_vars) == {"close", "anomaly_flag"}


def test_requesting_a_missing_dimension_raises_and_names_what_is_there():
    """Silently returning a differently-shaped dataset is how the original
    bug stayed invisible; the replacement must fail at the call."""
    backend = XrBackend().to_internal(_panel())
    with pytest.raises(ValueError) as excinfo:
        backend.get_xarray_dataset(["timestamp", "venue"])

    message = str(excinfo.value)
    assert "venue" in message
    assert "timestamp" in message and "symbol" in message


def test_the_backend_object_is_not_mutated_by_a_narrowing_request():
    """`filter_by_date`/`filter_by_symbol` on this same interface narrow IN
    PLACE, so an implementation written by analogy with them would truncate
    the store its caller shares with everything else (this is RV-01's failure
    mode). `get_xarray_dataset` must not."""
    panel = _panel()
    backend = XrBackend().to_internal(panel)
    backend.get_xarray_dataset(["timestamp"])

    assert set(backend.data.data_vars) == {"close", "anomaly_flag"}
    assert dict(backend.data.sizes) == {"timestamp": 4, "symbol": 2}


# --------------------------------------------------------------------------
# PlBackend: the implementation `indexes` always meant something to
# --------------------------------------------------------------------------


def test_plbackend_still_builds_its_dataset_from_the_named_index():
    frame = pl.LazyFrame(
        {
            "timestamp": list(TIMES) * 2,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4,
            "close": list(range(8)),
        }
    )
    result = PlBackend().to_internal(frame).get_xarray_dataset(
        ["timestamp", "symbol"]
    )
    assert result["close"].dims == ("timestamp", "symbol")


def test_plbackend_rejects_a_missing_indexes_argument():
    """The ABC's default is `None` so `XrBackend`'s no-arg callers are
    contract-legal. A LazyFrame has no dimensions to fall back on, so
    `PlBackend` must say so rather than fail somewhere inside pandas."""
    frame = pl.LazyFrame({"timestamp": list(TIMES), "close": list(range(4))})
    with pytest.raises(ValueError, match="indexes"):
        PlBackend().to_internal(frame).get_xarray_dataset()


# --------------------------------------------------------------------------
# The knock-on: RawDataset.time_interval
# --------------------------------------------------------------------------


def test_time_interval_works_under_xrbackend(tmp_path: Path):
    """The reason defect E mattered.

    `RawDataset.time_interval` is the only caller that asks for a single
    dimension. Under `XrBackend` it raised -- first `TypeError` from diffing
    the bool `anomaly_flag`, then, once that variable was removed,
    `AttributeError: 'Dataset' object has no attribute 'to_series'`.

    The panel here carries a bool variable AND a weekend-shaped gap, so it
    reproduces the first failure and exercises the `.mode()` the property
    uses for exactly that reason.
    """
    config = RawDatasetConfig(
        zarr_file_path=str(tmp_path / "panel" / "interval.zarr")
    )
    dataset = PanelDataset(config)
    dataset.from_raw_data()

    interval = dataset.time_interval
    assert isinstance(interval, np.timedelta64)
    assert interval == np.timedelta64(1, "D")


def test_time_interval_takes_the_mode_not_the_first_gap(tmp_path: Path):
    """The `.mode()` is load-bearing: a daily panel with a weekend hole must
    still report 1 day. Without it the property would report whatever the
    first gap happened to be."""
    config = RawDatasetConfig(
        zarr_file_path=str(tmp_path / "panel" / "gappy.zarr")
    )
    dataset = PanelDataset(config, gappy=True)
    dataset.from_raw_data()

    assert dataset.time_interval == np.timedelta64(1, "D")
