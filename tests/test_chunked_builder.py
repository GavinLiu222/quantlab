"""``ChunkedBuilder``: a factor's store built one chunk of time at a time.

When a range would not fit in memory, the builder splits it into chunks of
one calendar period each (``TimeChunkPlanner``), each warmed up on its own.
``build`` and ``extend`` write chunk by chunk through the wrapped factor's
own ``build`` and ``extend``, so the store is only ever the owner's;
``compute`` fills one in-memory panel chunk by chunk, or refuses when the
output alone would not fit. Joined in time order, the chunks equal the
whole range computed at once: NaN in the same cells, values within
floating-point tolerance. A factor that does not own its store (a view,
a factor pinned to some outputs) is refused before anything is computed.

The prices here are 91 daily bars over January to March 2024, so a
``"month"`` granularity gives three chunks.
"""

import dataclasses

import numpy as np
import pytest
import xarray as xr
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

import quantlab.factor.chunking as chunked_module
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import (
    FactorConfig, NeutralizedConfig, PolarsFactorConfig, RosterConfig,
)
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.chunking import ChunkedBuilder
from quantlab.factor.predefined.momentum import Momentum
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.factor.predefined.roster import RosterFactor

_T = 91
_TIMES = np.datetime64("2024-01-01") + np.arange(_T).astype("timedelta64[D]")
_START, _END = "2024-01-01", "2024-03-31"
_SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


class TwoOutputs(FactorKunQuant):
    """``ma_dev`` and ``close``: a store a factor pinned to one of them does not own."""

    def _get_factor_names(self):
        return ("ma_dev", "close")

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("Close")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev")
            Output(close, "close")
        return Function(builder.ops)


class RollingMean(FactorKunQuant):
    """``ma_dev``: the close over its 5-bar mean, minus 1."""

    def _get_factor_names(self):
        return ("ma_dev",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("Close")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev")
        return Function(builder.ops)


def _prices() -> xr.Dataset:
    rng = np.random.default_rng(7)
    close = 50 + rng.normal(0, 1, size=(_T, len(_SYMBOLS))).cumsum(axis=0)
    close[:20, 2] = np.nan  # a late listing: NaN must land in the same cells
    return xr.Dataset(
        {"Close": (("timestamp", "symbol"), close)},
        coords={"timestamp": _TIMES, "symbol": _SYMBOLS},
    )


@pytest.fixture
def prices(tmp_path):
    return FrameDataset(_prices()).to_zarr(tmp_path / "prices.zarr")


def _kunquant(prices, store) -> RollingMean:
    return RollingMean(FactorConfig(
        warmup_bars=4, dataset=prices, mode="batch", data_columns=("Close",),
        file_path=str(store), njobs=2,
    ))


def _polars(prices, store) -> Momentum:
    return Momentum(PolarsFactorConfig(
        warmup_bars=3, dataset=prices, file_path=str(store), kwargs={"n": 3},
    ))


@pytest.fixture(params=[_kunquant, _polars], ids=["kunquant", "polars"])
def make(request):
    return request.param


def _stored(factor) -> xr.Dataset:
    return xr.open_zarr(factor.store_path).load()


# A rolling sum is kept as a running total from the first bar computed, so a
# chunk that starts later differs in the last bits of its dtype: about one
# float32 step of the inputs' scale for KunQuant, which then survives the
# cancellation in a ratio minus one.
_TOLERANCE = {
    np.dtype("float32"): {"rtol": 1e-5, "atol": 1e-6},
    np.dtype("float64"): {"rtol": 1e-10, "atol": 1e-12},
}


def _assert_same(chunked: xr.Dataset, whole: xr.Dataset) -> None:
    assert list(chunked.data_vars) == list(whole.data_vars)
    xr.testing.assert_equal(chunked["timestamp"], whole["timestamp"])
    xr.testing.assert_equal(chunked["symbol"], whole["symbol"])
    for name in whole.data_vars:
        got, want = chunked[name].values, whole[name].values
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_allclose(got, want, equal_nan=True, **_TOLERANCE[got.dtype])


def test_a_chunked_build_equals_the_whole_build(tmp_path, prices, make):
    whole = make(prices, tmp_path / "whole.zarr").build(_START, _END)
    inner = make(prices, tmp_path / "chunked.zarr")

    ChunkedBuilder(inner, granularity="month").build(_START, _END)

    _assert_same(_stored(inner), _stored(whole))
    assert inner.store_range() == (_START, _END)


def test_a_chunked_extend_equals_the_whole_build(tmp_path, prices, make):
    whole = make(prices, tmp_path / "whole.zarr").build(_START, _END)
    inner = make(prices, tmp_path / "chunked.zarr").build(_START, "2024-01-10")

    ChunkedBuilder(inner, granularity="month").extend(_END)

    _assert_same(_stored(inner), _stored(whole))
    assert inner.store_range() == (_START, _END)


def test_the_chunks_are_one_calendar_period_each(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    calls = []
    build, extend = inner.build, inner.extend
    monkeypatch.setattr(inner, "build", lambda s, e: calls.append(("build", s, e)) or build(s, e))
    monkeypatch.setattr(inner, "extend", lambda e: calls.append(("extend", e)) or extend(e))

    ChunkedBuilder(inner, granularity="month").build(_START, _END)

    assert [(c[0], str(c[-1])[:10]) for c in calls] == [
        ("build", "2024-01-31"), ("extend", "2024-02-29"), ("extend", _END),
    ]
    assert calls[0][1] == _START


def test_a_failed_build_resumes_with_extend(tmp_path, prices, monkeypatch):
    whole = _kunquant(prices, tmp_path / "whole.zarr").build(_START, _END)
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    builder = ChunkedBuilder(inner, granularity="month")
    extend = inner.extend

    def crash_in_march(end):
        if str(end) >= "2024-03":
            raise MemoryError("killed")
        return extend(end)

    monkeypatch.setattr(inner, "extend", crash_in_march)
    with pytest.raises(MemoryError):
        builder.build(_START, _END)
    assert inner.store_range()[1].startswith("2024-02-29")

    monkeypatch.setattr(inner, "extend", extend)
    builder.extend(_END)

    _assert_same(_stored(inner), _stored(whole))


def test_a_chunked_compute_equals_the_whole_compute(tmp_path, prices, make):
    inner = make(prices, tmp_path / "chunked.zarr")
    whole = inner.compute(_START, _END)

    chunked = ChunkedBuilder(inner, granularity="month")

    _assert_same(chunked.compute("2024-01-05", _END), whole.sel(timestamp=slice("2024-01-05", None)))


def test_compute_refuses_when_the_output_alone_does_not_fit(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    # The output is 91 bars x 5 symbols x 1 variable x 8 bytes = 3640 bytes.
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: 3000)

    with pytest.raises(MemoryError, match=r"build\(start, end\)"):
        ChunkedBuilder(inner).compute(_START, _END)


def test_the_coarsest_granularity_that_fits_is_chosen(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    peak, _ = inner.cell_bytes()
    # Room for a month and its warm-up, not for a quarter.
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: (31 + 4) * 5 * peak)

    builder = ChunkedBuilder(inner)

    assert builder.plan(_START, _END)[0] == "month"


def test_a_range_that_fits_is_one_chunk(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    calls = []
    build = inner.build
    monkeypatch.setattr(inner, "build", lambda s, e: calls.append((s, e)) or build(s, e))
    monkeypatch.setattr(inner, "extend", lambda e: pytest.fail("one chunk needs no extend"))

    ChunkedBuilder(inner).build(_START, _END)

    assert calls == [(_START, _END)]


def test_too_little_memory_for_the_finest_granularity_is_refused(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: 1)

    with pytest.raises(MemoryError, match="hour"):
        ChunkedBuilder(inner).build(_START, _END)


def test_the_graph_is_compiled_once_for_every_chunk(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    compiles = []
    make = inner._make
    monkeypatch.setattr(inner, "_make", lambda: compiles.append(1) or make())

    ChunkedBuilder(inner, granularity="month").build(_START, _END)

    assert len(compiles) == 1
    assert inner._lib is None


# --- a resampled factor, built a chunk at a time ------------------------------

_HOURS = np.datetime64("2024-01-01T00") + np.arange(91 * 24).astype("timedelta64[h]")


@pytest.fixture
def hourly(tmp_path):
    rng = np.random.default_rng(11)
    close = 50 + rng.normal(0, 0.2, size=(len(_HOURS), len(_SYMBOLS))).cumsum(axis=0)
    close[: 20 * 24, 2] = np.nan
    panel = xr.Dataset(
        {"Close": (("timestamp", "symbol"), close)},
        coords={"timestamp": _HOURS, "symbol": _SYMBOLS},
    )
    return FrameDataset(panel).to_zarr(tmp_path / "hourly.zarr")


def _daily(hourly, store):
    return _kunquant(hourly, store).resample("1d", "last")


def test_a_resampled_factor_builds_in_chunks(tmp_path, hourly):
    whole = _daily(hourly, tmp_path / "whole.zarr").compute(_START, _END)
    inner = _daily(hourly, tmp_path / "chunked.zarr")

    ChunkedBuilder(inner, granularity="month").build(_START, _END)

    assert inner.store_range() == (_START, _END)
    _assert_same(inner.read(_START, _END), whole)


def test_a_resampled_factor_computes_in_chunks(tmp_path, hourly):
    whole = _daily(hourly, tmp_path / "whole.zarr").compute("2024-01-15", _END)
    chunked = ChunkedBuilder(_daily(hourly, tmp_path / "c.zarr"), granularity="month")

    _assert_same(chunked.compute("2024-01-15", _END), whole)


def test_chunks_finer_than_the_resampled_bars_are_refused(tmp_path, hourly):
    with pytest.raises(ValueError, match="finer than"):
        ChunkedBuilder(_daily(hourly, tmp_path / "c.zarr"), granularity="hour")


# --- chunks computed in parallel worker processes ------------------------------


def test_a_parallel_build_equals_the_whole_build(tmp_path, prices, make):
    whole = make(prices, tmp_path / "whole.zarr").build(_START, _END)
    inner = make(prices, tmp_path / "chunked.zarr")

    ChunkedBuilder(inner, granularity="month", workers=2).build(_START, _END)

    assert inner.store_range() == (_START, _END)
    _assert_same(_stored(inner), _stored(whole))


def test_a_parallel_extend_equals_the_whole_build(tmp_path, prices):
    whole = _kunquant(prices, tmp_path / "whole.zarr").build(_START, _END)
    inner = _kunquant(prices, tmp_path / "chunked.zarr").build(_START, "2024-01-20")

    ChunkedBuilder(inner, granularity="month", workers=2).extend(_END)

    _assert_same(_stored(inner), _stored(whole))


def test_a_parallel_resampled_build_equals_the_whole(tmp_path, hourly):
    whole = _daily(hourly, tmp_path / "whole.zarr").compute(_START, _END)
    inner = _daily(hourly, tmp_path / "chunked.zarr")

    ChunkedBuilder(inner, granularity="month", workers=3).build(_START, _END)

    _assert_same(inner.read(_START, _END), whole)


def test_a_parallel_compute_equals_the_whole_compute(tmp_path, prices):
    whole = _kunquant(prices, tmp_path / "whole.zarr").compute("2024-01-10", _END)
    chunked = ChunkedBuilder(_kunquant(prices, tmp_path / "c.zarr"), granularity="month", workers=2)

    _assert_same(chunked.compute("2024-01-10", _END), whole)


def test_parallel_workers_need_a_factor_that_rebuilds_from_its_config(tmp_path):
    held = _kunquant(FrameDataset(_prices()), tmp_path / "c.zarr")  # a panel in memory, no store

    with pytest.raises(ValueError, match="rebuild"):
        ChunkedBuilder(held, granularity="month", workers=2).build(_START, _END)


def test_workers_are_cut_to_the_chunks_that_fit_at_once(tmp_path, prices, monkeypatch):
    chunked = ChunkedBuilder(_kunquant(prices, tmp_path / "c.zarr"), granularity="month", workers=8)
    _, chunks = chunked.plan(_START, _END)
    assert chunked.parallel_workers(chunks) == 3  # three chunks

    one_chunk = chunked._peak_bytes(31)
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: 2 * one_chunk)
    assert chunked.parallel_workers(chunks) == 2


def test_workers_must_be_positive(tmp_path, prices):
    with pytest.raises(ValueError, match="workers"):
        ChunkedBuilder(_kunquant(prices, tmp_path / "c.zarr"), workers=0)


def test_build_and_extend_return_the_factor_whose_store_they_wrote(tmp_path, prices):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")
    builder = ChunkedBuilder(inner, granularity="month")

    assert builder.build(_START, "2024-02-10") is inner
    assert builder.extend(_END) is inner
    assert inner.store_range() == (_START, _END)
    assert builder.factor is inner


# --- a factor that does not own its store ------------------------------------


def _refuse_compute(factor, monkeypatch):
    monkeypatch.setattr(factor, "compute", lambda *a: pytest.fail("computed a chunk"))


def test_a_view_is_refused_before_anything_is_computed(tmp_path, prices, monkeypatch):
    inner = _kunquant(prices, tmp_path / "x.zarr").build(_START, _END)
    roster = FrameDataset(xr.Dataset(
        {"Close": (("timestamp", "symbol"), np.ones((_T, 2)))},
        coords={"timestamp": _TIMES, "symbol": ["BBB", "DDD"]},
    )).to_zarr(tmp_path / "roster.zarr")
    view = RosterFactor(RosterConfig(factor=inner, roster=roster))
    _refuse_compute(inner, monkeypatch)

    with pytest.raises(ValueError, match=r"RosterFactor\.build\(\): a view has no store of its own"):
        ChunkedBuilder(view, granularity="month", workers=2)


def test_a_pinned_factor_is_refused_before_anything_is_computed(tmp_path, prices, monkeypatch):
    two = TwoOutputs(FactorConfig(
        warmup_bars=4, dataset=prices, mode="batch", data_columns=("Close",),
        file_path=str(tmp_path / "two.zarr"), njobs=2,
    )).build(_START, _END)
    pinned = TwoOutputs(dataclasses.replace(two.config, factor_names=("ma_dev",)))
    _refuse_compute(pinned, monkeypatch)

    with pytest.raises(ValueError, match=r"TwoOutputs\.build\(\): the store at .* only the factor whose outputs"):
        ChunkedBuilder(pinned, granularity="month")


# --- the symbols a chunk is sized on (#251) --------------------------------


def _wide_exposures(tmp_path) -> FrameDataset:
    """Market cap and industry of the five priced symbols and ten more."""
    symbols = [*_SYMBOLS, *(f"X{i}" for i in range(10))]
    shape = (_T, len(symbols))
    return FrameDataset(xr.Dataset(
        {
            "marketcap": (("timestamp", "symbol"), np.full(shape, 1e9)),
            "industry": (("timestamp", "symbol"), np.tile(np.arange(len(symbols)) % 2, (_T, 1)).astype(float)),
        },
        coords={"timestamp": _TIMES, "symbol": symbols},
    )).to_zarr(tmp_path / "exposures.zarr")


def _neutral(tmp_path, prices) -> NeutralizedFactor:
    """The five-symbol ``RollingMean`` neutralized against fifteen symbols' exposures."""
    return NeutralizedFactor(NeutralizedConfig(
        factor=_kunquant(prices, tmp_path / "inner.zarr"), dataset=_wide_exposures(tmp_path),
        file_path=str(tmp_path / "neutral.zarr"), njobs=2,
    ))


def test_a_plain_factor_outputs_its_datasets_symbols(tmp_path, prices):
    inner = _kunquant(prices, tmp_path / "chunked.zarr")

    assert inner.output_symbols() == _SYMBOLS


def test_a_neutralized_factor_outputs_its_inner_factors_symbols(tmp_path, prices):
    neutral = _neutral(tmp_path, prices)

    assert len(neutral.config.dataset.stored_symbols()) == 15
    assert neutral.output_symbols() == _SYMBOLS


def test_a_roster_factor_outputs_the_kept_roster(tmp_path, prices):
    roster = FrameDataset(xr.Dataset(
        {"Close": (("timestamp", "symbol"), np.ones((_T, 3)))},
        coords={"timestamp": _TIMES, "symbol": ["DDD", "BBB", "ZZZ"]},
    )).to_zarr(tmp_path / "roster.zarr")
    factor = RosterFactor(RosterConfig(factor=_kunquant(prices, tmp_path / "x.zarr"), roster=roster))

    assert factor.output_symbols() == ["BBB", "DDD"]


def test_a_neutralized_factor_is_sized_on_its_inner_factors_symbols(tmp_path, prices, monkeypatch):
    neutral = _neutral(tmp_path, prices)
    peak, _ = neutral.cell_bytes()
    # Room for the whole range on the five output symbols, not on all fifteen.
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: (_T + neutral.warmup_bars) * 5 * peak)

    builder = ChunkedBuilder(neutral)

    assert builder.plan(_START, _END)[0] is None
    # The exposures' fifteen symbols, the old count, would have cut it.
    monkeypatch.setattr(neutral, "output_symbols", neutral.config.dataset.stored_symbols)
    assert builder.plan(_START, _END)[0] is not None


def test_workers_make_the_automatic_plan_cut_enough_chunks(tmp_path, prices):
    inner = _kunquant(prices, tmp_path / "c.zarr")

    assert ChunkedBuilder(inner).plan(_START, _END)[0] is None  # fits whole
    granularity, chunks = ChunkedBuilder(inner, workers=3).plan(_START, _END)
    assert granularity == "month" and len(chunks) == 3  # the coarsest with three chunks


def test_workers_fall_back_when_no_granularity_gives_them_all_a_chunk(tmp_path, prices):
    inner = _kunquant(prices, tmp_path / "c.zarr")

    granularity, chunks = ChunkedBuilder(inner, workers=500).plan(_START, _END)

    assert granularity is None and len(chunks) == 1  # as with one worker


def test_workers_take_a_finer_granularity_when_their_chunks_do_not_fit_side_by_side(tmp_path, prices, monkeypatch):
    chunked = ChunkedBuilder(_kunquant(prices, tmp_path / "c.zarr"), workers=3)
    month = chunked._peak_bytes(31)
    monkeypatch.setattr(chunked_module, "memory_budget", lambda: 3 * month - 1)

    granularity, _ = chunked.plan(_START, _END)

    assert granularity == "day"
