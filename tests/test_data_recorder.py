"""A run records the data it reads at the dataset seam.

A ``DataRecorder`` is a context a run opens. Every read through
``BaseDataset.panel`` and ``Factor.read`` while it is open is logged as a
request; when it closes, each distinct request is hashed once and, when an
expected record is given, compared with it by digest alone. Outside a recorder
a read costs nothing extra.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

import quantlab.runs.record as fingerprint
from quantlab.factor.config import PolarsFactorConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.stock import StockDataset
from quantlab.runs.record import DataRecorder
from quantlab.runs.record import _active_recorder as active_recorder
from tests.backtest_fixtures import PastReturnFactor, write_price_store


@pytest.fixture
def warnings_logged():
    """Collect the loguru warnings emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


@pytest.fixture
def hashes(monkeypatch):
    """Count the calls to ``dataset_fingerprint``."""
    calls = []
    original = fingerprint._dataset_fingerprint

    def counting(ds, variables):
        calls.append(sorted(variables))
        return original(ds, variables)

    monkeypatch.setattr(fingerprint, "_dataset_fingerprint", counting)
    return calls


@pytest.fixture
def prices(tmp_path):
    """A ``StockDataset`` over a 60-bar business-day store of six symbols."""
    return StockDataset(write_price_store(tmp_path / "a"))


def _frame(symbols=("AAA", "BBB"), scale=1.0):
    """Return a ten-business-day panel with ``close`` scaled by ``scale``."""
    timestamps = pd.bdate_range("2024-01-01", periods=10)
    values = np.arange(len(timestamps) * len(symbols), dtype=float).reshape(
        len(timestamps), len(symbols)
    )
    return xr.Dataset(
        {"close": (["timestamp", "symbol"], values * scale),
         "volume": (["timestamp", "symbol"], values + 100.0)},
        coords={"timestamp": timestamps, "symbol": list(symbols)},
    )


def test_each_distinct_request_is_logged_and_hashed_once(prices, hashes):
    with DataRecorder(keys=[(prices, "price_dataset")]) as recorder:
        prices.panel("2024-01-02", "2024-01-31")
        prices.panel("2024-01-02", "2024-01-31")
        prices.panel("2024-01-02", "2024-02-15")
        assert hashes == []  # nothing is hashed before the run ends

    entries = recorder.records["price_dataset"]
    assert [e["request"]["end"] for e in entries] == ["2024-01-31T00:00:00", "2024-02-15T00:00:00"]
    assert len(hashes) == 2
    assert entries[0]["n_timestamps"] == 22
    assert entries[0]["request"]["variables"] is None
    assert "adjClose" in entries[0]["variables"]  # no variables: every variable


def test_two_spellings_of_the_same_bars_are_one_request(prices, hashes):
    """The request is the bars read, not the dates as a caller spelled them."""
    with DataRecorder(keys=[(prices, "price_dataset")]) as recorder:
        prices.panel("2023-12-30", "2024-01-31")  # a weekend before the first bar
        prices.panel(pd.Timestamp("2024-01-01"), "2024-01-31 23:00")

    (entry,) = recorder.records["price_dataset"]
    assert entry["request"]["start"] == "2024-01-01T00:00:00"
    assert len(hashes) == 1


def test_outside_a_recorder_a_read_records_and_hashes_nothing(prices, hashes):
    prices.panel("2024-01-02", "2024-01-31")

    assert active_recorder() is None
    assert hashes == []


def test_nested_recorders_log_to_the_innermost_only(tmp_path, prices):
    other = StockDataset(write_price_store(tmp_path / "b"))
    with DataRecorder(keys=[(prices, "outer")]) as outer:
        prices.panel("2024-01-02", "2024-01-31")
        with DataRecorder(keys=[(other, "inner")]) as inner:
            assert active_recorder() is inner
            other.panel("2024-01-02", "2024-01-31")
        assert active_recorder() is outer

    assert list(outer.records) == ["outer"]
    assert list(inner.records) == ["inner"]
    assert active_recorder() is None


def test_variables_narrow_the_panel_and_the_hash(prices, hashes):
    panel = prices.panel("2024-01-02", "2024-01-31", variables=["adjOpen", "adjClose"])
    assert sorted(panel.data_vars) == ["adjClose", "adjOpen"]

    with DataRecorder(keys=[(prices, "price_dataset")]) as recorder:
        prices.panel("2024-01-02", "2024-01-31", variables=["adjClose", "adjOpen"])

    (entry,) = recorder.records["price_dataset"]
    assert entry["variables"] == ["adjClose", "adjOpen"]
    assert entry["request"]["variables"] == ["adjClose", "adjOpen"]
    assert hashes == [["adjClose", "adjOpen"]]


def test_symbols_are_part_of_the_request(prices):
    with DataRecorder(keys=[(prices, "p")]) as recorder:
        prices.panel("2024-01-02", "2024-01-31", symbols=["BBB", "AAA"])
        prices.panel("2024-01-02", "2024-01-31")

    entries = recorder.records["p"]
    assert [e["request"]["symbols"] for e in entries] == [["BBB", "AAA"], None]
    assert [e["n_symbols"] for e in entries] == [2, 6]


def test_a_merged_dataset_records_its_leaves_not_itself():
    left = FrameDataset(_frame(("AAA", "BBB")))
    right = FrameDataset(_frame(("CCC",)))
    merged = MergedDataset([left, right])

    with DataRecorder(keys=[(merged, "m"), (left, "m.0"), (right, "m.1")]) as recorder:
        panel = merged.panel("2024-01-02", "2024-01-05", variables=["close"])

    assert list(panel.data_vars) == ["close"]
    assert sorted(recorder.records) == ["m.0", "m.1"]
    assert recorder.records["m.0"][0]["variables"] == ["close"]


def test_an_in_memory_dataset_hashes_its_held_panel():
    held = FrameDataset(_frame())
    with DataRecorder(keys=[(held, "held")]) as recorder:
        held.panel("2024-01-02", "2024-01-05")

    expected = fingerprint._dataset_fingerprint(
        _frame().sel(timestamp=slice("2024-01-02", "2024-01-05")), ["close", "volume"]
    )
    assert recorder.records["held"][0]["digest"] == expected["digest"]


def test_an_unmapped_dataset_falls_back_to_its_class_and_store(prices):
    with DataRecorder() as recorder:
        prices.panel("2024-01-02", "2024-01-05")

    assert list(recorder.records) == [f"StockDataset:{prices.store_path}"]


def test_a_dataset_given_two_keys_takes_the_first(prices):
    with DataRecorder(keys=[(prices, "first"), (prices, "second")]) as recorder:
        prices.panel("2024-01-02", "2024-01-05")

    assert list(recorder.records) == ["first"]


def test_a_factor_store_read_is_recorded_under_the_factor_key(tmp_path, hashes):
    factor = PastReturnFactor(
        PolarsFactorConfig(
            warmup_bars=5,
            dataset=StockDataset(write_price_store(tmp_path)),
            kwargs={"n": 3},
            file_path=str(tmp_path / "factor.zarr"),
        )
    )
    factor.build("2024-01-10", "2024-02-20")
    hashes.clear()

    with DataRecorder(keys=[(factor, "model.factors.0")]) as recorder:
        factor.read("2024-01-15", "2024-02-01")

    (entry,) = recorder.records["model.factors.0"]
    assert entry["variables"] == ["past_ret_3"]
    assert entry["request"] == {
        "start": "2024-01-15T00:00:00", "end": "2024-02-01T00:00:00", "symbols": None,
        "variables": None,
    }
    assert len(hashes) == 1


def test_a_factor_reads_symbols_are_part_of_its_request_and_its_hash(tmp_path):
    factor = PastReturnFactor(
        PolarsFactorConfig(
            warmup_bars=5,
            dataset=StockDataset(write_price_store(tmp_path)),
            kwargs={"n": 3},
            file_path=str(tmp_path / "factor.zarr"),
        )
    )
    factor.build("2024-01-10", "2024-02-20")

    with DataRecorder(keys=[(factor, "f")]) as recorder:
        factor.read("2024-01-15", "2024-02-01", symbols=["BBB", "AAA"])
        factor.read("2024-01-15", "2024-02-01")

    cut, whole = recorder.records["f"]
    assert cut["request"]["symbols"] == ["BBB", "AAA"] and whole["request"]["symbols"] is None
    assert cut["n_symbols"] == 2
    assert cut["digest"] != whole["digest"]

def _record(dataset, *ranges):
    """Return the records of reading ``ranges`` of ``dataset`` under key ``data``."""
    with DataRecorder(keys=[(dataset, "data")]) as recorder:
        for start, end in ranges:
            dataset.panel(start, end)
    return recorder.records


def _compare(dataset, expected, *ranges, keys=None):
    """Read ``ranges`` of ``dataset`` inside a recorder expecting ``expected``."""
    with DataRecorder(keys=keys or [(dataset, "data")], expected=expected) as recorder:
        for start, end in ranges:
            dataset.panel(start, end)
    return recorder


def test_an_unchanged_record_is_silent(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))

    _compare(FrameDataset(_frame()), expected, ("2024-01-02", "2024-01-05"))

    assert warnings_logged == []


def test_a_changed_digest_warns_and_never_raises(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))

    _compare(FrameDataset(_frame(scale=2.0)), expected, ("2024-01-02", "2024-01-05"))

    (message,) = warnings_logged
    assert "data fingerprint mismatch" in message
    assert "'data'" in message and "digest differs" in message
    assert "2024-01-02" in message and "4 timestamps x 2 symbols" in message


def test_only_the_digest_decides(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))
    expected["data"][0]["n_timestamps"] = 99  # explanation only, never compared

    _compare(FrameDataset(_frame()), expected, ("2024-01-02", "2024-01-05"))

    assert warnings_logged == []


def test_a_missing_and_an_extra_request_warn(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))

    _compare(FrameDataset(_frame()), expected, ("2024-01-02", "2024-01-08"))

    assert len(warnings_logged) == 2
    assert any("not read by this run" in m and "2024-01-05" in m for m in warnings_logged)
    assert any("absent from the expected record" in m and "2024-01-08" in m
               for m in warnings_logged)


def test_a_missing_and_an_extra_key_warn(warnings_logged):
    expected = _record(FrameDataset(_frame()), ("2024-01-02", "2024-01-05"))
    held = FrameDataset(_frame())

    _compare(held, expected, ("2024-01-02", "2024-01-05"), keys=[(held, "renamed")])

    assert len(warnings_logged) == 2
    assert any("'data'" in m and "not read by this run" in m for m in warnings_logged)
    assert any("'renamed'" in m and "absent from the expected record" in m
               for m in warnings_logged)


def test_the_failure_path_compares_partially_and_the_error_propagates(warnings_logged):
    expected = _record(
        FrameDataset(_frame()), ("2024-01-02", "2024-01-05"), ("2024-01-02", "2024-01-12")
    )
    changed = FrameDataset(_frame(scale=2.0))

    with pytest.raises(RuntimeError, match="boom"):
        with DataRecorder(keys=[(changed, "data")], expected=expected) as recorder:
            changed.panel("2024-01-02", "2024-01-05")
            raise RuntimeError("boom")

    (message,) = warnings_logged  # the request not read yet is not reported
    assert "digest differs" in message
    assert "comparison is PARTIAL" in message
    assert len(recorder.records["data"]) == 1


def test_a_failing_diagnostic_never_replaces_the_error(warnings_logged, monkeypatch):
    held = FrameDataset(_frame())

    def broken(ds, variables):
        raise OSError("disk gone")

    monkeypatch.setattr(fingerprint, "_dataset_fingerprint", broken)
    with pytest.raises(RuntimeError, match="boom"):
        with DataRecorder(keys=[(held, "data")], expected={}):
            held.panel("2024-01-02", "2024-01-05")
            raise RuntimeError("boom")

    assert any("OSError" in m for m in warnings_logged)


def test_a_merge_input_reads_its_own_name_of_a_shared_variable(spot_kline_zarr):
    from quantlab.dataset.spot import SpotKlineDataset

    spot = SpotKlineDataset(spot_kline_zarr())
    held = FrameDataset(_frame(("ZZZ1", "ZZZ2")))
    merged = MergedDataset([spot, held])

    with DataRecorder(keys=[(spot, "spot"), (held, "held")]) as recorder:
        panel = merged.panel("2024-01-02", "2024-01-05", variables=["close"])

    assert list(panel.data_vars) == ["close"]
    assert recorder.records["spot"][0]["variables"] == ["Close"]
    assert recorder.records["held"][0]["variables"] == ["close"]


def test_a_failing_diagnostic_after_a_successful_run_only_warns(warnings_logged, monkeypatch):
    held = FrameDataset(_frame())

    def broken(ds, variables):
        raise OSError("disk gone")

    monkeypatch.setattr(fingerprint, "_dataset_fingerprint", broken)
    with DataRecorder(keys=[(held, "data")]):
        held.panel("2024-01-02", "2024-01-05")

    (message,) = warnings_logged
    assert "OSError" in message and "continuing" in message


def test_a_merge_input_holding_none_of_the_variables_is_not_read():
    left = FrameDataset(_frame(("AAA",)))
    right = FrameDataset(_frame(("BBB",)).rename({"close": "bid"}))
    merged = MergedDataset([left, right])

    with DataRecorder(keys=[(left, "left"), (right, "right")]) as recorder:
        panel = merged.panel("2024-01-02", "2024-01-05", variables=["bid"])

    assert list(panel.data_vars) == ["bid"]
    assert list(recorder.records) == ["right"]
    with pytest.raises(KeyError, match="no input holds"):
        merged.panel("2024-01-02", "2024-01-05", variables=["ask"])


def test_a_kunquant_factor_reads_only_its_data_columns(spot_kline_zarr):
    from tests.test_factor_merge import MaDeviation
    from quantlab.factor.config import FactorConfig
    from quantlab.dataset.spot import SpotKlineDataset

    spot = SpotKlineDataset(spot_kline_zarr())
    factor = MaDeviation(FactorConfig(
        dataset=spot, mode="batch", data_columns=("close",), warmup_bars=5, njobs=1,
    ))

    with DataRecorder(keys=[(spot, "dataset")]) as recorder:
        factor.compute("2024-01-10", "2024-01-20")

    (entry,) = recorder.records["dataset"]
    assert entry["request"]["variables"] == ["Close"]  # the store's own name


def test_unrecorded_reads_reach_no_recorder_and_are_not_hashed(prices, hashes):
    from quantlab.runs.record import unrecorded

    with DataRecorder(keys=[(prices, "p")]) as outer:
        with unrecorded():
            assert active_recorder() is None
            prices.panel("2024-01-02", "2024-01-31")
            with DataRecorder(keys=[(prices, "inner")]) as inner:
                prices.panel("2024-01-02", "2024-01-05")
        assert active_recorder() is outer

    assert outer.records == {}
    assert list(inner.records) == ["inner"]
    assert len(hashes) == 1


def _one_variable(values, dtype) -> xr.Dataset:
    """A 2 x 2 panel holding ``values`` as ``close`` in ``dtype``."""
    return xr.Dataset(
        {"close": (["timestamp", "symbol"], np.asarray(values, dtype=dtype))},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=2), "symbol": ["A", "B"]},
    )


def test_a_variable_is_hashed_in_its_stored_dtype():
    """No up-cast: the dtype is part of the digest, NaN and -0.0 are canonical in it."""
    values = [[1.5, np.nan], [-0.0, 2.0]]
    as32 = fingerprint._dataset_fingerprint(_one_variable(values, np.float32), ["close"])
    as64 = fingerprint._dataset_fingerprint(_one_variable(values, np.float64), ["close"])
    assert as32["digest"] != as64["digest"]

    other_nan = np.frombuffer(np.uint32(0x7FC00001).tobytes(), dtype=np.float32)[0]
    other_bits = _one_variable([[1.5, other_nan], [0.0, 2.0]], np.float32)
    assert fingerprint._dataset_fingerprint(other_bits, ["close"])["digest"] == as32["digest"]


def test_integer_and_boolean_variables_are_hashed_as_stored():
    ints = fingerprint._dataset_fingerprint(_one_variable([[1, 2], [3, 4]], np.int64), ["close"])
    flags = fingerprint._dataset_fingerprint(_one_variable([[True, False], [False, True]], bool), ["close"])
    assert len(ints["digest"]) == len(flags["digest"]) == 64
    assert ints["digest"] != fingerprint._dataset_fingerprint(
        _one_variable([[1, 2], [3, 5]], np.int64), ["close"]
    )["digest"]


def _wide_panel(n_timestamps=50, n_symbols=7, seed=0) -> xr.Dataset:
    """A panel of three float32 variables with NaNs, plus an int and a bool one."""
    rng = np.random.default_rng(seed)
    shape = (n_timestamps, n_symbols)
    floats = {
        name: rng.standard_normal(shape).astype(np.float32)
        for name in ("alpha", "beta", "gamma")
    }
    floats["beta"][rng.random(shape) < 0.3] = np.nan
    return xr.Dataset(
        {
            **{name: (["timestamp", "symbol"], values) for name, values in floats.items()},
            "count": (["timestamp", "symbol"], rng.integers(0, 9, shape)),
            "flag": (["timestamp", "symbol"], rng.random(shape) < 0.5),
        },
        coords={
            "timestamp": pd.date_range("2024-01-01", periods=n_timestamps, freq="h"),
            "symbol": [f"S{i}" for i in range(n_symbols)],
        },
    )


@pytest.mark.parametrize("workers", [1, 2, 8])
@pytest.mark.parametrize("block_rows", [1, 3, 50, None])
def test_the_digest_does_not_depend_on_threads_or_blocks(workers, block_rows):
    panel = _wide_panel()
    names = list(panel.data_vars)
    serial = fingerprint._dataset_fingerprint(panel, names, workers=1, block_rows=None)

    record = fingerprint._dataset_fingerprint(
        panel, names, workers=workers, block_rows=block_rows
    )

    assert record["digest"] == serial["digest"]
    assert record["variable_digests"] == serial["variable_digests"]


def test_each_variable_has_its_own_digest_and_dtype():
    panel = _wide_panel()
    before = fingerprint._dataset_fingerprint(panel, ["alpha", "count", "flag"])
    changed = panel.copy(deep=True)
    changed["alpha"][3, 2] += 1

    after = fingerprint._dataset_fingerprint(changed, ["flag", "alpha", "count"])

    assert sorted(before["variable_digests"]) == ["alpha", "count", "flag"]
    assert before["variable_dtypes"] == {"alpha": "<f4", "count": "<i8", "flag": "|b1"}
    assert after["digest"] != before["digest"]
    assert after["variable_digests"]["alpha"] != before["variable_digests"]["alpha"]
    assert after["variable_digests"]["count"] == before["variable_digests"]["count"]


def test_relabelled_axes_change_the_digest_but_no_variable_digest():
    panel = _wide_panel()
    before = fingerprint._dataset_fingerprint(panel, ["alpha"])
    shifted = panel.assign_coords(timestamp=panel["timestamp"] + pd.Timedelta("1min"))

    after = fingerprint._dataset_fingerprint(shifted, ["alpha"])

    assert after["digest"] != before["digest"]
    assert after["variable_digests"] == before["variable_digests"]


def test_a_lazily_read_variable_is_hashed_a_block_at_a_time(tmp_path, monkeypatch):
    import tracemalloc

    panel = _wide_panel(n_timestamps=20000, n_symbols=100)[["alpha"]].astype(np.float64)
    panel.to_zarr(tmp_path / "wide.zarr", encoding={"alpha": {"chunks": (100, 100)}})
    whole = panel["alpha"].nbytes  # 16 MB
    monkeypatch.setattr(fingerprint, "_BLOCK_BYTES", 150 * 100 * 8)  # 150 bars, rounded up to 200

    def peak(**kwargs):
        lazy = xr.open_zarr(tmp_path / "wide.zarr")
        tracemalloc.start()
        try:
            record = fingerprint._dataset_fingerprint(lazy, ["alpha"], **kwargs)
            return record, tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    blocked, blocked_peak = peak()
    eager, eager_peak = peak(block_rows=20000)

    assert blocked["digest"] == eager["digest"]
    assert eager_peak > whole
    assert blocked_peak < whole / 8


def _records_of(panel, variables=None):
    """``DataRecorder.records`` of one whole-panel request of ``panel`` under ``data``."""
    names = variables or sorted(panel.data_vars)
    request = {"start": "2024-01-01", "end": "2024-01-03", "symbols": None, "variables": variables}
    return {"data": [{"request": request, **fingerprint._dataset_fingerprint(panel, names)}]}


def _mismatch(expected_panel, actual_panel, warnings_logged, variables=None):
    fingerprint.compare(
        {"data_fingerprint": _records_of(expected_panel, variables)},
        {"data_fingerprint": _records_of(actual_panel, variables)},
        owner="run",
    )
    (message,) = warnings_logged
    assert "digest differs" in message
    return message


def test_a_mismatch_names_the_variables_whose_values_changed(warnings_logged):
    panel = _wide_panel()
    changed = panel.copy(deep=True)
    changed["beta"][0, 0] = 7.0
    changed["count"][1, 1] += 1

    message = _mismatch(panel, changed, warnings_logged)

    assert "values of ['beta', 'count'] changed" in message
    assert "alpha" not in message


def test_a_mismatch_names_variables_added_to_and_missing_from_the_store(warnings_logged):
    panel = _wide_panel()
    grown = panel.assign(delta=panel["alpha"] * 2).drop_vars("gamma")

    message = _mismatch(panel, grown, warnings_logged)

    assert "variables added ['delta']" in message
    assert "variables missing ['gamma']" in message
    assert "values of" not in message


def test_a_mismatch_of_identical_values_blames_the_labels(warnings_logged):
    panel = _wide_panel()
    renamed = panel.assign_coords(symbol=[f"T{i}" for i in range(panel.sizes["symbol"])])

    message = _mismatch(panel, renamed, warnings_logged, variables=["alpha"])

    assert "the bar or symbol labels changed" in message
    assert "values of" not in message


def test_a_mismatch_with_another_extent_blames_the_extent_not_the_values(warnings_logged):
    panel = _wide_panel()

    message = _mismatch(panel, panel.isel(timestamp=slice(0, -1)), warnings_logged)

    assert "the bars or symbols changed" in message
    assert "50 timestamps" in message and "49 timestamps" in message
    assert "values of" not in message


def test_a_mismatch_names_a_changed_dtype(warnings_logged):
    panel = _wide_panel()
    narrowed = panel.assign(count=panel["count"].astype(np.int32))

    message = _mismatch(panel, narrowed, warnings_logged, variables=["alpha", "count"])

    assert "dtype of 'count' <i8 -> <i4" in message
    assert "values of" not in message


def test_build_settings_are_recorded_and_decide_the_digest(prices, warnings_logged, monkeypatch):
    monkeypatch.setattr(StockDataset, "_build_settings", lambda self: {"window": "a"})
    expected = _record(prices, ("2024-01-02", "2024-01-05"))
    (entry,) = expected["data"]
    assert entry["settings"] == {"window": "a"}

    _compare(prices, expected, ("2024-01-02", "2024-01-05"))
    assert warnings_logged == []

    monkeypatch.setattr(StockDataset, "_build_settings", lambda self: {"window": "b"})
    recorder = _compare(prices, expected, ("2024-01-02", "2024-01-05"))
    (changed,) = recorder.records["data"]
    assert changed["variable_digests"] == entry["variable_digests"]
    assert changed["digest"] != entry["digest"]
    (message,) = warnings_logged
    assert "build settings changed from {'window': 'a'} to {'window': 'b'}" in message
    assert "values of" not in message


def test_a_dataset_without_build_settings_records_none(prices):
    (entry,) = _record(prices, ("2024-01-02", "2024-01-05"))["data"]
    assert "settings" not in entry


@pytest.fixture
def unsorted_store(tmp_path):
    """A zarr store whose symbol axis is not sorted (it grew at its end)."""
    path = tmp_path / "unsorted.zarr"
    _frame(symbols=("CCC", "AAA", "BBB")).to_zarr(path)
    return path


def test_an_unsorted_store_is_hashed_in_its_stored_order_without_a_copy(
    unsorted_store, monkeypatch
):
    def no_sorted_copy(self, *args, **kwargs):
        raise AssertionError("the fingerprint sorted a copy of the store")

    monkeypatch.setattr(xr.Dataset, "sortby", no_sorted_copy)
    first = fingerprint._dataset_fingerprint(xr.open_zarr(unsorted_store), ["close"])
    second = fingerprint._dataset_fingerprint(xr.open_zarr(unsorted_store), ["close"])

    assert first == second
    assert first["n_symbols"] == 3


def test_the_same_values_in_another_symbol_order_have_another_digest(unsorted_store):
    stored = xr.open_zarr(unsorted_store)
    reordered = stored.isel(symbol=[1, 2, 0])  # AAA, BBB, CCC: same cells

    stored_record = fingerprint._dataset_fingerprint(stored, ["close", "volume"])
    reordered_record = fingerprint._dataset_fingerprint(reordered, ["close", "volume"])

    assert stored_record["digest"] != reordered_record["digest"]
    assert stored_record["variable_digests"] != reordered_record["variable_digests"]
