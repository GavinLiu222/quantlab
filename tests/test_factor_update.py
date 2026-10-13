"""`Factor.extend(end)` -- the AUTOMATIC incremental interface of a factor store.

`build(start, end)` writes a factor store WHOLESALE and records its range;
`extend(end)` appends the bars after the recorded end. The incremental path
lives one layer down, in `XrBackend.widen_and_append`, and `extend()` is the
factor layer's route to it. It is AUTOMATIC: it works out for itself what
changed -- later dates, new symbols, new variables -- and reconciles each axis
without the caller naming which widening to perform:

    build(start, end)  writes WHOLESALE  (replaces the store)
    extend(end)        EXTENDS           (no route to overwrite a stored range)

Every test here names, in its docstring, the mutation that reddens it.

WHY EVERY STORE-TOUCHING TEST HERE RUNS TWICE (260908-dvv). `extend()` reaches
`widen_and_append` and through it `widen_data_vars`, which is where a
coordinate-encoding defect once shipped unseen, because the test factor built
its `symbol` coordinate from a python list. That round-trips through zarr to a
FIXED-WIDTH unicode store, while the store the real chunked ingest writes is
`object`-encoded and decodes to `StringDType()`. The coordinate now comes from
`conftest.symbol_coord` and the encoding is carried on the factor instance by
`_factor(..., encoding=...)`, so every store-touching test gets a
`[fixed_width]` and a `[variable_length]` id.
"""

import inspect
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import assert_stored_symbol_encoding, symbol_coord
from quantlab.factor.config import BaseFactorConfig
from quantlab.factor.base import Factor
from quantlab.backend.zarr import XrBackend

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeDataset:
    """A dataset `PanelFactor` never reads: its `compute` builds the panel."""


class PanelFactor(Factor):
    """The smallest concrete `Factor`: a `(timestamp, symbol)` panel.

    `prepare()` takes the dates, symbols and variables of the panel the next
    `compute()` returns, so one class can express every axis the extend path
    reconciles. `build()` and `extend()` are the real ones. No KunQuant graph,
    no compiled code, no real dataset.
    """

    SYMBOLS = ["AAA", "BBB"]

    #: Which of `conftest.SYMBOL_COORD_ENCODINGS` this instance's panels are
    #: built at. Deliberately NOT given a default: a factor constructed
    #: without one raises `AttributeError` inside `prepare()` rather than silently
    #: falling back to one arm, which is the failure mode this whole task
    #: exists to close.
    symbol_encoding: str

    def _get_factor_names(self) -> tuple[str, ...]:
        return ("alpha",)

    def compute(self, start, end) -> xr.Dataset:
        """Return the prepared panel, whatever range is asked for."""
        return self._prepared

    def prepare(
        self,
        dates: Sequence[str],
        symbols: Sequence[str] | None = None,
        variables: Mapping[str, str] | Sequence[str] = ("alpha",),
        offset: float = 0.0,
    ):
        symbols = list(self.SYMBOLS if symbols is None else symbols)
        if not isinstance(variables, Mapping):
            variables = {str(name): "float64" for name in variables}

        shape = (len(dates), len(symbols))
        data = {}
        for index, (name, dtype) in enumerate(variables.items()):
            if np.issubdtype(np.dtype(dtype), np.bool_):
                values = np.ones(shape, dtype=bool)
            else:
                values = (
                    np.arange(shape[0] * shape[1], dtype="float64").reshape(shape)
                    + offset
                    + index * 1000.0
                ).astype(dtype)
            data[name] = (["timestamp", "symbol"], values)

        self._prepared = xr.Dataset(
            data,
            coords={
                "timestamp": pd.to_datetime(list(dates)),
                "symbol": symbol_coord(symbols, self.symbol_encoding),
            },
        )
        return self


class FilledPanelFactor(PanelFactor):
    """A subclass that OVERRIDES the fill seam, the way a classification-label
    factor carrying a boolean variable would have to."""

    def _widen_fill_values(self) -> dict:
        return {"anomaly_flag": False}


def _factor(tmp_path: Path, cls=PanelFactor, *, encoding: str) -> PanelFactor:
    """Build a `PanelFactor` whose panels carry the named symbol encoding.

    `encoding` is keyword-only and REQUIRED, matching `_panel`'s signature in
    the two sibling widening suites, so a new test cannot forget the axis.
    """
    factor = cls(
        BaseFactorConfig(
            warmup_bars=1,
            dataset=_FakeDataset(),  # type: ignore[arg-type]
            file_path=str(tmp_path / "alpha.zarr"),
        )
    )
    factor.symbol_encoding = encoding
    return factor


def _stored(factor: Factor) -> xr.Dataset:
    return xr.open_zarr(factor.config.file_path).load()


EARLY = ["2024-02-01", "2024-02-02", "2024-02-03"]
LATER = ["2024-03-01", "2024-03-02"]

#: A stored range LONGER than `XrBackend.APPEND_DIM_CHUNK` (260908-g30).
#:
#: The length is load-bearing for the same reason it is in
#: `tests/test_symbol_axis_widening.py`: a widen forced over budget by
#: `MAX_WIDEN_BYTES = 0` floors its block at `APPEND_DIM_CHUNK`, so a shorter
#: store makes the chunked path a single write that no multi-block assertion
#: can tell from a correct one. This is an ARGUMENT to the existing
#: `PanelFactor.prepare()`, not a new builder.
LONG = [d.isoformat() for d in pd.date_range("2024-02-01", periods=600, freq="h")]
AFTER_LONG = ["2024-03-01", "2024-03-02"]


def _build(factor: PanelFactor, dates: Sequence[str], **kwargs) -> PanelFactor:
    """Write the store from ``dates`` with ``build()``, recording their range."""
    return factor.prepare(dates, **kwargs).build(dates[0], dates[-1])


def _extend(factor: PanelFactor, dates: Sequence[str], **kwargs) -> PanelFactor:
    """Append ``dates`` with ``extend()`` up to their last one."""
    return factor.prepare(dates, **kwargs).extend(dates[-1])


# ---------------------------------------------------------------------------
# The automatic extend interface
# ---------------------------------------------------------------------------


def test_extend_appends_a_later_date_range_to_the_store(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The tracer: "build February, then extend to March" simply works.

    Both ranges present, the axis unique and strictly increasing, and
    February's values bit-identical afterwards.

    RED under: routing `extend()` through a wholesale write, or through
    anything that does not extend.
    """
    factor = _factor(tmp_path, encoding=symbol_encoding)
    _build(factor, EARLY)
    before = _stored(factor)

    _extend(factor, LATER, offset=100.0)

    after = _stored(factor)
    assert after.sizes["timestamp"] == len(EARLY) + len(LATER)
    index = after["timestamp"].to_index()
    assert index.is_unique and index.is_monotonic_increasing
    np.testing.assert_array_equal(
        after["alpha"].values[: len(EARLY)], before["alpha"].values
    )


def test_extend_reconciles_a_new_symbol_without_being_told_to(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """AUTOMATIC is the whole point: the caller names no widening.

    A roster that grew between two runs is the ordinary case, not an
    exceptional one, and a factor-layer caller has no business knowing that
    the storage layer calls this "widening the symbol axis". It passes a
    panel; `extend()` works out what changed.

    RED under: M11 (route `extend()` through `append()` instead of
    `widen_and_append()`) -- the symbol-coordinate guard refuses.
    """
    factor = _factor(tmp_path, encoding=symbol_encoding)
    _build(factor, EARLY)

    _extend(factor, LATER, symbols=["AAA", "BBB", "CCC"], offset=100.0)

    after = _stored(factor)
    assert after["symbol"].values.tolist() == ["AAA", "BBB", "CCC"]
    assert after.sizes["timestamp"] == len(EARLY) + len(LATER)
    # The new symbol carries NaN across the historical block and real values
    # on the new one -- a widen backfills, it does not invent history.
    assert np.isnan(after["alpha"].sel(symbol="CCC").values[: len(EARLY)]).all()
    assert not np.isnan(
        after["alpha"].sel(symbol="CCC").values[len(EARLY) :]
    ).any()


def test_extend_reconciles_a_new_variable_without_being_told_to(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The third axis, reconciled just as automatically as the second.

    The SYMBOL axis AGREES on both sides -- the same roster before and after,
    with only the variable set growing. That is load-bearing rather than
    incidental: it is what routes this panel into `widen_and_append`'s fast
    path and therefore what makes M9 observable here at all. A panel that grew
    BOTH axes would be reconciled by the symbol widen regardless and would
    stay green under M9, proving nothing about the variable axis.

    This is also the property that proves `extend()` routes through the
    three-axis path rather than a plain append.

    RED under: M11 (via the Task 1 new-variable refusal) and M9 (leaving
    `widen_and_append`'s short-circuit testing the symbol axis alone -- the
    fast path is taken and the closing `append()` refuses `beta`).
    """
    factor = _factor(tmp_path, encoding=symbol_encoding)
    _build(factor, EARLY)

    _extend(factor, LATER, variables=["alpha", "beta"], offset=100.0)

    after = _stored(factor)
    assert sorted(after.data_vars) == ["alpha", "beta"]
    assert after["symbol"].values.tolist() == ["AAA", "BBB"]
    assert after.sizes["timestamp"] == len(EARLY) + len(LATER)
    # The new variable is backfilled across the historical block only.
    assert np.isnan(after["beta"].values[: len(EARLY)]).all()
    assert not np.isnan(after["beta"].values[len(EARLY) :]).any()


def test_extend_declares_no_overwrite_parameter(tmp_path: Path) -> None:
    """`extend()` EXTENDS, and offers no route to overwrite a range the store
    already holds. Recomputing is `build()`'s job; a `mode=` on `extend()`
    would blur the boundary the two interfaces exist to keep sharp. The
    behavioural half, refusing an end the store already covers, is
    `tests/test_factor_date_range.py::test_extend_refuses_an_end_the_store_already_covers`.

    RED under: giving `Factor.extend` any parameter besides `end`.

    THE ONE TEST IN THIS MODULE THAT DOES NOT TAKE `symbol_encoding`, and that
    is a DECISION rather than an oversight (260908-dvv). It never builds a
    panel and never opens a store: `inspect.signature(Factor.extend)` reads the
    same tuple whatever a coordinate is encoded as, so a second id here would
    be a case with byte-identical behaviour -- cost without coverage. It is the
    single member of the exemption set in
    `tests/test_widening_fixture_realism.py`, so removing it from there without
    also giving this test the fixture fails the family guard.
    """
    parameters = tuple(inspect.signature(Factor.extend).parameters)
    assert parameters == ("self", "end")


def test_the_widen_fill_seam_reaches_the_widening_call(
    tmp_path: Path,
    symbol_encoding: str,
) -> None:
    """DVAR-11: `_widen_fill_values()` is a real seam, asserted BEHAVIOURALLY.

    The default is `{}` because every factor and label panel is float today --
    KunQuant emits float arrays, and even `SpotBinaryReturn` builds its binary
    label from `op.ConstantOp(1.0)`/`(0.0)`, so it is float64 rather than
    bool. The seam is required regardless of that default: the widening
    refuses to backfill a non-float variable without an explicit fill, so
    without this hook a future non-float subclass has no way to widen AT ALL.

    Asserting only that the method exists and returns `{}` would not span
    that. So the span is a pair of runs on the same shape: the base subclass
    is REFUSED on a boolean variable, and the subclass overriding the seam
    succeeds and keeps the dtype. That is the mapping demonstrably arriving at
    the widening call.

    RED under: dropping the seam, hardcoding `{}` at the call site, or not
    passing `fill_values` through to `widen_and_append`.
    """
    assert Factor._widen_fill_values(  # type: ignore[misc]
        object.__new__(PanelFactor)
    ) == {}

    plain = _factor(tmp_path / "plain", encoding=symbol_encoding)
    _build(plain, EARLY)
    with pytest.raises(ValueError) as excinfo:
        _extend(
            plain, LATER, variables={"alpha": "float64", "anomaly_flag": "bool"}
        )
    assert "anomaly_flag" in str(excinfo.value), str(excinfo.value)
    assert "fill_values" in str(excinfo.value), str(excinfo.value)

    filled = _factor(
        tmp_path / "filled",
        cls=FilledPanelFactor,
        encoding=symbol_encoding,
    )
    _build(filled, EARLY)
    _extend(filled, LATER, variables={"alpha": "float64", "anomaly_flag": "bool"})

    after = _stored(filled)
    assert after["anomaly_flag"].dtype == np.dtype(bool)
    # False across the historical block: it was not flagged because there was
    # nothing to flag. True on the new window, where the panel says so.
    assert not after["anomaly_flag"].values[: len(EARLY)].any()
    assert after["anomaly_flag"].values[len(EARLY) :].all()


def test_extend_appends_a_new_symbol_in_place_with_no_call_site_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """A new listing reaches the in-place extension, never a rewrite (ADR 0032).

    `Factor` has no raw tier, so `Factor.extend()` ->
    `XrBackend.widen_and_append` is the only path a new listing can take here.
    The new symbol is appended at the end of the axis by
    `XrBackend.extend_symbol_axis`, which resizes the arrays in place; the
    rewrite (`widen_symbol_axis`) is never called, whatever the store's size.

    RED under: `widen_and_append` widening onto the sorted union again.
    """
    factor = _factor(tmp_path, encoding=symbol_encoding)
    _build(factor, LONG, symbols=["BBB", "DDD"])
    before = _stored(factor)

    def refuse(*args, **kwargs):
        raise AssertionError("extend() rewrote the store")

    monkeypatch.setattr(XrBackend, "widen_symbol_axis", refuse)
    _extend(factor, AFTER_LONG, symbols=["AAA", "BBB", "DDD"], offset=100.0)

    after = _stored(factor)
    assert after["symbol"].values.tolist() == ["BBB", "DDD", "AAA"]
    assert after.sizes["timestamp"] == len(LONG) + len(AFTER_LONG)
    for symbol in ("BBB", "DDD"):
        np.testing.assert_array_equal(
            after["alpha"].sel(symbol=symbol).values[: len(LONG)],
            before["alpha"].sel(symbol=symbol).values,
        )
    assert np.isnan(after["alpha"].sel(symbol="AAA").values[: len(LONG)]).all()
    assert not np.isnan(after["alpha"].sel(symbol="AAA").values[len(LONG) :]).any()
    assert_stored_symbol_encoding(str(factor.config.file_path), symbol_encoding)


def test_build_then_extend_with_a_new_symbol_equals_one_build_by_label(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The extended store holds what one build holds, compared by label.

    The axes differ in order (a new symbol is appended, a build sorts), and
    nothing reads a store by position (ADR 0032).

    RED under: a new symbol's history landing under another label.
    """
    symbols = ["AAA", "BBB", "CCC"]
    whole = _factor(tmp_path / "whole", encoding=symbol_encoding)
    _build(whole, EARLY + LATER, symbols=symbols)
    whole_panel = whole._prepared.copy()
    # The early bars of the late listing are missing, as an extension leaves them.
    late = (whole_panel["symbol"] == symbols[0]) & whole_panel["timestamp"].isin(
        pd.to_datetime(EARLY)
    )
    whole_panel["alpha"] = whole_panel["alpha"].where(~late)
    whole._prepared = whole_panel
    whole.build(EARLY[0], LATER[-1])

    pieces = _factor(tmp_path / "pieces", encoding=symbol_encoding)
    pieces._prepared = whole_panel.isel(timestamp=slice(None, len(EARLY))).sel(
        symbol=["BBB", "CCC"]
    )
    pieces.build(EARLY[0], EARLY[-1])
    pieces._prepared = whole_panel.isel(timestamp=slice(len(EARLY), None))
    pieces.extend(LATER[-1])

    expected, got = _stored(whole), _stored(pieces)
    assert got["symbol"].values.tolist() == ["BBB", "CCC", "AAA"]
    xr.testing.assert_identical(got.sortby("symbol"), expected.sortby("symbol"))


# ---------------------------------------------------------------------------
# One store layout whichever way it was written (#247)
# ---------------------------------------------------------------------------

UNSORTED = ["CCC", "AAA", "BBB"]


def test_build_writes_the_symbol_axis_in_sorted_order(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """A dataset whose symbol axis is not sorted still gets a sorted store.

    RED under: `build()` writing the computed panel in the dataset's order.
    """
    factor = _factor(tmp_path, encoding=symbol_encoding)
    _build(factor, EARLY, symbols=UNSORTED)

    stored = _stored(factor)
    assert stored["symbol"].values.tolist() == ["AAA", "BBB", "CCC"]
    expected = factor._prepared["alpha"].sel(symbol=["AAA", "BBB", "CCC"])
    np.testing.assert_array_equal(stored["alpha"].values, expected.values)


def test_build_then_extend_matches_one_build_position_for_position(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """Built whole or built then extended, the store has one layout.

    RED under: `build()` keeping the dataset's unsorted order, which
    `extend()` then rewrites sorted.
    """
    whole = _factor(tmp_path / "whole", encoding=symbol_encoding)
    _build(whole, EARLY + LATER, symbols=UNSORTED)

    pieces = _factor(tmp_path / "pieces", encoding=symbol_encoding)
    _build(pieces, EARLY, symbols=UNSORTED)
    pieces._prepared = whole._prepared.isel(timestamp=slice(len(EARLY), None))
    pieces.extend(LATER[-1])

    expected, got = _stored(whole), _stored(pieces)
    assert got["symbol"].values.tolist() == expected["symbol"].values.tolist()
    np.testing.assert_array_equal(got["alpha"].values, expected["alpha"].values)


def test_widen_and_append_creates_a_store_with_a_sorted_symbol_axis(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The first chunk `ChunkedFactor` writes lands sorted too.

    RED under: `widen_and_append` delegating a missing store straight to
    `append` with the panel's own order.
    """
    panel = xr.Dataset(
        {"alpha": (["timestamp", "symbol"], np.arange(6.0).reshape(2, 3))},
        coords={
            "timestamp": pd.to_datetime(EARLY[:2]),
            "symbol": symbol_coord(UNSORTED, symbol_encoding),
        },
    )
    path = str(tmp_path / "new.zarr")
    XrBackend().to_internal(panel).widen_and_append(path)

    stored = xr.open_zarr(path).load()
    assert stored["symbol"].values.tolist() == ["AAA", "BBB", "CCC"]
    np.testing.assert_array_equal(
        stored["alpha"].values,
        panel["alpha"].sel(symbol=["AAA", "BBB", "CCC"]).values,
    )
    assert_stored_symbol_encoding(path, symbol_encoding)
