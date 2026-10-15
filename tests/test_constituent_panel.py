"""Tests for the interval-to-daily-panel densification (DATA-05, D-04).

Every test in this file runs OFFLINE. The only route to the network in this
code path is `quantlab.universe.requests.get`, which the shared
`mock_universe_fetchers` fixture in `tests/conftest.py` replaces with a
URL-keyed fake that raises `AssertionError` on any unexpected URL. The
densification tests do not even need that: they drive module-local
`IndexConstituentDataset` subclasses whose `_build_intervals()` returns a
hand-written frame and which therefore perform no I/O at all.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from loguru import logger

from conftest import WHOLE_STORE
from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.raw import IndexConstituentDataset
from quantlab.dataset.raw import RawDataset
from quantlab.dataset._support.cleaning import clean_membership_panel
from quantlab.dataset.constituent import SP500ConstituentDataset

_COVERAGE_START = "1976-07-01"


def _make_config(tmp_path, **overrides) -> ConstituentDatasetConfig:
    """Module-local config constructor (03.1-PATTERNS.md §6: shared mocks live
    in conftest, per-module config constructors stay module-local)."""
    params = dict(
        zarr_file_path=str(tmp_path / "us_equity" / "sp500_constituent.zarr"),
        cache_dir=str(tmp_path / "reference" / "_cache"),
    )
    params.update(overrides)
    return ConstituentDatasetConfig(**params)  # type: ignore[arg-type]


def _hand_built_intervals() -> pl.DataFrame:
    """Four membership intervals chosen so every assertion in this module is
    arithmetic a reader can check by hand.

    - `OPEN1`   2000-01-03 -> open        (still a member)
    - `CLOSED1` 2000-01-03 -> 2010-06-15  (a removal on a known date)
    - `DLIST1` 1980-01-02 -> 1985-03-04 (ends decades before the right edge)
    - `LATE1`   2015-09-01 -> open        (joins after the panel's left edge)
    """
    return pl.DataFrame(
        [
            ("OPEN1", "2000-01-03", None),
            ("CLOSED1", "2000-01-03", "2010-06-15"),
            ("DLIST1", "1980-01-02", "1985-03-04"),
            ("LATE1", "2015-09-01", None),
        ],
        schema=["symbol", "start_date", "end_date"],
        orient="row",
    )


def _closed_only_intervals() -> pl.DataFrame:
    """The two CLOSED rows only, maximum observed date 2010-06-15.

    Exists so the no-open-membership branch of the right-edge rule is testable
    without any dependence on what today's date happens to be.
    """
    return pl.DataFrame(
        [
            ("CLOSED1", "2000-01-03", "2010-06-15"),
            ("DLIST1", "1980-01-02", "1985-03-04"),
        ],
        schema=["symbol", "start_date", "end_date"],
        orient="row",
    )


class _PanelFixture(IndexConstituentDataset):
    """Densification-only fixture: performs NO I/O of any kind.

    `_build_intervals()` returns a hand-written frame instead of reaching a
    fetcher, so these tests exercise `_densify()` in isolation from the
    network, the cache and the Zarr store.
    """

    def _pit_coverage_start(self) -> str:
        return _COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        return _hand_built_intervals()


class _ClosedOnlyPanelFixture(_PanelFixture):
    """Same no-I/O fixture, but with every membership closed.

    Performs no network, cache or store access either -- it exists purely so
    the `horizon = observed` branch of the right-edge rule can be asserted
    against a fixed date rather than against `today`.
    """

    def _build_intervals(self) -> pl.DataFrame:
        return _closed_only_intervals()


class _NullStartDatePanelFixture(_PanelFixture):
    """Same no-I/O fixture, with one interval carrying a null `start_date` --
    the shape a change-log row without a date produces.
    """

    def _build_intervals(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                ("OPEN1", "2000-01-03", None),
                ("NODATE1", None, "2010-06-15"),
            ],
            schema=["symbol", "start_date", "end_date"],
            orient="row",
        )


def _panel(dataset: IndexConstituentDataset) -> xr.Dataset:
    return dataset.from_raw_data().get_xarray_dataset()


def _captured_warnings():
    """Attach a temporary in-memory loguru sink.

    loguru does not propagate to stdlib `logging`, so pytest's `caplog` sees
    nothing (`tests/test_universe.py:60-73` records the same constraint).
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    return messages, sink_id


def test_sp500_panel_round_trips_through_zarr(mock_universe_fetchers, tmp_path):
    """DATA-05 / D-04: the S&P 500 membership panel survives a full
    build -> Zarr write -> fresh-instance read cycle as a boolean
    `(timestamp, symbol)` grid."""
    cfg = _make_config(tmp_path)
    SP500ConstituentDataset(cfg).from_raw_data().save()

    reloaded = (
        SP500ConstituentDataset(_make_config(tmp_path)).panel(*WHOLE_STORE)
    )

    assert isinstance(reloaded, xr.Dataset)
    assert tuple(reloaded["is_member"].dims) == ("timestamp", "symbol")
    assert set(reloaded.data_vars) == {"is_member"}
    assert reloaded["is_member"].dtype == np.dtype(bool)

    symbols = set(reloaded["symbol"].values.tolist())
    # `ADDED1` is the fixture's still-open member; `ZZZZ` was removed in 1985
    # and must still be a column, not an absent symbol.
    assert "ADDED1" in symbols
    assert "ZZZZ" in symbols


def test_removal_effective_date_is_still_a_membership_day(tmp_path):
    """RESEARCH Finding 6 bullet 1 -- interval half-openness.

    A removal's `effective_date` IS a membership day. This matches
    `quantlab/universe.py:UniverseCatalog.get_symbols_as_of`'s
    `end_date >= as_of_date` comparison exactly. An off-by-one here is
    invisible unless a fixture pins a known removal date and asserts the
    boolean on exactly that day.
    """
    panel = _panel(
        _PanelFixture(
            _make_config(
                tmp_path, start_date="2000-01-01", end_date="2020-12-31"
            )
        )
    )
    closed1 = panel["is_member"].sel(symbol="CLOSED1")

    assert bool(closed1.sel(timestamp="2010-06-14").item()) is True
    assert bool(closed1.sel(timestamp="2010-06-15").item()) is True
    assert bool(closed1.sel(timestamp="2010-06-16").item()) is False


def test_open_interval_densifies_true_to_the_right_edge(tmp_path):
    """RESEARCH Finding 6 bullet 2 -- `end_date is None` means "still a
    member" and must densify True through the panel's right edge rather than
    being dropped as null. A null-dropping implementation still produces a
    plausible-looking panel, so the right edge is asserted specifically.
    """
    panel = _panel(
        _PanelFixture(
            _make_config(
                tmp_path, start_date="2000-01-01", end_date="2020-12-31"
            )
        )
    )
    open1 = panel["is_member"].sel(symbol="OPEN1")

    assert bool(open1.isel(timestamp=-1).item()) is True
    assert bool(open1.sel(timestamp="2012-07-04").item()) is True

    after_start = open1.sel(timestamp=slice("2000-01-03", None))
    assert bool(after_start.all().item()) is True


def test_left_edge_is_clamped_to_the_index_coverage_start(tmp_path):
    """RESEARCH Finding 6 bullet 4 / CONFLICT 5 -- the panel never starts
    before the index's own `PIT_COVERAGE_START`, even though the inherited
    default is 1900-01-01. Without the clamp the panel gains a 76-year
    all-False region that reads as "nobody was a member" rather than
    "unknown".

    This construction leaves BOTH dates at their inherited sentinels
    (`Date.START_DATE` and `Date.END_DATE`), and the announcement rule exempts
    both -- the caller requested nothing on either edge -- so a correct
    implementation is completely silent here. That is what makes the
    unqualified "no warning at all" assertion safe.

    If this test ever goes red, the fix is NEVER to delete a warning: tests
    `..._pre_coverage_start_date_is_clamped_with_a_warning` and
    `test_right_edge_is_clamped_to_the_observed_horizon...` are the two that
    require the warnings to exist, and the announcement rule in 03.1-03-PLAN's
    `<interfaces>` is the arbiter.
    """
    messages, sink_id = _captured_warnings()
    try:
        dataset = _PanelFixture(_make_config(tmp_path, start_date=None))
        panel = _panel(dataset)
    finally:
        logger.remove(sink_id)

    assert dataset.config.start_date == _COVERAGE_START
    assert pd.Timestamp(
        panel["timestamp"].values[0]
    ) == pd.Timestamp(_COVERAGE_START)
    assert messages == []


def test_explicitly_requested_pre_coverage_start_date_is_clamped_with_a_warning(
    tmp_path,
):
    """An EXPLICITLY requested window narrowed without a signal is the same
    class of silent narrowing this phase exists to eliminate: a caller who
    asks for 1950 and silently receives 1976 has been given less than they
    requested with no way to notice.

    This is the asymmetry fix that pairs with the right-edge truncation
    warning -- one announcement rule, applied to both edges, exempting only
    each edge's inherited sentinel.
    """
    messages, sink_id = _captured_warnings()
    try:
        dataset = _PanelFixture(
            _make_config(tmp_path, start_date="1950-01-01")
        )
    finally:
        logger.remove(sink_id)

    assert dataset.config.start_date == _COVERAGE_START
    assert any(
        "1950-01-01" in message and _COVERAGE_START in message
        for message in messages
    )


def test_delisted_symbol_is_a_real_mostly_false_column(tmp_path):
    """RESEARCH Finding 6 bullet 5 -- the symbol axis is the ALL-TIME union.

    Survivorship bias re-enters exactly here: an absent column is
    indistinguishable from a symbol that was never a member. `DLIST1`'s
    membership ended in 1985, entirely before this window, so it must be
    present as a real, fully-False column.
    """
    panel = _panel(
        _PanelFixture(_make_config(tmp_path, start_date="2000-01-01"))
    )

    assert "DLIST1" in panel["symbol"].values.tolist()
    delisted = panel["is_member"].sel(symbol="DLIST1")
    assert bool(delisted.any().item()) is False


def test_right_edge_reaches_today_when_a_membership_is_still_open(tmp_path):
    """An open membership is by definition CURRENT, so the last observed
    change event is only a lower bound on the right edge. Stopping there would
    end the panel weeks or months short of the present -- precisely the live
    edge where a universe mask gets used.

    `end_date` is left at the inherited `Date.END_DATE` sentinel here, so the
    announcement rule exempts it and no truncation warning may be emitted.
    This assertion is what locks that exemption: re-introducing an
    unconditional truncation warning fails here rather than passing unnoticed.
    """
    today = pd.Timestamp.today().normalize()

    messages, sink_id = _captured_warnings()
    try:
        panel = _panel(
            _PanelFixture(
                _make_config(tmp_path, start_date="2000-01-01", end_date=None)
            )
        )
    finally:
        logger.remove(sink_id)

    assert pd.Timestamp(panel["timestamp"].values[-1]) == today
    assert bool(
        panel["is_member"].sel(symbol="OPEN1").isel(timestamp=-1).item()
    ) is True
    assert messages == []


def test_right_edge_is_clamped_to_the_observed_horizon_when_every_membership_is_closed(
    tmp_path,
):
    """With NO open membership there is no basis for extending to today, so
    the maximum observed date is the honest right edge -- extending past it
    would fabricate membership the source never reported.

    This is also the ONE test that forces the right-edge truncation warning to
    exist at all: every other construction in this module leaves `end_date` at
    its exempt `Date.END_DATE` sentinel. The `end_date="2030-01-01"` here is
    deliberately explicit for that reason.

    The warning is captured with a temporary in-memory loguru sink rather than
    `caplog` -- loguru does not propagate to stdlib logging
    (`tests/test_universe.py:60-73`).
    """
    messages, sink_id = _captured_warnings()
    try:
        panel = _panel(
            _ClosedOnlyPanelFixture(
                _make_config(tmp_path, end_date="2030-01-01")
            )
        )
    finally:
        logger.remove(sink_id)

    assert pd.Timestamp(
        panel["timestamp"].values[-1]
    ) == pd.Timestamp("2010-06-15")
    assert any(
        "2030-01-01" in message and "2010-06-15" in message
        for message in messages
    )


def test_calendar_day_axis_is_contiguous_and_sorted(tmp_path):
    """RESEARCH Finding 6 bullet 3 -- the axis is CALENDAR days, not trading
    days: contiguous, strictly increasing, weekends included.

    The downstream consequence a consumer must know: a join against OHLCV
    data, which only carries trading days, must reindex or `.sel()` this panel
    onto the price panel's timestamps. The weekend rows carry the last trading
    day's membership forward.
    """
    panel = _panel(
        _PanelFixture(
            _make_config(
                tmp_path, start_date="2000-01-01", end_date="2000-01-31"
            )
        )
    )
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)

    deltas = np.diff(timestamps.values)
    assert np.all(deltas == np.timedelta64(1, "D"))
    assert timestamps.is_monotonic_increasing
    # 2000-01-08 was a Saturday.
    assert pd.Timestamp("2000-01-08") in timestamps
    assert pd.Timestamp("2000-01-08").dayofweek == 5


def test_construction_performs_no_network_call_and_no_store_read(
    monkeypatch, tmp_path
):
    """CONFLICT 1 -- `_reset_symbols()` is overridden to a no-op.

    The inherited body is wrong three times over for this class: (1) the
    symbol axis is derived from the membership-interval table rather than from
    a store, so overwriting `config.symbols` with the store's symbols is
    meaningless; (2) its `FileNotFoundError` fallback calls `from_raw_data()`,
    which here reaches a REMOTE FETCH, so merely constructing the object on a
    fresh clone would perform an unannounced HTTP request; (3) only
    `FileNotFoundError` is caught, so any network or parse error would escape
    `__init__` and the object could not be constructed offline at all.

    Without the override, every test in this file would need the network mock
    active merely to construct the object.
    """

    def _explode(*args, **kwargs):
        raise AssertionError("network reached during construction")

    monkeypatch.setattr("quantlab.universe.requests.get", _explode)

    cfg = _make_config(
        tmp_path / "does-not-exist", symbols=("AAPL", "MSFT")
    )
    dataset = SP500ConstituentDataset(cfg)

    assert dataset.config.symbols == ("AAPL", "MSFT")


@pytest.mark.parametrize("bad_date", ["2000-1-3", "01/01/2005", "not-a-date"])
def test_non_iso_date_is_rejected_at_the_config_boundary(tmp_path, bad_date):
    """WR-12. Every date comparison in this pipeline is LEXICOGRAPHIC on
    strings, and nothing validated the format. A non-padded or non-ISO value
    therefore did not fail to match -- it compared WRONG and silently skipped
    the coverage clamp or the coverage guard: `"2007-2-1" >= "1976-07-01"` is
    True by string order, and `"1980-12-12" <= "01/01/2024"` is False.

    Note `"2000-1-3"` is REJECTED rather than repaired: a value that is nearly
    ISO is exactly the one a lexicographic comparison mishandles quietly, so
    guessing at it would preserve the ambiguity this guard exists to remove.
    """
    with pytest.raises(ValueError, match="ISO YYYY-MM-DD"):
        _PanelFixture(_make_config(tmp_path, start_date=bad_date))




def test_dates_are_normalized_to_canonical_iso_at_the_config_boundary(tmp_path):
    """WR-12, positive half. An accepted alternate ISO spelling is rewritten to
    canonical zero-padded YYYY-MM-DD, so every downstream comparison can stay a
    plain string comparison.
    """
    dataset = _PanelFixture(_make_config(tmp_path, start_date="20000103"))

    assert dataset.config.start_date == "2000-01-03"


def test_as_of_pins_the_right_edge_making_the_panel_reproducible(tmp_path):
    """WR-07. With an open membership the right edge was
    `max(observed, pd.Timestamp.today())`, so rebuilding the same config on two
    days produced two differently-shaped Zarr stores while `save()` overwrites
    with mode="w" -- the single most influential parameter of the output shape
    came from the clock, not the config (CLAUDE.md 可复现性).

    Pinning `as_of` makes the panel a function of the config alone.
    """
    pinned = _panel(_PanelFixture(_make_config(tmp_path, as_of="2020-06-15")))

    assert pd.Timestamp(pinned["timestamp"].values[-1]) == pd.Timestamp("2020-06-15")
    # And it is stable across rebuilds, which `today` by construction is not.
    again = _panel(_PanelFixture(_make_config(tmp_path, as_of="2020-06-15")))
    assert pinned["timestamp"].equals(again["timestamp"])


def test_null_start_date_is_rejected_rather_than_silently_densified(tmp_path):
    """WR-13. `max(pd.Timestamp(row["start_date"]) for row in rows)` does not
    raise on a null: `pd.Timestamp(None)` is NaT and every comparison against
    NaT is False, so `max()` returns whichever value it happened to hold first
    rather than the true maximum -- silently corrupting the horizon. The same
    null then reaches the fill loop and yields an all-False column
    indistinguishable from "never a member".

    `start_date` can legitimately be null: it comes from `effective_date`.
    """
    dataset = _NullStartDatePanelFixture(_make_config(tmp_path))

    with pytest.raises(ValueError, match="null start_date"):
        dataset.from_raw_data()


def test_the_symbol_axis_takes_its_dtype_and_order_from_the_intervals(tmp_path):
    """The CONTROL ARM of 03.11-05: a ticker-keyed index is unchanged.

    `_densify` stopped calling `str()` on the values that reach `column_of`
    and `coords` so a PERMNO-keyed universe can land on an int64 axis. The
    same path still serves every Wikipedia-sourced index, whose labels are
    tickers, and for those nothing may move: the axis stays textual and stays
    in the lexicographic order `sorted()` used to give it.

    Asserted by dtype KIND, not by an exact dtype: `O`, `U` and `T` are all
    ways numpy/pandas spell "text" depending on how the coordinate was built,
    and pinning one of them would make this test about the construction route
    rather than about the contract.
    """
    panel = _panel(
        _PanelFixture(
            _make_config(tmp_path, start_date="2000-01-01", end_date="2000-01-31")
        )
    )

    labels = panel["symbol"].values.tolist()

    assert panel["symbol"].dtype.kind in "OUST", panel["symbol"].dtype
    assert all(isinstance(label, str) for label in labels), labels
    assert labels == sorted(labels)


def test_a_null_start_date_names_the_offending_symbol_in_the_message(tmp_path):
    """The one `str()` in `_densify` that must SURVIVE the axis change.

    `undated` renders its symbols for an ERROR MESSAGE, not for the axis, and
    a message is text whatever the axis is. Removing this `str()` along with
    the other three would make the refusal read `{np.int64(7000)}` on an
    integer-keyed universe -- or raise while formatting -- at the exact moment
    the operator most needs to know WHICH row has no date.
    """

    class _IntKeyedNullStart(_PanelFixture):
        def _build_intervals(self) -> pl.DataFrame:
            return pl.DataFrame(
                [
                    (10107, "2000-01-03", None),
                    (7000, None, "2010-06-15"),
                ],
                schema={
                    "symbol": pl.Int64,
                    "start_date": pl.String,
                    "end_date": pl.String,
                },
                orient="row",
            )

    with pytest.raises(ValueError) as refusal:
        _IntKeyedNullStart(_make_config(tmp_path)).from_raw_data()

    message = str(refusal.value)
    assert "null start_date" in message
    assert "'7000'" in message, message
    assert "10107" not in message, message


def test_membership_panel_cleaning_replaces_market_data_cleaning():
    """`clean_market_data()` is not merely unnecessary here, it is unusable:
    `validate_schema()` hard-raises on the five missing OHLCV columns, and
    even past that `flag_anomalies()` would append a second, meaningless
    all-False `anomaly_flag` variable to a panel whose only variable is
    already a boolean.
    """
    assert IndexConstituentDataset._clean is not RawDataset._clean

    timestamps = pd.date_range("2020-01-01", periods=3, freq="D")
    values = np.array(
        [[True, False], [True, True], [False, True]], dtype=bool
    )
    panel = xr.Dataset(
        {"is_member": (["timestamp", "symbol"], values)},
        coords={"timestamp": timestamps, "symbol": ["A", "B"]},
    )

    assert clean_membership_panel(panel) is panel

    float_panel = xr.Dataset(
        {"is_member": (["timestamp", "symbol"], values.astype(float))},
        coords={"timestamp": timestamps, "symbol": ["A", "B"]},
    )
    with pytest.raises(ValueError):
        clean_membership_panel(float_panel)
