"""`quantlab.acquisition.registry.convert()` -- the registry-level raw->Zarr
entry point (DATA-07, phase 03.5).

The requirement these tests exist for is stated as a CALL SITE, not as a
behaviour: an in-process caller holding only a `SourceDescriptor` and a
`DatasetConfig` converts an already-persisted raw parquet tier into a Zarr
store, without naming `StockDataset`, `TiingoAcquisition` or any `ingest_*.py`
anywhere. Every test below therefore reaches the conversion through
`convert(DESCRIPTOR, config)` and never through the Dataset class -- a test
that constructed the dataset itself would pass while the requirement failed.

Offline, all of it. No credential is read, no network call is made, and every
path is under `tmp_path`. The raw tier is a hive-partitioned parquet tree
written by the shared `hive_raw_tree` fixture, in the shape
`StockDataset._scan_raw` asserts: `Path(raw_data_dir_path).name == vendor`.
"""

import dataclasses
from pathlib import Path
from typing import Callable

import pytest
import xarray as xr

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.raw import ConversionResult
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import convert

#: Two calendar years, a handful of observed days in each, so a `year` window
#: is cheap and there are exactly TWO planned windows to write, skip and count.
#: Sparse across symbols on purpose (`B` trades only in 2024): a window that
#: derived its own symbol axis would come back with one column instead of the
#: pinned two, which is the misalignment `from_raw_data_chunked`'s pinned axis
#: exists to prevent -- and which `pinned_symbols` in the result reports.
_YEARS = (2023, 2024)
_DAYS_PER_YEAR = ("01-04", "06-15", "12-28")
_EXPECTED_WINDOWS = len(_YEARS)
_EXPECTED_SYMBOLS = 2


def _raw_rows(stock_pqt_row: Callable[..., dict]) -> list[dict]:
    rows = []
    for year in _YEARS:
        for day in _DAYS_PER_YEAR:
            date_str = f"{year}-{day}"
            rows.append(stock_pqt_row(date_str, "A", close=100.0))
            if year == 2024:
                rows.append(stock_pqt_row(date_str, "B", close=100.0))
    return rows


@pytest.fixture
def tiingo_raw_tier(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> Callable[..., DatasetConfig]:
    """Factory. `tiingo_raw_tier()` writes the two-year raw parquet panel once
    and returns a `DatasetConfig` a caller could hand straight to `convert()`.

    Calling it twice returns configs pointing at the SAME raw tree and the SAME
    Zarr path, which is what makes the idempotency probe a genuine second run
    over the first run's store rather than a fresh one.
    """
    raw_dir = tmp_path / "raw"
    hive_raw_tree(raw_dir, "tiingo", _raw_rows(stock_pqt_row), batch_key="panel")

    def _build(store_name: str = "out.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(raw_dir / "tiingo"),
            zarr_file_path=str(tmp_path / store_name),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
        )

    return _build


def test_convert_writes_a_real_zarr_store_and_returns_a_conversion_result(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """DATA-07's tracer: descriptor + config in, Zarr store on disk out.

    The assertions are deliberately split between the STORE and the RESULT.
    Asserting only the returned object would pass against a `convert()` that
    computed counts and wrote nothing; asserting only the store would pass
    against one that wrote but reported nothing a caller could render (D-04).
    Both halves are the requirement.
    """
    descriptor = DataSourceRegistry.get("tiingo")
    config = tiingo_raw_tier()

    result = convert(descriptor, config)

    # -- the store is real --------------------------------------------------
    assert Path(config.zarr_file_path).exists()
    stored = xr.open_zarr(config.zarr_file_path)
    assert set(stored.dims) >= {"timestamp", "symbol"}
    assert stored.sizes["symbol"] == _EXPECTED_SYMBOLS
    assert stored.sizes["timestamp"] == len(_YEARS) * len(_DAYS_PER_YEAR)

    # -- the outcome is renderable without reading that store ---------------
    assert isinstance(result, ConversionResult)
    assert result.zarr_path == config.zarr_file_path
    assert Path(result.ledger_path).exists()
    assert result.granularity == "year"
    assert result.pinned_symbols == _EXPECTED_SYMBOLS
    assert result.windows_planned == _EXPECTED_WINDOWS
    assert result.windows_written == _EXPECTED_WINDOWS
    assert result.windows_skipped == 0
    assert result.rows_written == len(_YEARS) * len(_DAYS_PER_YEAR)
    assert result.peak_window_bytes is not None and result.peak_window_bytes > 0
    assert result.resumed is False
    assert result.cancelled is False
    # D-11: `convert()` runs no guard and invents no estimate. `None` here is
    # the honest answer when the caller offered none, not a missing value.
    assert result.predicted_peak_bytes is None


def test_a_second_convert_over_the_same_window_writes_nothing_twice(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """DATA-07's idempotency probe row, as an assertion rather than as prose.

    A second call over the same config must not re-append a single window --
    re-appending would duplicate every timestamp in the store, which is the
    silent corruption the chunk ledger exists to make impossible. The proof is
    that the second run's SKIPPED count equals the first run's WRITTEN count
    and its own written count is zero.
    """
    descriptor = DataSourceRegistry.get("tiingo")
    first = convert(descriptor, tiingo_raw_tier())

    second = convert(descriptor, tiingo_raw_tier())

    assert second.windows_written == 0
    assert second.windows_skipped == first.windows_written
    assert second.resumed is True
    assert second.rows_written == 0
    # Nothing was materialised, so there is no observed peak to report.
    assert second.peak_window_bytes is None

    # The store itself is unchanged -- the counts above would also be produced
    # by a run that skipped the ledger check and appended anyway if the ledger
    # were the only thing consulted.
    stored = xr.open_zarr(first.zarr_path)
    assert stored.sizes["timestamp"] == len(_YEARS) * len(_DAYS_PER_YEAR)
    assert stored.sizes["symbol"] == _EXPECTED_SYMBOLS


def test_predicted_peak_is_echoed_back_untouched(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """`predicted_peak_bytes` is the CALLER'S number, carried not computed.

    D-11 leaves the RAM guard at the call site, so the entry point has no
    roster category to size against. Echoing the caller's own estimate into
    the result is what lets a report put prediction beside outcome without
    `convert()` pretending to be self-protecting.
    """
    result = convert(
        DataSourceRegistry.get("tiingo"),
        tiingo_raw_tier(),
        predicted_peak_bytes=123_456,
    )

    assert result.predicted_peak_bytes == 123_456
    # The echo must not disturb anything that was measured.
    assert result.windows_written == _EXPECTED_WINDOWS


def test_run_still_returns_an_acquisition_result_and_converts_nothing() -> None:
    """SC-2's surviving first half: conversion is a SEPARATE function.

    Pinned structurally rather than by running an acquisition (which would
    need a credential): `download_symbols` and `convert` are two module-level functions
    with two different return annotations, and `download_symbols` grew no conversion
    keyword. A `mode=`/`to_zarr=` flag appearing on `download_symbols` is exactly the
    regression 03.4 D-14 forbids, and it would be invisible to every
    behavioural test in this file.
    """
    import inspect

    from quantlab.acquisition import registry
    from quantlab.acquisition.symbol_batch import AcquisitionResult

    assert registry.download_symbols.__annotations__["return"] is AcquisitionResult
    assert registry.convert.__annotations__["return"] is ConversionResult

    download_params = set(inspect.signature(registry.download_symbols).parameters)
    assert not download_params & {"mode", "to_zarr", "convert", "granularity"}


# ---------------------------------------------------------------------------
# The SECOND vendor, and the three ways a capability lookup can refuse.
#
# Every refusal below is `ValueError` and every assertion is on the MESSAGE,
# not merely on the type. Asserting the type alone would pass against a single
# generic raise, which is exactly the shape these three arms replace: a caller
# who cannot tell "you asked for a combination nobody serves" from "you asked
# for one nobody can convert yet" has to read this module's source to act.
# ---------------------------------------------------------------------------


@pytest.fixture
def alpaca_raw_tier(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> Callable[..., DatasetConfig]:
    """The same two-year panel under an `alpaca`-terminated raw root.

    `StockDataset._scan_raw` asserts `Path(raw_data_dir_path).name == vendor`
    and then asserts the literal `vendor` column agrees, so the vendor token
    has to be threaded through both the path and the config -- which is also
    the reason this cannot reuse `tiingo_raw_tier` with a renamed config.
    """
    # Alpaca names its data type, so the bars sit in `alpaca/bars/`, one
    # level below the vendor root the config points at; `convert()` fills
    # `kwargs["data_type"]` in from the capability and scans there.
    raw_dir = tmp_path / "raw"
    hive_raw_tree(
        raw_dir, "alpaca", _raw_rows(stock_pqt_row), batch_key="panel", data_type="bars"
    )

    def _build(frequency: str = "1d", store_name: str = "alpaca.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(raw_dir / "alpaca"),
            zarr_file_path=str(tmp_path / store_name),
            market="us_equity",
            frequency=frequency,  # type: ignore[arg-type]
            vendor="alpaca",
        )

    return _build


def test_convert_reaches_the_second_vendor_through_the_same_entry_point(
    alpaca_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """The tracer's sideways expansion: a SECOND descriptor, same call shape.

    `StockDataset` answering for both vendors is the D-01 claim made concrete
    -- capabilities are keyed by `(market, frequency, data_type)`, which is the
    axis Dataset subclasses divide on, so one class on three rows across two
    vendors is one correct answer to three questions rather than a duplicated
    decision.
    """
    result = convert(DataSourceRegistry.get("alpaca"), alpaca_raw_tier())

    assert isinstance(result, ConversionResult)
    assert result.windows_written == _EXPECTED_WINDOWS
    assert result.pinned_symbols == _EXPECTED_SYMBOLS
    assert Path(result.zarr_path).exists()


def test_tick_is_refused_because_no_capability_carries_a_conversion_target(
    alpaca_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """SC-7: the tick refusal names phase 03.3 and the irregular event axis.

    Asserted on the SUBSTRINGS rather than on `ValueError` alone, deliberately.
    The type alone is satisfied by any raise at all -- including the
    placeholder this arm replaced -- so a message assertion is the only thing
    that distinguishes "refused for the right reason" from "blew up".

    And no store is created: a refusal that had already written a
    partially-flattened panel would leave behind exactly the
    plausible-looking-but-wrong artefact the refusal exists to prevent.
    """
    config = alpaca_raw_tier(frequency="tick", store_name="tick.zarr")

    with pytest.raises(ValueError) as excinfo:
        convert(DataSourceRegistry.get("alpaca"), config)

    message = str(excinfo.value)
    assert "not supported yet" in message
    assert "irregular event" in message
    assert not Path(config.zarr_file_path).exists()


def test_the_tick_lookup_MATCHES_and_the_refusal_is_the_absent_target() -> None:
    """The refusal is DATA, not a failed lookup (D-01).

    This is the assertion that stops the SC-7 arm from being re-implemented as
    `if frequency == "tick"` one day: `capabilities_for` must still RETURN the
    tick rows -- the vendor genuinely serves them, and the registry must keep
    advertising what it serves -- while every one of them carries
    `dataset_cls is None`. Phase 03.3 lands tick conversion by FILLING those
    two fields; nothing in `convert()` has to change.
    """
    descriptor = DataSourceRegistry.get("alpaca")

    tick_rows = descriptor.capabilities_for("us_equity", "tick")

    assert len(tick_rows) == 2
    assert {row.data_type for row in tick_rows} == {"quotes", "trades"}
    assert all(row.dataset_cls is None for row in tick_rows)
    # ...and the lookup did not fail: `supports()` still says yes.
    assert descriptor.supports("us_equity", "tick")


def test_an_unserved_combination_names_the_request_and_what_is_served(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """NO MATCH: the message names the requested tuple AND the served ones.

    A caller who mistyped one element learns which from the same message,
    rather than getting a bare "not supported" about a registry they cannot
    see -- the same courtesy `DataSourceRegistry.get()` already extends for a
    mistyped vendor token.
    """
    descriptor = DataSourceRegistry.get("tiingo")
    config = dataclasses.replace(tiingo_raw_tier(), frequency="1m")

    with pytest.raises(ValueError) as excinfo:
        convert(descriptor, config)

    message = str(excinfo.value)
    assert "1m" in message
    assert "us_equity" in message
    # What it DOES serve, enumerated from `capabilities` rather than restated.
    assert "'1d'" in message or '"1d"' in message


def test_two_matching_targets_without_a_data_type_refuse_rather_than_guess(
    isolated_registry,
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """AMBIGUOUS: two rows, two targets, no `data_type` -> refuse.

    This is the case D-02's whole argument is about. No shipped vendor has this
    shape today, which is precisely why it needs a test rather than a comment:
    the day one does, `convert()` must refuse instead of letting capability
    ORDER decide what gets converted.

    Built through `isolated_registry`, which restores
    `DataSourceRegistry.SOURCES` by rebinding the saved tuple on teardown, so
    the synthetic descriptor cannot leak into a later test.
    """
    from quantlab.acquisition import tiingo as tiingo_module
    from quantlab.acquisition.base import Capability, SourceDescriptor, register_source
    from quantlab.dataset.spot import SpotKlineDataset
    from quantlab.dataset.stock import StockDataset

    register_source(
        SourceDescriptor(
            vendor="twotarget",  # type: ignore[arg-type]
            display_name="Two Target Vendor",
            symbol_batch_cls=tiingo_module.TiingoAcquisition,
            config_factory=lambda **kw: None,  # type: ignore[arg-type,return-value]
            capabilities=(
                Capability(
                    market="us_equity",
                    frequency="1d",
                    data_type="bars",
                    dataset_cls=StockDataset,
                ),
                Capability(
                    market="us_equity",
                    frequency="1d",
                    data_type="klines",
                    dataset_cls=SpotKlineDataset,
                ),
            ),
            required_env=(),
        )
    )
    descriptor = isolated_registry.get("twotarget")

    with pytest.raises(ValueError) as excinfo:
        convert(descriptor, tiingo_raw_tier(store_name="ambiguous.zarr"))

    message = str(excinfo.value)
    assert "data_type" in message
    assert "bars" in message and "klines" in message

    # Supplying the discriminator resolves it -- the refusal is about the
    # MISSING argument, not about the vendor being unusable.
    assert descriptor.capabilities_for("us_equity", "1d", "bars")[
        0
    ].dataset_cls is StockDataset


def test_alpaca_rows_carry_targets_on_bars_and_none_on_tick() -> None:
    """The four Alpaca rows, pinned as DATA.

    `dataset_cls` on exactly the two `bars` rows and `None` on exactly the two
    `tick` rows is the whole SC-7 mechanism; a runtime check on the rows is
    what keeps a future edit from filling a tick row before Phase 03.3 has
    landed a Dataset that can express the axis.
    """
    from quantlab.dataset.stock import StockDataset

    capabilities = DataSourceRegistry.get("alpaca").capabilities

    bars = [c for c in capabilities if c.data_type == "bars"]
    tick = [c for c in capabilities if c.frequency == "tick"]

    assert len(bars) == 2
    assert all(c.dataset_cls is StockDataset for c in bars)
    assert len(tick) == 2
    assert all(c.dataset_cls is None for c in tick)


# ---------------------------------------------------------------------------
# The two handles: `reporter` and `cancel` (03.5 D-05, plan 06).
#
# These reach `RawDataset.from_raw_data_chunked()` DIRECTLY rather than
# through `convert()`, and that is a deliberate exception to this module's
# opening rule. The contract under test is the CHUNK LOOP's -- where the
# cancel token is observed, what is emitted per window, and what a reporter
# that raises can and cannot do. `convert()` owns one line of it (the
# forwarding), and that line gets its own test at the bottom of this section:
# a test that only ever went through `convert()` could not tell "the loop
# never checks the token" from "`convert()` drops the argument".
# ---------------------------------------------------------------------------

from quantlab.utils.progress import CancelToken, ProgressEvent, ProgressReporter


class _RecordingReporter(ProgressReporter):
    """Every event, in arrival order. The whole reporter."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]


class _CancelAfterNWritten(_RecordingReporter):
    """Drives the token from the loop's OWN events.

    Cancelling from a second thread would make "after exactly two windows" a
    race the test cannot win on a fast machine; cancelling on the second
    `window_written` event lands the token deterministically between window
    two's ledger record and window three's top-of-loop check.
    """

    def __init__(self, token: CancelToken, after: int) -> None:
        super().__init__()
        self._token = token
        self._after = after

    def emit(self, event: ProgressEvent) -> None:
        super().emit(event)
        if self.kinds().count("window_written") >= self._after:
            self._token.cancel()


class _ExplodingReporter(ProgressReporter):
    """Raises on every event. A UI bug, as a three-line class."""

    def __init__(self) -> None:
        self.calls = 0

    def emit(self, event: ProgressEvent) -> None:
        self.calls += 1
        raise RuntimeError("the console screen blew up")


_FOUR_YEARS = (2021, 2022, 2023, 2024)


@pytest.fixture
def four_window_raw_tier(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> Callable[..., DatasetConfig]:
    """Four calendar years -> exactly FOUR `year` windows.

    Four rather than two because the interesting cancel lands in the MIDDLE:
    with two windows, "cancelled after the second" and "ran to completion" are
    the same store and the same counts.
    """
    raw_dir = tmp_path / "raw4"
    rows = [
        stock_pqt_row(f"{year}-{day}", symbol, close=100.0)
        for year in _FOUR_YEARS
        for day in ("01-04", "06-15")
        for symbol in ("A", "B")
    ]
    hive_raw_tree(raw_dir, "tiingo", rows, batch_key="panel")

    def _build(store_name: str = "four.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(raw_dir / "tiingo"),
            zarr_file_path=str(tmp_path / store_name),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
        )

    return _build


def _dataset(config: DatasetConfig):
    from quantlab.dataset.stock import StockDataset

    return StockDataset(config)


def test_a_conversion_without_handles_behaves_exactly_as_before(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """The default call is unchanged: same store, same counts, `cancelled` False.

    The regression this guards is the one every "just add a keyword" change
    risks -- a new parameter that is only inert when someone remembers to keep
    it inert. Asserted against the SAME expectations the tracer at the top of
    this file asserts, so the two cannot drift.
    """
    config = tiingo_raw_tier(store_name="nohandles.zarr")

    dataset = _dataset(config)
    dataset.from_raw_data_chunked()
    result = dataset.last_chunk_result

    assert result is not None
    assert result.windows_written == _EXPECTED_WINDOWS
    assert result.windows_skipped == 0
    assert result.pinned_symbols == _EXPECTED_SYMBOLS
    assert result.cancelled is False
    stored = xr.open_zarr(config.zarr_file_path)
    assert stored.sizes["timestamp"] == len(_YEARS) * len(_DAYS_PER_YEAR)


def test_a_recording_reporter_sees_start_then_one_event_per_window_then_finish(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """The event stream is the console's whole view of a multi-hour run.

    Asserted as an ORDERED sequence rather than as a set: a reporter renders a
    bar, and a bar built from a finish event that arrived before its windows
    is not a bar. `total` carries the planned window count from the very first
    event, because a bar with no total renders a spinner.
    """
    reporter = _RecordingReporter()
    config = tiingo_raw_tier(store_name="reported.zarr")

    _dataset(config).from_raw_data_chunked(reporter=reporter)

    assert reporter.kinds() == (
        ["conversion_started"]
        + ["window_written"] * _EXPECTED_WINDOWS
        + ["conversion_finished"]
    )
    started = reporter.events[0]
    assert started.total == _EXPECTED_WINDOWS
    assert started.detail["pinned_symbols"] == _EXPECTED_SYMBOLS
    # `completed` rises 1..N over the written windows, against a fixed total.
    written = [e for e in reporter.events if e.kind == "window_written"]
    assert [e.completed for e in written] == list(
        range(1, _EXPECTED_WINDOWS + 1)
    )
    assert {e.total for e in written} == {_EXPECTED_WINDOWS}
    # `vendor` is REQUIRED on ProgressEvent and a conversion has no vendor in
    # the acquisition sense; the config's own vendor token is what it reuses.
    assert {e.vendor for e in reporter.events} == {"tiingo"}


def test_a_fully_resumed_run_emits_one_skipped_event_per_recorded_window(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """A resume is not silence. It is N skips and a finish.

    An operator restarting an interrupted backfill watches the bar to learn
    how much of it was already done; a resumed run that emitted nothing until
    the first NEW window would look hung for exactly as long as the work
    already finished.
    """
    _dataset(tiingo_raw_tier(store_name="resumed.zarr")).from_raw_data_chunked()

    reporter = _RecordingReporter()
    dataset = _dataset(tiingo_raw_tier(store_name="resumed.zarr"))
    dataset.from_raw_data_chunked(reporter=reporter)

    assert reporter.kinds() == (
        ["conversion_started"]
        + ["window_skipped"] * _EXPECTED_WINDOWS
        + ["conversion_finished"]
    )
    assert dataset.last_chunk_result is not None
    assert dataset.last_chunk_result.windows_written == 0
    assert dataset.last_chunk_result.resumed is True
    assert dataset.last_chunk_result.cancelled is False


def test_a_token_cancelled_before_the_first_window_writes_nothing(
    four_window_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """Cancel-before-start is a real state, not an edge case.

    The console's operator can hit stop while the pinned-axis scan is still
    running, so the first thing the loop does must be to ask. No window is
    materialised, NO STORE IS CREATED, and the result says `cancelled` rather
    than reporting a clean zero-window run -- which is what a caller would see
    if the flag were omitted and is indistinguishable from "there was nothing
    to do".
    """
    token = CancelToken()
    token.cancel()
    reporter = _RecordingReporter()
    config = four_window_raw_tier(store_name="precancelled.zarr")

    dataset = _dataset(config)
    dataset.from_raw_data_chunked(reporter=reporter, cancel=token)
    result = dataset.last_chunk_result

    assert result is not None
    assert result.cancelled is True
    assert result.windows_written == 0
    assert result.rows_written == 0
    assert result.windows_planned == len(_FOUR_YEARS)
    assert "cancelled" in reporter.kinds()
    assert "window_written" not in reporter.kinds()
    assert not Path(config.zarr_file_path).exists()


def test_a_cancel_after_two_of_four_windows_stops_there_and_resumes(
    four_window_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """The whole D-05 claim, end to end.

    Two windows land, the third never starts, and a later run finishes the
    remaining two and reports `resumed`. That last half is what makes
    cancellation cheap: `ChunkLedger` already guarantees every window recorded
    before the break stays recorded, which is the precondition acquisition had
    to retrofit with atomic sidecars and the chunk loop gets for free.

    Asserted on the STORE as well as on the counters -- counters are what a
    loop that checked the token in the wrong place would still get right while
    leaving the ledger and the store disagreeing.
    """
    token = CancelToken()
    reporter = _CancelAfterNWritten(token, after=2)

    first_config = four_window_raw_tier()
    first = _dataset(first_config)
    first.from_raw_data_chunked(reporter=reporter, cancel=token)
    first_result = first.last_chunk_result

    assert first_result is not None
    assert first_result.cancelled is True
    assert first_result.windows_written == 2
    assert first_result.windows_planned == len(_FOUR_YEARS)
    assert reporter.kinds().count("window_written") == 2
    assert reporter.kinds()[-2:] == ["cancelled", "conversion_finished"]
    # The store holds exactly the two windows' rows, and no partial third.
    stored = xr.open_zarr(first_config.zarr_file_path)
    assert stored.sizes["timestamp"] == 2 * 2  # two years, two days each

    second = _dataset(four_window_raw_tier())
    second.from_raw_data_chunked()
    second_result = second.last_chunk_result

    assert second_result is not None
    assert second_result.windows_skipped == 2
    assert second_result.windows_written == 2
    assert second_result.resumed is True
    assert second_result.cancelled is False
    stored = xr.open_zarr(first_config.zarr_file_path)
    assert stored.sizes["timestamp"] == len(_FOUR_YEARS) * 2


def test_a_reporter_that_raises_on_every_event_cannot_end_the_conversion(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """T-03.5-18: a UI bug must not cost a multi-hour conversion.

    The same never-raises contract `SymbolBatchAcquisition._emit` holds one layer up,
    reintroduced here because the callback now runs inside a SECOND loop this
    repository owns. `calls` is asserted non-zero so the test cannot pass
    against an implementation that simply stopped emitting.
    """
    reporter = _ExplodingReporter()
    config = tiingo_raw_tier(store_name="exploding.zarr")

    dataset = _dataset(config)
    dataset.from_raw_data_chunked(reporter=reporter)
    result = dataset.last_chunk_result

    assert reporter.calls > 0
    assert result is not None
    assert result.windows_written == _EXPECTED_WINDOWS
    assert result.cancelled is False
    stored = xr.open_zarr(config.zarr_file_path)
    assert stored.sizes["timestamp"] == len(_YEARS) * len(_DAYS_PER_YEAR)


def test_neither_handle_is_ever_written_onto_the_config(
    tiingo_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """T-03.5-21: both are CALL ARGUMENTS, and `to_dict()` lands on disk.

    `RawDatasetConfig.to_dict()` is `asdict(self)` and is serialised beside
    model checkpoints. A `threading.Event` cannot be serialised at all, and a
    live reporter object is not reproducible configuration -- a config that
    round-trips through JSON only when nobody watched the run is worse than
    one that never carries the handles.
    """
    import dataclasses as _dc

    field_names = {f.name for f in _dc.fields(DatasetConfig)}
    assert "reporter" not in field_names
    assert "cancel" not in field_names

    config = tiingo_raw_tier(store_name="configclean.zarr")
    _dataset(config).from_raw_data_chunked(
        reporter=_RecordingReporter(), cancel=CancelToken()
    )

    as_dict = config.to_dict()
    assert "reporter" not in as_dict
    assert "cancel" not in as_dict


def test_convert_forwards_both_handles_into_the_chunk_loop(
    four_window_raw_tier: Callable[..., DatasetConfig],
) -> None:
    """D-05's forwarding, proved BEHAVIOURALLY rather than by signature.

    A test asserting only that `convert()` declares `reporter` and `cancel`
    would pass against a function that accepts both and drops them on the
    floor, which is the exact failure this test exists for. So it asserts the
    two OBSERVABLE consequences: the token passed to `convert()` actually stops
    the conversion at a window boundary, and the reporter passed to `convert()`
    actually receives the chunk loop's own events -- the same stream, in the
    same order, that the direct `from_raw_data_chunked` call produces.
    """
    token = CancelToken()
    reporter = _CancelAfterNWritten(token, after=2)
    config = four_window_raw_tier(store_name="forwarded.zarr")

    result = convert(
        DataSourceRegistry.get("tiingo"),
        config,
        reporter=reporter,
        cancel=token,
    )

    assert result.cancelled is True
    assert result.windows_written == 2
    assert result.windows_planned == len(_FOUR_YEARS)
    assert reporter.kinds() == [
        "conversion_started",
        "window_written",
        "window_written",
        "cancelled",
        "conversion_finished",
    ]
    # ...and it stopped between two appends, so the two finished windows are
    # on disk and resumable rather than rolled back.
    stored = xr.open_zarr(config.zarr_file_path)
    assert stored.sizes["timestamp"] == 2 * 2

    resumed = convert(DataSourceRegistry.get("tiingo"), config)
    assert resumed.resumed is True
    assert resumed.windows_skipped == 2
    assert resumed.windows_written == 2
    assert resumed.cancelled is False
