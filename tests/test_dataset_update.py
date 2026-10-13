"""`BaseDataset.update()` and the evidence-driven strategy resolver
(260908-0f4 Task 2).

`update()` is the AUTOMATIC incremental entry point, symmetric with
`Factor.update()`. It takes no strategy parameter, because the widen-vs-rebuild
choice is a FACT about the raw tier rather than a caller's preference: a symbol
newly admitted to a widened roster may already carry vendor history over the
window the store covers, and a `widen` NaNs that history out with nothing to
report it. Measured on this repo's own data at planning time -- 56 symbols
present in raw and absent from the store, one of them (`VSTD`) carrying raw
rows across every timestamp the store has.

The three-way rule these tests pin:

- ANY added symbol carrying raw rows inside the store's extent -> `rebuild`;
- NO added symbol carrying such rows                           -> `widen`;
- ANY REMOVED symbol                                           -> `refuse`.

`tests/test_dataset_update_evidence.py` pins the probe itself. This module
pins what the resolver does with its answer -- including, in
`test_the_probe_is_asked_about_the_stores_extent_not_the_config_range`, WHICH
window it hands the probe, which is the one invariant no probe-layer test can
observe.

Everything here is offline: `tmp_path` stores, `tmp_path` raw trees.
"""

import inspect
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.base import BaseDataset
from quantlab.dataset.stock import StockDataset

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_STORE_YEARS = (2022, 2023, 2024)
_LATE_YEAR = 2025
_DAYS = ("01-04", "06-15", "12-28")

#: The store built by `_built_over_ab` spans exactly this. Both edges are
#: strictly INSIDE the config range below, so a resolver that handed the probe
#: the config range instead would produce a visibly different pair rather than
#: an accidentally equal one.
_STORE_FIRST = pd.Timestamp(f"{_STORE_YEARS[0]}-{_DAYS[0]}")
_STORE_LAST = pd.Timestamp(f"{_STORE_YEARS[-1]}-{_DAYS[-1]}")

_CONFIG_START = "2021-01-01"
_CONFIG_END = "2026-12-31"

#: `C` trades on every day the store already holds -- 3 years x 3 days.
_SPANNING_ROWS = len(_STORE_YEARS) * len(_DAYS)


class _GrowingRoster:
    """Staged raw tree. A panel over `{A, B}` for three years, then whichever
    of three differently-shaped roster changes a test needs:

    - `write_spanning_listing` -- `C`, trading on every timestamp the store
      ALREADY holds. This is the shape a widen destroys silently.
    - `write_late_listing` -- `D`, trading only in a year AFTER the store's
      last timestamp. A genuine new listing, where NaN is correct.
    - `write_removable` / `drop_removable` -- `X`, present when the store is
      built and gone from raw afterwards, which is the only way to reach the
      REMOVED branch from a raw tier.
    """

    def __init__(self, raw_dir: Path, tmp_path: Path, row, hive) -> None:
        self._raw_dir = raw_dir
        self._tmp_path = tmp_path
        self._row = row
        self._hive = hive

    def config(self, store_name: str = "growing.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(self._raw_dir / "tiingo"),
            zarr_file_path=str(self._tmp_path / store_name),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
            start_date=_CONFIG_START,
            end_date=_CONFIG_END,
        )

    def _write(self, symbol: str, close: float, years, batch_key: str) -> None:
        rows = [
            self._row(f"{year}-{day}", symbol, close=close)
            for year in years
            for day in _DAYS
        ]
        self._hive(self._raw_dir, "tiingo", rows, batch_key=batch_key)

    def write_initial(self) -> None:
        self._write("A", 100.0, _STORE_YEARS, "panel-a")
        self._write("B", 200.0, _STORE_YEARS, "panel-b")

    def write_spanning_listing(self) -> None:
        self._write("C", 300.0, _STORE_YEARS, "spanning")

    def write_late_listing(self, symbol: str = "D") -> None:
        self._write(symbol, 400.0, (_LATE_YEAR,), f"late-{symbol}")

    def write_removable(self) -> None:
        self._write("X", 500.0, _STORE_YEARS, "removable")

    def drop_removable(self) -> None:
        removed = list((self._raw_dir / "tiingo").rglob("part-removable-*.pqt"))
        assert removed, "the removable shards must exist before they are dropped"
        for path in removed:
            path.unlink()


@pytest.fixture
def growing_roster(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> _GrowingRoster:
    return _GrowingRoster(
        tmp_path / "growing", tmp_path, stock_pqt_row, hive_raw_tree
    )


class _SpyStock(StockDataset):
    """Records every densify window. The count is the ONLY observable that
    separates `widen` from `rebuild` for a genuinely new listing -- measured,
    the two produce NUMERICALLY IDENTICAL values there, so a value-only
    assertion would stay green under an "always rebuild" mutation.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.window_calls: list[tuple] = []
        super().__init__(config)

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: Optional[list[str]] = None
    ) -> xr.Dataset:
        window = super()._raw_data_to_xr_window(start_date, end_date, symbols)
        self.window_calls.append((pd.Timestamp(start_date), pd.Timestamp(end_date)))
        return window


def _panel(path: str) -> xr.Dataset:
    return xr.open_zarr(path).load()


def _captured_warnings():
    """loguru does not propagate to stdlib `logging`, so pytest's `caplog` sees
    nothing -- the same constraint `tests/test_chunked_ingest.py` records.
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    return messages, sink_id


def _built_over_ab(growing_roster: _GrowingRoster) -> DatasetConfig:
    """A complete chunked store over `{A, B}` for the three store years."""
    config = growing_roster.config()
    growing_roster.write_initial()
    StockDataset(config).from_raw_data_chunked(granularity="year")
    stored = _panel(config.zarr_file_path)
    assert stored["symbol"].values.tolist() == ["A", "B"]
    assert pd.Timestamp(stored["timestamp"].values[0]) == _STORE_FIRST
    assert pd.Timestamp(stored["timestamp"].values[-1]) == _STORE_LAST
    return config


# ---------------------------------------------------------------------------
# The three-way rule
# ---------------------------------------------------------------------------


def test_history_inside_the_stores_extent_resolves_to_rebuild(
    growing_roster: _GrowingRoster,
) -> None:
    """`C` already has vendor rows across every timestamp the store holds, so
    a widen would replace real data with NaN and nothing would say so. The
    resolver reads that off the raw tier and rebuilds instead.

    RED under: resolving to `widen` regardless of the probe's answer (M2) --
    `C` comes back all-NaN.
    """
    config = _built_over_ab(growing_roster)
    growing_roster.write_spanning_listing()

    StockDataset(config).update(granularity="year")

    rebuilt = _panel(config.zarr_file_path)
    assert rebuilt["symbol"].values.tolist() == ["A", "B", "C"]
    assert not np.isnan(rebuilt["adjClose"].sel(symbol="C").values).any()
    assert (rebuilt["adjClose"].sel(symbol="C").values == 300.0).all()


def test_a_genuine_new_listing_resolves_to_widen(
    growing_roster: _GrowingRoster,
) -> None:
    """`D` trades only AFTER the store's last timestamp, so it carries no
    history to lose and NaN is the correct value over the store's own extent.
    A rebuild here would be pure waste.

    Asserted by the DENSIFY-WINDOW COUNT, not by values. Measured: for a
    genuinely new listing `widen` and `rebuild` produce numerically identical
    values, so a value assertion would stay green under an "always rebuild"
    mutation. A widen keeps the store and appends only the one new window; a
    rebuild re-densifies all four.

    RED under: resolving to `rebuild` regardless of the probe's answer (M3),
    or handing the probe the config range instead of the store's extent (M4) --
    `D`'s later rows would then count as evidence. Both raise the count from
    1 to 4.
    """
    config = _built_over_ab(growing_roster)
    growing_roster.write_late_listing()

    dataset = _SpyStock(config)
    dataset.update(granularity="year")

    assert len(dataset.window_calls) == 1
    assert dataset.window_calls[0][0].year == _LATE_YEAR

    after = _panel(config.zarr_file_path)
    assert after["symbol"].values.tolist() == ["A", "B", "D"]
    # The values are what BOTH strategies produce -- asserted so the fixture's
    # measured claim stays visible, not as the discriminator.
    np.testing.assert_array_equal(
        after["adjClose"].sel(symbol="D").values[-len(_DAYS):],
        np.full(len(_DAYS), 400.0),
    )
    assert np.isnan(
        after["adjClose"].sel(symbol="D").values[: -len(_DAYS)]
    ).all()


def test_one_qualifying_symbol_rebuilds_the_whole_store(
    growing_roster: _GrowingRoster,
) -> None:
    """Rebuild is not a per-symbol operation. When ANY added symbol carries
    history in the store's extent the WHOLE store is re-densified -- including
    the windows that qualifying symbol never traded in, and including a window
    added by an unrelated, genuinely-new symbol.

    RED under: rebuilding only the windows the qualifying symbol touches, or
    resolving per-symbol and widening the rest.
    """
    config = _built_over_ab(growing_roster)
    growing_roster.write_spanning_listing()  # C -- qualifies
    growing_roster.write_late_listing("E")  # E -- does not

    dataset = _SpyStock(config)
    dataset.update(granularity="year")

    # Four windows: the three the store already held, plus E's later year.
    assert len(dataset.window_calls) == len(_STORE_YEARS) + 1

    rebuilt = _panel(config.zarr_file_path)
    assert rebuilt["symbol"].values.tolist() == ["A", "B", "C", "E"]
    # `C`'s REAL history, recovered from raw across the store's ORIGINAL
    # extent -- the nine timestamps a widen would have NaN'd out. It is NaN
    # over `E`'s later year only because it genuinely never traded there.
    recovered = rebuilt["adjClose"].sel(symbol="C").values[:_SPANNING_ROWS]
    assert not np.isnan(recovered).any()
    assert (recovered == 300.0).all()

    scratch = growing_roster.config("scratch.zarr")
    StockDataset(scratch).from_raw_data_chunked(granularity="year")
    xr.testing.assert_identical(rebuilt, _panel(scratch.zarr_file_path))


def test_a_removed_symbol_resolves_to_refuse_and_leaves_the_store_alone(
    growing_roster: _GrowingRoster,
) -> None:
    """Neither other strategy is SAFE when a symbol left the roster. `widen`
    cannot express a dropped label at all -- `XrBackend.widen_symbol_axis`
    refuses a target axis that is not a superset of the stored one -- and
    `rebuild` would silently discard that label's stored history. The
    automatic path is allowed to be conservative; it is not allowed to be
    destructive.

    RED under: dropping the removed branch and falling through to widening
    (M5), which raises the SUPERSET refusal instead of the roster one.
    """
    config = growing_roster.config()
    growing_roster.write_initial()
    growing_roster.write_removable()
    StockDataset(config).from_raw_data_chunked(granularity="year")
    assert _panel(config.zarr_file_path)["symbol"].values.tolist() == [
        "A",
        "B",
        "X",
    ]
    before = _panel(config.zarr_file_path)

    growing_roster.drop_removable()

    with pytest.raises(ValueError) as excinfo:
        StockDataset(config).update(granularity="year")

    assert "roster" in str(excinfo.value).lower()
    xr.testing.assert_identical(_panel(config.zarr_file_path), before)


# ---------------------------------------------------------------------------
# The decision is spoken
# ---------------------------------------------------------------------------


def test_the_rebuild_decision_is_reported_with_symbols_and_row_counts(
    growing_roster: _GrowingRoster,
) -> None:
    """A silent strategy switch is the same opacity as a wrong flag, in the
    other direction. Before the rebuild runs, the log names how many added
    symbols qualified and -- for each -- the symbol and how many raw rows it
    carries INSIDE the store's extent.

    RED under: deleting the report (M7), or reporting a bare count with no
    symbol names.
    """
    config = _built_over_ab(growing_roster)
    growing_roster.write_spanning_listing()

    messages, sink_id = _captured_warnings()
    try:
        StockDataset(config).update(granularity="year")
    finally:
        logger.remove(sink_id)

    blob = "\n".join(messages)
    assert f"C={_SPANNING_ROWS}" in blob
    assert "rebuild" in blob


# ---------------------------------------------------------------------------
# The interface, and what cannot reach it
# ---------------------------------------------------------------------------


def test_update_exposes_no_strategy_parameter() -> None:
    """The whole point of `update()` is that a caller CANNOT tell it whether
    to widen or rebuild -- that is the decision this entry point removes from
    callers. No `mode`, `force` or `overwrite` either: the append-dim overlap
    refusal is inherited unconditionally.

    RED under: adding an `on_new_listing` passthrough, which would make
    `update()` a second spelling of `from_raw_data_chunked` rather than a
    different contract.
    """
    parameters = inspect.signature(BaseDataset.update).parameters

    assert set(parameters) == {"self", "granularity", "ledger_path", "append_dim"}
    for forbidden in ("on_new_listing", "mode", "force", "overwrite"):
        assert forbidden not in parameters


def test_no_string_reaches_the_automatic_branch(
    growing_roster: _GrowingRoster,
) -> None:
    """The sentinel is a non-string object precisely so it is UNREACHABLE from
    a CLI flag, a config file or a JSON round-trip. Every plausible spelling
    an operator might reach for is refused with the published three.

    One function containing a loop rather than a parametrized test, so the
    suite's count stays deterministic.

    RED under: making the sentinel the string `"automatic"` (M6) -- that case
    would then resolve automatically instead of raising.
    """
    config = growing_roster.config("never-built.zarr")
    growing_roster.write_initial()

    for spelling in ("auto", "automatic", "evidence"):
        with pytest.raises(ValueError) as excinfo:
            StockDataset(config).from_raw_data_chunked(on_new_listing=spelling)

        message = str(excinfo.value)
        assert spelling in message
        for accepted in BaseDataset.NEW_LISTING_STRATEGIES:
            assert accepted in message

    assert not Path(config.zarr_file_path).exists()


def test_the_sentinel_is_absent_from_the_published_strategies(
    growing_roster: _GrowingRoster,
) -> None:
    """`NEW_LISTING_STRATEGIES` is the CLI's `choices` source -- adding the
    sentinel to it would put an object no operator can type onto the
    command-line surface and advertise it in every error message.

    RED under: adding the sentinel to the tuple (M9).
    """
    config = growing_roster.config("never-built.zarr")
    growing_roster.write_initial()

    assert BaseDataset.NEW_LISTING_STRATEGIES == ("refuse", "rebuild", "widen")
    assert BaseDataset._AUTOMATIC not in BaseDataset.NEW_LISTING_STRATEGIES
    assert not isinstance(BaseDataset._AUTOMATIC, str)

    with pytest.raises(ValueError) as excinfo:
        StockDataset(config).from_raw_data_chunked(on_new_listing="nope")

    message = str(excinfo.value)
    assert repr(BaseDataset._AUTOMATIC) not in message
    assert "_AUTOMATIC" not in message


# ---------------------------------------------------------------------------
# The common cases the resolver must not make expensive
# ---------------------------------------------------------------------------


def test_update_against_an_absent_store_is_an_ordinary_first_ingest(
    growing_roster: _GrowingRoster,
) -> None:
    """No store means no history to lose, so there is nothing to resolve and
    nothing to reconcile: `update()` is exactly a first chunked run.

    RED under: requiring a store, or resolving to `refuse` when the extent
    helper returns None.
    """
    config = growing_roster.config("fresh.zarr")
    growing_roster.write_initial()

    StockDataset(config).update(granularity="year")

    scratch = growing_roster.config("scratch.zarr")
    StockDataset(scratch).from_raw_data_chunked(granularity="year")
    xr.testing.assert_identical(
        _panel(config.zarr_file_path), _panel(scratch.zarr_file_path)
    )


def test_an_unchanged_roster_never_touches_the_raw_evidence_probe(
    growing_roster: _GrowingRoster,
) -> None:
    """The probe costs a raw scan, and it is only ever worth paying when the
    symbol axes actually drifted. `_reconcile_new_listings` already owns the
    ONE drift-detection site and falls through before the resolver exists, so
    an ordinary incremental refresh -- new DATES, same roster -- pays nothing.

    Asserted by making the probe RAISE: if it is reached at all, the run dies.

    RED under: hoisting the resolution into `update()`, which would probe on
    every run and compute the pinned axis twice.
    """
    config = _built_over_ab(growing_roster)
    # New dates for the SAME two symbols -- real incremental work, no drift.
    growing_roster._write("A", 100.0, (_LATE_YEAR,), "later-a")
    growing_roster._write("B", 200.0, (_LATE_YEAR,), "later-b")

    class _ProbeIsForbidden(StockDataset):
        def _added_symbols_with_raw_history(self, added, start, end) -> dict:
            raise AssertionError(
                "the evidence probe must not run when the roster is unchanged"
            )

    _ProbeIsForbidden(config).update(granularity="year")

    after = _panel(config.zarr_file_path)
    assert after["symbol"].values.tolist() == ["A", "B"]
    assert pd.Timestamp(after["timestamp"].values[-1]).year == _LATE_YEAR


def test_the_probe_is_asked_about_the_stores_extent_not_the_config_range(
    growing_roster: _GrowingRoster,
) -> None:
    """D-07, asserted STRUCTURALLY -- and this is the only test in the suite
    that can be. The probe takes its window from its caller, so no probe-layer
    test can observe which window the resolver chose; only here is the choice
    visible.

    The config range is strictly wider than the store's extent on BOTH sides,
    so a resolver handing over the config range produces a visibly different
    pair rather than an accidentally equal one. Using the config range would
    make every genuinely-new listing look like evidence and trigger a needless
    whole-store rebuild -- measured on this repo's real store, those two
    windows differ by eleven months.

    RED under: passing the config's date range to the probe (M4).
    """
    config = _built_over_ab(growing_roster)
    growing_roster.write_spanning_listing()

    extent_before = BaseDataset._stored_append_extent(config.zarr_file_path)
    assert extent_before is not None
    # The premise: the config range genuinely brackets the store's extent.
    assert pd.Timestamp(_CONFIG_START) < pd.Timestamp(extent_before[0])
    assert pd.Timestamp(extent_before[1]) < pd.Timestamp(_CONFIG_END)

    windows: list[tuple] = []

    class _RecordsTheProbeWindow(StockDataset):
        def _added_symbols_with_raw_history(self, added, start, end) -> dict:
            windows.append((start, end))
            return super()._added_symbols_with_raw_history(added, start, end)

    _RecordsTheProbeWindow(config).update(granularity="year")

    assert len(windows) == 1
    asked_start, asked_end = windows[0]
    assert pd.Timestamp(asked_start) == pd.Timestamp(extent_before[0])
    assert pd.Timestamp(asked_end) == pd.Timestamp(extent_before[1])


# ---------------------------------------------------------------------------
# The pinned axis is the stored axis followed by the new symbols (#256, ADR 0032)
# ---------------------------------------------------------------------------


def _chunk_files(path: str) -> dict:
    """Every stored chunk file of the data variables: bytes, inode, mtime."""
    root = Path(path)
    files = {}
    for chunk in root.rglob("*"):
        relative = chunk.relative_to(root)
        if (
            chunk.is_file()
            and relative.parts[0] not in ("symbol", "timestamp")
            and chunk.name not in ("zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata")
        ):
            stat = chunk.stat()
            files[str(relative)] = (chunk.read_bytes(), stat.st_ino, stat.st_mtime_ns)
    return files


def test_a_new_listing_is_appended_at_the_end_without_rewriting_the_store(
    growing_roster: _GrowingRoster, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`AAA` sorts between `A` and `B` but is appended after them.

    RED under: pinning the sorted raw axis (the store is then rewritten onto
    `A, AAA, B` by `widen_symbol_axis`).
    """
    from quantlab.backend.zarr import XrBackend

    monkeypatch.setattr(XrBackend, "APPEND_DIM_CHUNK", 3)
    config = _built_over_ab(growing_roster)
    before = _chunk_files(config.zarr_file_path)
    growing_roster.write_late_listing("AAA")

    def refuse(*args, **kwargs):
        raise AssertionError("the store was rewritten")

    monkeypatch.setattr(XrBackend, "widen_symbol_axis", refuse)
    StockDataset(config).update(granularity="year")

    after = _panel(config.zarr_file_path)
    assert after["symbol"].values.tolist() == ["A", "B", "AAA"]
    assert np.isnan(after["adjClose"].sel(symbol="AAA").values[: -len(_DAYS)]).all()
    assert (after["adjClose"].sel(symbol="AAA").values[-len(_DAYS):] == 400.0).all()
    assert (after["adjClose"].sel(symbol="B").values[: -len(_DAYS)] == 200.0).all()
    files = _chunk_files(config.zarr_file_path)
    for name, content in before.items():
        assert files[name] == content, name


def test_an_update_after_an_extension_neither_widens_nor_refuses(
    growing_roster: _GrowingRoster, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ledger was rebased onto the extended axis, so the next run agrees.

    RED under: re-pinning the sorted raw axis on the next run (refused as a
    changed axis) or not rebasing the ledger.
    """
    from quantlab.backend.zarr import XrBackend

    config = _built_over_ab(growing_roster)
    growing_roster.write_late_listing("AAA")
    StockDataset(config).update(granularity="year")
    extended = _panel(config.zarr_file_path)
    before = _chunk_files(config.zarr_file_path)

    def refuse(*args, **kwargs):
        raise AssertionError("the axis was touched")

    monkeypatch.setattr(XrBackend, "widen_symbol_axis", refuse)
    monkeypatch.setattr(XrBackend, "extend_symbol_axis", refuse)
    dataset = _SpyStock(config)
    dataset.update(granularity="year")
    StockDataset(config).from_raw_data_chunked(granularity="year")

    assert dataset.window_calls == []
    assert _chunk_files(config.zarr_file_path) == before
    xr.testing.assert_identical(_panel(config.zarr_file_path), extended)


def test_a_rebuild_writes_a_sorted_axis(growing_roster: _GrowingRoster) -> None:
    """A listing with history inside the store's extent rebuilds, sorted.

    RED under: rebuilding on the stored-plus-new axis.
    """
    config = _built_over_ab(growing_roster)
    growing_roster._write("AA", 300.0, _STORE_YEARS, "spanning-aa")

    StockDataset(config).update(granularity="year")

    rebuilt = _panel(config.zarr_file_path)
    assert rebuilt["symbol"].values.tolist() == ["A", "AA", "B"]
    assert (rebuilt["adjClose"].sel(symbol="AA").values == 300.0).all()
