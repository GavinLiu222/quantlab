"""`{zarr}.crsp_tickers.json`: the write side, and `CrspTickerLookup` reading it.

The panel's `symbol` axis is the int64 PERMNO (D-01). A PERMNO is correct for a
machine and illegible for a human, and the fix D-03 chose is a SIDECAR rather
than a column: a 2-D `ticker(timestamp, symbol)` string variable is refused by
the backend's symbol-dim dtype guards, and a 1-D `ticker(symbol)` coord can only
hold ONE name per PERMNO -- which silently discards FB -> META, the exact case
this file pins twice.

So the sidecar is an INTERVAL TABLE, not a `{PERMNO: ticker}` map. That
distinction is the whole design: any last-name-wins mapping keeps one name per
PERMNO and is deliberately NOT the shape copied here.

**Every quantlab import is INSIDE a test or helper body**, following
`tests/test_crsp_dataset.py`: these tests are written before the names they
assert on exist, and a module-scope import would turn the RED run into a
collection error -- zero tests discovered, which proves nothing (TDD gate
#3770). The rule is about QUANTLAB names: `loguru` is third-party, already
installed, and cannot be the name a RED run is waiting for, so the
`warning_messages` fixture imports it at module scope like every other test
file that captures a warning.

**Provenance.** The FB -> META rows are VERBATIM `03.10-LIVE-CHECK.json` key
`C5_ticker_hist_crsp_a_stock.stksecurityinfohist`, reached through
`tests/crsp_fixtures.py:SECINFO_ROWS`; the same source
`tests/test_crsp_symbology.py` uses. Daily rows invented for a scenario carry a
`# SYNTHETIC` comment.
"""

from __future__ import annotations

from datetime import date

import pytest
from loguru import logger

from tests.test_crsp_dataset import (
    SYNTHETIC_PERMNO,
    _dataset_config,
    _pull,
    _synthetic_secinfo,
)

#: The three PERMNOs the converted panel below carries. 13407 is the rename
#: case (two intervals), 14593 is the multi-interval single-name case, and
#: 10107 is the synthetic one-interval case.
META_PERMNO = "13407"
AAPL_PERMNO = "14593"

PANEL_PERMNOS = (META_PERMNO, AAPL_PERMNO, SYNTHETIC_PERMNO)

#: A window straddling the FB -> META boundary (2022-06-09) so the panel's own
#: dates cross the rename. The sidecar's intervals come from the reference
#: tier and span the whole security history regardless.
WINDOW_START = "2022-06-01"
WINDOW_END = "2022-06-30"

#: Every trading day in that window, for every panel PERMNO. # SYNTHETIC --
#: the live check never sampled daily rows for these three securities in 2022.
_TRADING_DAYS = (
    "2022-06-01", "2022-06-02", "2022-06-03", "2022-06-06", "2022-06-07",
    "2022-06-08", "2022-06-09", "2022-06-10", "2022-06-13", "2022-06-14",
)


def _daily_rows():
    """Flat, priced, unsplit days -- everything `dsf_row` does not default.

    `dlyprc` is what makes the adjustment anchor usable (a PERMNO with no
    strictly-positive close inside the window is refused outright), and this
    file is about NAMES, so the prices are deliberately the least interesting
    ones available.
    """
    from tests.crsp_fixtures import dsf_row

    return [
        dsf_row(
            int(permno),
            day,
            dlyprc="100.000000",  # SYNTHETIC
            dlyret="0.001000",  # SYNTHETIC
            dlyretx="0.001000",  # SYNTHETIC
        )
        for permno in PANEL_PERMNOS
        for day in _TRADING_DAYS
    ]


@pytest.fixture
def converted(mock_crsp_session, tmp_path):
    """A converted CRSP store carrying exactly `PANEL_PERMNOS`.

    Returns the `CrspDatasetConfig`; the store, the filter report and the
    ticker sidecar are all on disk beside it.
    """
    from quantlab.dataset.crsp import CrspStockDataset

    cfg, reference_dir = _pull(
        tmp_path,
        _daily_rows(),
        list(PANEL_PERMNOS),
        start=WINDOW_START,
        end=WINDOW_END,
        extra_secinfo=_synthetic_secinfo(),
    )
    dataset_config = _dataset_config(
        tmp_path, cfg, reference_dir, start=WINDOW_START, end=WINDOW_END
    )
    CrspStockDataset(dataset_config).from_raw_data().save()
    return dataset_config


def _payload(dataset_config) -> dict:
    import json

    from quantlab.dataset.crsp import CrspStockDataset

    path = CrspStockDataset(dataset_config).ticker_sidecar_path()
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Task 1 -- the write side
# ---------------------------------------------------------------------------


def test_the_suffix_and_the_path_follow_the_filter_report_s_shape():
    """`.crsp_tickers.json`, a SIBLING of the store -- the same one-line
    `Path(str(zarr_file_path) + SUFFIX)` the filter report uses."""
    from quantlab.dataset.config import CrspDatasetConfig
    from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX, CrspStockDataset

    assert TICKER_SIDECAR_SUFFIX == ".crsp_tickers.json"

    config = CrspDatasetConfig(
        zarr_file_path="/tmp/does-not-exist/crsp.zarr",
        raw_data_dir_path="/tmp/does-not-exist",
        reference_dir="/tmp/does-not-exist/reference",
        start_date="2022-06-01",
        end_date="2022-06-30",
    )
    path = CrspStockDataset(config).ticker_sidecar_path()
    assert str(path) == "/tmp/does-not-exist/crsp.zarr.crsp_tickers.json"


def test_a_conversion_writes_the_three_top_level_keys(converted):
    """`generated_from` / `vintage_product_end` / `intervals`.

    The vintage rides along for the same reason the adjustment anchor records
    it: the SAME PERMNO read against a newer CRSP vintage can carry a later
    interval, so "which names" is only answerable together with "as of which
    vintage".
    """
    payload = _payload(converted)

    assert sorted(payload) == [
        "generated_from",
        "intervals",
        "vintage_product_end",
    ]
    assert payload["generated_from"] == "stksecurityinfohist"
    assert payload["vintage_product_end"] == "2025-12-31"


def test_permno_13407_is_two_intervals_fb_then_meta(converted):
    """VERBATIM C5, and the reason this sidecar is not a `{PERMNO: ticker}` map.

    One PERMNO, two names, a hard boundary. A shape that kept only the LAST
    ticker per PERMNO would answer "META" for 2012, which is the same defect
    D-03 rejected a 1-D `ticker(symbol)` coord for.
    """
    intervals = _payload(converted)["intervals"][META_PERMNO]

    assert len(intervals) == 2
    assert intervals[0] == {
        "ticker": "FB",
        "start": "2012-05-18",
        "end": "2022-06-08",
    }
    assert intervals[1] == {
        "ticker": "META",
        "start": "2022-06-09",
        "end": "2025-12-31",
    }


def test_intervals_are_keyed_by_permno_string_and_sorted_by_start(converted):
    """JSON object keys can only be strings, so `str(permno)` -- the same
    spelling `_permno_breakdown` uses. Within a PERMNO, `start` ascends."""
    intervals = _payload(converted)["intervals"]

    assert all(isinstance(key, str) for key in intervals)
    assert set(intervals) == {str(int(p)) for p in PANEL_PERMNOS}
    for spans in intervals.values():
        starts = [span["start"] for span in spans]
        assert starts == sorted(starts)
        assert all(span["start"] <= span["end"] for span in spans)


def test_only_the_panel_s_permnos_are_written(converted):
    """Not the whole reference tier.

    `SECINFO_ROWS` carries BRK (83443), GOOGL (90319), Lehman (80599) and
    WestRock (21186) as well, and the live table has 40,518 PERMNOs. A sidecar
    that named every security CRSP has ever issued would be megabytes of names
    for a panel holding three.
    """
    intervals = _payload(converted)["intervals"]

    assert len(intervals) == 3
    assert "83443" not in intervals
    assert "90319" not in intervals


def test_the_sidecar_is_written_indented_and_key_sorted(converted):
    """`indent=2, sort_keys=True`, passed EXPLICITLY.

    `quantlab/utils/atomic.py` forwards `**json_kwargs` verbatim so every
    caller keeps its own formatting; relying on a default here would make the
    format a property of the writer rather than of this sidecar.
    """
    from quantlab.dataset.crsp import CrspStockDataset

    text = CrspStockDataset(converted).ticker_sidecar_path().read_text(
        encoding="utf-8"
    )

    assert text.startswith('{\n  "generated_from"')
    assert '\n  "intervals": {' in text
    assert '\n  "vintage_product_end"' in text


def test_an_existing_store_blocks_the_write(tmp_path):
    """WR-03: `_write_identity_reports` returns before writing anything when
    the store already exists.

    The cost is an append no longer refreshing the sidecar, and it is the
    lesser harm -- without the guard a REFUSED re-conversion would overwrite
    the surviving store's audit files with numbers for a panel that was never
    written. A rebuild deletes the store and its `.crsp_*.json` siblings first;
    `quantlab/acquisition/wrds/rebuild.py:CrspStoreRebuilder` is what does that.
    """
    from quantlab.dataset.config import CrspDatasetConfig
    from quantlab.dataset.crsp import CrspStockDataset

    store = tmp_path / "crsp.zarr"
    store.mkdir()
    config = CrspDatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(tmp_path),
        reference_dir=str(tmp_path / "reference"),
        start_date="2022-06-01",
        end_date="2022-06-30",
    )
    dataset = CrspStockDataset(config)
    dataset._ticker_intervals = {
        "generated_from": "stksecurityinfohist",
        "vintage_product_end": "2025-12-31",
        "intervals": {"13407": [{"ticker": "META", "start": "2022-06-09",
                                 "end": "2025-12-31"}]},
    }

    dataset._write_identity_reports()

    assert not dataset.ticker_sidecar_path().exists()


# ---------------------------------------------------------------------------
# Task 2 -- the read side
# ---------------------------------------------------------------------------


def _lookup(converted):
    """The lookup the CRSP dataset names, so the read-side tests go through it."""
    from quantlab.dataset.crsp import CrspStockDataset

    return CrspStockDataset(converted).ticker_lookup()


def test_a_crsp_dataset_names_a_ticker_lookup_over_its_sidecar(converted):
    """#226: the dataset tells a caller which lookup applies, as the interface."""
    from quantlab.dataset.base import TickerLookup
    from quantlab.dataset.crsp import CrspStockDataset

    dataset = CrspStockDataset(converted)
    lookup = dataset.ticker_lookup()

    assert isinstance(lookup, TickerLookup)
    assert lookup.sidecar_path == dataset.ticker_sidecar_path()


def test_names_gives_ticker_and_company_and_falls_back_to_the_id(converted):
    """#226: `names` answers ticker and company as of a day; the CRSP sidecar
    records no company, and an unknown PERMNO is named by its own digits."""
    from quantlab.dataset.base import SymbolName

    lookup = _lookup(converted)

    assert lookup.names([13407, 99999], date(2022, 6, 8)) == [
        SymbolName(ticker="FB", company=None),
        SymbolName(ticker="99999", company=None),
    ]
    assert lookup.names([13407], date(2022, 6, 9)) == [SymbolName("META", None)]


def test_a_dataset_without_a_ticker_sidecar_names_no_lookup(tmp_path):
    """#226: a store-backed dataset of another vendor and one held in memory
    both name no lookup, so their symbols are shown as they are."""
    import pandas as pd

    from quantlab.dataset.config import DatasetConfig
    from quantlab.dataset.memory import FrameDataset
    from quantlab.dataset.stock import StockDataset

    stock = StockDataset(DatasetConfig(
        zarr_file_path=str(tmp_path / "us_all.zarr"),
        raw_data_dir_path=str(tmp_path / "raw"),
        market="us_equity", frequency="1d", vendor="tiingo",
    ))
    frame = FrameDataset(pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]), "symbol": ["AAA"],
        "close": [10.0]}))

    assert stock.ticker_lookup() is None
    assert frame.ticker_lookup() is None


def test_as_of_answers_fb_and_meta_on_either_side_of_the_rename(converted):
    """The sidecar's whole reason for existing: a PERMNO's PERIOD-CORRECT name.

    2022-06-08 is FB and 2022-06-09 is META for the same 13407, so an audit
    record dated inside the FB era cannot be labelled META.
    """
    lookup = _lookup(converted)

    assert lookup.as_of(13407, date(2022, 6, 8)) == "FB"
    assert lookup.as_of(13407, date(2022, 6, 9)) == "META"


def test_as_of_answers_aapl_for_a_single_name_permno(converted):
    lookup = _lookup(converted)

    assert lookup.as_of(14593, date(2020, 1, 1)) == "AAPL"


def test_as_of_returns_none_when_no_interval_covers_the_day(converted):
    """A day before the security existed is an ABSENCE, not an error: the
    caller is a display layer and has a perfectly good fallback."""
    lookup = _lookup(converted)

    assert lookup.as_of(13407, date(1990, 1, 1)) is None


def test_as_of_returns_none_for_a_permno_the_sidecar_never_heard_of(converted):
    lookup = _lookup(converted)

    assert lookup.as_of(99999, date(2022, 6, 9)) is None


def test_label_is_positional_and_falls_back_to_the_permno(converted):
    """`label` returns one string per input, in order, and spells an unknown
    PERMNO as its own digits -- a display layer wants a readable line, not a
    `KeyError` in the middle of a log message."""
    lookup = _lookup(converted)

    assert lookup.label([13407, 14593, 99999], date(2022, 6, 9)) == [
        "META",
        "AAPL",
        "99999",
    ]


def test_a_missing_sidecar_is_raised_on_first_query_not_on_construction(
    tmp_path,
):
    """Lazy, like `CrspReference.manifest`: constructing the lookup touches no
    disk, and the refusal names the class, the path and what to do about it."""
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    missing = tmp_path / "crsp.zarr.crsp_tickers.json"
    lookup = CrspTickerLookup(missing)  # must not raise

    with pytest.raises(FileNotFoundError) as excinfo:
        lookup.as_of(13407, date(2022, 6, 9))

    message = str(excinfo.value)
    assert "CrspTickerLookup:" in message
    assert str(missing) in message
    assert ".crsp_*.json" in message


def test_a_corrupt_sidecar_names_the_exception_type_and_the_way_out(tmp_path):
    """This repo's shaped-error convention for an unreadable sidecar:
    `type(exc).__name__`, what the file records, and the rebuild route."""
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    corrupt = tmp_path / "crsp.zarr.crsp_tickers.json"
    corrupt.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        CrspTickerLookup(corrupt).as_of(13407, date(2022, 6, 9))

    message = str(excinfo.value)
    assert "JSONDecodeError" in message
    assert ".crsp_*.json" in message


def test_the_sidecar_is_read_once_per_instance(converted, monkeypatch):
    """The `None`-sentinel cache `CrspSymbology._intervals` and
    `CrspReference.manifest` both use: repeated display lookups must not
    re-read and re-parse the file once per logged line."""
    from pathlib import Path

    lookup = _lookup(converted)
    reads: list[str] = []
    original = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        reads.append(str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    lookup.as_of(13407, date(2022, 6, 9))
    lookup.as_of(14593, date(2022, 6, 9))
    lookup.label([13407, 14593], date(2022, 6, 9))

    assert len(reads) == 1, reads


def test_label_falls_back_without_raising_when_the_sidecar_is_absent(tmp_path):
    """The display contract (T-03.11-30): the call sites that reach `label()`
    may NOT break because an audit file is missing.

    They are `quantlab/backtest/engine_vectorbt.py:303` and `quantlab/model/base.py`'s
    `_spell` (entered from both the `missing` and the `extra` branch of
    `predict_panel`). `browse_zarr`'s refusal and the `--symbols` CLI help name
    this class in prose without ever calling it.

    `as_of` still raises -- it is the strict, single-value question. `label` is
    the display entry point and answers with the digits.
    """
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    lookup = CrspTickerLookup(tmp_path / "absent.crsp_tickers.json")

    assert lookup.label([13407, 99999], date(2022, 6, 9)) == ["13407", "99999"]


def test_label_falls_back_without_raising_when_the_sidecar_is_corrupt(tmp_path):
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    corrupt = tmp_path / "crsp.zarr.crsp_tickers.json"
    corrupt.write_text("{not json", encoding="utf-8")

    assert CrspTickerLookup(corrupt).label([13407], date(2022, 6, 9)) == ["13407"]


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text.

    VERBATIM from `tests/test_model_predict_panel.py` -- this repo has one way
    of capturing a `loguru` warning in a test, and a second spelling of it
    would be a fixture to keep in sync for no benefit.
    """
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    yield messages
    logger.remove(handler_id)


#: A sidecar that is entirely well-formed -- the control for the tests that
#: assert NOTHING was warned. Hand-written rather than taken from `converted`
#: because those tests are about the absence of a log line, and a full CRSP
#: conversion would be a minute of fixture work to prove it.
VALID_SIDECAR = (
    '{"generated_from": "stksecurityinfohist", '
    '"vintage_product_end": "2025-12-31", '
    '"intervals": {"13407": [{"ticker": "FB", "start": "2012-05-18", '
    '"end": "2022-06-08"}]}}'
)

#: One PERMNO's spans are fine and the other's is missing `start`.
#:
#: The point is that damage is not all-or-nothing: `intervals` reads cleanly,
#: 13407 answers, and only 14593 falls into the per-PERMNO `except`. A guard
#: that degraded the whole CALL on the first bad span would turn one unreadable
#: security into a line of digits for every security beside it.
PARTLY_BROKEN_SIDECAR = (
    '{"intervals": {'
    '"13407": [{"ticker": "FB", "start": "2012-05-18", "end": "2022-06-08"}], '
    '"14593": [{"ticker": "AAPL", "end": "2025-12-31"}]}}'
)


#: Sidecars that PARSE as JSON and are still unusable -- the half of "corrupt"
#: the contract claimed to cover and did not (G-03.11-3). All three were
#: reproduced by hand against the shipped 03.11-09 code.
#:
#: They travel TWO DIFFERENT paths, which is the whole reason the guard has to
#: be in two places rather than one:
#:
#: | payload | reading `intervals` | the per-PERMNO `as_of` call |
#: |---|---|---|
#: | `{"intervals": [1, 2, 3]}` | breaks here (`list.get`) | never reached |
#: | `[]` | breaks here (`list.get`) | never reached |
#: | `{"intervals": {"13407": [{"ticker": "FB"}]}}` | PASSES, returns a non-empty dict | breaks here (`span["start"]`) |
#:
#: A guard on the first step alone leaves the third one crashing, which is what
#: `label()`'s single `try` around the intervals read used to do. Do not delete
#: either half of the guard thinking the other one already covers it.
#:
#: THREE ENTRY POINTS, the same three inputs, three deliberately different
#: postures. This second table is the one to read before "unifying" anything:
#:
#: | payload | `label` | `as_of` | `product_end` |
#: |---|---|---|---|
#: | `{"intervals": [1, 2, 3]}` | digits, plus ONE warning | shaped refusal (`intervals` is not an object) | `None` |
#: | `[]` | digits, plus ONE warning | shaped refusal (top level is not an object) | shaped refusal (the SAME check, via `_object_payload()`) |
#: | `{"intervals": {"13407": [{"ticker": "FB"}]}}` | digits, plus ONE warning | shaped refusal (the span lacks `start`) | `None` |
#:
#: The `None`s are not gaps. `product_end` shares the TOP-LEVEL check and
#: nothing below it: a sidecar whose interval table is wrong can still say
#: truthfully which CRSP vintage it was read against, so refusing there would
#: be `product_end` inventing a second shape check of its own -- the thing
#: G-03.11-6 / WR-04 was about. What it must never do again is answer with a
#: BARE exception, and `[]` is the row that used to.
#:
#: The warning column is the other half of G-03.11-6: `label` degrading is not
#: silent any more, and says so exactly once per lookup instance -- the digits
#: alone are also what a store with NO sidecar prints, and a console that
#: cannot tell the two apart never gets the broken one rebuilt.
MALFORMED_SIDECARS = [
    pytest.param('{"intervals": [1, 2, 3]}', id="intervals-is-a-list"),
    pytest.param(
        '{"intervals": {"13407": [{"ticker": "FB"}]}}', id="span-has-no-start"
    ),
    pytest.param("[]", id="payload-is-a-list"),
]


def _written(tmp_path, text):
    """A lookup over a sidecar whose exact bytes the test chose, and its path."""
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    path = tmp_path / "crsp.zarr.crsp_tickers.json"
    path.write_text(text, encoding="utf-8")
    return CrspTickerLookup(path), path


@pytest.mark.parametrize("text", MALFORMED_SIDECARS)
def test_label_falls_back_when_the_sidecar_parses_but_is_shaped_wrong(
    tmp_path, text, warning_messages
):
    """The display contract says "missing OR CORRUPT", and corrupt includes
    "parsed fine, shaped wrong" -- not just "not JSON".

    The worst caller is `base/model.py:1281`: a BARE `_spell` on the happy path,
    where a panel carrying untrained symbols is merely dropped with a warning
    and the prediction completes. A malformed audit sidecar used to turn that
    successful `predict_panel` into a crash.

    The fall-back is never raised AND never silent (G-03.11-6 / WR-03): the
    digits alone are also the supported output of a store with no sidecar at
    all, so one WARNING naming the class and the original exception is what
    tells the two apart on a console.
    """
    lookup, _ = _written(tmp_path, text)

    assert lookup.label([13407], date(2022, 6, 9)) == ["13407"]
    assert len(warning_messages) == 1, warning_messages
    assert "CrspTickerLookup" in warning_messages[0]


@pytest.mark.parametrize("text", MALFORMED_SIDECARS)
def test_as_of_refuses_a_structurally_broken_sidecar_with_a_shaped_error(
    tmp_path, text
):
    """The strict entry point stays strict -- but its refusal is READABLE.

    A bare `AttributeError: 'list' object has no attribute 'get'` names neither
    the file nor the way out. Structural damage must NOT be rounded down to
    `None` here either: "this sidecar cannot be read" and "that PERMNO had no
    name that day" are different answers and `as_of` is the question that cares.
    """
    lookup, path = _written(tmp_path, text)

    with pytest.raises(ValueError) as excinfo:
        lookup.as_of(13407, date(2022, 6, 9))

    message = str(excinfo.value)
    assert "CrspTickerLookup:" in message
    assert str(path) in message
    assert ".crsp_*.json" in message


def test_a_payload_with_no_intervals_key_keeps_its_current_behaviour(tmp_path):
    """An ABSENT `intervals` key is not damage: `.get("intervals", {})` has
    always answered "this sidecar knows no names" and both entry points already
    have a good answer for that. Tightening it into a refusal would break a
    sidecar written for a roster this store does not carry.

    `{}` is one of THREE spellings of "empty" this file keeps apart, and the
    third entry point has to stay legible on all of them: `product_end` is
    `None` here because "this sidecar records no vintage" is an answer, not
    damage. `[]` (top level is not an object) and a zero-byte file (never
    parsed) are the other two, and each gets a shaped refusal instead -- see
    the two `product_end` parametrisations below.
    """
    lookup, _ = _written(tmp_path, "{}")

    assert lookup.as_of(13407, date(2022, 6, 9)) is None
    assert lookup.label([13407], date(2022, 6, 9)) == ["13407"]
    assert lookup.product_end is None


#: Sidecars that never get as far as a shape at all -- the OTHER half of
#: "corrupt", and a DIFFERENT path from `MALFORMED_SIDECARS` above.
#:
#: The dividing line between the two lists is one question: **did `json.loads`
#: return?** If it returned and handed back something that is not shaped like a
#: sidecar, the damage is structural and is caught downstream by `_intervals()`
#: or by the per-PERMNO `as_of` -- that is `MALFORMED_SIDECARS`. If it never
#: returned, the failure happened in the `payload` property, strictly upstream
#: of every structural guard, and none of those guards is even reached. Keeping
#: the two lists apart is what keeps `MALFORMED_SIDECARS`' path table honest;
#: merging them would make that table describe rows it does not cover.
#:
#: Bytes, not `str`, because one of the three is not decodable text.
#:
#: | payload | what `payload` sees |
#: |---|---|
#: | 20,000 nested `[` | `RecursionError` out of `json.loads` (G-03.11-3 / WR-01) |
#: | an isolated UTF-8 continuation byte | `UnicodeDecodeError` out of `read_text` (a `ValueError`) |
#: | zero bytes | `JSONDecodeError` out of `json.loads` (a `ValueError`) |
#:
#: The zero-byte row is deliberately NOT the same case as `{}` in
#: `test_a_payload_with_no_intervals_key_keeps_its_current_behaviour` above:
#: `{}` is a sidecar that parsed and knows no names, an empty FILE is a sidecar
#: that could not be read at all, and the two entry points answer them
#: differently on the `as_of` side. Both spellings of "empty" are pinned so the
#: difference stays visible.
UNPARSEABLE_SIDECARS = [
    pytest.param(b"[" * 20000 + b"]" * 20000, id="nested-past-the-parser"),
    pytest.param(b"\x80\x81\x82", id="not-valid-utf-8"),
    pytest.param(b"", id="zero-bytes"),
]


def _written_bytes(tmp_path, payload: bytes):
    """A lookup over a sidecar whose exact BYTES the test chose, and its path.

    The `bytes` twin of `_written`: `write_text` cannot express a file that is
    not decodable as UTF-8, and that is one of the three cases here.
    """
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    path = tmp_path / "crsp.zarr.crsp_tickers.json"
    path.write_bytes(payload)
    return CrspTickerLookup(path), path


@pytest.mark.parametrize("payload", UNPARSEABLE_SIDECARS)
def test_label_falls_back_when_the_sidecar_never_parses(tmp_path, payload):
    """The display contract holds for the parse stage too, not just for shapes.

    `RecursionError` is the one that was escaping (G-03.11-3 / WR-01): it is a
    `RuntimeError` subclass, so neither the `payload` property's original
    `(OSError, ValueError)` nor `label()`'s tuple caught it, and it walked out
    of the bare call sites -- `backtest/engine_vectorbt.py:303` mid-simulation
    and `base/model.py:1315`
    via `_spell` on the happy path.
    """
    lookup, _ = _written_bytes(tmp_path, payload)

    assert lookup.label([13407], date(2020, 1, 1)) == ["13407"]


@pytest.mark.parametrize("payload", UNPARSEABLE_SIDECARS)
def test_as_of_refuses_an_unparseable_sidecar_with_a_shaped_error(
    tmp_path, payload
):
    """Strict stays strict, and the refusal is SHAPED -- the module docstring's
    "each refusal names the class, the path and the rebuild" has to hold for
    the parse stage as well, or it is simply false.

    A bare `RecursionError: maximum recursion depth exceeded while decoding a
    JSON array` names neither the file nor the way out; the same three
    assertions the structural refusal already carries are what make it an
    answer a reader can act on.
    """
    lookup, path = _written_bytes(tmp_path, payload)

    with pytest.raises(ValueError) as excinfo:
        lookup.as_of(13407, date(2020, 1, 1))

    message = str(excinfo.value)
    assert "CrspTickerLookup:" in message
    assert str(path) in message
    assert ".crsp_*.json" in message


def test_a_bug_inside_as_of_reaches_the_caller_instead_of_becoming_digits(
    converted,
):
    """G-03.11-3 / WR-02: the guard must not swallow THIS module's own bugs.

    `label()`'s docstring has always said `except Exception` is deliberately
    avoided "so a genuine programming bug in this module still reaches the
    caller". Once 03.11-12's structural guards landed, `KeyError` /
    `AttributeError` / `TypeError` could no longer come from data damage at all
    -- every structural defect funnels through `_malformed` (a `ValueError`) or
    the `payload` property (`FileNotFoundError` / `ValueError`) -- so the only
    thing those three could still catch was the bug the rationale says they
    exist to surface.

    Injected into a SUBCLASS rather than the module, and over a VALID sidecar
    on purpose: a damaged sidecar would leave "did the guard swallow it, or was
    the data simply unreadable?" undecidable, which is exactly the ambiguity
    that let this survive. Here the sidecar is known good, so a digit in the
    output can only mean the guard ate a bug.
    """
    from quantlab.dataset.crsp import CrspStockDataset
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    class TypoInAsOf(CrspTickerLookup):
        def as_of(self, permno, day):
            raise KeyError("tikcer")  # a one-character typo in a span index

    lookup = TypoInAsOf(CrspStockDataset(converted).ticker_sidecar_path())

    with pytest.raises(KeyError):
        lookup.label([13407], date(2022, 6, 9))


def test_a_bug_inside_intervals_reaches_the_caller_too(converted):
    """The other half of the same path.

    Damage arrives by two routes and so does a bug: `label()` has TWO `except`
    sites, one around the intervals read and one around the per-PERMNO `as_of`.
    Narrowing one and not the other would leave a typo in `_intervals`
    (`self.paylaod` -> `AttributeError`) still degrading silently to digits.
    """
    from quantlab.dataset.crsp import CrspStockDataset
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    class TypoInIntervals(CrspTickerLookup):
        def _intervals(self):
            raise AttributeError("paylaod")  # a typo in an attribute name

    lookup = TypoInIntervals(CrspStockDataset(converted).ticker_sidecar_path())

    with pytest.raises(AttributeError):
        lookup.label([13407], date(2022, 6, 9))


# ---------------------------------------------------------------------------
# G-03.11-6 / WR-03 -- the degradation is visible, and says its piece ONCE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", MALFORMED_SIDECARS)
def test_a_degraded_lookup_warns_at_most_once_per_instance(
    tmp_path, text, warning_messages
):
    """T-03.11-61: one note per INSTANCE, not one per record.

    The most expensive caller renders a RUN of records on a single date --
    `backtest/engine_vectorbt.py:303` hands a whole forced-liquidation batch to
    one `label()` call mid-simulation -- and the instance outlives the call.
    A warning per record would bury the signal it exists to raise, which is
    operationally the same as having no warning at all.

    Both damage routes are exercised by the parametrisation, and they reach
    `_degrade` from different places: the two payloads that break while
    `intervals` is read enter it ONCE per call from the outer `except`, while
    the one that breaks inside the per-PERMNO `as_of` would enter it 500 times
    in the second call alone without the instance flag.
    """
    lookup, _ = _written(tmp_path, text)

    assert lookup.label([13407], date(2022, 6, 9)) == ["13407"]
    assert lookup.label([13407] * 500, date(2022, 6, 9)) == ["13407"] * 500
    assert lookup.label([13407, 14593], date(2022, 6, 9)) == ["13407", "14593"]

    assert len(warning_messages) == 1, warning_messages


def test_an_absent_sidecar_warns_as_well_as_falling_back(
    tmp_path, warning_messages
):
    """T-03.11-57: the OTHER half of telling the two states apart.

    `label()` answering with digits is the normal, documented, supported output
    for a store that simply has no sidecar -- a Tiingo or Alpaca panel, or a
    CRSP store built before 03.11-09. If only the CORRUPT branch warned, an
    operator reading digits would still have to guess which of the two they
    were looking at. The absent branch warns too, so the question a console can
    answer is "is this store missing a sidecar?" rather than "is this store
    missing a sidecar, or is the one it has unreadable?".
    """
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    lookup = CrspTickerLookup(tmp_path / "absent.crsp_tickers.json")

    assert lookup.label([13407, 99999], date(2022, 6, 9)) == ["13407", "99999"]
    assert len(warning_messages) == 1, warning_messages
    assert "CrspTickerLookup" in warning_messages[0]


def test_a_non_integer_label_passes_through_without_a_warning(
    tmp_path, warning_messages
):
    """The narrow `int(value)` guard is NOT a degradation and must stay quiet.

    It catches the CALLER's argument, not the file: a string symbol axis from
    another vendor reaching a shared display path is a supported, healthy case,
    and the sidecar underneath is perfectly readable. Warning here would fire
    once per lookup instance on every Tiingo panel that renders a symbol list,
    and a warning that fires on healthy input is a warning operators learn to
    ignore -- taking the corrupt-sidecar signal down with it.
    """
    lookup, _ = _written(tmp_path, VALID_SIDECAR)

    assert lookup.label(["QQQ", "SPY"], date(2015, 1, 1)) == ["QQQ", "SPY"]
    assert warning_messages == []


def test_a_partly_broken_sidecar_degrades_per_permno_and_keeps_order(
    tmp_path, warning_messages
):
    """Damage to ONE security costs that security its name and nothing else.

    `label` is positional -- one string per input, in order -- and the
    per-PERMNO `except` keeps that true through a partial failure: 13407 is
    still spelled FB in both of the positions it occupies, 14593 falls back to
    its digits in the one it occupies, and the list is the same length as the
    input. The single warning is what says the file, not the roster, is the
    problem.
    """
    lookup, _ = _written(tmp_path, PARTLY_BROKEN_SIDECAR)

    assert lookup.label([13407, 14593, 13407], date(2015, 1, 1)) == [
        "FB",
        "14593",
        "FB",
    ]
    assert len(warning_messages) == 1, warning_messages


def test_the_degraded_flag_is_a_log_throttle_and_not_a_state_machine():
    """A SOURCE assertion, because the property is about reach, not behaviour.

    `_degraded` exists only to keep `_degrade` from repeating itself. It is
    written in `__init__` and in `_degrade`, read in `_degrade`, and nowhere
    else -- in particular it never reaches a query, so no answer this class
    gives depends on whether a warning has already been printed.

    That is what makes the flag safe without a lock. `label()` is reached from
    `joblib`'s threading backend by way of the model and backtest layers, and
    the read-modify-write here is not atomic: two threads can both see `False`
    and both warn. The cost of that race is a duplicate log line. If the flag
    ever became an input to a RETURN value, the same race would start deciding
    whether a PERMNO gets its ticker, and this test is what stands between the
    two situations.

    No behavioural test can catch that drift -- a flag quietly consulted in
    `as_of` would keep every existing assertion green -- so the lock has to
    read the source.
    """
    import ast
    from pathlib import Path

    import quantlab.dataset.crsp.tickers as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    touched: dict[str, int] = {}
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        hits = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Attribute) and node.attr == "_degraded"
        ]
        if hits:
            touched[function.name] = len(hits)

    assert touched == {"__init__": 1, "_degrade": 2}, touched

    degrade = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_degrade"
    )
    assert not [
        node
        for node in ast.walk(degrade)
        if isinstance(node, ast.Return) and node.value is not None
    ]


def test_product_end_is_parsed_from_the_recorded_vintage(converted):
    """A derived value behind a `@property`, like `CrspReference.product_end`."""
    lookup = _lookup(converted)

    assert lookup.product_end == date(2025, 12, 31)


# ---------------------------------------------------------------------------
# G-03.11-6 / WR-04 -- the third entry point uses the same shape check
# ---------------------------------------------------------------------------


def test_product_end_refuses_a_non_object_payload_with_a_shaped_error(tmp_path):
    """The bare `AttributeError` 03.11-12 set out to delete, on 03.11-12's own
    input.

    `_intervals()` called itself "the one place the payload's top-level shape
    is checked" while `product_end` read the payload directly, so a `[]`
    sidecar answered with `AttributeError: 'list' object has no attribute
    'get'` -- naming neither the file nor the way out, from an entry point the
    module docstring already promised a shaped refusal for. `[]` was one of the
    three inputs in `MALFORMED_SIDECARS` the whole time; the plan reproduced it
    and fixed two of the three public methods that consume it.
    """
    lookup, path = _written(tmp_path, "[]")

    with pytest.raises(ValueError) as excinfo:
        lookup.product_end

    message = str(excinfo.value)
    assert "CrspTickerLookup:" in message
    assert str(path) in message
    assert ".crsp_*.json" in message


@pytest.mark.parametrize("text", MALFORMED_SIDECARS)
def test_product_end_never_answers_with_a_bare_exception(tmp_path, text):
    """The same three inputs, through the THIRD entry point.

    The conclusions differ by input and that is the point: `[]` fails the
    top-level shape check and is refused, while the other two have a perfectly
    good top level and simply record no vintage, so `None` is the right answer
    for them. `product_end` has no opinion about `intervals` -- a sidecar whose
    interval table is wrong can still say honestly which CRSP vintage it was
    read against, and turning that into a refusal would be `product_end`
    inventing a second shape check of its own.

    What must hold for ALL three is that neither conclusion is a BARE
    exception: every answer is either a value or a refusal that names the
    class, the path and the rebuild. An `AttributeError` escaping here fails
    this test on both branches, which is exactly how the defect presented.
    """
    lookup, path = _written(tmp_path, text)

    try:
        answer = lookup.product_end
    except ValueError as exc:
        message = str(exc)
        assert "CrspTickerLookup:" in message
        assert str(path) in message
        assert ".crsp_*.json" in message
    else:
        assert answer is None


@pytest.mark.parametrize("payload", UNPARSEABLE_SIDECARS)
def test_product_end_refuses_an_unparseable_sidecar_too(tmp_path, payload):
    """The parse stage reaches the third entry point exactly as it reaches the
    other two.

    Nothing new happens here -- `product_end` goes through the `payload`
    property like everything else, and that property has raised shaped
    refusals since 03.11-15. It is pinned because the "three entry points,
    three deliberate postures" claim in the module docstring is only checkable
    if all three are actually checked, and a zero-byte sidecar is one of the
    three spellings of "empty" this file keeps apart.
    """
    lookup, path = _written_bytes(tmp_path, payload)

    with pytest.raises(ValueError) as excinfo:
        lookup.product_end

    message = str(excinfo.value)
    assert "CrspTickerLookup:" in message
    assert str(path) in message
    assert ".crsp_*.json" in message


def test_beside_store_builds_the_lookup_from_a_store_path(converted):
    """For a caller holding only a store path: the suffix is appended here,
    not spelled by the caller."""
    from quantlab.dataset.crsp.tickers import CrspTickerLookup

    lookup = CrspTickerLookup.beside_store(converted.zarr_file_path)

    assert lookup.as_of(13407, date(2022, 6, 9)) == "META"
