import json
import os
from pathlib import Path

import pytest
from loguru import logger

from quantlab.acquisition.config import AcquisitionConfig


def _make_config(tmp_path: Path) -> AcquisitionConfig:
    return AcquisitionConfig(
        market="us_equity",
        frequency="1d",
        vendor="tiingo",
        raw_data_dir_path=str(tmp_path / "raw" / "tiingo"),
        watermark_path=str(tmp_path / "watermark"),
        symbols=("AAPL",),
        start_date="2024-01-01",
        end_date="2024-01-31",
    )


#: The hive partition every row of `tiingo_json_response` lands in -- the
#: fixture's dates are all in January 2024 and `RAW_HIVE_KEYS["1d"]` is
#: `("month",)`, so `month=2024-01` is the one leaf directory a download here
#: produces.
_SHARD_PARTITION = "month=2024-01"


def _raw_root(tmp_path: Path) -> Path:
    """The vendor-terminated raw root every config in this module points at."""
    return tmp_path / "raw" / "tiingo"


def _shard_symbols(tmp_path: Path) -> set[str]:
    """Every symbol present in the raw hive tree.

    03.2 D-08 replaced the pre-existing `{raw}/{symbol}/data.pqt` layout with a
    hive-partitioned one whose shard filenames carry a content-derived batch
    key (`part-{batch_key}-{page:05d}.pqt`). Asserting on a hardcoded filename
    would pin the hash rather than the behaviour, so these tests assert on what
    actually landed: the symbols readable back out of the tree.
    """
    import polars as pl

    root = _raw_root(tmp_path)
    files = sorted(root.rglob("*.pqt"))
    if not files:
        return set()
    frame = pl.concat(
        [pl.read_parquet(path) for path in files], how="vertical_relaxed"
    )
    return set(frame.get_column("symbol").unique().to_list())


def test_missing_api_key_raises_before_network_call(
    monkeypatch, mock_tiingo_client, tmp_path
):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)

    from quantlab.acquisition.tiingo import TiingoAcquisition

    config = _make_config(tmp_path)
    with pytest.raises(RuntimeError, match="TIINGO_API_KEY"):
        TiingoAcquisition(config)

    assert mock_tiingo_client.calls == []


def test_download_writes_parquet_and_watermark(mock_tiingo_client, tmp_path):
    from quantlab.acquisition.tiingo import TiingoAcquisition
    from quantlab.enums.data import TiingoColumns

    config = _make_config(tmp_path)
    acq = TiingoAcquisition(config)
    acq.download(["AAPL"])

    # D-08: one hive-partitioned shard per (partition, page), at a
    # deterministic `part-{batch_key}-{page:05d}.pqt` name under the
    # vendor-terminated raw root -- not the pre-03.2 `{symbol}/data.pqt`.
    partition = _raw_root(tmp_path) / _SHARD_PARTITION
    shards = sorted(partition.glob("part-*-00000.pqt"))
    assert len(shards) == 1, sorted(_raw_root(tmp_path).rglob("*"))
    assert _shard_symbols(tmp_path) == {"AAPL"}

    assert len(mock_tiingo_client.calls) == 1
    call = mock_tiingo_client.calls[0]
    assert call["ticker"] == "AAPL"
    assert call["columns"] == TiingoColumns.EOD
    assert call["frequency"] == "daily"
    assert call["startDate"] == "2024-01-01"
    assert call["endDate"] == "2024-01-31"

    watermark_file = tmp_path / "watermark" / "AAPL.json"
    assert watermark_file.exists()
    with open(watermark_file) as f:
        assert json.load(f)["last_date"] == "2024-01-31"


def test_refresh_uses_watermark_not_config_start_date(
    mock_tiingo_client, tmp_path
):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    config = _make_config(tmp_path)
    acq = TiingoAcquisition(config)

    watermark_dir = tmp_path / "watermark"
    watermark_dir.mkdir(parents=True, exist_ok=True)
    with open(watermark_dir / "AAPL.json", "w") as f:
        json.dump({"last_date": "2024-01-15"}, f)

    acq.refresh(["AAPL"])

    assert len(mock_tiingo_client.calls) == 1
    call = mock_tiingo_client.calls[0]
    assert call["startDate"] == "2024-01-15"
    assert call["startDate"] != config.start_date


def test_credential_never_exposed_on_config_surface(mock_tiingo_client, tmp_path):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    config = _make_config(tmp_path)
    acq = TiingoAcquisition(config)

    # Check dict *keys*, not the full serialized string -- pytest's tmp_path
    # embeds the test function name in its path, which can coincidentally
    # contain one of these substrings and produce a false positive.
    keys = set(config.to_dict().keys())
    for forbidden in ("api_key", "credential", "token", "secret"):
        assert forbidden not in keys

    for attr_name, attr_value in vars(acq).items():
        if attr_name == "_client":
            continue
        assert "test-key-not-real" not in repr(attr_value)


# ---------------------------------------------------------------------------
# Concurrent, resumable, failure-isolated bulk acquisition
# (260906-0iy Task 2, D-03)
#
# These behaviours belonged to a separate `ConcurrentTiingoAcquisition` until
# 03.2-03 hoisted the orchestration onto `SymbolBatchAcquisition` and retired that name
# (D-02, 03.1 D-03). They are unchanged; only the class that carries them is.
#
# A ~15k-symbol, multi-hour backfill makes resumability and per-symbol failure
# isolation mandatory: one delisted ticker returning a 404 must not abort the
# other 15,000, and a job killed at ticker 20,000 must resume near ticker
# 20,000 rather than re-downloading from the top.
#
# No test here makes a real network call or requires a real credential --
# every one drives the `mock_tiingo_client` fixture.
# ---------------------------------------------------------------------------

_FIVE = ("AAPL", "MSFT", "GOOG", "AMZN", "META")


def _make_concurrent_config(
    tmp_path: Path,
    symbols: tuple[str, ...] = _FIVE,
    kwargs: dict | None = None,
) -> AcquisitionConfig:
    return AcquisitionConfig(
        market="us_equity",
        frequency="1d",
        vendor="tiingo",
        raw_data_dir_path=str(tmp_path / "raw" / "tiingo"),
        watermark_path=str(tmp_path / "watermark"),
        symbols=symbols,
        start_date="2024-01-01",
        end_date="2024-01-31",
        kwargs=kwargs if kwargs is not None else {"max_workers": 3},
    )


def test_concurrent_download_writes_every_parquet_and_watermark(
    mock_tiingo_client, tmp_path
):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    config = _make_concurrent_config(tmp_path)
    TiingoAcquisition(config).download()

    assert _shard_symbols(tmp_path) == set(_FIVE)
    for symbol in _FIVE:
        assert (tmp_path / "watermark" / f"{symbol}.json").exists()

    assert len(mock_tiingo_client.calls) == len(_FIVE)
    assert {call["ticker"] for call in mock_tiingo_client.calls} == set(_FIVE)


def test_second_download_skips_symbols_already_at_the_watermark(
    mock_tiingo_client, tmp_path
):
    """D-03's resumability property, asserted by COUNTING VENDOR CALLS rather
    than by timing -- a job killed at ticker 20,000 and restarted must issue
    zero requests for the 20,000 already at the target watermark.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    config = _make_concurrent_config(tmp_path)
    TiingoAcquisition(config).download()
    assert len(mock_tiingo_client.calls) == len(_FIVE)

    mock_tiingo_client.calls.clear()
    TiingoAcquisition(config).download()

    assert mock_tiingo_client.calls == []


def test_resume_false_re_fetches_symbols_already_at_the_watermark(
    mock_tiingo_client, tmp_path
):
    """The knob is read from `config.kwargs` -- the escape hatch
    `AcquisitionConfig` already documents -- so it stays config-driven rather
    than becoming a constructor argument no config file can reach.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    TiingoAcquisition(_make_concurrent_config(tmp_path)).download()
    mock_tiingo_client.calls.clear()

    no_resume = _make_concurrent_config(
        tmp_path, kwargs={"max_workers": 3, "resume": False}
    )
    TiingoAcquisition(no_resume).download()

    assert len(mock_tiingo_client.calls) == len(_FIVE)


def _fail_one(mock_client, bad_symbol: str, message: str) -> None:
    """Make `mock_client.get_ticker_price` raise for exactly one ticker."""
    original = mock_client.get_ticker_price

    def failing(self, ticker, **kwargs):
        if ticker == bad_symbol:
            raise RuntimeError(message)
        return original(self, ticker, **kwargs)

    mock_client.get_ticker_price = failing


def test_one_symbol_failure_does_not_abort_the_others(
    mock_tiingo_client, tmp_path
):
    """One delisted ticker returning a 404 must not abort the other ~15,000.

    The failed symbol gets NO watermark, so the next run retries it.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    _fail_one(mock_tiingo_client, "GOOG", "404 Not Found")

    config = _make_concurrent_config(tmp_path)
    result = TiingoAcquisition(config).download()

    assert result is not None  # the run returns normally, it does not raise

    landed = _shard_symbols(tmp_path)
    for symbol in ("AAPL", "MSFT", "AMZN", "META"):
        assert symbol in landed
        assert (tmp_path / "watermark" / f"{symbol}.json").exists()

    # The failed symbol wrote no shard at all -- `_fetch_batch` raises before
    # any write when `_fetch_page` does.
    assert "GOOG" not in landed

    # No watermark for the failure => the next run retries it.
    assert not (tmp_path / "watermark" / "GOOG.json").exists()

    manifest_path = tmp_path / "watermark" / "_failures.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        failures = json.load(f)
    assert set(failures) == {"GOOG"}
    assert "404" in failures["GOOG"]


def test_failure_manifest_never_contains_the_api_key(
    mock_tiingo_client, tmp_path, capsys
):
    """T-0iy-01. This repo has already leaked one real Tiingo key, and an
    exception string is a path a credential travels that nobody audits: the
    vendor client embeds the token in the request URL it echoes back on an
    auth error.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    key = os.environ["TIINGO_API_KEY"]
    _fail_one(
        mock_tiingo_client,
        "GOOG",
        f"401 Client Error for url: https://api.tiingo.com/tiingo/daily/goog/prices?token={key}",
    )

    config = _make_concurrent_config(tmp_path)
    TiingoAcquisition(config).download()

    manifest_text = (tmp_path / "watermark" / "_failures.json").read_text()
    assert key not in manifest_text
    assert "REDACTED" in manifest_text

    captured = capsys.readouterr()
    assert key not in captured.out
    assert key not in captured.err


def test_concurrent_refresh_starts_each_symbol_from_its_own_watermark(
    mock_tiingo_client, tmp_path
):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    config = _make_concurrent_config(tmp_path, symbols=("AAPL", "MSFT"))

    watermark_dir = tmp_path / "watermark"
    watermark_dir.mkdir(parents=True, exist_ok=True)
    with open(watermark_dir / "AAPL.json", "w") as f:
        json.dump({"last_date": "2024-01-15"}, f)
    with open(watermark_dir / "MSFT.json", "w") as f:
        json.dump({"last_date": "2024-01-20"}, f)

    TiingoAcquisition(config).refresh()

    starts = {
        call["ticker"]: call["startDate"] for call in mock_tiingo_client.calls
    }
    assert starts == {"AAPL": "2024-01-15", "MSFT": "2024-01-20"}


# ---------------------------------------------------------------------------
# Range-aware watermarks (260906-26o Task 1, D-01/D-03/D-04)
#
# The defect: watermarks recorded only the END date, so `_run` skipped a symbol
# whenever `_read_watermark(symbol) == config.end_date`. Widening
# `--start-date` on a later run therefore silently skipped every
# already-fetched symbol and shipped a dataset whose per-symbol history depth
# was inconsistent, with no warning at all.
#
# The fix records the covered RANGE. The migration deliberately does NOT
# guess: a legacy `{"last_date": ...}` sidecar reads back with an ABSENT start,
# and no code path anywhere fills it in from `config.start_date` or any other
# fallback (D-04). `test_read_coverage_on_a_legacy_watermark_never_invents_a_start`
# exists to make adding such a "helpful" default turn this suite red.
#
# Every test here is offline via `mock_tiingo_client`, writes only under
# `tmp_path`, and touches nothing on any real data volume.
# ---------------------------------------------------------------------------


def _captured_warnings():
    """Attach a temporary in-memory loguru sink.

    loguru does not propagate to stdlib `logging`, so pytest's `caplog` sees
    nothing (the same constraint recorded in tests/test_universe.py and
    tests/test_chunked_ingest.py).
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    return messages, sink_id


def _write_legacy_watermark(tmp_path: Path, symbol: str, last_date: str) -> None:
    """Write a sidecar in the pre-26o on-disk format: END DATE ONLY.

    This is the exact shape of all 4,638 files observed on the user's volume
    at planning time -- key set `('last_date',)`, nothing else.
    """
    watermark_dir = tmp_path / "watermark"
    watermark_dir.mkdir(parents=True, exist_ok=True)
    with open(watermark_dir / f"{symbol}.json", "w") as f:
        json.dump({"last_date": last_date}, f)


def _read_sidecar(tmp_path: Path, symbol: str) -> dict:
    with open(tmp_path / "watermark" / f"{symbol}.json") as f:
        return json.load(f)


def test_read_coverage_on_a_legacy_watermark_never_invents_a_start(
    mock_tiingo_client, tmp_path
):
    """D-04, pinned. An assumed covered start that is WRONG reproduces exactly
    the silent gap this change exists to eliminate -- and reproduces it
    invisibly. Only the user knows what window those files were fetched over,
    which is why stamping is an explicit, user-supplied step.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    _write_legacy_watermark(tmp_path, "AAPL", "2024-01-31")
    acq = TiingoAcquisition(_make_config(tmp_path))

    coverage = acq._read_coverage("AAPL")
    assert coverage is not None
    assert coverage["last_date"] == "2024-01-31"
    assert coverage["start_date"] is None
    # Not merely falsy -- explicitly not the config value, not a default.
    assert coverage["start_date"] != acq.config.start_date


def test_read_coverage_on_a_new_format_watermark_returns_both_components(
    mock_tiingo_client, tmp_path
):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    acq._write_watermark("AAPL", "2024-01-31", start_date="2024-01-01")

    # `no_data` joined this dict in 03.2-05 as the fourth read-time state
    # (D-04/SC-4). It is asserted here as an EQUALITY rather than dropped from
    # the comparison, because the point of this test is that `_read_coverage`
    # returns the whole recorded coverage and nothing invented -- a `False`
    # here is the correct reading of a sidecar whose marker key is absent,
    # which is exactly what `_write_watermark` wrote above.
    assert acq._read_coverage("AAPL") == {
        "start_date": "2024-01-01",
        "last_date": "2024-01-31",
        "no_data": False,
    }


def test_read_coverage_returns_none_for_an_absent_or_corrupt_sidecar(
    mock_tiingo_client, tmp_path
):
    """Matches `_read_watermark`'s existing tolerant fallback rather than
    introducing a second failure policy: a corrupt sidecar must never crash a
    15k-symbol run, and the worst case is a wider-than-necessary re-fetch.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    assert acq._read_coverage("AAPL") is None

    watermark_dir = tmp_path / "watermark"
    watermark_dir.mkdir(parents=True, exist_ok=True)
    (watermark_dir / "MSFT.json").write_text("{not json at all")
    assert acq._read_coverage("MSFT") is None


def test_write_watermark_stays_readable_by_the_unchanged_read_watermark(
    mock_tiingo_client, tmp_path
):
    """The schema is purely ADDITIVE: `last_date` is deliberately not renamed,
    so new code reads old files and old code reads new files.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))

    acq._write_watermark("AAPL", "2024-01-31", start_date="2024-01-01")
    assert acq._coverage.read_watermark("AAPL") == "2024-01-31"

    # Omitting the start writes NO key at all -- an unknown start is
    # represented by absence, never by a null that a reader could mistake for
    # a recorded value.
    acq._write_watermark("MSFT", "2024-01-31")
    assert acq._coverage.read_watermark("MSFT") == "2024-01-31"
    assert "start_date" not in _read_sidecar(tmp_path, "MSFT")


def test_sequential_download_records_the_requested_start(
    mock_tiingo_client, tmp_path
):
    """`_fetch_and_write` overwrites the parquet wholesale, so after a
    successful fetch the file contains exactly the requested range -- recording
    `config.start_date` as the covered start is a true statement about it.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    TiingoAcquisition(_make_config(tmp_path)).download(["AAPL"])

    assert _read_sidecar(tmp_path, "AAPL") == {
        "start_date": "2024-01-01",
        "last_date": "2024-01-31",
    }


def test_sequential_refresh_against_a_legacy_watermark_leaves_the_start_absent(
    mock_tiingo_client, tmp_path
):
    """`refresh()` fetches from the symbol's own `last_date` forward, so the
    covered start is whatever it already was. If it was unknown it STAYS
    unknown -- refresh does not invent coverage it did not fetch (D-04).
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    _write_legacy_watermark(tmp_path, "AAPL", "2024-01-15")
    TiingoAcquisition(_make_config(tmp_path)).refresh(["AAPL"])

    sidecar = _read_sidecar(tmp_path, "AAPL")
    assert sidecar["last_date"] == "2024-01-31"
    assert "start_date" not in sidecar


def test_sequential_refresh_carries_forward_a_known_start(
    mock_tiingo_client, tmp_path
):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    acq._write_watermark("AAPL", "2024-01-15", start_date="2020-01-01")
    acq.refresh(["AAPL"])

    assert _read_sidecar(tmp_path, "AAPL") == {
        "start_date": "2020-01-01",
        "last_date": "2024-01-31",
    }


def test_second_download_with_an_earlier_start_re_fetches(
    mock_tiingo_client, tmp_path
):
    """THE regression test for the reported defect (D-03).

    Before this change, widening the window silently skipped every symbol
    already at the end-date watermark and shipped a dataset with inconsistent
    per-symbol history depth. Asserted by COUNTING and INSPECTING vendor
    calls, because a call count is the only evidence a passing write cannot
    fake.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    TiingoAcquisition(_make_concurrent_config(tmp_path)).download()
    assert len(mock_tiingo_client.calls) == len(_FIVE)
    mock_tiingo_client.calls.clear()

    widened = _make_concurrent_config(tmp_path)
    widened.start_date = "2020-01-01"
    TiingoAcquisition(widened).download()

    assert len(mock_tiingo_client.calls) == len(_FIVE)
    # The WIDENED start is what actually reaches the vendor -- re-fetching the
    # same narrow window would leave the gap in place while looking busy.
    assert {call["startDate"] for call in mock_tiingo_client.calls} == {
        "2020-01-01"
    }
    assert _read_sidecar(tmp_path, "AAPL")["start_date"] == "2020-01-01"


def test_second_download_with_the_same_start_issues_zero_vendor_calls(
    mock_tiingo_client, tmp_path
):
    """D-01. The 4,621 already-downloaded symbols must not be re-fetched by
    default -- re-downloading them costs an entire hourly window for nothing.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    config = _make_concurrent_config(tmp_path)
    TiingoAcquisition(config).download()
    mock_tiingo_client.calls.clear()

    TiingoAcquisition(_make_concurrent_config(tmp_path)).download()
    assert mock_tiingo_client.calls == []


def test_second_download_with_a_later_start_issues_zero_vendor_calls(
    mock_tiingo_client, tmp_path
):
    """A narrower request inside proven coverage is not work."""
    from quantlab.acquisition.tiingo import TiingoAcquisition

    TiingoAcquisition(_make_concurrent_config(tmp_path)).download()
    mock_tiingo_client.calls.clear()

    narrowed = _make_concurrent_config(tmp_path)
    narrowed.start_date = "2024-01-10"
    TiingoAcquisition(narrowed).download()

    assert mock_tiingo_client.calls == []


def test_legacy_watermarks_are_skipped_by_default_and_reported_loudly(
    mock_tiingo_client, tmp_path
):
    """D-04 option (c). What makes the D-03 failure mode dangerous is SILENCE,
    not the skip. A run that skips these while printing their exact count and
    the setting that resolves it is a REPORTED gap with a named cure.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    for symbol in _FIVE:
        _write_legacy_watermark(tmp_path, symbol, "2024-01-31")

    messages, sink_id = _captured_warnings()
    try:
        TiingoAcquisition(_make_concurrent_config(tmp_path)).download()
    finally:
        logger.remove(sink_id)

    assert mock_tiingo_client.calls == []

    text = "\n".join(messages)
    assert str(len(_FIVE)) in text
    assert "legacy_watermarks='refetch'" in text


def test_legacy_watermarks_are_re_fetched_under_the_refetch_policy(
    mock_tiingo_client, tmp_path
):
    """The opt-in escape hatch. Making it a knob is what turns "unknown
    coverage is treated as covered" from an accident into a choice.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    for symbol in _FIVE:
        _write_legacy_watermark(tmp_path, symbol, "2024-01-31")

    config = _make_concurrent_config(
        tmp_path, kwargs={"max_workers": 3, "legacy_watermarks": "refetch"}
    )
    TiingoAcquisition(config).download()

    assert len(mock_tiingo_client.calls) == len(_FIVE)


def test_concurrent_refresh_is_not_forced_to_re_fetch_by_a_widened_start(
    mock_tiingo_client, tmp_path
):
    """`refresh()` requests `[watermark, end_date]` per symbol, NOT
    `config.start_date`, so judging its coverage against a widened
    `config.start_date` would mark every symbol pending on every run while the
    re-fetch could not close the gap -- an endless, silent quota burn. Refresh
    keeps the end-date-only rule; widening is `download()`'s job.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    TiingoAcquisition(_make_concurrent_config(tmp_path)).download()
    mock_tiingo_client.calls.clear()

    widened = _make_concurrent_config(tmp_path)
    widened.start_date = "2010-01-01"
    TiingoAcquisition(widened).refresh()

    assert mock_tiingo_client.calls == []


def test_coverage_report_counts_without_issuing_a_single_vendor_call(
    mock_tiingo_client, tmp_path
):
    """The read-only form `--dry-run` prints, so "would widening --start-date
    re-fetch anything?" is answerable BEFORE committing to a multi-hour job.

    It shares `_partition_by_coverage` with the real run, so the dry run and
    the run it predicts can never disagree.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_concurrent_config(tmp_path))
    acq._write_watermark("AAPL", "2024-01-31", start_date="2020-01-01")
    acq._write_watermark("MSFT", "2024-01-31", start_date="2024-01-15")
    _write_legacy_watermark(tmp_path, "GOOG", "2024-01-31")
    # AMZN and META have no watermark at all.

    report = acq.coverage_report()

    assert report["requested"] == len(_FIVE)
    assert report["covered"] == 1  # AAPL covers 2024-01-01
    assert report["widened"] == 1  # MSFT's coverage starts after it
    assert report["legacy"] == 1  # GOOG's start is unknown
    assert report["pending"] == 3  # MSFT + AMZN + META
    assert report["skipped"] == 2  # AAPL + the skipped legacy GOOG
    assert mock_tiingo_client.calls == []


# ---------------------------------------------------------------------------
# WR-05 -- the raw tier pins DTYPES, not only names and order.
# ---------------------------------------------------------------------------


def test_two_symbols_whose_json_infers_different_dtypes_share_one_schema(
    mock_tiingo_client, tmp_path
):
    """WR-05. `RAW_COLUMNS`' docstring claims the raw tier is "schema-stable
    BY CONSTRUCTION" and warns that one differently-shaped shard "makes the
    whole vendor root unreadable". Column set and order were pinned; DTYPE was
    not.

    `pl.DataFrame(json_rows)` infers per response, so a symbol whose `divCash`
    is all integer `0`, or whose `volume` is all null over the window, yields
    `Int64`/`Null` where its siblings yield `Float64`. A directory scan derives
    ONE schema from the first file it opens and enforces it across the rest --
    and `dataset/stock.py` deliberately leaves `extra_columns`/`missing_columns`
    at their raising defaults, so the scan fails with a `SchemaError` naming a
    file rather than a cause. It also looks intermittent, because which file
    polars opens first is filename-ordering dependent.

    Asserted on the SHARDS (identical dtypes) and then through a real scan of
    the whole root, which is the failure that would actually be reported.
    """
    import polars as pl

    from quantlab.acquisition.tiingo import TiingoAcquisition

    original = mock_tiingo_client.get_ticker_price

    def per_symbol(self, ticker, **kwargs):
        rows = original(self, ticker, **kwargs)
        if ticker != "MSFT":
            return rows
        # MSFT's window happens to carry whole-number dividends/splits and no
        # volume at all -- both entirely legal responses.
        degenerate = []
        for row in rows:
            row = dict(row)
            for name in ("divCash", "splitFactor"):
                if name in row:
                    row[name] = 0
            if "volume" in row:
                row["volume"] = None
            degenerate.append(row)
        return degenerate

    mock_tiingo_client.get_ticker_price = per_symbol

    config = _make_concurrent_config(tmp_path, symbols=("AAPL", "MSFT"))
    TiingoAcquisition(config).download()

    shards = sorted(Path(config.raw_data_dir_path).rglob("*.pqt"))
    assert len(shards) >= 2, [str(path) for path in shards]

    schemas = {tuple(pl.read_parquet(path).schema.items()) for path in shards}
    assert len(schemas) == 1, (
        "two shards under one vendor root carry different schemas:\n"
        + "\n".join(
            f"{path}: {dict(pl.read_parquet(path).schema)}" for path in shards
        )
    )
    assert dict(pl.read_parquet(shards[0]).schema)["divCash"] == pl.Float64

    # And the scan that a `Dataset` actually performs succeeds -- strictness
    # left at its raising defaults, exactly as `dataset/stock.py` does it.
    scanned = pl.scan_parquet(
        Path(config.raw_data_dir_path), hive_partitioning=True
    ).collect()
    assert scanned.height > 0


def _leading_null_low_response(null_rows: int = 150, total: int = 200) -> list[dict]:
    """Tiingo's real shape for a symbol whose early history has no `low`.

    A CONTIGUOUS LEADING run of nulls longer than polars' default
    `infer_schema_length` of 100, then real floats. Measured against the live
    API on 2026-09-13: `CAB` is null in `low`/`adjLow` for its first 409
    sessions (first real value 1.5, 2009-04-03) and `VNLPY` for its first 337
    (first real value 0.04, 2008-11-13). 150/200 reproduces the same shape at
    test speed -- what matters is only that the run crosses 100.
    """
    rows = []
    for index in range(total):
        nulled = index < null_rows
        day = 1 + index
        rows.append(
            {
                "date": f"2024-01-01T00:00:00.000Z".replace(
                    "01-01", f"{1 + day // 28:02d}-{1 + day % 28:02d}"
                ),
                "open": 100.0,
                "high": 102.5,
                "low": None if nulled else 1.5,
                "close": 101.0,
                "volume": 1000000,
                "adjOpen": 100.0,
                "adjHigh": 102.5,
                "adjLow": None if nulled else 1.5,
                "adjClose": 101.0,
                "adjVolume": 1000000,
                "divCash": 0.0,
                "splitFactor": 1.0,
            }
        )
    return rows


def test_a_leading_null_run_longer_than_the_infer_window_still_fetches(
    monkeypatch, tmp_path
):
    """The `CAB` / `VNLPY` regression: nulls at the START must not fail the fetch.

    polars infers dtypes from the first `infer_schema_length` rows (default
    100). When a column is null across that whole window and real afterwards,
    the column types `Null` and the first real value raises AT CONSTRUCTION:

        ComputeError: could not append value: 1.5 of type: f64 to the builder

    WR-05 pinned dtypes with a `.cast()` AFTER construction and argued the two
    vendors could then not drift on what "schema-stable" means. Only half was
    ported: a cast repairs a frame built with the wrong dtype and cannot run
    when the build itself throws. `acquisition/alpaca.py` had already reached
    the right spelling from the other direction (an OMITTED field rather than a
    nulled one); this pins that Tiingo keeps it too.

    Both symbols failed on EVERY full-market run before this, recording a
    polars builder error in `_failures.json` -- a message naming neither the
    symbol's data shape nor anything an operator could act on.
    """
    import polars as pl

    from quantlab.acquisition.tiingo import TiingoAcquisition

    rows = _leading_null_low_response()

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_ticker_price(self, ticker, **kwargs):
            return rows

    monkeypatch.setattr(
        "quantlab.acquisition.tiingo.TiingoClient", FakeClient, raising=False
    )
    monkeypatch.setenv("TIINGO_API_KEY", "test-key-not-real")

    config = _make_config(tmp_path)
    acq = TiingoAcquisition(config)
    acq.download(["CAB"])

    assert acq.last_result.failures == {}, "the leading null run must not fail"
    assert acq.last_result.succeeded == ("CAB",)

    frame = pl.read_parquet(sorted(_raw_root(tmp_path).rglob("*.pqt")))
    assert frame["low"].dtype == pl.Float64, "the cast still pins the dtype"
    assert frame["low"].null_count() > 0, "the real nulls are preserved, not filled"
    assert frame["low"].drop_nulls().to_list()[0] == 1.5


def test_a_column_the_vendor_nulls_in_every_row_still_types_float(
    monkeypatch, tmp_path
):
    """The other half, and the reason the cast is NOT redundant with the
    construction fix.

    `infer_schema_length=None` scans every row, but an all-null column has no
    row that says otherwise -- it still infers `Null`. The `.cast()` against
    `RAW_SCHEMA` is what turns it into `Float64`, which is what keeps the shard
    readable beside its siblings: `dataset/stock.py` leaves
    `extra_columns`/`missing_columns` at their raising defaults, so one
    `Null`-typed shard makes the whole `month=` directory unscannable.
    """
    import polars as pl

    from quantlab.acquisition.tiingo import TiingoAcquisition

    rows = _leading_null_low_response(null_rows=200, total=200)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_ticker_price(self, ticker, **kwargs):
            return rows

    monkeypatch.setattr(
        "quantlab.acquisition.tiingo.TiingoClient", FakeClient, raising=False
    )
    monkeypatch.setenv("TIINGO_API_KEY", "test-key-not-real")

    acq = TiingoAcquisition(_make_config(tmp_path))
    acq.download(["ALLNULL"])

    assert acq.last_result.failures == {}
    frame = pl.read_parquet(sorted(_raw_root(tmp_path).rglob("*.pqt")))
    assert frame["low"].dtype == pl.Float64
    assert frame["low"].null_count() == frame.height
