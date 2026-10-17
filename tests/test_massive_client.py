"""The Massive client (#238, #240): daily files from S3 and the condition table over REST.

Seam 3 of spec #237, with the S3 and HTTP layers faked. What is locked here:

- a day's file is asked for under Massive's key layout and lands, byte for
  byte, at the raw tier's path; a file already there whole is not fetched
  again;
- a file is fetched in byte ranges, several at once, and the ranges
  assemble the exact bytes; a broken stream resumes from the bytes it
  already wrote;
- a file is recorded only when its size is the vendor's and its gzip decodes
  to the end: a short or corrupt download leaves nothing in the raw tier;
- a run over several days asks only for XNYS sessions, fetches several files
  at once, oldest first, skips a day Massive has not published yet without
  moving the watermark past it, records a watermark per data type and
  resumes after it, and runs at most ``files`` days ahead of its caller;
- "not published" (404), "not entitled" (403) and a transport error are told
  apart, and a transport error, throttling included, is retried after a
  back-off;
- the condition table is pulled with the key in a header, never in a URL,
  across pages, and stored as a snapshot the conversion reads;
- a missing credential fails before any transfer, naming the variables; the
  S3 secret defaults to the API key.

Every byte and value is SYNTHETIC.
"""

from __future__ import annotations

import itertools
import json
import random
import threading
from datetime import date

import pytest

from tests.massive_fixtures import CONDITION_RECORDS, TRADE_HEADER, gzip_text

DAY = date(2024, 11, 29)
KEY = "us_stocks_sip/trades_v1/2024/11/2024-11-29.csv.gz"
BODY = gzip_text([TRADE_HEADER, "AAA,,0,12,1,1,10.0,1,1,100,1,0,0"])  # SYNTHETIC


def _big_body(rows: int = 400) -> bytes:
    """A gzip of incompressible SYNTHETIC rows, a few kilobytes long."""
    rng = random.Random(7)
    return gzip_text([TRADE_HEADER] + [f"T{rng.getrandbits(64):x},,0,12,{i},1,10.0,{i},1,100,1,0,0" for i in range(rows)])


def _key(day: date, data_type: str = "trades") -> str:
    from quantlab.dataset.massive.raw import s3_key

    return s3_key(data_type, day)


ENV = {
    "MASSIVE_API_KEY": "synthetic-api-key",  # SYNTHETIC
    "MASSIVE_S3_ACCESS_KEY_ID": "synthetic-access-id",  # SYNTHETIC
}


class FakeStore:
    """An object store answering from ``objects``.

    ``errors`` are raised once per key, in order, by ``size`` and ``read``;
    ``sizes`` overrides the size a key reports; ``breaks`` maps a key to a
    list of byte counts: each of its next reads delivers that many bytes and
    then breaks. ``barrier`` makes the first two reads wait for each other.
    """

    def __init__(self, objects: dict[str, bytes], *, errors: dict[str, list[Exception]] | None = None,
                 sizes: dict[str, int] | None = None, breaks: dict[str, list[int]] | None = None,
                 barrier: threading.Barrier | None = None):
        self.objects = objects
        self.errors = errors or {}
        self.sizes = sizes or {}
        self.breaks = breaks or {}
        self.barrier = barrier
        self.heads: list[str] = []
        self.reads: list[tuple[str, int, int]] = []
        self._lock = threading.Lock()

    def _raise(self, key: str) -> None:
        with self._lock:
            error = self.errors[key].pop(0) if self.errors.get(key) else None
        if error is not None:
            raise error

    def size(self, key: str) -> int:
        self.heads.append(key)
        self._raise(key)
        if key not in self.objects:
            from quantlab.acquisition.massive.client import MassiveNotPublishedError

            raise MassiveNotPublishedError(key)
        return self.sizes.get(key, len(self.objects[key]))

    def read(self, key: str, start: int, stop: int):
        from quantlab.acquisition.massive.client import MassiveTransportError

        with self._lock:
            self.reads.append((key, start, stop))
            cut = self.breaks[key].pop(0) if self.breaks.get(key) else None
            first_two = len(self.reads) <= 2
        if self.barrier is not None and first_two:
            try:
                self.barrier.wait()
            except threading.BrokenBarrierError:
                pass
        self._raise(key)
        body = self.objects[key][start:stop]
        if cut is not None:
            yield body[:cut]
            raise MassiveTransportError("connection reset")
        for offset in range(0, len(body), 7):
            yield body[offset : offset + 7]

    def fetched(self, key: str) -> bool:
        return any(read[0] == key for read in self.reads)


@pytest.fixture
def env(monkeypatch):
    for name in ("MASSIVE_API_KEY", "MASSIVE_S3_ACCESS_KEY_ID", "MASSIVE_S3_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


def _client(store=None, http_get=None, sleeps=None, **settings):
    from quantlab.acquisition.massive.client import MassiveClient

    return MassiveClient(
        store=store,
        http_get=http_get,
        sleep=(sleeps.append if sleeps is not None else (lambda seconds: None)),
        **settings,
    )


def test_a_days_file_lands_at_the_raw_tier_path(env, tmp_path):
    from quantlab.dataset.massive.raw import raw_days, raw_file

    store = FakeStore({KEY: BODY})
    path = _client(store).download_day("trades", DAY, tmp_path)
    assert path == raw_file(tmp_path / "massive", "trades", DAY)
    assert path.read_bytes() == BODY
    assert store.reads == [(KEY, 0, len(BODY))]
    assert raw_days(tmp_path / "massive", "trades") == [DAY]


def test_the_minute_aggregates_key(env, tmp_path):
    key = "us_stocks_sip/minute_aggs_v1/2024/11/2024-11-29.csv.gz"
    store = FakeStore({key: BODY})
    path = _client(store).download_day("minute_aggs", DAY, tmp_path)
    assert path.parts[-3:] == ("minute_aggs", "2024", "2024-11-29.csv.gz")


def test_a_file_already_whole_is_not_fetched_again(env, tmp_path):
    store = FakeStore({KEY: BODY})
    client = _client(store)
    client.download_day("trades", DAY, tmp_path)
    client.download_day("trades", DAY, tmp_path)
    assert len(store.reads) == 1


def test_ranges_fetched_in_parallel_assemble_the_exact_bytes(env, tmp_path):
    body = _big_body()
    # Two ranges must be in flight together, or the barrier times out and breaks.
    store = FakeStore({KEY: body}, barrier=threading.Barrier(2, timeout=5))
    path = _client(store, streams=4, part_bytes=1000).download_day("trades", DAY, tmp_path)
    assert path.read_bytes() == body
    assert not store.barrier.broken
    ranges = sorted((start, stop) for _, start, stop in store.reads)
    assert len(ranges) == -(-len(body) // 1000)
    assert ranges[0][0] == 0 and ranges[-1][1] == len(body)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(ranges))


def test_a_broken_stream_resumes_from_the_bytes_it_wrote(env, tmp_path):
    body = _big_body()
    sleeps: list[float] = []
    store = FakeStore({KEY: body}, breaks={KEY: [300, 0]})
    path = _client(store, sleeps=sleeps, streams=1, part_bytes=len(body)).download_day("trades", DAY, tmp_path)
    assert path.read_bytes() == body
    # 300 bytes kept: asked again from byte 300; nothing kept: asked again from there after a wait.
    assert [start for _, start, _ in store.reads] == [0, 300, 300]
    assert len(sleeps) == 1


def test_a_stream_that_keeps_breaking_without_progress_fails(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import raw_days

    store = FakeStore({KEY: BODY}, breaks={KEY: [0] * 20})
    with pytest.raises(MassiveTransportError, match="reset"):
        _client(store, retries=3).download_day("trades", DAY, tmp_path)
    assert len(store.reads) == 4
    assert raw_days(tmp_path / "massive", "trades") == []
    assert not list((tmp_path / "massive").rglob("*.part"))


def test_a_file_shorter_than_the_vendor_lists_leaves_nothing(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import raw_days

    store = FakeStore({KEY: BODY}, sizes={KEY: len(BODY) + 50})
    with pytest.raises(MassiveTransportError, match="bytes"):
        _client(store, retries=2).download_day("trades", DAY, tmp_path)
    assert raw_days(tmp_path / "massive", "trades") == []
    assert not list((tmp_path / "massive").rglob("*.part"))


def test_a_corrupt_gzip_leaves_nothing(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import raw_days

    corrupt = BODY[:-12] + b"\x00" * 12
    with pytest.raises(MassiveTransportError, match="gzip"):
        _client(FakeStore({KEY: corrupt})).download_day("trades", DAY, tmp_path)
    assert raw_days(tmp_path / "massive", "trades") == []
    assert not list((tmp_path / "massive").rglob("*.part"))


def test_not_published_and_not_entitled_are_told_apart(env, tmp_path):
    from quantlab.acquisition.massive.client import (
        MassiveEntitlementError,
        MassiveNotPublishedError,
    )

    with pytest.raises(MassiveNotPublishedError):
        _client(FakeStore({})).download_day("trades", date(2024, 11, 28), tmp_path)
    store = FakeStore({KEY: BODY}, errors={KEY: [MassiveEntitlementError("403")]})
    with pytest.raises(MassiveEntitlementError):
        _client(store).download_day("trades", DAY, tmp_path)


def test_the_size_of_a_published_file_is_asked_without_downloading_it(env):
    from quantlab.acquisition.massive.client import (
        MassiveEntitlementError,
        MassiveNotPublishedError,
    )

    store = FakeStore({KEY: BODY}, errors={KEY: [MassiveEntitlementError("403")]})
    client = _client(store)
    with pytest.raises(MassiveEntitlementError):
        client.published_size("trades", DAY)
    assert client.published_size("trades", DAY) == len(BODY)
    with pytest.raises(MassiveNotPublishedError):
        client.published_size("trades", date(2024, 11, 28))
    assert store.reads == []


def test_a_transport_error_is_retried_after_a_back_off(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError

    sleeps: list[float] = []
    store = FakeStore({KEY: BODY}, errors={KEY: [MassiveTransportError("reset"), MassiveTransportError("reset")]})
    path = _client(store, sleeps=sleeps).download_day("trades", DAY, tmp_path)
    assert path.read_bytes() == BODY
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_the_condition_table_is_pulled_across_pages_with_the_key_in_a_header(env, tmp_path):
    from quantlab.acquisition.massive.client import CONDITIONS_URL, HttpResponse
    from quantlab.dataset.massive.raw import latest_conditions, read_conditions

    calls = []
    next_url = "https://api.massive.com/v3/reference/conditions?cursor=SYNTHETIC"

    def http_get(url, *, params, headers):
        calls.append((url, dict(params), dict(headers)))
        if url == CONDITIONS_URL:
            body = {"status": "OK", "results": CONDITION_RECORDS[:5], "next_url": next_url}
        else:
            body = {"status": "OK", "results": CONDITION_RECORDS[5:]}
        return HttpResponse(200, json.dumps(body).encode())

    path = _client(http_get=http_get).condition_table(tmp_path)
    assert path == latest_conditions(tmp_path / "massive")
    assert json.loads(path.read_text()) == CONDITION_RECORDS
    assert [url for url, _, _ in calls] == [CONDITIONS_URL, next_url]
    assert calls[0][1] == {"asset_class": "stocks", "limit": "1000"}
    assert all(headers == {"Authorization": "Bearer synthetic-api-key"} for _, _, headers in calls)
    assert all("synthetic-api-key" not in url + json.dumps(params) for url, params, _ in calls)
    assert 37 in read_conditions(path)["id"].to_list()


def test_a_refused_key_on_the_condition_table_says_so(env, tmp_path):
    from quantlab.acquisition.massive.client import (
        HttpResponse,
        MassiveEntitlementError,
    )

    def http_get(url, *, params, headers):
        return HttpResponse(401, b'{"status":"ERROR","error":"Unknown API Key"}')

    with pytest.raises(MassiveEntitlementError, match="Unknown API Key"):
        _client(http_get=http_get).condition_table(tmp_path)


def test_a_throttled_condition_pull_is_retried(env, tmp_path):
    from quantlab.acquisition.massive.client import HttpResponse

    replies = [HttpResponse(429, b""), HttpResponse(200, json.dumps({"results": CONDITION_RECORDS}).encode())]
    sleeps: list[float] = []
    _client(http_get=lambda url, *, params, headers: replies.pop(0), sleeps=sleeps).condition_table(tmp_path)
    assert len(sleeps) == 1


def test_a_missing_credential_fails_before_any_transfer_naming_the_variables(monkeypatch):
    from quantlab.acquisition.massive.client import (
        MassiveClient,
        MassiveCredentialError,
    )

    for name in ("MASSIVE_API_KEY", "MASSIVE_S3_ACCESS_KEY_ID", "MASSIVE_S3_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    store = FakeStore({KEY: BODY})
    with pytest.raises(MassiveCredentialError, match="MASSIVE_API_KEY.*MASSIVE_S3_ACCESS_KEY_ID"):
        MassiveClient(store=store)
    monkeypatch.setenv("MASSIVE_API_KEY", "synthetic-api-key")  # SYNTHETIC
    with pytest.raises(MassiveCredentialError, match="MASSIVE_S3_ACCESS_KEY_ID") as raised:
        MassiveClient(store=store)
    assert "MASSIVE_API_KEY" not in str(raised.value).split("set")[0]
    assert store.reads == []


def test_the_s3_secret_defaults_to_the_api_key(env, monkeypatch):
    from quantlab.acquisition.massive.client import MassiveCredentials

    credentials = MassiveCredentials.from_env()
    assert credentials.s3_secret_access_key == "synthetic-api-key"
    monkeypatch.setenv("MASSIVE_S3_SECRET_ACCESS_KEY", "synthetic-secret")  # SYNTHETIC
    assert MassiveCredentials.from_env().s3_secret_access_key == "synthetic-secret"
    assert "synthetic" not in repr(credentials)


# -- Several days: files at once, watermarks, outcomes ---------------------

#: The XNYS sessions of Mon 2024-11-25 .. Fri 2024-11-29: Thanksgiving (Thu 28th) is a holiday.
WEEK = [date(2024, 11, 25), date(2024, 11, 26), date(2024, 11, 27), date(2024, 11, 29)]


def _week_store(data_type: str = "trades", days=WEEK, **kwargs) -> FakeStore:
    return FakeStore({_key(day, data_type): gzip_text([TRADE_HEADER, f"A{day.day},,0,12,1,1,10.0,1,1,100,1,0,0"])
                      for day in days}, **kwargs)


def test_several_days_download_oldest_first_on_xnys_sessions_only(env, tmp_path):
    from quantlab.dataset.massive.raw import raw_days, read_watermark

    store = _week_store()
    days = list(_client(store, files=3, part_bytes=16).download_days(
        "trades", date(2024, 11, 23), date(2024, 12, 1), tmp_path))
    assert [d.day for d in days] == WEEK
    assert all(d.published and d.path.read_bytes() == store.objects[_key(d.day)] for d in days)
    assert raw_days(tmp_path / "massive", "trades") == WEEK
    assert read_watermark(tmp_path / "massive", "trades") == WEEK[-1]
    # Weekends and the holiday are not asked for.
    assert sorted(store.heads) == sorted(_key(day) for day in WEEK)


@pytest.mark.parametrize("data_type", ["minute_aggs", "day_aggs"])
def test_aggregate_files_download_through_the_same_path(env, tmp_path, data_type):
    from quantlab.dataset.massive.raw import raw_days, read_watermark

    store = _week_store(data_type)
    days = list(_client(store, part_bytes=16).download_days(data_type, WEEK[0], WEEK[-1], tmp_path))
    assert all(d.path.read_bytes() == store.objects[_key(d.day, data_type)] for d in days)
    assert raw_days(tmp_path / "massive", data_type) == WEEK
    assert read_watermark(tmp_path / "massive", data_type) == WEEK[-1]


def test_a_restart_resumes_after_the_watermark(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import read_watermark

    # The third day keeps failing: the run stops, the watermark stays on the second.
    store = _week_store(errors={_key(WEEK[2]): [MassiveTransportError("reset")] * 10})
    client = _client(store, files=1, retries=2)
    with pytest.raises(MassiveTransportError):
        list(client.download_days("trades", WEEK[0], WEEK[-1], tmp_path))
    assert read_watermark(tmp_path / "massive", "trades") == WEEK[1]

    healed = _week_store()
    days = list(_client(healed, files=1).download_days("trades", WEEK[0], WEEK[-1], tmp_path))
    assert [d.day for d in days] == WEEK[2:]
    assert not healed.fetched(_key(WEEK[0])) and not healed.fetched(_key(WEEK[1]))
    assert _key(WEEK[0]) not in healed.heads
    assert read_watermark(tmp_path / "massive", "trades") == WEEK[-1]


def test_the_watermark_is_per_data_type(env, tmp_path):
    from quantlab.dataset.massive.raw import read_watermark

    list(_client(_week_store()).download_days("trades", WEEK[0], WEEK[1], tmp_path))
    assert read_watermark(tmp_path / "massive", "trades") == WEEK[1]
    assert read_watermark(tmp_path / "massive", "minute_aggs") is None


def test_days_not_published_yet_do_not_move_the_watermark(env, tmp_path):
    from quantlab.dataset.massive.raw import read_watermark

    # Nothing after the 29th is published yet: a later run asks for those days again.
    store = _week_store()
    days = list(_client(store).download_days("trades", WEEK[3], date(2024, 12, 3), tmp_path))
    assert [(d.day, d.published) for d in days] == [(WEEK[3], True), (date(2024, 12, 2), False),
                                                    (date(2024, 12, 3), False)]
    assert read_watermark(tmp_path / "massive", "trades") == WEEK[3]
    later = _week_store()
    list(_client(later).download_days("trades", WEEK[3], date(2024, 12, 3), tmp_path))
    assert _key(date(2024, 12, 2)) in later.heads


def test_a_late_file_holds_the_watermark_and_is_asked_for_again(env, tmp_path):
    from quantlab.dataset.massive.raw import read_watermark

    # The 26th is missing on the first run; the days after it still download.
    first = _week_store(days=[WEEK[0], *WEEK[2:]])
    days = list(_client(first).download_days("trades", WEEK[0], WEEK[-1], tmp_path))
    assert [d.published for d in days] == [True, False, True, True]
    assert read_watermark(tmp_path / "massive", "trades") == WEEK[0]

    second = _week_store()
    days = list(_client(second).download_days("trades", WEEK[0], WEEK[-1], tmp_path))
    assert [d.day for d in days] == WEEK[1:]
    assert [d.fetched_bytes > 0 for d in days] == [True, False, False]
    assert read_watermark(tmp_path / "massive", "trades") == WEEK[-1]


def test_downloads_run_at_most_files_days_ahead_of_the_caller(env, tmp_path):
    import time

    # Eight sessions; the caller holds the first (converting it) and asks for no more.
    days = [date(2024, 12, d) for d in (2, 3, 4, 5, 6, 9, 10, 11)]
    store = _week_store(days=days)
    run = _client(store, files=2).download_days("trades", days[0], days[-1], tmp_path)
    first = next(run)
    time.sleep(0.3)
    # The day being converted and one more: never a week of raw trades on disk.
    assert first.day == days[0]
    assert sorted(set(store.heads)) == [_key(days[0]), _key(days[1])]
    assert [d.day for d in run] == days[1:]


def test_a_day_outside_the_plan_fails_the_run_saying_so(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveEntitlementError
    from quantlab.dataset.massive.raw import read_watermark

    store = _week_store(errors={_key(WEEK[1]): [MassiveEntitlementError("outside the plan's window")]})
    with pytest.raises(MassiveEntitlementError, match="plan"):
        list(_client(store, files=1).download_days("trades", WEEK[0], WEEK[-1], tmp_path))
    assert read_watermark(tmp_path / "massive", "trades") == WEEK[0]


# -- The S3 layer ----------------------------------------------------------


class FakeS3:
    """A boto3 S3 client stand-in answering from one object.

    ``errors`` (error codes) are raised by HEAD requests first, in order;
    ``get_errors`` likewise by ranged GETs, ``list_errors`` by listings.
    ``listed`` are the keys a listing returns, filtered by prefix.
    """

    def __init__(self, body: bytes, errors: list[str] | None = None, get_errors: list[str] | None = None,
                 listed: list[str] | None = None, list_errors: list[str] | None = None):
        self.body = body
        self.listed = listed or []
        self.errors = {"HeadObject": list(errors or []), "GetObject": list(get_errors or []),
                       "ListObjectsV2": list(list_errors or [])}
        self.ranges: list[str] = []
        self._lock = threading.Lock()

    def _raise(self, operation: str) -> None:
        from botocore.exceptions import ClientError

        with self._lock:
            code = self.errors[operation].pop(0) if self.errors[operation] else None
        if code is not None:
            raise ClientError({"Error": {"Code": code, "Message": code}}, operation)

    def head_object(self, Bucket, Key):
        self._raise("HeadObject")
        return {"ContentLength": len(self.body)}

    def list_objects_v2(self, Bucket, Prefix):
        self._raise("ListObjectsV2")
        return {"Contents": [{"Key": key} for key in self.listed if key.startswith(Prefix)]}

    def get_object(self, Bucket, Key, Range):
        import io

        from botocore.response import StreamingBody

        self._raise("GetObject")
        self.ranges.append(Range)
        first, last = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        part = self.body[first : last + 1]
        return {"Body": StreamingBody(io.BytesIO(part), len(part))}


def _s3_client(s3, sleeps, today=date(2024, 12, 2), **settings):
    from quantlab.acquisition.massive.client import MassiveCredentials, S3ObjectStore

    store = S3ObjectStore(MassiveCredentials("k", "i", "s"), client=s3, today=lambda: today)
    return _client(store, sleeps=sleeps, **settings)


def test_throttling_is_backed_off_and_retried(env, tmp_path):
    body = _big_body()
    s3 = FakeS3(body, errors=["SlowDown", "503"])
    sleeps: list[float] = []
    path = _s3_client(s3, sleeps, streams=2, part_bytes=2000).download_day("trades", DAY, tmp_path)
    assert path.read_bytes() == body
    assert sleeps == [2.0, 4.0]
    assert "bytes=0-1999" in s3.ranges


def test_a_throttled_range_is_backed_off_and_retried(env, tmp_path):
    body = _big_body()
    s3 = FakeS3(body, get_errors=["SlowDown", "SlowDown"])
    sleeps: list[float] = []
    path = _s3_client(s3, sleeps, streams=1, part_bytes=len(body)).download_day("trades", DAY, tmp_path)
    assert path.read_bytes() == body
    assert sleeps == [2.0, 4.0]


def test_s3_refusals_and_missing_keys_are_told_apart(env, tmp_path):
    from quantlab.acquisition.massive.client import (
        MassiveEntitlementError,
        MassiveNotPublishedError,
    )

    sleeps: list[float] = []
    with pytest.raises(MassiveNotPublishedError):
        _s3_client(FakeS3(BODY, ["NoSuchKey"]), sleeps).download_day("trades", DAY, tmp_path)
    with pytest.raises(MassiveEntitlementError, match="credentials"):
        _s3_client(FakeS3(BODY, ["403"], list_errors=["403"]), sleeps).download_day("trades", DAY, tmp_path)
    with pytest.raises(MassiveEntitlementError, match="credentials"):
        _s3_client(FakeS3(BODY, ["InvalidAccessKeyId"]), sleeps).download_day("trades", DAY, tmp_path)
    assert sleeps == []


# Massive answers 404 for a missing day inside the plan's window (a holiday), and 403 for
# every day outside it: before the oldest day, and after the newest published one (today
# before it is published, the future). A listing of the month tells the two ends apart.
MONTH = [_key(day) for day in (date(2024, 11, 25), date(2024, 11, 26), date(2024, 11, 27))]


def test_a_day_after_the_newest_published_is_not_published_yet(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveNotPublishedError

    s3 = FakeS3(BODY, ["403"], listed=MONTH)
    with pytest.raises(MassiveNotPublishedError, match="not published"):
        _s3_client(s3, [], today=date(2024, 11, 29)).download_day("trades", date(2024, 11, 29), tmp_path)


def test_a_recent_day_of_a_month_with_nothing_published_yet_is_not_published(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveNotPublishedError

    with pytest.raises(MassiveNotPublishedError):
        _s3_client(FakeS3(BODY, ["403"]), [], today=date(2024, 12, 3)).download_day(
            "trades", date(2024, 12, 2), tmp_path)


def test_a_day_before_the_oldest_published_is_outside_the_plan(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveEntitlementError

    with pytest.raises(MassiveEntitlementError, match="window"):
        _s3_client(FakeS3(BODY, ["403"], listed=MONTH), []).download_day("trades", date(2024, 11, 22), tmp_path)
    # A whole month before the window lists nothing.
    with pytest.raises(MassiveEntitlementError, match="window"):
        _s3_client(FakeS3(BODY, ["403"]), [], today=date(2026, 10, 9)).download_day(
            "trades", date(2016, 9, 30), tmp_path)


def test_a_run_up_to_today_stops_at_the_newest_published_day(env, tmp_path):
    from quantlab.dataset.massive.raw import read_watermark

    published = {_key(day): gzip_text([TRADE_HEADER, "A,,0,12,1,1,10.0,1,1,100,1,0,0"]) for day in WEEK[:3]}

    class Store(FakeS3):
        def head_object(self, Bucket, Key):
            if Key not in published:
                from botocore.exceptions import ClientError

                raise ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject")
            self.body = published[Key]
            return {"ContentLength": len(self.body)}

    s3 = Store(b"", listed=list(published))
    days = list(_s3_client(s3, [], today=date(2024, 11, 29), files=1).download_days(
        "trades", WEEK[0], date(2024, 11, 29), tmp_path))
    assert [(d.day, d.published) for d in days] == [(WEEK[0], True), (WEEK[1], True), (WEEK[2], True),
                                                    (WEEK[3], False)]
    assert read_watermark(tmp_path / "massive", "trades") == WEEK[2]


def test_concurrency_settings_are_checked(env):
    with pytest.raises(ValueError, match="streams"):
        _client(FakeStore({}), streams=0)


def test_a_stalled_stream_times_out_soon_and_its_resume_is_logged(env, tmp_path):
    from loguru import logger

    from quantlab.acquisition.massive.client import (
        MassiveClient,
        MassiveCredentials,
        S3ObjectStore,
    )

    # A stream that stops sending is given up after read_timeout seconds, not five minutes.
    store = S3ObjectStore(MassiveCredentials("k", "i", "s"), read_timeout=30)
    assert store._client.meta.config.read_timeout == 30
    assert MassiveClient(read_timeout=12).store._client.meta.config.read_timeout == 12

    # A break after some bytes resumes at once, and says so: stalls are counted in the log.
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        body = _big_body()
        store = FakeStore({KEY: body}, breaks={KEY: [300]})
        _client(store, streams=1, part_bytes=len(body)).download_day("trades", DAY, tmp_path)
    finally:
        logger.remove(sink)
    assert any("resuming at byte 300" in message for message in messages)
