"""The shared HTTP GET of the REST-style vendor clients (`quantlab.acquisition._support.http`).

`requests.get` is replaced inside the module, so nothing here opens a
connection.
"""

from __future__ import annotations

import pytest
import requests


class _FakeRequestsResponse:
    """What `requests.get(..., stream=True)` returns, reduced to what `http_get` reads."""

    def __init__(self, status, chunks, headers, broken=False):
        self.status_code = status
        self.headers = headers
        self._chunks = chunks
        self._broken = broken
        self.closed = False

    def iter_content(self, chunk_size):
        yield from self._chunks
        if self._broken:
            raise requests.exceptions.ChunkedEncodingError("Connection broken (SYNTHETIC)")

    def close(self):
        self.closed = True


def _patch_get(monkeypatch, answer):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("quantlab.acquisition._support.http.requests.get", fake_get)
    return calls


def test_the_get_streams_the_body_and_passes_timeout_headers_and_the_redirect_rule(monkeypatch):
    from quantlab.acquisition._support.http import http_get

    raw = _FakeRequestsResponse(206, [b"ab", b"cd"], {"Content-Range": "bytes 0-3/9"})
    calls = _patch_get(monkeypatch, raw)

    response = http_get(
        "https://h.invalid/p", params={"a": "1"}, headers={"x-api-key": "k"}, timeout=(3, 4)
    )

    assert response.status == 206
    assert response.headers["Content-Range"] == "bytes 0-3/9"
    assert response.read() == b"abcd"
    assert raw.closed
    url, kwargs = calls[0]
    assert url == "https://h.invalid/p"
    assert kwargs == {
        "params": {"a": "1"},
        "headers": {"x-api-key": "k"},
        "stream": True,
        "allow_redirects": False,
        "timeout": (3, 4),
    }


def test_the_get_follows_redirects_only_when_asked(monkeypatch):
    from quantlab.acquisition._support.http import http_get

    calls = _patch_get(monkeypatch, _FakeRequestsResponse(200, [], {}))
    http_get("https://h.invalid/p", params={}, headers={}, timeout=(1, 1), follow_redirects=True)
    assert calls[0][1]["allow_redirects"] is True


def test_a_connection_that_fails_to_open_raises_the_clients_error_without_the_query(monkeypatch):
    from quantlab.acquisition._support.http import http_get
    from quantlab.acquisition._support.request_policy import TransientError

    class VendorError(TransientError):
        pass

    _patch_get(monkeypatch, requests.exceptions.ConnectTimeout("timed out ?Signature=SECRET"))
    with pytest.raises(VendorError, match=r"ConnectTimeout on https://h.invalid/p$") as raised:
        http_get("https://h.invalid/p?Signature=SECRET", params={}, headers={}, timeout=(1, 1),
                 error=VendorError)
    assert "SECRET" not in str(raised.value)


def test_a_body_that_breaks_raises_the_clients_error_while_it_is_read(monkeypatch):
    from quantlab.acquisition._support.http import http_get
    from quantlab.acquisition._support.request_policy import TransientError

    _patch_get(monkeypatch, _FakeRequestsResponse(200, [b"ab"], {}, broken=True))
    response = http_get("https://h.invalid/p", params={}, headers={}, timeout=(1, 1))
    chunks = iter(response.body)
    assert next(chunks) == b"ab"
    with pytest.raises(TransientError, match="ChunkedEncodingError"):
        next(chunks)


def test_a_response_built_in_memory_keeps_whole_bytes_as_one_chunk():
    from quantlab.acquisition._support.http import HttpResponse

    assert list(HttpResponse(200, b"abc").body) == [b"abc"]
    assert HttpResponse(200, iter([b"a", b"b"])).read() == b"ab"
    assert HttpResponse(404).read() == b""
    assert HttpResponse(429, b"", {"Retry-After": "7"}).headers["Retry-After"] == "7"
