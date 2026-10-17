"""Keep Massive's REST raw tier current: every endpoint, each from its watermark.

Each endpoint (``quantlab.acquisition.massive.rest.ENDPOINTS``) is pulled by
``MassiveClient.pull``: pages from the day after its watermark, written as the
vendor returned them under ``<download-dir>/massive/rest/<endpoint>/``, the
watermark moved after every page. A first refresh backfills the whole history,
a later one asks only for what is new, and an interrupted one resumes where it
stopped. Every endpoint is tried even when one fails.

The daily trade, minute and day aggregate files are not refreshed here: their
download runs beside the Trade bar conversion (``scripts/massive/convert.py
--download``), which converts each day while the next downloads, so a refresh
that downloaded them all first would hold every file on disk before
converting any. ``refresh_raw_tier`` is registered on ``MASSIVE_SOURCE``.

Examples
--------
Needs ``MASSIVE_API_KEY``, ``MASSIVE_S3_ACCESS_KEY_ID`` and the network::

    from quantlab.acquisition.registry import refresh_raw_tier
    from quantlab.acquisition.base import DataSourceRegistry

    result = refresh_raw_tier(DataSourceRegistry.get("massive"), "/data/quantlab/downloads")
    result.failed  # the endpoints not refreshed
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from quantlab.acquisition.massive.client import (
    MassiveClient,
    MassiveEntitlementError,
    MassiveTransportError,
)
from quantlab.acquisition.massive.rest import ENDPOINTS
from quantlab.acquisition.refresh import RawTierRefreshResult, RefreshedPart


def refresh_raw_tier(
    download_dir: str | Path,
    *,
    end: date | None = None,
    endpoints: Iterable[str] | None = None,
    client: MassiveClient | None = None,
    today: date | None = None,
) -> RawTierRefreshResult:
    """Pull the REST endpoints into the raw tier under ``<download_dir>/massive/rest/``.

    Parameters
    ----------
    download_dir : str or Path
        The download root.
    end : date or None, default None
        The last day asked for (a backfill in steps); no bound when omitted.
    endpoints : iterable of str or None, default None
        The endpoints to pull, by name; every one of ``ENDPOINTS`` when
        omitted, in its order.
    client : MassiveClient or None, default None
        The client to pull through; one reading the credentials from the
        environment when omitted.
    today : date or None, default None
        Today in New York, passed to ``MassiveClient.pull``; the system
        clock's when omitted.

    Returns
    -------
    RawTierRefreshResult
        One part per endpoint: the rows and pages pulled and the watermark
        after, or why it failed (a refusal, a transport error past the
        retries, a page the client cannot read).

    Raises
    ------
    ValueError
        If an endpoint is not in ``ENDPOINTS``; nothing is requested.
    MassiveCredentialError
        If ``client`` is omitted and a credential variable is unset.

    Examples
    --------
    Needs the credentials and the network::

        refresh_raw_tier("/data/quantlab/downloads", endpoints=("dividends",), end=date(2010, 12, 31))
    """
    names = list(ENDPOINTS) if endpoints is None else list(endpoints)
    if unknown := [name for name in names if name not in ENDPOINTS]:
        raise ValueError(f"unknown endpoint(s) {unknown}; expected some of {list(ENDPOINTS)}")
    client = client if client is not None else MassiveClient()
    parts = []
    for name in names:
        started = time.monotonic()
        try:
            done = client.pull(name, download_dir, end=end, today=today)
        except (MassiveEntitlementError, MassiveTransportError, ValueError) as error:
            parts.append(RefreshedPart.failure(name, str(error)))
            continue
        parts.append(RefreshedPart(
            name,
            f"{done.rows:,} row(s) in {done.pages} page(s), through {done.through}; "
            f"{time.monotonic() - started:.0f}s",
        ))
    return RawTierRefreshResult("massive", tuple(parts))
