"""Keep the Sharadar raw tier current: the daily table plan.

Each table is pulled the way its updates arrive:

- ``WHOLE_TABLES`` whole, in bulk: TICKERS (new listings and ticker
  changes), INDICATORS (units and the 8-K event codes), SP500 and SF3A (small;
  SF3A's newest quarter fills in place) and ACTIONS (the vendor records an
  action under its effective date, which can be weeks before it publishes it,
  so a date window would never see it);
- ``WINDOW_TABLES`` as a trailing date window over REST, from the
  ``trading_days``-th most recent raw date before each table's watermark
  through ``end``, so a late vendor correction to a recent day is seen;
- ``UPDATED_TABLES`` as every row the vendor changed since the watermark (by
  ``lastupdated``), or whole in bulk when that query fails: Sharadar stops a
  query after 15 seconds, and after a mass re-stamp of ``lastupdated`` the bulk
  zip is the cheaper copy of the changed rows.

SF3 and SF3B feed no store and are not refreshed; ``bulk_table`` pulls them.
``refresh_raw_tier`` is registered on ``SHARADAR_SOURCE``; the stores are
brought up to date afterwards by ``quantlab.dataset.sharadar.stores``.

Examples
--------
Needs ``SHARADAR_API_KEY`` and the network::

    from quantlab.acquisition.registry import refresh_raw_tier
    from quantlab.acquisition.base import DataSourceRegistry

    result = refresh_raw_tier(DataSourceRegistry.get("sharadar"), "/data/quantlab/downloads")
    result.summary()
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from loguru import logger

from quantlab.acquisition.refresh import RawTierRefreshResult, RefreshedPart
from quantlab.acquisition.sharadar.client import SharadarClient, SharadarHttpError

#: Pulled whole every run: reference and membership tables are small.
WHOLE_TABLES = ("tickers", "indicators", "sp500", "sf3a", "actions")
#: Pulled as a trailing date window.
WINDOW_TABLES = ("sep", "sfp", "events", "sf2")
#: Pulled as the rows changed since the watermark (``lastupdated``).
UPDATED_TABLES = ("sf1", "daily")


def refresh_raw_tier(
    download_dir: str | Path,
    *,
    end: date | None = None,
    trading_days: int = 10,
    client: SharadarClient | None = None,
) -> RawTierRefreshResult:
    """Pull the table plan into the raw tier under ``<download_dir>/sharadar/``.

    The tables are pulled in the order ``WHOLE_TABLES``, ``WINDOW_TABLES``,
    ``UPDATED_TABLES``. A failure stops the refresh at that table (the
    tables after it keep their watermarks), except a failed updated pull,
    which is retried as a bulk pull.

    Parameters
    ----------
    download_dir : str or Path
        The download root, holding a bulk pull of every table already.
    end : date or None, default None
        The last day of the window pulls; ``None`` is today in Sharadar's
        time zone (US/Eastern).
    trading_days : int, default 10
        Raw dates re-pulled before each window table's watermark.
    client : SharadarClient or None, default None
        The client to pull through; one reading ``SHARADAR_API_KEY`` when
        omitted.

    Returns
    -------
    RawTierRefreshResult
        One part per table; its summary is the raw file written.

    Raises
    ------
    SharadarEntitlementError
        If the key is missing or refused.
    SharadarHttpError
        If a whole or window pull failed, or the bulk fallback of an updated
        pull.
    ValueError
        If a table has no bulk pull yet, or the vendor's CSV is not its schema.

    Examples
    --------
    Needs the key and the network::

        refresh_raw_tier("/data/quantlab/downloads", trading_days=10).failed  # ()
    """
    client = client if client is not None else SharadarClient()
    parts = []
    for code in WHOLE_TABLES:
        parts.append(RefreshedPart(code, str(client.bulk_table(code, download_dir))))
    for code in WINDOW_TABLES:
        path = client.window_table(code, download_dir, through=end, trading_days=trading_days)
        parts.append(RefreshedPart(code, str(path)))
    for code in UPDATED_TABLES:
        try:
            parts.append(RefreshedPart(code, str(client.updated_table(code, download_dir))))
        except SharadarHttpError as exc:
            logger.warning(f"Sharadar {code}: updated pull failed ({exc}); pulling it in bulk instead")
            path = client.bulk_table(code, download_dir)
            parts.append(RefreshedPart(code, f"updated pull failed ({exc}); pulled in bulk: {path}"))
    return RawTierRefreshResult("sharadar", tuple(parts))
