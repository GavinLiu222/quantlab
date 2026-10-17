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

A table Sharadar fails (one of ``TABLE_FAILURES``: a refusal, an error
status or a connection failure past the retries, a file not the table's
schema) is recorded as failed (``RefreshedPart.failure``) and the tables after
it are still pulled; any other error, a bug of ours or a table never pulled in
bulk, is raised. A missing key raises before any table is asked for.

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

from quantlab.acquisition._support.request_policy import require_env
from quantlab.acquisition.refresh import RawTierRefreshResult, RefreshedPart
from quantlab.acquisition.sharadar.client import (
    API_KEY_ENV,
    SharadarClient,
    SharadarEntitlementError,
    SharadarHttpError,
    SharadarResponseError,
    SharadarTransportError,
)

#: Pulled whole every run: reference and membership tables are small.
WHOLE_TABLES = ("tickers", "indicators", "sp500", "sf3a", "actions")
#: Pulled as a trailing date window.
WINDOW_TABLES = ("sep", "sfp", "events", "sf2")
#: Pulled as the rows changed since the watermark (``lastupdated``).
UPDATED_TABLES = ("sf1", "daily")
#: The errors that mean Sharadar failed one table; the others are still pulled.
TABLE_FAILURES = (SharadarEntitlementError, SharadarHttpError, SharadarTransportError, SharadarResponseError)


def refresh_raw_tier(
    download_dir: str | Path,
    *,
    end: date | None = None,
    trading_days: int = 10,
    client: SharadarClient | None = None,
) -> RawTierRefreshResult:
    """Pull the table plan into the raw tier under ``<download_dir>/sharadar/``.

    The tables are pulled in the order ``WHOLE_TABLES``, ``WINDOW_TABLES``,
    ``UPDATED_TABLES``. A table that fails with one of ``TABLE_FAILURES`` is
    listed as failed and keeps its watermark, and the tables after it are
    still pulled; a failed updated pull (``SharadarHttpError``) is first
    retried as a bulk pull.

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
        One part per table: the raw file written, or why it failed.

    Raises
    ------
    SharadarEntitlementError
        If ``SHARADAR_API_KEY`` is unset; nothing is requested.
    ValueError
        If a table has no bulk pull yet; the tables before it are pulled.

    Examples
    --------
    Needs the key and the network::

        refresh_raw_tier("/data/quantlab/downloads", trading_days=10).failed  # ()
    """
    require_env(
        [API_KEY_ENV],
        hint="cannot refresh the Sharadar raw tier; set it in the environment. Nothing was requested.",
        error=SharadarEntitlementError,
    )
    client = client if client is not None else SharadarClient()

    def pull(code: str) -> RefreshedPart:
        try:
            if code in WHOLE_TABLES:
                return RefreshedPart(code, str(client.bulk_table(code, download_dir)))
            if code in WINDOW_TABLES:
                path = client.window_table(code, download_dir, through=end, trading_days=trading_days)
                return RefreshedPart(code, str(path))
            try:
                return RefreshedPart(code, str(client.updated_table(code, download_dir)))
            except SharadarHttpError as exc:
                logger.warning(f"Sharadar {code}: updated pull failed ({exc}); pulling it in bulk instead")
                path = client.bulk_table(code, download_dir)
                return RefreshedPart(code, f"updated pull failed ({exc}); pulled in bulk: {path}")
        except TABLE_FAILURES as exc:
            logger.warning(f"Sharadar {code}: not refreshed ({exc}); pulling the other tables")
            return RefreshedPart.failure(code, str(exc))

    parts = tuple(pull(code) for code in (*WHOLE_TABLES, *WINDOW_TABLES, *UPDATED_TABLES))
    return RawTierRefreshResult("sharadar", parts)
