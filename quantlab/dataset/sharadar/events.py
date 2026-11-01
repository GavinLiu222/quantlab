"""Sharadar 8-K events (EVENTS) as a daily boolean panel, keyed by permaticker.

EVENTS holds one row per company and 8-K filing date, with the filing's
items as a pipe-joined list of two-digit event codes (``22|91``: results of
operations, and financial statements). The codes are Sharadar's published
list, the ``EVENTCODES`` rows of INDICATORS.

``SharadarEventsDataset`` has one boolean variable per published code,
``event_<code>``, whose ``title`` attribute is the code's title. A cell is
True on the available date of a filing that lists the code (the first SEP
trading day on or after the filing date) and False on every other day
(``quantlab.dataset.sharadar.filings.FilingPanelDataset``). A code missing
from the list is refused rather than dropped: re-pull INDICATORS.

TICKERS has no rows of EVENTS, which covers SF1's filers, so its tickers are
mapped through the SF1 rows. EVENTS has no ``lastupdated``; it is refreshed
by trailing date windows (``SharadarClient.window_table``), and ``update()``
appends the new trading days. A code the vendor publishes after the store
was built becomes a new variable on the next update, False on the days
already stored: no filing could list it then.

Examples
--------
Build the store and see which companies reported results on a day::

    config = SharadarEventsConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_events_1d/sharadar_events_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarEventsDataset(config).update()
    panel = SharadarEventsDataset(config).panel("2024-01-02", "2024-12-31")
    panel["event_22"].attrs["title"]  # 'Results of Operations and Financial Condition'
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset.config import SharadarEventsConfig
from quantlab.dataset.sharadar.filings import FilingPanelDataset
from quantlab.dataset.sharadar.tables import scan_raw_table

#: The INDICATORS ``table`` value of the published event codes.
EVENTCODES_LABEL = "EVENTCODES"

#: Prefix of each code's variable name.
VARIABLE_PREFIX = "event_"


class SharadarEventsDataset(FilingPanelDataset):
    """Daily panel of 8-K event codes, True on each filing's available date.

    Parameters
    ----------
    config : SharadarEventsConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``events``, ``sep`` (the calendar), ``tickers`` and ``indicators``
        raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarEventsDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)[:2]
        # ['event_11', 'event_12']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarEventsConfig

    TABLE = "events"

    #: One row per company and filing date.
    KEY = ("date",)

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._codes_cache: dict[str, str] | None = None

    def _codes(self) -> dict[str, str]:
        """Return the published event codes and their titles, sorted by code (cached)."""
        if self._codes_cache is None:
            rows = (
                scan_raw_table(self.config.raw_data_dir_path, "indicators")
                .filter(pl.col("table") == EVENTCODES_LABEL)
                .select("indicator", "title")
                .sort("indicator")
                .collect()
            )
            if rows.height == 0:
                raise ValueError(
                    f"{self.class_name}: INDICATORS has no {EVENTCODES_LABEL} "
                    f"rows; re-pull INDICATORS."
                )
            self._codes_cache = dict(rows.iter_rows())
        return self._codes_cache

    def _empty(self) -> dict[str, object]:
        """Return False for every code's variable: no filing listed it."""
        return {f"{VARIABLE_PREFIX}{code}": False for code in self._codes()}

    def _attrs(self) -> dict[str, dict]:
        """Return each code variable's ``title``, from the published list."""
        return {f"{VARIABLE_PREFIX}{code}": {"title": title} for code, title in self._codes().items()}

    def _filings(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Return one row per filing with one boolean per published code.

        Raises
        ------
        ValueError
            If a filing lists a code the published list does not have.
        """
        codes = self._codes()
        frame = raw.select(
            "permaticker", "date", pl.col("eventcodes").fill_null("").str.split("|").alias("_codes")
        )
        listed = frame.get_column("_codes").explode().unique().drop_nulls().to_list()
        unknown = sorted(code for code in listed if code and code not in codes)
        if unknown:
            raise ValueError(
                f"{self.class_name}: EVENTS lists event code(s) {unknown} that "
                f"INDICATORS' {EVENTCODES_LABEL} rows do not publish; re-pull "
                f"INDICATORS."
            )
        return frame.select(
            "permaticker",
            "date",
            *(
                pl.col("_codes").list.contains(code).alias(f"{VARIABLE_PREFIX}{code}")
                for code in codes
            ),
        )

    def _combine(self) -> list[pl.Expr]:
        """Set a code on a day when any of the day's filings lists it."""
        return [pl.col(name).any() for name in self._empty()]
