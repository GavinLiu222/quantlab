"""Point-in-time panel of Sharadar fundamentals (SF1), keyed by permaticker.

SF1 holds one row per company, *dimension* and filing. Only the as-reported
dimensions are point-in-time: ARQ (each fiscal quarter) and ART (trailing
twelve months). Their ``date`` is the SEC filing date, the row's *release
date*, and a later filing that restates a period is a new row rather than a
rewrite of the old one. The most-recent dimensions (MRQ, MRY, MRT) are dated
at the period end and rewritten on restatement, so they would leak a value
one to three months early; they stay in the raw tier and never enter a store.

``SharadarFundamentalsDataset`` places each row of one as-reported dimension
at its *available date*, the first SEP trading day on or after its release
date, and uses only information available at the time: on
every trading day it shows, per security, the row with the latest fiscal
period among the rows available by that day, and within that period the
latest filing. So:

- a restatement of a period is shown from its own release date, never
  earlier;
- a later filing of an older period never replaces a newer period's row;
- a row no newer row has replaced stops being shown ``stale_after_days``
  after its release date (365 by default, as for Compustat).

The panel's calendar is SEP's trading days, so it lines up with the price
panels bar for bar, and its ``symbol`` axis is the int64 permaticker, as
theirs is. Each of SF1's 105 indicators is a float64 variable whose
``unit`` attribute is the vendor's unit type from INDICATORS (``currency``
is the reporting currency, ``USD``, ``ratio``, ``units``, per share);
``release_date`` and ``reportperiod`` (the fiscal period end) say which row
is shown. The store grows with ``update()``: new trading days are appended
and a stored day is never rewritten, so a value the vendor changes later
(SF1 rows are refreshed by ``lastupdated``, ``SharadarClient.updated_table``)
reaches only the days after the update.

Examples
--------
Build the quarterly store from a downloaded raw tier and read a quarter::

    config = SharadarFundamentalsConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_sf1_arq/sharadar_sf1_arq.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarFundamentalsDataset(config).update()
    panel = SharadarFundamentalsDataset(config).panel("2024-01-02", "2024-03-28")
    panel["revenue"].attrs["unit"]  # 'currency'

SF1 also holds DAILY's valuation indicators, as of each filing. Merged with
the DAILY panel (``MergedDataset``), they are renamed ``sf1_marketcap``,
``sf1_ev``, ``sf1_pe`` and so on (``COLUMN_MAP``), so ``marketcap`` is
DAILY's daily value.
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset.config import DatasetConfig, SharadarFundamentalsConfig
from quantlab.dataset.sharadar.permatickers import map_raw_table
from quantlab.dataset.sharadar.point_in_time import ReleasePanelDataset, ReleaseSpec
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.dataset.sharadar.tables import (
    DAILY_INDICATORS,
    SF1_INDICATORS,
    scan_raw_table,
    table,
)
from quantlab.utils.timer import Timer

#: The dimensions a store may hold: as reported, quarterly or trailing twelve months.
AS_REPORTED: tuple[str, ...] = ("ARQ", "ART")

#: The date variables saying which row a cell shows.
DATE_VARIABLES: tuple[str, ...] = ("release_date", "reportperiod")

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


class SharadarFundamentalsDataset(ReleasePanelDataset):
    """Daily point-in-time panel of one as-reported SF1 dimension, keyed by permaticker.

    Parameters
    ----------
    config : SharadarFundamentalsConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``sf1``, ``sep`` (the calendar), ``tickers`` and ``indicators`` raw
        tables; ``dimension`` is ``"ARQ"`` or ``"ART"``.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarFundamentalsDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)[:3]
        # ['accoci', 'assets', 'assetsavg']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarFundamentalsConfig

    #: SF1 and SEP, the calendar, must be complete through a day for it to be built.
    RAW_TABLES = ("sf1", "sep")

    #: SF1's valuation indicators that DAILY also holds (``marketcap``,
    #: ``ev``, ``pe``, ...) go by ``sf1_<name>`` when merged or exported, so
    #: a merge with the DAILY panel keeps DAILY's daily values under the
    #: plain names. The store and ``panel()`` keep SF1's own names.
    COLUMN_MAP = {
        name: f"sf1_{name}" for name in DAILY_INDICATORS if name in SF1_INDICATORS
    }

    def _normalize_config(self, config: DatasetConfig) -> SharadarFundamentalsConfig:
        """Normalise as the base class does, then check the SF1 fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarFundamentalsConfig``.
        ValueError
            If ``table`` is not ``"sf1"``, ``dimension`` is not as reported,
            ``stale_after_days`` is below 1, or a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``).

        Examples
        --------
        A most-recent dimension is refused::

            SharadarFundamentalsDataset(dataclasses.replace(config, dimension="MRQ"))
            # ValueError: SharadarFundamentalsDataset: dimension 'MRQ' is not as reported ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarFundamentalsConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarFundamentalsConfig, got "
                f"{type(config).__name__}."
            )
        if config.table != "sf1":
            raise ValueError(f"{self.class_name}: table must be 'sf1', got {config.table!r}.")
        if config.dimension not in AS_REPORTED:
            raise ValueError(
                f"{self.class_name}: dimension {config.dimension!r} is not as "
                f"reported; a store holds {' or '.join(AS_REPORTED)}. The "
                f"most-recent dimensions are dated at the period end, not the "
                f"release date, and would leak values early."
            )
        if config.stale_after_days is not None and config.stale_after_days < 1:
            raise ValueError(
                f"{self.class_name}: stale_after_days must be at least 1 or "
                f"None, got {config.stale_after_days}."
            )
        return normalize_universe(config, self.class_name)

    # -- the release table ------------------------------------------------------

    def _released(self) -> pl.DataFrame:
        """Return the rows the panel ever shows, each at its available date.

        A row is kept when its fiscal period is the latest of its
        permaticker's rows released by then, so a later filing of an older
        period is dropped. Of several rows reaching one available date, the
        last filed (and so the latest period) is kept.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` (a trading day), the ``DATE_VARIABLES``
            and ``SF1_INDICATORS``, sorted by ``available``.

        Raises
        ------
        ValueError
            If a raw ticker has two permatickers, or two rows of one ticker
            and permaticker share a release date and fiscal period (rows
            under two tickers are settled, ``PermatickerResolver.settle``).
        """
        config = self.config
        root = config.raw_data_dir_path
        calendar = self._trading_calendar()
        with Timer(f"{self.class_name}: shown rows"):
            frame = map_raw_table(
                root,
                "sf1",
                owner=self.class_name,
                store_path=config.zarr_file_path,
                query=lambda scan: scan.filter(pl.col("dimension") == config.dimension).filter(
                    pl.col("date") <= pl.lit(calendar.through)
                ),
                key=("dimension", "date", "reportperiod"),
            )
            frame = frame.filter(pl.col("permaticker").is_in(universe(config)))
            self._assert_unique_keys(frame)
            frame = frame.sort("permaticker", "date", "reportperiod").filter(
                pl.col("reportperiod")
                == pl.col("reportperiod").cum_max().over("permaticker")
            )
            return (
                calendar.place(frame)
                # A join does not keep row order; the last filing must be last.
                .sort("permaticker", "date", "reportperiod")
                .unique(subset=["permaticker", "available"], keep="last", maintain_order=True)
                .select(
                    pl.col("permaticker").alias("symbol"),
                    "available",
                    pl.col("date").cast(pl.Datetime("ns")).alias("release_date"),
                    pl.col("reportperiod").cast(pl.Datetime("ns")),
                    *(pl.col(name).cast(pl.Float64) for name in SF1_INDICATORS),
                )
                .sort("available")
            )

    def _spec(self) -> ReleaseSpec:
        """Return the indicators (each with its vendor unit) and the date variables."""
        units = self._units()
        variables = {
            name: ({"unit": units[name]} if name in units else {})
            for name in (*SF1_INDICATORS, *DATE_VARIABLES)
        }
        variables["release_date"]["description"] = "SEC filing date of the row shown"
        variables["reportperiod"]["description"] = "fiscal period end of the row shown"
        return ReleaseSpec(variables=variables, dates=DATE_VARIABLES)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return f"no {self.config.dimension} row is shown"

    def _assert_unique_keys(self, frame: pl.DataFrame) -> None:
        """Raise if two rows of one permaticker share a release date and fiscal period."""
        duplicates = (
            frame.group_by("permaticker", "date", "reportperiod")
            .agg(pl.col("ticker").sort())
            .filter(pl.col("ticker").list.len() > 1)
            .sort("permaticker", "date")
        )
        if duplicates.height:
            sample = [
                f"{row['permaticker']}@{row['date']}/{row['reportperiod']}:{row['ticker']}"
                for row in duplicates.head(_ERROR_SAMPLE).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {duplicates.height} (permaticker, release "
                f"date, fiscal period) key(s) have several raw rows, first "
                f"{sample}; refusing rather than choosing one."
            )

    def _units(self) -> dict[str, str]:
        """Return each SF1 indicator's unit type from the INDICATORS table."""
        rows = (
            scan_raw_table(self.config.raw_data_dir_path, "indicators")
            .filter(pl.col("table").is_in(table("sf1").tickers_labels))
            .select("indicator", "unittype")
            .collect()
        )
        return dict(rows.iter_rows())
