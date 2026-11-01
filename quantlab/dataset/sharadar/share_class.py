"""The security whose filings carry each Sharadar security's firm values, keyed by permaticker.

A firm with several traded share classes (Alphabet's GOOGL and GOOG,
Berkshire's BRK.B and BRK.A) has one SF1 security, the *primary* class:
its DAILY ``marketcap`` is the firm's total market cap (GOOGL's $2,251B on
2024-06-28 is every share of both classes) and its SF1 rows the firm's
fundamentals. A *secondary* class (TICKERS ``category`` ending in
"Secondary Class") has SEP rows only.

``SharadarShareClassDataset`` stores one float variable, ``firm``: the
permaticker whose DAILY and SF1 rows hold the security's firm values.

- A security that is not a secondary class is its own firm.
- A secondary class's firm is the SF1 security with the same SEC CIK (the
  ``CIK=`` of TICKERS ``secfilings``) that is priced on the day, between
  its TICKERS ``firstpricedate`` and ``lastpricedate``. A CIK can name
  successive issuers (a company restructured into a new security), so the
  firm can change from one day to the next.
- The firm is NaN on a day where no such SF1 security is priced, on a day
  where two or more are, and for a secondary class without a CIK. Each day
  is decided from that day's issuers alone, so the panel holds no
  look-ahead.

On the 2026-10-05 pull, 1,334 of the 1,338 domestic secondary classes have
exactly one SF1 security with their CIK; three have no CIK, and one
(OSGB) has two, priced in turn. The CIK was checked against TICKERS
``relatedtickers`` on GOOG/GOOGL, BRK.A/BRK.B and FOX/FOXA (issue #186).

Like any spell panel (``quantlab.dataset.sharadar.spells``), the value is
shown from the security's first to its last price date, on SEP's trading
days, and the store grows with ``update()``.

Examples
--------
Build the store and read a year::

    config = SharadarShareClassConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_share_class_1d/sharadar_share_class_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarShareClassDataset(config).update()
    panel = SharadarShareClassDataset(config).panel("2024-01-02", "2024-12-31")
    panel["firm"].sel(symbol=119496).values[0]  # GOOG -> 195146.0, GOOGL
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset.config import SharadarShareClassConfig
from quantlab.dataset.sharadar.spells import ERROR_SAMPLE, SpellPanelDataset
from quantlab.dataset.sharadar.tables import scan_raw_table, table
from quantlab.dataset.sharadar.universe import universe

#: Name of the panel's one variable.
VARIABLE = "firm"

#: TICKERS ``category`` suffix of a secondary share class.
SECONDARY_CLASS = "Secondary Class"

#: The SEC CIK of a TICKERS row, from its ``secfilings`` URL.
_CIK = pl.col("secfilings").str.extract(r"CIK=(\d+)").alias("cik")

#: Stand-ins for an issuer's unknown first or last price date: priced throughout.
_EARLIEST = pl.date(1, 1, 1)
_LATEST = pl.date(9999, 12, 31)


class SharadarShareClassDataset(SpellPanelDataset):
    """Daily panel of the permaticker holding each security's firm values.

    Parameters
    ----------
    config : SharadarShareClassConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``tickers`` and ``sep`` (the calendar) raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarShareClassDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)  # ['firm']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarShareClassConfig
    VARIABLE = VARIABLE

    def _variable_attrs(self) -> dict:
        """Return what the variable holds."""
        return {"description": "permaticker whose DAILY and SF1 rows hold the firm's values"}

    def _securities(self) -> pl.DataFrame:
        """Return each kept SEP security's category, CIK and price dates.

        Raises
        ------
        ValueError
            If a permaticker has two SEP TICKERS rows.
        """
        securities = (
            scan_raw_table(self.config.raw_data_dir_path, "tickers")
            .filter(
                pl.col("table").is_in(table("sep").tickers_labels)
                & pl.col("permaticker").is_in(universe(self.config))
            )
            .select(
                "permaticker", "category", "firstpricedate", "lastpricedate",
                _CIK,
            )
            .unique()
            .collect()
        )
        twice = securities.filter(pl.col("permaticker").is_duplicated())
        if twice.height:
            raise ValueError(
                f"{self.class_name}: permaticker(s) "
                f"{twice.get_column('permaticker').unique().sort().head(ERROR_SAMPLE).to_list()} "
                f"have several SEP rows in TICKERS; re-pull TICKERS."
            )
        return securities

    def _issuers(self) -> pl.DataFrame:
        """Return every SF1 security with a CIK: ``cik``, ``firm`` and its price dates."""
        return (
            scan_raw_table(self.config.raw_data_dir_path, "tickers")
            .filter(pl.col("table").is_in(table("sf1").tickers_labels))
            .select(
                _CIK,
                pl.col("permaticker").alias("firm"),
                pl.col("firstpricedate").alias("issuer_first"),
                pl.col("lastpricedate").alias("issuer_last"),
            )
            .filter(pl.col("cik").is_not_null())
            .unique()
            .collect()
        )

    def _build_spells(self) -> pl.DataFrame:
        """Return every security's firm spells.

        A security that is not a secondary class has one open spell of
        itself. A secondary class's timeline is cut at every start and end
        of the SF1 securities with its CIK; a piece covered by exactly one
        of them is a spell of that security, and a piece covered by two or
        more has no firm (it is ambiguous on those days only).
        """
        securities = self._securities().with_columns(
            pl.col("category").fill_null("").str.ends_with(SECONDARY_CLASS).alias("secondary")
        )
        own = securities.filter(~pl.col("secondary")).select(
            pl.col("permaticker").alias("symbol"),
            pl.lit(None, dtype=pl.Date).alias("start"),
            pl.lit(None, dtype=pl.Date).alias("end"),
            pl.col("firstpricedate").alias("first"),
            pl.col("lastpricedate").alias("last"),
            pl.col("permaticker").cast(pl.Float64).alias(VARIABLE),
        )
        # Each issuer's priced span as [low, high), an open end at a stand-in;
        # ``issuer_last`` is a priced day, so the span runs through it.
        issuers = securities.filter(pl.col("secondary")).join(
            self._issuers(), on="cik", how="inner"
        ).select(
            "permaticker",
            "firm",
            pl.col("issuer_first").fill_null(_EARLIEST).alias("low"),
            pl.col("issuer_last").dt.offset_by("1d").fill_null(_LATEST).alias("high"),
        )
        # The pieces between consecutive boundaries of one secondary class.
        bounds = (
            pl.concat([
                issuers.select("permaticker", pl.col("low").alias("at")),
                issuers.select("permaticker", pl.col("high").alias("at")),
            ])
            .unique()
            .sort("permaticker", "at")
        )
        pieces = bounds.select(
            "permaticker",
            pl.col("at").alias("start"),
            pl.col("at").shift(-1).over("permaticker").alias("end"),
        ).drop_nulls("end")
        covered = (
            pieces.join(issuers, on="permaticker")
            .filter((pl.col("low") <= pl.col("start")) & (pl.col("high") >= pl.col("end")))
            .group_by("permaticker", "start", "end")
            .agg(pl.len().alias("issuers"), pl.col("firm").first())
        )
        secondary = covered.join(
            securities.select("permaticker", "firstpricedate", "lastpricedate"), on="permaticker"
        ).select(
            pl.col("permaticker").alias("symbol"),
            "start",
            "end",
            pl.col("firstpricedate").alias("first"),
            pl.col("lastpricedate").alias("last"),
            pl.when(pl.col("issuers") == 1)
            .then(pl.col("firm").cast(pl.Float64))
            .otherwise(float("nan"))
            .alias(VARIABLE),
        )
        # A secondary class with no spell still belongs on the axis: an empty
        # spell keeps it there, NaN on every day.
        unmatched = securities.filter(pl.col("secondary")).join(
            secondary.select(pl.col("symbol").alias("permaticker")), on="permaticker", how="anti"
        ).select(
            pl.col("permaticker").alias("symbol"),
            pl.col("firstpricedate").alias("start"),
            pl.col("firstpricedate").alias("end"),
            pl.col("firstpricedate").alias("first"),
            pl.col("lastpricedate").alias("last"),
            pl.lit(float("nan")).alias(VARIABLE),
        )
        return pl.concat([own, secondary, unmatched], how="vertical_relaxed")
