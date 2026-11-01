"""Sharadar insider transactions (SF2) as a daily panel of net shares and value per transaction code, keyed by permaticker.

SF2 holds one row per security line of an insider's Form 3, 4 or 5. Its
``date`` is the SEC filing date, the day the transaction became public;
``transactiondate`` is earlier (two days at the median). A line is a
non-derivative transaction (``securityadcode`` ``NA`` acquired or ``ND``
disposed), a derivative one (``DA``, ``DD``), or a holding with no
transaction (no ``transactioncode``, ``N``/``D``), which never counts.

``SharadarInsidersDataset`` places each line at the first SEP trading day
on or after its filing date (never its transaction date) and sums a day's
lines per security (``quantlab.dataset.sharadar.filings.FilingPanelDataset``):

- ``net_shares``, ``net_value``: open-market trades in the stock, shares
  and USD bought (``P``) minus sold (``S``) on a non-derivative line. The
  vendor's ``transactionvalue`` is unsigned USD (INDICATORS calls it USD
  millions, but it equals shares times price), so a sale counts negative.
- ``net_shares_<code>``, ``net_value_<code>``: the non-derivative lines of
  one transaction code, ``P``, ``S``, ``A`` (grant), ``M`` (exercise or
  conversion), ``F`` (tax or exercise price paid in stock), ``G`` (gift),
  ``D`` (disposed to the issuer), and ``other`` for every remaining code.
  ``P`` and ``S`` keep their own sign, as in ``net_shares``; every other
  code counts positive when acquired (``NA``) and negative when disposed
  (``ND``). A line without a price adds nothing to the value.
- ``net_shares_derivative``: derivative units (options, units, warrants)
  acquired minus disposed, every code together.
- ``insiders_buying``, ``insiders_selling``: the distinct owners
  (``ownername``) with a non-derivative ``P`` or ``S`` line on the day.

A day without a transaction is 0. A transaction is counted on the first
filing that shows it. When an amendment (4/A) restates a filing, it repeats
the lines; a later line repeating one exactly (same insider, transaction
date, code, shares and price, ``TRADE_KEY``, on the same side of the
derivative split) is not counted again. The vendor also relabels the
superseded original ``RESTATED - <form>``, but only once the amendment
arrives, so the label is ignored: a live update, which saw the original as
a plain form on its filing date, and a rebuild of the same days give the
same panel. An amendment that changes a line's shares or price does not
repeat it, and is counted again on its own filing date.

TICKERS has rows of SF2 (``table`` ``SF2``), so its tickers are mapped
through them. SF2 has no ``lastupdated``; it is refreshed by trailing date
windows (``SharadarClient.window_table``), and ``update()`` appends the new
trading days.

Examples
--------
Build the store and read the net buying of a quarter::

    config = SharadarInsidersConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_insiders_1d/sharadar_insiders_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarInsidersDataset(config).update()
    panel = SharadarInsidersDataset(config).panel("2024-01-02", "2024-03-28")
    panel["net_value"].attrs["unit"]  # 'USD'
    panel["net_shares_M"].attrs["unit"]  # 'shares'
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset.config import SharadarInsidersConfig
from quantlab.dataset.sharadar.filings import FilingPanelDataset

#: Transaction codes of open-market trades, and the sign each counts with.
OPEN_MARKET: dict[str, int] = {"P": 1, "S": -1}

#: ``securityadcode`` values of a non-derivative line: acquired, disposed.
NON_DERIVATIVE: tuple[str, ...] = ("NA", "ND")

#: ``securityadcode`` values of a derivative line: acquired, disposed.
DERIVATIVE: tuple[str, ...] = ("DA", "DD")

#: The sign a line counts with by its ``securityadcode``.
ADCODE_SIGN: dict[str, int] = {"NA": 1, "ND": -1, "DA": 1, "DD": -1}

#: Transaction codes with variables of their own and what each means (SEC
#: Form 4 instructions); every other code is ``other``.
CODES: dict[str, str] = {
    "P": "open-market or private purchase",
    "S": "open-market or private sale",
    "A": "grant or award from the issuer",
    "M": "exercise or conversion of a derivative",
    "F": "exercise price or tax paid by delivering stock",
    "G": "gift",
    "D": "disposition to the issuer",
}

#: The per-code variable suffixes, in variable order.
BUCKETS: tuple[str, ...] = (*CODES, "other")

#: The columns that identify one trade across the filings that show it.
TRADE_KEY: tuple[str, ...] = (
    "permaticker", "ownername", "transactiondate", "transactioncode",
    "transactionshares", "transactionpricepershare",
)


class SharadarInsidersDataset(FilingPanelDataset):
    """Daily panel of insiders' transactions per code, on each filing's available date.

    Parameters
    ----------
    config : SharadarInsidersConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``sf2``, ``sep`` (the calendar) and ``tickers`` raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarInsidersDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)[:4]
        # ['net_shares', 'net_value', 'net_shares_P', 'net_value_P']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarInsidersConfig

    TABLE = "sf2"

    def _empty(self) -> dict[str, object]:
        """Return 0 for every variable: no transaction."""
        empty: dict[str, object] = {"net_shares": 0.0, "net_value": 0.0}
        for bucket in BUCKETS:
            empty[f"net_shares_{bucket}"] = 0.0
            empty[f"net_value_{bucket}"] = 0.0
        empty["net_shares_derivative"] = 0.0
        empty["insiders_buying"] = 0.0
        empty["insiders_selling"] = 0.0
        return empty

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's unit and meaning."""
        attrs = {
            "net_shares": {"unit": "shares", "description": "open-market shares bought minus sold"},
            "net_value": {"unit": "USD", "description": "open-market USD bought minus sold"},
        }
        others = "codes other than " + ", ".join(CODES)
        for bucket, meaning in {**CODES, "other": "every other code"}.items():
            codes = others if bucket == "other" else f"code {bucket}"
            attrs[f"net_shares_{bucket}"] = {
                "unit": "shares",
                "description": f"non-derivative shares acquired minus disposed, {codes} ({meaning})",
            }
            attrs[f"net_value_{bucket}"] = {
                "unit": "USD",
                "description": f"non-derivative USD acquired minus disposed, {codes} ({meaning})",
            }
        attrs["net_shares_derivative"] = {
            "unit": "shares",
            "description": "derivative units acquired minus disposed, every transaction code",
        }
        attrs["insiders_buying"] = {
            "unit": "insiders",
            "description": "distinct insiders with an open-market purchase (P) of the stock",
        }
        attrs["insiders_selling"] = {
            "unit": "insiders",
            "description": "distinct insiders with an open-market sale (S) of the stock",
        }
        return attrs

    def _filings(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Return every transaction line, signed, each from the first filing showing it.

        A holding line (no transaction code) is left out. Several identical
        lines of one filing are several transactions and all count; the
        same line in a later filing is a repeat and does not. Derivative and
        non-derivative lines are checked for repeats apart, so the two lines
        of an option exercise never hide each other.
        """
        code = pl.col("transactioncode")
        lines = raw.filter(
            code.is_not_null()
            & pl.col("securityadcode").is_in([*NON_DERIVATIVE, *DERIVATIVE])
        ).with_columns(pl.col("securityadcode").is_in(DERIVATIVE).alias("_derivative"))
        first_shown = pl.col("date") == pl.col("date").min().over([*TRADE_KEY, "_derivative"])
        lines = lines.filter(first_shown).with_columns(
            pl.when(pl.col("_derivative"))
            .then(pl.lit(None, dtype=pl.String))
            .when(code.is_in(list(CODES)))
            .then(code)
            .otherwise(pl.lit("other"))
            .alias("_bucket"),
            # P and S count by their own sign, as net_shares does; every other
            # code by whether the line acquired or disposed.
            pl.coalesce(
                code.replace_strict(OPEN_MARKET, default=None, return_dtype=pl.Int64),
                pl.col("securityadcode").replace_strict(
                    ADCODE_SIGN, default=None, return_dtype=pl.Int64
                ),
            ).alias("_sign"),
        )
        shares = (pl.col("_sign") * pl.col("transactionshares").abs()).cast(pl.Float64)
        value = (pl.col("_sign") * pl.col("transactionvalue").abs()).cast(pl.Float64)
        open_market = pl.col("_bucket").is_in(list(OPEN_MARKET))

        def only(condition: pl.Expr, amount: pl.Expr) -> pl.Expr:
            return pl.when(condition).then(amount).otherwise(0.0)

        per_code = []
        for bucket in BUCKETS:
            is_bucket = pl.col("_bucket") == bucket
            per_code.append(only(is_bucket, shares).alias(f"net_shares_{bucket}"))
            per_code.append(only(is_bucket, value).alias(f"net_value_{bucket}"))
        return lines.select(
            "permaticker",
            "date",
            only(open_market, shares).alias("net_shares"),
            only(open_market, value).alias("net_value"),
            *per_code,
            only(pl.col("_derivative"), shares).alias("net_shares_derivative"),
            pl.when(pl.col("_bucket") == "P").then(pl.col("ownername")).alias("_buyer"),
            pl.when(pl.col("_bucket") == "S").then(pl.col("ownername")).alias("_seller"),
        )

    def _combine(self) -> list[pl.Expr]:
        """Sum a day's lines and count its distinct buyers and sellers.

        A line without a value adds nothing to the value variables.
        """
        sums = [name for name in self._empty() if name.startswith("net_")]
        return [
            *(pl.col(name).sum() for name in sums),
            pl.col("_buyer").drop_nulls().n_unique().cast(pl.Float64).alias("insiders_buying"),
            pl.col("_seller").drop_nulls().n_unique().cast(pl.Float64).alias("insiders_selling"),
        ]
