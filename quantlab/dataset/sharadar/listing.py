"""Point-in-time listing of each Sharadar security: SIC series and exchange, keyed by permaticker.

TICKERS gives each security's *current* SIC code and exchange; ACTIONS
records their changes (``sicchangefrom``/``sicchangeto`` with the code in
``value``, ``exchangefrom``/``exchangeto`` with the exchange name in
``contraname``). ``SharadarListingDataset`` walks each history back from the
current value (``quantlab.dataset.sharadar.history``; the SIC history is the
one the Fama-French 48 industry panel reads, so the two never disagree) and
stores five float variables on SEP's trading days:

``sic_code``
    The SIC code itself.
``sic_sector``
    The SIC division of the code, 1..11 (``names``: Sharadar's
    ``sicsector`` spellings). A code in no division's range is NaN.
``sic_industry``
    The SIC industry, coded by its SIC code: Sharadar's ``sicindustry`` is
    one title per code (on the 2026-10-16 pull, 440 codes and 440 titles),
    so the value is the code itself and ``names`` maps each code to its
    title. ``names`` holds the codes some TICKERS row carries today; a past
    code no row carries any longer keeps its value without a title, so a
    day's value never depends on the day it was built.
``fama_industry``
    The Fama-French 48 industry of the code, 1..48, unmerged
    (``quantlab.dataset._support.ff48``); ``names`` are French's short names.
    Sharadar's own ``famaindustry`` is not used: it is a current snapshot
    not kept in line with ``siccode`` (it differs from the code's most
    common label on 450 of 20,861 SEP rows of the 2026-10-05 pull).
    ``fama_industry`` is the industry panel's code before its merge.
``exchange``
    The listing exchange, a fixed code per vendor name (``names``). An
    exchange name outside ``EXCHANGES`` refuses the build naming it, so a
    new venue is never silently dropped.

A value is shown from the security's TICKERS ``firstpricedate`` to its
``lastpricedate`` and is NaN outside them, or where it is unknown. A change
shows from its own date, or the next trading day when that is a weekend or
holiday. TICKERS attributes with no history in ACTIONS (``sector``,
``industry``, ``category``, ``location``, ``currency``, the market-cap and
revenue scales, ``famaindustry``) are not in the panel: they would carry
today's value back over the whole history. The store grows with
``update()``: new days are appended and a stored day is never rewritten.

Examples
--------
Build the store and read a year::

    config = SharadarListingConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_listing_1d/sharadar_listing_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarListingDataset(config).update()
    panel = SharadarListingDataset(config).panel("2024-01-02", "2024-12-31")
    panel["exchange"].attrs["names"]["2"]  # 'NASDAQ'
"""

from __future__ import annotations

import numpy as np
import polars as pl

from quantlab.dataset._support.ff48 import ff48_codes, short_names
from quantlab.dataset.config import SharadarListingConfig
from quantlab.dataset.sharadar.history import (
    ERROR_SAMPLE,
    EXCHANGE_FROM,
    changes,
    sep_securities,
    sic_history,
    walk_back,
)
from quantlab.dataset.sharadar.spells import SpellPanelDataset
from quantlab.dataset.sharadar.tables import scan_raw_table

#: The panel's variables.
VARIABLES = ("sic_code", "sic_sector", "sic_industry", "fama_industry", "exchange")

#: The SIC divisions: code, first and last SIC of the range, Sharadar's
#: ``sicsector`` spelling (VERBATIM; Public Administration does not occur on
#: the 2026-10-16 pull and is spelled as the SIC manual names it).
SIC_DIVISIONS: tuple[tuple[int, int, int, str], ...] = (
    (1, 100, 999, "Agriculture Forestry And Fishing"),
    (2, 1000, 1499, "Mining"),
    (3, 1500, 1799, "Construction"),
    (4, 2000, 3999, "Manufacturing"),
    (5, 4000, 4999, "Transportation Communications Electric Gas And Sanitary Service"),
    (6, 5000, 5199, "Wholesale Trade"),
    (7, 5200, 5999, "Retail Trade"),
    (8, 6000, 6799, "Finance Insurance And Real Estate"),
    (9, 7000, 8999, "Services"),
    (10, 9100, 9729, "Public Administration"),
    (11, 9900, 9999, "Nonclassifiable"),
)

#: Each exchange name TICKERS and ACTIONS use (VERBATIM) and its fixed code.
#: Codes never change meaning, so a stored day keeps its meaning; a new
#: name is added with a new code.
EXCHANGES: dict[str, int] = {
    "NYSE": 1,
    "NASDAQ": 2,
    "NYSEMKT": 3,
    "NYSEARCA": 4,
    "BATS": 5,
    "IEX": 6,
    "TXSE": 7,
    "AMEX": 8,
    "OTC": 9,
}

#: How the vendor fills an ACTIONS text field it has nothing for.
_MISSING = "N/A"


def _sic_sector(sic: np.ndarray) -> np.ndarray:
    """Return each SIC's division code, NaN for a SIC in no division (or NaN)."""
    sector = np.full(sic.shape, np.nan)
    for code, low, high, _ in SIC_DIVISIONS:
        sector[(sic >= low) & (sic <= high)] = code
    return sector


class SharadarListingDataset(SpellPanelDataset):
    """Daily panel of each security's point-in-time SIC series and exchange.

    Parameters
    ----------
    config : SharadarListingConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``tickers``, ``actions`` and ``sep`` (the calendar) raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarListingDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['sic_code', 'sic_sector', 'sic_industry', 'fama_industry', 'exchange']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarListingConfig
    VARIABLES = VARIABLES
    RAW_TABLES = ("sep", "actions")

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._titles_cache: dict[str, str] | None = None

    def _titles(self) -> dict[str, str]:
        """Return each SIC code's TICKERS ``sicindustry`` title, keyed by the code as a string."""
        if self._titles_cache is None:
            titles = (
                scan_raw_table(self.config.raw_data_dir_path, "tickers")
                .select(pl.col("siccode").cast(pl.Int64), pl.col("sicindustry").cast(pl.String))
                .drop_nulls()
                .unique()
                .sort("siccode", "sicindustry")
                .unique("siccode", keep="first")
                .collect()
            )
            self._titles_cache = {
                str(code): title for code, title in titles.iter_rows()
            }
        return self._titles_cache

    def _attrs(self, variable: str) -> dict:
        """Return what each variable holds and, for a coded one, its code table."""
        if variable == "sic_code":
            return {"description": "SIC code"}
        if variable == "sic_sector":
            return {
                "description": "SIC division of the SIC code",
                "names": {str(code): name for code, _, _, name in SIC_DIVISIONS},
            }
        if variable == "sic_industry":
            return {"description": "SIC industry title, coded by its SIC code", "names": self._titles()}
        if variable == "fama_industry":
            return {"scheme": "Fama-French 48", "names": short_names(())}
        return {
            "description": "listing exchange",
            "names": {str(code): name for name, code in EXCHANGES.items()},
        }

    def _exchange_codes(self, names: pl.Series, where: str) -> pl.Series:
        """Return each exchange name's code, NaN for a missing one.

        Raises
        ------
        ValueError
            If a name is not in ``EXCHANGES``.
        """
        unknown = sorted(
            set(names.drop_nulls().to_list()) - set(EXCHANGES) - {_MISSING}
        )
        if unknown:
            raise ValueError(
                f"{self.class_name}: unknown exchange name(s) {unknown[:ERROR_SAMPLE]} in "
                f"{where}; add them to quantlab.dataset.sharadar.listing.EXCHANGES."
            )
        return names.replace_strict(
            {name: float(code) for name, code in EXCHANGES.items()},
            default=float("nan"),
            return_dtype=pl.Float64,
        )

    def _build_spells(self) -> pl.DataFrame:
        """Return every security's SIC spells and exchange spells.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``start``, ``end``, ``first``, ``last`` and the five
            variables: a SIC spell carries the four SIC variables and NaN
            for ``exchange``, an exchange spell the reverse.

        Raises
        ------
        ValueError
            If a permaticker has two SEP TICKERS rows, two SIC or two
            exchange changes on one date, or an exchange name is unknown.
        """
        securities = sep_securities(self.config, self.class_name, ("siccode", "exchange"))
        nan = pl.lit(float("nan"))
        sic = sic_history(self.config, securities, self.class_name)
        codes = sic.get_column("sic").fill_null(np.nan).to_numpy()
        sic = sic.select(
            "permaticker", "start", "end",
            pl.Series("sic_code", codes),
            pl.Series("sic_sector", _sic_sector(codes)),
            pl.Series("sic_industry", codes),
            pl.Series("fama_industry", ff48_codes(codes)),
            nan.alias("exchange"),
        )
        history = changes(
            self.config, securities, EXCHANGE_FROM,
            pl.col("contraname").alias("exchange"), "exchange", self.class_name,
        )
        history = history.with_columns(
            self._exchange_codes(history.get_column("exchange"), "ACTIONS exchangefrom")
        )
        current = securities.select("permaticker").with_columns(
            self._exchange_codes(securities.get_column("exchange"), "TICKERS exchange")
        )
        exchange = walk_back(current, "exchange", history).select(
            "permaticker", "start", "end",
            *(nan.alias(v) for v in VARIABLES[:-1]),
            "exchange",
        )
        return pl.concat([sic, exchange]).join(
            securities.select(
                "permaticker",
                pl.col("firstpricedate").alias("first"),
                pl.col("lastpricedate").alias("last"),
            ),
            on="permaticker",
        ).select(pl.col("permaticker").alias("symbol"), "start", "end", "first", "last", *VARIABLES)
