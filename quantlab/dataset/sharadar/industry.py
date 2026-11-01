"""Point-in-time Fama-French 48 industry of each Sharadar security, keyed by permaticker.

TICKERS gives each security's *current* SIC code (``siccode``); its
``famaindustry``, ``sicindustry`` and ``sector`` are current snapshots too,
and none of them is used here. The SIC history is rebuilt from ACTIONS'
``sicchangefrom`` rows by ``quantlab.dataset.sharadar.history.sic_history``,
which the listing panel (``quantlab.dataset.sharadar.listing``) reads too:
on a day, a security's SIC is the ``sicchangefrom`` of its first change
dated after that day, or the current code when no change is later, so a
change dated on a weekend or holiday shows from the next trading day.

``SharadarIndustryDataset`` classifies each day's SIC into the Fama-French
48 industries (``quantlab.dataset._support.ff48``; a SIC inside no range is
48, "Other"), merges the thin industries named in
``config.industry_merge``, and stores the code as one float variable,
``industry``, on SEP's trading days. A security's industry is shown from its
TICKERS ``firstpricedate`` to its ``lastpricedate`` and is NaN outside them,
or where its SIC is unknown. The ``industry`` variable's ``names`` attribute maps
each code that can appear (as a string) to French's short name.

ACTIONS rows of a ticker outside SEP are dropped, and two SIC changes of
one security on one date are refused (``sic_history``). The
store grows with ``update()``: new days are appended and a stored day is
never rewritten.

Examples
--------
Build the store and read a year::

    config = SharadarIndustryConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_industry_1d/sharadar_industry_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarIndustryDataset(config).update()
    panel = SharadarIndustryDataset(config).panel("2024-01-02", "2024-12-31")
    panel["industry"].attrs["names"]["44"]  # 'Banks'
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl

from quantlab.dataset._support.ff48 import (
    apply_merge,
    check_merge,
    ff48_codes,
    short_names,
)
from quantlab.dataset.config import DatasetConfig, SharadarIndustryConfig
from quantlab.dataset.sharadar.history import sep_securities, sic_history
from quantlab.dataset.sharadar.spells import SpellPanelDataset

#: Name of the panel's one variable.
VARIABLE = "industry"


class SharadarIndustryDataset(SpellPanelDataset):
    """Daily panel of each security's point-in-time Fama-French 48 industry code.

    Parameters
    ----------
    config : SharadarIndustryConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``tickers``, ``actions`` and ``sep`` (the calendar) raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarIndustryDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)  # ['industry']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarIndustryConfig
    VARIABLE = VARIABLE
    RAW_TABLES = ("sep", "actions")

    def _normalize_config(self, config: DatasetConfig) -> SharadarIndustryConfig:
        """Normalise as ``SpellPanelDataset`` does, then check the merge mapping.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarIndustryConfig``.
        ValueError
            If ``table`` is not ``"sep"``, a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``) or
            ``industry_merge`` is refused (``quantlab.dataset._support.ff48.check_merge``).

        Examples
        --------
        A merge naming a code outside 1..48 is refused::

            SharadarIndustryDataset(dataclasses.replace(config, industry_merge=((27, 49),)))
            # ValueError: SharadarIndustryDataset: the industry merge mapping names unknown code(s) [49]; ...
        """
        config = super()._normalize_config(config)
        return dataclasses.replace(
            config, industry_merge=check_merge(config.industry_merge, self.class_name)
        )

    def _variable_attrs(self) -> dict:
        """Return the scheme and each code's short name."""
        return {"scheme": "Fama-French 48", "names": short_names(self.config.industry_merge)}

    def _build_spells(self) -> pl.DataFrame:
        """Return every security's industry spells.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the permaticker), ``start`` (inclusive, null when
            open), ``end`` (exclusive, null when open), ``industry`` (the
            merged code, NaN for an unknown SIC), ``first`` and ``last``
            (the security's first and last price dates).

        Raises
        ------
        ValueError
            If a permaticker has two SEP TICKERS rows, or two SIC changes
            on one date.
        """
        securities = sep_securities(self.config, self.class_name, ("siccode",))
        spells = sic_history(self.config, securities, self.class_name).join(
            securities.select(
                "permaticker",
                pl.col("firstpricedate").alias("first"),
                pl.col("lastpricedate").alias("last"),
            ),
            on="permaticker",
        )
        sic = spells.get_column("sic").fill_null(np.nan).to_numpy()
        industry = apply_merge(ff48_codes(sic), self.config.industry_merge)
        return spells.select(
            pl.col("permaticker").alias("symbol"), "start", "end", "first", "last"
        ).with_columns(pl.Series(VARIABLE, industry))
