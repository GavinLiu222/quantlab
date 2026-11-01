"""Massive's 8-K disclosure categories as a daily panel on the permaticker axis.

Massive's ``/stocks/filings/8-K/vX/disclosures`` records, as
``MassiveClient.pull`` keeps them in the REST raw tier
(``quantlab.dataset.massive.raw.read_rest_records``), become one boolean per
primary category of Massive's disclosure taxonomy
(``/stocks/taxonomies/vX/disclosures``, its newest snapshot), on SEP's
trading calendar, one column per permaticker. A variable is named by its
primary category as the taxonomy spells it (``financial_results``,
``leadership_and_governance``) and described by the secondary and tertiary
categories the taxonomy lists under it.

A variable is true on the first trading day after the filing date of an 8-K
with a disclosure in its category, and false on every other day: an 8-K is
often filed after the close, so a filing shows on the next bar (one filed
on a Friday shows on Monday). A filing with disclosures in two primary
categories sets both. Secondary and tertiary categories stay raw.

A record names its filer's tickers (several for a filer with several share
classes); it is placed once for each ticker, on the permaticker the Ticker
map gives that ticker on the day the filing shows
(``quantlab.dataset.massive.placement.TickerPlacement``). A ticker it cannot
place, and a record naming no ticker, is left out and counted in
``<store>.unplaced.json``. Only the universe's permatickers (by default
SEP's domestic common stock) are columns. A primary category the taxonomy
does not list refuses the build, naming it, so a category the vendor adds is
never dropped; the taxonomy's categories are the store's variables, so a new
one needs the store rebuilt.

The store is built through the last day SEP, the disclosure pull and the
Ticker map all reach, so ``update()`` appends a day only once each of them
holds it.

Examples
--------
Build the store and read the earnings 8-Ks of one security::

    config = MassiveDisclosuresConfig(
        zarr_file_path="/data/quantlab/market/massive/massive_disclosures_1d/massive_disclosures_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
        massive_dir="/data/quantlab/downloads/massive",
        ticker_map_path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
        start_date="2020-01-02",
    )
    panel = MassiveDisclosureDataset(config).update().panel("2024-01-02", "2024-12-31")
    panel["financial_results"].sel(symbol=199059)
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from quantlab.dataset.config import MassiveDisclosuresConfig
from quantlab.dataset.massive.placement import TickerPlacement
from quantlab.dataset.massive.raw import read_rest_snapshot, rest_frame
from quantlab.dataset.massive.rest_panel import MassiveRestPanel
from quantlab.dataset.sharadar.filings import FilingPanelDataset
from quantlab.dataset.sharadar.universe import universe
from quantlab.utils.timer import Timer

__all__ = ["MassiveDisclosureDataset"]

#: The REST endpoint of the disclosures (``quantlab.acquisition.massive.rest.ENDPOINTS``).
DISCLOSURES = "8k_disclosures"

#: The REST snapshot endpoint of the taxonomy.
TAXONOMY = "disclosure_taxonomy"

#: The fields of a disclosure record read.
DISCLOSURE_SCHEMA = {
    "filing_date": pl.String,
    "primary_category": pl.String,
    "tickers": pl.List(pl.String),
}


class MassiveDisclosureDataset(MassiveRestPanel, FilingPanelDataset):
    """Daily panel of Massive's 8-K disclosure categories, one flag per primary category, by permaticker.

    See the module docstring for the variables and the placement.

    Parameters
    ----------
    config : MassiveDisclosuresConfig
        Sharadar's raw tier (calendar, universe), Massive's raw tier and the
        Ticker map.

    Examples
    --------
    With ``config`` as in the module example::

        ds = MassiveDisclosureDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['capital_markets', 'financial_results', 'leadership_and_governance', ...]
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MassiveDisclosuresConfig

    #: The table of the config: SEP gives the calendar and the universe.
    TABLE = "sep"

    #: The REST endpoints read (``MassiveRestPanel``).
    REST_ENDPOINTS = (DISCLOSURES,)

    #: The latest first day of the endpoints read (``MassiveRestPanel.EARLIEST``).
    EARLIEST = date(2020, 1, 1)

    #: The store folder under ``market/massive`` (``MassiveRestPanel.from_data_root``).
    STORE_NAME = "massive_disclosures_1d"

    #: The store folder's ``README.md``.
    STORE_README = """# massive_disclosures_1d

Massive's 8-K disclosure categories (its REST `/stocks/filings/8-K/vX/disclosures`, classified
by `/stocks/taxonomies/vX/disclosures`) as a daily panel on Sharadar's permaticker axis and
SEP's trading calendar, the whole market's domestic common stock (SEP's default category
filter). One boolean per primary category of the taxonomy, named as the taxonomy spells it (for
example `financial_results`, `leadership_and_governance`) and described by the secondary and
tertiary categories under it: true on the first trading day after the filing date of an 8-K with
a disclosure in that category (a Friday filing shows on Monday), false otherwise. Secondary and
tertiary categories stay raw.

A filing is placed once for each ticker it names, on the permaticker the Ticker map gives that
ticker on the day it shows; a ticker the map cannot place, or a filing naming none, is left out
and counted in `massive_disclosures_1d.zarr.unplaced.json`. A primary category the taxonomy does
not list refuses the build. Built by
`quantlab.dataset.massive.disclosures.MassiveDisclosureDataset` from
`downloads/massive/rest/8k_disclosures`, `downloads/massive/rest/disclosure_taxonomy`,
`market/massive/massive_ticker_map` and Sharadar's SEP and TICKERS, from the later of the Ticker
map's first day and 2020-01-01 (Massive's disclosures start in 2020-01); first built by
`scripts/massive/disclosures.py`, then appended day by day by the daily data update.
docs/user-guide/massive.md, "8-K disclosures". Licensed data: personal use.
"""

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._categories: dict[str, dict] | None = None

    # -- the taxonomy: the variables ---------------------------------------------

    def _taxonomy(self) -> dict[str, dict]:
        """Return each primary category's attributes from the newest taxonomy snapshot, sorted by name.

        Raises
        ------
        ValueError
            If the taxonomy was never pulled.
        """
        if self._categories is None:
            try:
                rows = read_rest_snapshot(self.config.massive_dir, TAXONOMY)
            except ValueError:
                raise ValueError(
                    f"{self.class_name}: Massive's {TAXONOMY!r} endpoint was never pulled into "
                    f"{self.config.massive_dir}; run scripts/massive/rest_pull.py first."
                ) from None
            grouped: dict[str, dict[str, set]] = {}
            for row in rows:
                primary = row.get("primary_category")
                if not primary:
                    continue
                parts = grouped.setdefault(primary, {"secondary": set(), "tertiary": set(), "taxonomy": set()})
                for level in ("secondary", "tertiary"):
                    if row.get(f"{level}_category"):
                        parts[level].add(row[f"{level}_category"])
                if row.get("taxonomy"):
                    parts["taxonomy"].add(str(row["taxonomy"]))
            self._categories = {
                primary: {
                    "description": (
                        f"An 8-K with a disclosure in Massive's primary category {primary} "
                        f"(secondary categories: {', '.join(sorted(parts['secondary']))}), "
                        "shown on the first trading day after its filing date."
                    ),
                    "secondary_categories": ", ".join(sorted(parts["secondary"])),
                    "tertiary_categories": ", ".join(sorted(parts["tertiary"])),
                    "taxonomy": ", ".join(sorted(parts["taxonomy"])),
                }
                for primary, parts in sorted(grouped.items())
            }
        return self._categories

    def _empty(self) -> dict[str, object]:
        """Return False for every primary category: nothing was filed."""
        return {primary: False for primary in self._taxonomy()}

    def _attrs(self) -> dict[str, dict]:
        """Return each primary category's description and the taxonomy's categories under it."""
        return self._taxonomy()

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no Massive 8-K disclosure is placed"

    # -- the calendar: the last day every input reaches -----------------------

    def _raw_tables(self) -> tuple[str, ...]:
        """Return SEP, the calendar."""
        return ("sep",)

    # -- the rows ---------------------------------------------------------------

    def _records(self) -> pl.DataFrame:
        """Return the disclosure records' filing date, primary category and tickers.

        Raises
        ------
        ValueError
            If a record's primary category is not in the taxonomy.
        """
        frame = rest_frame(self.config.massive_dir, DISCLOSURES, DISCLOSURE_SCHEMA, key=None)
        known = set(self._taxonomy())
        unknown = sorted(set(frame.get_column("primary_category").drop_nulls().unique()) - known)
        if unknown:
            raise ValueError(
                f"{self.class_name}: Massive's 8-K disclosures name primary categories its taxonomy "
                f"does not list: {', '.join(unknown)}; pull the taxonomy again (scripts/massive/rest_pull.py) "
                "and rebuild the store."
            )
        return frame.filter(pl.col("filing_date").is_not_null() & pl.col("primary_category").is_not_null())

    def _released(self) -> pl.DataFrame:
        """Return every day with a disclosure, per security, one flag per primary category.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` (the first trading day after the
            filing date) and one boolean per primary category, one row per
            pair, sorted by both.
        """
        calendar = self._trading_calendar()
        window = calendar.days_in(self.config.start_date, self.config.end_date)
        categories = list(self._taxonomy())
        placement = TickerPlacement(self.ticker_map, owner=self.class_name)
        schema = {"symbol": pl.Int64, "available": pl.Datetime("ns"), **{name: pl.Boolean for name in categories}}
        frame = pl.DataFrame(schema=schema)
        with Timer(f"{self.class_name}: place 8-K disclosures"):
            if window.len():
                rows = (
                    self._on_trading_days(self._records(), calendar, "filing_date", window.min(),
                                          lag=timedelta(days=1))
                    .explode("tickers", empty_as_null=True)
                    .rename({"tickers": "ticker"})
                )
                placement.leave_out(rows.filter(pl.col("ticker").is_null()), source=DISCLOSURES,
                                    reason="the filing names no ticker", day="filing_date")
                placed = placement.place(rows.filter(pl.col("ticker").is_not_null()), source=DISCLOSURES)
                keep = [int(s) for s in universe(self.config)]
                frame = (
                    placed.filter(pl.col("permaticker").is_in(keep))
                    .group_by(pl.col("permaticker").alias("symbol"), "available")
                    .agg((pl.col("primary_category") == name).any().alias(name) for name in categories)
                    .select(list(schema))
                )
            placement.write_report(self.config.zarr_file_path)
        return frame.sort("symbol", "available")
