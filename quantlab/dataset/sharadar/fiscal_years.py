"""Point-in-time history of each company's latest fiscal years (SF1 ARY), keyed by permaticker.

SF1's ARY dimension holds one row per company, fiscal year and filing, each
dated at its SEC filing date (its *release date*); a restatement of a year
is a new row with a later date, never a rewrite. The quarterly fundamentals
panel (``quantlab.dataset.sharadar.fundamentals``) shows only the latest
period's row, so it cannot say what a company earned three years ago as
known today. ``SharadarFiscalYearsDataset`` shows that history: on every
trading day, per security, it holds the latest ``years`` fiscal years whose
rows are available by that day, newest first, as ``<indicator>_fy0`` (the
latest fiscal year) .. ``<indicator>_fy<years - 1>``.

A row is placed at its *available date*, the first SEP trading day on or
after its release date, the same rule as the fundamentals panel. So:

- a fiscal year appears from its own release date, never earlier, and its
  arrival shifts the older years one slot back;
- each slot shows its fiscal year's latest filing available by that day, so
  a restatement of any year, the newest or an older one, is shown from its
  own release date;
- a slot whose year is not yet known (fewer than ``years`` are released) is
  NaN, and its ``reportperiod_fy<k>`` is NaT;
- a security shows nothing once the row shown in ``fy0`` (the newest
  fiscal year's latest filing) was released more than ``stale_after_days``
  ago.

A fiscal year is identified by its ``reportperiod``: a company that moves
its fiscal year end has two slots a few months apart.

The calendar is SEP's trading days and the ``symbol`` axis the int64
permaticker, as for the other Sharadar panels. Besides the indicator
variables (float64, ``unit`` from INDICATORS), ``reportperiod_fy<k>`` holds
each slot's fiscal year end and ``release_date`` the filing date of the row
shown in ``fy0``. The store grows with ``update()``: new trading days are
appended and a stored day is never rewritten.

Examples
--------
Build the store from a downloaded raw tier and read a quarter::

    config = SharadarFiscalYearsConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_sf1_fiscal_years/sharadar_sf1_fiscal_years.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarFiscalYearsDataset(config).update()
    panel = SharadarFiscalYearsDataset(config).panel("2024-01-02", "2024-03-28")
    panel["eps_fy0"].attrs["unit"]  # 'USD/share'
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset.config import DatasetConfig, SharadarFiscalYearsConfig
from quantlab.dataset.sharadar.permatickers import map_raw_table
from quantlab.dataset.sharadar.point_in_time import ReleasePanelDataset, ReleaseSpec
from quantlab.dataset.sharadar.tables import (
    SF1_INDICATORS,
    scan_raw_table,
    table,
)
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.timer import Timer

#: The SF1 dimension the panel is built from: annual, as reported.
DIMENSION = "ARY"

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


def slot_name(name: str, slot: int) -> str:
    """Return the variable name of ``name`` in fiscal-year slot ``slot``.

    Examples
    --------
    >>> slot_name("eps", 0)
    'eps_fy0'
    """
    return f"{name}_fy{slot}"


class SharadarFiscalYearsDataset(ReleasePanelDataset):
    """Daily point-in-time panel of each company's latest fiscal years, keyed by permaticker.

    Parameters
    ----------
    config : SharadarFiscalYearsConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``sf1``, ``sep`` (the calendar), ``tickers`` and ``indicators`` raw
        tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarFiscalYearsDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)[:3]
        # ['eps_fy0', 'eps_fy1', 'eps_fy2']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarFiscalYearsConfig

    #: SF1 and SEP, the calendar, must be complete through a day for it to be built.
    RAW_TABLES = ("sf1", "sep")

    def _normalize_config(self, config: DatasetConfig) -> SharadarFiscalYearsConfig:
        """Normalise as the base class does, then check the fiscal-year fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarFiscalYearsConfig``.
        ValueError
            If ``table`` is not ``"sf1"``, ``indicators`` is empty, repeats a
            name or names one SF1 does not have, ``years`` or
            ``stale_after_days`` is below 1, or a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``).

        Examples
        --------
        An unknown indicator is refused::

            SharadarFiscalYearsDataset(dataclasses.replace(config, indicators=("epss",)))
            # ValueError: SharadarFiscalYearsDataset: indicators ['epss'] are not SF1 indicators.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarFiscalYearsConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarFiscalYearsConfig, got "
                f"{type(config).__name__}."
            )
        if config.table != "sf1":
            raise ValueError(f"{self.class_name}: table must be 'sf1', got {config.table!r}.")
        indicators = tuple(config.indicators)
        if not indicators or len(set(indicators)) != len(indicators):
            raise ValueError(
                f"{self.class_name}: indicators must name at least one SF1 "
                f"indicator, each once; got {indicators!r}."
            )
        unknown = [name for name in indicators if name not in SF1_INDICATORS]
        if unknown:
            raise ValueError(f"{self.class_name}: indicators {unknown} are not SF1 indicators.")
        if config.years < 1:
            raise ValueError(f"{self.class_name}: years must be at least 1, got {config.years}.")
        if config.stale_after_days is not None and config.stale_after_days < 1:
            raise ValueError(
                f"{self.class_name}: stale_after_days must be at least 1 or "
                f"None, got {config.stale_after_days}."
            )
        return normalize_universe(config, self.class_name)

    # -- the release table ------------------------------------------------------

    def _rows(self) -> pl.DataFrame:
        """Return the universe's ARY rows, each with its available date.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` (a trading day), ``release_date``,
            ``reportperiod`` and the configured indicators; rows released
            after the calendar's last day are dropped.

        Raises
        ------
        ValueError
            If a raw ticker has two permatickers, or two rows of one ticker
            and permaticker share a release date and fiscal year (rows under
            two tickers are settled, ``PermatickerResolver.settle``).
        """
        config = self.config
        root = config.raw_data_dir_path
        calendar = self._trading_calendar()
        frame = map_raw_table(
            root,
            "sf1",
            owner=self.class_name,
            store_path=config.zarr_file_path,
            query=lambda scan: scan.filter(pl.col("dimension") == DIMENSION)
            .filter(pl.col("date") <= pl.lit(calendar.through))
            .select("ticker", "date", "reportperiod", *config.indicators, "permaticker"),
            key=("date", "reportperiod"),
        )
        frame = frame.filter(pl.col("permaticker").is_in(universe(config)))
        self._assert_unique_keys(frame)
        return calendar.place(frame).select(
            pl.col("permaticker").alias("symbol"),
            "available",
            pl.col("date").cast(pl.Datetime("ns")).alias("release_date"),
            pl.col("reportperiod").cast(pl.Datetime("ns")),
            *(pl.col(name).cast(pl.Float64) for name in config.indicators),
        )

    def _released(self) -> pl.DataFrame:
        """Return each security's fiscal-year slots at every day they change.

        A security's state changes on every available date of one of its
        rows. The state on such a day takes, per fiscal year, the latest
        filing available by then, and keeps the ``years`` latest fiscal
        years, newest in slot 0.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available``, ``release_date`` (of slot 0's row) and
            per slot ``k`` the ``reportperiod_fy<k>`` and indicator
            ``<name>_fy<k>`` columns, sorted by ``available``.
        """
        config = self.config
        with Timer(f"{self.class_name}: fiscal-year states"):
            rows = self._rows()
            events = rows.select("symbol", pl.col("available").alias("as_of")).unique()
            candidates = (
                events.join(rows, on="symbol", how="inner")
                .filter(pl.col("available") <= pl.col("as_of"))
                .sort("symbol", "as_of", "reportperiod", "release_date")
                .unique(subset=["symbol", "as_of", "reportperiod"], keep="last", maintain_order=True)
                .sort(
                    ["symbol", "as_of", "reportperiod"], descending=[False, False, True]
                )
                .with_columns(slot=pl.int_range(pl.len()).over("symbol", "as_of"))
                .filter(pl.col("slot") < config.years)
            )
            states = events.sort("symbol", "as_of").join(
                candidates.filter(pl.col("slot") == 0).select("symbol", "as_of", "release_date"),
                on=["symbol", "as_of"],
                how="left",
            )
            for slot in range(config.years):
                states = states.join(
                    candidates.filter(pl.col("slot") == slot).select(
                        "symbol",
                        "as_of",
                        *(
                            pl.col(name).alias(slot_name(name, slot))
                            for name in ("reportperiod", *config.indicators)
                        ),
                    ),
                    on=["symbol", "as_of"],
                    how="left",
                )
            return states.rename({"as_of": "available"}).sort("available")

    def _spec(self) -> ReleaseSpec:
        """Return the slot variables (each with its vendor unit) and the date variables."""
        units = self._units()
        variables = {}
        for name in self.config.indicators:
            for slot in range(self.config.years):
                attrs = {"description": f"{name} of fiscal-year slot {slot} (0 is the latest)"}
                if name in units:
                    attrs["unit"] = units[name]
                variables[slot_name(name, slot)] = attrs
        for slot in range(self.config.years):
            variables[slot_name("reportperiod", slot)] = {
                "description": f"fiscal year end of slot {slot}"
            }
        variables["release_date"] = {
            "description": "SEC filing date of the row shown in slot 0"
        }
        return ReleaseSpec(variables=variables, dates=self._date_variables())

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return f"no {DIMENSION} row is shown"

    def _assert_unique_keys(self, frame: pl.DataFrame) -> None:
        """Raise if two rows of one permaticker share a release date and fiscal year."""
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
                f"date, fiscal year) key(s) have several raw rows, first "
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

    def _date_variables(self) -> tuple[str, ...]:
        """Return the names of the panel's datetime variables."""
        return (
            "release_date",
            *(slot_name("reportperiod", slot) for slot in range(self.config.years)),
        )
