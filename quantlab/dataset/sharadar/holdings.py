"""Sharadar 13F institutional ownership (SF3A) as a daily panel, keyed by permaticker.

Institutions file their 13F holdings up to 45 days after each calendar
quarter's end. SF3 holds every holding (security, investor, security type,
quarter); SF3A is the vendor's sum of SF3 by security and quarter, and is
what this panel reads: ``shrholders`` (institutions holding the common
stock), ``shrunits`` (their shares, in thousands) and ``shrvalue`` (the
shares' value, in USD millions). On the 2026-10-05 pull, SF3A's holders
and shares equal SF3 summed by hand up to rounding.

SF3 has no filing date, and a quarter's rows fill in as the filings arrive:
40 days after a quarter end, SF3A's newest quarter can hold a few dozen
holders of a stock that has six thousand. ``SharadarHoldingsDataset``
therefore shows a quarter from its *available date*, the first SEP trading
day on or after quarter end + 45 days (``AVAILABILITY_LAG``), until the
next quarter's available date. So a partial quarter is never shown, and a
security with no row in the shown quarter shows nothing (NaN) rather than
an older quarter's count. Quarters due before SEP's first trading day are
all available on it, and the latest of them is the one shown. Variables:

- ``holders``: institutions holding the stock;
- ``shares_held``: the shares they hold (``shrunits`` times 1,000);
- ``value_held``: the value of those shares in USD (``shrvalue`` times
  1,000,000), as the filers report it at the quarter end;
- ``quarter_end``: the quarter shown.

The ``symbol`` axis is the int64 permaticker. SF3 keys holdings by the
security's ticker, which TICKERS lists only under the price tables (its
``SF3B`` rows are investors), so tickers are mapped through SEP's rows; a
ticker SEP does not know (a fund, or a CUSIP Sharadar has no prices for)
is left out and counted in the log. SF3A is small and has no
``lastupdated``: ``update.py`` pulls it whole every run, and ``update()``
appends the new trading days.

A rebuilt store shows each quarter as the vendor holds it now, filings made
after the 45 days included; an updated store shows it as it stood on the
update. Both show a quarter only from its available date.

Examples
--------
Build the store and read the holders of a stock::

    config = SharadarHoldingsConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_holdings_1d/sharadar_holdings_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarHoldingsDataset(config).update()
    panel = SharadarHoldingsDataset(config).panel("2024-01-02", "2024-12-31")
    panel["holders"].sel(symbol=199059)  # AAPL
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.config import DatasetConfig, SharadarHoldingsConfig
from quantlab.dataset.sharadar.permatickers import map_raw_table
from quantlab.dataset.sharadar.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    on_grid,
    to_panel,
)
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.timer import Timer

#: Days after a quarter's end until its 13F filings are all due.
AVAILABILITY_LAG = timedelta(days=45)

#: The panel's variables and their attributes.
VARIABLES: dict[str, dict] = {
    "holders": {"unit": "institutions", "description": "13F filers holding the common stock"},
    "shares_held": {"unit": "shares", "description": "common shares 13F filers hold"},
    "value_held": {
        "unit": "USD",
        "description": "value of the common shares 13F filers hold, as reported at the quarter end",
    },
    "quarter_end": {"description": "calendar quarter end of the 13F holdings shown"},
}

#: SF3A writes shares in thousands.
UNITS_SCALE = 1_000.0

#: SF3A writes values in USD millions (INDICATORS, ``shrvalue``).
VALUE_SCALE = 1_000_000.0

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


class SharadarHoldingsDataset(ReleasePanelDataset):
    """Daily panel of 13F holders and shares held, each quarter from quarter end + 45 days.

    Parameters
    ----------
    config : SharadarHoldingsConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``sf3a``, ``sep`` (the calendar) and ``tickers`` raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarHoldingsDataset(config).update()
        sorted(ds.panel("2024-01-02", "2024-01-05").data_vars)
        # ['holders', 'quarter_end', 'shares_held', 'value_held']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarHoldingsConfig

    #: SF3A and SEP, the calendar, must be complete through a day for it to be built.
    RAW_TABLES = ("sf3a", "sep")

    def _normalize_config(self, config: DatasetConfig) -> SharadarHoldingsConfig:
        """Normalise as the base class does, then check the table and the universe fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarHoldingsConfig``.
        ValueError
            If ``table`` is not ``"sf3a"`` or a universe field is invalid.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarHoldingsConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarHoldingsConfig, got "
                f"{type(config).__name__}."
            )
        if config.table != "sf3a":
            raise ValueError(f"{self.class_name}: table must be 'sf3a', got {config.table!r}.")
        return normalize_universe(config, self.class_name)

    # -- the release table ------------------------------------------------------

    def _released(self) -> pl.DataFrame:
        """Return every security's holdings per quarter, at the quarter's available date.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``quarter_end``, ``available`` (a trading day; a
            quarter the calendar does not reach yet is left out),
            ``holders``, ``shares_held`` and ``value_held``.

        Raises
        ------
        ValueError
            If two rows of one ticker and permaticker share a quarter; rows
            of two tickers of one security are settled first
            (``PermatickerResolver.settle``).
        """
        root = self.config.raw_data_dir_path
        with Timer(f"{self.class_name}: quarters"):
            # SF3A holds funds and securities without Sharadar prices: their
            # rows map to nothing and are left out quietly, unreported. SF3A
            # can hold a quarter of one security under its current and a
            # former ticker: the rows are settled (``key``), and those that
            # cannot be are left out and listed in ``<store>.unmapped.json``.
            # A ticker that maps to two permatickers is still refused.
            frame = map_raw_table(
                root,
                "sf3a",
                owner=self.class_name,
                query=lambda scan: scan.select(
                    "ticker",
                    pl.col("date").str.to_date().alias("quarter_end"),
                    "shrholders",
                    "shrunits",
                    "shrvalue",
                    "permaticker",
                ),
                store_path=self.config.zarr_file_path,
                date_column="quarter_end",
                key=("quarter_end",),
                quiet=True,
                report_unmapped=False,
            )
            frame = frame.filter(pl.col("permaticker").is_in(universe(self.config)))
            self._assert_unique_keys(frame)
            return (
                self._trading_calendar()
                .place(frame, "quarter_end", lag=AVAILABILITY_LAG)
                .select(
                    pl.col("permaticker").alias("symbol"),
                    pl.col("quarter_end").cast(pl.Datetime("ns")),
                    "available",
                    pl.col("shrholders").cast(pl.Float64).alias("holders"),
                    (pl.col("shrunits") * UNITS_SCALE).alias("shares_held"),
                    (pl.col("shrvalue") * VALUE_SCALE).alias("value_held"),
                )
            )

    def _spec(self) -> ReleaseSpec:
        """Return the variables; a quarter is replaced by the next, never stale."""
        return ReleaseSpec(variables=VARIABLES, dates=("quarter_end",), stamp=None)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no 13F quarter is shown"

    def _assert_unique_keys(self, frame: pl.DataFrame) -> None:
        """Raise if two rows of one permaticker share a quarter."""
        duplicates = (
            frame.group_by("permaticker", "quarter_end")
            .agg(pl.col("ticker").sort())
            .filter(pl.col("ticker").list.len() > 1)
            .sort("permaticker", "quarter_end")
        )
        if duplicates.height:
            sample = [
                f"{row['permaticker']}@{row['quarter_end']}:{row['ticker']}"
                for row in duplicates.head(_ERROR_SAMPLE).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {duplicates.height} (permaticker, quarter) "
                f"pair(s) have several SF3A rows, first {sample}; refusing "
                f"rather than choosing one."
            )

    # -- windows: one quarter a day for every security ------------------------

    def _shown_by(self, last) -> pl.DataFrame:
        """Return the quarters' rows available on or before ``last`` (every one when ``None``)."""
        quarters = self._released_rows()
        return quarters.filter(pl.col("available") <= last) if last is not None else quarters

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Return one window of the panel: each day shows the latest quarter available by then.

        The quarter is the market's, the same for every security: a security
        without a row in it shows nothing that day. Of several quarters
        available on one day (those due before the calendar's first day),
        the latest is the one shown.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list of int, optional
            Permatickers to build; ``None`` is every one shown by the
            window's end.
        """
        days, symbols = self._window_axes(start_date, end_date, symbols)
        quarters = self._shown_by(days.max() if days.len() else None)
        current = days.to_frame().join_asof(
            quarters.group_by("available").agg(pl.col("quarter_end").max()).sort("available"),
            left_on="timestamp",
            right_on="available",
            strategy="backward",
        ).select("timestamp", "quarter_end")
        cells = (
            on_grid(days, symbols)
            .join(current, on="timestamp", how="left")
            .join(
                quarters.select(
                    "symbol", "quarter_end", "holders", "shares_held", "value_held"
                ),
                on=["symbol", "quarter_end"],
                how="left",
            )
            .sort("_day", "_column")
        )
        shown = cells.get_column("holders").is_not_null()
        cells = cells.with_columns(
            pl.when(shown).then(pl.col("quarter_end")).alias("quarter_end")
        )
        return to_panel(cells, days, symbols, self._spec().variables)

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count, per added permaticker, its quarters shown between ``start`` and ``end``.

        A quarter shown from before ``start`` counts too: it is still shown on ``start``.
        """
        if not added:
            return {}
        quarters = self._shown_by(pd.Timestamp(end).to_datetime64())
        starts = quarters.select("available").unique().sort("available").get_column("available")
        # The quarter shown on ``start`` is the latest available by then.
        # polars compares a Datetime series with a Python datetime, not a numpy datetime64.
        first = starts.filter(starts <= pd.Timestamp(start).to_pydatetime()).max()
        counts = (
            quarters.filter(
                pl.col("symbol").is_in([int(s) for s in added])
                & (pl.col("available") >= (first if first is not None else starts.min()))
            )
            .group_by("symbol")
            .len()
        )
        return {str(symbol): int(rows) for symbol, rows in counts.iter_rows()}
