"""Massive's daily short volume as a panel on the permaticker axis (#356).

Massive's ``/stocks/v1/short-volume`` records (FINRA's daily short sale
volume, one record per ticker and day, from 2024-02-06), as
``MassiveClient.pull`` keeps them in the REST raw tier, become four
variables on SEP's trading calendar, one column per permaticker, each on
the record's own trading day:

- ``short_volume``: the shares sold short across all venues (the vendor's
  ``short_volume``);
- ``short_total_volume``: the total volume reported across all venues
  (``total_volume``);
- ``short_exempt_volume``: the short volume marked exempt from Regulation
  SHO (``exempt_volume``);
- ``short_volume_ratio``: the percent of the total volume sold short
  (``short_volume_ratio``, ``100 * short_volume / total_volume``).

A cell without a record is NaN. The per-venue columns (ADF, Nasdaq
Carteret, Nasdaq Chicago, NYSE) stay in the raw tier. A record re-read by a
later pull (FINRA's revision, the pull's overlap days) replaces the earlier
copy of its ticker and day; a revision of a day already in the store
reaches it only when the store is rebuilt, as the store only appends days.
When two tickers place on one permaticker on one day their volumes are
summed and the ratio is recomputed from the sums.

Each record is placed on the permaticker the Ticker map gives its ticker on
the record's day, through
``quantlab.dataset.massive.placement.TickerPlacement``; a record it cannot
place is left out and counted in ``<store>.unplaced.json``. Only the
universe's permatickers (by default SEP's domestic common stock) are
columns.

The store is built through the last day SEP, the REST pull and the Ticker
map all reach, so ``update()`` appends a day only once each of them holds
it.

Examples
--------
Build the store and read one security's short volume ratio::

    config = MassiveShortVolumeConfig(
        zarr_file_path="/data/quantlab/market/massive/massive_short_volume_1d/massive_short_volume_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
        massive_dir="/data/quantlab/downloads/massive",
        ticker_map_path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
        start_date="2024-02-06",
    )
    panel = MassiveShortVolumeDataset(config).update().panel("2024-03-01", "2024-03-29")
    panel["short_volume_ratio"].sel(symbol=199059)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.config import (
    DatasetConfig,
    MassiveShortVolumeConfig,
    TickerMapConfig,
)
from quantlab.dataset.massive.placement import UNPLACED_SUFFIX, TickerPlacement
from quantlab.dataset.massive.raw import read_rest_watermark, rest_page_files
from quantlab.dataset.massive.ticker_map import TickerMap
from quantlab.dataset.sharadar.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
    on_grid,
    to_panel,
)
from quantlab.dataset.sharadar.tables import raw_through, trading_days
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.timer import Timer

__all__ = ["VARIABLES", "MassiveShortVolumeDataset"]

#: The panel's variables and their attributes, in panel order.
VARIABLES: dict[str, dict] = {
    "short_volume": {
        "unit": "shares",
        "description": "Shares sold short across all venues on the day (Massive's short_volume).",
    },
    "short_total_volume": {
        "unit": "shares",
        "description": "Total volume reported across all venues on the day (Massive's total_volume).",
    },
    "short_exempt_volume": {
        "unit": "shares",
        "description": "Short volume marked exempt from Regulation SHO (Massive's exempt_volume).",
    },
    "short_volume_ratio": {
        "unit": "percent",
        "description": "100 * short_volume / short_total_volume (Massive's short_volume_ratio).",
    },
}

#: The REST endpoint read (``quantlab.acquisition.massive.rest.ENDPOINTS``).
ENDPOINT = "short_volume"

#: The fields of a record read; the per-venue columns are not among them.
SCHEMA = {
    "ticker": pl.String,
    "date": pl.String,
    "short_volume": pl.Float64,
    "total_volume": pl.Float64,
    "exempt_volume": pl.Float64,
}


def _records(vendor_root) -> pl.DataFrame:
    """Return every (ticker, date)'s newest record, ``date`` as ``pl.Date``.

    Pages are read oldest pull first, so the last copy of a ticker and day
    is the newest.
    """
    pages = [
        pl.read_ndjson(path, schema=SCHEMA).with_columns(pl.lit(index, pl.Int64).alias("_page"))
        for index, path in enumerate(rest_page_files(vendor_root, ENDPOINT))
    ]
    frame = pl.concat([pl.DataFrame(schema={**SCHEMA, "_page": pl.Int64}), *pages])
    return (
        frame.with_columns(pl.col("date").str.slice(0, 10).str.to_date(strict=False))
        .filter(pl.col("ticker").is_not_null() & pl.col("date").is_not_null())
        .with_row_index("_row")
        .sort("_page", "_row")
        .unique(["ticker", "date"], keep="last", maintain_order=True)
        .drop("_page", "_row")
    )


class MassiveShortVolumeDataset(ReleasePanelDataset):
    """Daily panel of Massive's short volume totals, by permaticker.

    See the module docstring for the variables and the placement.

    Parameters
    ----------
    dataset_config : MassiveShortVolumeConfig
        Sharadar's raw tier (calendar, universe), Massive's raw tier and the
        Ticker map.

    Examples
    --------
    With ``config`` as in the module example::

        ds = MassiveShortVolumeDataset(config).update()
        list(ds.panel("2024-03-01", "2024-03-05").data_vars)
        # ['short_volume', 'short_total_volume', 'short_exempt_volume', 'short_volume_ratio']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MassiveShortVolumeConfig

    def _normalize_config(self, config: DatasetConfig) -> MassiveShortVolumeConfig:
        """Normalise as the base class does, then check the table and the universe fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``MassiveShortVolumeConfig``.
        ValueError
            If ``table`` is not ``"sep"`` or a universe field is invalid.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, MassiveShortVolumeConfig):
            raise TypeError(f"{self.class_name} needs a MassiveShortVolumeConfig, got {type(config).__name__}.")
        if config.table != "sep":
            raise ValueError(f"{self.class_name}: table must be 'sep', got {config.table!r}.")
        return normalize_universe(config, self.class_name)

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._ticker_map: TickerMap | None = None

    @property
    def ticker_map(self) -> TickerMap:
        """The Ticker map the records are placed through.

        Examples
        --------
        With ``config`` as in the module example::

            MassiveShortVolumeDataset(config).ticker_map.path.name
            # 'massive_ticker_map'
        """
        if self._ticker_map is None:
            self._ticker_map = TickerMap(TickerMapConfig(
                path=self.config.ticker_map_path,
                massive_dir=self.config.massive_dir,
                sharadar_dir=self.config.raw_data_dir_path,
            ))
        return self._ticker_map

    @property
    def unplaced_path(self) -> str:
        """The report of the records left out (``<store>.unplaced.json``).

        Examples
        --------
        With ``config`` as in the module example::

            MassiveShortVolumeDataset(config).unplaced_path.endswith("massive_short_volume_1d.zarr.unplaced.json")
            # True
        """
        return f"{self.config.zarr_file_path}{UNPLACED_SUFFIX}"

    # -- the calendar: the last day every input reaches -----------------------

    def _trading_calendar(self) -> TradingCalendar:
        """Return SEP's trading days through the last day SEP, the pull and the Ticker map reach.

        Raises
        ------
        ValueError
            If the REST endpoint was never pulled or the Ticker map is empty.
        """
        if self._calendar_cache is None:
            root = self.config.raw_data_dir_path
            pulled = read_rest_watermark(self.config.massive_dir, ENDPOINT)
            if pulled is None:
                raise ValueError(
                    f"{self.class_name}: Massive's {ENDPOINT!r} endpoint was never pulled into "
                    f"{self.config.massive_dir}; run scripts/massive/rest_pull.py first."
                )
            days = self.ticker_map.days()
            if not days:
                raise ValueError(
                    f"{self.class_name}: the Ticker map at {self.config.ticker_map_path} maps no day; "
                    f"run scripts/massive/ticker_map.py first."
                )
            through = min(raw_through(root, ("sep",)), pulled, days[-1])
            self._calendar_cache = TradingCalendar(
                trading_days(root, through), through=through, start=self.config.start_date, end=self.config.end_date
            )
        return self._calendar_cache

    # -- the rows ---------------------------------------------------------------

    def _released(self) -> pl.DataFrame:
        """Return every security's totals per day with a record, placed and summed.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` and the four variables, one row per
            pair, sorted by both.
        """
        calendar = self._trading_calendar()
        window = calendar.days_in(self.config.start_date, self.config.end_date)
        placement = TickerPlacement(self.ticker_map, owner=self.class_name)
        schema = {"symbol": pl.Int64, "available": pl.Datetime("ns"), **{n: pl.Float64 for n in VARIABLES}}
        frame = pl.DataFrame(schema=schema)
        with Timer(f"{self.class_name}: place short volume"):
            if window.len():
                rows = calendar.place(_records(self.config.massive_dir), on="date").filter(
                    pl.col("available").is_between(window.min(), window.max())
                )
                rows = placement.place(rows, source=ENDPOINT)
                frame = (
                    rows.group_by(pl.col("permaticker").alias("symbol"), "available")
                    .agg(
                        pl.col("short_volume").sum(),
                        pl.col("total_volume").sum().alias("short_total_volume"),
                        pl.col("exempt_volume").sum().alias("short_exempt_volume"),
                    )
                    .with_columns(
                        pl.when(pl.col("short_total_volume") > 0)
                        .then(100 * pl.col("short_volume") / pl.col("short_total_volume"))
                        .alias("short_volume_ratio")
                    )
                )
            placement.write_report(self.config.zarr_file_path)
        keep = [int(s) for s in universe(self.config)]
        return (
            pl.concat([pl.DataFrame(schema=schema), frame], how="diagonal_relaxed")
            .filter(pl.col("symbol").is_in(keep))
            .select(list(schema))
            .sort("symbol", "available")
        )

    # -- the panel: a value on its own day ----------------------------------------

    def _spec(self) -> ReleaseSpec:
        """Return the four variables and their attributes; no staleness."""
        return ReleaseSpec(variables=VARIABLES, stamp=None)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no Massive short volume record is placed"

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list[int] | None = None) -> xr.Dataset:
        """Return one window: each day's totals on that day, NaN elsewhere."""
        days, symbols = self._window_axes(start_date, end_date, symbols)
        cells = (
            on_grid(days, symbols)
            .join(self._released_rows(), left_on=["timestamp", "symbol"], right_on=["available", "symbol"],
                  how="left")
            .sort("_day", "_column")
            .with_columns(pl.col(name).cast(pl.Float64).fill_null(float("nan")) for name in VARIABLES)
        )
        return to_panel(cells, days, symbols, VARIABLES)

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count each added permaticker's days with a record between ``start`` and ``end``."""
        if not added:
            return {}
        counts = (
            self._released_rows()
            .filter(
                pl.col("symbol").is_in([int(s) for s in added])
                & pl.col("available").is_between(
                    pl.lit(pd.Timestamp(start).to_datetime64()), pl.lit(pd.Timestamp(end).to_datetime64())
                )
            )
            .group_by("symbol")
            .len()
        )
        return {str(symbol): int(rows) for symbol, rows in counts.iter_rows()}

    def _widen_fill_values(self) -> dict:
        """Fill a new permaticker's stored days with NaN: it had no record then."""
        return {name: np.nan for name in VARIABLES}
