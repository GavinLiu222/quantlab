"""What the Massive REST panels on the permaticker axis share.

Each of Massive's REST panels (dividends, IPOs, news, short volume, short
interest, 8-K disclosures) reads one or more endpoints of the REST raw tier
(``quantlab.dataset.massive.raw``), places each record on a permaticker
through the Ticker map (``quantlab.dataset.massive.placement.TickerPlacement``)
and shows it on SEP's trading calendar over SEP's universe. Two classes hold
the shared part:

- ``MassiveRestPanel``, a mixin for any Sharadar release or filing panel:
  the Ticker map, the report of the records left out, the calendar built
  through the last day SEP, every endpoint read and the Ticker map all
  reach, and the placement of an event on its trading day;
- ``MassiveRestEventPanel``, a ``ReleasePanelDataset`` whose rows are
  events shown on their own day only: the config checks, the window and
  the symbols an update adds.

Examples
--------
A panel names the endpoints it reads and its variables::

    class MassiveExampleDataset(MassiveRestEventPanel):
        config_cls = MassiveExampleConfig
        REST_ENDPOINTS = ("example",)
        VARIABLES = {"example_value": {"unit": "USD", "description": "..."}}

        def _released(self):
            ...  # symbol, available and the variables, one row per pair
"""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.config import DatasetConfig, MassiveRestPanelConfig, TickerMapConfig
from quantlab.dataset.massive.placement import UNPLACED_SUFFIX
from quantlab.dataset.massive.raw import read_rest_watermark
from quantlab.dataset.massive.ticker_map import TickerMap
from quantlab.dataset.sharadar.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
    on_grid,
    to_panel,
)
from quantlab.dataset.sharadar.tables import raw_through, trading_days
from quantlab.dataset.sharadar.universe import normalize_universe

__all__ = ["MassiveRestEventPanel", "MassiveRestPanel"]


class MassiveRestPanel:
    """Mixin of a Sharadar panel built from Massive's REST records placed through the Ticker map.

    Put it before the panel's base class (``ReleasePanelDataset`` or a
    subclass) and set ``REST_ENDPOINTS``. The config is a
    ``MassiveRestPanelConfig``.

    Examples
    --------
    ``MassiveDisclosureDataset`` is one::

        class MassiveDisclosureDataset(MassiveRestPanel, FilingPanelDataset):
            REST_ENDPOINTS = ("8k_disclosures",)
    """

    #: The REST endpoints read (``quantlab.acquisition.massive.rest.ENDPOINTS``);
    #: the store is built only through the day each one is pulled through.
    REST_ENDPOINTS: ClassVar[tuple[str, ...]] = ()

    config: MassiveRestPanelConfig

    def _on_config_installed(self) -> None:
        """Drop what was cached for the previous config."""
        super()._on_config_installed()
        self._ticker_map: TickerMap | None = None

    @property
    def ticker_map(self) -> TickerMap:
        """The Ticker map the records are placed through.

        Examples
        --------
        With ``config`` a ``MassiveDividendsConfig``::

            MassiveDividendDataset(config).ticker_map.path.name
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
        With ``config`` a ``MassiveDividendsConfig`` of the store
        ``massive_dividends_1d.zarr``::

            MassiveDividendDataset(config).unplaced_path
            # '.../massive_dividends_1d.zarr.unplaced.json'
        """
        return f"{self.config.zarr_file_path}{UNPLACED_SUFFIX}"

    def _trading_calendar(self) -> TradingCalendar:
        """Return SEP's trading days through the last day SEP, every endpoint read and the Ticker map reach.

        Raises
        ------
        ValueError
            If an endpoint was never pulled (naming ``rest_pull.py``) or the
            Ticker map maps no day (naming ``ticker_map.py``).
        """
        if self._calendar_cache is None:
            root = self.config.raw_data_dir_path
            ends = {"SEP": raw_through(root, ("sep",))}
            for endpoint in self.REST_ENDPOINTS:
                through = read_rest_watermark(self.config.massive_dir, endpoint)
                if through is None:
                    raise ValueError(
                        f"{self.class_name}: Massive's {endpoint!r} endpoint was never pulled into "
                        f"{self.config.massive_dir}; run scripts/massive/rest_pull.py first."
                    )
                ends[endpoint] = through
            days = self.ticker_map.days()
            if not days:
                raise ValueError(
                    f"{self.class_name}: the Ticker map at {self.config.ticker_map_path} maps no day; "
                    f"run scripts/massive/ticker_map.py first."
                )
            ends["Ticker map"] = days[-1]
            through = min(ends.values())
            self._calendar_cache = TradingCalendar(
                trading_days(root, through), through=through, start=self.config.start_date, end=self.config.end_date
            )
        return self._calendar_cache

    @staticmethod
    def _on_trading_days(
        rows: pl.DataFrame, calendar: TradingCalendar, on: str, first, *, lag: timedelta | None = None
    ) -> pl.DataFrame:
        """Return ``rows`` placed on the trading day of ``on`` (+ ``lag``), from the store's first day.

        An event placed before ``first``, the window's first day, is dropped.
        """
        return calendar.place(rows, on=on, lag=lag).filter(pl.col("available") >= first)


class MassiveRestEventPanel(MassiveRestPanel, ReleasePanelDataset):
    """Daily panel of Massive REST events, each shown on its own trading day only.

    A subclass sets ``config_cls`` (a ``MassiveRestPanelConfig``),
    ``REST_ENDPOINTS``, ``VARIABLES`` and, when a day without an event is
    not NaN, ``FILL_VALUE``; it implements ``_released`` (``symbol``,
    ``available`` and the variables, one row per pair) and
    ``_nothing_shown``. A panel that shows a row on later days too (short
    interest, shown until the next release) overrides
    ``_raw_data_to_xr_window`` and ``_added_symbols_with_raw_history``.

    Examples
    --------
    ``MassiveNewsDataset`` counts 0 on a day without an article::

        class MassiveNewsDataset(MassiveRestEventPanel):
            REST_ENDPOINTS = ("news",)
            FILL_VALUE = 0.0
    """

    #: The panel's variables and their attributes, in panel order.
    VARIABLES: ClassVar[dict[str, dict]] = {}
    #: The value of a cell without an event, and of a new permaticker's stored days.
    FILL_VALUE: ClassVar[float] = float("nan")

    def _normalize_config(self, config: DatasetConfig) -> MassiveRestPanelConfig:
        """Normalise as the base class does, then check the config class, the table and the universe fields.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of the panel's ``config_cls``.
        ValueError
            If ``table`` is not ``"sep"`` or a universe field is invalid.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.config_cls):
            raise TypeError(f"{self.class_name} needs a {self.config_cls.__name__}, got {type(config).__name__}.")
        if config.table != "sep":
            raise ValueError(f"{self.class_name}: table must be 'sep', got {config.table!r}.")
        return normalize_universe(config, self.class_name)

    def _spec(self) -> ReleaseSpec:
        """Return the variables and their attributes; no staleness."""
        return ReleaseSpec(variables=self.VARIABLES, stamp=None)

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list[int] | None = None) -> xr.Dataset:
        """Return one window: each day's values on that day, ``FILL_VALUE`` elsewhere."""
        days, symbols = self._window_axes(start_date, end_date, symbols)
        cells = (
            on_grid(days, symbols)
            .join(self._released_rows(), left_on=["timestamp", "symbol"], right_on=["available", "symbol"],
                  how="left")
            .sort("_day", "_column")
            .with_columns(pl.col(name).cast(pl.Float64).fill_null(self.FILL_VALUE) for name in self.VARIABLES)
        )
        return to_panel(cells, days, symbols, self.VARIABLES)

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count each added permaticker's days with an event between ``start`` and ``end``."""
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
        """Fill a new permaticker's stored days with ``FILL_VALUE``: it had no event then."""
        return {name: np.float64(self.FILL_VALUE) for name in self.VARIABLES}
