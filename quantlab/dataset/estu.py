"""Index membership from a BarraStyle store's estimation universe (``estu``).

``BarraStyle`` (``quantlab.factor.predefined.barra``) writes, beside its
styles, the estimation universe it standardises them over: ``estu`` on
``(timestamp, symbol)``, positive where a security is among the largest
domestic common stocks by the previous bar's market cap (the 3,000 largest
for the Sharadar build). ``EstuConstituentDataset`` reads that variable as a
point-in-time index, the "us3000" universe of the Sharadar research.

``estu`` at bar t uses the cap of t-1, so the membership of t is known at t.
The panel ends at the store's last bar and moves on with it: extend the
BarraStyle store (``Factor.extend``), then ``update()`` this dataset.

Examples
--------
>>> from quantlab.dataset.config import ConstituentDatasetConfig
>>> members = EstuConstituentDataset(ConstituentDatasetConfig(
...     zarr_file_path="/data/universe/us3000/membership_estu/membership_estu.zarr",
...     start_date="2010-01-01",
...     kwargs={"barra_store": "/data/factors/market/barra_style/barra_style.zarr"},
... ))
>>> members.update().panel("2026-10-01", "2026-10-02")["is_member"].sum("symbol").values
array([3000, 3000])
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr

from quantlab.dataset._support.member_runs import member_runs
from quantlab.dataset.raw import IndexConstituentDataset

__all__ = ["ESTU_COVERAGE_START", "EstuConstituentDataset"]

#: The first bar of the Sharadar BarraStyle store's ``estu``.
ESTU_COVERAGE_START = "2000-02-02"


class EstuConstituentDataset(IndexConstituentDataset):
    """Membership = the ``estu`` of a BarraStyle store, one interval per run of member bars.

    A run of consecutive bars in the universe is one interval, both ends
    included, so the calendar days inside a run (a weekend) read as member.
    A security still in the universe on the store's last bar has an
    interval ending on that bar, so the panel's last day is the store's
    last bar (no ``as_of`` needed).

    Parameters
    ----------
    config : ConstituentDatasetConfig
        ``kwargs["barra_store"]`` is the BarraStyle Zarr store holding
        ``estu``. ``start_date`` and ``end_date`` limit the bars read; leave
        ``end_date`` unset for a panel that follows the store.

    Examples
    --------
    >>> members = EstuConstituentDataset(config)
    >>> members.update()
    >>> members.panel("2026-10-02", "2026-10-02")["is_member"].sum().item()
    3000
    """

    def _pit_coverage_start(self, config) -> str:
        """Return the first bar of the Sharadar BarraStyle ``estu``."""
        return ESTU_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        """Return one ``symbol``/``start_date``/``end_date`` row per run of member bars.

        Raises
        ------
        KeyError
            If the config has no ``kwargs["barra_store"]``.
        """
        estu = xr.open_zarr(self.config.kwargs["barra_store"])["estu"]
        estu = estu.sel(timestamp=slice(self.config.start_date, self.config.end_date))
        mask = estu.transpose("timestamp", "symbol").values > 0
        return member_runs(mask, estu["timestamp"].values, estu["symbol"].values.astype(np.int64))
