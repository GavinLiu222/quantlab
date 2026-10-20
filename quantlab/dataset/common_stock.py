"""The common-stock membership: every security of a whole-market price store.

The whole-market Sharadar store (``market/sharadar/sharadar_sep_1d``) holds,
under its default ``category_filter``, the domestic common stocks: primary
and secondary share classes, no ADRs, Canadian companies or preferred
shares. ``CommonStockConstituentDataset`` reads it as a point-in-time index:
a security is a member from its first to its last bar with a finite
``close``, halted days (no price) included, one interval per security. Its
panel feeds ``MembershipCapWeight`` unchanged, giving the market portfolio's
cap weights.

The panel ends at the store's last bar and moves on with it: extend the
store, then ``update()`` this dataset.

Examples
--------
>>> from quantlab.dataset.config import ConstituentDatasetConfig
>>> members = CommonStockConstituentDataset(ConstituentDatasetConfig(
...     zarr_file_path="/data/quantlab/universe/sharadar/sharadar_common_stock_membership/sharadar_common_stock_membership.zarr",
...     kwargs={"price_store": "/data/quantlab/market/sharadar/sharadar_sep_1d/sharadar_sep_1d.zarr"},
... ))
>>> members.update().panel("2026-10-14", "2026-10-15")["is_member"].sum("symbol").values
array([4661, 4600])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.raw import IndexConstituentDataset

__all__ = ["CommonStockConstituentDataset"]


class CommonStockConstituentDataset(IndexConstituentDataset):
    """Membership = every security of a price store, from its first to its last priced bar.

    A priced bar is one with a finite ``close``. The interval holds both
    ends, so a halt inside it (bars without a price) and the calendar days
    between bars read as member, and a delisted security stops being a
    member the day after its last priced bar. A security priced on the
    store's last bar has an interval ending on that bar, so the panel's last
    day is the store's last bar (no ``as_of`` needed). The coverage starts at
    the store's first bar.

    A security without a price on the store's last bars (a halt or a late
    print at the store's end) is not a member on them: whether it trades
    again is not known yet. ``update()`` appends the new bars and keeps the
    stored ones, so those days stay non-member once it trades again (what
    was known at the time), while a build from scratch reads them as a halt
    inside the interval.

    Parameters
    ----------
    config : ConstituentDatasetConfig
        ``kwargs["price_store"]`` is the whole-market price Zarr store (for
        Sharadar, ``sharadar_sep_1d.zarr``, whose symbol axis is the
        permaticker). ``start_date`` and ``end_date`` limit the bars read;
        leave ``end_date`` unset for a panel that follows the store.

    Examples
    --------
    A store of two securities, ``202`` halted on the middle bar and ``303``
    delisted after it:

    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.dataset.config import ConstituentDatasetConfig
    >>> days = pd.bdate_range("2024-01-02", periods=3)
    >>> xr.Dataset(
    ...     {"close": (("timestamp", "symbol"), [[1.0, 2.0], [np.nan, 2.0], [1.0, np.nan]])},
    ...     coords={"timestamp": days, "symbol": np.array([202, 303])},
    ... ).to_zarr("data/sep.zarr", mode="w")  # doctest: +ELLIPSIS
    <xarray.backends.zarr.ZarrStore object at ...>
    >>> members = CommonStockConstituentDataset(ConstituentDatasetConfig(
    ...     zarr_file_path="data/common_stock.zarr",
    ...     kwargs={"price_store": "data/sep.zarr"},
    ... ))
    >>> members.from_raw_data().get_xarray_dataset()["is_member"].to_pandas()
    symbol       202    303
    timestamp
    2024-01-02  True   True
    2024-01-03  True   True
    2024-01-04  True  False
    """

    def _normalize_config(self, config: ConstituentDatasetConfig) -> ConstituentDatasetConfig:
        """Read the price store's first bar (the coverage start), then normalise as the base class does.

        Raises
        ------
        KeyError
            If the config has no ``kwargs["price_store"]``.
        """
        first = xr.open_zarr(config.kwargs["price_store"])["timestamp"].values[0]
        self._first_bar = pd.Timestamp(first).strftime("%Y-%m-%d")
        return super()._normalize_config(config)

    def _pit_coverage_start(self) -> str:
        """Return the price store's first bar."""
        return self._first_bar

    def _build_intervals(self) -> pl.DataFrame:
        """Return one ``symbol``/``start_date``/``end_date`` row per security with a priced bar."""
        close = xr.open_zarr(self.config.kwargs["price_store"])["close"]
        close = close.sel(timestamp=slice(self.config.start_date, self.config.end_date))
        priced = close.notnull().transpose("timestamp", "symbol").values
        days = close["timestamp"].values.astype("datetime64[D]")
        listed = priced.any(axis=0)
        first = priced.argmax(axis=0)
        last = len(days) - 1 - priced[::-1].argmax(axis=0)
        symbols = close["symbol"].values[listed]
        return pl.DataFrame(
            {
                "symbol": symbols.astype(np.int64) if symbols.dtype.kind in "iu" else symbols.astype(str),
                "start_date": days[first[listed]],
                "end_date": days[last[listed]],
            }
        ).with_columns(pl.col("start_date").cast(pl.Date), pl.col("end_date").cast(pl.Date))
