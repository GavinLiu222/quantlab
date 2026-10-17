"""The adjusted price chain of a raw daily price panel, driven by any vendor's events.

SEP (``quantlab.dataset.sharadar.stock``) adjusts its raw prices from
Sharadar's ACTIONS, and Massive's Day aggregates on permatickers
(``quantlab.dataset.massive.mapped_bars``) from Massive's own splits and
dividends (ADR 0048). Both chain the same variables by the same rule, the
CRSP path's convention, here once:

- ``divCash`` is the cash paid per share held on the ex-date (0.0 on other
  rows) and ``splitFactor`` the split's new shares per old share on its
  effective date (1.0 on other rows); the caller gives both, from its own
  vendor's events.
- The day's total return is ``(close + divCash) * splitFactor / close_prev - 1``,
  with ``close_prev`` the last earlier positive raw close, so a halt is
  spanned.
- ``adjClose`` is the security's anchor (its first positive close) grown by
  the product of those returns since, NaN where ``close`` is not positive;
  ``adjOpen``/``adjHigh``/``adjLow`` are scaled by ``adjClose / close``.
- ``cumfacshr`` is 1.0 on the security's first row and divided by
  ``splitFactor`` on every row since, so ``cumfacshr[t-1] / cumfacshr[t] ==
  splitFactor[t]``; ``adjVolume`` is the raw volume in the anchor's shares.

``chain_adjusted`` without a ``carry`` chains a window from its own first
rows (SEP derives its whole window this way). With a ``carry``, the chain
continues each security from the state its stored history ended in
(``carry_state``): the first row's return is taken against the carried
close, ``adjClose`` grows from the carried ``adjClose`` and ``cumfacshr`` from
the carried ``cumfacshr``, so appending a window never rewrites a stored bar
(the Adjusted price rule).

Examples
--------
>>> import polars as pl
>>> from datetime import datetime
>>> frame = pl.DataFrame({
...     "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 3)], "symbol": [1, 1],
...     "open": [10.0, 5.0], "high": [10.0, 5.0], "low": [10.0, 5.0], "close": [10.0, 5.0],
...     "volume": [100.0, 200.0], "divCash": [0.0, 0.0], "splitFactor": [1.0, 2.0],
... })
>>> chain_adjusted(frame, owner="demo")["adjClose"].to_list()   # a 2:1 split is no return
[10.0, 10.0]
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr
from loguru import logger

__all__ = ["ADJUSTED_VARIABLES", "CARRY_SCHEMA", "carry_state", "chain_adjusted"]

#: The variables the chain adds to the raw rows.
ADJUSTED_VARIABLES: tuple[str, ...] = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume", "cumfacshr")

#: The state a security's chain continues from: its last positive raw close and the
#: ``adjClose`` there, its last ``cumfacshr``, and ``volume_scale``, the constant
#: ``adjVolume / (volume * cumfacshr)`` of its chain. Null where it has none yet.
CARRY_SCHEMA = {
    "symbol": pl.Int64,
    "close": pl.Float64,
    "adjClose": pl.Float64,
    "cumfacshr": pl.Float64,
    "volume_scale": pl.Float64,
}

#: Number of securities a log line names.
_SAMPLE = 5


def chain_adjusted(frame: pl.DataFrame, *, owner: str, carry: pl.DataFrame | None = None) -> pl.DataFrame:
    """Return ``frame`` with the adjusted variables chained from its raw prices and events.

    Parameters
    ----------
    frame : pl.DataFrame
        One row per ``timestamp`` and ``symbol``, with the raw ``open``,
        ``high``, ``low``, ``close``, ``volume`` and the events ``divCash``
        and ``splitFactor`` (no nulls). Other columns are kept.
    owner : str
        Named in log lines.
    carry : pl.DataFrame, optional
        ``CARRY_SCHEMA`` rows, the state each security's stored chain ended
        in (``carry_state``). A security without a row starts a chain of its
        own. ``None`` chains the window from its own first rows.

    Returns
    -------
    pl.DataFrame
        ``frame`` sorted by ``symbol`` and ``timestamp``, with
        ``ADJUSTED_VARIABLES`` added.

    Examples
    --------
    A carried chain continues from its stored state (a dividend of 1 on a
    close of 99 after a stored close of 100 is a zero return):

    >>> import polars as pl
    >>> from datetime import datetime
    >>> row = pl.DataFrame({
    ...     "timestamp": [datetime(2024, 1, 4)], "symbol": [1], "open": [99.0], "high": [99.0],
    ...     "low": [99.0], "close": [99.0], "volume": [10.0], "divCash": [1.0], "splitFactor": [1.0],
    ... })
    >>> carry = pl.DataFrame({"symbol": [1], "close": [100.0], "adjClose": [50.0],
    ...                       "cumfacshr": [0.5], "volume_scale": [2.0]}, schema=CARRY_SCHEMA)
    >>> chain_adjusted(row, owner="demo", carry=carry).select("adjClose", "cumfacshr").row(0)
    (50.0, 0.5)
    """
    frame = frame.sort("symbol", "timestamp")
    positive = pl.when(pl.col("close") > 0).then(pl.col("close"))
    close_prev = positive.shift(1).forward_fill().over("symbol")
    carried = carry is not None
    if carried:
        frame = frame.join(
            carry.select(pl.col("symbol").cast(frame.schema["symbol"]), *(
                pl.col(name).alias(f"_carry_{name}") for name in list(CARRY_SCHEMA)[1:]
            )),
            on="symbol",
            how="left",
        )
        close_prev = pl.coalesce(close_prev, pl.col("_carry_close"))
    ret = ((pl.col("close") + pl.col("divCash")) * pl.col("splitFactor")) / close_prev - 1.0
    frame = frame.with_columns(
        (1.0 + ret.fill_null(0.0)).cum_prod().over("symbol").alias("_G"),
        pl.col("splitFactor").cum_prod().over("symbol").alias("_S"),
    )
    base = pl.col("_S").first().over("symbol")
    if carried:
        base = pl.coalesce(pl.col("_carry_cumfacshr"), base)
    frame = frame.with_columns((base / pl.col("_S")).alias("cumfacshr"))
    anchor = (
        frame.filter(pl.col("close") > 0)
        .group_by("symbol")
        .agg(
            pl.col("close").first().alias("_close_anchor"),
            pl.col("_G").first().alias("_G_anchor"),
            pl.col("_S").first().alias("_S_anchor"),
        )
    )
    frame = frame.join(anchor, on="symbol", how="left")
    grown = pl.col("_close_anchor") * pl.col("_G") / pl.col("_G_anchor")
    volume = pl.col("volume") * pl.col("_S_anchor") / pl.col("_S")
    if carried:
        grown = pl.when(pl.col("_carry_adjClose").is_not_null()).then(
            pl.col("_carry_adjClose") * pl.col("_G")
        ).otherwise(grown)
        volume = pl.when(pl.col("_carry_volume_scale").is_not_null()).then(
            pl.col("volume") * pl.col("_carry_volume_scale") * pl.col("cumfacshr")
        ).otherwise(volume)
        unanchored = frame.filter(pl.col("_close_anchor").is_null() & pl.col("_carry_adjClose").is_null())
    else:
        unanchored = frame.filter(pl.col("_close_anchor").is_null())
    unanchored = unanchored.select("symbol").unique()
    if unanchored.height:
        logger.warning(
            f"{owner}: {unanchored.height} permaticker(s) have no "
            f"positive close in the window, so no adjusted price, first "
            f"{unanchored.sort('symbol').head(_SAMPLE)['symbol'].to_list()}."
        )
    frame = frame.with_columns(pl.when(pl.col("close") > 0).then(grown).alias("adjClose"))
    factor = pl.col("adjClose") / pl.col("close")
    frame = frame.with_columns(
        (pl.col("open") * factor).alias("adjOpen"),
        (pl.col("high") * factor).alias("adjHigh"),
        (pl.col("low") * factor).alias("adjLow"),
        volume.alias("adjVolume"),
    )
    helpers = [name for name in frame.columns if name.startswith(("_G", "_S", "_close_anchor", "_carry_"))]
    return frame.drop(helpers)


def carry_state(panel: xr.Dataset, earlier: pl.DataFrame | None = None) -> pl.DataFrame:
    """Return the state each security's chain ends in after ``panel``.

    Parameters
    ----------
    panel : xr.Dataset
        Chained bars on ``(timestamp, symbol)`` in time order, with
        ``close``, ``volume`` and ``ADJUSTED_VARIABLES``.
    earlier : pl.DataFrame, optional
        The state before ``panel`` (``CARRY_SCHEMA``); a security or field
        ``panel`` gives no value of keeps it.

    Returns
    -------
    pl.DataFrame
        ``CARRY_SCHEMA`` rows, one per security with any state.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> panel = xr.Dataset(
    ...     {name: (("timestamp", "symbol"), np.array([[v], [np.nan]])) for name, v in
    ...      {"close": 10.0, "volume": 5.0, "adjClose": 20.0, "adjVolume": 10.0, "cumfacshr": 1.0,
    ...       "adjOpen": 20.0, "adjHigh": 20.0, "adjLow": 20.0}.items()},
    ...     coords={"timestamp": np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[ns]"), "symbol": [7]},
    ... )
    >>> carry_state(panel).row(0)
    (7, 10.0, 20.0, 1.0, 2.0)
    """
    symbols = np.asarray(panel["symbol"].values, dtype=np.int64)

    def last(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """The last value of each column where ``valid``, NaN where never."""
        rows = np.where(valid, np.arange(len(values))[:, None], -1).max(axis=0)
        found = rows >= 0
        out = np.full(values.shape[1], np.nan)
        out[found] = values[rows[found], np.flatnonzero(found)]
        return out

    close = panel["close"].values
    adj = panel["adjClose"].values
    cum = panel["cumfacshr"].values
    volume = panel["volume"].values
    adj_volume = panel["adjVolume"].values
    priced = (close > 0) & np.isfinite(adj)
    scaled = (volume > 0) & np.isfinite(adj_volume) & np.isfinite(cum) & (cum != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = adj_volume / (volume * cum)
    now = pl.DataFrame(
        {
            "symbol": symbols,
            "close": last(close, priced),
            "adjClose": last(adj, priced),
            "cumfacshr": last(cum, np.isfinite(cum)),
            "volume_scale": last(scale, scaled),
        },
        schema=CARRY_SCHEMA,
    ).fill_nan(None)
    if earlier is None or earlier.height == 0:
        state = now
    else:
        state = (
            earlier.join(now, on="symbol", how="full", coalesce=True, suffix="_now")
            .select(
                "symbol",
                *(pl.coalesce(f"{name}_now", name).alias(name) for name in list(CARRY_SCHEMA)[1:]),
            )
        )
    known = pl.any_horizontal(pl.col(name).is_not_null() for name in list(CARRY_SCHEMA)[1:])
    return state.filter(known).select(list(CARRY_SCHEMA)).sort("symbol")
