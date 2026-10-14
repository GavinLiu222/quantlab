"""Cleaning and validation rules shared by every dataset class.

A *panel* is an ``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``.
It is *dense*: it has a cell for every timestamp and symbol pair, and a
symbol that did not trade at a timestamp is NaN there. OHLCV means the open,
high, low, close and volume columns of a price bar.

Raw market data passes through this module at two stages.
``dedup_raw_frame`` runs on the long-format polars frame before it is
converted to xarray, because a duplicated ``(timestamp, symbol)`` pair makes
that conversion fail. ``clean_market_data`` runs once on the resulting panel;
it only checks the schema and flags anomalies, and leaves the NaN gaps of the
dense panel alone. ``clean_membership_panel`` and ``clean_nbbo_panel`` are the
matching validators for an index-membership panel and for an NBBO quote
panel (NBBO, the National Best Bid and Offer, is the best bid and ask across
all US exchanges). Neither of those has OHLCV columns.

Nothing here fills, interpolates or corrects a value. A gap or an outlier is
reported, never repaired, so the pipeline only ever holds observed data.
The module imports only third-party libraries.
"""

from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Price variables checked for zero or negative values and extreme jumps,
# including the split- and dividend-adjusted versions some vendors supply.
_PRICE_LIKE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "adjOpen",
    "adjHigh",
    "adjLow",
    "adjClose",
)

# A single-step percentage change in ``close`` beyond this fraction is
# flagged as an extreme jump.
_EXTREME_JUMP_THRESHOLD = 0.5


def dedup_raw_frame(
    data: pl.LazyFrame, keep: Literal["first", "last"] = "last"
) -> pl.LazyFrame:
    """Drop duplicate ``(timestamp, symbol)`` rows, keeping a fixed one.

    This must run before the frame is converted to xarray, which rejects a
    non-unique index. The default ``keep="last"`` is chosen because when a
    vendor delivers files periodically, a later file is more likely to hold
    corrected data than an earlier one.

    Parameters
    ----------
    data : pl.LazyFrame
        A long-format frame with ``timestamp`` and ``symbol`` columns.
    keep : {"first", "last"}, default "last"
        Which duplicate to keep.

    Returns
    -------
    pl.LazyFrame
        The frame with at most one row per ``(timestamp, symbol)`` pair.

    Examples
    --------
    >>> frame = pl.DataFrame({
    ...     "timestamp": ["2024-01-02", "2024-01-02"],
    ...     "symbol": ["AAA", "AAA"],
    ...     "close": [10.0, 10.5],
    ... }).lazy()
    >>> dedup_raw_frame(frame).collect()["close"].to_list()
    [10.5]
    >>> dedup_raw_frame(frame, keep="first").collect()["close"].to_list()
    [10.0]
    """
    return data.unique(subset=["timestamp", "symbol"], keep=keep)


def flag_anomalies(data: xr.Dataset) -> xr.Dataset:
    """Add a boolean ``anomaly_flag`` variable marking suspicious prices.

    A cell is flagged when any price variable present is zero or negative,
    or when ``close`` changes by more than ``_EXTREME_JUMP_THRESHOLD`` (50%)
    from a strictly positive previous close. The values themselves are never
    changed or dropped, so an anomaly stays visible for later investigation.
    If any cell is flagged, a warning with the count is logged.

    Parameters
    ----------
    data : xr.Dataset
        A panel on ``(timestamp, symbol)``.

    Returns
    -------
    xr.Dataset
        ``data`` with an additional ``anomaly_flag`` variable on the same
        dimensions.

    Examples
    --------
    ``BBB`` prints a zero close and ``AAA`` jumps from 10.5 to 20.0:

    >>> panel["close"].values
    array([[10. ,  5. ],
           [10.5,  0. ],
           [20. ,  5.2],
           [20.5,  5.3]])
    >>> flag_anomalies(panel)["anomaly_flag"].values
    array([[False, False],
           [False,  True],
           [ True, False],
           [False, False]])
    """
    present_price_columns = [
        c for c in _PRICE_LIKE_COLUMNS if c in data.data_vars
    ]

    if present_price_columns:
        anomaly = xr.zeros_like(data[present_price_columns[0]], dtype=bool)
    else:
        anomaly = xr.DataArray(
            np.zeros(
                (data.sizes.get("timestamp", 0), data.sizes.get("symbol", 0)),
                dtype=bool,
            ),
            dims=["timestamp", "symbol"],
            coords={"timestamp": data["timestamp"], "symbol": data["symbol"]},
        )

    for col in present_price_columns:
        anomaly = anomaly | (data[col] <= 0)

    if "close" in data.data_vars:
        close = data["close"]
        shifted = close.shift(timestamp=1)
        pct_change = close.diff(dim="timestamp") / shifted
        # A percentage change from a zero, negative or NaN close is
        # meaningless, so only a strictly positive previous close counts.
        # Comparisons with NaN are False.
        valid_prior = shifted > 0
        jump = (np.abs(pct_change) > _EXTREME_JUMP_THRESHOLD) & valid_prior
        # ``diff`` drops the first timestamp; put it back as "no jump".
        jump = jump.reindex(timestamp=data["timestamp"], fill_value=False)
        anomaly = anomaly | jump

    anomaly = anomaly.astype(bool)
    anomaly.name = "anomaly_flag"

    flagged_count = int(anomaly.sum().item())
    if flagged_count > 0:
        logger.warning(
            f"flag_anomalies: flagged {flagged_count} anomalous "
            f"(timestamp, symbol) data point(s) (zero/negative price or "
            f"extreme jump); values left unmodified, see `anomaly_flag`."
        )

    return data.assign(anomaly_flag=anomaly)


def check_bad_print_rule(jump: float, volume_ratio: float, lookback: int) -> None:
    """Raise ``ValueError`` for a parameter ``bad_print_mask`` cannot use.

    Examples
    --------
    >>> check_bad_print_rule(5.0, 20.0, 20)
    >>> check_bad_print_rule(1.0, 20.0, 20)
    Traceback (most recent call last):
    ValueError: bad print rule: jump must be above 1, got 1.0.
    """
    if not jump > 1:
        raise ValueError(f"bad print rule: jump must be above 1, got {jump}.")
    if not volume_ratio > 0:
        raise ValueError(f"bad print rule: volume_ratio must be positive, got {volume_ratio}.")
    if lookback < 1:
        raise ValueError(f"bad print rule: lookback must be at least 1, got {lookback}.")


def bad_print_mask(
    price: np.ndarray,
    volume: np.ndarray,
    *,
    jump: float = 5.0,
    volume_ratio: float = 20.0,
    lookback: int = 20,
) -> np.ndarray:
    """Return where a bar is a bad print: a large one-bar move on ordinary volume.

    A bar of a symbol is a *candidate* when its price moves more than
    ``jump`` times, up or down, from the symbol's last priced bar among the
    ``lookback`` bars before it, on a volume below ``volume_ratio`` times the
    mean volume of those bars (a halted bar's 0 counted in the mean). A
    candidate is a bad print unless that last priced bar is itself a
    candidate: then it is the price coming back, a real one. A vendor's bad
    print (a close of $0.01 between two of $7 and $9) trades like any other
    day, while a real jump of that size trades many times its usual volume.
    A bar whose lookback has no traded bar (a halt longer than the lookback)
    is never flagged: there is no volume to judge it by. Row ``t`` reads only
    rows ``t - 2 lookback`` to ``t``, so a live feed and a backtest flag the
    same bars, and any window starting ``2 lookback`` rows earlier gives the
    same flags.

    A bad print lasting several bars has only its first bar flagged: the
    bars after it move little from it and keep their price. The bar back
    from its last bar is flagged as well when it is a candidate, since the
    bar before it is not.

    The defaults are our choice (#223). On the Sharadar SEP history (adjusted
    close and adjusted volume, 7,233 daily bars across 17,067 permatickers)
    they flag 535 bars of 321 permatickers. Of the 75 one-bar moves of more
    than 5 times that return within 0.8 to 1.25 times the previous close
    the next bar, 63 are flagged: most of the rest belong to one series that
    swings between two levels 800 times apart every bar (permaticker 116211
    in 2002), where every other bar is taken as the price coming back. Of
    the 1,101 such moves that do not return, 472 are flagged, among them bad
    prints lasting several bars and real moves of thinly traded stocks. Real
    jumps such as TPST on 2023-10-11 (40 times, on
    246 times the mean volume) and SIVB's first print after its halt on
    2023-03-28 (265 times lower, on 33 times the mean) are not flagged.

    Parameters
    ----------
    price : np.ndarray
        ``[T, S]`` split- and dividend-adjusted prices, NaN where a symbol
        has none. Adjusted, so a split is no move.
    volume : np.ndarray
        ``[T, S]`` volumes on the same share basis.
    jump : float, default 5.0
        Fold move, above 1, beyond which a bar is a candidate.
    volume_ratio : float, default 20.0
        A candidate trades less than this many times the lookback's mean
        volume. Positive.
    lookback : int, default 20
        Bars before a bar the previous price and the mean volume are read
        from. At least 1.

    Returns
    -------
    np.ndarray
        ``[T, S]`` booleans; ``True`` marks a bad print.

    Raises
    ------
    ValueError
        If ``jump`` is not above 1, ``volume_ratio`` not positive or
        ``lookback`` below 1.

    Examples
    --------
    One bar at a hundredth of the price, on its usual volume; the bar back
    is real:

    >>> price = np.array([[10.0], [10.1], [0.1], [10.2], [10.1]])
    >>> volume = np.full((5, 1), 1_000.0)
    >>> bad_print_mask(price, volume).ravel().tolist()
    [False, False, True, False, False]

    A hundredfold rise that stays, on a hundred times the volume, is real:

    >>> price = np.array([[10.0], [10.1], [1010.0], [1020.0], [1015.0]])
    >>> volume[2] = 100_000.0
    >>> bad_print_mask(price, volume).ravel().tolist()
    [False, False, False, False, False]
    """
    check_bad_print_rule(jump, volume_ratio, lookback)
    price = np.asarray(price, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    def at_last_priced_bar(values: np.ndarray) -> np.ndarray:
        """Each bar's ``values`` at the last priced bar among the ``lookback`` before it."""
        priced = pd.DataFrame(np.where(np.isfinite(price), values, np.nan))
        return priced.shift(1).ffill(limit=lookback - 1).to_numpy()

    before = pd.DataFrame(volume).shift(1).rolling(lookback, min_periods=1)
    mean_volume, most_volume = before.mean().to_numpy(), before.max().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        move = price / at_last_priced_bar(price)
        ordinary = (most_volume > 0) & ~(volume >= volume_ratio * mean_volume)
    candidate = np.isfinite(move) & ((move > jump) | (move < 1.0 / jump)) & ordinary
    return candidate & ~(at_last_priced_bar(candidate.astype(np.float64)) == 1.0)


def intrabar_bad_print_mask(price: np.ndarray, close: np.ndarray, *, jump: float = 5.0, lookback: int = 20) -> np.ndarray:
    """Return where an intrabar price (an open, high or low) is a bad print of its own.

    A vendor can get one price of a bar wrong and the rest right: Sharadar
    SEP opens SLND (permaticker 636783) at 2000.00 on 2023-02-16, a bar
    that closed at 8.601 after a close of 8.94, while its trades that day
    ran from 8.10 to 10.00. ``bad_print_mask`` reads the close and cannot
    see it. A price is flagged when it is more than ``jump`` times away, up
    or down, both from its bar's close and from the last priced close among
    the ``lookback`` bars before it: a real gap is near one of them (a stock
    that gaps up and holds closes near its open). It reads the bar and the
    bars before it only, all known at the bar's close.

    On the Sharadar SEP history, at 5 times, it finds 174 opens of 129
    permatickers the close rule leaves (25 since 2016-10, every one of them
    wrong: SLND, opens of fractions of a cent), 135 highs and 446 lows. A
    real day's high or low can fail it: TPST gapped up 7 times on
    2023-10-11 and closed 40 times up, so its low is more than 5 times from
    either close. ``BadPrintMaskedDataset`` checks the open only by default.

    Parameters
    ----------
    price : np.ndarray
        ``[T, S]`` adjusted intrabar prices.
    close : np.ndarray
        ``[T, S]`` adjusted closes, on the same basis.
    jump : float, default 5.0
        Fold distance, above 1, beyond which a price is far.
    lookback : int, default 20
        Bars before a bar the last priced close is read from. At least 1.

    Returns
    -------
    np.ndarray
        ``[T, S]`` booleans; ``True`` marks a bad intrabar price.

    Raises
    ------
    ValueError
        If ``jump`` is not above 1 or ``lookback`` below 1.

    Examples
    --------
    >>> close = np.array([[8.9], [8.94], [8.6], [8.4]])
    >>> opens = np.array([[9.0], [8.9], [2000.0], [8.5]])
    >>> intrabar_bad_print_mask(opens, close).ravel().tolist()
    [False, False, True, False]
    """
    check_bad_print_rule(jump, 1.0, lookback)
    price = np.asarray(price, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    before = pd.DataFrame(close).shift(1).ffill(limit=lookback - 1).to_numpy()

    def far(reference: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = price / reference
        return np.isfinite(ratio) & ((ratio > jump) | (ratio < 1.0 / jump))

    return far(close) & far(before)


def validate_schema(
    data: xr.Dataset, required_columns: tuple[str, ...] = REQUIRED_COLUMNS
) -> xr.Dataset:
    """Check that the required columns exist and report unexpected nulls.

    A dense panel has a cell for every timestamp and symbol, so a symbol that
    did not trade at a timestamp is null in every variable there. Such cells
    are expected and are left out of the null counts. A cell counts as "no
    bar" when every column in ``required_columns`` is null there, and only
    nulls on cells that do have a bar are warned about. A column laid out on
    other dimensions than the required columns is counted in full instead.

    If every required column is null in every cell, the panel is treated as
    an empty ingest (for example an empty vendor response). That is logged
    as an error, the "no bar" exclusion is switched off, and each column's
    full null count is reported. Nulls never raise; only a missing column
    does, because a data problem must not abort an ingest running unattended.

    Parameters
    ----------
    data : xr.Dataset
        A panel on ``(timestamp, symbol)``.
    required_columns : tuple of str, default ``REQUIRED_COLUMNS``
        The variables that must be present. Callers whose columns are
        spelled differently pass their own tuple; the "no bar" test uses
        these columns too.

    Returns
    -------
    xr.Dataset
        ``data``, unchanged.

    Raises
    ------
    ValueError
        If any of ``required_columns`` is missing.

    Examples
    --------
    >>> validate_schema(ohlcv) is ohlcv
    True
    >>> validate_schema(ohlcv.drop_vars("volume"))
    Traceback (most recent call last):
    ValueError: validate_schema: required column(s) missing from dataset: ['volume']
    >>> validate_schema(quotes, required_columns=("bid", "ask")) is quotes
    True
    """
    missing = [col for col in required_columns if col not in data.data_vars]
    if missing:
        raise ValueError(
            f"validate_schema: required column(s) missing from dataset: "
            f"{missing}"
        )

    structural_mask = None
    for col in required_columns:
        is_null = data[col].isnull()
        structural_mask = (
            is_null if structural_mask is None else (structural_mask & is_null)
        )

    empty_ingest = False
    if structural_mask is not None:
        structural_cells = int(structural_mask.sum().item())
        total_cells = int(structural_mask.size)
        if total_cells > 0 and structural_cells == total_cells:
            empty_ingest = True
            logger.error(
                f"validate_schema: EVERY required column is null on every one "
                f"of the {total_cells} (timestamp, symbol) cell(s); no bar "
                f"exists anywhere in this panel. That is not the normal gaps "
                f"of a dense panel; it is an empty ingest: an empty vendor "
                f"response, a mis-parsed file, or a fully failed backfill. "
                f"Required columns: {list(required_columns)}. Empty cells are "
                f"not excluded for this panel, so the per-column counts below "
                f"are whole-column null counts. Not raising: data problems "
                f"are flagged, never deleted or fatal."
            )
        elif structural_cells > 0 and total_cells > 0:
            logger.info(
                f"validate_schema: {structural_cells}/{total_cells} "
                f"({structural_cells / total_cells:.1%}) "
                f"(timestamp, symbol) cell(s) hold no bar at all (null in "
                f"every required column). Those are the normal gaps of a dense "
                f"panel, not anomalies; nulls on those cells are excluded "
                f"from the counts below."
            )

    for col in data.data_vars:
        column = data[col]
        if empty_ingest:
            # With every cell masked, the next branch would report zero for
            # every column, so fall back to whole-column counts.
            null_count = int(column.isnull().sum().item())
            scope = (
                "raw null value(s), whole-column count, because empty cells "
                "are not excluded on an empty-ingest panel (see the ERROR "
                "above)"
            )
        elif structural_mask is not None and tuple(column.dims) == tuple(
            structural_mask.dims
        ):
            null_count = int(
                (column.isnull() & ~structural_mask).sum().item()
            )
            scope = (
                "null value(s) on (timestamp, symbol) cells where a bar does "
                "exist"
            )
        else:
            null_count = int(column.isnull().sum().item())
            scope = (
                f"null value(s); its dims {tuple(column.dims)} differ from "
                f"the required-column grid, so the whole column is counted"
            )
        if null_count > 0:
            logger.warning(
                f"validate_schema: column '{col}' has {null_count} {scope} "
                f"(not raising: data problems are flagged, never deleted or fatal)."
            )

    return data


def clean_market_data(data: xr.Dataset) -> xr.Dataset:
    """Validate the schema of a market panel and flag its anomalies.

    This is the one cleaning step applied to every market-data panel after
    it has been converted to xarray. Duplicates were already removed from
    the polars frame, and the NaN gaps of the dense panel are expected, so
    nothing here fills or changes a value.

    Parameters
    ----------
    data : xr.Dataset
        A panel on ``(timestamp, symbol)`` carrying the OHLCV columns.

    Returns
    -------
    xr.Dataset
        ``data`` with an ``anomaly_flag`` variable added.

    Examples
    --------
    >>> cleaned = clean_market_data(ohlcv)
    >>> list(cleaned.data_vars)
    ['open', 'high', 'low', 'close', 'volume', 'anomaly_flag']
    >>> cleaned["anomaly_flag"].dtype
    dtype('bool')
    """
    data = validate_schema(data)
    data = flag_anomalies(data)
    return data


def clean_membership_panel(data: xr.Dataset) -> xr.Dataset:
    """Validate an index-membership panel and return it unchanged.

    An index-membership panel says, for each day and symbol, whether the
    symbol belonged to the index. The OHLCV checks do not apply:
    ``validate_schema`` would raise on the missing price columns, and
    ``flag_anomalies`` would add a meaningless all-False flag. Nothing is
    modified, filled or re-sorted. A panel that breaks these rules points to
    a bug in the code that built it, and repairing it here would hide that.

    Parameters
    ----------
    data : xr.Dataset
        A panel whose only variable is a boolean ``is_member`` on
        ``(timestamp, symbol)``.

    Returns
    -------
    xr.Dataset
        ``data``, unchanged.

    Raises
    ------
    ValueError
        If the variables are not exactly ``{"is_member"}``, if
        ``is_member`` is not boolean or not on ``("timestamp", "symbol")``,
        or if the ``timestamp`` coordinate is not strictly increasing.
        The last case is refused because slicing by date label silently
        returns wrong results on an unsorted index.

    Examples
    --------
    >>> clean_membership_panel(membership) is membership
    True
    >>> clean_membership_panel(membership.astype(int))
    Traceback (most recent call last):
    ValueError: clean_membership_panel: 'is_member' must have dtype bool, got int64
    """
    variables = set(data.data_vars)
    if variables != {"is_member"}:
        raise ValueError(
            f"clean_membership_panel: expected exactly one data variable "
            f"'is_member', got {sorted(variables)}"
        )

    is_member = data["is_member"]
    if is_member.dtype != np.dtype(bool):
        raise ValueError(
            f"clean_membership_panel: 'is_member' must have dtype bool, got "
            f"{is_member.dtype}"
        )
    if tuple(is_member.dims) != ("timestamp", "symbol"):
        raise ValueError(
            f"clean_membership_panel: 'is_member' must have dims "
            f"('timestamp', 'symbol'), got {tuple(is_member.dims)}"
        )

    timestamps = data["timestamp"].values
    # Compare neighbours directly instead of ``np.diff(...) > 0``, which works
    # for any dtype (comparing a timedelta64 with a bare ``0`` is deprecated).
    if timestamps.size > 1 and not np.all(timestamps[:-1] < timestamps[1:]):
        raise ValueError(
            "clean_membership_panel: the 'timestamp' coordinate must be "
            "strictly increasing (no duplicates, no out-of-order rows), "
            "because XrBackend.filter_by_date slices it with "
            ".sel(slice(...)), which silently returns wrong results on an "
            "unsorted index."
        )

    return data


#: The exact variables of an NBBO bar panel, all float64. The two counts,
#: ``n_updates`` and ``n_ambiguous_ties``, are float64 too: integers are
#: promoted to float when data is appended anyway, and a count must be able
#: to hold NaN for a symbol that joins the panel later.
NBBO_PANEL_VARIABLES = (
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "mid",
    "spread",
    "spread_bps",
    "imbalance",
    "n_updates",
    "tw_spread",
    "tw_bid_size",
    "tw_ask_size",
    "n_ambiguous_ties",
)


def clean_nbbo_panel(data: xr.Dataset) -> xr.Dataset:
    """Validate an NBBO quote-bar panel and return it unchanged.

    An NBBO quote-bar panel holds, per bar and symbol, the best bid and ask
    across US exchanges and statistics derived from them. Like
    ``clean_membership_panel``, this never modifies, fills or re-sorts. The
    OHLCV checks do not apply because the panel has no price columns.
    Carrying the last quote forward over a bar with no updates is done by
    the resampler that builds the panel, not here.

    Parameters
    ----------
    data : xr.Dataset
        A panel whose variables are exactly ``NBBO_PANEL_VARIABLES``.

    Returns
    -------
    xr.Dataset
        ``data``, unchanged.

    Raises
    ------
    ValueError
        If the variable set differs from ``NBBO_PANEL_VARIABLES``,
        if any variable is not float64 or not on ``("timestamp",
        "symbol")``, or if the ``timestamp`` coordinate is not strictly
        increasing.

    Examples
    --------
    >>> clean_nbbo_panel(quotes) is quotes
    True
    >>> clean_nbbo_panel(quotes.drop_vars("mid"))
    Traceback (most recent call last):
    ValueError: clean_nbbo_panel: expected exactly the variables ['ask', ...
    """
    variables = set(data.data_vars)
    expected = set(NBBO_PANEL_VARIABLES)
    if variables != expected:
        raise ValueError(
            f"clean_nbbo_panel: expected exactly the variables "
            f"{sorted(expected)}, got {sorted(variables)} (missing "
            f"{sorted(expected - variables)}, unexpected "
            f"{sorted(variables - expected)})"
        )

    for name in NBBO_PANEL_VARIABLES:
        variable = data[name]
        if variable.dtype != np.dtype("float64"):
            raise ValueError(
                f"clean_nbbo_panel: {name!r} must have dtype float64, got "
                f"{variable.dtype}"
            )
        if tuple(variable.dims) != ("timestamp", "symbol"):
            raise ValueError(
                f"clean_nbbo_panel: {name!r} must have dims "
                f"('timestamp', 'symbol'), got {tuple(variable.dims)}"
            )

    timestamps = data["timestamp"].values
    # Compare neighbours directly, as in ``clean_membership_panel``.
    if timestamps.size > 1 and not np.all(timestamps[:-1] < timestamps[1:]):
        raise ValueError(
            "clean_nbbo_panel: the 'timestamp' coordinate must be strictly "
            "increasing (no duplicates, no out-of-order rows), because "
            "XrBackend.filter_by_date slices it with .sel(slice(...)), which "
            "silently returns wrong results on an unsorted index."
        )

    return data
