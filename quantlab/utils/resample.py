"""Shared pieces of the resample contract: config checks, bucket labels, paths.

A *resample* aggregates a ``(timestamp, symbol)`` panel onto a coarser
regular time grid, one aggregation method per variable. A factor carries the
request in its config (``resample_freq`` and ``resample_how``), its dataset
maps each source timestamp to the bar it belongs to (``resample_labels``),
and the storage backend does the grouping (its ``resample``). The helpers
here are the parts of that contract the layers share.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.enums.data import RESAMPLE_FREQUENCY_SECONDS, RESAMPLE_METHODS


def validate_resample_config(
    freq: str | None, how: dict[str, str] | str | None, owner: str
) -> None:
    """Check a config's ``resample_freq`` and ``resample_how`` pair.

    Parameters
    ----------
    freq : str or None
        The ``resample_freq`` field.
    how : dict[str, str] or str or None
        The ``resample_how`` field.
    owner : str
        Class name quoted in error messages.

    Raises
    ------
    ValueError
        If ``freq`` is not a ``ResampleFrequency`` token, if ``how`` is set
        without ``freq`` or ``freq`` without ``how``, or if any method is
        not a ``ResampleMethod`` token.

    Examples
    --------
    >>> validate_resample_config("1d", {"close": "last"}, "IntradayStats")
    >>> validate_resample_config("1d", None, "IntradayStats")
    Traceback (most recent call last):
    ValueError: IntradayStats: resample_freq='1d' needs resample_how ...
    """
    if freq is None and how is None:
        return
    if freq is None:
        raise ValueError(
            f"{owner}: resample_how is set but resample_freq is None. Set "
            f"both to resample, or neither to keep the store's own bars."
        )
    if freq not in RESAMPLE_FREQUENCY_SECONDS:
        raise ValueError(
            f"{owner}: resample_freq {freq!r} is not one of "
            f"{list(RESAMPLE_FREQUENCY_SECONDS)}."
        )
    if how is None:
        raise ValueError(
            f"{owner}: resample_freq={freq!r} needs resample_how: one "
            f"method for every variable (a str) or a {{variable: method}} "
            f"dict. Methods: {list(RESAMPLE_METHODS)}."
        )
    methods = [how] if isinstance(how, str) else list(how.values())
    if not isinstance(how, (str, dict)):
        raise ValueError(
            f"{owner}: resample_how must be a str or a dict, got "
            f"{type(how).__name__}."
        )
    bad = sorted({m for m in methods if m not in RESAMPLE_METHODS})
    if bad:
        raise ValueError(
            f"{owner}: resample_how uses unknown method(s) {bad}; choose "
            f"from {list(RESAMPLE_METHODS)}."
        )


def resolve_resample_how(
    how: dict[str, str] | str, variables: list[str], owner: str
) -> dict[str, str]:
    """Return one method per variable of the panel being resampled.

    A string applies to every variable. A dict must name every variable:
    an unlisted variable is refused rather than silently dropped or given
    a default, and a listed name the panel does not have is refused too.

    Parameters
    ----------
    how : dict[str, str] or str
        The ``resample_how`` field.
    variables : list[str]
        The panel's data variables.
    owner : str
        Class name quoted in error messages.

    Examples
    --------
    >>> resolve_resample_how("last", ["bid", "ask"], "IntradayStats")
    {'bid': 'last', 'ask': 'last'}
    >>> resolve_resample_how({"bid": "last"}, ["bid", "ask"], "IntradayStats")
    Traceback (most recent call last):
    ValueError: IntradayStats: resample_how does not name ['ask'] ...
    """
    if isinstance(how, str):
        return {name: how for name in variables}
    missing = [name for name in variables if name not in how]
    if missing:
        raise ValueError(
            f"{owner}: resample_how does not name {missing}; every variable "
            f"of the panel needs a method (or pass one method as a str)."
        )
    unknown = [name for name in how if name not in variables]
    if unknown:
        raise ValueError(
            f"{owner}: resample_how names {unknown}, which the panel does "
            f"not have. Variables: {list(variables)}."
        )
    return {name: how[name] for name in variables}


def resample_seconds(freq: str) -> int:
    """Return the length of a ``ResampleFrequency`` token in seconds.

    Examples
    --------
    >>> resample_seconds("1h")
    3600
    """
    return RESAMPLE_FREQUENCY_SECONDS[freq]


def source_interval_seconds(timestamps: np.ndarray) -> float:
    """Return the most common spacing of ``timestamps``, in seconds.

    The mode is used rather than the minimum so that gaps such as
    weekends do not distort the answer.

    Examples
    --------
    >>> source_interval_seconds(pd.date_range("2024-01-01", periods=3, freq="min"))
    60.0
    """
    index = pd.DatetimeIndex(timestamps)
    if len(index) < 2:
        raise ValueError(
            "a panel needs at least two timestamps to know its bar size"
        )
    return float(pd.Series(np.diff(index.values)).mode().iloc[0] / np.timedelta64(1, "s"))


def assert_coarser(timestamps: np.ndarray, freq: str, owner: str) -> None:
    """Refuse a resample onto a grid no coarser than the panel's own bars.

    Raises
    ------
    ValueError
        If ``freq`` is not longer than the panel's bar size.

    Examples
    --------
    >>> minutes = pd.date_range("2024-01-01", periods=10, freq="min")
    >>> assert_coarser(minutes, "5m", "IntradayStats")
    >>> assert_coarser(minutes, "1m", "IntradayStats")
    Traceback (most recent call last):
    ValueError: IntradayStats: resample_freq='1m' (60s) is not coarser ...
    """
    source = source_interval_seconds(timestamps)
    target = resample_seconds(freq)
    if target <= source:
        raise ValueError(
            f"{owner}: resample_freq={freq!r} ({target}s) is not coarser "
            f"than the panel's own bars ({source:g}s). A resample only "
            f"aggregates onto a coarser grid."
        )


def clock_labels(timestamps: np.ndarray, freq: str) -> np.ndarray:
    """Return each timestamp's bar, floored on the UTC clock.

    This is the default bucketing: a bar starts at a multiple of ``freq``
    counted from midnight UTC and is labelled at its start, which matches
    bars labelled at their open time and daily stores stamped at midnight.

    Examples
    --------
    >>> ts = pd.to_datetime(["2024-01-01 09:31", "2024-01-01 09:34", "2024-01-01 09:35"])
    >>> clock_labels(ts, "5m").astype("datetime64[m]").tolist()  # doctest: +SKIP
    [2024-01-01T09:30, 2024-01-01T09:30, 2024-01-01T09:35]
    """
    index = pd.DatetimeIndex(timestamps)
    return index.floor(f"{resample_seconds(freq)}s").values


def session_labels(
    timestamps: np.ndarray, freq: str, sessions: pd.DataFrame, owner: str
) -> np.ndarray:
    """Return each timestamp's bar inside its trading session.

    Bars here are labelled at their *end*, as ``NbboResampler`` labels them,
    so a source timestamp belongs to the session whose ``open < t <=
    close``. ``"1d"`` labels the whole session with its date at midnight, so
    a daily panel lines up with daily stores. Any other frequency cuts the
    session into right-closed bars from its open, labelled at their end; a
    partial last bar is labelled at the session close.

    Parameters
    ----------
    timestamps : np.ndarray
        Source timestamps, naive UTC.
    freq : str
        A ``ResampleFrequency`` token.
    sessions : pd.DataFrame
        Columns ``date``, ``open`` and ``close`` (naive UTC), one row per
        session, sorted by ``open``.
    owner : str
        Class name quoted in error messages.

    Raises
    ------
    ValueError
        If a timestamp falls in no session.

    Examples
    --------
    >>> sessions = pd.DataFrame({
    ...     "date": [pd.Timestamp("2024-01-24").date()],
    ...     "open": [pd.Timestamp("2024-01-24 14:30")],
    ...     "close": [pd.Timestamp("2024-01-24 21:00")],
    ... })
    >>> ts = pd.to_datetime(["2024-01-24 14:31", "2024-01-24 21:00"])
    >>> session_labels(ts, "1d", sessions, "Demo").astype("datetime64[D]").tolist()  # doctest: +SKIP
    [2024-01-24, 2024-01-24]
    """
    index = pd.DatetimeIndex(timestamps).values.astype("datetime64[ns]")
    opens = pd.to_datetime(sessions["open"]).values.astype("datetime64[ns]")
    closes = pd.to_datetime(sessions["close"]).values.astype("datetime64[ns]")
    dates = pd.to_datetime(sessions["date"]).values.astype("datetime64[ns]")

    # The session whose close is the first at or after t.
    idx = np.searchsorted(closes, index, side="left")
    outside = idx >= len(closes)
    idx_safe = np.where(outside, 0, idx)
    outside |= ~(opens[idx_safe] < index)
    if outside.any():
        first = pd.Timestamp(index[outside][0])
        raise ValueError(
            f"{owner}: timestamp {first} falls in no trading session; a "
            f"session-based resample needs every bar inside a session."
        )

    if freq == "1d":
        return dates[idx]

    step = np.timedelta64(resample_seconds(freq), "s").astype("timedelta64[ns]")
    offset = index - opens[idx]
    bars = -(-offset // step)  # ceil: a bar is labelled at its end
    labels = opens[idx] + bars * step
    return np.minimum(labels, closes[idx])


def resample_store_path(path: str | None, freq: str | None) -> str | None:
    """Return the store a resampled object reads and writes.

    The resampled store sits beside the source store with ``_resample_<freq>``
    added to its name: ``klines.zarr`` becomes ``klines_resample_1d.zarr``.
    With ``freq`` unset the source path is returned unchanged; a ``None``
    path stays ``None``.

    Examples
    --------
    >>> resample_store_path("data/klines.zarr", "1d")
    'data/klines_resample_1d.zarr'
    >>> resample_store_path("data/klines.zarr", None)
    'data/klines.zarr'
    """
    if path is None or freq is None:
        return path
    source = Path(path)
    return str(source.with_name(f"{source.stem}_resample_{freq}{source.suffix}"))
