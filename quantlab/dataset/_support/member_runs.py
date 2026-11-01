"""Membership intervals from a per-bar membership grid: one interval per run of member bars."""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["member_runs"]


def member_runs(mask: np.ndarray, days: np.ndarray, symbols: np.ndarray) -> pl.DataFrame:
    """Return one ``symbol``/``start_date``/``end_date`` row per run of consecutive member bars.

    A run's first and last bars are its interval's ends, both included, so
    the calendar days between two bars of one run (a weekend) read as member
    once ``IndexConstituentDataset._densify`` fills the interval.

    Parameters
    ----------
    mask : np.ndarray
        ``[T, S]`` booleans, True where a symbol is a member on a bar.
    days : np.ndarray
        The ``T`` bars, ``datetime64``.
    symbols : np.ndarray
        The ``S`` symbol labels; integer labels come back as int64.

    Returns
    -------
    pl.DataFrame
        The intervals, ``start_date`` and ``end_date`` as ``pl.Date``.

    Examples
    --------
    >>> days = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], "datetime64[D]")
    >>> mask = np.array([[True, False], [True, True], [False, True]])
    >>> member_runs(mask, days, np.array([7, 9])).rows()
    [(7, datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)), (9, datetime.date(2024, 1, 2), datetime.date(2024, 1, 3))]
    """
    days = np.asarray(days).astype("datetime64[D]")
    edge = np.zeros((1, mask.shape[1]), bool)
    step = np.diff(np.vstack([edge, mask, edge]).astype(np.int8), axis=0)
    start_t, start_s = np.nonzero(step == 1)
    end_t, end_s = np.nonzero(step == -1)
    # Both sides in (symbol, time) order, so the k-th start pairs with the k-th end.
    order_start = np.lexsort((start_t, start_s))
    order_end = np.lexsort((end_t, end_s))
    labels = symbols[start_s[order_start]]
    return pl.DataFrame(
        {
            "symbol": labels.astype(np.int64) if labels.dtype.kind in "iu" else labels.astype(str),
            "start_date": days[start_t[order_start]],
            "end_date": days[end_t[order_end] - 1],
        }
    ).with_columns(pl.col("start_date").cast(pl.Date), pl.col("end_date").cast(pl.Date))
