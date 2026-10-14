"""Evaluation: score a trained unit's predictions against its raw labels.

A model and an ensemble are both scored here, after training, by one rule:
``evaluate`` takes the unit's prediction panel (a model's ``predict_panel``
over its collected panel, an ensemble's combined prediction), the panel of
the labels' raw values and the unit's train / validation / test segments,
returns the metrics and writes the unit's two evaluation files,
``ic_series.csv`` and ``test_predictions.zarr``. It needs no model
instance, so predictions can be scored without training anything.

Scoring rules, one set for every unit:

- Every label is scored on every non-empty segment. The first label's keys
  are ``{split}_{metric}``, every other label's ``{split}_{label}_{metric}``.
- The IC family (``ic``, ``rank_ic``, ``icir``, ``rank_icir``, see
  ``ic_panel_metrics``) always, and ``rank_autocorrelation``, the mean of
  ``rank_autocorrelation_series`` over the segment: how much the
  prediction's order persists from one bar to the next, which bounds the
  turnover of a book that follows it.
- The error metrics ``mse``, ``rmse``, ``mae`` and ``r2`` only for a label
  whose prediction is on its own scale (label scale ``"raw"``): a
  standardized prediction is in other units than the label.
- ``qlike`` and ``variance_ratio`` (``volatility_level_metrics``) where
  ``scores_volatility_level`` holds: a volatility label predicted raw.
- whatever ``extra_metrics`` returns for a label and split, when given: the
  caller's own metrics on the same bars and symbols (an ensemble adds how much
  its members agree, ``member_correlation``). ``evaluate`` itself knows no
  ensemble.

**The panel metrics.** MSE, RMSE, MAE, R2, IC and RankIC, and the level metrics
of a volatility prediction, are defined here too.

The return models score their predictions with these functions. Predictions
and targets are ``[T, S]`` panels (time by symbol), and every function follows
the same conventions: inputs are cast to float64 and must share a shape; only
cells where both prediction and target are finite take part in any sum or
ranking; and an empty set of usable cells yields NaN without raising a
``RuntimeWarning``.

The IC (information coefficient) measures how well predictions order the
symbols. Cross-sectional IC is the Pearson correlation between prediction and
target across the symbols of one timestamp, averaged over time. RankIC ranks
each row first (average ranks on ties) and then computes IC on the ranks, so
it is a per-timestamp Spearman correlation and is not dominated by
outliers. Both are fully vectorised, with no per-row Python loop, since a
panel can hold tens of thousands of timestamps.

The per-timestamp values behind both means are available as series
(``cross_sectional_ic_series``, ``cross_sectional_rank_ic_series``), NaN on a
skipped row. The ICIR (information ratio of the IC) divides the mean of the
valid values of such a series by their sample standard deviation
(``information_ratio``), so it measures how stable a signal is, not only how
strong.

The IC of a volatility prediction measures only how well it ranks the
symbols' volatility. A consumer that uses the predicted level, such as a
covariance built from it, also needs the level right, which
``volatility_level_metrics`` scores: the QLIKE loss and the ratio of realised
to predicted variance.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata

from quantlab.model.split import purge_segments
from quantlab.runs.trained_run import evaluation_paths


def _joint(pred, target) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast both inputs to float64, check shapes, and return them with the joint mask.

    The joint mask marks cells where both inputs are finite.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(pred, target, mask)`` where ``mask`` is True where both are finite.

    Raises
    ------
    ValueError
        If the two inputs differ in shape.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {p.shape} vs {t.shape}"
        )
    return p, t, np.isfinite(p) & np.isfinite(t)


def mse(pred, target) -> float:
    """Return the mean squared error over the jointly finite cells, or NaN if none.

    Parameters
    ----------
    pred : array_like
        Predictions, any shape.
    target : array_like
        Realised values, the same shape as ``pred``.

    Examples
    --------
    >>> mse([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    1.0
    >>> mse([1.0, np.nan], [np.nan, 2.0])
    nan
    """
    p, t, mask = _joint(pred, target)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    diff = p[mask] - t[mask]
    return float(np.dot(diff, diff) / n)


def rmse(pred, target) -> float:
    """Return ``sqrt(mse(pred, target))``, or NaN when the MSE is undefined.

    Parameters
    ----------
    pred : array_like
        Predictions, any shape.
    target : array_like
        Realised values, the same shape as ``pred``.

    Examples
    --------
    >>> rmse([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    1.0
    """
    value = mse(pred, target)
    return float(np.sqrt(value)) if np.isfinite(value) else float("nan")


def mae(pred, target) -> float:
    """Return the mean absolute error over the jointly finite cells, or NaN if none.

    Parameters
    ----------
    pred : array_like
        Predictions, any shape.
    target : array_like
        Realised values, the same shape as ``pred``.

    Examples
    --------
    >>> mae([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    0.5
    """
    p, t, mask = _joint(pred, target)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.abs(p[mask] - t[mask]).sum() / n)


def r2(pred, target) -> float:
    """Return the coefficient of determination ``1 - SS_res / SS_tot``.

    NaN is returned when fewer than two cells are usable or when the target is
    constant over the usable cells (``SS_tot == 0``), because R2 is undefined
    in both cases.

    Parameters
    ----------
    pred : array_like
        Predictions, any shape.
    target : array_like
        Realised values, the same shape as ``pred``.

    Examples
    --------
    >>> r2([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
    0.7142857142857143
    >>> r2([1.0, 2.0], [3.0, 3.0])
    nan
    """
    p, t, mask = _joint(pred, target)
    n = int(mask.sum())
    if n < 2:
        return float("nan")
    tv = t[mask]
    ss_tot = float(np.sum((tv - tv.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((tv - p[mask]) ** 2))
    return 1.0 - ss_res / ss_tot


def cross_sectional_ic_series(pred, target) -> np.ndarray:
    """Return the per-row Pearson correlation of a panel, NaN on a skipped row.

    A row (timestamp) is skipped when it has fewer than two usable symbols or
    when either the prediction or the target is constant over its usable
    symbols. Constancy is tested exactly (masked max equals masked min) rather
    than with a variance threshold, so genuinely small cross-sectional spreads
    are not misread as constant.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.

    Returns
    -------
    np.ndarray
        A float64 array of length ``T``.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.

    Examples
    --------
    >>> pred = np.array([[1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [1.0, 2.0, 3.0]])
    >>> target = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 4.0], [1.0, 3.0, 2.0]])
    >>> cross_sectional_ic_series(pred, target)
    array([1. , 1. , 0.5])
    """
    p, t, mask = _joint(pred, target)
    if p.ndim != 2:
        raise ValueError(
            f"cross_sectional_ic expects a 2-D [T, S] panel, got shape {p.shape}"
        )

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        n = mask.sum(axis=1)
        safe_n = np.maximum(n, 1)
        p_mean = np.where(mask, p, 0.0).sum(axis=1) / safe_n
        t_mean = np.where(mask, t, 0.0).sum(axis=1) / safe_n
        p_dev = np.where(mask, p - p_mean[:, None], 0.0)
        t_dev = np.where(mask, t - t_mean[:, None], 0.0)
        cov = (p_dev * t_dev).sum(axis=1)
        var_p = (p_dev * p_dev).sum(axis=1)
        var_t = (t_dev * t_dev).sum(axis=1)

        p_varies = np.max(np.where(mask, p, -np.inf), axis=1, initial=-np.inf) > np.min(
            np.where(mask, p, np.inf), axis=1, initial=np.inf
        )
        t_varies = np.max(np.where(mask, t, -np.inf), axis=1, initial=-np.inf) > np.min(
            np.where(mask, t, np.inf), axis=1, initial=np.inf
        )
        valid = (n >= 2) & p_varies & t_varies & (var_p > 0) & (var_t > 0)

        series = np.full(p.shape[0], np.nan)
        series[valid] = cov[valid] / np.sqrt(var_p[valid] * var_t[valid])
        return series


def cross_sectional_ic(pred, target) -> float:
    """Return the mean over time of the per-row Pearson correlation.

    The mean runs over the rows ``cross_sectional_ic_series`` does not skip
    (see there for when a row is skipped). NaN is returned when every row is
    skipped.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.

    Examples
    --------
    >>> pred = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    >>> target = np.array([[9.0, 1.0, 4.0], [1.0, 9.0, 4.0]])
    >>> cross_sectional_ic(pred, target)
    0.989743318610787
    """
    return _finite_mean(cross_sectional_ic_series(pred, target))


def _finite_mean(series: np.ndarray) -> float:
    """Return the mean of the finite values of ``series``, NaN when there are none."""
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return float("nan")
    return float(finite.sum() / finite.size)


def information_ratio(series) -> float:
    """Return the mean of the finite values of ``series`` over their sample std.

    This is the ICIR when ``series`` is a per-bar IC series. NaN values (the
    skipped bars of an IC series) are left out rather than counted as 0. The
    standard deviation uses ``ddof=1``, so fewer than two finite values give
    NaN, and so do finite values that are all equal (tested exactly, as for
    the IC's constancy test), since the ratio is then undefined.

    Parameters
    ----------
    series : array_like
        A 1-D sequence of per-bar values.

    Examples
    --------
    >>> information_ratio([0.1, np.nan, 0.3])
    1.4142135623730951
    >>> information_ratio([0.1, np.nan])
    nan
    """
    values = np.asarray(series, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size < 2 or finite.max() == finite.min():
        return float("nan")
    return float(finite.mean() / finite.std(ddof=1))


def cross_sectional_rank_ic_series(pred, target) -> np.ndarray:
    """Return the per-row IC computed on per-row ranks, NaN on a skipped row.

    Cells outside the joint mask are set to NaN on both panels before ranking.
    The order matters: a symbol missing only on the target side would
    otherwise still occupy a rank on the prediction side and shift every other
    rank in that row. Ties receive their average rank.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.

    Returns
    -------
    np.ndarray
        A float64 array of length ``T``.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.

    Examples
    --------
    >>> cross_sectional_rank_ic_series([[1.0, 10.0, 100.0]], [[1.0, 3.0, 2.0]])
    array([0.5])
    """
    p, t, mask = _joint(pred, target)
    if p.ndim != 2:
        raise ValueError(
            f"cross_sectional_rank_ic expects a 2-D [T, S] panel, got shape {p.shape}"
        )
    p_ranks = rankdata(np.where(mask, p, np.nan), axis=1, nan_policy="omit")
    t_ranks = rankdata(np.where(mask, t, np.nan), axis=1, nan_policy="omit")
    return cross_sectional_ic_series(p_ranks, t_ranks)


def cross_sectional_rank_ic(pred, target) -> float:
    """Return the cross-sectional IC computed on per-row ranks.

    The mean of ``cross_sectional_rank_ic_series`` over the rows it does not
    skip, or NaN when every row is skipped.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or differ in shape.

    Examples
    --------
    The panel from ``cross_sectional_ic`` orders every row the same way
    on both sides, so its rank correlation is exactly 1:

    >>> cross_sectional_rank_ic(pred, target)
    1.0
    """
    return _finite_mean(cross_sectional_rank_ic_series(pred, target))


def rank_autocorrelation_series(pred, lag: int = 1) -> np.ndarray:
    """Return each row's rank correlation with the row ``lag`` rows before, NaN where there is none.

    The prediction's own persistence, as the factor report's
    ``rank_autocorrelation``: row ``t`` is the cross-sectional rank IC of
    ``pred[t]`` against ``pred[t - lag]`` on the symbols finite in both. The
    first ``lag`` rows have no earlier row and are NaN, as is a row skipped
    by the rank IC. A signal whose order changes little from bar to bar
    (values near 1) needs little trading to follow.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    lag : int, default 1
        How many rows back each row is compared with; at least 1.

    Returns
    -------
    np.ndarray
        A float64 array of length ``T``.

    Raises
    ------
    ValueError
        If ``pred`` is not 2-D or ``lag`` is below 1.

    Examples
    --------
    The second row swaps the top two of the first:

    >>> rank_autocorrelation_series([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0]])
    array([nan, 0.8])
    """
    p = np.asarray(pred, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"rank_autocorrelation_series expects a 2-D [T, S] panel, got shape {p.shape}")
    if lag < 1:
        raise ValueError(f"lag must be at least 1, got {lag}")
    out = np.full(p.shape[0], np.nan)
    if p.shape[0] > lag:
        out[lag:] = cross_sectional_rank_ic_series(p[lag:], p[:-lag])
    return out


def regression_panel_metrics(
    pred, target, *, return_series: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return the eight panel metrics keyed ``mse, rmse, mae, r2, ic, rank_ic, icir, rank_icir``.

    ``ic`` and ``icir`` are the mean and the ``information_ratio`` of one
    ``cross_sectional_ic_series``, and ``rank_ic`` and ``rank_icir`` of one
    ``cross_sectional_rank_ic_series``, so each series is computed once.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.
    return_series : bool, default False
        Also return the two per-row series the IC metrics were computed
        from, as ``{"ic": ..., "rank_ic": ...}`` (length ``T``, NaN on a
        skipped row).

    Returns
    -------
    dict or tuple
        The metrics, or ``(metrics, series)`` when ``return_series`` is True.

    Examples
    --------
    >>> metrics = regression_panel_metrics(pred, target)
    >>> sorted(metrics)
    ['ic', 'icir', 'mae', 'mse', 'r2', 'rank_ic', 'rank_icir', 'rmse']
    >>> metrics["rank_ic"]
    1.0
    >>> metrics, series = regression_panel_metrics(pred, target, return_series=True)
    >>> series["rank_ic"]
    array([1., 1.])
    """
    ic_metrics, series = ic_panel_metrics(pred, target, return_series=True)
    metrics = {
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "r2": r2(pred, target),
        **ic_metrics,
    }
    if return_series:
        return metrics, series
    return metrics


def ic_panel_metrics(
    pred, target, *, return_series: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return the four IC metrics keyed ``ic, rank_ic, icir, rank_icir``.

    The IC half of ``regression_panel_metrics``, which calls it: the mean and
    the ``information_ratio`` of one ``cross_sectional_ic_series`` and of one
    ``cross_sectional_rank_ic_series``. It suits predictions whose scale
    means nothing, such as an ensemble's average of z-scores, where an error
    metric would compare units that differ from the target's.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predictions.
    target : array_like
        A 2-D ``[T, S]`` panel of realised values.
    return_series : bool, default False
        Also return the two per-row series, as ``{"ic": ..., "rank_ic":
        ...}`` (length ``T``, NaN on a skipped row).

    Returns
    -------
    dict or tuple
        The metrics, or ``(metrics, series)`` when ``return_series`` is True.

    Examples
    --------
    On the panel from ``cross_sectional_ic``, whose two rows have the same
    IC, so the ICIR is undefined:

    >>> ic_panel_metrics(pred, target)
    {'ic': 0.989743318610787, 'rank_ic': 1.0, 'icir': nan, 'rank_icir': nan}
    """
    ic = cross_sectional_ic_series(pred, target)
    rank_ic = cross_sectional_rank_ic_series(pred, target)
    metrics = {
        "ic": _finite_mean(ic),
        "rank_ic": _finite_mean(rank_ic),
        "icir": information_ratio(ic),
        "rank_icir": information_ratio(rank_ic),
    }
    if return_series:
        return metrics, {"ic": ic, "rank_ic": rank_ic}
    return metrics


def volatility_level_metrics(pred, target) -> dict[str, float]:
    """Return the level metrics of a volatility prediction, ``qlike`` and ``variance_ratio``.

    Both compare predicted and realised variance on the cells where the
    prediction and the target are finite and positive (a non-positive
    volatility has no variance ratio). ``qlike`` is the mean over those
    cells of ``q - log(q) - 1`` with ``q = target**2 / pred**2``: zero for a
    perfect prediction, and it penalises an under-prediction of variance
    more than an over-prediction of the same size; a prediction close to
    zero where the target is not makes it very large, and infinite (written
    as null in a run's ``run.json``) once ``q`` overflows, which flags a model
    predicting next to no risk. ``variance_ratio`` is ``mean(target**2) /
    mean(pred**2)``, pooled over cells, so the most volatile symbols weigh
    most: 1 when the predicted variance is unbiased, above 1 when risk is
    under-predicted.

    Parameters
    ----------
    pred : array_like
        A 2-D ``[T, S]`` panel of predicted volatilities.
    target : array_like
        A 2-D ``[T, S]`` panel of realised volatilities, on the same scale.

    Returns
    -------
    dict[str, float]
        ``{"qlike": ..., "variance_ratio": ...}``, NaN without a usable cell.

    Examples
    --------
    A prediction of half the realised volatility on every cell, so a
    quarter of its variance:

    >>> volatility_level_metrics([[0.1, 0.2]], [[0.2, 0.4]])
    {'qlike': 1.6137056388801092, 'variance_ratio': 4.0}
    """
    p, t, mask = _joint(pred, target)
    with np.errstate(invalid="ignore"):
        mask &= (p > 0) & (t > 0)
    if not mask.any():
        return {"qlike": float("nan"), "variance_ratio": float("nan")}
    predicted, realised = p[mask] ** 2, t[mask] ** 2
    q = realised / predicted
    return {
        "qlike": float(np.mean(q - np.log(q) - 1.0)),
        "variance_ratio": float(realised.mean() / predicted.mean()),
    }


def scores_volatility_level(label, scale: str | None) -> bool:
    """Return whether a label's predictions get ``volatility_level_metrics``.

    They do when the label's ``kind`` is ``"volatility"`` and the prediction
    is on the label's own scale (``scale`` is ``"raw"``, as a predictor's
    ``label_scales`` reports it): a level metric on a standardized
    prediction compares units that differ from the label's. An object
    without ``kind`` counts as a return label.

    Parameters
    ----------
    label : object
        The label object, such as a ``Forward``.
    scale : str or None
        The prediction's scale for that label, ``"raw"`` or
        ``"standardized"``.

    Returns
    -------
    bool

    Examples
    --------
    >>> from quantlab.label.predefined.fret import Return, Volatility
    >>> scores_volatility_level(Volatility, "raw"), scores_volatility_level(Volatility, "standardized")
    (True, False)
    >>> scores_volatility_level(Return, "raw")
    False
    """
    return getattr(label, "kind", "return") == "volatility" and scale == "raw"


SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class Segments:
    """The bars of a unit's train, validation and test segments.

    Each field is an array of timestamps, empty when the unit has no such
    segment; a model's come from its purged split of the training window
    and its test dates.

    Examples
    --------
    >>> segments = Segments(train=stamps[:20], val=stamps[:0], test=stamps[25:])
    >>> [split for split, _ in segments.splits()]
    ['train', 'test']
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def splits(self) -> list[tuple[str, np.ndarray]]:
        """Return ``(split, timestamps)`` for each non-empty segment, in ``SPLITS`` order.

        Examples
        --------
        >>> Segments(train=stamps[:2], val=stamps[:0], test=stamps[2:3]).splits()[1][0]
        'test'
        """
        return [
            (split, np.asarray(stamps))
            for split, stamps in zip(SPLITS, (self.train, self.val, self.test))
            if len(stamps)
        ]


def evaluate(
    predictions: xr.Dataset,
    truth: xr.Dataset | Mapping[str, xr.Dataset],
    *,
    labels: Mapping[str, object],
    label_scales: Mapping[str, str],
    segments: Segments | Mapping[str, Segments],
    test_bounds: tuple,
    run_dir: Path | str | None,
    extra_metrics: Callable[[str, np.ndarray, np.ndarray], Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Score a unit's predictions of every label and write its evaluation files.

    Each label's prediction is taken on its segment's bars and on the
    symbols of ``truth``, so a predicted symbol without truth is ignored and
    a missing prediction counts as missing.

    Parameters
    ----------
    predictions : xr.Dataset
        One variable per label name on ``(timestamp, symbol)``.
    truth : xr.Dataset or Mapping[str, xr.Dataset]
        The labels' raw values, one variable per label name on
        ``(timestamp, symbol)``, covering every segment's bars; or each
        label's own panel holding it (an ensemble scores a label on the
        symbols of the first member predicting it).
    labels : Mapping[str, object]
        Label name to the label object predicting it (read for its
        ``kind``), in label order: the first key is the first label.
    label_scales : Mapping[str, str]
        Label name to its prediction scale, ``"raw"`` or ``"standardized"``.
    segments : Segments or Mapping[str, Segments]
        The unit's segments, or each label's own (an ensemble scores a label
        on the segments of the first member predicting it).
    test_bounds : tuple
        The unit's ``(test_start, test_end)``; ``test_predictions.zarr``
        keeps the test bars inside them.
    run_dir : Path or str or None
        Directory receiving ``ic_series.csv`` and ``test_predictions.zarr``;
        None writes nothing.
    extra_metrics : callable, optional
        ``extra_metrics(label, timestamps, symbols)`` is called for every label
        and scored split with the bars and symbols that split is scored on, and
        the metrics it returns are added under the same keys as the others
        (``{split}_{key}``, prefixed by the label after the first). An ensemble
        adds ``member_correlation`` this way.

    Returns
    -------
    dict[str, float]
        The metrics, NaN where a metric is undefined.

    Notes
    -----
    Written into ``run_dir``:

    - ``ic_series.csv``, columns ``split, timestamp, ic, rank_ic,
      rank_autocorrelation``: the first label's per-bar IC, rank IC and
      rank autocorrelation behind its ``{split}_ic`` / ``{split}_icir`` /
      ``{split}_rank_autocorrelation``, one row per bar of each scored split
      in ``SPLITS`` order; a bar where the IC and the rank IC are both
      undefined has no row.
    - ``test_predictions.zarr``: every label's prediction on the first
      label's test bars inside ``test_bounds``; not written when there are
      none.

    Examples
    --------
    >>> metrics = evaluate(
    ...     predictions, truth, labels={"ret": label}, label_scales={"ret": "raw"},
    ...     segments=segments, test_bounds=("2024-01-31", "2024-02-09"), run_dir=None,
    ... )
    >>> sorted(metrics)[:4]
    ['test_ic', 'test_icir', 'test_mae', 'test_mse']
    """
    names = [str(name) for name in labels]
    per_label = (
        segments if not isinstance(segments, Segments) else {name: segments for name in names}
    )
    truths = truth if not isinstance(truth, xr.Dataset) else {name: truth for name in names}
    metrics: dict[str, float] = {}
    series: dict[str, tuple[np.ndarray, ...]] = {}
    for i, name in enumerate(names):
        prefix = "" if i == 0 else f"{name}_"
        scale = label_scales.get(name)
        level = scores_volatility_level(labels[name], scale)
        for split, stamps in per_label[name].splits():
            target = truths[name][name].sel(timestamp=stamps)
            symbols = target.symbol.values
            pred = (
                predictions[name]
                .reindex(timestamp=stamps, symbol=symbols)
                .values
            )
            target = target.values
            values, per_bar = ic_panel_metrics(pred, target, return_series=True)
            autocorrelation = rank_autocorrelation_series(pred)
            values["rank_autocorrelation"] = _finite_mean(autocorrelation)
            if scale == "raw":
                values.update(
                    mse=mse(pred, target), rmse=rmse(pred, target),
                    mae=mae(pred, target), r2=r2(pred, target),
                )
            if level:
                values.update(volatility_level_metrics(pred, target))
            if extra_metrics is not None:
                values.update(extra_metrics(name, stamps, symbols))
            metrics.update({f"{split}_{prefix}{key}": value for key, value in values.items()})
            if i == 0:
                series[split] = (stamps, per_bar["ic"], per_bar["rank_ic"], autocorrelation)

    if run_dir is not None:
        ic_series_path, test_predictions_path = evaluation_paths(run_dir)
        _write_ic_series(ic_series_path, series)
        test = np.sort(np.asarray(per_label[names[0]].test)) if names else np.array([])
        # Bounds resolve as the segments do: a date string covers its whole day.
        (test,) = purge_segments(test, [tuple(test_bounds)], 0)
        if len(test):
            predictions[names].reindex(timestamp=test).to_zarr(
                test_predictions_path, mode="w"
            )
    return metrics


def _write_ic_series(path: Path, series: Mapping) -> None:
    """Write per-bar IC series to ``path`` as ``ic_series.csv``, atomically.

    ``series`` maps a split name to ``(timestamps, ic, rank_ic,
    rank_autocorrelation)`` arrays of one length. The rows follow the splits of ``SPLITS`` that ``series``
    holds, each in its given order; a bar where neither value is finite has
    no row.

    Examples
    --------
    >>> ic_series_path, _ = evaluation_paths(unit)
    >>> _write_ic_series(ic_series_path, {"test": (stamps, ic, rank_ic, autocorrelation)})
    >>> pd.read_csv(ic_series_path).columns.tolist()
    ['split', 'timestamp', 'ic', 'rank_ic', 'rank_autocorrelation']
    """
    rows = []
    for split in SPLITS:
        if split not in series:
            continue
        stamps, ic, rank_ic, autocorrelation = series[split]
        keep = np.isfinite(ic) | np.isfinite(rank_ic)
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "timestamp": np.asarray(stamps)[keep],
                    "ic": ic[keep],
                    "rank_ic": rank_ic[keep],
                    "rank_autocorrelation": autocorrelation[keep],
                }
            )
        )
    frame = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["split", "timestamp", "ic", "rank_ic", "rank_autocorrelation"])
    )
    path = Path(path)
    staging = path.with_name(path.name + ".tmp")
    frame.to_csv(staging, index=False)
    os.replace(staging, path)
