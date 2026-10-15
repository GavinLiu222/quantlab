"""Smooth a predictor's predictions in time, causally.

A portfolio rule that trades every bar on the latest predictions trades
their fast part as well as their slow part. ``SmoothedPredictor`` wraps any
``Predictor`` and replaces each bar's prediction with an exponentially
weighted mean of the symbol's own predictions up to and including that bar,
so a memoryless rule on it trades the slow part only. Bar t reads bars up to
t and nothing later.

Each bar is first standardised over its symbols (a cross-sectional z-score),
so a change of scale between bars, between the checkpoints of a walk-forward
run for example, does not weigh one stretch of bars more than another; the
smoothed panel is then on a standardised scale. ``standardize=False``
smooths the raw values in their own units instead.

The mean at a window's first bars needs the predictions of the bars before
it, the warm-up, which the wrapped predictor is asked for too. A model
predicts them with its loaded checkpoint; a ``RecordedPredictor`` with
``history=True`` serves what the checkpoints live at those bars predicted
there (in a ``run_cv``, the earlier folds), the history a live account
smoothing its own past predictions would hold.
"""

import dataclasses
import math
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.core.component import Component, component
from quantlab.utils.cross_section import cross_sectional_zscore

if TYPE_CHECKING:  # type hints only
    from pathlib import Path

#: Calendar days first requested per warm-up bar (about 1.45 on an exchange calendar), plus a margin.
_DAYS_PER_BAR = 1.5
_MARGIN_DAYS = 10
#: The request is widened (doubled) at most this many times when it returns too few warm-up bars.
_MAX_WIDENINGS = 8


@dataclasses.dataclass(frozen=True)
class SmoothedPredictorConfig:
    """What a ``SmoothedPredictor`` is rebuilt from.

    Examples
    --------
    >>> sorted(SmoothedPredictorConfig(predictor=None).__dataclass_fields__)
    ['halflife_bars', 'predictor', 'standardize', 'warmup_bars']
    """

    #: The predictor whose predictions are smoothed.
    predictor: object = component()
    #: The halflife of the exponential weights, in bars.
    halflife_bars: float = 1.0
    #: Bars before the window the mean also reads; ``None`` means ten halflives.
    warmup_bars: int | None = None
    #: Whether each bar is z-scored over its symbols before smoothing.
    standardize: bool = True


class SmoothedPredictor(Component):
    """A ``Predictor`` whose predictions are a causal exponential mean of the wrapped ones.

    ``predict_window(start, end)`` asks the wrapped predictor for the window
    and up to ``warmup_bars`` bars before it, z-scores every bar over its
    symbols (``cross_sectional_zscore``, ``ddof=1``) when ``standardize``,
    and returns, for every bar of the window, each symbol's exponentially
    weighted mean of those values over the bars up to that one: weight
    ``2 ** (-age / halflife_bars)`` for a value ``age`` bars old, normalised
    over the values the symbol has (pandas ``ewm(halflife=halflife_bars,
    adjust=True, ignore_na=False)``). A bar where a symbol has no value adds
    nothing to its mean, but its older values keep ageing. A cell NaN in the
    wrapped predictions at t is NaN in the output at t, so a mask at t (a
    ``MembershipMaskedPredictor``) still decides who may be entered.

    The warm-up is counted in the wrapped predictor's bars: the request is
    widened until ``warmup_bars`` bars before the window come back (ten
    halflives by default, the weight left beyond them at most ``2 ** -10``)
    and cut to them. When the wrapped predictor holds fewer (its first bar,
    a record's start), the mean starts at its first bar, a warning gives
    the count, and ``last_warmup_bars`` holds it.

    The warm-up must be out of sample. A wrapped model predicts the bars
    before the window with its loaded checkpoint, so a warm-up bar on or
    before the end of the checkpoint's fitted training window
    (``fitted_train_bounds``) would blend an in-sample fit into the
    smoothed values, and is refused. A ``RecordedPredictor`` with
    ``history=True`` serves each earlier bar as the checkpoint live then
    predicted it, so it is not checked. Every other
    protocol member forwards to the wrapped predictor; ``label_scales`` is
    ``"standardized"`` for every label when standardising.

    Parameters
    ----------
    predictor : Predictor
        The model or ensemble whose predictions are smoothed. In a
        ``run_cv`` replay, wrap a ``RecordedPredictor(..., history=True)``
        so each fold's warm-up is what the earlier folds predicted.
    halflife_bars : float
        The halflife of the weights, in bars; positive.
    warmup_bars : int, optional
        Bars before the window the mean reads; non-negative. Ten
        halflives, rounded up, by default.
    standardize : bool, default True
        Whether to z-score each bar over its symbols before smoothing.

    Raises
    ------
    ValueError
        If ``halflife_bars`` is not positive and finite or ``warmup_bars``
        is negative; from ``predict_window``, if a warm-up bar lies inside
        the loaded checkpoint's fitted training window.

    Attributes
    ----------
    last_warmup_bars : int or None
        The warm-up bars the last ``predict_window`` read.

    Examples
    --------
    With ``whole`` a ``run_cv`` result, ``model`` the predictor it ran and
    ``backtester`` a ``run_cv`` backtester over the same folds::

        >>> recorded = RecordedPredictor(model, whole.run_dir, history=True)
        >>> backtester.config = dataclasses.replace(
        ...     backtester.config, model=SmoothedPredictor(recorded, halflife_bars=20)
        ... )
        >>> smoothed = backtester.run_cv()  # every fold's warm-up from the earlier folds

    >>> SmoothedPredictor(None, halflife_bars=2.5).warmup_bars
    25
    """

    #: The config dataclass the wrapper is serialised and rebuilt with.
    config_cls = SmoothedPredictorConfig

    def __init__(
        self,
        predictor,
        halflife_bars: float,
        warmup_bars: int | None = None,
        standardize: bool = True,
    ) -> None:
        """Initialize the wrapper; see the class docstring for parameters."""
        if not (isinstance(halflife_bars, (int, float)) and math.isfinite(halflife_bars)
                and halflife_bars > 0):
            raise ValueError(
                f"{type(self).__name__}: halflife_bars must be positive and finite, "
                f"got {halflife_bars!r}"
            )
        if warmup_bars is None:
            warmup_bars = math.ceil(10 * halflife_bars)
        if int(warmup_bars) != warmup_bars or warmup_bars < 0:
            raise ValueError(
                f"{type(self).__name__}: warmup_bars must be a non-negative integer, "
                f"got {warmup_bars!r}"
            )
        self.predictor = predictor
        self.halflife_bars = halflife_bars
        self.warmup_bars = int(warmup_bars)
        self.standardize = bool(standardize)
        self.last_warmup_bars: int | None = None

    # -- the smoothing ---------------------------------------------------

    def smooth(self, predictions: xr.Dataset) -> xr.Dataset:
        """Return the causal exponential mean of every variable of ``predictions``.

        Every bar of ``predictions`` is read; nothing before it. See the
        class docstring for the weights.

        Parameters
        ----------
        predictions : xr.Dataset
            Predictions on ``(timestamp, symbol)``, in time order.

        Returns
        -------
        xr.Dataset
            The smoothed panel on the same coordinates, NaN where the input
            is NaN (or, standardising, where the bar is degenerate).

        Examples
        --------
        >>> panel = xr.Dataset(
        ...     {"fwd_ret_1": (("timestamp", "symbol"), [[1.0, 2.0, 3.0], [3.0, 2.0, np.nan]])},
        ...     coords={"timestamp": pd.bdate_range("2024-01-01", periods=2),
        ...             "symbol": ["A", "B", "C"]},
        ... )
        >>> SmoothedPredictor(None, halflife_bars=1).smooth(panel)["fwd_ret_1"].values.round(4)
        array([[-1.    ,  0.    ,  1.    ],
               [ 0.1381, -0.4714,     nan]])
        """
        out = predictions.copy()
        for name, variable in predictions.data_vars.items():
            panel = variable.transpose("timestamp", "symbol")
            values = np.asarray(panel.values, dtype=float)
            if self.standardize:
                values = cross_sectional_zscore(values)
            mean = pd.DataFrame(values).ewm(
                halflife=self.halflife_bars, adjust=True, ignore_na=False
            ).mean().to_numpy()
            out[name] = panel.copy(data=np.where(np.isfinite(values), mean, np.nan))
        return out

    # -- Predictor protocol ----------------------------------------------

    def predict_window(self, start, end) -> xr.Dataset:
        """The smoothed predictions for every bar from ``start`` to ``end``.

        The wrapped predictor is asked for the window and days before it,
        widened until ``warmup_bars`` bars before ``start`` come back or it
        holds no earlier bar; the last ``warmup_bars`` are kept.

        Raises
        ------
        ValueError
            If a warm-up bar lies inside the loaded checkpoint's fitted
            training window (an in-sample fit), unless the wrapped predictor
            serves recorded history (``RecordedPredictor(history=True)``).

        Examples
        --------
        With ``model`` a loaded model over business days:

        >>> out = SmoothedPredictor(model, halflife_bars=5).predict_window("2024-02-12", "2024-03-11")
        >>> str(out.timestamp.values[0])[:10]  # the warm-up bars are not returned
        '2024-02-12'
        """
        start_day = pd.Timestamp(start).normalize()
        pad_days = math.ceil(self.warmup_bars * _DAYS_PER_BAR) + _MARGIN_DAYS
        widenings = 0
        while True:
            first_day = start_day - pd.Timedelta(days=pad_days) if self.warmup_bars else start_day
            raw = self.predictor.predict_window(str(first_day.date()), end)
            days = pd.DatetimeIndex(raw["timestamp"].values).normalize()
            before = int((days < start_day).sum())
            # Too few bars came back although the request reached its first day:
            # the padding, not the history, fell short; widen it.
            reached = days.size > 0 and days[0] - first_day <= pd.Timedelta(days=_MARGIN_DAYS)
            if before >= self.warmup_bars or not reached or widenings == _MAX_WIDENINGS:
                break
            pad_days *= 2
            widenings += 1
        raw = raw.isel(timestamp=slice(max(before - self.warmup_bars, 0), None))
        self.last_warmup_bars = min(before, self.warmup_bars)
        if self.last_warmup_bars < self.warmup_bars:
            logger.warning(
                f"{type(self).__name__}: {self.last_warmup_bars} of {self.warmup_bars} warm-up "
                f"bars before {start_day.date()}; the wrapped predictor holds no earlier bar, "
                f"so the mean starts there"
            )
        if self.last_warmup_bars:
            self._check_out_of_sample(pd.Timestamp(raw["timestamp"].values[0]))
        smoothed = self.smooth(raw)
        days = pd.DatetimeIndex(smoothed["timestamp"].values).normalize()
        return smoothed.isel(timestamp=np.flatnonzero(days >= start_day))

    def _check_out_of_sample(self, first_bar: pd.Timestamp) -> None:
        """Refuse a warm-up reaching into the loaded checkpoint's fitted training window."""
        if getattr(self.predictor, "history", False) is True:
            return  # a recorded history: each bar as the checkpoint live then predicted it
        try:
            fitted_end = self.predictor.fitted_train_bounds[1]
        except (AttributeError, IndexError, TypeError, ValueError):
            return  # nothing fitted is known
        if fitted_end is not None and first_bar.normalize() <= pd.Timestamp(fitted_end).normalize():
            raise ValueError(
                f"{type(self).__name__}: the warm-up starts {first_bar.date()}, inside the "
                f"loaded checkpoint's fitted training window (to {fitted_end}); its "
                f"predictions there are in-sample fits. Shorten warmup_bars, start the "
                f"window later, or wrap a RecordedPredictor(..., history=True)"
            )

    @property
    def labels(self) -> list:
        """The wrapped predictor's labels."""
        return self.predictor.labels

    @property
    def train_bounds(self) -> tuple:
        """The wrapped predictor's ``(start, end)`` training window."""
        return self.predictor.train_bounds

    @property
    def test_bounds(self) -> tuple:
        """The wrapped predictor's ``(start, end)`` test window."""
        return self.predictor.test_bounds

    @property
    def fitted_train_bounds(self) -> tuple:
        """The wrapped predictor's fitted ``(start, end)`` training window."""
        return self.predictor.fitted_train_bounds

    @property
    def label_delays(self) -> tuple[int, ...]:
        """The wrapped predictor's label delays; smoothing moves no bar."""
        return self.predictor.label_delays

    @property
    def label_scales(self) -> dict[str, str]:
        """``"standardized"`` for every label when standardising, else the wrapped predictor's.

        Examples
        --------
        >>> SmoothedPredictor(model, halflife_bars=5).label_scales
        {'fwd_ret_1': 'standardized'}
        """
        scales = dict(self.predictor.label_scales)
        if self.standardize:
            return {name: "standardized" for name in scales}
        return scales

    def collect(self) -> Self:
        """Collect the wrapped predictor's training data; returns the wrapper."""
        self.predictor.collect()
        return self

    def train(self) -> "Path":
        """Train the wrapped predictor; returns its checkpoint."""
        return self.predictor.train()

    def load(self, path) -> Self:
        """Load ``path`` into the wrapped predictor; returns the wrapper."""
        self.predictor.load(path)
        return self

    def check_checkpoint(self, path) -> None:
        """Validate ``path`` with the wrapped predictor, without loading it."""
        self.predictor.check_checkpoint(path)

    @property
    def config(self) -> SmoothedPredictorConfig:
        """The wrapped predictor and the smoothing; ``get_config()`` serialises it."""
        return SmoothedPredictorConfig(
            predictor=self.predictor,
            halflife_bars=self.halflife_bars,
            warmup_bars=self.warmup_bars,
            standardize=self.standardize,
        )

    @classmethod
    def from_config(cls, config: dict, run_dir=None) -> Self:
        """Rebuild the wrapper and its predictor from ``get_config()``."""
        return cls(**cls._rebuilt_fields(config, run_dir))
