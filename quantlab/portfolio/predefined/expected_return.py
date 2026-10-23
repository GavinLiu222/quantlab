"""The shipped expected return models of the mean-variance rule.

An ``ExpectedReturnModel`` (``quantlab.portfolio.base``) turns the prediction
of a mean-variance rule's ``expected_return_label`` into the expected return
``mu`` its optimiser maximises, on ``[bars, symbols]`` arrays, so one
implementation serves the per-bar rule (one row) and the approximate
backtest. Five are shipped:

- ``GrinoldExpectedReturn``: ``mu = ic * sigma * z``, ``z`` the prediction's
  cross-sectional z-score over the bar's candidates and ``sigma`` each
  candidate's volatility on the label's span (Grinold's rule; a riskier name
  gets a larger ``mu``);
- ``ZScoreExpectedReturn``: ``mu = ic * median(sigma) * z``, the same scale
  without the tilt to risky names: within a bar ``mu`` ranks as the
  prediction;
- ``UnscaledZScoreExpectedReturn``: ``mu = z``, the z-score itself,
  unscaled: the same units on every bar, so ``turnover_penalty`` is in z
  units;
- ``RawExpectedReturn``: ``mu`` is the prediction, which must be in return
  units (its label spec's scale ``"raw"``);
- ``PredictionExpectedReturn``: ``mu`` is the prediction as it is, in its own
  units (a standardized ensemble score, say); ``risk_aversion`` and
  ``turnover_penalty`` are then in those units too.
"""

import numpy as np

from quantlab.portfolio.base import ExpectedReturnModel
from quantlab.portfolio.config import (
    GrinoldExpectedReturnConfig,
    PredictionExpectedReturnConfig,
    RawExpectedReturnConfig,
    UnscaledZScoreExpectedReturnConfig,
    ZScoreExpectedReturnConfig,
)
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.utils.cross_section import cross_sectional_zscore


def _zscore(predictions: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Each row's cross-sectional z-score (``ddof=1``) over its candidates.

    0.0 off the candidates and where the z-score is undefined.
    """
    return np.nan_to_num(cross_sectional_zscore(np.where(candidates, predictions, np.nan)), nan=0.0)


def _check_ic(ic: float, name: str) -> None:
    """Refuse an ``ic`` that is not a finite positive number, naming the model ``name``."""
    if not np.isfinite(ic) or ic <= 0:
        raise ValueError(f"{name} needs a positive ic, got {ic!r}")


class _ScaledByIC(ExpectedReturnModel):
    """A model whose ``mu`` is ``ic`` times a scaled z-score: Grinold's and the z-score model.

    It refuses a non-positive ``ic`` when it is built.
    """

    def __init__(self, config):
        """Initialize the model from its config and check its ``ic``."""
        super().__init__(config)
        _check_ic(config.ic, type(self).__name__)


class GrinoldExpectedReturn(_ScaledByIC):
    """``mu = ic * sigma * z``: Grinold's rule, a riskier candidate getting a larger ``mu``.

    ``z`` is the prediction's cross-sectional z-score over the bar's
    candidates (``ddof=1``; 0.0 when the bar has fewer than two or a constant
    cross-section), ``sigma`` each candidate's volatility on the label's
    span, so any score will do.

    Parameters
    ----------
    config : GrinoldExpectedReturnConfig
        ``ic``, the model's information coefficient (a CV run's mean IC).

    Raises
    ------
    ValueError
        If ``ic`` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> model = GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05))
    >>> model.expected_returns(
    ...     np.array([[1.0, 2.0, 3.0]]), np.array([[0.02, 0.02, 0.04]]), np.ones((1, 3), dtype=bool)
    ... ).round(4)
    array([[-0.001,  0.   ,  0.002]])
    """

    config_cls = GrinoldExpectedReturnConfig

    def expected_returns(self, predictions, sigma, candidates) -> np.ndarray:
        """``ic * sigma * z`` on the candidates, 0.0 elsewhere.

        See ``ExpectedReturnModel.expected_returns``.

        Examples
        --------
        >>> model.expected_returns(
        ...     np.array([[1.0, 3.0]]), np.array([[0.01, 0.01]]), np.ones((1, 2), dtype=bool)
        ... ).round(4)
        array([[-0.0004,  0.0004]])
        """
        candidates = np.asarray(candidates, dtype=bool)
        return np.where(candidates, self.config.ic * sigma * _zscore(predictions, candidates), 0.0)


class ZScoreExpectedReturn(_ScaledByIC):
    """``mu = ic * median(sigma) * z``: Grinold's scale without the tilt to risky names.

    ``z`` is as in ``GrinoldExpectedReturn``; each bar's ``z`` is scaled by
    the median volatility of its candidates, so ``mu`` has Grinold's size
    (the same ``risk_aversion`` and ``turnover_penalty`` mean the same) and,
    within a bar, ranks exactly as the prediction.

    Parameters
    ----------
    config : ZScoreExpectedReturnConfig
        ``ic``, the model's information coefficient.

    Raises
    ------
    ValueError
        If ``ic`` is not positive.

    Examples
    --------
    >>> model = ZScoreExpectedReturn(ZScoreExpectedReturnConfig(ic=0.05))
    >>> model.expected_returns(
    ...     np.array([[1.0, 2.0, 3.0]]), np.array([[0.02, 0.02, 0.04]]), np.ones((1, 3), dtype=bool)
    ... ).round(4)
    array([[-0.001,  0.   ,  0.001]])
    """

    config_cls = ZScoreExpectedReturnConfig

    def expected_returns(self, predictions, sigma, candidates) -> np.ndarray:
        """``ic * median candidate sigma * z`` on the candidates, 0.0 elsewhere.

        Examples
        --------
        >>> model.expected_returns(
        ...     np.array([[1.0, 3.0]]), np.array([[0.01, 0.03]]), np.ones((1, 2), dtype=bool)
        ... ).round(4)
        array([[-0.0007,  0.0007]])
        """
        candidates = np.asarray(candidates, dtype=bool)
        with np.errstate(all="ignore"):
            median = np.nanmedian(np.where(candidates, sigma, np.nan), axis=-1, keepdims=True)
        scaled = self.config.ic * np.nan_to_num(median, nan=0.0) * _zscore(predictions, candidates)
        return np.where(candidates, scaled, 0.0)


class UnscaledZScoreExpectedReturn(ExpectedReturnModel):
    """``mu = z``: the prediction's cross-sectional z-score over the candidates, unscaled.

    ``z`` is the z-score (``ddof=1``) of each bar's prediction over its
    candidates, 0.0 off them. ``ZScoreExpectedReturn`` scales it by
    ``ic * median(sigma)``, which changes from bar to bar; here ``mu`` is the
    z-score itself, so ``turnover_penalty`` (and ``risk_aversion``) weigh
    against it in z units on every bar. No volatility is read.

    Examples
    --------
    >>> UnscaledZScoreExpectedReturn(UnscaledZScoreExpectedReturnConfig()).expected_returns(
    ...     np.array([[1.0, 2.0, 3.0, 9.0]]), None, np.array([[True, True, True, False]])
    ... )
    array([[-1.,  0.,  1.,  0.]])
    """

    config_cls = UnscaledZScoreExpectedReturnConfig
    needs_sigma = False

    def expected_returns(self, predictions, sigma, candidates) -> np.ndarray:
        """The candidates' z-score, 0.0 elsewhere; see the class docstring."""
        candidates = np.asarray(candidates, dtype=bool)
        return np.where(candidates, _zscore(predictions, candidates), 0.0)


class PredictionExpectedReturn(ExpectedReturnModel):
    """``mu`` is the prediction as it is, whatever its units.

    For a prediction that is not a return (a standardized ensemble score, a
    rank): ``risk_aversion``, ``turnover_penalty`` and the rest weigh against
    ``mu`` in the prediction's units, so they are not comparable with a
    Grinold rule's. Any label scale is taken.

    Examples
    --------
    >>> PredictionExpectedReturn(PredictionExpectedReturnConfig()).expected_returns(
    ...     np.array([[1.2, -0.4]]), None, np.ones((1, 2), dtype=bool)
    ... )
    array([[ 1.2, -0.4]])
    """

    config_cls = PredictionExpectedReturnConfig
    needs_sigma = False

    def expected_returns(self, predictions, sigma, candidates) -> np.ndarray:
        """The prediction on the candidates, 0.0 elsewhere; see the class docstring."""
        return np.where(np.asarray(candidates, dtype=bool), predictions, 0.0)


class RawExpectedReturn(PredictionExpectedReturn):
    """``mu`` is the prediction, which must be a return in the label's own units.

    Bound to a label whose spec reports a scale other than ``"raw"`` (a model
    fitted on a transformed target, or a label an ensemble averages), it
    refuses; ``PredictionExpectedReturn`` takes such a prediction as it is.

    Examples
    --------
    >>> RawExpectedReturn(RawExpectedReturnConfig()).bind(
    ...     LabelSpec(name="ret_5", scale="standardized", delay=1, span=5)
    ... )
    Traceback (most recent call last):
    ValueError: RawExpectedReturn reads the prediction of 'ret_5' as a return, but its label spec reports its scale as 'standardized', not 'raw' (a model fitted on a transformed target, or a label an ensemble averages); use PredictionExpectedReturn to take it as it is, or GrinoldExpectedReturn
    """

    config_cls = RawExpectedReturnConfig

    def bind(self, spec: LabelSpec) -> None:
        """Refuse a label whose spec's scale is not ``"raw"``; see the class docstring."""
        if spec.scale != "raw":
            raise ValueError(
                f"{type(self).__name__} reads the prediction of {spec.name!r} as a return, "
                f"but its label spec reports its scale as {spec.scale!r}, not 'raw' (a model "
                f"fitted on a transformed target, or a label an ensemble averages); use "
                f"PredictionExpectedReturn to take it as it is, or GrinoldExpectedReturn"
            )
