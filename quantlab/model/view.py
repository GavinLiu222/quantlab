"""Predictor views: predictors that answer through the predictor they wrap.

A **predictor view** wraps another predictor (a model, an ensemble or
another view) and is a predictor itself: it drops into
``BacktestConfig.model`` in place of the predictor it wraps and changes only
what it says it changes about the predictions, such as which cells it masks
(``MembershipMaskedPredictor``), where they come from (``RecordedPredictor``)
or how they are smoothed in time (``SmoothedPredictor``). ``PredictorView``
forwards every member of the backtester's ``Predictor`` protocol
(``quantlab.backtest.base``) to the wrapped predictor once, so views of
views compose.

The protocol lives in the backtest layer, which the model layer does not
import (ADR 0022): the view forwards it structurally, and
``tests/test_predictor_view.py`` checks it forwards every protocol member.
"""

import dataclasses
from pathlib import Path
from typing import Self

import xarray as xr

from quantlab.core.component import Component
from quantlab.model.config import PredictorViewConfig


class PredictorView(Component):
    """Root class of the views of ``config.predictor``.

    Every protocol member is the wrapped predictor's, through the view:

    =========================  ==============================================
    ``labels``                 the wrapped predictor's
    ``train_bounds``           the wrapped predictor's
    ``test_bounds``            the wrapped predictor's
    ``fitted_train_bounds``    the wrapped predictor's
    ``label_delays``           the wrapped predictor's
    ``label_scales``           the wrapped predictor's
    ``predict_window``         the wrapped predictor's, then
                               ``_view_predictions``
    ``collect``, ``load``      called on the wrapped predictor; return the view
    ``train``                  the wrapped predictor's checkpoint
    ``check_checkpoint``       the wrapped predictor's
    ``config``                 ``config_cls`` filled from the view's
                               attributes of the same names
    ``from_config``            the config's fields, components rebuilt,
                               passed to the constructor
    =========================  ==============================================

    A subclass declares ``config_cls`` (a subclass of
    ``PredictorViewConfig``), keeps each config field as an attribute of the
    same name, takes the fields as constructor arguments, and overrides
    ``_view_predictions`` to change the predictions; a view that predicts
    differently (a wider request, a record) overrides ``predict_window``,
    and one that changes a label's scale or the checkpoint handling
    overrides that member.

    Parameters
    ----------
    predictor : Predictor
        The wrapped predictor.

    Attributes
    ----------
    predictor : Predictor
        The wrapped predictor.

    Examples
    --------
    With ``model`` a loaded predictor:

    >>> class Doubled(PredictorView):
    ...     def _view_predictions(self, predictions):
    ...         return predictions * 2
    >>> view = Doubled(model)
    >>> view.train_bounds == model.train_bounds, view.label_scales == model.label_scales
    (True, True)
    >>> window = ("2024-01-01", "2024-01-12")
    >>> bool(view.predict_window(*window).equals(model.predict_window(*window) * 2))
    True
    >>> view.load("ckpt.joblib") is view
    True
    """

    #: The config class the view is serialised and rebuilt with.
    config_cls = PredictorViewConfig

    def __init__(self, predictor) -> None:
        """Initialize the view; see the class docstring for parameters."""
        self.predictor = predictor

    def _view_predictions(self, predictions: xr.Dataset) -> xr.Dataset:
        """Return ``predictions``, the wrapped predictor's, as the view gives them."""
        return predictions

    @property
    def labels(self) -> list:
        """The wrapped predictor's labels.

        Examples
        --------
        >>> view.labels is model.labels
        True
        """
        return self.predictor.labels

    @property
    def train_bounds(self) -> tuple:
        """The wrapped predictor's ``(start, end)`` training window.

        Examples
        --------
        >>> view.train_bounds == model.train_bounds
        True
        """
        return self.predictor.train_bounds

    @property
    def test_bounds(self) -> tuple:
        """The wrapped predictor's ``(start, end)`` test window.

        Examples
        --------
        >>> view.test_bounds == model.test_bounds
        True
        """
        return self.predictor.test_bounds

    @property
    def fitted_train_bounds(self) -> tuple:
        """The wrapped predictor's fitted ``(start, end)`` training window.

        Examples
        --------
        >>> view.fitted_train_bounds == model.fitted_train_bounds
        True
        """
        return self.predictor.fitted_train_bounds

    @property
    def label_delays(self) -> tuple[int, ...]:
        """The wrapped predictor's label delays; a view moves no bar.

        Examples
        --------
        >>> view.label_delays == model.label_delays
        True
        """
        return self.predictor.label_delays

    @property
    def label_scales(self) -> dict[str, str]:
        """The wrapped predictor's label scales.

        Examples
        --------
        >>> view.label_scales == model.label_scales
        True
        """
        return self.predictor.label_scales

    def predict_window(self, start, end) -> xr.Dataset:
        """The wrapped predictor's predictions over the window, through ``_view_predictions``.

        Examples
        --------
        >>> bool(view.predict_window(*window).equals(model.predict_window(*window) * 2))
        True
        """
        return self._view_predictions(self.predictor.predict_window(start, end))

    def collect(self) -> Self:
        """Collect the wrapped predictor's training data; returns the view.

        Examples
        --------
        >>> view.collect() is view
        True
        """
        self.predictor.collect()
        return self

    def train(self) -> Path:
        """Train the wrapped predictor; returns its checkpoint.

        Examples
        --------
        >>> view.train() == model.train()
        True
        """
        return self.predictor.train()

    def load(self, path) -> Self:
        """Load ``path`` into the wrapped predictor; returns the view.

        Examples
        --------
        >>> view.load("ckpt.joblib") is view
        True
        """
        self.predictor.load(path)
        return self

    def check_checkpoint(self, path) -> None:
        """Validate ``path`` with the wrapped predictor, without loading it.

        Examples
        --------
        >>> view.check_checkpoint("ckpt.joblib")  # raises if it does not fit
        """
        self.predictor.check_checkpoint(path)

    @property
    def config(self) -> PredictorViewConfig:
        """``config_cls`` filled from the view's attributes; ``get_config()`` serialises it.

        Examples
        --------
        >>> view.config.predictor is model
        True
        """
        return self.config_cls(
            **{
                spec.name: getattr(self, spec.name)
                for spec in dataclasses.fields(self.config_cls)
            }
        )

    @classmethod
    def from_config(cls, config: dict, run_dir=None) -> Self:
        """Rebuild the view and the predictor it wraps from ``get_config()``.

        The component fields are rebuilt by the component rule, with
        ``run_dir`` passed down, and every field is passed to the
        constructor by name.

        Examples
        --------
        >>> from quantlab.model.predefined.smoothed import SmoothedPredictor
        >>> smoothed = SmoothedPredictor(None, halflife_bars=2.5)
        >>> rebuilt = SmoothedPredictor.from_config(smoothed.get_config())
        >>> rebuilt.halflife_bars, rebuilt.warmup_bars, rebuilt.predictor
        (2.5, 25, None)
        """
        return cls(**cls._rebuilt_fields(config, run_dir))
