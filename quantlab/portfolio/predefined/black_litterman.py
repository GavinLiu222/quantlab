"""Black-Litterman on a factor risk model: ``BlackLittermanOptimizer``.

A mean-variance rule whose expected return and covariance are a
Black-Litterman posterior: an equilibrium prior from the declared market
portfolio, blended with the views of the rule's expected return model by
Bayes' rule, in the risk model's factor space. Returns are ``r = B f + e``,
the factor returns and the specific returns with prior means ``pi_f`` and
``pi_e`` and prior covariances ``tau F`` and ``tau diag(D)``, so the
expected return has the prior ``N(pi, tau Sigma)``, ``Sigma = B F B' +
diag(D)`` the forecast over the expected-return label's span. A symbol view
says ``mu_j = q_j``, ``q_j`` the expected return model's ``mu_j`` (a total
return) or the equilibrium return plus it (an alpha,
``views_relative_to_prior``), with variance ``omega_j``: ``tau Sigma_jj``
(He-Litterman), ``tau (1 - C)/C Sigma_jj`` (Walters' closed form of
Idzorek's confidence ``C``) or ``tau D_j`` (the specific variance).
The posterior mean and the posterior covariance ``Sigma + M``, ``M`` the
posterior mean's uncertainty, are solved on the factor dimension: one
``K x K`` solve, never a symbol-by-symbol inverse, and the posterior
covariance is diagonal plus rank ``2K``, the factor form the optimiser's
risk term reads.

The equilibrium ``pi = delta (Sigma + M) w`` is the expected return that
makes the market ``w`` optimal under the covariance the rule optimises on,
``delta`` the rule's ``risk_aversion``: with no views (``M = tau Sigma``)
the decided book is the market, and views equal to ``pi`` leave it there.
``pi`` splits as ``pi_f = delta F x`` and ``pi_e = pi - B pi_f``, ``x =
B' w`` the market's factor exposures (``market_exposures``).

The user's views (``ReturnView``: ``FactorReturnView``,
``RelativeFactorReturnView``, ``MarketReturnView``) are views on the factor
returns alone: a factor view says ``f_k = q``, a relative one ``f_k - f_l =
q``, a market view ``x' f = q``
(the market's specific return neglected), ``q`` an annualised return times
``span / 252``, with variance ``tau (1 - C)/C p' F p`` (Idzorek's
confidence ``C``, ``p`` the view's row) or the squared standard deviation,
converted the same way. They update the factor prior first, ``N(pi_f, tau
F)`` to ``N(f_1, F_1)``, in the form that takes a zero variance; the
symbol views then update that, which is the one Bayes update over all the
views, since the views' errors are independent. A view is constant or a
series read point-in-time from a store, the last finite value dated at or
before the bar; a view not known yet at a bar, or on a factor the bar's
forecast lacks, is left out at that bar.

The approximate backtest runs this rule (``quantlab.portfolio.predefined.approximate.black_litterman``),
handing ``moments`` each bar's arrays.
"""

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.portfolio.base import (
    InputDeclaration,
    PortfolioConstructionError,
    PortfolioConstructor,
    PortfolioContext,
    _Configured,
)
from quantlab.portfolio.config import (
    BlackLittermanConfig,
    FactorReturnViewConfig,
    MarketReturnViewConfig,
    RelativeFactorReturnViewConfig,
    ReturnViewConfig,
)
from quantlab.portfolio.predefined.expected_return import (
    GrinoldExpectedReturn,
    UnscaledZScoreExpectedReturn,
    ZScoreExpectedReturn,
)
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.risk.base import FactorRiskForecast
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.runs.record import record_read

#: Bars in a year: an annualised view times ``span / BARS_PER_YEAR`` is a view over the span.
BARS_PER_YEAR = 252


class ReturnView(_Configured, ABC):
    """Base class of the user's views of a ``BlackLittermanOptimizer``: one portfolio's return.

    A view says ``p' f = q``, ``f`` the risk model's factor returns: a
    subclass says what ``p`` is (``row``) and which risk-model factors it
    names (``risk_factors``). The value and its uncertainty are shared: an
    annualised return, a constant (``annual_return``) or a series read
    point-in-time from a zarr store (``store``, ``variable``), whose value
    at a bar is the last finite one dated at or before the bar, each read
    logged to the open ``DataRecorder``; the uncertainty is Idzorek's
    ``confidence`` or a standard deviation ``std``.

    Parameters
    ----------
    config : ReturnViewConfig
        An instance of the subclass's ``config_cls``.

    Raises
    ------
    ValueError
        If not exactly one of ``annual_return`` and ``store`` is set,
        ``annual_return`` is not finite, ``variable`` is missing with
        ``store`` or set without it, not exactly one of ``confidence`` and
        ``std`` is set, ``confidence`` is outside ``(0, 1]`` or ``std`` is
        not positive.

    Examples
    --------
    >>> from quantlab.portfolio.config import MarketReturnViewConfig
    >>> view = MarketReturnView(MarketReturnViewConfig(annual_return=0.06, confidence=0.5))
    >>> view.annual_return_at(pd.Timestamp("2024-03-01"))
    0.06
    >>> view.variance(prior=4e-4, tau=0.05, span=5)
    2e-05
    """

    def __init__(self, config: ReturnViewConfig):
        """Check the view's value and uncertainty; see the class docstring."""
        super().__init__(config)
        owner = type(self).__name__
        if (config.annual_return is None) == (config.store is None):
            raise ValueError(
                f"{owner} takes exactly one of annual_return (a constant) and store (a series), "
                f"got annual_return={config.annual_return!r}, store={config.store!r}"
            )
        if config.annual_return is not None and not math.isfinite(config.annual_return):
            raise ValueError(f"{owner}'s annual_return must be finite, got {config.annual_return}")
        if (config.store is None) != (config.variable is None):
            raise ValueError(
                f"{owner}'s variable names the store's series: set it with store and only "
                f"then, got store={config.store!r}, variable={config.variable!r}"
            )
        if (config.confidence is None) == (config.std is None):
            raise ValueError(
                f"{owner} takes exactly one of confidence and std, got "
                f"confidence={config.confidence!r}, std={config.std!r}"
            )
        if config.confidence is not None and not 0 < config.confidence <= 1:
            raise ValueError(f"{owner}'s confidence must be in (0, 1], got {config.confidence}")
        if config.std is not None and not config.std > 0:
            raise ValueError(f"{owner}'s std must be positive, got {config.std}")
        self._series: xr.Dataset | None = None

    @property
    def store_path(self) -> str | None:
        """The store the series is read from, ``None`` for a constant view.

        Examples
        --------
        >>> view.store_path is None
        True
        """
        return self.config.store

    def _read(self, start, end) -> xr.Dataset:
        """Read the series from the store over ``start`` to ``end``."""
        rows = xr.open_zarr(self.config.store)[[self.config.variable]]
        return rows.sel(timestamp=slice(start, end)).load()

    def annual_return_at(self, timestamp) -> float:
        """Return the view's annualised return known at ``timestamp``, NaN before the first.

        Parameters
        ----------
        timestamp : pd.Timestamp or np.datetime64
            The bar.

        Returns
        -------
        float
            The constant, or the series' last finite value dated at or
            before ``timestamp``.

        Examples
        --------
        With ``view`` reading a series whose first value is dated
        2024-03-04:

        >>> view.annual_return_at(pd.Timestamp("2024-03-01"))  # doctest: +SKIP
        nan
        """
        if self.config.annual_return is not None:
            return float(self.config.annual_return)
        if self._series is None:
            self._series = self._read(None, None)
        rows = self._series.sel(timestamp=slice(None, timestamp))
        record_read(self, rows, store=self.config.store, reread_range=self._read)
        values = np.asarray(rows[self.config.variable].values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite[-1]) if finite.size else math.nan

    @property
    def risk_factors(self) -> tuple[str, ...]:
        """The risk-model factors the view names, checked against the risk model; none by default.

        Examples
        --------
        >>> view.risk_factors
        ()
        """
        return ()

    @abstractmethod
    def row(self, factor_names: Sequence[str], market_exposures: np.ndarray) -> np.ndarray | None:
        """Return the view's row ``p`` on the bar's factor returns, ``None`` when the bar cannot price it.

        Parameters
        ----------
        factor_names : Sequence[str]
            The factors of the bar's forecast.
        market_exposures : np.ndarray
            The market's exposure to each of them.

        Returns
        -------
        np.ndarray or None
            ``p`` with the view ``p' f = q``; ``None`` when the bar's
            forecast cannot price it.

        Examples
        --------
        >>> view.row(("market", "value"), np.array([1.0, 0.2]))
        array([1. , 0.2])
        """

    def variance(self, prior: float, tau: float, span: int) -> float:
        """Return the view's variance over the span.

        Parameters
        ----------
        prior : float
            ``p' F p``, the viewed return's variance over the span.
        tau : float
            The rule's ``tau``.
        span : int
            The span in bars.

        Returns
        -------
        float
            ``tau (1 - C)/C prior`` with a confidence ``C``, else ``(std *
            span / 252) ** 2``.

        Examples
        --------
        >>> view.variance(prior=4e-4, tau=0.05, span=5)
        2e-05
        """
        confidence = self.config.confidence
        if confidence is not None:
            return tau * (1.0 - confidence) / confidence * prior
        return (self.config.std * span / BARS_PER_YEAR) ** 2


class MarketReturnView(ReturnView):
    """The user's view on the market portfolio's return, for ``BlackLittermanOptimizer``.

    A ``ReturnView`` whose row is the market's factor exposures ``x = B'
    w``: the market's specific return is neglected.

    Parameters
    ----------
    config : MarketReturnViewConfig
        The view's value and its uncertainty.

    Raises
    ------
    ValueError
        As ``ReturnView``.

    Examples
    --------
    >>> from quantlab.portfolio.config import MarketReturnViewConfig
    >>> view = MarketReturnView(MarketReturnViewConfig(annual_return=0.06, std=0.04))
    >>> view.row(("market", "value"), np.array([1.0, 0.2]))
    array([1. , 0.2])
    """

    config_cls = MarketReturnViewConfig

    def row(self, factor_names: Sequence[str], market_exposures: np.ndarray) -> np.ndarray:
        """Return the market's exposures ``x``.

        Examples
        --------
        >>> view.row(("market",), np.array([0.9]))
        array([0.9])
        """
        return np.asarray(market_exposures, dtype=np.float64)


def _unit_rows(factor_names: Sequence[str], weights: dict[str, float]) -> np.ndarray | None:
    """Return the row with ``weights`` on the named factors, ``None`` when one is not among ``factor_names``."""
    names = list(factor_names)
    if any(name not in names for name in weights):
        return None
    row = np.zeros(len(names))
    for name, weight in weights.items():
        row[names.index(name)] = weight
    return row


class FactorReturnView(ReturnView):
    """The user's view on one risk-model factor's return, for ``BlackLittermanOptimizer``.

    A ``ReturnView`` on the factor ``factor`` (its name in the risk model):
    the view's row is that factor's unit vector, and a bar whose forecast
    lacks the factor leaves the view out.

    Parameters
    ----------
    config : FactorReturnViewConfig
        The factor, the view's value and its uncertainty.

    Raises
    ------
    ValueError
        As ``ReturnView``.

    Examples
    --------
    >>> from quantlab.portfolio.config import FactorReturnViewConfig
    >>> view = FactorReturnView(FactorReturnViewConfig(factor="value", annual_return=0.02, std=0.01))
    >>> view.row(("market", "value"), np.array([1.0, 0.2]))
    array([0., 1.])
    >>> view.row(("market",), np.array([1.0])) is None
    True
    """

    config_cls = FactorReturnViewConfig

    @property
    def risk_factors(self) -> tuple[str, ...]:
        """``(factor,)``.

        Examples
        --------
        >>> view.risk_factors
        ('value',)
        """
        return (self.config.factor,)

    def row(self, factor_names: Sequence[str], market_exposures: np.ndarray) -> np.ndarray | None:
        """Return the view's row on the factor returns: ``factor``'s unit vector, ``None`` without it.

        Examples
        --------
        >>> view.row(("value", "market"), np.zeros(2))
        array([1., 0.])
        """
        return _unit_rows(factor_names, {self.config.factor: 1.0})


class RelativeFactorReturnView(ReturnView):
    """The user's view on one risk-model factor's return less another's, for ``BlackLittermanOptimizer``.

    A ``ReturnView`` on ``f_outperformer - f_underperformer``: "value earns
    2% a year more than momentum". Its row is ``+1`` on ``outperformer``
    and ``-1`` on ``underperformer``; a bar whose forecast lacks either
    leaves the view out.

    Parameters
    ----------
    config : RelativeFactorReturnViewConfig
        The two factors, the view's value and its uncertainty.

    Raises
    ------
    ValueError
        If ``outperformer`` and ``underperformer`` are one factor, and as
        ``ReturnView``.

    Examples
    --------
    >>> from quantlab.portfolio.config import RelativeFactorReturnViewConfig
    >>> view = RelativeFactorReturnView(RelativeFactorReturnViewConfig(
    ...     outperformer="value", underperformer="momentum", annual_return=0.02, confidence=0.5,
    ... ))
    >>> view.row(("market", "momentum", "value"), np.zeros(3))
    array([ 0., -1.,  1.])
    >>> view.risk_factors
    ('value', 'momentum')
    """

    config_cls = RelativeFactorReturnViewConfig

    def __init__(self, config: RelativeFactorReturnViewConfig):
        """Check the two factors differ, then as ``ReturnView``."""
        super().__init__(config)
        if config.outperformer == config.underperformer:
            raise ValueError(
                f"RelativeFactorReturnView's outperformer and underperformer are both "
                f"{config.outperformer!r}; a view on one factor is a FactorReturnView"
            )

    @property
    def risk_factors(self) -> tuple[str, ...]:
        """``(outperformer, underperformer)``.

        Examples
        --------
        >>> view.risk_factors
        ('value', 'momentum')
        """
        return (self.config.outperformer, self.config.underperformer)

    def row(self, factor_names: Sequence[str], market_exposures: np.ndarray) -> np.ndarray | None:
        """Return ``+1`` on ``outperformer`` and ``-1`` on ``underperformer``, ``None`` without either.

        Examples
        --------
        >>> view.row(("value", "momentum"), np.zeros(2))
        array([ 1., -1.])
        """
        return _unit_rows(
            factor_names, {self.config.outperformer: 1.0, self.config.underperformer: -1.0}
        )


class BlackLittermanOptimizer(MeanVarianceOptimizer):
    """Mean-variance weights on a Black-Litterman posterior.

    See the module docstring for the model. On each rebalance bar the
    candidates are every selectable symbol (tradable, not locked, covered by
    the forecast, with every bounded exposure): a symbol without a view gets
    its equilibrium return plus what the views on correlated symbols move
    it by. The symbol views are the ``expected_return`` model's ``mu`` on
    the predicted candidates over the span, each a total return or, with
    ``views_relative_to_prior``, an alpha added to the symbol's equilibrium
    return; ``omega`` sets their uncertainty. The user's views are
    ``factor_views`` (``FactorReturnView``, ``RelativeFactorReturnView``)
    and ``market_views`` (``MarketReturnView``). Without ``expected_return`` the rule reads no
    prediction and its span is ``span``. The market's
    weights are ``market``'s output ``market_weight`` at the bar on the
    symbols the forecast covers (NaN counts as 0), renormalised to sum to
    one. ``candidate_top_k`` ranks the posterior expected return. Every other
    field works as in ``MeanVarianceOptimizer``; ``risk_aversion`` is both
    the optimiser's and the equilibrium's ``delta``.

    Parameters
    ----------
    config : BlackLittermanConfig
        ``MeanVarianceConfig``'s fields, the ``market`` factor, its
        ``market_weight`` output, ``tau``, ``omega`` (with ``confidence``),
        ``views_relative_to_prior``, ``span`` and the user's views.

    Raises
    ------
    ValueError
        If there are no views (no ``expected_return`` and no user views),
        ``expected_return`` and ``expected_return_label`` are not set
        together, ``span`` is missing without ``expected_return`` or not a
        positive integer, a ``factor_views`` entry is not a ``FactorReturnView``
        or ``RelativeFactorReturnView`` or a ``market_views`` one not a
        ``MarketReturnView``, a factor view names a factor the risk model
        lacks, ``bind`` finds a ``span`` other than
        the label's, ``covariance`` is not
        a ``FactorRiskStoreEstimator``, ``market`` is missing or lacks
        ``market_weight``, ``tau`` is not positive, ``omega`` is unknown,
        ``confidence`` is missing or outside ``(0, 1]`` with ``"idzorek"``
        or set with another ``omega``, or the rule measures risk against a
        reference book (``reference_weights``); and as
        ``MeanVarianceOptimizer``.

    Warns
    -----
    UserWarning
        If a z-score expected return model (``GrinoldExpectedReturn``,
        ``ZScoreExpectedReturn``, ``UnscaledZScoreExpectedReturn``) gives
        total-return views: its ``mu`` is a cross-sectionally centred alpha,
        so as a total return it pulls every viewed name towards zero; set
        ``views_relative_to_prior=True``.

    Examples
    --------
    ``forecast`` is a ``FactorRiskForecast`` over ``AAA``..``FFF``,
    ``risk`` a ``FactorRiskStoreEstimator`` returning it, ``cap_weight`` a
    factor whose output ``benchmark_weight`` is 0.3, 0.25, 0.2, 0.15, 0.07
    and 0.03, and ``context`` a bar whose predictions are all NaN. With no
    views the book is the market:

    >>> from quantlab.portfolio.config import BlackLittermanConfig, RawExpectedReturnConfig
    >>> from quantlab.portfolio.predefined.expected_return import RawExpectedReturn
    >>> rule = BlackLittermanOptimizer(BlackLittermanConfig(
    ...     expected_return_label="ret_5", covariance=risk,
    ...     expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
    ...     risk_aversion=3.0, market=cap_weight, tau=0.05,
    ... ))  # doctest: +SKIP
    >>> rule.bind(specs)  # doctest: +SKIP
    >>> rule.construct(context).values.round(4)  # doctest: +SKIP
    array([0.3 , 0.25, 0.2 , 0.15, 0.07, 0.03])
    """

    config_cls = BlackLittermanConfig
    _requires_expected_return = False

    def __init__(self, config: BlackLittermanConfig):
        """Check the views, the covariance estimator, the market, ``tau`` and ``omega``; see the class docstring."""
        if config.expected_return is None and not config.factor_views and not config.market_views:
            raise ValueError(
                "BlackLittermanOptimizer has no views: set expected_return (the symbol views), "
                "factor_views or market_views"
            )
        if (config.expected_return is None) != (config.expected_return_label is None):
            raise ValueError(
                f"expected_return and expected_return_label give the symbol views together: "
                f"set both or neither, got expected_return_label="
                f"{config.expected_return_label!r} and expected_return="
                f"{type(config.expected_return).__name__}"
            )
        span = config.span
        if span is None:
            if config.expected_return is None:
                raise ValueError(
                    "without expected_return the rule has no label to read its span from: set span"
                )
        elif isinstance(span, bool) or not isinstance(span, (int, np.integer)) or span < 1:
            raise ValueError(f"span must be a positive integer or None, got {span!r}")
        for field, views, classes in (
            ("factor_views", config.factor_views, (FactorReturnView, RelativeFactorReturnView)),
            ("market_views", config.market_views, (MarketReturnView,)),
        ):
            for view in views:
                if not isinstance(view, classes):
                    raise ValueError(
                        f"{field} holds {' or '.join(cls.__name__ for cls in classes)}, got "
                        f"{type(view).__name__}"
                    )
        risk_model = getattr(getattr(config.covariance, "config", None), "risk_model", None)
        if risk_model is not None and config.factor_views:
            names = tuple(risk_model.factor_names)
            unknown = [
                name for view in config.factor_views for name in view.risk_factors
                if name not in names
            ]
            if unknown:
                raise ValueError(
                    f"factor_views names {unknown}, which the risk model's factors {list(names)} "
                    f"do not include"
                )
        if not isinstance(config.covariance, FactorRiskStoreEstimator):
            raise ValueError(
                f"BlackLittermanOptimizer's covariance must be a FactorRiskStoreEstimator, got "
                f"{type(config.covariance).__name__}: the prior and the posterior price risk "
                f"from one factor risk model's forecast"
            )
        if config.market is None:
            raise ValueError("BlackLittermanOptimizer needs a market factor (market)")
        if config.market_weight not in config.market.get_factor_names():
            raise ValueError(
                f"the market factor has no output market_weight={config.market_weight!r} "
                f"({list(config.market.get_factor_names())})"
            )
        if not config.tau > 0:
            raise ValueError(f"tau must be positive, got {config.tau}")
        if config.omega not in ("he_litterman", "idzorek", "specific"):
            raise ValueError(
                f"omega must be 'he_litterman', 'idzorek' or 'specific', got {config.omega!r}"
            )
        if config.omega == "idzorek":
            if config.confidence is None or not 0 < config.confidence <= 1:
                raise ValueError(
                    f"omega='idzorek' needs a confidence in (0, 1], got {config.confidence}"
                )
        elif config.confidence is not None:
            raise ValueError(
                f"confidence={config.confidence} is Idzorek's; omega={config.omega!r} does not "
                f"read it"
            )
        if not config.views_relative_to_prior and isinstance(
            config.expected_return, (GrinoldExpectedReturn, ZScoreExpectedReturn, UnscaledZScoreExpectedReturn)
        ):
            warnings.warn(
                f"{type(config.expected_return).__name__} gives a cross-sectionally centred "
                f"alpha, used here as total-return views (views_relative_to_prior=False): every "
                f"viewed name is pulled towards a zero return; set views_relative_to_prior=True "
                f"to add the alpha to the equilibrium return",
                UserWarning,
                stacklevel=2,
            )
        if type(self).reference_weights is not MeanVarianceOptimizer.reference_weights:
            raise ValueError(
                f"{type(self).__name__} overrides reference_weights: an active-risk reference "
                f"prices lambda * Sigma * b, the equilibrium prior again, so Black-Litterman "
                f"refuses it"
            )
        super().__init__(config)

    def declared_inputs(self) -> InputDeclaration:
        """The optimiser's declaration, and the market among the factors.

        Examples
        --------
        >>> rule.declared_inputs().factors[-1] is rule.config.market  # doctest: +SKIP
        True
        """
        return super().declared_inputs().merged(InputDeclaration(factors=(self.config.market,)))

    def axis_factors(self) -> tuple[str, ...]:
        """The market's weights: its symbols without a prediction are in the equilibrium portfolio.

        Examples
        --------
        >>> rule.axis_factors() == (rule.config.market_weight,)  # doctest: +SKIP
        True
        """
        return (*super().axis_factors(), self.config.market_weight)

    def ranking_label(self, names: Sequence[str]) -> str:
        """Return ``expected_return_label``, or without symbol views the first label.

        Examples
        --------
        >>> rule.ranking_label(["ret_1", "ret_5"])  # doctest: +SKIP
        'ret_5'
        """
        if self.config.expected_return_label is None:
            return PortfolioConstructor.ranking_label(self, names)
        return super().ranking_label(names)

    def bind(self, labels: Sequence[LabelSpec]) -> None:
        """Read the span: the expected-return label's, or ``span`` without symbol views.

        Parameters
        ----------
        labels : Sequence[LabelSpec]
            The specs of the predicted labels; not read without
            ``expected_return``.

        Raises
        ------
        ValueError
            If ``span`` is set and differs from the label's span, and as
            ``MeanVarianceOptimizer.bind``.

        Examples
        --------
        >>> rule.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])  # doctest: +SKIP
        >>> rule.span  # doctest: +SKIP
        5
        """
        config = self.config
        if config.expected_return is None:
            self._span = int(config.span)
            return
        super().bind(labels)
        if config.span is not None and int(config.span) != self._span:
            label_span, self._span = self._span, None
            raise ValueError(
                f"span={config.span} differs from the span of expected_return_label "
                f"{config.expected_return_label!r}, {label_span} bars; the two must match"
            )

    def market_weights(self, context: PortfolioContext, forecast: FactorRiskForecast) -> np.ndarray:
        """Return the market's weights on the forecast's symbols, summing to one.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context; ``context.factors`` holds ``market_weight``.
        forecast : FactorRiskForecast
            The bar's forecast.

        Returns
        -------
        np.ndarray
            One weight per ``forecast.symbols``.

        Raises
        ------
        PortfolioConstructionError
            If the market has no weight on the covered symbols.

        Examples
        --------
        >>> rule.market_weights(context, forecast).sum()  # doctest: +SKIP
        1.0
        """
        weights = context.factors[self.config.market_weight].reindex(symbol=forecast.symbols)
        weights = np.nan_to_num(np.asarray(weights.values, dtype=np.float64))
        total = weights.sum()
        if not total > 0:
            raise PortfolioConstructionError(
                f"the market ({self.config.market_weight!r}) has no weight on the "
                f"{len(forecast.symbols)} symbols the forecast covers at {context.timestamp}"
            )
        return weights / total

    def market_exposures(
        self, context: PortfolioContext, forecast: FactorRiskForecast, weights: np.ndarray
    ) -> np.ndarray:
        """Return the market's factor exposures ``x``: ``B' w`` on the forecast's symbols.

        A market wider than the roster (the whole market, say) overrides it
        together with ``market_weights``, the market's weights on the
        roster, so that ``x`` and the weights describe one portfolio.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context.
        forecast : FactorRiskForecast
            The bar's forecast.
        weights : np.ndarray
            The market's weights on ``forecast.symbols`` (``market_weights``).

        Returns
        -------
        np.ndarray
            One exposure per factor of the forecast.

        Examples
        --------
        >>> rule.market_exposures(context, forecast, weights).shape  # doctest: +SKIP
        (2,)
        """
        return forecast.exposures.T @ weights

    def posterior(
        self,
        context: PortfolioContext,
        symbols: np.ndarray,
        expected_return: np.ndarray,
        estimate: FactorRiskForecast,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, FactorRiskForecast, np.ndarray]:
        """Return the posterior expected return, the posterior covariance and every selectable symbol.

        See the module docstring for the model; the views are the finite
        entries of ``expected_return``, total returns or alphas
        (``views_relative_to_prior``), with ``omega``'s uncertainty.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context; ``context.factors`` holds ``market_weight``.
        symbols : np.ndarray
            The selectable symbols.
        expected_return : np.ndarray
            The ``expected_return`` model's ``mu`` per selectable symbol on
            the span, NaN without a prediction: the symbol views.
        estimate : FactorRiskForecast
            The bar's forecast on the span.
        candidates : np.ndarray
            Ignored: every selectable symbol is a candidate.

        Returns
        -------
        expected_return : np.ndarray
            The posterior mean per selectable symbol.
        estimate : FactorRiskForecast
            The posterior covariance ``Sigma + M`` over ``estimate``'s
            symbols, its factors the forecast's and their posterior copies.
        candidates : np.ndarray
            All ``True``.

        Examples
        --------
        With no views, the posterior mean is the equilibrium and the
        covariance ``(1 + tau) Sigma``:

        >>> mu, posterior, chosen = rule.posterior(
        ...     context, symbols, np.full(len(symbols), np.nan), estimate, candidates,
        ... )  # doctest: +SKIP
        >>> np.allclose(posterior.covariance, (1 + rule.config.tau) * estimate.covariance)  # doctest: +SKIP
        True
        >>> bool(chosen.all())  # doctest: +SKIP
        True
        """
        exposures, factor_covariance, specific = estimate.factor_form()
        weights = self.market_weights(context, estimate)
        x = self.market_exposures(context, estimate, weights)
        position = pd.Index(estimate.symbols).get_indexer(symbols)
        viewed = np.isfinite(expected_return)
        rows = position[viewed]
        variance = None if self.config.omega == "specific" else estimate.subset(rows).variance
        mean, factor_posterior, keep, specific_posterior = self.moments(
            context.timestamp, estimate.factor_names, (exposures, factor_covariance, specific),
            (weights, x), rows, variance, np.asarray(expected_return, dtype=np.float64)[viewed],
        )
        k = len(estimate.factor_names)
        covariance = np.zeros((2 * k, 2 * k))
        covariance[:k, :k] = factor_covariance
        covariance[k:, k:] = factor_posterior
        forecast = FactorRiskForecast(
            symbols=estimate.symbols,
            factor_names=tuple(estimate.factor_names)
            + tuple(f"{name} (posterior)" for name in estimate.factor_names),
            exposures=np.concatenate([exposures, keep[:, None] * exposures], axis=1),
            factor_covariance=covariance,
            specific_variance=specific + specific_posterior,
        )
        return mean[position], forecast, np.ones(len(symbols), dtype=bool)

    def moments(
        self,
        timestamp,
        factor_names: Sequence[str],
        factor_form: tuple[np.ndarray, np.ndarray, np.ndarray],
        market: tuple[np.ndarray, np.ndarray],
        rows: np.ndarray,
        variance: np.ndarray | None,
        views: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the posterior on one bar's arrays: the mean, the factor covariance and the diagonal parts of ``M``.

        The model's arithmetic (see the module docstring) on the bar's
        forecast in factor form, shared by ``posterior`` and the approximate
        backtest's implementation of the rule, which hands it each bar's
        arrays; the user's views known at ``timestamp`` are read here.

        Parameters
        ----------
        timestamp : pd.Timestamp or np.datetime64
            The bar, at which the user's views are read.
        factor_names : Sequence[str]
            The forecast's factors.
        factor_form : tuple of np.ndarray
            ``(B, F, D)`` over the forecast's ``n`` symbols on the span:
            ``[n, k]`` exposures, ``[k, k]`` factor covariance, ``[n]``
            specific variances.
        market : tuple of np.ndarray
            ``(w, x)``: the market's weights on the ``n`` symbols, summing to
            one, and its factor exposures.
        rows : np.ndarray
            The viewed symbols' positions among the ``n``.
        variance : np.ndarray or None
            ``Sigma_jj`` of each viewed symbol; not read (``None``) with
            ``omega="specific"``.
        views : np.ndarray
            The expected return model's ``mu`` of each viewed symbol.

        Returns
        -------
        mean : np.ndarray
            ``[n]``: the posterior expected return.
        factor_posterior : np.ndarray
            ``[k, k]``: the posterior covariance of ``f``, ``F_post``.
        keep : np.ndarray
            ``[n]``: how much of each symbol's specific prior variance a view
            keeps, so ``M = U F_post U' + diag(specific)``, ``U = diag(keep) B``.
        specific : np.ndarray
            ``[n]``: the diagonal of ``M``.

        Raises
        ------
        PortfolioConstructionError
            If the user's views cannot be blended (two fully confident views
            on one return, or a confident view on a factor without variance).

        Examples
        --------
        With no views the mean is the equilibrium ``delta (1 + tau) Sigma w``:

        >>> mean, _, keep, _ = rule.moments(
        ...     day, forecast.factor_names, forecast.factor_form(), (w, forecast.exposures.T @ w),
        ...     np.array([], dtype=int), np.array([]), np.array([]),
        ... )  # doctest: +SKIP
        >>> np.allclose(mean, 3.0 * 1.05 * forecast.covariance @ w), bool((keep == 1).all())  # doctest: +SKIP
        (True, True)
        """
        exposures, factor_covariance, specific = factor_form
        weights, x = market
        tau, delta = self.config.tau, self.config.risk_aversion
        if self.config.omega == "specific":
            omega = tau * specific[rows]
        else:
            omega = tau * variance  # He-Litterman, tau Sigma_jj
            if self.config.omega == "idzorek":  # scaled by (1 - C) / C
                confidence = self.config.confidence
                omega = omega * ((1.0 - confidence) / confidence)

        # The specific prior's variance, and how much of it a view keeps.
        tau_specific = tau * specific
        keep = np.ones(len(specific))
        keep[rows] = omega / (omega + tau_specific[rows])
        specific_posterior = tau_specific * keep  # the diagonal of M
        tau_factor = tau * factor_covariance
        # The user's views update the factor prior first: F_1 = tau F - G P tau F,
        # G = tau F P' (P tau F P' + Omega_u)^-1, which takes a zero Omega_u.
        user_rows, user_q, user_omega = self._user_views(timestamp, factor_names, x, tau_factor)
        prior_factor = tau_factor
        if len(user_q):
            spread = user_rows @ tau_factor @ user_rows.T + np.diag(user_omega)
            try:
                gain = np.linalg.solve(spread, user_rows @ tau_factor).T
            except np.linalg.LinAlgError as error:
                raise PortfolioConstructionError(
                    f"the user views at {timestamp} cannot be blended: two fully "
                    f"confident views on one return, or a confident view on a factor without "
                    f"variance, leave P tau F P' + Omega singular ({error})"
                ) from error
            prior_factor = tau_factor - gain @ user_rows @ tau_factor
            prior_factor = (prior_factor + prior_factor.T) / 2
        # The posterior factor covariance (I + F_1 H)^-1 F_1, H the symbol views' precision on f.
        viewed_exposures = exposures[rows]
        precision = viewed_exposures.T @ (viewed_exposures / (omega + tau_specific[rows])[:, None])
        system = np.eye(len(tau_factor)) + prior_factor @ precision
        factor_posterior = np.linalg.solve(system, prior_factor)
        factor_posterior = (factor_posterior + factor_posterior.T) / 2
        shrunk_exposures = keep[:, None] * exposures  # M = U F_post U' + diag(...), U = diag(keep) B

        # The equilibrium against the posterior covariance Sigma + M, from x and w.
        shrunk_market = x - exposures.T @ ((1.0 - keep) * weights)  # U' w
        pi = delta * (
            exposures @ (factor_covariance @ x)
            + specific * weights
            + shrunk_exposures @ (factor_posterior @ shrunk_market)
            + specific_posterior * weights
        )
        pi_factor = delta * factor_covariance @ x
        pi_specific = pi - exposures @ pi_factor
        # The views, here because an alpha view is added to pi.
        q = views
        if self.config.views_relative_to_prior:
            q = q + pi[rows]

        # The posterior mean of f, then of e, then of r = B f + e.
        factor_prior_mean = pi_factor
        if len(user_q):
            factor_prior_mean = pi_factor + gain @ (user_q - user_rows @ pi_factor)
        surprise = (q - pi_specific[rows]) / (omega + tau_specific[rows])
        factor_mean = np.linalg.solve(
            system, factor_prior_mean + prior_factor @ (viewed_exposures.T @ surprise)
        )
        specific_mean = keep * pi_specific
        specific_mean[rows] += (1.0 - keep[rows]) * (q - viewed_exposures @ factor_mean)
        mean = exposures @ factor_mean + specific_mean
        return mean, factor_posterior, keep, specific_posterior

    def _user_views(
        self,
        timestamp,
        factor_names: Sequence[str],
        market_exposures: np.ndarray,
        tau_factor: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the user views known at the bar: their rows on ``f``, values and variances over the span."""
        span = self._span
        rows, values, variances = [], [], []
        for view in (*self.config.factor_views, *self.config.market_views):
            row = view.row(factor_names, market_exposures)
            if row is None:
                continue
            annual = view.annual_return_at(timestamp)
            if not math.isfinite(annual):
                continue
            prior = float(row @ tau_factor @ row) / self.config.tau
            rows.append(row)
            values.append(annual * span / BARS_PER_YEAR)
            variances.append(view.variance(prior, self.config.tau, span))
        k = len(factor_names)
        return (
            np.asarray(rows, dtype=np.float64).reshape(len(rows), k),
            np.asarray(values, dtype=np.float64),
            np.asarray(variances, dtype=np.float64),
        )


__all__ = [
    "BlackLittermanOptimizer",
    "FactorReturnView",
    "MarketReturnView",
    "RelativeFactorReturnView",
    "ReturnView",
]
