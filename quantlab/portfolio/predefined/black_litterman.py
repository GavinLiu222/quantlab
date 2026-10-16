"""Black-Litterman on a factor risk model: ``BlackLittermanOptimizer``.

A mean-variance rule (ADR 0047) whose expected return and covariance are a
Black-Litterman posterior: an equilibrium prior from the declared market
portfolio, blended with the views of the rule's expected return model by
Bayes' rule, in the risk model's factor space. Returns are ``r = B f + e``,
the factor returns and the specific returns with prior means ``pi_f`` and
``pi_e`` and prior covariances ``tau F`` and ``tau diag(D)``, so the
expected return has the prior ``N(pi, tau Sigma)``, ``Sigma = B F B' +
diag(D)`` the forecast over the expected-return label's span. A symbol view
says ``mu_j = q_j`` with variance ``omega_j = tau Sigma_jj`` (He-Litterman).
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

The approximate backtest does not run this rule.
"""

import numpy as np
import pandas as pd

from quantlab.portfolio.base import InputDeclaration, PortfolioConstructionError, PortfolioContext
from quantlab.portfolio.config import BlackLittermanConfig
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.risk.base import FactorRiskForecast


class BlackLittermanOptimizer(MeanVarianceOptimizer):
    """Mean-variance weights on a Black-Litterman posterior.

    See the module docstring for the model. On each rebalance bar the
    candidates are every selectable symbol (tradable, not locked, covered by
    the forecast, with every bounded exposure): a symbol without a view gets
    its equilibrium return plus what the views on correlated symbols move
    it by. The views are the ``expected_return`` model's ``mu`` on the
    predicted candidates, each a total return over the span. The market's
    weights are ``market``'s output ``market_weight`` at the bar on the
    symbols the forecast covers (NaN counts as 0), renormalised to sum to
    one. ``candidate_top_k`` ranks the posterior expected return. Every other
    field works as in ``MeanVarianceOptimizer``; ``risk_aversion`` is both
    the optimiser's and the equilibrium's ``delta``.

    Parameters
    ----------
    config : BlackLittermanConfig
        ``MeanVarianceConfig``'s fields, the ``market`` factor, its
        ``market_weight`` output and ``tau``.

    Raises
    ------
    ValueError
        If ``expected_return`` is missing (no views), ``covariance`` is not
        a ``FactorRiskStoreEstimator``, ``market`` is missing or lacks
        ``market_weight``, ``tau`` is not positive, or the rule measures
        risk against a reference book (``reference_weights``); and as
        ``MeanVarianceOptimizer``.

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

    def __init__(self, config: BlackLittermanConfig):
        """Check the views, the covariance estimator, the market and ``tau``; see the class docstring."""
        if config.expected_return is None:
            raise ValueError(
                "BlackLittermanOptimizer has no views: set expected_return, the model whose "
                "expected returns are the symbol views"
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
        entries of ``expected_return``.

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
        tau, delta = self.config.tau, self.config.risk_aversion
        weights = self.market_weights(context, estimate)
        x = self.market_exposures(context, estimate, weights)
        position = pd.Index(estimate.symbols).get_indexer(symbols)
        viewed = np.isfinite(expected_return)
        rows, q = position[viewed], np.asarray(expected_return, dtype=np.float64)[viewed]
        omega = tau * estimate.subset(rows).variance  # He-Litterman

        # The specific prior's variance, and how much of it a view keeps.
        tau_specific = tau * specific
        keep = np.ones(len(specific))
        keep[rows] = omega / (omega + tau_specific[rows])
        specific_posterior = tau_specific * keep  # the diagonal of M
        # The posterior factor covariance (I + tau F H)^-1 tau F, H the views' precision on f.
        viewed_exposures = exposures[rows]
        precision = viewed_exposures.T @ (viewed_exposures / (omega + tau_specific[rows])[:, None])
        tau_factor = tau * factor_covariance
        system = np.eye(len(tau_factor)) + tau_factor @ precision
        factor_posterior = np.linalg.solve(system, tau_factor)
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

        # The posterior mean of f, then of e, then of r = B f + e.
        surprise = (q - pi_specific[rows]) / (omega + tau_specific[rows])
        factor_mean = np.linalg.solve(
            system, pi_factor + tau_factor @ (viewed_exposures.T @ surprise)
        )
        specific_mean = keep * pi_specific
        specific_mean[rows] += (1.0 - keep[rows]) * (q - viewed_exposures @ factor_mean)
        mean = exposures @ factor_mean + specific_mean

        k = len(estimate.factor_names)
        covariance = np.zeros((2 * k, 2 * k))
        covariance[:k, :k] = factor_covariance
        covariance[k:, k:] = factor_posterior
        forecast = FactorRiskForecast(
            symbols=estimate.symbols,
            factor_names=tuple(estimate.factor_names)
            + tuple(f"{name} (posterior)" for name in estimate.factor_names),
            exposures=np.concatenate([exposures, shrunk_exposures], axis=1),
            factor_covariance=covariance,
            specific_variance=specific + specific_posterior,
        )
        return mean[position], forecast, np.ones(len(symbols), dtype=bool)


__all__ = ["BlackLittermanOptimizer"]
