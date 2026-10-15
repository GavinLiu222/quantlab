"""The US-equity cross-sectional stock-selection backtester.

``USEquityCrossectionSelectStockVectorBt`` is the concrete backtester of the
pipeline for daily US equities. It composes the ``US_EQUITY_MARKET`` price
conventions, the portfolio construction rule of its config that turns
predictions into target weights, and the vectorbt simulation engine
inherited from ``VectorBtBacktester``. A saved run rebuilds it through
``quantlab.runs.backtest_run.BacktestRun.rebuild_backtester``.
``USEquityApproximateBacktest`` is the approximate backtest on the same
conventions.
"""

import xarray as xr

from quantlab.backtest.approximate import ApproximateBacktest
from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.base import MarketSpec, label_specs
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.base import PortfolioConstructor
from quantlab.portfolio.decision_inputs import DecisionInputs

#: Price conventions for US equities. Orders fill at the split- and
#: dividend-adjusted open and the portfolio is valued at the adjusted close;
#: the unadjusted price columns never enter the profit and loss. The column
#: names are spelled only here, never inside a method body.
US_EQUITY_MARKET = MarketSpec(
    fill_price_column="adjOpen",
    valuation_price_column="adjClose",
    trading_days_per_year=252,
    session_minutes_per_day=390,
)


class USEquityCrossectionSelectStockVectorBt(VectorBtBacktester):
    """Daily US-equity backtester: a portfolio construction rule simulated with vectorbt.

    On every rebalance bar the config's ``constructor`` turns the
    predictions of that bar into target weights, for example the top
    ``top_n`` symbols with ``TopNConstructor``, and the targets are held
    until the next rebalance bar. A symbol is tradable at a bar as the price
    dataset's ``tradable_bars`` says (by default, when it has a fill price
    at that bar), and a held symbol that is not tradable stays locked. The
    class only wires the pieces together: the market conventions are
    ``MARKET``, the rule is the config's ``constructor``, and every engine
    behaviour comes from ``VectorBtBacktester``.

    Parameters
    ----------
    config : CrossSectionBacktestConfig
        The backtest configuration: price dataset, model, date window and
        output directory, plus ``rebalance_periods`` and ``constructor``.

    Raises
    ------
    TypeError
        If ``constructor`` is not a ``PortfolioConstructor``.
    ValueError
        If the constructor refuses the model's labels.

    Examples
    --------
    >>> from quantlab.backtest.config import CrossSectionBacktestConfig
    >>> from quantlab.portfolio.config import TopNConfig
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> backtester = USEquityCrossectionSelectStockVectorBt(
    ...     CrossSectionBacktestConfig(
    ...         price_dataset=prices,  # a StockDataset over a Zarr store
    ...         model=model,  # a model whose checkpoint is given below
    ...         model_mode="load",
    ...         checkpoint="models/first_feature/checkpoint.joblib",
    ...         start_date="2024-02-12",
    ...         end_date="2024-03-11",
    ...         output_dir="runs",
    ...         rebalance_periods=5,
    ...         constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
    ...     )
    ... )
    >>> result = backtester.run()
    >>> BacktestRun.open(result.run_dir).kind
    'run'
    >>> result.weights["weight"].dims
    ('timestamp', 'symbol')
    """

    config_cls = CrossSectionBacktestConfig
    MARKET = US_EQUITY_MARKET

    def _validate_config(self) -> None:
        """Check the constructor against the model's labels at construction time.

        A label the rule needs and the model does not predict is reported
        before any data is read or any model trained. A config without a
        model (for ``run_weights()``) has no labels to check.
        """
        config = self.config
        if not isinstance(config.constructor, PortfolioConstructor):
            raise TypeError(
                f"{self.class_name}: constructor must be a PortfolioConstructor, "
                f"got {type(config.constructor).__name__}"
            )
        if config.model is not None:
            config.constructor.bind(label_specs(config.model))

    def _generate_signals(
        self, predictions: xr.Dataset, prices: xr.Dataset, delisted: xr.DataArray
    ) -> xr.Dataset:
        """Turn the model's predictions into target weights through ``DecisionInputs``.

        The decision inputs (tradability, the price windows and their
        warm-up, the rule's factor panels, the holdings replayed with the
        run's sizing basis, fees and slippage) are assembled by
        ``quantlab.portfolio.decision_inputs.DecisionInputs``, anchored on
        the predictions' first bar and handed ``delisted``, the marks the
        simulation settles. Bars the rule failed on are kept for
        ``_signal_metrics``.
        """
        weights = DecisionInputs(
            self.config.price_dataset,
            self.config.constructor,
            fill_column=self.MARKET.fill_price_column,
            valuation_column=self.MARKET.valuation_price_column,
            rebalance_periods=self.config.rebalance_periods,
            anchor=predictions.timestamp.values[0],
            execution=self.config.execution,
        ).weights(predictions, delisted=delisted)
        self._failed_bars = list(weights.attrs.pop("failed_bars", []))
        self._events = dict(weights.attrs.pop("events", {}))
        return weights

    def _signal_metrics(self) -> dict:
        """Report the bars the constructor could not decide, and its events.

        ``{"portfolio_construction": {"failed_bar_count": n, "failed_bars":
        [...], <event>: {"count": m, "bars": [...]}}}``, the bars as ISO
        timestamps; an event appears only when it happened, ``count`` being
        the number of symbols it involved over all its bars and ``bars`` one
        record per bar, ``{"bar", "symbols"}`` for an event that names them
        (the optimiser's ``closed_without_risk``) or ``{"bar", "count"}`` for
        one that counts them (the top-n rule's ``tie_at_cutoff``).
        """
        failed = list(getattr(self, "_failed_bars", []))
        block = {"failed_bar_count": len(failed), "failed_bars": failed}
        for name, records in getattr(self, "_events", {}).items():
            block[name] = {
                "count": sum(
                    record["count"] if "count" in record else len(record["symbols"])
                    for record in records
                ),
                "bars": list(records),
            }
        return {"portfolio_construction": block}


class USEquityApproximateBacktest(ApproximateBacktest):
    """The approximate backtest of daily US equities: ``ApproximateBacktest`` on ``US_EQUITY_MARKET``.

    Fills at the adjusted open, values at the adjusted close, as
    ``USEquityCrossectionSelectStockVectorBt`` does, so the same rule and
    predictions can be confirmed with a backtest run.

    Examples
    --------
    >>> from quantlab.backtest.config import ApproximateBacktestConfig
    >>> backtest = USEquityApproximateBacktest(
    ...     ApproximateBacktestConfig(
    ...         price_dataset=prices, start_date="2024-03-25", end_date="2024-06-26",
    ...         rebalance_periods=5,
    ...     )
    ... )
    >>> backtest.MARKET.fill_price_column
    'adjOpen'
    """

    MARKET = US_EQUITY_MARKET
