# Portfolio construction

English | [简体中文](zh-CN/portfolio.md)

Portfolio construction is the step between a model and a backtest. On every rebalance bar it takes the model's predictions for that bar and the weights currently held, and returns the weights to hold after the bar. The backtest then trades those weights.

Every rule derives from `PortfolioConstructor` (`quantlab/portfolio/base.py`). Three rules ship in `quantlab/portfolio/predefined/`:

- `TopNConstructor`: equal weights on the `top_n` best scores, long-only or long-short.
- `TopkDropoutConstructor`: qlib's TopkDropout. It holds `topk` names and replaces at most `n_drop` of them per rebalance, long-only or long-short.
- `MeanVarianceOptimizer`: Markowitz weights with a turnover penalty, solved with cvxpy. It prices risk with a covariance estimator. Two ship: `LedoitWolfEstimator`, a Ledoit-Wolf shrunk covariance of trailing returns, and `FactorRiskStoreEstimator`, which reads a factor risk model's stores (`quantlab/risk/`).

A backtest holds its rule in the `constructor` field of its config (see [Backtesting](backtest.md)).

## Prerequisites

Run the sessions from the repository root with `uv run python`. They build each bar's input by hand, so they need no store, model or GPU. In a backtest, the backtester builds the same input from the price dataset and the model.

## The basics

### One bar at a time

A rule has one method to implement, `construct(context)`. The `PortfolioContext` it receives describes one bar and holds nothing from a later bar:

| Field | Content |
| --- | --- |
| `timestamp` | The bar. The weights returned fill at the next bar. |
| `predictions` | Every label's prediction at the bar, one variable per label, on `symbol`. |
| `tradable` | Whether each symbol can be traded at the bar: by default, whether it has a real fill price there. |
| `current_weights` | The weights held now, valued at the bar. They are all 0.0 before the first rebalance. |
| `returns` | The last `lookback_bars` one-bar returns, ending at the bar, from the rule's last `history_bars` raw valuation prices forward-filled within that window. It is empty for a rule with `lookback_bars` of 0. |
| `staleness` | Bars since each symbol's last real price within the last `history_bars` bars; NaN when it has none there. |
| `factors` | The values at the bar of the factors the rule declares (`declared_inputs().factors`), or `None`. |
| `risk_exposures` | The exposures at the bar of the factor risk model the rule declares (`declared_inputs().risk_model`), as the model gives them (`FactorRiskModel.exposures`: read from the exposures factor's store or computed, per `exposure_data_strategy`), or `None`. |
| `risk_estimate` | The same model's estimate store row at the bar, its forecast of the next bar's risk: `factor_covariance` on `(factor_i, factor_j)` and `specific_risk` on `symbol`. It is `None` when the rule declares no risk model or the store has no row at the bar. |

A rule returns one weight per symbol:

- The weights are finite, 0.0 where nothing is held, with a gross exposure of at most one.
- An all-NaN row means "hold the current position".
- A symbol that is held but not tradable is a *locked position*; `context.locked` marks them. It must keep its current weight. A symbol that is neither tradable nor held gets 0.0.

A rule keeps no state between bars. Because it only ever sees one bar, it cannot look ahead. It can also be called from the bar handler of an event-driven engine.

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.portfolio.config import TopNConfig
>>> from quantlab.portfolio.base import PortfolioContext
>>> from quantlab.portfolio.predefined.top_n import TopNConstructor
>>> symbols = ["AAA", "BBB", "CCC", "DDD"]
>>> def on_symbols(values):
...     return xr.DataArray(values, dims="symbol", coords={"symbol": symbols})
>>> context = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, -0.1, 0.5, 0.2])}),
...     tradable=on_symbols([True, True, True, False]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.25]),
... )
>>> context.locked.values
array([False, False, False,  True])
>>> TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(context).values
array([0.375, 0.   , 0.375, 0.25 ])

```

`DDD` is halted while held, so it keeps its 0.25. The two best tradable scores share the remaining 0.75.

Equal scores are ranked by symbol order. When a book's cut falls inside a group of equal scores, the symbol order, not the model, decides which of them are held, and the row counts the tied symbols left out as a `tie_at_cutoff` event. Below, `BBB`, `CCC` and `DDD` score the same, `BBB` comes first, and two are left out. A model that predicts only a handful of distinct values makes this event appear on most rebalances.

```python
>>> tied = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, 0.5, 0.5, 0.5])}),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.0]),
... )
>>> weights = TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(tied)
>>> weights.values
array([0.5, 0.5, 0. , 0. ])
>>> weights.attrs
{'events': {'tie_at_cutoff': 2}}

```

### In a backtest

A rule only decides. Its *decision inputs*, everything it may read at a bar except the holdings, are assembled by one module, `DecisionInputs` in `quantlab/portfolio/decision_inputs.py`, for the backtest and for an executor alike. It is built from the price dataset, the fill and valuation columns, the bound rule, the rebalance period, the *anchor* (the first bar of the prediction panel, from which the rebalance schedule counts) and the execution settings. The vectorised backtest calls its `weights(predictions, delisted=...)`, which loops the rule's one-bar decision (`decide`, see [One bar outside a backtest](#one-bar-outside-a-backtest)) over the rebalance bars and builds each bar's context:

- The tradability is the price dataset's `tradable_bars`, the factor values are the rule's declared factors computed over the window with their own warm-up, and the risk exposures are its declared risk model's `exposures` over the window, one request. All of it comes from one value the rule gives, `declared_inputs()` (an `InputDeclaration`: `lookback_bars`, `history_bars`, `factors`, `risk_model`), asked once.
- The current weights are the holdings the earlier rebalances really left. They are replayed by the Execution module (`quantlab.execution.rules`) exactly as the simulation trades them, including rejected orders, delisting settlements, the sizing basis, fees and slippage. The backtester passes its config's `execution` settings and the delisting marks it hands the engine; without settings the replay sizes at the fill price and charges no costs.
- The rebalance bars are every `rebalance_periods`-th bar from the anchor; the last bar never rebalances, since an order decided there has no next bar to fill on. `rebalances(t)` asks the same schedule for one bar.
- The result records the rebalances beside `weight`: `scheduled` (the bars the schedule named) and `decided` (those the rule decided), booleans on `timestamp`. A scheduled bar the rule held, after a `PortfolioConstructionError` or an all-NaN row, is a *held bar*: scheduled and not decided. Every reader of a run's rebalances takes them from this record (see the target-weight contract in [Backtesting](backtest.md#the-target-weight-contract)).
- The return window comes from the last known price of each symbol, so a halt shows as zero returns and then the whole move on the day trading resumes.
- Each bar reads only the last `history_bars` raw valuation prices up to and including it (`lookback_bars + 1` by default; Ledoit-Wolf `lookback_bars + 1 + max_stale_bars`; mean-variance its covariance estimator's), so a decision does not depend on where the price history starts. The backtest's warm-up holds the `history_bars - 1` bars before its first bar, so that bar already has a full window.

The backtester calls the rule's `bind(labels)` when it is built, before any data is read or any model trained. `labels` holds one `LabelSpec(name, scale, delay, span)` per prediction variable, which the backtester derives from its predictor with `quantlab.backtest.base.label_specs`; `span` is `None` for a label that is not a `Forward` label. The specs are the only thing a rule may know about a prediction: it is never handed the model. This is where a rule checks the labels it needs, so a misconfigured rule fails at once.

A bar the rule cannot decide raises `PortfolioConstructionError`, for example when an optimisation is infeasible or the solver fails. `decide` turns it into a hold of the current position on that bar and logs a warning. `metrics.json` lists every such bar under `portfolio_construction` (`failed_bar_count`, `failed_bars`), along with any event a rule reported, such as `tie_at_cutoff` above or `closed_without_risk` below, with its `count` (the symbols it involved over all its bars) and one record per bar.

The run's recipe records the rule with all its parameters and its covariance estimator; `BacktestRun.rebuild("constructor")` rebuilds the rule and `rebuild_backtester()` the whole backtester (see the backtest guide).

### A run's decision inputs without the model

A run with a model (`run()` or `run_cv()`) also keeps the predictions the rule read with their label specs, as a `PredictionPanel` that `BacktestRun.predictions()` returns. `DecisionInputs.from_run(run_dir)` reads the run through `BacktestRun`: it rebuilds the rule (bound to the panel's specs) and the price dataset (an in-memory one from its copy under the run directory), takes the market columns, the execution settings and the rebalance period, and takes the anchor from the panel's first bar. It never imports the model, factor, label or backtest layers, so an executor outside the research pipeline, such as an event-driven backtest, can replay the run's decisions from the run directory alone; `weights` on the panel's predictions reproduces the run's weights. With `result` the `run()` of the first session of the [backtest guide](backtest.md#what-a-run-does), a top-2 rule rebalancing every five bars:

```python
>>> from quantlab.portfolio.decision_inputs import DecisionInputs
>>> from quantlab.runs.backtest_run import BacktestRun
>>> run = BacktestRun.open(result.run_dir)
>>> run_inputs = DecisionInputs.from_run(run.path)
>>> run_inputs.constructor == run.rebuild("constructor"), run_inputs.rebalance_periods
(True, 5)
>>> panel = run.predictions()
>>> panel.labels
(LabelSpec(name='open_ret_1', scale='raw', delay=1, span=1),)
>>> run_inputs.anchor == panel.predictions.timestamp.values[0]
True
>>> run_inputs.weights(panel.predictions).equals(run.weights())
True

```

The panel holds one variable per label on `(timestamp, symbol)`, stored with its specs. A `run_weights()` run has no model and keeps no panel: its `predictions()` is `None`.

### One bar outside a backtest

An executor that keeps its own book, such as an event-driven backtest or a live account, decides a bar with the same `DecisionInputs` and the rule's `decide`:

- `rebalances(t)` says whether bar `t` is a rebalance bar of the schedule; pass `end=` to the constructor to keep a replay's last bar from rebalancing.
- `context(t, predictions, current_weights)` builds the bar's `PortfolioContext`. `predictions` and `current_weights` are the bar's values on `symbol`; a symbol missing from `current_weights` is not held, and a held symbol without a prediction joins the bar with NaN predictions. The tradability, the return window and staleness of the last `history_bars` valuation prices up to `t`, and the factor values at `t` are read from the dataset, so an executor keeps at most `history_bars` bars of prices. Given the holdings the backtest replayed, the context equals the backtest's.
- `preloaded(start, end)`, a context manager, is for a replay that builds a context per bar (an event-driven backtest, a recheck of its decisions). On entering it reads, once over the window and through the same recorded read seams, the prices with their warm-up, their tradability, the declared factors, and the declared risk model's exposures and estimate rows (over the part of the window its estimate store covers). Inside, `context` at a bar of the window is sliced from memory and equals the context built without the preload; a bar outside the window raises `ValueError`. A live run, deciding one new bar at a time, does not preload.
- `decide(context)` runs `construct` and checks the row against the weights contract. It returns a `Decision(weights, failure, events)`: the weights on the context's symbols, all NaN for a hold; the message of a `PortfolioConstructionError` that made the bar a hold, or `None`; and the events the row reported. A row that breaks the contract (NaN mixed with finite weights, a moved locked position, weight on a symbol neither tradable nor held) is a bug in the rule and raises `ValueError`.

Here `inputs` is built directly, `DecisionInputs(dataset, rule, fill_column=..., valuation_column=..., rebalance_periods=..., anchor=...)`, over three bars of prices on which `DDD` has no price at `context.timestamp` (`from_run` builds the same from a run):

```python
>>> from quantlab.runs.prediction_panel import LabelSpec
>>> from quantlab.dataset.memory import FrameDataset
>>> bars = pd.bdate_range(end=context.timestamp, periods=3)
>>> close = xr.DataArray(
...     [[10.0, 20.0, 30.0, 40.0], [10.5, 20.5, 30.5, 40.0], [11.0, 21.0, 31.0, np.nan]],
...     dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": symbols},
... )
>>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
>>> rule.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
>>> inputs = DecisionInputs(
...     FrameDataset(xr.Dataset({"open": close, "close": close})), rule,
...     fill_column="open", valuation_column="close", rebalance_periods=1,
...     anchor=context.timestamp,
... )
>>> inputs.rebalances(context.timestamp)
True
>>> bar = inputs.context(
...     context.timestamp,
...     context.predictions,
...     xr.DataArray([0.25], dims="symbol", coords={"symbol": ["DDD"]}),
... )
>>> bar.locked.values
array([False, False, False,  True])
>>> decision = rule.decide(bar)
>>> decision.weights.values, decision.failure, decision.events
(array([0.375, 0.   , 0.375, 0.25 ]), None, {})

```

## TopkDropout

`TopkDropoutConstructor` (`quantlab/portfolio/predefined/topk_dropout.py`) follows qlib's `TopkDropoutStrategy` with `method_sell="bottom"`, `method_buy="top"` and `only_tradable=True`. It caps turnover at about `n_drop / topk` of the book per rebalance, where `TopNConstructor` rebuilds an equal-weight book every time. On each rebalance bar, for a book holding `last`:

1. The candidates are the `n_drop + topk - len(last)` best-scored symbols that are tradable, have a finite score and are not held.
2. The holdings among the `n_drop` worst of the holdings and candidates ranked together are sold. A holding is therefore sold only for a better-scored symbol. A holding without a finite score ranks worst. A locked position is never sold.
3. The first `len(sell) + topk - len(last)` candidates are bought, so an empty book fills up to `topk`.

Kept holdings stay at their current weight. The buys split equally what the kept holdings leave of the budget (1 long-only, 0.5 per side long-short), or the sold weight when that is more. The short side runs the same rule on the negated scores, and never buys a held symbol or one of the bar's long buys. qlib's `hold_thresh`, random methods and price-limit rules are not reproduced.

```python
>>> from quantlab.portfolio.config import TopkDropoutConfig
>>> from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
>>> rule = TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=2, n_drop=1))
>>> held = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, -0.1, 0.5, 0.2])}),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.55, 0.45, 0.0, 0.0]),
... )
>>> rule.construct(held).values  # BBB makes way for CCC; AAA keeps its weight
array([0.55, 0.  , 0.45, 0.  ])
```

## Mean-variance optimisation

### The problem

On each rebalance bar, `MeanVarianceOptimizer` solves

    maximise    w'mu - risk_aversion / 2 * w'Sigma w - turnover_penalty * |w - w_current|_1

- `mu` is the expected return of each symbol.
- `Sigma` is their covariance.
- `w_current` is the context's current weights.

The constraints depend on `direction`:

| `direction` | Constraints |
| --- | --- |
| `"long_only"` | `sum(w) = 1`, `0 <= w <= weight_cap`: fully invested. |
| `"long_short"` | `sum(w) = 0`, `sum(abs(w)) <= 1`, `abs(w) <= weight_cap`: dollar-neutral. |

In a long-short book the gross exposure of one is a ceiling, not a target. When the expected returns do not pay for the risk and the trading, part of the book stays uninvested, down to no position at all.

The turnover penalty is charged on the change from the weights actually held, so it prices the trades that will really happen. A locked position keeps its weight: it counts in the risk term, and the other symbols share what is left of the budget.

The candidates are the symbols that pass all of these checks:

- They are tradable.
- They are not locked.
- The covariance estimator covers them.
- They have a finite expected-return prediction or are held.

A held candidate without a prediction gets an expected return of 0.0, so its turnover cost decides whether it is closed. Every other symbol gets 0.0.

### A first optimisation

The optimiser needs to know the span and scale of the labels it reads, which a backtest hands it through `bind` as label specs. Here the specs describe a 5-bar return `ret_5` and a 5-bar volatility `vol_5`, both predicted in the labels' own units (`"raw"`, see [Expected return models](#expected-return-models)).

```python
>>> from quantlab.portfolio.config import (
...     GrinoldExpectedReturnConfig, LedoitWolfEstimatorConfig, MeanVarianceConfig,
... )
>>> from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
>>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
>>> from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
>>> from quantlab.runs.prediction_panel import LabelSpec
>>> specs = [
...     LabelSpec(name="ret_5", scale="raw", delay=1, span=5),
...     LabelSpec(name="vol_5", scale="raw", delay=1, span=5),
... ]
>>> rng = np.random.default_rng(0)
>>> window = rng.normal(0.0, 1.0, size=(60, 4)) * [0.010, 0.015, 0.020, 0.025]
>>> context = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-26"),
...     predictions=xr.Dataset({
...         "ret_5": on_symbols([0.8, -0.1, -0.3, 0.2]),
...         "vol_5": on_symbols([0.05, 0.03, 0.04, 0.06]),
...     }),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.0]),
...     returns=xr.DataArray(window, dims=("timestamp", "symbol"), coords={
...         "timestamp": pd.bdate_range("2023-12-29", periods=60), "symbol": symbols}),
... )
>>> optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.4,
... ))
>>> optimizer.bind(specs)
>>> optimizer.declared_inputs().lookback_bars, optimizer.span
(60, 5)
>>> weights = optimizer.construct(context)
>>> weights.values.round(3)
array([0.4  , 0.4  , 0.009, 0.191])

```

The book is fully invested and within the cap. With so small an expected return it leans toward the low-volatility symbols `AAA` and `BBB`. `optimizer.problem_inputs(context)` returns the candidates with their `mu`, `Sigma` and current weights, which is useful to see why the optimiser chose what it did.

### Expected return models

A model's prediction is usually a score, not a return. It ranks symbols, but its scale depends on the model and on how its training target was transformed. The optimiser's `expected_return` field holds an expected return model (`ExpectedReturnModel`, `quantlab.portfolio.predefined.expected_return`, ADR 0046) that says how `expected_return_label`'s prediction becomes the expected return `mu`:

- `GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=...))` sets `mu = ic * sigma * z`. Here `z` is the prediction's cross-sectional z-score over the candidates, `sigma` each symbol's volatility over the span (from the covariance's diagonal), and `ic` the model's information coefficient, for example the mean IC of a walk-forward cross-validation run. Any model's output can feed it, and `risk_aversion` keeps its meaning from one model to the next. A riskier name gets a larger `mu`, so without a risk term the book leans to volatile names.
- `ZScoreExpectedReturn(ZScoreExpectedReturnConfig(ic=...))` sets `mu = ic * median(sigma) * z`, the median over the bar's candidates: Grinold's scale, so the same `risk_aversion` and `turnover_penalty` mean the same, without the tilt; within a bar `mu` ranks exactly as the prediction.
- `RawExpectedReturn(RawExpectedReturnConfig())` takes the prediction itself as `mu`. It only makes sense for a model that predicts returns in their own units.
- `PredictionExpectedReturn(PredictionExpectedReturnConfig())` takes the prediction as it is, whatever its units (an ensemble's standardized score, say). `risk_aversion` and `turnover_penalty` then weigh against `mu` in the prediction's units, so they are not comparable with a Grinold rule's.

A model works on `[bars, symbols]` arrays (`expected_returns(predictions, sigma, candidates)`), so the per-bar rule and the approximate backtest call the same code. To write one, subclass `ExpectedReturnModel`, set `config_cls` and `needs_sigma`, implement `expected_returns`, and override `bind` to refuse a label it cannot read.

Every predictor reports, per label, whether its prediction is in the label's own units: `label_scales` maps each label to `"raw"` or `"standardized"`, and the label's spec carries it as `scale`. A model is `"raw"` when it was fitted on the label unchanged. An ensemble reports `"standardized"` for a label it averages over several members, because it z-scores each member first. `bind` refuses `RawExpectedReturn` on a `"standardized"` label:

```python
>>> import dataclasses
>>> from quantlab.portfolio.config import RawExpectedReturnConfig
>>> from quantlab.portfolio.predefined.expected_return import RawExpectedReturn
>>> ranked = [dataclasses.replace(specs[0], scale="standardized"), specs[1]]
>>> MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5", expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     risk_aversion=5.0,
... )).bind(ranked)
Traceback (most recent call last):
    ...
ValueError: RawExpectedReturn reads the prediction of 'ret_5' as a return, but its label spec reports its scale as 'standardized', not 'raw' (a model fitted on a transformed target, or a label an ensemble averages); use PredictionExpectedReturn to take it as it is, or GrinoldExpectedReturn

```

### Spans

`mu`, `sigma` and `Sigma` are all expressed over the span of `expected_return_label`: 5 bars for `ret_5`. The span is read from the label's spec, never configured, and a label without a span (not a `Forward` label) is refused. A covariance estimator estimates the covariance of one-bar returns, and the optimiser multiplies it by the span, because variance grows linearly with time:

```python
>>> inputs = optimizer.problem_inputs(context)
>>> one_bar = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)).estimate(context)
>>> bool(np.allclose(inputs.covariance, 5 * one_bar.covariance))
True

```

Rebalancing every span (`rebalance_periods` equal to the span) makes the horizon the optimiser plans over the one the portfolio is held for.

### Volatility from a model

Volatility moves fast and is fairly predictable. Correlations move slowly and are better estimated from history. With `volatility_label` set, the optimiser takes each symbol's volatility from a model's prediction:

- The covariance becomes the predicted volatilities around the covariance estimator's historical correlations: `Sigma = D C D`.
- The Grinold `sigma` becomes the prediction itself.

The volatility label, such as `Volatility`, is a span-scale volatility. It must have the span of the expected-return label and a `"raw"` scale, and `bind` checks both.

```python
>>> with_volatility = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5", volatility_label="vol_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.4,
... ))
>>> with_volatility.bind(specs)
>>> covariance = with_volatility.problem_inputs(context).covariance
>>> np.sqrt(np.diag(covariance)).round(4)
array([0.05, 0.03, 0.04, 0.06])
>>> with_volatility.construct(context).values.round(3)
array([0.4  , 0.399, 0.   , 0.201])

```

A symbol without a finite, positive volatility prediction has no risk estimate. It is treated like a symbol without enough history, as described under [Covariance estimators](#covariance-estimators).

The covariance estimator must take volatilities: `FactorRiskStoreEstimator` does not, since the factor risk model's forecast is its own, so `volatility_label` with it is refused when the optimiser is constructed.

One predictor has to supply both labels. A `ModelEnsemble` of a return model and a volatility model does this: a label that only one member predicts is passed through unchanged and keeps that member's scale (see [Models](model.md)).

### Long-short and the candidate pool

With `direction="long_short"` the book is dollar-neutral:

```python
>>> long_short = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.4, direction="long_short",
... ))
>>> long_short.bind(specs)
>>> weights = long_short.construct(context)
>>> weights.values.round(3)
array([ 0.4  , -0.212, -0.261,  0.073])
>>> bool(abs(weights.sum()) < 1e-12), float(abs(weights).sum().round(3))
(True, 0.945)

```

A universe of thousands of symbols is slow to optimise at every rebalance. `candidate_top_k` restricts each bar to a pool:

- the `candidate_top_k` symbols with the largest `mu` (largest `abs(mu)` long-short);
- plus every symbol currently held (`|w| > HELD_WEIGHT_TOLERANCE`, 1e-8, so a solver's residue is not a holding), so that a holding that fell out of the top can still be closed at its turnover cost.

Symbols outside the pool get 0.0. The z-score of the expected return model is taken over every candidate before the pool is cut, so `mu` does not depend on `candidate_top_k`.

### Minimum trade

The solver rarely returns an exact 0: a symbol it does not want often comes back at 1e-10 to 1e-6, and every such residue would be an order, a holding, and, through the pool's held symbols, a candidate at every later bar. `min_trade` (long-only, 0 by default) skips every change smaller than itself: a candidate whose solved weight differs from its current one by less than `min_trade` keeps its current weight, so no order is placed for it.

| current | solved | `min_trade=1e-4` |
|---|---|---|
| 0 | 1e-9 | 0 (not bought) |
| 0.05 | 0.04995 | 0.05 (not traded) |
| 0.05 | 0.03 | 0.03 (sold) |
| 0.05 | 2e-9 | 0 (closed: a traded target below `min_trade` is 0) |

A skipped buy leaves its budget in cash. Skipped sells can leave the book above one; the candidates that do trade are then scaled down to absorb the excess, and when they cannot, those sells are made as solved. The bounded exposures can move by about the skipped trades' total.

### Solver

Each bar's problem is solved by the cvxpy solver `solver` names: `"CLARABEL"` by default, an interior-point solver whose constraints hold to about 1e-9. `"OSQP"`, a first-order solver and the one cvxpy 1.9 picks for these problems, holds them to about 1e-5; on simulated problems of 900 to 1,500 candidates Clarabel solved about twice as fast, at 3,000 the two are close. An interior-point solution leaves residues of 1e-10 to 1e-6 on nearly every symbol it does not want; `min_trade` (see [Minimum trade](#minimum-trade)) keeps them from becoming orders. The solver is named so a run reproduces when cvxpy changes its default or another machine has other solvers installed, and `None` is refused. A solver that is not installed (`cvxpy.installed_solvers()`) is refused when the optimizer is built.

### Name-count cap

`max_names` caps the name count of the book each bar decides: the symbols with a non-zero weight, locked positions included. The optimiser then picks the names together with their weights, under the risk, the turnover penalty, the cap and the exposure bounds at once, so the book is the best one of at most `max_names` names, not the top `max_names` by expected return. Long-short, long and short names count together. Each bar is solved exactly as a mixed-integer program: one boolean `z` per candidate, `|w| <= weight_cap * z` and `sum(z) + locked names <= max_names`.

```python
>>> capped = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.6, max_names=2, solver="MOSEK",
... ))
>>> capped.bind(specs)
>>> capped.construct(context).values.round(3)
array([0.6, 0.4, 0. , 0. ])

```

- **Solver.** A mixed-integer conic program needs a solver that solves one, named in `solver`: `"MOSEK"` (the `mosek` extra; it needs a licence at `~/mosek/mosek.lic`), or `GUROBI`, `CPLEX`, `XPRESS`, `COPT`, `SCIP` where installed. The default `"CLARABEL"` cannot, and a config with `max_names` and a solver that cannot is refused when the optimiser is built, naming the installed ones.
- **Gap.** `mip_gap` (MOSEK and GUROBI) stops the solve once the book's objective is proven within that fraction of the best book's; `None` keeps the solver's default (1e-4 for MOSEK). No time limit is set: a bar takes as long as the proof does, and a looser gap is the way to make it faster. A solve that ends without an optimal status holds the bar, recorded in `metrics.json`'s `portfolio_construction`.
- **Budget.** Long-only, `weight_cap * max_names` must be at least one, or the book cannot be fully invested; it is refused when the optimiser is built. Near one the weights are all near the cap (20 names at 5% must each be 5%), and the choice of names is all that is left. Long-short there is no such check: the gross exposure is a ceiling, so a small cap leaves part of the book uninvested.
- **Locked positions** take their slots first. A bar whose locked positions alone exceed `max_names`, or long-only fill it while the book has budget left, holds and is recorded. Long-short, locked names may fill the cap: the free names then hold nothing, which works when the locked net is zero.
- **Exact zeros.** A name the solve leaves out gets exactly 0 (the solver's integer tolerance would leave residues near `weight_cap` times it), and the capped-simplex cleanup (long-short, the clip-and-rescale) runs on the kept names only. A held name the solve leaves out is sold in full, whatever its size; `min_trade` applies to the kept names with its usual meaning.
- With `candidate_top_k` the program is solved on the pool, which is much faster. The approximate backtest caps the name count locally, in both directions (see [What it keeps and what it gives up](backtest.md#what-it-keeps-and-what-it-gives-up)).

### Mean-variance in the approximate backtest

The approximate backtest (see [Sweep with the approximate backtest](backtest.md#sweep-with-the-approximate-backtest)) reads the optimiser's config and decides a chunk of bars with its own mean-variance implementation (`quantlab/portfolio/predefined/approximate/mean_variance.py`, its `decide`); `MeanVarianceOptimizer` itself only decides one bar at a time. It builds every bar's covariance in low-rank form (`quantlab/portfolio/predefined/approximate/covariance.py`, a `LowRankCovariance` of `quantlab.portfolio.approximate`: `B F B' + diag(D)` per bar) scaled to the expected-return label's span, and computes `mu` as the per-bar rule does: a symbol tradable and covered by the estimate is selectable, and `mu = ic * sigma * z` on the selectable symbols with a prediction. It then maximises `w @ mu - risk_aversion / 2 * w @ Sigma @ w - turnover_penalty * |w - w_held|_1` under the budget and the cap on each bar's candidates, in compiled code (numba, `quantlab/portfolio/predefined/approximate/_support/mean_variance_kernels.py`, imported only by this path), only on the columns some bar of the chunk can select. Each bar is solved by accelerated proximal gradient (FISTA with adaptive restart) with backtracking: the step's curvature starts below a bound on the budget's tangent space, `{sum(dw) = 0}`, where the market factor, on which every symbol loads alike, mostly cancels, and is lowered after every accepted step and doubled when the sufficient-decrease test fails, never past the bound, so a bar whose binding caps leave little curvature takes long steps. The prox is the exact projection onto the capped simplex, with the turnover penalty as soft-thresholding toward the holdings, its shift found by safeguarded Newton steps. A bar has converged when its gradient-mapping residual is at most the backtest's `tolerance` (default 1e-5) times its starting gradient's largest entry, its objective then within about twice that of the optimum. With a turnover penalty, or `candidate_top_k` (which keeps the held names in its pool), the bars are solved forward in time, each once: its holdings are its book (the last decided bar's weights at their fill prices, marked at its close; `quantlab.execution.approximate`), the selectable ones joining the candidates (at `mu = 0` when unpredicted), and its solve starts from them; a held bar leaves the book as it was. Without either, the bars are independent and solved at once, in parallel over the bars except on macOS (`numba.set_num_threads` caps the threads), a held name without a prediction then not a candidate. On a problem of 3,000 candidates and 61 factors whose largest eigenvalue is about 4,300 times the median specific variance, a bar takes about 20 iterations from its caps and 4 from its drifted holdings, against 2,600 and 50 for the previous solver. A bar with too few candidates holds, as in a backtest run. On synthetic stores the weights are within 1e-5 (L1) of a backtest run solved to tight tolerances and the value within 2e-4; at CLARABEL's default tolerances a run's own solution can stop about 2e-3 from the optimum on a bar whose objective is nearly flat, so the two backtests can differ by that much there. With a turnover penalty a weight inside the no-trade band stays at its holding, so the holdings each backtest hands the rule enter its weights: with fees, a backtest run cuts a fill's last buys to the cash the fees leave (the cash cap, which the approximate backtest gives up), and the penalised weights then differ by up to about the cost times the turnover (7.8e-4 with 5 bp fees on the synthetic store). Both backtests count a name as held above `HELD_WEIGHT_TOLERANCE` (1e-8), so a residue the run's solver leaves on a name (about 1e-9, platform-dependent) puts it in neither pool.

Supported: both directions, any expected return model, any `turnover_penalty`, `candidate_top_k` and `volatility_label`, and `max_names`, with `LedoitWolfEstimator` or `FactorRiskStoreEstimator`. Long-short, the solve maximises over `sum(w) = 0`, `|w|_1 <= 1` and `|w| <= weight_cap`, the projection being the entrywise prox of the L1 terms (the gross ceiling's multiplier, and the turnover penalty's pull toward the holdings) with a shift meeting the zero sum; the ceiling's multiplier is found by Newton steps, the gross being piecewise linear in it. `candidate_top_k` keeps each bar's top `candidate_top_k` by `mu` (`|mu|` long-short) among its predicted and held candidates, ranked with a stable sort as per bar, and the held ones. `volatility_label` hands the covariance estimate the prediction over `sqrt(span)` as one-bar volatilities: a symbol without a finite positive one is not covered, and the covered ones keep the estimate's correlations at the predicted volatilities (`LowRankCovariance.with_volatility`); with `FactorRiskStoreEstimator` the optimiser refuses `volatility_label` when it is constructed. Every expected return model is the per-bar rule's own array code (`RawExpectedReturn` takes the prediction as `mu`, for example). `max_names` projects each step onto the books of at most `max_names` names, long and short counted together: the direction's prox first (the capped simplex, or the long-short set above), then, when it holds more names, the `max_names` names whose zeroing costs the prox most (`1/2 (w - u)^2` plus the turnover pull and, long-short, the gross ceiling's multiplier times `|w|`, at 0 against at the prox's weight) kept, on either side, and the prox run again on them; the steps have no momentum (acceleration oscillates on that non-convex set) and the first one is a gradient step from the start, so a start from cash ranks the names by the gradient rather than by column order. The result is a local optimum of the program `run()` solves exactly. Speed (#319): on a us3000-shaped synthetic store (10,000 symbols on the axis, 3,000 priced, 60 factors with eigenvalues up to about 4,500 times the specific variance) the paper-trading configuration (risk aversion 20, `weight_cap` 0.05, `turnover_penalty` 0.002, ic 0.0122) over 67 daily rebalances at 10 bp took 1.0 s against `run()`'s 532 s, the value within 5e-4, the annualised return within 0.25 percentage points and the weights within 2.4e-3 (L1, the cash cap's gap). Refused before any data is read, the message naming it: a tracking-error cap (`tracking_error`), `exposure_bounds`, a subclass of the optimiser or of a supported estimator (one overriding `reference_weights` or `risk_constraints`, say), any other covariance estimator. `solver` is not read, and `min_trade` is not modelled: configs that differ only in `min_trade` give the same approximate result.

The active-risk rule runs too, without a `tracking_error`. `DecisionInputs.panel` holds the values of the factors a rule declares (`DecisionPanel.factors`), and the approximate backtest reads the benchmark's weights there (its `benchmark_weight` output), as the per-bar rule's `reference_weights` reads each bar's. It then prices `(w - b)' Sigma (w - b)` at every bar, `b` on the bar's covered symbols: a benchmark symbol the estimate does not cover drops out, as in the per-bar rule, and no event is recorded. On a synthetic store its weights equal those of a tightly solved backtest run to 6e-7 (L1), and its value to 1.5e-4.

With `FactorRiskStoreEstimator`, `DecisionInputs.panel` carries the factor risk model's exposures at the bars and its estimate store's rows there (`DecisionPanel.risk_exposures`, `risk_estimates`), read as a backtest run reads them (ADR 0037): the approximate backtest reads each once over its decision bars (`DecisionInputs.panel_reads`, with the tradability and the declared factors), whatever its chunks. The approximate backtest builds every bar's `B F B' + diag(D)` with array operations, without asking `forecast` bar by bar: `B` from the model's `exposure_matrix` over all bars at once, `F` and `D` from the estimate rows, on the model's factor axis. Its coverage is a simplified version of the per-bar forecast's, so the two backtests can differ where the stores have gaps: a factor is kept at a bar when its variance is finite (a dropped factor has zero exposures and a zero row and column of `F`, and a symbol exposed to it is not covered there, as per bar), but a missing covariance between two kept factors is taken as 0, where the per-bar forecast drops factors until the block is finite; a symbol is covered when `exposure_matrix` covers it and it has a finite specific risk and finite exposures on the kept factors, as per bar. A bar the estimate store has no row at is refused, as the per-bar `estimate` refuses it. On the synthetic factor risk model of the tests (two styles, two industries) the weights equal a tightly solved backtest run's to 5e-8 (L1) on 14 symbols and 1.6e-6 on 300, and the value to 2.5e-4; 300 symbols over 67 daily rebalances take 0.25 s against the run's 4.0 s.

### Exposure bounds

`exposure_bounds` holds the book's exposure to a factor between two bounds: for each named output `x` of the `exposure_factors`, `lower <= sum_i w_i * x_i <= upper`, the locked positions' exposure included (a locked symbol without one makes the bar fail, since the book's exposure is then unknown). The rule declares the `exposure_factors` among its factors (`declared_inputs()`), so a backtest computes them over its window and hands their values at the bar in `context.factors`. A candidate without an exposure gets no weight, and a held one is closed, listed in the bar's `closed_without_exposure` event. A factor of the covariance estimator's factor risk model (one of its `factor_names`) can be bounded too, without declaring a factor: its exposures are the forecast's answer, `forecast.factor_exposure(name, symbols)`: the column of `B`, none for a symbol the forecast does not cover, and 0 when the factor has no covariance at the bar and so left the forecast. A dense or Ledoit-Wolf estimate has no risk factors. With a factor risk model as the covariance, `exposure_bounds={"style_beta": (-0.1, 0.1)}` holds the book's USE4 Beta exposure near the cap-weighted estimation universe's, which is 0, and an industry factor can be bounded the same way. A bounded name that is both an `exposure_factors` output and a factor of the risk model is refused. Bounds the candidates cannot reach make the bar infeasible, and the backtest holds it (see [In a backtest](#in-a-backtest)). The bounds hold to the solver's tolerance: about 1e-9 with `"CLARABEL"`, 1e-5 with `"OSQP"`.

The common use is a beta near 1, so that a long-only book keeps the benchmark's market risk and its excess comes from the selection rather than from a lower beta. `BenchmarkBeta` (see the [factor guide](factor.md#benchmark-beta)) gives each symbol's beta on a single-symbol benchmark; with `stocks` the symbols' price dataset and `vt` the benchmark's:

```python
>>> from quantlab.factor.config import BenchmarkBetaConfig
>>> from quantlab.factor.predefined.benchmark_beta import BenchmarkBeta
>>> beta = BenchmarkBeta(BenchmarkBetaConfig(dataset=stocks, benchmark=vt))
>>> bounded = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.4,
...     exposure_factors=(beta,), exposure_bounds={"beta": (0.95, 1.05)},
... ))
>>> bounded.bind(specs)
>>> bounded.declared_inputs().factors == (beta,)
True

```

On the context of [A first optimisation](#a-first-optimisation), whose book had a beta of 0.858 at these exposures, the bound lifts it to 0.95:

```python
>>> exposure = on_symbols([0.6, 0.8, 1.3, 1.5])
>>> context = dataclasses.replace(context, factors=xr.Dataset({"beta": exposure}))
>>> weights = bounded.construct(context)
>>> weights.values.round(3), float((weights * exposure).sum().round(3))
(array([0.4  , 0.248, 0.08 , 0.271]), 0.95)

```

## Covariance estimators

A covariance estimator (`CovarianceEstimator`) estimates the covariance of one-bar returns at one bar. Its `estimate(context, volatility=None)` returns a `ReturnCovariance` (`quantlab.risk.base`): the symbols it covers, their covariance and variances, `scaled`, `subset`, `factor_form()` (`None` for a dense estimate) and `factor_exposure(name, symbols)`. Three types satisfy it and stay separate (ADR 0028): the dense `CovarianceEstimate`, the low-rank `LowRankCovarianceEstimate` (`B B' + diag(D)`, without risk factors) and a factor risk model's `FactorRiskForecast`. A rule asks the estimate and never tells them apart. It is not trained and has no checkpoint. A forecast that feeds it, such as predicted volatility, comes from a model through the predictor. It returns only a covariance, so any rule that needs one can hold it, not only the mean-variance optimiser.

It is not a factor risk model. A factor risk model (`quantlab/risk/`, ADR 0024) is estimated ahead of any backtest as stores of one row per bar; `FactorRiskStoreEstimator` is the covariance estimator that reads them.

`LedoitWolfEstimator` reads the last `lookback_bars` one-bar returns and shrinks their sample covariance toward a scaled identity with the Ledoit-Wolf coefficient (`sklearn.covariance.ledoit_wolf`), so the estimate stays well conditioned when there are more symbols than bars. It then splits the covariance into correlations and volatilities, and replaces the volatilities with the given ones when `volatility` is passed. The estimate is a `LowRankCovarianceEstimate`: the shrunk covariance of a `lookback_bars` window is the scaled demeaned window times its transpose plus a multiple of the identity, so it never builds the `[n, n]` matrix (`covariance` builds it on request). Its `factor_form()` is `(B, I, D)` once there are at least `LOW_RANK_MIN_RATIO` (6) symbols per bar of the window, and the optimiser then prices risk in low-rank form; with fewer the dense risk term solves faster and `factor_form()` is `None`. With CLARABEL on a long-only problem and a 252-bar window that is 4.5 s against 5.6 s dense at 3,000 candidates and 14 s against 22 s at 5,000; the interior-point solve, not the covariance, is most of a bar's time either way.

A symbol is covered only when all of these hold:

- Every return in its window is finite.
- The returns are not all equal.
- Its staleness is at most `max_stale_bars` (default 5).

The other symbols are left out, and the optimiser never selects them. A held symbol that the covariance estimator does not cover is closed, and the row reports it as a `closed_without_risk` event:

```python
>>> short_history = window.copy()
>>> short_history[:5, 2] = np.nan  # CCC listed five bars into the window
>>> import dataclasses
>>> late = dataclasses.replace(
...     context,
...     current_weights=on_symbols([0.0, 0.0, 0.3, 0.0]),
...     returns=context.returns.copy(data=short_history),
... )
>>> weights = optimizer.construct(late)
>>> weights.values.round(3)
array([0.4, 0.4, 0. , 0.2])
>>> weights.attrs["events"]
{'closed_without_risk': ['CCC']}

```

### Factor form: reading a factor risk model

`FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model))` reads the stores of a factor risk model (`quantlab.risk.base.FactorRiskModel`, for example `Use4RiskModel`):

- It returns the model's forecast at the bar, `model.forecast(row, exposures)`, a `FactorRiskForecast` (`quantlab.risk.base`): the covariance in factor form `B F B' + diag(D)`, exposures `B`, factor covariance `F` and specific variances `D`, from the context's estimate row (`context.risk_estimate`) and risk exposures. It reads no store itself, so it can be called on a context built by hand. Its `factor_form()` makes the optimiser price risk as `|F^(1/2) B' w|^2 + w' diag(D) w`, without building the dense matrix.
- It declares the model (`declared_inputs().risk_model`), and no factor. The backtest takes the model's exposures from the model, `model.exposures(start, end)`, the source its stores, factor attribution and bias statistics read: from the exposures factor's store under `exposure_data_strategy="read"`, which must then cover the backtest, computed under `"cal"`. It puts their values at the bar in `context.risk_exposures`, and the estimate store's row at the bar in `context.risk_estimate`: read once over a panel's rebalance bars, once per bar for `context`, once over a preloaded window. A bar the store has no row for gets `None`, and the estimator refuses it.

Coverage is the model's rule, the one factor attribution and the bias statistics use: a symbol with every exposure and a specific risk at the bar, not exposed to a factor without a covariance there (such a factor leaves the forecast). It reads no return window, so it has no staleness filter, and a locked position is priced like any other. The model's stores must cover the backtest. [Factor risk model](developer-guide/risk-model.md) describes `Use4RiskModel` and compares it with Ledoit-Wolf in a mean-variance backtest.

## A full example

`examples/wrds_us_equity/sp500_xgb_mvo.py` is an enhanced S&P 500 index: it runs the whole pipeline on the point-in-time index members.

1. Alpha101 and Alpha158 factors.
2. A `Return` and a `Volatility` label, both over 5 bars.
3. A `ModelEnsemble` of two `XGBoostRegressor` heads, one per label.
4. The mean-variance optimiser over the day's members, with `volatility_label` set and a pool of 200 candidates, rebalancing every 5 bars against buy-and-hold SPY. A stock removed from the index is settled at its last close, like a delisting.

It needs the S&P 500 and SPY stores of `scripts/wrds/index.py --index sp500` and `scripts/wrds/etf.py --etf spy` (see the [examples README](../examples/wrds_us_equity/README.md)).

A run on the training server (one GPU; commit `211b324`, 2026-09-30; about 8 minutes once the stores exist) gave, over the out-of-sample window 2020-2024:

| Model | Test IC | Test rank IC |
| --- | --- | --- |
| Return model (`ret_5`) | -0.0005 | -0.0004 |
| Volatility model (`vol_5`) | 0.513 | 0.481 |

| Backtest | Strategy | SPY |
| --- | --- | --- |
| Total return | 41.6% | 96.8% |
| Sharpe ratio | 0.51 | |
| Maximum drawdown | 32.3% | |
| Annualised excess return | -6.4% | |
| Tracking error | 10.1% | |
| Information ratio | -0.75 | |
| Beta | 0.68 | |

No rebalance failed and no order was rejected. On 10 rebalances, 11 holdings in all were closed because the covariance estimator had no estimate for them (`closed_without_risk`).

Read these numbers as a working pipeline, not as a strategy. The return model has no out-of-sample skill: its test IC is zero. With no signal in `mu`, the risk term decides the weights, and the book becomes a low-volatility portfolio with a beta of 0.68, which trails the index in a rising market. The volatility model does carry information: its IC is the correlation of predicted with realised volatility, which measures how well it ranks risk, not alpha. The optimiser also constrains neither the tracking error nor the active weights against the index, so this is an index-member portfolio, not yet a controlled enhanced index.

## Writing a rule

Subclass `PortfolioConstructor`:

1. Set `config_cls` to a frozen dataclass of the rule's parameters.
2. Implement `construct`.
3. Override `declared_inputs` when the rule reads a return window, factor panels or a factor risk model's exposures: it returns an `InputDeclaration` (`lookback_bars`, `history_bars` when it needs more raw prices than `lookback_bars + 1`, `factors`, `risk_model`); a rule holding parts merges theirs with `InputDeclaration.merged`. Override `axis_factors` when the rule reads a declared factor on symbols it has no prediction for (a reference book's weights): a backtest decides only on the symbols with a prediction and those with a value of these outputs (`DecisionInputs.decision_symbols`); `ActiveRiskOptimizer` names its `benchmark_weight` and `BlackLittermanOptimizer` its `market_weight`. Override `bind` to check the label specs, and `ranking_label` when the rule ranks by a label other than the first one predicted (a run's attribution ranks its universe and score groups by it).

`MeanVarianceOptimizer` has three extension points for a rule built on it. `posterior(context, symbols, expected_return, estimate, candidates)` is called after `mu` is computed and before `candidate_top_k`: over the bar's selectable symbols (tradable, not locked, covered, with every bounded exposure) it receives `mu` (NaN without a prediction), the span-scaled covariance estimate and the candidates (a prediction, or held), and returns the three to optimise; a rule may replace `mu` and the estimate (which must still cover every candidate) and widen or narrow the candidates, a candidate left at NaN getting `mu = 0`. It returns them unchanged by default. The approximate backtest's mean-variance `decide` takes the matching hook, `posterior(rule, panel, problem)` on every bar's `PanelProblem`, ahead of its kernels; `None` by default. `reference_weights(context)` returns the book the risk term measures against, so the optimiser prices `(w - b)' Sigma (w - b)`, the active risk, instead of `w' Sigma w`; a benchmark symbol the covariance estimator does not cover is left out and reported in the `reference_without_risk` event. `risk_constraints(risk)` returns constraints on that risk term, a cvxpy expression on the expected-return label's span, such as `[risk <= cap**2]`. All three are unused by default, and the problem is then unchanged. `ActiveRiskOptimizer` (`quantlab.portfolio.predefined.active_risk`) is built on them: a benchmark-relative mean-variance rule whose benchmark weights come from a declared factor, with an optional tracking-error cap. Its config `ActiveRiskConfig` is `MeanVarianceConfig` plus `benchmark` (the factor), `benchmark_weight` (its output holding the weights, default `"benchmark_weight"`) and `tracking_error` (a cap on the forecast active volatility over the expected-return label's span, `None` for none). It maximises `w @ mu - risk_aversion / 2 * (w - b)' Sigma (w - b) - turnover_penalty * |w - w_current|_1` under the optimiser's constraints, and with `tracking_error` also `(w - b)' Sigma (w - b) <= tracking_error**2`. `MembershipCapWeight` (see [factor.md](factor.md)) gives an index's capitalization weights as such a benchmark.

```python
from quantlab.factor.config import MembershipCapWeightConfig
from quantlab.factor.predefined.membership_cap_weight import MembershipCapWeight
from quantlab.portfolio.config import (
    ActiveRiskConfig, FactorRiskStoreEstimatorConfig, GrinoldExpectedReturnConfig,
)
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.predefined.active_risk import ActiveRiskOptimizer
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator

rule = ActiveRiskOptimizer(ActiveRiskConfig(
    expected_return_label="ret_1",
    covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4)),
    expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.0122)), risk_aversion=20.0, turnover_penalty=0.002, weight_cap=0.05,
    benchmark=MembershipCapWeight(MembershipCapWeightConfig(dataset=daily, membership=us3000)),
    tracking_error=0.06 / 252**0.5,
))
```

`BlackLittermanOptimizer` (`quantlab.portfolio.predefined.black_litterman`, ADR 0047) is a mean-variance rule on a Black-Litterman posterior, built on `posterior`. Its config `BlackLittermanConfig` is `MeanVarianceConfig` plus `market` (a weight factor, declared like `benchmark`: usually `MembershipCapWeight` over the roster's membership), `market_weight` (its output, default `"benchmark_weight"`) and `tau` (required, positive: the prior's covariance is `tau` times the risk model's; one over the length of the covariance's estimation window, or 0.025 to 0.05 by convention). In the risk model's factor space, returns are `r = B f + e` with prior `f ~ N(pi_f, tau F)` and `e ~ N(pi_e, tau diag(D))`, `Sigma = B F B' + diag(D)` the bar's forecast over the expected-return label's span. The symbol views are the `expected_return` model's `mu` on the predicted candidates: each a total return over the span (`views_relative_to_prior=False`, the default) or an alpha added to the symbol's equilibrium return `pi` (`views_relative_to_prior=True`; `pi` is priced against the posterior covariance, so it moves slightly with `omega` and with which names have views). An IC-scaled model (`GrinoldExpectedReturn`, `ZScoreExpectedReturn`) gives a cross-sectionally centred alpha, so as a total return it pulls every viewed name towards zero; the rule warns when it is constructed that way. `omega` sets each view's uncertainty `omega_j`: `"he_litterman"` (default, `tau Sigma_jj`), `"idzorek"` (`tau (1 - C)/C Sigma_jj`, Walters' closed form of Idzorek's confidence `C = confidence`, required in `(0, 1]`: 1 puts the posterior on the views, near 0 leaves it at the prior) or `"specific"` (`tau D_j`, the specific variance); `confidence` is refused with another `omega`. Do not scale `Omega` by the IC when the views already come from an IC-scaled model: Grinold's `ic * sigma * z` is itself the one-name posterior of an Idzorek view with `C = IC^2` on the score in return units (`sigma z / IC`), so shrinking it again counts the IC twice; with an IC-scaled model, use alpha views and a confidence near 1, and keep `confidence = IC^2` (Grinold's shrinkage) for views not yet scaled by the IC. One Bayes update gives the posterior mean and the posterior covariance `Sigma + M`, `M` the uncertainty of the posterior mean; it is solved on the factor dimension (one `K x K` solve, no symbol-by-symbol inverse), and the posterior covariance is diagonal plus rank `2K`, the factor form the risk term reads. The rule optimises on both, with every mean-variance field. The equilibrium return is `pi = risk_aversion * (Sigma + M) * w`, `w` the market's weights on the covered symbols renormalised to sum to one: the return that makes the market optimal under the covariance the rule optimises on, so with no views (`M = tau Sigma`) the decided book is the market, and views equal to `pi` leave it there (where the constraints allow: `max_names`, `candidate_top_k`, a cap below a market weight or a turnover penalty from a book other than the market move it). As `Omega` tends to 0 with a view on every covered symbol the market holds, `pi` tends to `risk_aversion * Sigma * w`; a market symbol without a view (locked, not tradable or without a prediction) keeps its share of `M`. Its split is `pi_f = risk_aversion * F x` with `x = B' w` (`market_exposures`; a market wider than the roster overrides it together with `market_weights`) and `pi_e = pi - B pi_f`. Every selectable symbol is a candidate: one without a view gets its equilibrium return plus what views on correlated symbols move it by, and `candidate_top_k` ranks the posterior. The rule refuses, when it is constructed: no views at all (no `expected_return` and no user views), a covariance estimator other than `FactorRiskStoreEstimator` (the prior and the posterior price risk from one forecast), a missing `market` or `market_weight`, a `tau` that is not positive, an unknown `omega`, a `confidence` missing or outside `(0, 1]` with `"idzorek"` or set with another `omega`, and a subclass with a `reference_weights` (an active-risk reference prices `lambda Sigma b`, the equilibrium again). The approximate backtest runs it (see [Black-Litterman in the approximate backtest](backtest.md#black-litterman-in-the-approximate-backtest)).

```python
from quantlab.factor.config import MembershipCapWeightConfig
from quantlab.factor.predefined.membership_cap_weight import MembershipCapWeight
from quantlab.portfolio.config import (
    BlackLittermanConfig, FactorRiskStoreEstimatorConfig, RawExpectedReturnConfig,
)
from quantlab.portfolio.predefined.black_litterman import BlackLittermanOptimizer
from quantlab.portfolio.predefined.expected_return import RawExpectedReturn
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator

rule = BlackLittermanOptimizer(BlackLittermanConfig(
    expected_return_label="ret_1",
    covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4)),
    expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
    risk_aversion=20.0, weight_cap=0.05, tau=0.05, omega="idzorek", confidence=0.5,
    market=MembershipCapWeight(MembershipCapWeightConfig(dataset=daily, membership=us3000)),
))
```

The user's views (each a `ReturnView`) are `factor_views` (each a `FactorReturnView`, a view on one risk-model factor's return, `f_k`, by the factor's name in the risk model, or a `RelativeFactorReturnView`, a view on one factor's return less another's, `f_k - f_l`: "value earns 2% a year more than momentum", `outperformer` and `underperformer`) and `market_views` (each a `MarketReturnView`, a view on the market portfolio's return, `x' f` with `x = B' w`, the market's specific return neglected). A view is an annualised return, converted linearly to the span (times `span / 252`): a constant (`annual_return`) or a series read point-in-time from a zarr store (`store`, `variable`, a variable on `timestamp`), whose value at a bar is the last finite one dated at or before it; each read is logged to the run's data record. Its uncertainty is Idzorek's `confidence` `C` (the view's variance `tau (1 - C)/C p' F p`, `p` its row on the factor returns; 1 puts the posterior on the view) or a standard deviation `std`, annualised like the return and converted the same way (`(std * span / 252)^2`). A view not known yet at a bar, or on a factor the bar's forecast does not have, is left out at that bar. All the views are one Bayes update (the user's views update the factor prior, in a form that takes a zero variance, and the symbol views update that); the posterior covariance keeps its rank `2K` plus diagonal form. `expected_return` is optional: without it the rule reads no prediction, `expected_return_label` stays unset, and `span` (in bars) gives the span; with it, a `span` set too must equal the label's. The rule also refuses, when it is constructed, `expected_return` without `expected_return_label` or the reverse, no `span` without `expected_return`, a `factor_views` entry that is not a `FactorReturnView` or `RelativeFactorReturnView` (or a `market_views` one that is not a `MarketReturnView`) and a factor view naming a factor the risk model lacks; `bind` refuses a `span` other than the label's. A view itself refuses a relative view whose two factors are one, not exactly one of `annual_return` and `store`, `variable` without `store` (or the reverse), not exactly one of `confidence` and `std`, a `confidence` outside `(0, 1]` and a `std` that is not positive.

```python
from quantlab.portfolio.config import (
    FactorReturnViewConfig, MarketReturnViewConfig, RelativeFactorReturnViewConfig,
)
from quantlab.portfolio.predefined.black_litterman import (
    FactorReturnView, MarketReturnView, RelativeFactorReturnView,
)

rule = BlackLittermanOptimizer(BlackLittermanConfig(
    covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4)),
    risk_aversion=20.0, weight_cap=0.05, tau=0.05, span=5,
    market=MembershipCapWeight(MembershipCapWeightConfig(dataset=daily, membership=us3000)),
    # Book-to-price earns 2% a year, with a 50% confidence.
    factor_views=(
        FactorReturnView(FactorReturnViewConfig(factor="style_book_to_price", annual_return=0.02, confidence=0.5)),
        # Momentum earns 1% a year more than size, give or take 2% a year.
        RelativeFactorReturnView(RelativeFactorReturnViewConfig(
            outperformer="style_momentum", underperformer="style_size", annual_return=0.01, std=0.02,
        )),
    ),
    # The market earns what the series says at each bar, give or take 4% a year.
    market_views=(MarketReturnView(MarketReturnViewConfig(store="views/market.zarr", variable="annual_return", std=0.04)),),
))
```

On us3000 (quantlab-experiment `2026-10-17-us3000-black-litterman`, #344), with the paper-trading ensemble's predictions as the only views, the market the cap-weighted us3000, `risk_aversion` 20, a 5% cap, a 0.002 turnover penalty, no exposure bounds and SPY's total return as the benchmark, Black-Litterman does not beat `ActiveRiskOptimizer` with `GrinoldExpectedReturn(ic=0.0122)` on the same settings. Grinold alpha views with `omega="idzorek"` and `confidence=1` give the active-risk book (its objective is a total-return mean-variance on `mu + risk_aversion * Sigma * b`); `"he_litterman"` equals `"idzorek"` at `confidence=0.5`; `tau` from 0.01 to 0.1 moves the excess IR by less than 0.01; total-return views from an IC-scaled model lose 3 to 6 points a year. The best configuration on the selection folds, alpha views (`views_relative_to_prior=True`) with `"he_litterman"` and `tau=0.025`, shrinks the alphas and trades 184% a year against 435%: in a backtest run its selection excess IR is 0.20 against -0.05 (+1.4 points a year, t 0.82), and on the holdout -0.26 against 0.54 (-4.5 points, t -1.79). With one model's scores as views, use `ActiveRiskOptimizer`; use Black-Litterman to blend views of several kinds with their confidences, with alpha views, `"he_litterman"` and `tau=0.025`.

Do not override `decide`: it is the one decision path a backtest and an executor share. A rule holds no assembly code; `DecisionInputs` builds its contexts.

The approximate backtest (see [Sweep with the approximate backtest](backtest.md#sweep-with-the-approximate-backtest)) runs the shipped rules it has its own implementations of, chosen by their exact class: a new rule, or a subclass of a shipped one, runs in backtest runs and is refused there. To sweep it, add its implementation to `quantlab/portfolio/predefined/approximate` (a module with a `decide(rule, panel, prices, stopping)` function) and its field table to `quantlab/portfolio/approximate.py`.

`get_config` and `from_config` serialise the rule as its config's fields plus the class's import path. A field holding another component, such as a covariance estimator, is nested, so no extra code is needed to rebuild a run.

A new covariance estimator subclasses `CovarianceEstimator` in the same way and implements `estimate`; the approximate backtest estimates only with `LedoitWolfEstimator` and `FactorRiskStoreEstimator` (`quantlab/portfolio/predefined/approximate/covariance.py`). See [Extending quantlab](developer-guide/extending.md).

## See also

- [Backtesting](backtest.md): the `constructor` field, execution and the run directory.
- [Models](model.md): `label_scales` and ensembles that combine a return model and a volatility model.
- [Factors](factor.md): the `Return` and `Volatility` labels.
