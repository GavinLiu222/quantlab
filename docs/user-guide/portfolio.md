# Portfolio construction

Portfolio construction sits between a model and a backtest. On every rebalance bar a rule
takes the model's predictions for that bar and the weights currently held, and returns the
weights to hold after the bar; the backtest then trades them. A backtest holds its rule in the
`constructor` field of its config (see [Backtesting](backtesting.md)).

Every rule derives from {py:class}`~quantlab.portfolio.base.PortfolioConstructor`. The
shipped rules are:

- {py:class}`~quantlab.portfolio.predefined.top_n.TopNConstructor`: equal weights on the
  `top_n` best scores, long-only or long-short.
- {py:class}`~quantlab.portfolio.predefined.topk_dropout.TopkDropoutConstructor`: qlib's
  TopkDropout, which holds `topk` names and replaces at most `n_drop` of them per rebalance.
- {py:class}`~quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer`: Markowitz
  weights with a turnover penalty, solved with cvxpy, pricing risk with a covariance
  estimator.
- {py:class}`~quantlab.portfolio.predefined.active_risk.ActiveRiskOptimizer`: mean-variance
  against a benchmark book, with an optional tracking-error cap.
- {py:class}`~quantlab.portfolio.predefined.black_litterman.BlackLittermanOptimizer`:
  mean-variance on a Black-Litterman posterior.

The sessions on this page build each bar's input by hand, so they need no store, model or
GPU. Run them from the repository root with `uv run python`.

## One bar at a time

A rule implements one method, `construct(context)`. The
{py:class}`~quantlab.portfolio.base.PortfolioContext` it receives describes one bar and
holds nothing from a later bar: the bar's `timestamp`, every label's `predictions` on
`symbol`, which symbols are `tradable` (by default, those with a real fill price at the bar),
the `current_weights`, and, when the rule asks for them, a window of past `returns`, the
values of declared `factors` and a factor risk model's exposures and estimate row.

A rule returns one weight per symbol:

- The weights are finite, 0.0 where nothing is held, with a gross exposure of at most one.
- An all-NaN row means "hold the current position".
- A symbol that is held but not tradable is a *locked position* (`context.locked`). It keeps
  its current weight. A symbol neither tradable nor held gets 0.0.

A rule keeps no state between bars. Because it sees one bar only, it cannot look ahead, and
the same rule can be called from the bar handler of an event-driven engine or a live account.

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

`DDD` is halted while held, so it keeps its 0.25. The two best tradable scores share the
remaining 0.75. Long-short, the `top_n` best get `+0.5 / top_n` each and the `top_n` worst
`-0.5 / top_n` each.

Equal scores are ranked by symbol order. When a book's cut falls inside a group of equal
scores, symbol order rather than the model decides which are held, and the row reports the
tied symbols left out as a `tie_at_cutoff` event. A model that predicts only a handful of
distinct values triggers it on most rebalances.

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

## Rules in a backtest

A rule only decides. Everything it may read at a bar except the holdings, its *decision
inputs*, is assembled by {py:class}`~quantlab.portfolio.decision_inputs.DecisionInputs`, for
the backtest and for any other executor alike:

- The rule says what it reads through `declared_inputs()`, an
  {py:class}`~quantlab.portfolio.base.InputDeclaration`: how many bars of returns
  (`lookback_bars`) and raw prices (`history_bars`), which factors, which factor risk model.
  Each bar reads only the last `history_bars` prices up to and including it, so a decision
  does not depend on where the price history starts.
- The current weights are the holdings the earlier rebalances really left, replayed under the
  same execution rules as the simulation (rejected orders, delisting settlements, sizing,
  fees and slippage).
- The rebalance bars are every `rebalance_periods`-th bar from the first bar of the
  prediction panel. The last bar never rebalances, since an order decided there has no next
  bar to fill on.
- A halt shows in the return window as zero returns, then the whole move on the day trading
  resumes.

When the backtester is built, before any data is read, it calls the rule's `bind(labels)`
with one {py:class}`~quantlab.runs.prediction_panel.LabelSpec` (`name`, `scale`, `delay`,
`span`) per prediction variable. The specs are all a rule knows of the model; this is where a
rule checks the labels it needs, so a misconfigured rule fails at once.

A bar the rule cannot decide (an infeasible optimisation, a solver failure) raises
`PortfolioConstructionError`. The backtest holds the current position on that bar, logs a
warning and lists the bar in the `portfolio_construction` block of the run's metrics, beside
every event the rule reported (`tie_at_cutoff`, `closed_without_risk`, ...).

### Decisions outside a backtest

A run with a model keeps the predictions its rule read, as a
{py:class}`~quantlab.runs.prediction_panel.PredictionPanel`.
{py:meth}`DecisionInputs.from_run <quantlab.portfolio.decision_inputs.DecisionInputs.from_run>`
rebuilds the run's bound rule, price dataset, execution settings and schedule from the run
directory alone, without importing the model, factor, label or backtest layers. An
event-driven backtest can therefore replay a run's decisions. With `result` the first run of
[Backtesting](backtesting.md#a-first-run), a top-2 rule rebalancing every five bars:

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

An executor that keeps its own book, such as a live account, decides one bar at a time:
`rebalances(t)` says whether `t` is a rebalance bar, `context(t, predictions,
current_weights)` builds the bar's context from the dataset (so the executor keeps at most
`history_bars` bars of prices), and the rule's `decide(context)` runs `construct`, checks the
row against the weight contract and returns a {py:class}`~quantlab.portfolio.base.Decision`.
For a replay over many bars, `preloaded(start, end)` reads the window's inputs once. Here
`inputs` is built directly over three bars of prices on which `DDD` has no price at the
decision bar:

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

`TopkDropoutConstructor` follows qlib's `TopkDropoutStrategy` with `method_sell="bottom"`,
`method_buy="top"` and `only_tradable=True`. It caps turnover at about `n_drop / topk` of the
book per rebalance, where `TopNConstructor` rebuilds the book every time. For a book holding
`last`:

1. The candidates are the `n_drop + topk - len(last)` best-scored tradable symbols with a
   finite score that are not held.
2. The holdings among the `n_drop` worst of holdings and candidates ranked together are sold,
   so a holding is sold only for a better-scored symbol. A locked position is never sold.
3. The first `len(sell) + topk - len(last)` candidates are bought, so an empty book fills up
   to `topk`.

Kept holdings stay at their current weight; the buys split equally what they leave of the
budget. The short side runs the same rule on the negated scores. qlib's `hold_thresh`, random
methods and price-limit rules are not reproduced.

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

On each rebalance bar `MeanVarianceOptimizer` solves

```text
maximise    w'mu - risk_aversion / 2 * w'Sigma w - turnover_penalty * |w - w_current|_1
```

where `mu` is each symbol's expected return, `Sigma` their covariance and `w_current` the
weights held. The constraints depend on `direction`:

| `direction` | Constraints |
| --- | --- |
| `"long_only"` | `sum(w) = 1`, `0 <= w <= weight_cap`: fully invested. |
| `"long_short"` | `sum(w) = 0`, `sum(abs(w)) <= 1`, `abs(w) <= weight_cap`: dollar-neutral. |

Long-short, a gross exposure of one is a ceiling, not a target: when the expected returns do
not pay for the risk and the trading, part of the book stays uninvested. The turnover penalty
prices the change from the weights actually held. A locked position keeps its weight and
counts in the risk term.

The candidates are the symbols that are tradable, not locked, covered by the covariance
estimator, and have a finite prediction or are held. A held candidate without a prediction
gets `mu = 0`, so its turnover cost decides whether it is closed.

### A first optimisation

The optimiser needs the span and scale of the labels it reads, which a backtest hands it
through `bind`. Here the specs describe a 5-bar return `ret_5` and a 5-bar volatility
`vol_5`, both in the labels' own units:

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

The book is fully invested and within the cap; with so small an expected return it leans
toward the low-volatility `AAA` and `BBB`. `optimizer.problem_inputs(context)` returns the
candidates with their `mu`, `Sigma` and current weights, to see why the optimiser chose what
it did.

### Expected return models

A model's prediction is usually a score, not a return: it ranks symbols, but its scale
depends on the model and its training target. The optimiser's `expected_return` field holds
an {py:class}`~quantlab.portfolio.base.ExpectedReturnModel` that turns the prediction of
`expected_return_label` into `mu`:

- {py:class}`~quantlab.portfolio.predefined.expected_return.GrinoldExpectedReturn`:
  `mu = ic * sigma * z`, with `z` the prediction's cross-sectional z-score over the
  candidates, `sigma` each symbol's volatility over the span and `ic` the model's information
  coefficient (for example the mean IC of a walk-forward cross-validation). Any model's output
  can feed it, and `risk_aversion` keeps its meaning from one model to the next. Riskier names
  get a larger `mu`.
- {py:class}`~quantlab.portfolio.predefined.expected_return.ZScoreExpectedReturn`:
  `mu = ic * median(sigma) * z`, Grinold's scale without the tilt toward volatile names.
- {py:class}`~quantlab.portfolio.predefined.expected_return.UnscaledZScoreExpectedReturn`:
  `mu = z`, the z-score itself with no volatility read, so `mu` is in the same units on every
  bar and `risk_aversion` and `turnover_penalty` weigh against it in z units.
- {py:class}`~quantlab.portfolio.predefined.expected_return.RawExpectedReturn`: the
  prediction itself, for a model that predicts returns in their own units.
- {py:class}`~quantlab.portfolio.predefined.expected_return.PredictionExpectedReturn`: the
  prediction as it is, whatever its units; `risk_aversion` and `turnover_penalty` then weigh
  against `mu` in those units.

Every predictor reports per label whether its prediction is in the label's own units
(`label_scales`: `"raw"` or `"standardized"`), and the label spec carries it as `scale`. An
ensemble that averages z-scored members reports `"standardized"`. `bind` refuses
`RawExpectedReturn` on such a label:

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

`mu`, `sigma` and `Sigma` are all expressed over the span of `expected_return_label`, read
from its spec (5 bars for `ret_5`); a label without a span is refused. A covariance estimator
estimates one-bar returns, and the optimiser multiplies by the span, since variance grows
linearly with time:

```python
>>> inputs = optimizer.problem_inputs(context)
>>> one_bar = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)).estimate(context)
>>> bool(np.allclose(inputs.covariance, 5 * one_bar.covariance))
True
```

Rebalancing every span (`rebalance_periods` equal to the span) makes the horizon the
optimiser plans over the one the portfolio is held for.

### Volatility from a model

Volatility moves fast and is fairly predictable; correlations move slowly and are better
estimated from history. With `volatility_label` set, the covariance becomes the predicted
volatilities around the estimator's historical correlations (`Sigma = D C D`), and Grinold's
`sigma` becomes the prediction. The volatility label must have the expected-return label's
span and a `"raw"` scale.

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

One predictor supplies both labels; a `ModelEnsemble` of a return model and a volatility
model does this (see [Models](models.md)). `FactorRiskStoreEstimator` keeps its own
forecast, so it refuses `volatility_label`.

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

Thousands of symbols are slow to optimise at every rebalance. `candidate_top_k` restricts
each bar to the `candidate_top_k` largest `mu` (largest `abs(mu)` long-short) plus every
symbol currently held, so a holding that fell out of the top can still be closed at its
turnover cost. The z-score is taken before the pool is cut, so `mu` does not depend on
`candidate_top_k`.

### Minimum trade and the solver

The solver rarely returns an exact 0: a symbol it does not want often comes back at 1e-10 to
1e-6, and each residue would become an order and a holding. `min_trade` (long-only) skips
every change smaller than itself:

| current | solved | `min_trade=1e-4` |
|---|---|---|
| 0 | 1e-9 | 0 (not bought) |
| 0.05 | 0.04995 | 0.05 (not traded) |
| 0.05 | 0.03 | 0.03 (sold) |
| 0.05 | 2e-9 | 0 (closed: a traded target below `min_trade` is 0) |

`solver` names the cvxpy solver and is part of the recipe, so a run reproduces when cvxpy
changes its default. The default `"CLARABEL"`, an interior-point solver, holds the
constraints to about 1e-9; `"OSQP"` to about 1e-5.

### Name-count cap

`max_names` caps the number of names with a non-zero weight, locked positions included and
long and short names counted together. The optimiser then chooses the names together with
their weights, so the book is the best one of at most `max_names` names, not the top
`max_names` by expected return. Each bar is solved exactly as a mixed-integer program, which
needs a solver that handles one: `"MOSEK"` (the `mosek` extra, with a licence at
`~/mosek/mosek.lic`), or `GUROBI`, `CPLEX`, `XPRESS`, `COPT` or `SCIP` where installed.

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

Long-only, `weight_cap * max_names` must be at least one, or the book cannot be fully
invested. `mip_gap` stops a solve once the objective is proven within that fraction of the
optimum; no time limit is set, so a looser gap is how to make it faster. Exact solves are
slow: on us3000 with 20 names at 5% and the top 1,000 candidates, a bar took a median of 6 to
9 seconds and a ten-year daily window over 13 hours. The approximate backtest solves the cap
locally and much faster (see [Backtesting](backtesting.md#the-approximate-backtest)).

### Exposure bounds

`exposure_bounds` holds the book's exposure to a factor between two bounds,
`lower <= sum_i w_i * x_i <= upper`, locked positions included. The bounded factor is either
an output of one of the rule's `exposure_factors`, which the backtest computes over its window
and hands over in `context.factors`, or a factor of the covariance estimator's factor risk
model, such as `{"style_beta": (-0.1, 0.1)}`. A candidate without an exposure gets no weight,
and a held one is closed (the `closed_without_exposure` event). Bounds the candidates cannot
reach make the bar infeasible, and the backtest holds it.

A common use is a beta near 1, so that a long-only book keeps the benchmark's market risk and
its excess comes from selection. {py:class}`~quantlab.factor.predefined.benchmark_beta.BenchmarkBeta`
gives each symbol's beta on a single-symbol benchmark; with `stocks` the symbols' price
dataset and `vt` the benchmark's:

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

On the context of [A first optimisation](#a-first-optimisation), whose book had a beta of
0.858 at these exposures, the bound lifts it to 0.95:

```python
>>> exposure = on_symbols([0.6, 0.8, 1.3, 1.5])
>>> context = dataclasses.replace(context, factors=xr.Dataset({"beta": exposure}))
>>> weights = bounded.construct(context)
>>> weights.values.round(3), float((weights * exposure).sum().round(3))
(array([0.4  , 0.248, 0.08 , 0.271]), 0.95)
```

### Active risk

`ActiveRiskOptimizer` measures risk against a benchmark book `b`: it maximises
`w @ mu - risk_aversion / 2 * (w - b)' Sigma (w - b) - turnover_penalty * |w - w_current|_1`,
with every mean-variance constraint and, with `tracking_error`, also
`(w - b)' Sigma (w - b) <= tracking_error**2` over the label's span. The benchmark weights
come from a declared factor; {py:class}`~quantlab.factor.predefined.membership_cap_weight.MembershipCapWeight`
gives an index's capitalization weights. With `use4` a factor risk model, `daily` the price
dataset and `us3000` the membership:

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

A benchmark symbol the covariance estimator does not cover is left out and reported as the
`reference_without_risk` event.

### Black-Litterman

`BlackLittermanOptimizer` optimises on a Black-Litterman posterior in the factor space of a
factor risk model (it requires `FactorRiskStoreEstimator`). Returns are `r = B f + e`; the
prior's covariance is `tau` times the risk model's, and its mean is the equilibrium return
`pi`, the return that makes the `market` portfolio (a weight factor, usually
`MembershipCapWeight` over the roster) optimal. With no views the decided book is the market.

Views come from two places:

- **Symbol views**: the `expected_return` model's `mu` on the predicted candidates, read as
  total returns over the span or, with `views_relative_to_prior=True`, as alphas added to
  `pi`. A z-score model (`GrinoldExpectedReturn`, `ZScoreExpectedReturn`,
  `UnscaledZScoreExpectedReturn`) gives centred alphas, so use it with alpha views. `omega` sets each view's uncertainty: `"he_litterman"`
  (the default, `tau Sigma_jj`), `"idzorek"` with a `confidence` in `(0, 1]`, or
  `"specific"` (`tau D_j`).
- **User views** ({py:class}`~quantlab.portfolio.predefined.black_litterman.ReturnView`):
  `factor_views` on one risk factor's return (`FactorReturnView`) or one factor's return
  less another's (`RelativeFactorReturnView`), and `market_views` on the market's return
  (`MarketReturnView`). Each is an annualised return, a constant or a series read
  point-in-time from a Zarr store, with a `confidence` or a `std`.

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

Without `expected_return` the rule reads no prediction and takes its horizon from `span`:

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

```{note}
Grinold's `ic * sigma * z` is already the one-name posterior of an Idzorek view with
`C = IC^2`, so do not shrink IC-scaled views by the IC again. On us3000, with one model's
scores as the only views, Black-Litterman did not beat `ActiveRiskOptimizer` with
`GrinoldExpectedReturn` on the same settings. Use `ActiveRiskOptimizer` for one model's
scores, and Black-Litterman to blend views of several kinds with their confidences (alpha
views, `"he_litterman"`, `tau=0.025`).
```

## Covariance estimators

A covariance estimator ({py:class}`~quantlab.portfolio.base.CovarianceEstimator`) estimates
the covariance of one-bar returns at one bar. Its `estimate(context, volatility=None)`
returns a {py:class}`~quantlab.risk.base.ReturnCovariance`: the symbols it covers and their
covariance, in dense, low-rank or factor form. A rule asks the estimate and never tells the
forms apart. An estimator is not trained and has no checkpoint, and any rule that needs a
covariance can hold one.

{py:class}`~quantlab.portfolio.predefined.ledoit_wolf.LedoitWolfEstimator` reads the last
`lookback_bars` one-bar returns and shrinks their sample covariance toward a scaled identity
with the Ledoit-Wolf coefficient, so the estimate stays well conditioned with more symbols
than bars. It keeps the estimate in low-rank form and never builds the full matrix unless
asked. A symbol is covered only when every return in its window is finite, the returns are
not all equal and its staleness is at most `max_stale_bars` (default 5). A held symbol the
estimator does not cover is closed and reported as `closed_without_risk`:

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

{py:class}`~quantlab.portfolio.predefined.factor_risk.FactorRiskStoreEstimator` reads the
stores of a factor risk model ({py:class}`~quantlab.risk.base.FactorRiskModel`, for example
{py:class}`~quantlab.risk.predefined.use4.Use4RiskModel`), which is estimated ahead of any
backtest as one row per bar. It returns the model's forecast at the bar, `B F B' + diag(D)`,
which the optimiser prices as `|F^(1/2) B' w|^2 + w' diag(D) w` without building the dense
matrix. It declares the model, and the backtest hands it each bar's exposures and estimate
row; the stores must cover the backtest. Coverage is the risk model's: a symbol with every
exposure and a specific risk at the bar. [Factor risk model](../dev/risk-model.md)
describes `Use4RiskModel` and compares it with Ledoit-Wolf in a mean-variance backtest.

## Rules in the approximate backtest

The [approximate backtest](backtesting.md#the-approximate-backtest) decides every rebalance of
a window at once with its own implementation of each shipped rule, chosen by the rule's exact
class: `TopNConstructor`, `TopkDropoutConstructor`, `MeanVarianceOptimizer` (both directions,
any expected return model, turnover penalty, `candidate_top_k`, `volatility_label` and
`max_names`, over `LedoitWolfEstimator` or `FactorRiskStoreEstimator`),
`ActiveRiskOptimizer` without a tracking-error cap, and `BlackLittermanOptimizer` with every
view type. It refuses, before reading any data, `exposure_bounds`, a tracking-error cap, any
other estimator, a custom rule and any subclass of a shipped one. It ignores `solver`,
`mip_gap` and `min_trade`.

Its mean-variance solver is accelerated proximal gradient on a low-rank covariance,
compiled with numba. On synthetic stores its weights are within about 1e-5 (L1) of a tightly
solved backtest run and its value within about 2e-4. With `max_names` it finds a local
optimum rather than the exact one: on us3000 its books shared 83% of the exact run's names
on average, at the same volatility. A us3000-shaped problem of 67 daily rebalances took 1.0 s
against a backtest run's 532 s.

## Extending

To write a rule or a covariance estimator of your own, including rules built on the
mean-variance extension points, see [Extending quantlab](../dev/extending.md).
