"""``DecisionInputs`` assembles a rule's decision inputs, for a panel and for one bar (#127).

What is locked here, and what turns it red (an in-memory price dataset with a
halt, a late listing and a delisting; no vectorbt):

- On every rebalance bar, the context ``weights()`` built equals ``context()``
  at that bar given the replayed holdings, value for value (top-n,
  mean-variance with Ledoit-Wolf, a rule declaring a factor, one reading a
  factor risk model's exposures and one reading a market portfolio's rows), and
  ``decide`` on it returns the panel's row.
- ``context()`` puts a held symbol without a prediction on the bar's symbols,
  locked where it cannot trade.
- ``rebalances(t)`` reads the dataset's calendar once over a replay, and is
  the schedule ``weights()`` follows: every
  ``rebalance_periods`` bars from the anchor, never the last bar or ``end``,
  counted on the dataset's calendar when the panel starts after the anchor.
- ``weights()`` warns once when the dataset holds fewer bars than the first
  window needs; ``context()`` does not.
- One module assembles: no ``PortfolioContext`` is built outside it, the
  backtester, the backtest base and the API reference none of the assembly
  names (tradability, schedule, warm-up, factors, the replay book), and a
  rule has no ``build_context`` or ``construct_panel``.
- Refused: a panel before the anchor, a bar off the calendar, non-finite
  holdings, a panel handed as one bar's predictions, and factor panels
  lacking a declared name.
"""

import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.portfolio.base import InputDeclaration, PortfolioContext
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.dataset.memory import FrameDataset
from quantlab.portfolio.decision_inputs import DecisionInputs
from tests.decision_fixtures import schedule
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor

SYMBOLS = [f"S{i}" for i in range(8)]
WARMUP = 30
BARS = 60
PERIODS = 3
SPECS = [LabelSpec(name="ret_5", scale="raw", delay=1, span=5)]
TS = pd.bdate_range("2024-01-01", periods=WARMUP + BARS)


def _price():
    """Prices with a halt, a late listing and a delisting."""
    rng = np.random.default_rng(7)
    prices = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(len(TS), len(SYMBOLS))), axis=0))
    prices[WARMUP + 10 : WARMUP + 16, 1] = np.nan  # S1 halts
    prices[: WARMUP + 5, 2] = np.nan  # S2 lists late
    prices[WARMUP + 30 :, 3] = np.nan  # S3 delists
    return xr.DataArray(prices, dims=("timestamp", "symbol"), coords={"timestamp": TS, "symbol": SYMBOLS})


def _predictions():
    rng = np.random.default_rng(8)
    scores = rng.normal(size=(BARS, len(SYMBOLS)))
    scores[:, [1, 3]] += 3.0  # S1 and S3 are held into the halt and the delisting
    return xr.Dataset(
        {"ret_5": (("timestamp", "symbol"), scores)},
        coords={"timestamp": TS[WARMUP:], "symbol": SYMBOLS},
    )


def _dataset(price=None):
    price = _price() if price is None else price
    return FrameDataset(xr.Dataset({"open": price, "close": price}))


def _inputs(rule, dataset=None, *, anchor=None, end=None):
    return DecisionInputs(
        _dataset() if dataset is None else dataset,
        rule,
        fill_column="open",
        valuation_column="close",
        rebalance_periods=PERIODS,
        anchor=TS[WARMUP] if anchor is None else anchor,
        end=end,
    )


class _LogPrice:
    """Stands in for a ``Factor``: ``size`` is the log close, from the first bar asked for."""

    def get_factor_names(self):
        return ["size"]

    def compute(self, first, last):
        return xr.Dataset({"size": np.log(_price()).sel(timestamp=slice(first, last))})


#: The last bar the stand-in estimate store covers: a few rebalance bars lie after it.
ESTIMATE_END = TS[-8]
#: The symbols the stand-in estimate store has a specific risk for.
ESTIMATE_SYMBOLS = SYMBOLS[2:]


def _estimate_rows():
    """The stand-in estimate store: a factor covariance and specific risks of each bar to ``ESTIMATE_END``."""
    bars = TS[TS <= ESTIMATE_END]
    rng = np.random.default_rng(9)
    return xr.Dataset(
        {
            "factor_covariance": (
                ("timestamp", "factor_i", "factor_j"),
                np.repeat(np.array([[[0.04, 0.01], [0.01, 0.09]]]), len(bars), axis=0),
            ),
            "specific_risk": (("timestamp", "symbol"), rng.uniform(0.01, 0.03, (len(bars), len(ESTIMATE_SYMBOLS)))),
            "diagnostic": ("timestamp", np.arange(len(bars), dtype=float)),
        },
        coords={"timestamp": bars, "symbol": ESTIMATE_SYMBOLS, "factor_i": ["a", "b"], "factor_j": ["a", "b"]},
    )


class _EstimateStore:
    """Stands in for a ``RiskStore``: the recorded range and the rows of ``_estimate_rows``, reads recorded."""

    calls: list

    def store_range(self):
        return TS[0].date().isoformat(), ESTIMATE_END.date().isoformat()

    def read(self, start, end):
        self.calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        assert pd.Timestamp(start) >= TS[0] and pd.Timestamp(end) <= ESTIMATE_END + pd.Timedelta(1, "D")
        return _estimate_rows().sel(timestamp=slice(start, end))


class _RiskModel:
    """Stands in for a ``FactorRiskModel``: its exposures are the log close, read once per call."""

    calls: list

    def __init__(self):
        self.estimate = _EstimateStore()
        self.estimate.calls = []

    def exposures(self, first, last):
        self.calls.append((pd.Timestamp(first), pd.Timestamp(last)))
        return xr.Dataset({"size": np.log(_price()).sel(timestamp=slice(first, last))})


#: The last bar the stand-in market store covers, before the estimate store's end.
MARKET_END = TS[-14]


def _market_rows():
    """The stand-in market store: the market's exposures to ``a`` and ``b`` and its covered share."""
    bars = TS[TS <= MARKET_END]
    rng = np.random.default_rng(10)
    return xr.Dataset(
        {
            "market_exposure": (("timestamp", "factor"), rng.normal(size=(len(bars), 2))),
            "covered_share": ("timestamp", rng.uniform(0.9, 1.0, len(bars))),
        },
        coords={"timestamp": bars, "factor": ["a", "b"]},
    )


class _MarketStore(_EstimateStore):
    """Stands in for a ``MarketPortfolio``'s ``RiskStore``: the rows of ``_market_rows``, reads recorded."""

    def store_range(self):
        return TS[0].date().isoformat(), MARKET_END.date().isoformat()

    def read(self, start, end):
        self.calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        assert pd.Timestamp(start) >= TS[0] and pd.Timestamp(end) <= MARKET_END + pd.Timedelta(1, "D")
        return _market_rows().sel(timestamp=slice(start, end))


class _Market:
    """Stands in for a ``MarketPortfolio`` over ``risk``: its config names the risk model, its store the rows."""

    def __init__(self, risk):
        self.config = type("Config", (), {"risk_model": risk})()
        self.market = _MarketStore()
        self.market.calls = []


class _Recording:
    """Records every context ``construct`` is handed."""

    seen: list

    def construct(self, context):
        self.seen.append(context)
        return super().construct(context)


class _TopN(_Recording, TopNConstructor):
    pass


class _MeanVariance(_Recording, MeanVarianceOptimizer):
    pass


class _WithFactor(_TopN):
    def declared_inputs(self):
        return InputDeclaration(factors=(_LogPrice(),))


class _WithRiskModel(_TopN):
    risk = None

    def declared_inputs(self):
        return InputDeclaration(risk_model=self.risk)


class _WithMarket(_TopN):
    risk = market = None

    def declared_inputs(self):
        return InputDeclaration(risk_model=self.risk, market=self.market)


def _rule(kind):
    if kind == "mean_variance":
        rule = _MeanVariance(
            MeanVarianceConfig(
                expected_return_label="ret_5",
                covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
                expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
                risk_aversion=5.0,
                turnover_penalty=0.002,
                weight_cap=0.4,
            )
        )
    else:
        rule = {"factor": _WithFactor, "risk": _WithRiskModel, "market": _WithMarket}.get(kind, _TopN)(
            TopNConfig(direction="long_only", top_n=3)
        )
    if kind in ("risk", "market"):
        rule.risk = _RiskModel()
        rule.risk.calls = []
    if kind == "market":
        rule.market = _Market(rule.risk)
    rule.bind(SPECS)
    rule.seen = []
    return rule


def _assert_same_context(built: PortfolioContext, looped: PortfolioContext):
    assert built.timestamp == looped.timestamp
    for name in (
        "predictions", "tradable", "current_weights", "returns", "staleness", "factors",
        "risk_exposures", "risk_estimate", "market",
    ):
        a, b = getattr(built, name), getattr(looped, name)
        if b is None:
            assert a is None, name
        else:
            xr.testing.assert_identical(a, b)


@pytest.mark.parametrize("kind", ["top_n", "mean_variance", "factor", "risk", "market"])
def test_context_reproduces_every_rebalance_bar_of_weights(kind):
    rule = _rule(kind)
    inputs = _inputs(rule)
    predictions = _predictions()

    panel = inputs.weights(predictions)

    looped = list(rule.seen)
    assert len(looped) == schedule(BARS, PERIODS).sum()
    assert any((c.current_weights.values != 0).any() for c in looped)
    assert any(bool(c.locked.any()) for c in looped), "the replay must cover a locked position"
    for context in looped:
        t = context.timestamp
        built = inputs.context(t, predictions.sel(timestamp=t), context.current_weights)
        _assert_same_context(built, context)
        decision = rule.decide(built)
        assert decision.failure is None
        np.testing.assert_array_equal(decision.weights.values, panel["weight"].sel(timestamp=t).values)
    if kind == "mean_variance":
        assert all(c.returns.sizes["timestamp"] == 20 for c in looped)
    if kind == "factor":
        assert all(c.factors is not None for c in looped)
    if kind in ("risk", "market"):
        assert all(c.factors is None and c.risk_exposures is not None for c in looped)
    else:
        assert all(c.risk_exposures is None for c in looped)
    if kind == "market":
        assert any(c.market is not None for c in looped)
    else:
        assert all(c.market is None for c in looped)


def test_a_risk_models_exposures_are_read_once_over_the_panel_and_reindexed():
    rule = _rule("risk")
    predictions = _predictions()
    _inputs(rule).weights(predictions)
    # One request over the panel's bars, never one per bar.
    assert rule.risk.calls == [(TS[WARMUP], TS[-1])]
    for context in rule.seen:
        assert context.risk_exposures["size"].dims == ("symbol",)
        assert context.risk_exposures.symbol.values.tolist() == SYMBOLS

    # A held symbol without a prediction joins the bar with its exposure.
    t = TS[WARMUP + 3]
    held = xr.DataArray([0.4], dims="symbol", coords={"symbol": ["S1"]})
    context = _inputs(rule).context(t, predictions.sel(timestamp=t).drop_sel(symbol="S1"), held)
    assert rule.risk.calls[-1] == (t, t)
    assert context.risk_exposures.symbol.values.tolist() == context.symbols.tolist()
    assert context.risk_exposures["size"].sel(symbol="S1") == np.log(_price()).sel(timestamp=t, symbol="S1")


def test_a_risk_models_estimate_rows_are_read_once_over_the_scheduled_bars_and_reindexed():
    rule = _rule("risk")
    predictions = _predictions()
    _inputs(rule).weights(predictions)
    scheduled = TS[WARMUP:][schedule(BARS, PERIODS)]
    # One request from the first scheduled bar to the store's end, never one per bar.
    ((first, last),) = rule.risk.estimate.calls
    assert first == scheduled[0] and last.normalize() == ESTIMATE_END
    stored = _estimate_rows()
    covered = [c for c in rule.seen if c.timestamp <= ESTIMATE_END]
    assert covered and len(covered) < len(rule.seen)
    for context in covered:
        estimate = context.risk_estimate
        # The store contract's variables only, specific risks on the bar's symbols.
        assert sorted(estimate.data_vars) == ["factor_covariance", "specific_risk"]
        assert estimate["specific_risk"].dims == ("symbol",)
        assert estimate.symbol.values.tolist() == SYMBOLS
        row = stored.sel(timestamp=context.timestamp)
        xr.testing.assert_equal(
            estimate["specific_risk"], row["specific_risk"].reindex(symbol=SYMBOLS).drop_vars("timestamp")
        )
        np.testing.assert_array_equal(estimate["factor_covariance"].values, row["factor_covariance"].values)
    # A bar past the store has no estimate row.
    assert all(c.risk_estimate is None for c in rule.seen if c.timestamp > ESTIMATE_END)

    # One bar reads its own row, on the bar's symbols with a held one.
    t = TS[WARMUP + 3]
    held = xr.DataArray([0.4], dims="symbol", coords={"symbol": ["S1"]})
    context = _inputs(rule).context(t, predictions.sel(timestamp=t).drop_sel(symbol="S7"), held)
    assert rule.risk.estimate.calls[-1] == (t, t)
    assert context.risk_estimate.symbol.values.tolist() == context.symbols.tolist()
    assert context.risk_estimate["specific_risk"].sel(symbol="S6") == stored["specific_risk"].sel(timestamp=t, symbol="S6")
    assert np.isnan(context.risk_estimate["specific_risk"].sel(symbol="S1"))
    # A bar past the store reads nothing and has none.
    late = TS[-5]
    calls = len(rule.risk.estimate.calls)
    assert _inputs(rule).context(late, predictions.sel(timestamp=late), held).risk_estimate is None
    assert len(rule.risk.estimate.calls) == calls


def test_a_market_portfolios_rows_are_read_once_over_the_scheduled_bars():
    rule = _rule("market")
    predictions = _predictions()
    _inputs(rule).weights(predictions)
    scheduled = TS[WARMUP:][schedule(BARS, PERIODS)]
    # One request from the first scheduled bar to the store's end, never one per bar.
    ((first, last),) = rule.market.market.calls
    assert first == scheduled[0] and last.normalize() == MARKET_END
    stored = _market_rows()
    covered = [c for c in rule.seen if c.timestamp <= MARKET_END]
    assert covered and len(covered) < len(rule.seen)
    for context in covered:
        xr.testing.assert_identical(context.market, stored.sel(timestamp=context.timestamp, drop=True))
    # A bar past the store has no market row.
    assert all(c.market is None for c in rule.seen if c.timestamp > MARKET_END)

    # One bar reads its own row; a bar past the store reads nothing and has none.
    t = TS[WARMUP + 3]
    held = xr.DataArray([0.4], dims="symbol", coords={"symbol": ["S1"]})
    context = _inputs(rule).context(t, predictions.sel(timestamp=t), held)
    assert rule.market.market.calls[-1] == (t, t)
    xr.testing.assert_identical(context.market, stored.sel(timestamp=t, drop=True))
    late = TS[-5]
    calls = len(rule.market.market.calls)
    assert _inputs(rule).context(late, predictions.sel(timestamp=late), held).market is None
    assert len(rule.market.market.calls) == calls


def test_a_decision_panel_holds_the_market_rows_of_its_bars():
    rule = _rule("market")
    predictions = _predictions()
    inputs = _inputs(rule)
    bars = [TS[WARMUP + 2], TS[WARMUP + 5], TS[-3]]

    panel = inputs.panel(predictions.sel(timestamp=bars))

    stored = _market_rows()
    assert panel.market_rows.timestamp.values.tolist() == stored.timestamp.values[
        np.isin(stored.timestamp.values, bars)].tolist()
    xr.testing.assert_identical(panel.market_rows, stored.sel(timestamp=bars[:2]))
    assert inputs.panel(predictions.sel(timestamp=[TS[-3]])).market_rows is None
    assert _inputs(_rule("risk")).panel(predictions.sel(timestamp=bars)).market_rows is None


def test_a_market_portfolio_is_declared_with_its_own_risk_model():
    risk, other = _RiskModel(), _RiskModel()
    market = _Market(risk)

    assert InputDeclaration(risk_model=risk, market=market).market is market
    with pytest.raises(ValueError, match="declared with its own risk model"):
        InputDeclaration(market=market)
    with pytest.raises(ValueError, match="declared with its own risk model"):
        InputDeclaration(risk_model=other, market=market)
    merged = InputDeclaration(lookback_bars=3).merged(InputDeclaration(risk_model=risk, market=market))
    assert merged.market is market and merged.risk_model is risk
    with pytest.raises(ValueError, match="two different market portfolios"):
        InputDeclaration(risk_model=risk, market=market).merged(
            InputDeclaration(risk_model=risk, market=_Market(risk))
        )


def test_a_held_symbol_without_a_prediction_joins_the_bar_locked():
    rule = _rule("top_n")
    t = TS[WARMUP + 12]  # S1 halted
    predictions = _predictions().sel(timestamp=t).drop_sel(symbol="S1")
    held = xr.DataArray([0.4], dims="symbol", coords={"symbol": ["S1"]})

    context = _inputs(rule).context(t, predictions, held)

    assert context.symbols.tolist() == [s for s in SYMBOLS if s != "S1"] + ["S1"]
    assert np.isnan(context.predictions["ret_5"].sel(symbol="S1"))
    assert context.locked.sel(symbol="S1")
    assert rule.decide(context).weights.sel(symbol="S1") == 0.4


def test_rebalances_is_the_schedule_weights_follows():
    rule = _rule("top_n")
    inputs = _inputs(rule)
    predictions = _predictions()

    panel = inputs.weights(predictions)

    decided = np.isfinite(panel["weight"].values).all(axis=1)
    asked = np.array([inputs.rebalances(t) for t in TS[WARMUP:]])
    np.testing.assert_array_equal(asked, schedule(BARS, PERIODS))
    assert not (decided & ~asked).any()
    assert not inputs.rebalances(TS[WARMUP - 1])  # before the anchor
    assert not _inputs(rule, end=TS[WARMUP + 3]).rebalances(TS[WARMUP + 3])


def test_every_decided_bar_is_logged_with_its_count_names_and_seconds():
    from loguru import logger

    rule = _rule("top_n")
    inputs = _inputs(rule)
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="INFO", format="{message}",
                            filter=lambda r: r["name"] == "quantlab.portfolio.decision_inputs")
    try:
        panel = inputs.weights(_predictions())
    finally:
        logger.remove(handler_id)

    scheduled = np.flatnonzero(panel["scheduled"].values)
    lines = [m.strip() for m in messages if " bar " in m]
    assert len(lines) == len(scheduled)
    for k, (t, line) in enumerate(zip(scheduled, lines), start=1):
        row = panel["weight"].values[t]
        names = int((np.abs(row) > 0).sum())
        bar = pd.Timestamp(panel.timestamp.values[t]).isoformat()
        assert line.startswith(f"{type(rule).__name__}: bar {k}/{len(scheduled)} {bar}: {names} names, ")
        assert line.endswith(" s")


def test_rebalances_reads_the_calendar_once_over_a_replay(monkeypatch):
    dataset = _dataset()
    reads = []
    original = type(dataset).calendar

    def counting(self, start, end):
        reads.append((start, end))
        return original(self, start, end)

    monkeypatch.setattr(type(dataset), "calendar", counting)
    inputs = _inputs(_rule("top_n"), dataset)

    asked = [inputs.rebalances(t) for t in TS[WARMUP:]]

    assert len(reads) == 1
    np.testing.assert_array_equal(asked[:-1], schedule(BARS, PERIODS)[:-1])


def test_a_panel_starting_after_the_anchor_keeps_the_anchors_schedule():
    rule = _rule("top_n")
    inputs = _inputs(rule, anchor=TS[WARMUP - 2])
    predictions = _predictions()

    inputs.weights(predictions)

    decided = [c.timestamp for c in rule.seen]
    expected = [t for i, t in enumerate(TS[WARMUP - 2 :]) if i % PERIODS == 0 and t >= TS[WARMUP]]
    assert decided == [pd.Timestamp(t) for t in expected if t != TS[-1]]


def test_only_weights_warns_about_a_short_history():
    rule = _rule("mean_variance")  # history_bars 26: 25 bars before the first
    short = _dataset(_price().isel(timestamp=slice(WARMUP - 10, None)))
    inputs = _inputs(rule, short)
    predictions = _predictions()

    with pytest.warns(UserWarning, match=r"reads 25 bar\(s\) of prices .* holds only 10; .* short by 15"):
        inputs.weights(predictions)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = TS[WARMUP]
        inputs.context(t, predictions.sel(timestamp=t), xr.zeros_like(predictions["ret_5"].sel(timestamp=t)))


def test_refusals():
    rule = _rule("top_n")
    inputs = _inputs(rule)
    predictions = _predictions()
    t = TS[WARMUP + 1]
    bar = predictions.sel(timestamp=t)
    flat = xr.zeros_like(bar["ret_5"])

    with pytest.raises(ValueError, match="before the anchor"):
        _inputs(rule, anchor=TS[WARMUP + 1]).weights(predictions)
    with pytest.raises(ValueError, match="not a bar"):
        inputs.context(pd.Timestamp("2024-01-06"), bar, flat)  # a Saturday
    with pytest.raises(ValueError, match="finite"):
        inputs.context(t, bar, flat.where(flat.symbol != "S0"))
    with pytest.raises(ValueError, match="one bar"):
        inputs.context(t, predictions, flat)
    with pytest.raises(ValueError, match="rebalance_periods"):
        DecisionInputs(_dataset(), rule, fill_column="open", valuation_column="close", rebalance_periods=0, anchor=t)


def test_factor_panels_lacking_a_declared_name_are_refused():
    class _Other(_LogPrice):
        def get_factor_names(self):
            return ["beta"]

    class _Declares(_TopN):
        def declared_inputs(self):
            return InputDeclaration(factors=(_Other(),))

    rule = _Declares(TopNConfig(direction="long_only", top_n=1))
    rule.seen = []
    with pytest.raises(ValueError, match=r"\['beta'\]"):
        _inputs(rule).weights(_predictions())


# -- one module assembles the inputs --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = "quantlab/portfolio/decision_inputs.py"
CALLERS = ("quantlab/backtest/predefined/us_equity.py", "quantlab/backtest/base.py", "quantlab/api/_backtest.py")
#: Names only the module may use to assemble decision inputs.
ASSEMBLY = {"tradable_bars", "_scheduled", "bar_before", "declared_inputs", "history_bars", "lookback_bars", "ExecutionBook"}


def _names(path: Path) -> set[str]:
    """Every name and attribute the code of ``path`` refers to (docstrings are not code)."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
    return names


def test_no_context_is_built_outside_the_module():
    builders = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "quantlab").rglob("*.py")
        if any(
            isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "PortfolioContext"
            for node in ast.walk(ast.parse(path.read_text()))
        )
    )
    assert builders == [MODULE]


@pytest.mark.parametrize("caller", CALLERS)
def test_the_callers_hold_no_assembly_wiring(caller):
    assert _names(REPO_ROOT / caller) & ASSEMBLY == set()


def test_the_rules_hold_no_assembly_code():
    from quantlab.portfolio.base import PortfolioConstructor

    assert not hasattr(PortfolioConstructor, "build_context")
    assert not hasattr(PortfolioConstructor, "construct_panel")
