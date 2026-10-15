"""``DecisionInputs.preloaded`` reads a replay window once; each bar's context is sliced from it (#232).

What is locked here, and what turns it red:

- Inside ``preloaded(start, end)``, ``context`` at every bar of the window
  equals the context built without the preload, value for value (prices
  window, returns, staleness, tradability, factors, risk exposures and
  estimate row), for
  top-n, mean-variance on Ledoit-Wolf, a rule declaring a factor and one
  reading a risk model's exposures, at every bar of a window starting
  mid-panel (its first bars' history lies before it) and on a dataset holding
  less history than the rule reads; and ``decide`` gives the same weights.
- No look-ahead: prices after a bar, changed, change no preloaded context up
  to that bar.
- A bar outside the window, or a second preload, raises; after the context
  closes, ``context`` reads its sources again.
- On a real factor risk model (USE4 stores, ``FactorRiskStoreEstimator``)
  rebuilt from a backtest run, under ``"read"`` and ``"cal"``: every
  preloaded context and the estimator's forecast through it equal the
  unpreloaded ones bit for bit, a decided bar opens no store (it opened
  several without the preload), and a ``DataRecorder`` records the window's
  exposures and estimate reads as one request each.
- The estimate rows are decision inputs (#287): a panel opens the estimate
  store once (it opened it once per decided bar) and records the request the
  per-bar reads made; the risk layer holds no rows in memory (``RiskStore``
  has no ``held``, no ``ContextVar``).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.record import DataRecorder
from tests.test_decision_inputs import (
    TS,
    WARMUP,
    _assert_same_context,
    _dataset,
    _inputs,
    _predictions,
    _price,
    _rule,
)


def _assert_same_decision(rule, a, b):
    x, y = rule.decide(a), rule.decide(b)
    assert x.failure == y.failure
    np.testing.assert_array_equal(x.weights.values, y.weights.values)


def _holdings(rule, inputs, predictions):
    """The replayed holdings of every rebalance bar of ``weights``, by bar."""
    rule.seen = []
    inputs.weights(predictions)
    return {c.timestamp: c.current_weights for c in rule.seen}


@pytest.mark.parametrize("kind", ["top_n", "mean_variance", "factor", "risk"])
def test_every_preloaded_context_equals_the_unpreloaded_one(kind):
    rule = _rule(kind)
    inputs = _inputs(rule)
    predictions = _predictions()
    held = _holdings(rule, inputs, predictions)
    assert any((w.values != 0).any() for w in held.values())

    # Every bar of a window starting mid-panel: its first bars read history before it.
    window = TS[WARMUP + 10 :]
    flat = xr.zeros_like(predictions["ret_5"].isel(timestamp=0, drop=True))
    unpreloaded = {}
    for t in window:
        weights = held.get(t, flat)
        unpreloaded[t] = inputs.context(t, predictions.sel(timestamp=t), weights)
    with inputs.preloaded(window[0], window[-1]):
        preloaded = {
            t: inputs.context(t, predictions.sel(timestamp=t), held.get(t, flat)) for t in window
        }
    for t in window:
        _assert_same_context(preloaded[t], unpreloaded[t])
        _assert_same_decision(rule, preloaded[t], unpreloaded[t])
    if kind == "risk":
        assert all(c.risk_exposures is not None for c in preloaded.values())
    if kind == "factor":
        assert all(c.factors is not None for c in preloaded.values())


def test_a_preload_on_a_short_history_counts_the_history_as_context_does():
    rule = _rule("mean_variance")  # history_bars 26
    short = _dataset(_price().isel(timestamp=slice(WARMUP - 10, None)))  # 10 bars before the panel
    inputs = _inputs(rule, short)
    predictions = _predictions()
    flat = xr.zeros_like(predictions["ret_5"].isel(timestamp=0, drop=True))
    window = TS[WARMUP : WARMUP + 20]

    unpreloaded = [inputs.context(t, predictions.sel(timestamp=t), flat) for t in window]
    with inputs.preloaded(window[0], window[-1]):
        preloaded = [inputs.context(t, predictions.sel(timestamp=t), flat) for t in window]

    for a, b in zip(preloaded, unpreloaded):
        _assert_same_context(a, b)
    # The first windows are short, as without the preload.
    assert preloaded[0].returns.sizes["timestamp"] < preloaded[-1].returns.sizes["timestamp"] == 20


def test_prices_after_a_bar_change_no_preloaded_context_up_to_it():
    rule = _rule("mean_variance")
    predictions = _predictions()
    flat = xr.zeros_like(predictions["ret_5"].isel(timestamp=0, drop=True))
    cut = WARMUP + 20
    changed = _price()
    changed[cut + 1 :] = changed[cut + 1 :] * 1.5
    window = TS[WARMUP:]

    contexts = []
    for price in (_price(), changed):
        inputs = _inputs(rule, _dataset(price))
        with inputs.preloaded(window[0], window[-1]):
            contexts.append(
                [inputs.context(t, predictions.sel(timestamp=t), flat) for t in TS[WARMUP : cut + 1]]
            )
    for a, b in zip(*contexts):
        _assert_same_context(a, b)


def test_a_bar_outside_the_window_or_a_second_preload_is_refused():
    rule = _rule("top_n")
    inputs = _inputs(rule)
    predictions = _predictions()
    flat = xr.zeros_like(predictions["ret_5"].isel(timestamp=0, drop=True))
    inside, before, after = TS[WARMUP + 3], TS[WARMUP + 2], TS[WARMUP + 9]

    with inputs.preloaded(inside, TS[WARMUP + 8]):
        with pytest.raises(ValueError, match="outside the preloaded window"):
            inputs.context(before, predictions.sel(timestamp=before), flat)
        with pytest.raises(ValueError, match="outside the preloaded window"):
            inputs.context(after, predictions.sel(timestamp=after), flat)
        with pytest.raises(ValueError, match="not a bar"):
            inputs.context(pd.Timestamp("2024-02-17"), predictions.sel(timestamp=inside), flat)
        with pytest.raises(ValueError, match="already preloaded"), inputs.preloaded(inside, inside):
            pass
    # Closed: the sources answer again.
    inputs.context(after, predictions.sel(timestamp=after), flat)
    with pytest.raises(ValueError, match="no bar"), inputs.preloaded("2030-01-01", "2030-01-31"):
        pass


# -- a real factor risk model, rebuilt from a backtest run ------------------------------


class _StoreOpens:
    """Counts Zarr store opens (``xr.open_zarr`` and ``xr.open_dataset``)."""

    def __init__(self, monkeypatch):
        self.count = 0
        for name in ("open_zarr", "open_dataset"):
            original = getattr(xr, name)

            def counting(*args, _original=original, **kwargs):
                self.count += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(xr, name, counting)


@pytest.fixture(scope="module", params=["read", "cal"])
def factor_risk_run(request, tmp_path_factory):
    from tests.test_backtest_mean_variance import _factor_risk_run

    root = tmp_path_factory.mktemp(f"preload_{request.param}")
    result = _factor_risk_run(root, request.param, exposure_bounds={"style_a": (-0.3, 0.3)}).run()
    return request.param, result


def _decided_bars(result):
    weights = result.weights["weight"]
    return weights, weights.timestamp.values[np.isfinite(weights.values).all(axis=1)]


def test_a_factor_risk_rule_decides_identically_preloaded_and_opens_no_store(factor_risk_run, monkeypatch):
    strategy, result = factor_risk_run
    inputs = DecisionInputs.from_run(result.run_dir)
    predictions = BacktestRun.open(result.run_dir).predictions().predictions
    weights, decided = _decided_bars(result)
    rule = inputs.constructor
    estimator = rule.config.covariance

    def bars(previous=None):
        out = []
        for t in decided:
            t = pd.Timestamp(t)
            # The previous decision as the holdings: a book with positions.
            held = xr.zeros_like(weights.sel(timestamp=t)) if previous is None else previous
            context = inputs.context(t, predictions.sel(timestamp=t), held.drop_vars("timestamp", errors="ignore"))
            out.append(context)
            previous = weights.sel(timestamp=t)
        return out

    opens = _StoreOpens(monkeypatch)
    unpreloaded = bars()
    per_bar_before = opens.count / len(decided)
    opens.count = 0
    with inputs.preloaded(decided[0], weights.timestamp.values[-1]):
        preload_opens = opens.count
        opens.count = 0
        preloaded = bars()
        per_bar_after = opens.count / len(decided)
        forecasts = [estimator.estimate(c) for c in preloaded]
        decisions = [rule.decide(c) for c in preloaded]
        assert opens.count == 0

    # The estimate store, and the exposures store under "read".
    assert per_bar_before >= (2 if strategy == "read" else 1), per_bar_before
    assert per_bar_after == 0
    assert preload_opens > 0
    print(
        f"\n[{strategy}] store opens per decided bar: {per_bar_before:g} without the preload, "
        f"{per_bar_after:g} preloaded ({preload_opens:g} once, on entering; {len(decided)} bars)"
    )
    for a, b, forecast, decision in zip(preloaded, unpreloaded, forecasts, decisions):
        _assert_same_context(a, b)
        assert a.risk_exposures is not None
        for x, y in zip(forecast.factor_form(), estimator.estimate(b).factor_form()):
            np.testing.assert_array_equal(x, y)
        again = rule.decide(b)
        assert decision.failure == again.failure
        np.testing.assert_array_equal(decision.weights.values, again.weights.values)
    # Holding-independent first bar: the run's own row.
    np.testing.assert_array_equal(
        decisions[0].weights.values, weights.sel(timestamp=decided[0]).values
    )


def _keys(inputs):
    model = inputs.declared.risk_model
    return [
        (inputs.dataset, "price_dataset"),
        (model, "constructor.covariance.risk_model"),
        (model.config.exposures, "constructor.covariance.risk_model.exposures"),
    ]


def test_a_preloaded_replay_records_each_read_once_over_the_window(factor_risk_run):
    strategy, result = factor_risk_run
    inputs = DecisionInputs.from_run(result.run_dir)
    predictions = BacktestRun.open(result.run_dir).predictions().predictions
    weights, decided = _decided_bars(result)
    first, last = pd.Timestamp(decided[0]), pd.Timestamp(weights.timestamp.values[-1])

    with DataRecorder(keys=_keys(inputs)) as recorder, inputs.preloaded(first, last):
        for t in decided:
            inputs.constructor.decide(
                inputs.context(t, predictions.sel(timestamp=t), xr.zeros_like(weights.sel(timestamp=t, drop=True)))
            )
    records = recorder.records
    window = (first.isoformat(), last.isoformat())
    estimate, = records["constructor.covariance.risk_model.estimate"]
    assert (estimate["request"]["start"], estimate["request"]["end"]) == window
    if strategy == "read":
        exposures, = records["constructor.covariance.risk_model.exposures"]
        assert (exposures["request"]["start"], exposures["request"]["end"]) == window
    assert all(entry["digest"] for entries in records.values() for entry in entries)


def test_a_panel_reads_the_estimate_store_once_and_records_the_request_per_bar_reads_made(
    factor_risk_run, monkeypatch
):
    strategy, result = factor_risk_run
    inputs = DecisionInputs.from_run(result.run_dir)
    predictions = BacktestRun.open(result.run_dir).predictions().predictions
    model = inputs.declared.risk_model
    path = model.estimate.path
    weights, decided = _decided_bars(result)
    assert len(decided) > 3
    opened = []
    original = xr.open_zarr

    def counting(store, *args, **kwargs):
        opened.append(str(store))
        return original(store, *args, **kwargs)

    monkeypatch.setattr(xr, "open_zarr", counting)
    with DataRecorder(keys=_keys(inputs)) as recorder:
        again = inputs.weights(predictions)
        # Once over the panel, not once per decided bar (#287); the recorder
        # re-reads it once more on closing, to hash it.
        assert opened.count(path) == 1
    np.testing.assert_array_equal(again["weight"].values, weights.values)
    # The request the per-bar reads made: the first to the last rebalance bar.
    scheduled = again.timestamp.values[again["scheduled"].values]
    estimate, = recorder.records["constructor.covariance.risk_model.estimate"]
    assert (estimate["request"]["start"], estimate["request"]["end"]) == (
        pd.Timestamp(scheduled[0]).isoformat(), pd.Timestamp(scheduled[-1]).isoformat()
    )
    # And the run recorded the same request when it ran.
    run_fingerprint = BacktestRun.open(result.run_dir).data_fingerprint
    recorded, = run_fingerprint["constructor.covariance.risk_model.estimate"]
    assert recorded["request"] == estimate["request"] and recorded["digest"] == estimate["digest"]


def test_the_risk_store_holds_nothing_in_memory():
    from quantlab.risk import base

    assert not hasattr(base.RiskStore, "held")
    assert not any(isinstance(v, __import__("contextvars").ContextVar) for v in vars(base).values())
