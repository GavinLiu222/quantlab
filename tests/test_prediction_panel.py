"""Label specs and the prediction panel file (#107, ADR 0005 of quantlab-ibkr).

What is locked here, and what turns it red:

- `PredictionPanel.write`/`read` round-trip the predictions and their label specs
  through a Zarr store (one variable per label on `(timestamp, symbol)`, attrs
  `format_version` and JSON `labels`);
- a panel whose variables are not exactly its specs' names is refused;
- portfolio constructors bind to `LabelSpec`s: TopN refuses an unknown `score_label`,
  mean-variance refuses a label without a span;
- `label_specs(predictor)` derives one spec per label variable from `labels`,
  `label_delays` and `label_scales`;
- `run()` and `run_cv()` (stitched run only) write a prediction panel, read back
  through `BacktestRun.predictions()`; `run_weights()` writes none;
- `DecisionInputs.from_run(run_dir)` rebuilds the run's decision inputs (the rule bound
  to the panel's specs, the price dataset, an in-memory one from the run directory, the
  market columns, the execution settings, the rebalance period and the anchor), whose
  `weights()` reproduce the run's weights; importing and calling it loads no model,
  factor, label or backtest module, nor torch, xgboost, KunQuant or vectorbt.

Everything is synthetic, CPU-only and offline.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.runs.prediction_panel import LabelSpec, PredictionPanel
from quantlab.runs.backtest_run import BacktestRun

SPECS = (
    LabelSpec(name="ret_5", scale="raw", delay=1, span=5),
    LabelSpec(name="rank_1", scale="standardized", delay=1, span=None),
)


def _predictions(symbols) -> xr.Dataset:
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2024-01-02", periods=6, freq="B")
    shape = (len(timestamps), len(symbols))
    values = {name: rng.normal(size=shape) for name in ("ret_5", "rank_1")}
    values["ret_5"][2, 1] = np.nan
    return xr.Dataset(
        {name: (("timestamp", "symbol"), value) for name, value in values.items()},
        coords={"timestamp": timestamps, "symbol": symbols},
    )


@pytest.mark.parametrize(
    "symbols",
    [np.array(["AAA", "BBB", "CCC"], dtype=object), np.array([10001, 10002, 10107])],
    ids=["tickers", "permnos"],
)
def test_a_prediction_panel_round_trips_through_its_file(tmp_path, symbols):
    panel = PredictionPanel(_predictions(symbols), SPECS)

    panel.write(tmp_path / "panel.zarr")
    back = PredictionPanel.read(tmp_path / "panel.zarr")

    assert back.labels == SPECS
    xr.testing.assert_equal(back.predictions, panel.predictions)
    assert back.predictions.symbol.values.tolist() == symbols.tolist()


def test_the_file_holds_one_variable_per_label_and_json_specs(tmp_path):
    PredictionPanel(_predictions(np.array(["AAA", "BBB"], dtype=object)), SPECS).write(
        tmp_path / "panel.zarr"
    )

    stored = xr.open_zarr(tmp_path / "panel.zarr")
    assert sorted(stored.data_vars) == ["rank_1", "ret_5"]
    assert all(stored[name].dims == ("timestamp", "symbol") for name in stored.data_vars)
    assert stored.attrs["format_version"] == 1
    assert json.loads(stored.attrs["labels"]) == [
        {"name": "ret_5", "scale": "raw", "delay": 1, "span": 5},
        {"name": "rank_1", "scale": "standardized", "delay": 1, "span": None},
    ]


def test_a_panel_whose_variables_are_not_its_labels_is_refused():
    predictions = _predictions(np.array(["AAA", "BBB"], dtype=object))
    with pytest.raises(ValueError, match="rank_1"):
        PredictionPanel(predictions[["ret_5"]], SPECS)


def test_top_n_binds_to_label_specs_and_refuses_an_unknown_score_label():
    from quantlab.portfolio.config import TopNConfig
    from quantlab.portfolio.predefined.top_n import TopNConstructor

    TopNConstructor(TopNConfig(direction="long_only", top_n=1, score_label="rank_1")).bind(
        SPECS
    )
    rule = TopNConstructor(TopNConfig(direction="long_only", top_n=1, score_label="ret_20"))
    with pytest.raises(ValueError, match=r"score_label 'ret_20' is not one of .*\['ret_5', 'rank_1'\]"):
        rule.bind(SPECS)
    with pytest.raises(ValueError, match="no predicted labels"):
        TopNConstructor(TopNConfig(direction="long_only", top_n=1)).bind([])


def _optimizer(label: str):
    from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig
    from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
    from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer

    return MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label=label,
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
            risk_aversion=5.0,
        )
    )


def test_mean_variance_reads_the_span_of_its_label_spec():
    optimizer = _optimizer("ret_5")
    optimizer.bind(SPECS)
    assert optimizer.span == 5


def test_mean_variance_refuses_a_label_spec_without_a_span():
    with pytest.raises(
        ValueError, match="expected_return_label 'rank_1' has no span; it needs a Forward label"
    ):
        _optimizer("rank_1").bind(SPECS)


class _Label:
    def __init__(self, names, span=None):
        self._names = names
        if span is not None:
            self.span_bars = lambda: span

    def get_factor_names(self):
        return list(self._names)


class _Predictor:
    labels = [_Label(["ret_5"], span=5), _Label(["up_1", "down_1"])]
    label_delays = (1, 2)
    label_scales = {"ret_5": "raw", "up_1": "standardized", "down_1": "raw"}


def test_label_specs_derives_one_spec_per_label_variable():
    from quantlab.backtest.base import label_specs

    assert label_specs(_Predictor()) == (
        LabelSpec(name="ret_5", scale="raw", delay=1, span=5),
        LabelSpec(name="up_1", scale="standardized", delay=2, span=None),
        LabelSpec(name="down_1", scale="raw", delay=2, span=None),
    )


# --- run directories ---------------------------------------------------------


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _model_run(root, *, frame=False):
    """One load-mode `run()` with a top-2 rule over a store-backed price dataset, or with
    ``frame`` over an in-memory copy of it with fees and valuation sizing."""
    from quantlab.backtest.config import CrossSectionBacktestConfig
    from quantlab.portfolio.config import TopNConfig
    from quantlab.backtest.predefined.us_equity import (
        USEquityCrossectionSelectStockVectorBt,
    )
    from quantlab.portfolio.predefined.top_n import TopNConstructor
    from tests.backtest_fixtures import (
        make_model,
        make_stock_dataset,
        train_checkpoint,
        write_price_store,
    )

    from quantlab.dataset.memory import FrameDataset

    dataset_config = write_price_store(root, n_bars=60)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    dates = dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )
    checkpoint = train_checkpoint(make_model(root / "train", dataset_config, **dates))
    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=(
                FrameDataset(xr.open_zarr(dataset_config.zarr_file_path).load())
                if frame
                else make_stock_dataset(dataset_config)
            ),
            model=make_model(root / "backtest", dataset_config, **dates),
            model_mode="load",
            checkpoint=str(checkpoint),
            start_date=_day(bars[30]),
            end_date=_day(bars[50]),
            output_dir=str(root / "runs"),
            rebalance_periods=5,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
            **(dict(fees=0.001, slippage=0.0005, sizing_basis="valuation") if frame else {}),
        )
    )
    return backtester.run()


@pytest.fixture(scope="module")
def model_run(tmp_path_factory):
    """The store-backed run, shared read-only by the tests below."""
    return _model_run(tmp_path_factory.mktemp("model_run"))


@pytest.fixture(scope="module")
def frame_run(tmp_path_factory):
    """The in-memory run, shared read-only by the tests below."""
    return _model_run(tmp_path_factory.mktemp("frame_run"), frame=True)


def test_run_writes_the_window_predictions_and_their_label_specs(model_run):
    panel = BacktestRun.open(model_run.run_dir).predictions()

    assert panel.labels == (LabelSpec(name="fwd_ret_1", scale="raw", delay=1, span=1),)
    xr.testing.assert_equal(panel.predictions, model_run.predictions)


@pytest.mark.parametrize("run", ["model_run", "frame_run"])
def test_from_run_rebuilds_inputs_that_reproduce_the_runs_weights(run, request):
    from quantlab.backtest.config import BacktestConfig
    from quantlab.portfolio.config import TopNConfig
    from quantlab.portfolio.decision_inputs import DecisionInputs
    from quantlab.portfolio.predefined.top_n import TopNConstructor
    from quantlab.execution.rules import ExecutionSettings

    result = request.getfixturevalue(run)
    inputs = DecisionInputs.from_run(result.run_dir)

    assert inputs.constructor == TopNConstructor(TopNConfig(direction="long_only", top_n=2))
    assert (inputs.fill_column, inputs.valuation_column) == ("adjOpen", "adjClose")
    assert inputs.rebalance_periods == 5
    assert inputs.anchor == pd.Timestamp(result.predictions.timestamp.values[0])
    # Open-ended by default (a live run keeps counting past the run's end); a replay
    # passes its last bar, which then never rebalances.
    assert inputs.end is None
    bars = result.weights.timestamp.values
    replay = DecisionInputs.from_run(result.run_dir, end=bars[-1])
    decided = np.isfinite(result.weights["weight"].values).all(axis=1)
    asked = np.array([replay.rebalances(t) for t in bars])
    np.testing.assert_array_equal(decided & asked, decided)
    assert not asked[-1]
    assert [inputs.rebalances(t) for t in bars[:-1]] == asked[:-1].tolist()
    assert inputs.execution == (
        ExecutionSettings("valuation", 0.001, 0.0005)
        if run == "frame_run"
        else ExecutionSettings("fill", BacktestConfig.fees, BacktestConfig.slippage)
    )
    predictions = BacktestRun.open(result.run_dir).predictions().predictions
    weights = inputs.weights(predictions)["weight"]
    xr.testing.assert_equal(weights, result.weights["weight"].sel(symbol=weights.symbol.values))
    assert np.isfinite(weights.values).any()


def test_from_run_refuses_a_run_without_a_prediction_panel(tmp_path, model_run):
    from quantlab.portfolio.decision_inputs import DecisionInputs

    # A run_weights run has no model, so no prediction panel.
    weights_only = BacktestRun.open(model_run.run_dir).rebuild_backtester(
        model=None, model_mode=None, checkpoint=None, output_dir=str(tmp_path)
    )
    run_dir = weights_only.run_weights(model_run.weights).run_dir
    assert BacktestRun.open(run_dir).predictions() is None
    with pytest.raises(FileNotFoundError, match="no prediction panel"):
        DecisionInputs.from_run(run_dir)


def test_from_run_loads_no_model_factor_label_or_engine_module(model_run):
    import subprocess
    import sys

    from tests.test_backtest_contracts import REPO_ROOT

    code = (
        "import sys\n"
        "from quantlab.portfolio.decision_inputs import DecisionInputs\n"
        f"inputs = DecisionInputs.from_run({str(model_run.run_dir)!r})\n"
        "print(type(inputs.constructor).__name__)\n"
        "banned = ('quantlab.model', 'quantlab.factor', 'quantlab.label',\n"
        "          'quantlab.backtest', 'quantlab.model.base', 'quantlab.factor.base',\n"
        "          'quantlab.backtest.base', 'torch', 'xgboost', 'KunQuant', 'vectorbt')\n"
        "print(sorted(m for m in sys.modules\n"
        "             if any(m == b or m.startswith(b + '.') for b in banned)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-2:] == ["TopNConstructor", "[]"]
