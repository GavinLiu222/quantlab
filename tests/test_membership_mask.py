"""Index membership masks the predictions, never the prices (quantlab-ibkr ADR 0006).

What is locked here, and what turns it red:

- ``MembershipMaskedPredictor(model, membership)`` backtested over an UNMASKED
  price dataset: a security that leaves the index keeps its prices, is never
  selected after it left, and the holding it had is sold at the next open
  (an order, not a delisting settlement);
- a security that is never a member in the window is never predicted, so the
  run leaves it off its axis and never buys it;
- the run's prediction panel (``BacktestRun.predictions``) carries the masked predictions;
- the masked run equals the unmasked one up to the leaving date;
- the membership panel is fingerprinted, and the run's
  ``BacktestRun.rebuild_backtester`` rebuilds the wrapper and replays the run;
- ``run_cv`` masks every fold and the stitched prediction panel;
- predicting a bar the membership panel does not cover is refused (unknown
  membership is not "not a member").

Everything is synthetic, CPU-only and offline.
"""


import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.raw import IndexConstituentDataset
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import make_model, train_checkpoint
from tests.test_backtest_predictor_protocol import _backtester, _model_dates, _setup

COVERAGE_START = "2023-01-02"


class IntervalMembership(IndexConstituentDataset):
    """A membership panel built from the class-level ``intervals``.

    Only ``from_raw_data()`` reads them; a rebuilt instance
    (``rebuild``) reads the saved store.
    """

    intervals: list[tuple] = []

    def _pit_coverage_start(self) -> str:
        return COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        return pl.DataFrame(
            self.intervals, schema=["symbol", "start_date", "end_date"], orient="row"
        )


def _membership(tmp_path, intervals, *, as_of="2024-03-29") -> IntervalMembership:
    IntervalMembership.intervals = intervals
    dataset = IntervalMembership(
        ConstituentDatasetConfig(
            zarr_file_path=str(tmp_path / "membership.zarr"),
            cache_dir=str(tmp_path / "membership_cache"),
            as_of=as_of,
        )
    )
    dataset.from_raw_data().save()
    return dataset


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _rebalance_rows(weights: xr.Dataset) -> pd.DataFrame:
    frame = weights["weight"].to_pandas()
    return frame[frame.notna().any(axis=1)]


@pytest.fixture
def scenario(tmp_path):
    """An unmasked baseline run, a leaver it holds on two consecutive
    rebalance bars, and a symbol it selects that is never a member."""
    dataset_config, bars = _setup(tmp_path)
    dates = _model_dates(bars)
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **dates))
    baseline = _backtester(
        tmp_path, dataset_config,
        make_model(tmp_path / "plain", dataset_config, **dates),
        bars, name="plain", checkpoint=checkpoint,
    ).run()

    rebalances = _rebalance_rows(baseline.weights)
    leaver = first_held = None
    for (t1, row1), (t2, row2) in zip(rebalances.iterrows(), rebalances.iloc[1:].iterrows()):
        both = [s for s in rebalances.columns if row1[s] > 0 and row2[s] > 0]
        if both:
            leaver, first_held, second_held = both[0], t1, t2
            break
    assert leaver is not None, "fixture: no symbol held on two consecutive rebalances"
    outsider = next(
        s for s in rebalances.columns
        if s != leaver and (rebalances[s] > 0).any()
    )

    # The leaver is a member through the first holding bar and leaves the day
    # after; the outsider is never a member; everybody else always is.
    intervals = [
        (s, COVERAGE_START, None)
        for s in rebalances.columns
        if s not in (leaver, outsider)
    ] + [(leaver, COVERAGE_START, _day(first_held))]
    membership = _membership(tmp_path, intervals)

    def masked_backtester(name):
        return _backtester(
            tmp_path, dataset_config,
            MembershipMaskedPredictor(
                make_model(tmp_path / name, dataset_config, **dates), membership
            ),
            bars, name=name, checkpoint=checkpoint,
        )

    return {
        "baseline": baseline,
        "masked_backtester": masked_backtester,
        "leaver": leaver,
        "outsider": outsider,
        "first_held": first_held,
        "second_held": second_held,
        "dataset_config": dataset_config,
        "bars": bars,
        "dates": dates,
        "checkpoint": checkpoint,
        "membership": membership,
        "tmp_path": tmp_path,
    }


def test_a_leaver_is_not_selected_after_leaving_and_is_sold(scenario):
    result = scenario["masked_backtester"]("masked").run()
    leaver, outsider = scenario["leaver"], scenario["outsider"]
    first_held, second_held = scenario["first_held"], scenario["second_held"]
    weights = _rebalance_rows(result.weights)

    # Up to the leaving date the masked run decides like the baseline.
    assert weights.loc[first_held, leaver] > 0
    # After leaving it is not selected: the next rebalance targets zero.
    assert (weights.loc[weights.index > first_held, leaver] == 0).all()
    # The baseline would have kept it.
    assert _rebalance_rows(scenario["baseline"].weights).loc[second_held, leaver] > 0

    # Its prices were never masked, so the holding is sold by an order at
    # the open after the decision, not settled as a delisting.
    orders = result.simulation.orders.to_dataframe()
    sells = orders[(orders["symbol"] == leaver) & (orders["side"] == "Sell")]
    fill_bar = scenario["bars"][list(scenario["bars"]).index(np.datetime64(second_held)) + 1]
    assert pd.Timestamp(fill_bar) in set(pd.to_datetime(sells["timestamp"]))
    assert not any(s["symbol"] == leaver for s in result.simulation.settlements)
    assert not any(r["symbol"] == leaver for r in result.simulation.rejected_orders)

    # Never a member: never predicted, off the run's axis, never bought.
    assert outsider not in weights.columns
    assert not (orders["symbol"] == outsider).any()


def test_the_prediction_panel_carries_the_masked_predictions(scenario):
    result = scenario["masked_backtester"]("masked").run()
    leaver, outsider, first_held = (
        scenario["leaver"], scenario["outsider"], scenario["first_held"]
    )
    stored = BacktestRun.open(result.run_dir).predictions().predictions
    (name,) = stored.data_vars
    panel = stored[name].to_pandas()

    assert panel.loc[panel.index <= first_held, leaver].notna().all()
    assert panel.loc[panel.index > first_held, leaver].isna().all()
    assert outsider not in panel.columns
    members = [s for s in panel.columns if s != leaver]
    assert panel[members].notna().all().all()
    xr.testing.assert_identical(stored[name], result.predictions[name])


def test_the_masked_run_is_rebuilt_from_its_run_directory(scenario):
    first = scenario["masked_backtester"]("masked").run()
    run = BacktestRun.open(first.run_dir)

    # Recorded under its component path, its one variable only.
    (entry,) = run.data_fingerprint["model.membership"]
    assert entry["variables"] == ["is_member"]

    rebuilt = run.rebuild_backtester()

    assert type(rebuilt.config.model) is MembershipMaskedPredictor
    again = rebuilt.run()
    xr.testing.assert_identical(first.predictions, again.predictions)
    xr.testing.assert_identical(first.weights, again.weights)


def test_bars_outside_the_membership_panel_are_refused(scenario):
    tmp_path = scenario["tmp_path"]
    short = _membership(
        tmp_path / "short",
        [(s, COVERAGE_START, None) for s in ["AAA", "BBB"]],
        as_of=_day(scenario["first_held"]),
    )
    predictor = MembershipMaskedPredictor(
        make_model(tmp_path / "short_model", scenario["dataset_config"], **scenario["dates"]),
        short,
    )
    predictor.load(scenario["checkpoint"])
    bars = scenario["bars"]
    with pytest.raises(ValueError, match="membership"):
        predictor.predict_window(_day(bars[30]), _day(bars[50]))


def test_run_cv_masks_every_fold_and_the_stitched_predictions(tmp_path):
    from quantlab.backtest.predefined.us_equity import (
        USEquityCrossectionSelectStockVectorBt,
    )
    from quantlab.backtest.config import CrossSectionBacktestConfig
    from quantlab.portfolio.config import TopNConfig
    from quantlab.portfolio.predefined.top_n import TopNConstructor
    from tests.backtest_fixtures import make_stock_dataset, write_price_store

    dataset_config = write_price_store(tmp_path, n_bars=80)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    dates = dict(
        start_date=_day(bars[0]), end_date=_day(bars[79]),
        train_start=_day(bars[0]), train_end=_day(bars[29]),
        test_start=_day(bars[30]), test_end=_day(bars[79]),
    )
    trainer = make_model(tmp_path / "train", dataset_config, **dates)
    trainer.collect()
    project = trainer.train_cv(train_periods=30).path

    outsider = "AAA"
    membership = _membership(
        tmp_path,
        [(s, COVERAGE_START, None) for s in ["BBB", "CCC", "DDD", "EEE", "FFF"]],
        as_of="2024-06-28",
    )
    result = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=MembershipMaskedPredictor(
                make_model(tmp_path / "backtest", dataset_config, **dates), membership
            ),
            model_mode="load", cv_project_dir=str(project),
            start_date=_day(bars[30]), end_date=_day(bars[77]),
            output_dir=str(tmp_path / "runs"), rebalance_periods=2,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
            fees=0.0, slippage=0.0, init_cash=1_000_000.0,
        )
    ).run_cv()

    assert len(result.folds) > 1
    # Never a member: no fold predicts it, so the stitched run leaves it off its axis.
    for fold in result.folds:
        assert outsider not in fold["predictions"].symbol.values
    stored = BacktestRun.open(result.run_dir).predictions().predictions
    assert outsider not in stored.symbol.values
    assert stored.to_array().notnull().any()
    assert outsider not in result.weights.symbol.values
