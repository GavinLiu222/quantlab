"""`BaseEnsemble` combines each label over the members that predict it (ADR 0013).

What is locked here, and what turns it red:

- `labels` is the union of the members' labels in first-appearance order; a
  label predicted by several members is averaged (per-bar z-score, NaN-ignoring
  mean), one predicted by a single member is passed through unchanged.
- `label_scales`: a model is `raw` exactly when its training-target transform
  is the identity; an ensemble reports `standardized` for an averaged label
  and the member's own scale for a passed-through one.
- A same-name label with a different config is refused at construction,
  naming the member.
- The training end is the latest member's, the test window the intersection
  of the members'.
- `train_cv` lays out its folds with the largest lookahead among the
  members, every member purges its own window by its own lookahead, and a
  fold unit records what each member fitted and, for the ensemble, the
  window covering them (#123).
- The evaluation files score each label against the truth of a member that
  predicts it; `member_correlation` is reported only for labels with at least
  two members.
- An ensemble whose every label one member predicts reports, label by label,
  the same metric keys and values as that member does on its own (one
  Evaluation, #143): error metrics included for a raw label, the training
  loss excepted.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from quantlab.factor.config import PolarsFactorConfig
from quantlab.model.predefined.model_ensemble import ModelEnsemble
from quantlab.model.ensemble import average_predictions
from quantlab.model.evaluation import ic_panel_metrics
from quantlab.runs.trained_run import TrainedRun
from tests.backtest_fixtures import (
    FirstFeatureHead,
    ForwardReturnLabel,
    SeededHead,
    make_model,
    make_stock_dataset,
    write_price_store,
)

N_BARS = 60


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


class RankTargetHead(FirstFeatureHead):
    """A head trained on each bar's cross-sectional rank: not in the label's units."""

    def _transform_target(self, y: torch.Tensor, training: bool):
        return torch.argsort(torch.argsort(y, dim=0), dim=0).float(), None


@pytest.fixture
def setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values

    def member(
        name, *, head=FirstFeatureHead, n=1, horizon=1, train_end=24, test=(25, 29),
        hyperparameters=None,
    ):
        return make_model(
            tmp_path / name,
            dataset_config,
            head=head,
            n=n,
            n_forward_periods=horizon,
            start_date=_day(bars[0]),
            end_date=_day(bars[29]),
            train_start=_day(bars[0]),
            train_end=_day(bars[train_end]),
            test_start=_day(bars[test[0]]),
            test_end=_day(bars[test[1]]),
            hyperparameters=hyperparameters,
        )

    return member, bars, dataset_config


def test_a_model_reports_raw_for_an_identity_training_target(setup):
    member, _, _ = setup

    assert member("a").label_scales == {"fwd_ret_1": "raw"}
    assert member("b", head=RankTargetHead).label_scales == {"fwd_ret_1": "standardized"}


def test_disjoint_labels_are_both_exposed_and_passed_through_exactly(setup):
    member, bars, _ = setup
    ret, vol = member("ret"), member("vol", head=SeededHead, n=3, horizon=2)
    ensemble = ModelEnsemble([ret, vol])

    assert [label.get_factor_names() for label in ensemble.labels] == [
        ("fwd_ret_1",),
        ("fwd_ret_2",),
    ]
    assert ensemble.label_delays == (1, 1)
    assert ensemble.label_scales == {"fwd_ret_1": "raw", "fwd_ret_2": "raw"}

    ensemble.collect().train()
    start, end = _day(bars[25]), _day(bars[29])
    out = ensemble.predict_window(start, end)

    assert list(out.data_vars) == ["fwd_ret_1", "fwd_ret_2"]
    for model, name in ((ret, "fwd_ret_1"), (vol, "fwd_ret_2")):
        own = model.predict_window(start, end)[name]
        np.testing.assert_array_equal(
            out[name].sel(timestamp=own.timestamp, symbol=own.symbol).values, own.values
        )


def test_a_passed_through_label_keeps_its_members_scale(setup):
    member, _, _ = setup
    ensemble = ModelEnsemble([member("a"), member("b", head=RankTargetHead, horizon=2)])

    assert ensemble.label_scales == {"fwd_ret_1": "raw", "fwd_ret_2": "standardized"}


def test_a_rank_target_return_member_and_a_raw_volatility_member_keep_their_scales(setup):
    """The hyperparameter, not a subclass, makes the return member standardized;
    the other member, without it, stays raw (issue #95)."""
    member, _, _ = setup
    ret = member("ret", hyperparameters={"training_target": "cs_rank"})
    vol = member("vol", horizon=2)
    ensemble = ModelEnsemble([ret, vol])

    assert ensemble.label_scales == {"fwd_ret_1": "standardized", "fwd_ret_2": "raw"}


def test_mixed_overlap_averages_the_shared_label_and_passes_the_other(setup):
    member, bars, _ = setup
    a, b = member("a"), member("b", head=SeededHead, n=3)
    c = member("c", n=2, horizon=2)
    ensemble = ModelEnsemble([a, b, c])

    assert ensemble.label_scales == {"fwd_ret_1": "standardized", "fwd_ret_2": "raw"}

    ensemble.collect().train()
    start, end = _day(bars[25]), _day(bars[29])
    out = ensemble.predict_window(start, end)

    shared = average_predictions(
        [a.predict_window(start, end), b.predict_window(start, end)]
    )
    xr.testing.assert_allclose(out[["fwd_ret_1"]], shared.sel(timestamp=slice(start, end)))
    own = c.predict_window(start, end)["fwd_ret_2"]
    np.testing.assert_array_equal(
        out["fwd_ret_2"].sel(timestamp=own.timestamp, symbol=own.symbol).values, own.values
    )


def test_a_same_name_label_with_a_different_config_is_refused(setup):
    member, _, dataset_config = setup
    a, b = member("a"), member("b")
    other = ForwardReturnLabel(
        PolarsFactorConfig(
            warmup_bars=3,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": 1},
        )
    )
    b.config = dataclasses.replace(b.config, labels=[other])

    with pytest.raises(ValueError, match=r"member 1 \(FirstFeatureHead\).*fwd_ret_1"):
        ModelEnsemble([a, b])


def test_train_end_is_the_latest_and_test_window_the_intersection(setup):
    member, bars, _ = setup
    early = member("early", train_end=22, test=(23, 28))
    late = member("late", head=SeededHead, n=3, train_end=24, test=(25, 29))

    ensemble = ModelEnsemble([early, late])

    assert ensemble.train_bounds == (_day(bars[0]), _day(bars[24]))
    assert ensemble.test_bounds == (_day(bars[25]), _day(bars[28]))


def test_disjoint_test_windows_are_refused(setup):
    member, _, _ = setup
    with pytest.raises(ValueError, match="test window"):
        ModelEnsemble([member("a", train_end=10, test=(11, 14)), member("b", test=(25, 29))])


def test_train_cv_folds_record_what_each_member_fitted(setup):
    member, bars, _ = setup
    short = member("short")  # lookahead 2
    long = member("long", n=2, horizon=3)  # lookahead 4
    calendar = [pd.Timestamp(b) for b in bars]

    def gap(window, test_window):
        return calendar.index(pd.Timestamp(str(test_window[0]))) - calendar.index(
            pd.Timestamp(str(window[1]))
        )

    run = ModelEnsemble([short, long]).collect().train_cv(train_periods=10)

    assert run.folds
    for fold in run.folds:
        short_fit, long_fit = (m.fitted_train_window for m in fold.members)
        assert gap(short_fit, fold.test_window) == 1 + 2
        assert gap(long_fit, fold.test_window) == 1 + 4
        # The ensemble's fitted window covers its members': no member
        # fitted a bar after its end.
        assert gap(fold.fitted_train_window, fold.test_window) == 1 + 2


def _metrics(checkpoint) -> dict:
    return TrainedRun.open(checkpoint).metrics


def _expected_ic(model, name, stamps) -> float:
    data = model.collected_panel.data
    own = model.predict_panel(data)
    expected, _ = ic_panel_metrics(
        own[name].sel(timestamp=stamps).values,
        data[name].sel(timestamp=stamps).values,
        return_series=True,
    )
    return expected["ic"]


def test_evaluation_scores_each_label_against_its_own_truth(setup):
    member, _, _ = setup
    ret, vol = member("ret"), member("vol", n=3, horizon=2)
    ensemble = ModelEnsemble([ret, vol])

    checkpoint = ensemble.collect().train()
    metrics = _metrics(checkpoint)
    saved = xr.open_zarr(TrainedRun.open(checkpoint).test_predictions).load()

    assert set(saved.data_vars) == {"fwd_ret_1", "fwd_ret_2"}
    assert not any("member_correlation" in key for key in metrics)
    ret_stamps = ret.evaluation_segments().test
    vol_stamps = vol.evaluation_segments().test
    assert metrics["test_ic"] == pytest.approx(_expected_ic(ret, "fwd_ret_1", ret_stamps))
    assert metrics["test_fwd_ret_2_ic"] == pytest.approx(
        _expected_ic(vol, "fwd_ret_2", vol_stamps)
    )


def test_member_correlation_is_reported_only_for_shared_labels(setup):
    member, _, _ = setup
    ensemble = ModelEnsemble(
        [member("a"), member("b", head=SeededHead, n=3), member("c", n=2, horizon=2)]
    )

    metrics = _metrics(ensemble.collect().train())

    assert "test_member_correlation" in metrics
    assert "train_member_correlation" in metrics
    assert not any(key.endswith("fwd_ret_2_member_correlation") for key in metrics)


def test_a_one_member_per_label_ensemble_reports_its_members_metrics(setup):
    member, _, _ = setup
    ret, vol = member("ret"), member("vol", head=SeededHead, n=3, horizon=2)
    ensemble = ModelEnsemble([ret, vol])

    checkpoint = ensemble.collect().train()
    metrics = _metrics(checkpoint)
    run = TrainedRun.open(checkpoint)
    own = [
        {k: v for k, v in saved.metrics.items() if not k.endswith("_loss")}
        for saved in run.members
    ]

    assert "test_mse" in own[0] and "test_mse" in own[1]
    expected = {
        **own[0],
        **{
            key.replace("_", "_fwd_ret_2_", 1): value
            for key, value in own[1].items()
        },
    }
    assert metrics == expected
