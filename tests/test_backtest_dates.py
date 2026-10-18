"""Model preparation and date alignment in `BaseBacktester.run()` (phase 03.7, plan 06).

This module locks D-06, D-13, D-14 and D-15.

- **D-15, bar-accurate warm-up.** Each factor warms itself up: the
  backtester asks it for the window by date range and `compute` reads
  `warmup_bars` bars before the start, counted on the factor dataset's own
  calendar (#25). The warm-up test uses a 5-bar lookback, so a calendar-day
  count (5 days before a Monday reach back only 3 bars) leaves the first
  window bar NaN. A short history warns with the shortfall in bars.
- **D-14, date-range requests.** No factor config is re-dated and nothing is
  re-read with `overwrite=True`. The two strategy tests put the backtest
  window BEFORE the model's own start date: under "cal" the factor computes
  the window from its dataset, under "read" it reads the window from its
  store, and either way the window's first bar is predicted and traded.
- **D-06, the universe is every price symbol.** Predictions are reindexed onto
  the price dataset's symbol and timestamp axes before selection. A symbol the
  factor never saw scores NaN and gets 0.0 on every rebalance row. Predictions
  always come back symbol-sorted from `predict_panel`, so a price store written
  in a different symbol order has the same shape and a different order.
  Removing the reindex then either raises in the weights contract or selects
  by position. The alignment test checks both values and chosen names by
  symbol label.
- **D-13, model preparation.** "train" trains on the model's own
  train/test dates and never rewrites them. "load" refuses a missing file
  before any feature work. A torch head loads without any collected panel (its
  network depends only on the feature and label counts), and its window
  warm-up gives the window's first bar a full history.
- **Fold-style dates.** `'2026-08-07T00:00:00.000000000'`, numpy datetimes
  and timestamps with a time all normalize to an ISO date through one helper
  (RESEARCH Pitfall 10).

Everything is synthetic, CPU-only and offline. Configs are constructed
directly, never through the factories in `quantlab/config/__init__.py` (D-32).
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from loguru import logger

from quantlab.backtest.base import BaseBacktester
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.model.config import ModelConfig, ModelConfig
from quantlab.factor.config import PolarsFactorConfig
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.model.collected_panel import FEATURE_DTYPE
from tests.torch_heads import MeanContextHead
from tests.backtest_fixtures import (
    SYMBOLS,
    FirstFeatureHead,
    ForwardReturnLabel,
    PastReturnFactor,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
TOP_N = 2
REBALANCE_PERIODS = 5


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _bars(dataset_config) -> np.ndarray:
    return xr.open_zarr(dataset_config.zarr_file_path).timestamp.values


def _model_dates(bars, first: int, train_last: int, last: int) -> dict:
    return dict(
        start_date=_day(bars[first]),
        end_date=_day(bars[last]),
        train_start=_day(bars[first]),
        train_end=_day(bars[train_last]),
        test_start=_day(bars[train_last + 1]),
        test_end=_day(bars[last]),
    )


def _backtester(
    tmp_path: Path,
    price_config,
    model,
    bars,
    *,
    start_bar: int,
    end_bar: int,
    model_mode: str = "load",
    checkpoint=None,
) -> USEquityCrossectionSelectStockVectorBt:
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(price_config),
            model=model,
            model_mode=model_mode,
            checkpoint=None if checkpoint is None else str(checkpoint),
            start_date=_day(bars[start_bar]),
            end_date=_day(bars[end_bar]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=REBALANCE_PERIODS,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=TOP_N)),
            fees=0.0,
            slippage=0.0,
            init_cash=1_000_000.0,
        )
    )


def _loaded_model(tmp_path: Path, dataset_config, dates: dict, **model_kwargs):
    """A checkpoint trained by one model, and a fresh twin that will load it."""
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **dates, **model_kwargs)
    )
    model = make_model(tmp_path / "backtest", dataset_config, **dates, **model_kwargs)
    return model, checkpoint


def _adj_close(dataset_config) -> xr.DataArray:
    return (
        xr.open_zarr(dataset_config.zarr_file_path)["adjClose"]
        .load()
        .transpose("timestamp", "symbol")
    )


def _factor_configs(model) -> list[tuple]:
    """Each factor's config without its dataset object, and its dataset's config."""
    return [
        (
            {k: v for k, v in f.config.to_dict().items() if k != "dataset"},
            copy.deepcopy(f.config.dataset.config),
        )
        for f in model.config.factors
    ]


# --------------------------------------------------------------------------
# D-15: bar-accurate warm-up
# --------------------------------------------------------------------------


def test_warmup_counts_bars_on_the_price_calendar_not_calendar_days(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar = 30
    assert pd.Timestamp(bars[start_bar]).day_name() == "Monday"

    model, checkpoint = _loaded_model(
        tmp_path, dataset_config, _model_dates(bars, 0, 24, 29), n=5, warmup_bars=5
    )
    factor_configs = _factor_configs(model)
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=50, checkpoint=checkpoint,
    ).run()

    # The 5-bar lookback of the first window bar is fully inside the warm-up:
    # 5 bars before Monday 2024-02-12 is Monday 2024-02-05 (7 calendar days),
    # while 5 calendar days would reach only Wednesday 2024-02-07.
    first = result.predictions["fwd_ret_1"].isel(timestamp=0)
    assert np.isfinite(first.values).all(), first.values
    # Neither the factor config nor its dataset's config is touched.
    assert _factor_configs(model) == factor_configs


def test_warmup_warns_with_the_shortfall_in_bars_on_short_history(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)

    model, checkpoint = _loaded_model(
        tmp_path, dataset_config, _model_dates(bars, 0, 24, 29), warmup_bars=5
    )
    backtester = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=2, end_bar=20, checkpoint=checkpoint,
    )
    with pytest.warns(UserWarning) as caught:
        backtester.run()

    shortfall = [str(w.message) for w in caught if "warm-up" in str(w.message)]
    assert shortfall, [str(w.message) for w in caught]
    message = shortfall[0]
    assert "5 warm-up bar(s)" in message
    assert "only 2" in message
    assert "short by 3" in message


# --------------------------------------------------------------------------
# D-14: factor re-dating
# --------------------------------------------------------------------------


def _two_factor_model(root: Path, dataset_config, dates: dict) -> FirstFeatureHead:
    factors = [
        PastReturnFactor(
            PolarsFactorConfig(
                warmup_bars=warmup_bars,
                dataset=make_stock_dataset(dataset_config),
                kwargs={"n": n},
            )
        )
        for n, warmup_bars in ((1, 3), (2, 7))
    ]
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            warmup_bars=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": 1},
        )
    )
    return FirstFeatureHead(
        ModelConfig(
            factors=factors,
            labels=[label],
            model_save_dir=str(root / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            val_size=0.0,
            **dates,
        )
    )


def test_predictions_cover_exactly_the_window_without_redating_factors(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar, end_bar = 30, 50

    model = _two_factor_model(tmp_path, dataset_config, _model_dates(bars, 0, 24, 29))
    factor_configs = _factor_configs(model)
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=end_bar, model_mode="train",
    ).run()

    # Each factor warms itself up; no factor or dataset config is touched.
    assert _factor_configs(model) == factor_configs

    np.testing.assert_array_equal(
        result.predictions.timestamp.values.astype("datetime64[ns]"),
        bars[start_bar : end_bar + 1].astype("datetime64[ns]"),
    )
    # The model's own end date is bar 29; the last window bar is predicted
    # because the factors are asked for the backtest window.
    last = result.predictions["fwd_ret_1"].isel(timestamp=-1)
    assert np.isfinite(last.values).all(), last.values


def _strategy_model(tmp_path: Path, dataset_config, bars, strategy: str):
    """A model whose own dates (bars 35..59) start AFTER the backtest warm-up."""
    file_path = None
    if strategy == "read":
        file_path = str(tmp_path / "factors" / "past_ret.zarr")
        PastReturnFactor(
            PolarsFactorConfig(
                warmup_bars=5,
                dataset=make_stock_dataset(dataset_config),
                file_path=file_path,
                kwargs={"n": 1},
            )
        ).build(_day(bars[0]), _day(bars[-1]))

    factor = PastReturnFactor(
        PolarsFactorConfig(
            warmup_bars=5,
            dataset=make_stock_dataset(dataset_config),
            file_path=file_path,
            kwargs={"n": 1},
        )
    )
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            warmup_bars=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": 1},
        )
    )
    return FirstFeatureHead(
        ModelConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(tmp_path / "models"),
            factor_data_strategy=strategy,
            label_data_strategy="cal",
            val_size=0.0,
            **_model_dates(bars, 35, 49, 59),
        )
    )


def _assert_first_window_bar_is_predicted_and_traded(result, bars, start_bar):
    first = result.predictions["fwd_ret_1"].sel(timestamp=bars[start_bar])
    assert np.isfinite(first.values).all(), first.values
    first_rebalance = result.weights["weight"].isel(timestamp=0).values
    assert np.count_nonzero(first_rebalance) == TOP_N, first_rebalance


def test_read_strategy_predicts_over_the_widened_range_after_training(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar = 25

    model = _strategy_model(tmp_path, dataset_config, bars, "read")
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=45, model_mode="train",
    ).run()

    _assert_first_window_bar_is_predicted_and_traded(result, bars, start_bar)


def test_cal_strategy_predicts_over_the_widened_range_after_training(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar = 25

    model = _strategy_model(tmp_path, dataset_config, bars, "cal")
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=45, model_mode="train",
    ).run()

    _assert_first_window_bar_is_predicted_and_traded(result, bars, start_bar)


# --------------------------------------------------------------------------
# The run's axis is the symbols the rule decides on (D-06 kept every price
# symbol; a symbol without a prediction got 0.0 on every rebalance row)
# --------------------------------------------------------------------------


def test_a_symbol_absent_from_predictions_is_left_off_the_runs_axis(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    absent = SYMBOLS[-1]
    dates = _model_dates(bars, 0, 24, 29)

    _, checkpoint = _loaded_model(tmp_path, dataset_config, dates)
    narrowed = write_price_store(tmp_path / "narrowed", symbols=SYMBOLS[:-1], n_bars=N_BARS)
    model = make_model(tmp_path / "backtest", narrowed, **dates)

    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoint,
    ).run()

    assert result.weights.symbol.values.tolist() == SYMBOLS[:-1]
    assert absent not in result.predictions.symbol.values
    assert absent not in result.simulation.holdings.symbol.values
    rebalance = np.isfinite(result.weights["weight"].values).all(axis=1)
    assert rebalance.sum() == 4


def test_predictions_align_to_price_symbols_by_label_not_position(tmp_path):
    """Carried from plans 01 and 03: the former top-n selector paired
    scores with fill prices by shape only. `predict_panel` returns symbols
    sorted, so a price store written in reverse order has the same shape and
    a different order. Only the reindex onto `prices.symbol` keeps them paired.
    """
    reversed_symbols = list(reversed(SYMBOLS))
    dataset_config = write_price_store(tmp_path, symbols=reversed_symbols, n_bars=N_BARS)
    bars = _bars(dataset_config)
    start_bar = 30

    model, checkpoint = _loaded_model(
        tmp_path, dataset_config, _model_dates(bars, 0, 24, 29)
    )
    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=start_bar, end_bar=50, checkpoint=checkpoint,
    ).run()

    assert result.predictions.symbol.values.tolist() == reversed_symbols
    assert result.weights.symbol.values.tolist() == reversed_symbols

    adj_close = _adj_close(dataset_config)
    past_return = (
        adj_close.isel(timestamp=start_bar) / adj_close.isel(timestamp=start_bar - 1)
        - 1.0
    )
    for symbol in reversed_symbols:
        assert float(
            result.predictions["fwd_ret_1"]
            .isel(timestamp=0)
            .sel(symbol=symbol)
        ) == pytest.approx(float(past_return.sel(symbol=symbol).astype(FEATURE_DTYPE)), rel=1e-12)

    expected_top = set(past_return.to_series().nlargest(TOP_N).index)
    first_row = result.weights["weight"].isel(timestamp=0)
    chosen = {str(s) for s in first_row.symbol.values[first_row.values > 0]}
    assert chosen == expected_top


# --------------------------------------------------------------------------
# Pitfall 10: fold-style dates
# --------------------------------------------------------------------------


def test_iso_date_normalizes_fold_style_strings():
    fold_style = np.datetime_as_string(np.datetime64("2026-08-07", "ns"))
    assert fold_style == "2026-08-07T00:00:00.000000000"
    assert BaseBacktester._iso_date(fold_style) == "2026-08-07"
    assert BaseBacktester._iso_date(np.str_(fold_style)) == "2026-08-07"
    assert BaseBacktester._iso_date(np.datetime64("2026-08-07")) == "2026-08-07"
    assert BaseBacktester._iso_date(pd.Timestamp("2026-08-07 15:30")) == "2026-08-07"


# --------------------------------------------------------------------------
# D-13: model preparation
# --------------------------------------------------------------------------


class TinyLinearTorchHead(MeanContextHead):
    """A tiny cross-section head on a three-bar window (two bars of warm-up)."""


def _torch_model(root: Path, dataset_config, dates: dict) -> TinyLinearTorchHead:
    factor = PastReturnFactor(
        PolarsFactorConfig(
            warmup_bars=5, dataset=make_stock_dataset(dataset_config), kwargs={"n": 1}
        )
    )
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            warmup_bars=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": 1},
        )
    )
    return TinyLinearTorchHead(
        ModelConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(root / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            hyperparameters={"epochs": 1},
            val_size=0.0,
            **dates,
        )
    )


def test_train_mode_uses_the_models_own_dates_and_leaves_them_unchanged(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    dates = _model_dates(bars, 0, 24, 29)
    model = make_model(tmp_path, dataset_config, **dates)
    fields = ("train_start", "train_end", "test_start", "test_end")

    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, model_mode="train",
    ).run()

    assert {f: getattr(model.config, f) for f in fields} == {f: dates[f] for f in fields}
    checkpoints = sorted(Path(model.config.model_save_dir).rglob("*.joblib"))
    assert len(checkpoints) == 1, checkpoints
    first = result.predictions["fwd_ret_1"].isel(timestamp=0)
    assert np.isfinite(first.values).all(), first.values


def test_load_mode_with_missing_checkpoint_file_fails_before_predicting(
    tmp_path, monkeypatch
):
    """The existence check comes before any feature work, for a torch head too."""
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    model = _torch_model(tmp_path, dataset_config, _model_dates(bars, 0, 24, 29))
    missing = tmp_path / "never_trained" / "TinyLinearTorchHead_total.pth"

    calls = {"collect": 0, "predict_panel": 0}
    collect, predict_panel = model._collect_all_features, model.predict_panel

    def spy_collect(*args):
        calls["collect"] += 1
        return collect(*args)

    def spy_predict_panel(features):
        calls["predict_panel"] += 1
        return predict_panel(features)

    monkeypatch.setattr(model, "_collect_all_features", spy_collect)
    monkeypatch.setattr(model, "predict_panel", spy_predict_panel)

    backtester = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, checkpoint=missing,
    )
    with pytest.raises(FileNotFoundError, match="never_trained"):
        backtester.run()
    assert calls == {"collect": 0, "predict_panel": 0}


def test_load_mode_without_checkpoint_is_refused_at_construction(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    model = make_model(tmp_path, dataset_config, **_model_dates(bars, 0, 24, 29))

    with pytest.raises(ValueError, match="checkpoint"):
        _backtester(
            tmp_path, dataset_config, model, bars,
            start_bar=30, end_bar=50, model_mode="load", checkpoint=None,
        )


@pytest.mark.parametrize(
    "backtest_model_kwargs",
    [{"n": 2}, {"n_forward_periods": 3}],
    ids=["different-factor", "different-label"],
)
def test_load_refuses_a_checkpoint_trained_on_other_variables(
    tmp_path, monkeypatch, backtest_model_kwargs
):
    """Code review WR-01: a checkpoint is checked against config.model's variables.

    The checkpoint is trained on `past_ret_1` -> `fwd_ret_1`. The backtest
    model declares the same feature COUNT but a different factor (or a
    different label). The deterministic head only reads feature 0 by
    position, exactly like xgboost only checks the feature count, so the old
    `run()` predicted on the wrong inputs without complaint and this test
    went red. The refusal must name both variable lists and come before any
    feature is computed.
    """
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    dates = _model_dates(bars, 0, 24, 29)
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **dates))
    model = make_model(
        tmp_path / "backtest", dataset_config, **dates, **backtest_model_kwargs
    )
    computed: list[int] = []
    collect = model._collect_all_features
    monkeypatch.setattr(
        model, "_collect_all_features", lambda *a: computed.append(1) or collect(*a)
    )
    backtester = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoint,
    )

    with pytest.raises(ValueError, match="was trained on") as excinfo:
        backtester.run()

    message = str(excinfo.value)
    declared = model.get_factor_names() + model.get_label_names()
    assert any(name not in ("past_ret_1", "fwd_ret_1") and name in message for name in declared)
    assert str(checkpoint) in message
    assert computed == []


def test_load_uses_the_checkpoints_train_dates_over_stale_config_dates(
    tmp_path, warning_messages
):
    """Code review WR-01: D-17 classifies bars with the dates the checkpoint trained on.

    The checkpoint trained through bar 24 (fitted through bar 22 after the
    purge; its 2-bar label lookahead reads up to bar 24). The backtest model
    carries a stale `train_end` of bar 10. Trusting config.model, the old code
    put the training window end near bar 10 and reported the bars after it,
    which the model had trained on, as out-of-sample. It goes red here on the window, the in-sample range and the
    missing warning that names both date pairs.
    """
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **_model_dates(bars, 0, 24, 29))
    )
    stale = make_model(
        tmp_path / "backtest", dataset_config, **_model_dates(bars, 0, 10, 29)
    )

    result = _backtester(
        tmp_path, dataset_config, stale, bars,
        start_bar=20, end_bar=45, checkpoint=checkpoint,
    ).run()

    assert tuple(result.metrics["training_window"]) == (_day(bars[0]), _day(bars[24]))
    assert tuple(result.metrics["in_sample_range"]) == (_day(bars[20]), _day(bars[24]))
    stale_warnings = [
        m
        for m in warning_messages
        if "using the checkpoint's dates" in m and _day(bars[24]) in m and _day(bars[10]) in m
    ]
    assert len(stale_warnings) == 1, warning_messages


def test_load_refuses_a_checkpoint_without_run_json(tmp_path):
    """A checkpoint whose unit has no run.json records neither its training
    dates nor its variables, so it is refused with a message to retrain
    rather than backtested on config.model's dates."""
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    dates = _model_dates(bars, 0, 24, 29)
    model, checkpoint = _loaded_model(tmp_path, dataset_config, dates)
    (checkpoint.parent / "run.json").unlink()

    backtester = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoint,
    )
    with pytest.raises(ValueError, match="no run.json.*retrain"):
        backtester.run()


def _ns(bar) -> str:
    """The text shape `train_cv` writes: `np.datetime_as_string` of a ns bar."""
    return np.datetime_as_string(np.asarray(bar).astype("datetime64[ns]"))


def _ns_model_dates(bars, first: int, train_last: int, last: int) -> dict:
    dates = _model_dates(bars, first, train_last, last)
    dates.update(
        train_start=_ns(bars[first]),
        train_end=_ns(bars[train_last]),
        test_start=_ns(bars[train_last + 1]),
        test_end=_ns(bars[last]),
    )
    return dates


def _t00_model_dates(bars, first: int, train_last: int, last: int) -> dict:
    dates = _model_dates(bars, first, train_last, last)
    for key in ("train_start", "train_end", "test_start", "test_end"):
        dates[key] = dates[key] + "T00:00:00"
    return dates


@pytest.mark.parametrize(
    ("checkpoint_dates", "config_dates"),
    [(_ns_model_dates, _model_dates), (_model_dates, _t00_model_dates)],
    ids=["ns-checkpoint-vs-plain-config", "plain-checkpoint-vs-T00-config"],
)
def test_load_mode_same_train_dates_in_another_text_format_log_no_date_warning(
    tmp_path, warning_messages, checkpoint_dates, config_dates
):
    """UAT gap G-03.7-7: the same training bars written differently are the same window.

    The checkpoint's config.json records one text form of the dates (a train_cv
    nanosecond string, or a plain date) and config.model carries another form
    of the SAME instants. The old string comparison warned "using the
    checkpoint's dates" for both. On daily bars both forms select the same
    training bars, so nothing may warn, and the numbers stay as they were.
    """
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **checkpoint_dates(bars, 0, 24, 29))
    )
    model = make_model(
        tmp_path / "backtest", dataset_config, **config_dates(bars, 0, 24, 29)
    )

    result = _backtester(
        tmp_path, dataset_config, model, bars,
        start_bar=20, end_bar=45, checkpoint=checkpoint,
    ).run()

    stale = [m for m in warning_messages if "using the checkpoint's dates" in m]
    assert stale == [], stale
    assert tuple(result.metrics["training_window"]) == (_day(bars[0]), _day(bars[24]))


def test_same_training_bars_resolves_endpoints_on_the_calendar():
    """G-03.7-7: two date pairs match iff they select the same bars (slice semantics).

    Bare Timestamp equality is not enough: on intraday bars "2024-02-09"
    includes the whole day while the nanosecond midnight stops at the previous
    session's last bar. Endpoints past the calendar are compared as instants,
    so a stale date that clips to the same last bar still counts as different.
    """
    same = BaseBacktester._same_training_bars

    sessions = [
        pd.date_range(f"{day} 09:30", f"{day} 16:00", freq="30min")
        for day in ("2024-02-07", "2024-02-08", "2024-02-09")
    ]
    intraday = np.concatenate([s.values for s in sessions]).astype("datetime64[ns]")
    assert not same(
        intraday,
        ("2024-02-07", "2024-02-09"),
        ("2024-02-07", "2024-02-09T00:00:00.000000000"),
    )

    daily = pd.bdate_range("2024-01-01", "2024-03-29").values.astype("datetime64[ns]")
    plain = ("2024-01-01", "2024-02-09")
    assert same(
        daily, plain, ("2024-01-01T00:00:00.000000000", "2024-02-09T00:00:00.000000000")
    )
    assert same(daily, plain, ("2024-01-01T00:00:00", "2024-02-09T00:00:00"))
    assert not same(daily, plain, ("2024-01-01", "2024-02-08"))

    # Both train_end endpoints lie after the calendar's last bar (2024-03-29).
    assert same(
        daily, ("2024-01-01", "2024-04-05"), ("2024-01-01", "2024-04-05T00:00:00.000000000")
    )
    assert not same(daily, ("2024-01-01", "2024-04-05"), ("2024-01-01", "2024-04-08"))

    assert same(daily, (None, "2024-02-09"), (None, "2024-02-09"))
    assert not same(daily, plain, (None, "2024-02-09"))


def test_torch_head_loads_without_a_collected_panel_and_predicts_the_first_bar(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    trainer = _torch_model(tmp_path, dataset_config, _model_dates(bars, 0, 24, 29))
    trainer.collect()
    trainer.train()
    checkpoints = sorted(Path(trainer.config.model_save_dir).rglob("*.pth"))
    assert len(checkpoints) == 1, checkpoints

    fresh = TinyLinearTorchHead(trainer.config)
    assert fresh.model is None
    result = _backtester(
        tmp_path, dataset_config, fresh, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoints[0],
    ).run()

    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(fresh.model.state_dict()[name].cpu(), tensor.cpu()), name
    first = result.predictions["fwd_ret_1"].isel(timestamp=0)
    assert np.isfinite(first.values).all(), first.values


def _trained_torch_checkpoint(tmp_path: Path, dataset_config, bars) -> tuple[TinyLinearTorchHead, Path]:
    trainer = _torch_model(tmp_path, dataset_config, _model_dates(bars, 0, 24, 29))
    trainer.collect()
    trainer.train()
    checkpoints = sorted(Path(trainer.config.model_save_dir).rglob("*.pth"))
    assert len(checkpoints) == 1, checkpoints
    return trainer, checkpoints[0]


def test_backtester_delegates_the_variable_check_to_the_model(tmp_path, monkeypatch):
    """G-03.7-9: one variable check, in the model layer, keyed on `trained_on`.

    The backtester used to carry its own comparison against the factor config
    field `factors[].factor_names`. It refused a model's own checkpoint when
    that field was ordered differently from the derived names, and accepted
    permuted inputs when the derived names had drifted. It must be gone, and
    `_load_model_checkpoint` must call `model.check_checkpoint` before
    any feature collection, so a refusal still precedes any feature work. The old code has both attributes and never calls the model check
    before collection, so this goes red.
    """
    assert not hasattr(BaseBacktester, "_assert_checkpoint_variables")
    assert not hasattr(BaseBacktester, "_saved_variable_names")

    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    trainer, checkpoint = _trained_torch_checkpoint(tmp_path, dataset_config, bars)
    fresh = TinyLinearTorchHead(trainer.config)
    events: list[str] = []
    check, collect = fresh.check_checkpoint, fresh._collect_all_features
    monkeypatch.setattr(
        fresh, "check_checkpoint", lambda p: events.append("check") or check(p)
    )
    monkeypatch.setattr(
        fresh, "_collect_all_features", lambda *a: events.append("collect") or collect(*a)
    )

    _backtester(
        tmp_path, dataset_config, fresh, bars,
        start_bar=30, end_bar=50, checkpoint=checkpoint,
    ).run()

    assert "check" in events and "collect" in events, events
    assert events.index("check") < events.index("collect"), events


