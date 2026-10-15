"""`BaseBacktester.run_cv()`, the model-CV backtest (phase 03.7, plan 10).

`run_cv` answers "how well does the model's cross-validation actually trade".
It opens the walk-forward unit a `train_cv` run wrote (through `TrainedRun`,
#123), backtests every fold's out-of-sample test segment with that fold's own
checkpoint, and stitches the segments into one out-of-sample curve.

What is locked here, and what turns it red:

- **The walk-forward unit is a versioned persisted format.** A missing
  directory, an old layout without `run.json`, an unknown `format_version`, a
  unit of another kind and an empty fold list are each refused with a message
  naming the problem. A reader that guesses at an unknown version would
  silently misread an old or future training run.
- **D-16, one checkpoint and one test segment per fold.** Each fold loads its
  own checkpoint, in fold order, and its weights cover exactly its own test
  bars. A fold that trades bars outside its test segment trades data its
  model was trained on. A trial directory copied elsewhere backtests from its
  new location, never from the working directory.
- **D-17 per fold.** In/out-of-sample is decided with THAT fold's train dates.
  `train_cv` purges the last 2 bars of every training window for the 2-bar
  label (issue #34), so the last fitted label reads up to the bar before the
  test segment: every test bar is out-of-sample and no fold warns about an
  overlap. Before the purge the first two test bars of every fold were
  in-sample.
- **D-35, contiguity before stitching.** The selected folds' test segments must
  follow each other bar for bar on the price calendar. A gap or an overlap is
  refused before any checkpoint is loaded or any simulation runs. Stitching
  across a gap would silently drop the uncovered bars from the "out-of-sample"
  curve; stitching an overlap would trade some bars twice with two different
  models. Either way the stitched curve would describe no real trading path.
- **Fold selection.** Only folds whose test segment meets the config's
  backtest window are run, and a fold straddling the window's start or end is
  cut to it (#301), so a run_cv can start from cash on any bar.
- **D-35, the stitched curve.** Its weights come from one pass over the
  concatenated fold predictions and it is ONE continuous simulation, so
  holdings and capital carry across fold boundaries (#90). The portfolio is
  decided and simulated once: a fold is that account over its test segment,
  never a second account started from cash (ADR 0033), and its metrics chain
  to the stitched total.
- **D-24 / D-35, the run directory.** The top-level artifacts describe the
  stitched curve, and each fold is a child run of kind `fold` (`BacktestRun.folds`,
  #133) holding its metrics and what its prediction read; its weights, equity,
  holdings and settlements are the run's, cut to its window. Results are
  read through `BacktestRun`; run file names appear only where the subject is
  the layout itself (the directory listing) or a record is edited to check a
  refusal (`_edited_project`).

Everything is synthetic, CPU-only and offline. Configs are constructed
directly, never through the factories in `quantlab/config/__init__.py` (D-32).
"""

import dataclasses
import json
import shutil
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

import quantlab.backtest.engine_vectorbt as engine_module
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.base import PortfolioConstructor
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun, Market
from quantlab.utils.date_range import bar_label
from tests.decision_fixtures import schedule
from tests.backtest_fixtures import (
    make_model,
    make_stock_dataset,
    write_price_store,
)

N_BARS = 80
TRAIN_PERIODS = 30
#: `walk_forward_folds`: test_periods = 30 // 5 = 6, (80 - 30) // 6 = 8 folds, whose
#: test segments cover bars 30..77 with no gap between them.
TEST_PERIODS = 6
N_FOLDS = 8
FIRST_TEST_BAR = TRAIN_PERIODS
LAST_TEST_BAR = TRAIN_PERIODS + N_FOLDS * TEST_PERIODS - 1
HORIZON = 2
REBALANCE_PERIODS = 2
TOP_N = 2
INIT_CASH = 1_000_000.0

OVERLAP_WARNING = "overlaps the model's effective training window"

_UNSET = object()


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _model_dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[N_BARS - 1]),
        train_start=_day(bars[0]),
        train_end=_day(bars[TRAIN_PERIODS - 1]),
        test_start=_day(bars[TRAIN_PERIODS]),
        test_end=_day(bars[N_BARS - 1]),
    )


@pytest.fixture(scope="module")
def cv_project(tmp_path_factory):
    """One real `train_cv` run, shared read-only by every test in the module.

    Tests never write into this directory: an edited unit is a copy in the
    test's own tmp_path.
    """
    root = tmp_path_factory.mktemp("cv_project")
    dataset_config = write_price_store(root, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    model = make_model(
        root / "train",
        dataset_config,
        n_forward_periods=HORIZON,
        **_model_dates(bars),
    )
    model.collect()
    run = model.train_cv(train_periods=TRAIN_PERIODS)

    assert len(run.folds) == N_FOLDS
    return types.SimpleNamespace(
        root=root,
        dataset_config=dataset_config,
        bars=bars,
        project_dir=run.path,
        run=run,
        checkpoints=[str(fold.checkpoint) for fold in run.folds],
    )


def _test_bars(cv, fold: int) -> np.ndarray:
    first = FIRST_TEST_BAR + fold * TEST_PERIODS
    return cv.bars[first : first + TEST_PERIODS].astype("datetime64[ns]")


def _backtester(
    tmp_path: Path,
    cv,
    *,
    cv_project_dir=_UNSET,
    checkpoint=None,
    start_bar: int = FIRST_TEST_BAR,
    end_bar: int = LAST_TEST_BAR,
    sizing_basis: str = "fill",
) -> USEquityCrossectionSelectStockVectorBt:
    project_dir = cv.project_dir if cv_project_dir is _UNSET else cv_project_dir
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(cv.dataset_config),
            model=make_model(
                tmp_path / "backtest",
                cv.dataset_config,
                n_forward_periods=HORIZON,
                **_model_dates(cv.bars),
            ),
            model_mode="load",
            checkpoint=None if checkpoint is None else str(checkpoint),
            cv_project_dir=None if project_dir is None else str(project_dir),
            start_date=_day(cv.bars[start_bar]),
            end_date=_day(cv.bars[end_bar]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=REBALANCE_PERIODS,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=TOP_N)),
            fees=0.0,
            slippage=0.0,
            init_cash=INIT_CASH,
            sizing_basis=sizing_basis,
        )
    )


def _edited_project(tmp_path: Path, cv, edit) -> Path:
    """Copy the walk-forward unit into tmp_path and ``edit`` its ``run.json``."""
    project = tmp_path / "edited_project"
    shutil.copytree(cv.project_dir, project)
    path = project / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return project


def _spy_load(monkeypatch, backtester) -> list[str]:
    model = backtester.config.model
    real_load = model.load
    loaded: list[str] = []

    def _spy(p):
        loaded.append(str(p))
        return real_load(p)

    monkeypatch.setattr(model, "load", _spy)
    return loaded


def _spy_from_orders(monkeypatch) -> list[pd.Index]:
    real_from_orders = engine_module.vbt.Portfolio.from_orders
    calls: list[pd.Index] = []

    def _spy(*args, **kwargs):
        calls.append(kwargs["close"].index)
        return real_from_orders(*args, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "vbt",
        types.SimpleNamespace(Portfolio=types.SimpleNamespace(from_orders=_spy)),
    )
    return calls


# --------------------------------------------------------------------------
# The walk-forward unit reader refuses bad input
# --------------------------------------------------------------------------


def test_run_cv_refuses_a_missing_project_dir(tmp_path, cv_project):
    missing = tmp_path / "not_a_cv_project"
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=missing)

    with pytest.raises(FileNotFoundError) as excinfo:
        backtester.run_cv()
    assert str(missing) in str(excinfo.value)


def test_run_cv_refuses_an_old_layout_and_says_to_retrain(tmp_path, cv_project):
    """A pre-#123 trial directory has cv_folds.json and no run.json."""
    old = tmp_path / "old_project"
    old.mkdir()
    (old / "cv_folds.json").write_text(json.dumps({"format_version": 2, "folds": []}))
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=old)

    with pytest.raises(ValueError, match=r"no run.json.*retrain"):
        backtester.run_cv()


def test_run_cv_refuses_an_unknown_format_version(tmp_path, cv_project):
    project = _edited_project(
        tmp_path, cv_project, lambda payload: payload.update(format_version=99)
    )
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=project)

    with pytest.raises(ValueError, match=r"format_version 99.*retrain"):
        backtester.run_cv()


def test_run_cv_refuses_a_unit_that_is_not_a_walk_forward_run(tmp_path, cv_project):
    fold_unit = cv_project.project_dir / "fold_0"
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=fold_unit)

    with pytest.raises(ValueError, match=r"'model' trained run.*walk-forward"):
        backtester.run_cv()


def test_run_cv_refuses_an_empty_fold_list(tmp_path, cv_project):
    project = _edited_project(tmp_path, cv_project, lambda payload: payload.update(folds=[]))
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=project)

    with pytest.raises(ValueError, match=r"no folds"):
        backtester.run_cv()


def test_run_requires_a_checkpoint_and_run_cv_requires_a_project_dir(
    tmp_path, cv_project
):
    """model_mode="load" needs a checkpoint (run) or a cv_project_dir (run_cv),
    and each entry point refuses when its own input is the one missing."""
    only_project = _backtester(tmp_path / "a", cv_project)
    with pytest.raises(ValueError, match=r"run\(\).*checkpoint"):
        only_project.run()

    checkpoint = cv_project.checkpoints[0]
    only_checkpoint = _backtester(
        tmp_path / "b", cv_project, cv_project_dir=None, checkpoint=checkpoint
    )
    with pytest.raises(ValueError, match=r"run_cv\(\).*cv_project_dir"):
        only_checkpoint.run_cv()

    with pytest.raises(ValueError, match=r"checkpoint.*cv_project_dir"):
        _backtester(tmp_path / "c", cv_project, cv_project_dir=None)


# --------------------------------------------------------------------------
# D-16 / D-17: per-fold backtests
# --------------------------------------------------------------------------


def test_each_fold_loads_its_own_checkpoint_and_trades_only_its_test_segment(
    tmp_path, cv_project, monkeypatch
):
    backtester = _backtester(tmp_path, cv_project)
    loaded = _spy_load(monkeypatch, backtester)

    result = backtester.run_cv()

    assert loaded == cv_project.checkpoints
    assert [record["fold"] for record in result.folds] == list(range(N_FOLDS))
    for record in result.folds:
        fold = record["fold"]
        np.testing.assert_array_equal(
            record["predictions"].timestamp.values.astype("datetime64[ns]"),
            _test_bars(cv_project, fold),
        )
        assert record["checkpoint"] == cv_project.checkpoints[fold]


def test_a_copied_trial_directory_backtests_from_its_new_location(
    tmp_path, cv_project, monkeypatch
):
    """A trial directory copied elsewhere (as from the training server) opens and
    backtests from its new location, and the working directory plays no part:
    the original is renamed away and the process runs from an unrelated
    directory."""
    moved = tmp_path / "moved" / cv_project.project_dir.name
    shutil.copytree(cv_project.project_dir, moved)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    hidden = cv_project.project_dir.with_name(cv_project.project_dir.name + "_hidden")
    cv_project.project_dir.rename(hidden)
    try:
        backtester = _backtester(tmp_path, cv_project, cv_project_dir=moved)
        loaded = _spy_load(monkeypatch, backtester)
        result = backtester.run_cv()
    finally:
        hidden.rename(cv_project.project_dir)

    expected = [
        str(moved / Path(path).relative_to(cv_project.project_dir))
        for path in cv_project.checkpoints
    ]
    assert loaded == expected
    assert [record["checkpoint"] for record in result.folds] == expected


def test_a_failed_cv_persist_leaves_no_run_directory(tmp_path, cv_project, monkeypatch):
    """Code review WR-08: an interrupted `run_cv` persist leaves nothing that looks like a run.

    `_persist_cv` used to create `output_dir/{class}_{ts}/` and write
    config.json first, then the zarr stores, `folds/`, metrics and the report.
    A failure in any later step (a zarr error, plotly, a full disk, Ctrl-C)
    left a directory holding a valid config.json but no metrics or
    fingerprint. Nobody could tell it from a finished run, and
    `rebuild` would happily "reproduce" it. Report
    writing is made to fail here: `output_dir` must end up empty, with no
    final directory and no staging leftover. Red on the old code.
    """
    import quantlab.backtest.base as backtest_module

    def _fail(*args, **kwargs):
        raise RuntimeError("simulated failure while writing report.html")

    monkeypatch.setattr(backtest_module, "write_backtest_report", _fail)
    backtester = _backtester(tmp_path, cv_project)

    with pytest.raises(RuntimeError, match="simulated failure"):
        backtester.run_cv()

    runs = tmp_path / "runs"
    leftovers = sorted(p.name for p in runs.iterdir()) if runs.exists() else []
    assert leftovers == []


def test_per_fold_split_uses_the_folds_own_train_dates(
    tmp_path, cv_project, warning_messages
):
    result = _backtester(tmp_path, cv_project).run_cv()

    assert len(result.folds) == N_FOLDS
    for record in result.folds:
        bars = _test_bars(cv_project, record["fold"])
        metrics = record["metrics"]
        assert metrics["in_sample_range"] is None
        assert [tuple(r) for r in metrics["out_of_sample_ranges"]] == [
            (_day(bars[0]), _day(bars[-1]))
        ]
        # The purged training end plus the 2-bar label reaches the bar just
        # before the fold's first test bar.
        assert tuple(metrics["training_window"])[1] == _day(
            cv_project.bars[np.flatnonzero(cv_project.bars == bars[0])[0] - 1]
        )

    overlaps = [m for m in warning_messages if OVERLAP_WARNING in m]
    assert overlaps == []


CHECKPOINT_DATES_WARNING = "using the checkpoint's dates"


def test_a_normal_run_cv_logs_no_checkpoint_date_warning(
    tmp_path, cv_project, warning_messages
):
    """UAT gap G-03.7-7: every fold trains on its own window, so comparing each
    fold with config.model's single training window used to log one "using the
    checkpoint's dates" warning per fold. Each fold's window comes from its own
    record, so nothing warns."""
    _backtester(tmp_path, cv_project).run_cv()

    checkpoint_dates = [m for m in warning_messages if CHECKPOINT_DATES_WARNING in m]
    assert checkpoint_dates == []


# --------------------------------------------------------------------------
# D-35: contiguity is asserted before anything runs
# --------------------------------------------------------------------------


def test_non_contiguous_folds_are_refused_before_any_simulation(
    tmp_path, cv_project, monkeypatch
):
    folds = cv_project.run.folds
    project = _edited_project(
        tmp_path, cv_project, lambda payload: payload["folds"].pop(3)
    )
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=project)
    calls = _spy_from_orders(monkeypatch)
    loaded = _spy_load(monkeypatch, backtester)

    with pytest.raises(ValueError, match=r"gap") as excinfo:
        backtester.run_cv()

    message = str(excinfo.value)
    assert _day(folds[2].test_window[1]) in message
    assert _day(folds[4].test_window[0]) in message
    assert calls == []
    assert loaded == []


def test_overlapping_folds_are_refused(tmp_path, cv_project, monkeypatch):
    folds = cv_project.run.folds
    moved_end = FIRST_TEST_BAR + 3 * TEST_PERIODS  # one bar past fold 2's end
    new_end = np.datetime_as_string(cv_project.bars[moved_end])
    project = _edited_project(
        tmp_path,
        cv_project,
        lambda payload: payload["folds"][2]["test_window"].__setitem__(1, new_end),
    )
    # The fold's own record must agree with the walk-forward record's copy.
    fold_record = project / "fold_2" / "run.json"
    payload = json.loads(fold_record.read_text(encoding="utf-8"))
    payload["test_window"][1] = new_end
    fold_record.write_text(json.dumps(payload), encoding="utf-8")
    backtester = _backtester(tmp_path, cv_project, cv_project_dir=project)
    calls = _spy_from_orders(monkeypatch)

    with pytest.raises(ValueError, match=r"overlap") as excinfo:
        backtester.run_cv()

    message = str(excinfo.value)
    assert _day(new_end) in message
    assert _day(folds[3].test_window[0]) in message
    assert calls == []


def test_folds_outside_the_config_window_are_skipped(
    tmp_path, cv_project, monkeypatch
):
    first_kept = N_FOLDS - 3
    backtester = _backtester(
        tmp_path,
        cv_project,
        start_bar=FIRST_TEST_BAR + first_kept * TEST_PERIODS,
        end_bar=LAST_TEST_BAR,
    )
    loaded = _spy_load(monkeypatch, backtester)

    result = backtester.run_cv()

    assert [record["fold"] for record in result.folds] == list(range(first_kept, N_FOLDS))
    assert loaded == cv_project.checkpoints[first_kept:]


def test_a_fold_straddling_the_window_is_cut_to_it_and_the_account_starts_from_cash(
    tmp_path, cv_project, monkeypatch
):
    """#301: a start inside fold 2's test segment and an end inside fold 5's.

    Fold 2 trades from the start bar and fold 5 up to the end bar, each with its
    own checkpoint; the account starts from cash on the start bar, and on the
    bars both runs trade the cut folds predict what the whole run predicts.
    """
    start_bar = FIRST_TEST_BAR + 2 * TEST_PERIODS + 3
    end_bar = FIRST_TEST_BAR + 5 * TEST_PERIODS + 1
    backtester = _backtester(tmp_path / "cut", cv_project, start_bar=start_bar, end_bar=end_bar)
    loaded = _spy_load(monkeypatch, backtester)

    result = backtester.run_cv()

    assert [record["fold"] for record in result.folds] == [2, 3, 4, 5]
    assert loaded == cv_project.checkpoints[2:6]
    bars = cv_project.bars.astype("datetime64[ns]")
    first, last = result.folds[0], result.folds[-1]
    np.testing.assert_array_equal(
        first["predictions"].timestamp.values.astype("datetime64[ns]"),
        bars[start_bar : FIRST_TEST_BAR + 3 * TEST_PERIODS],
    )
    np.testing.assert_array_equal(
        last["predictions"].timestamp.values.astype("datetime64[ns]"),
        bars[FIRST_TEST_BAR + 5 * TEST_PERIODS : end_bar + 1],
    )
    assert first["test_start"] == _day(bars[start_bar])
    assert last["test_end"] == _day(bars[end_bar])
    np.testing.assert_array_equal(
        result.simulation.value.timestamp.values.astype("datetime64[ns]"),
        bars[start_bar : end_bar + 1],
    )
    assert float(result.weights["weight"].isel(timestamp=0).sum()) == pytest.approx(1.0)
    whole = result.metrics["stitched"]["whole"]
    assert pd.Timestamp(whole["Start"]) == pd.Timestamp(bars[start_bar])

    full = _backtester(tmp_path / "full", cv_project).run_cv()
    for record in (first, last):
        cut = record["predictions"]
        xr.testing.assert_allclose(
            cut, full.folds[record["fold"]]["predictions"].sel(timestamp=cut.timestamp)
        )
    run = BacktestRun.open(result.run_dir)
    assert run.window == (_day(bars[start_bar]), _day(bars[end_bar]))
    assert run.folds[0].window == (_day(bars[start_bar]), _day(bars[FIRST_TEST_BAR + 3 * TEST_PERIODS - 1]))


def test_a_window_inside_one_fold_runs_that_fold_cut_on_both_sides(tmp_path, cv_project):
    start_bar = FIRST_TEST_BAR + 4 * TEST_PERIODS + 1
    end_bar = start_bar + 3

    result = _backtester(tmp_path, cv_project, start_bar=start_bar, end_bar=end_bar).run_cv()

    assert [record["fold"] for record in result.folds] == [4]
    np.testing.assert_array_equal(
        result.simulation.value.timestamp.values.astype("datetime64[ns]"),
        cv_project.bars[start_bar : end_bar + 1].astype("datetime64[ns]"),
    )


def test_a_window_that_meets_no_test_segment_is_refused(tmp_path, cv_project):
    backtester = _backtester(
        tmp_path, cv_project, start_bar=FIRST_TEST_BAR - 10, end_bar=FIRST_TEST_BAR - 1
    )

    with pytest.raises(ValueError, match=r"no fold's test segment overlaps"):
        backtester.run_cv()


# --------------------------------------------------------------------------
# D-35: one continuous stitched simulation
# --------------------------------------------------------------------------


def test_stitched_curve_is_one_continuous_simulation(
    tmp_path, cv_project, monkeypatch
):
    calls = _spy_from_orders(monkeypatch)

    result = _backtester(tmp_path, cv_project).run_cv()

    # Only the stitched account is simulated (ADR 0033): once with costs and
    # once without (the attribution's cost part).
    assert len(calls) == 2
    stitched_bars = cv_project.bars[FIRST_TEST_BAR : LAST_TEST_BAR + 1].astype(
        "datetime64[ns]"
    )
    for call in calls:
        np.testing.assert_array_equal(call.to_numpy().astype("datetime64[ns]"), stitched_bars)
    np.testing.assert_array_equal(
        result.simulation.value.timestamp.values.astype("datetime64[ns]"),
        stitched_bars,
    )


def test_stitched_capital_is_not_reset_at_fold_boundaries(tmp_path, cv_project):
    """The stitched curve arrives at fold 1 still holding fold 0's last book, so
    its value moves across the boundary bar. A curve glued from per-fold values,
    whether reset to init_cash or rescaled to the previous end value, stays
    flat there."""
    result = _backtester(tmp_path, cv_project).run_cv()

    stitched = result.simulation.value
    boundary = _test_bars(cv_project, 1)[0]
    before = _test_bars(cv_project, 0)[-1]

    assert abs(float(stitched.sel(timestamp=boundary)) - INIT_CASH) > 1.0
    assert (
        abs(
            float(stitched.sel(timestamp=boundary))
            - float(stitched.sel(timestamp=before))
        )
        > 1e-6
    )


def test_a_folds_metrics_are_the_stitched_account_over_its_test_segment(tmp_path, cv_project):
    """ADR 0033: a fold's return runs from the bar before its first test bar
    (``init_cash`` for the first fold) to its last, so the folds' returns
    chain to the stitched total."""
    result = _backtester(tmp_path, cv_project).run_cv()
    value = result.simulation.value

    chained = 1.0
    for record in result.folds:
        bars = _test_bars(cv_project, record["fold"])
        whole = record["metrics"]["whole"]
        position = int(np.flatnonzero(value.timestamp.values == bars[0])[0])
        start = INIT_CASH if position == 0 else float(value.values[position - 1])
        end = float(value.sel(timestamp=bars[-1]))
        assert whole["Total Return [%]"] == pytest.approx((end / start - 1.0) * 100.0)
        assert pd.Timestamp(whole["Start"]) == pd.Timestamp(bars[0])
        assert pd.Timestamp(whole["End"]) == pd.Timestamp(bars[-1])
        chained *= 1.0 + whole["Total Return [%]"] / 100.0
    stitched = result.metrics["stitched"]["whole"]["Total Return [%]"]
    assert (chained - 1.0) * 100.0 == pytest.approx(stitched)


def test_run_cv_stitches_its_folds_on_the_valuation_basis(tmp_path, cv_project):
    """#119: run_cv() accepts the valuation basis and stitches as under the
    fill basis: the stitched curve is the weights backtest of its weights on
    the valuation basis, and it is not the fill basis's curve."""
    backtester = _backtester(tmp_path, cv_project, sizing_basis="valuation")
    result = backtester.run_cv()

    replayed = backtester.run_weights(result.weights)
    np.testing.assert_array_equal(replayed.simulation.value.values, result.simulation.value.values)
    by_open = _backtester(tmp_path, cv_project).run_cv()
    assert not np.allclose(by_open.simulation.value.values, result.simulation.value.values)


def test_the_stitched_block_carries_order_count_and_the_positions_view(
    tmp_path, cv_project
):
    """The stitched block goes through the same `run_metrics` as `run()`.

    A WIRING lock, not a second arithmetic proof: the partition identities for
    `Total Orders` are proved against a real `run()` in
    tests/test_backtest_metrics.py, and the stitched block is built by the same
    function, so what is worth asserting on this path is that `run_cv` reaches it
    at all and reports one trade vocabulary (D-02).
    """
    result = _backtester(tmp_path, cv_project).run_cv()
    whole = result.metrics["stitched"]["whole"]

    assert "Total Orders" in whole, sorted(whole)
    assert isinstance(whole["Total Orders"], int) and not isinstance(
        whole["Total Orders"], bool
    )
    assert whole["Total Orders"] >= 0
    assert "positions" not in whole, sorted(whole)


def test_run_cv_run_directory_contents(tmp_path, cv_project):
    result = _backtester(tmp_path, cv_project).run_cv()
    run_dir = result.run_dir

    assert run_dir.parent == tmp_path / "runs"
    assert {p.name for p in run_dir.iterdir()} == {
        "config.json",
        "weights.zarr",
        "equity.zarr",
        "holdings.zarr",
        "metrics.json",
        "settlements.json",
        "run.json",
        "report.html",
        "predictions.zarr",
        "folds",
    }

    run = BacktestRun.open(run_dir)

    # --- the prediction panel: the concatenated fold predictions, stitched run only
    panel = run.predictions()
    assert panel.labels == (
        LabelSpec(name=f"fwd_ret_{HORIZON}", scale="raw", delay=1, span=HORIZON),
    )
    xr.testing.assert_equal(
        panel.predictions,
        xr.concat([record["predictions"] for record in result.folds], dim="timestamp"),
    )
    # ... from which DecisionInputs.from_run reproduces the stitched weights.
    from quantlab.portfolio.decision_inputs import DecisionInputs

    replayed = DecisionInputs.from_run(run_dir).weights(panel.predictions)["weight"]
    xr.testing.assert_equal(replayed, result.weights["weight"].sel(symbol=replayed.symbol.values))

    # --- stitched artifacts --------------------------------------------------
    np.testing.assert_array_equal(
        run.weights()["weight"].values, result.weights["weight"].values
    )
    np.testing.assert_array_equal(
        run.equity()["value"].values, result.simulation.value.values
    )

    # --- per-fold child runs: the stitched account cut to each fold (ADR 0033) --
    assert run.kind == "run_cv"
    assert all(fold.predictions() is None for fold in run.folds)
    assert [fold.index for fold in run.folds] == [record["fold"] for record in result.folds]
    for fold, record in zip(run.folds, result.folds):
        assert fold.kind == "fold"
        assert {p.name for p in fold.path.iterdir()} == {"run.json", "metrics.json"}
        bars = _test_bars(cv_project, record["fold"])
        assert fold.window == (_day(bars[0]), _day(bars[-1]))
        xr.testing.assert_identical(fold.weights(), run.weights().sel(timestamp=bars))
        xr.testing.assert_identical(fold.equity(), run.equity().sel(timestamp=bars))
        assert fold.settlements() == [
            settlement for settlement in run.settlements()
            if bars[0] <= np.datetime64(pd.Timestamp(settlement["settlement_timestamp"]), "ns") <= bars[-1]
        ]
        assert fold.metrics()["whole"]["Total Return [%]"] == pytest.approx(
            record["metrics"]["whole"]["Total Return [%]"]
        )

    # --- metrics -------------------------------------------------------------
    metrics = run.metrics()
    assert set(metrics) == {"stitched", "folds", "notes"}
    assert "Total Return [%]" in metrics["stitched"]["whole"]
    assert metrics["notes"]
    assert [entry["fold"] for entry in metrics["folds"]] == list(range(N_FOLDS))
    for entry in metrics["folds"]:
        bars = _test_bars(cv_project, entry["fold"])
        assert entry["test_start"] == _day(bars[0])
        assert entry["test_end"] == _day(bars[-1])
        assert entry["checkpoint"] == cv_project.checkpoints[entry["fold"]]
        assert entry["metrics"]["in_sample_range"] is None
        assert "Total Return [%]" in entry["metrics"]["whole"]
    assert metrics["stitched"]["in_sample_ranges"] == []

    # --- settlements: the stitched list ------------------------------------------
    assert len(run.settlements()) == len(result.simulation.settlements)

    # --- the data fingerprint: the stitched pass on top, each fold's in its run --
    first_day = _day(cv_project.bars[FIRST_TEST_BAR])
    last_day = _day(cv_project.bars[LAST_TEST_BAR])
    # The stitched pass reads prices only: its predictions are the folds'.
    assert set(run.data_fingerprint) == {"price_dataset"}
    window = [
        entry for entry in run.data_fingerprint["price_dataset"]
        if pd.Timestamp(entry["request"]["start"]) == pd.Timestamp(first_day)
    ]
    assert [_day(entry["end"]) for entry in window] == [last_day]
    for fold, record in zip(run.folds, result.folds):
        # A fold only predicts: it records the model's reads, never the prices.
        assert "price_dataset" not in fold.data_fingerprint
        (factor,) = fold.data_fingerprint["model.factors.0.dataset"]
        assert _day(factor["start"]) < record["test_start"], "the factor range must include warm-up"
        assert _day(factor["end"]) == record["test_end"]

    assert run.rebuild_backtester().config.cv_project_dir == str(cv_project.project_dir)
    assert run.market == Market(fill_price_column="adjOpen", valuation_price_column="adjClose")
    assert run.trained_run().path == cv_project.project_dir


def _book_identity(run: BacktestRun, simulation, prices: xr.Dataset) -> None:
    """Holdings of ``run`` are the order records' positions over the value, plus cash one.

    The positions and cash come from ``simulation.orders`` and the price
    store alone, as in tests/test_backtest_holdings.py (#225); on each fill
    bar the holdings valued at the fill prices are the targets (no costs).
    """
    holdings = run.holdings()["holding"]
    bars = holdings.timestamp.values
    symbols = [str(s) for s in holdings.symbol.values]
    np.testing.assert_array_equal(bars, run.equity().timestamp.values)
    assert symbols == [str(s) for s in run.weights().symbol.values]
    orders = simulation.orders
    shares = np.zeros((bars.size, len(symbols)))
    cash = np.zeros(bars.size)
    rows = np.searchsorted(bars, orders["timestamp"].values.astype("datetime64[ns]"))
    columns = [symbols.index(str(s)) for s in orders["symbol"].values]
    sign = np.where(orders["side"].values.astype(str) == "Buy", 1.0, -1.0)
    size = orders["size"].values
    np.add.at(shares, (rows, columns), sign * size)
    np.add.at(cash, rows, -sign * size * orders["price"].values - orders["fees"].values)
    shares, cash = np.cumsum(shares, axis=0), INIT_CASH + np.cumsum(cash)

    def panel(name):
        return (
            prices[name].transpose("timestamp", "symbol")
            .sel(timestamp=bars, symbol=symbols).values.astype(np.float64)
        )

    close, fill_open = panel("adjClose"), panel("adjOpen")
    holding = holdings.values
    value = run.equity()["value"].values
    position = np.where(shares != 0.0, shares * close, 0.0)
    np.testing.assert_allclose(holding * value[:, None], position, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(holding.sum(axis=1) + cash / value, 1.0, rtol=0, atol=1e-9)
    weights = run.weights()["weight"].transpose("timestamp", "symbol").values
    for signal in np.flatnonzero(np.isfinite(weights).any(axis=1)):
        fill = signal + 1
        if fill >= bars.size:
            continue
        at_open = np.where(shares[fill] != 0.0, shares[fill] * fill_open[fill], 0.0)
        np.testing.assert_allclose(
            at_open / (cash[fill] + at_open.sum()),
            np.where(np.isfinite(weights[signal]), weights[signal], 0.0),
            atol=1e-9,
        )


def test_run_cv_writes_holdings_for_the_stitched_run_and_cuts_them_for_every_fold(
    tmp_path, cv_project
):
    from tests.test_backtest_holdings import holdings_data

    result = _backtester(tmp_path, cv_project).run_cv()
    run = BacktestRun.open(result.run_dir)
    prices = xr.open_zarr(cv_project.dataset_config.zarr_file_path).load()

    # The stitched holdings are the stitched simulation's, with no reset at a
    # fold boundary: the book entering fold 1 is still invested.
    _book_identity(run, result.simulation, prices)
    stitched = run.holdings()["holding"]
    boundary = pd.Timestamp(_test_bars(cv_project, 1)[0])
    assert float(np.abs(stitched.sel(timestamp=boundary)).sum()) == pytest.approx(1.0, abs=1e-6)

    # Each fold's holdings are the stitched ones over its test segment.
    assert len(run.folds) == N_FOLDS
    for fold in run.folds:
        assert not (fold.path / "holdings.zarr").exists()
        xr.testing.assert_identical(
            fold.holdings(), run.holdings().sel(timestamp=_test_bars(cv_project, fold.index))
        )

    # The stitched report's Holdings tab covers every stitched bar, from holdings.zarr.
    data = holdings_data(run.report())
    days = data["days"]
    assert [day["d"] for day in days] == [
        bar_label(b) for b in run.equity().timestamp.values
    ]
    assert len(days) == LAST_TEST_BAR - FIRST_TEST_BAR + 1
    names = data["names"]
    symbols = [str(s) for s in stitched.symbol.values]
    for i, day in enumerate(days):
        row = stitched.values[i]
        shown = {names[k][2]: h for k, _, h, _ in day["h"]}
        assert shown == {s: row[j] for j, s in enumerate(symbols) if s in shown}
        assert {s for j, s in enumerate(symbols) if row[j] != 0.0} <= shown.keys()
        assert day["cash"] == pytest.approx(1.0 - row.sum(), abs=1e-12)
    summary = dict(data["summary"])
    assert summary["Bars"] == str(len(days))
    assert run.metrics()["stitched"]["whole"]["Total Return [%]"] == pytest.approx(
        (float(run.equity()["value"].values[-1]) / INIT_CASH - 1.0) * 100.0
    )


#: Contexts every `_Recorder` saw, in call order.
_SEEN: list = []


@dataclasses.dataclass(frozen=True)
class _RecorderConfig:
    top_n: int = TOP_N


class _Recorder(PortfolioConstructor):
    """Top-n by the first label, recording every context it is handed."""

    config_cls = _RecorderConfig

    def construct(self, context):
        _SEEN.append(context)
        rule = TopNConstructor(TopNConfig(direction="long_only", top_n=self.config.top_n))
        return rule.construct(context)


def test_the_rule_decides_each_rebalance_bar_once_on_the_holdings_the_previous_fold_left(
    tmp_path, cv_project
):
    """ADR 0033: the portfolio is constructed once, on the stitched account.

    No fold is decided again from cash: the rule sees each rebalance bar of the
    stitched span exactly once, and at a fold boundary it is handed the book
    the previous fold left.
    """
    _SEEN.clear()
    backtester = _backtester(tmp_path, cv_project)
    backtester.config = dataclasses.replace(
        backtester.config, constructor=_Recorder(_RecorderConfig())
    )

    result = backtester.run_cv()

    n_bars = LAST_TEST_BAR - FIRST_TEST_BAR + 1
    rebalances = int(schedule(n_bars, REBALANCE_PERIODS).sum())
    assert len(_SEEN) == rebalances
    boundary = pd.Timestamp(_test_bars(cv_project, 1)[0])
    carried = next(c for c in _SEEN if c.timestamp == boundary)
    assert float(carried.current_weights.sum()) == pytest.approx(1.0, abs=1e-9)
    assert result.metrics["stitched"]["portfolio_construction"]["failed_bar_count"] == 0


# --------------------------------------------------------------------------
# #302: RecordedPredictor serves a recorded run's predictions
# --------------------------------------------------------------------------


def _recorded(tmp_path, cv, run_dir):
    from quantlab.model.predefined.recorded import RecordedPredictor

    model = make_model(
        tmp_path / "recorded_model", cv.dataset_config, n_forward_periods=HORIZON,
        **_model_dates(cv.bars),
    )
    return RecordedPredictor(model, run_dir)


def test_a_recorded_predictor_replays_a_run_cv_from_another_start(tmp_path, cv_project):
    """#302: the folds of a run_cv started inside fold 2 are served from a whole
    run_cv's predictions, and trade exactly as the same run_cv predicting them."""
    whole = _backtester(tmp_path / "whole", cv_project).run_cv()
    start_bar = FIRST_TEST_BAR + 2 * TEST_PERIODS + 3
    direct = _backtester(tmp_path / "direct", cv_project, start_bar=start_bar).run_cv()
    backtester = _backtester(tmp_path / "replay", cv_project, start_bar=start_bar)
    backtester.config = dataclasses.replace(
        backtester.config, model=_recorded(tmp_path, cv_project, whole.run_dir)
    )

    replay = backtester.run_cv()

    assert [r["fold"] for r in replay.folds] == [r["fold"] for r in direct.folds]
    for a, b in zip(replay.folds, direct.folds):
        xr.testing.assert_allclose(a["predictions"], b["predictions"])
    xr.testing.assert_allclose(replay.weights, direct.weights)
    xr.testing.assert_allclose(replay.simulation.value, direct.simulation.value)
    rebuilt = BacktestRun.open(replay.run_dir).rebuild("model")
    assert type(rebuilt).__name__ == "RecordedPredictor"


def test_a_recorded_predictor_serves_only_its_checkpoints_fold(tmp_path, cv_project):
    whole = _backtester(tmp_path / "whole", cv_project).run_cv()
    recorded = _recorded(tmp_path, cv_project, whole.run_dir)

    with pytest.raises(ValueError, match="no checkpoint loaded"):
        recorded.predict_window(_day(_test_bars(cv_project, 0)[0]), _day(_test_bars(cv_project, 0)[-1]))
    recorded.load(cv_project.checkpoints[0])
    inside = recorded.predict_window(
        _day(_test_bars(cv_project, 0)[1]), _day(_test_bars(cv_project, 0)[-1])
    )
    np.testing.assert_array_equal(
        inside.timestamp.values.astype("datetime64[ns]"), _test_bars(cv_project, 0)[1:]
    )
    with pytest.raises(ValueError, match="outside"):
        recorded.predict_window(_day(_test_bars(cv_project, 1)[0]), _day(_test_bars(cv_project, 1)[-1]))
    with pytest.raises(ValueError, match="predicts nothing new"):
        recorded.train()
    late = _backtester(
        tmp_path / "late", cv_project, start_bar=FIRST_TEST_BAR + 5 * TEST_PERIODS
    ).run_cv()
    other = _recorded(tmp_path / "other", cv_project, late.run_dir)
    other.load(cv_project.checkpoints[0])
    with pytest.raises(ValueError, match="holds no prediction of checkpoint"):
        other.predict_window(_day(_test_bars(cv_project, 0)[0]), _day(_test_bars(cv_project, 0)[-1]))
