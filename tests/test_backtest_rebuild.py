"""A backtest rebuilds from its persisted config.json and re-runs identically (phase 03.7, plan 11).

What is locked here:

- **D-25, the recipe rebuilds everything.** `BacktestRun.rebuild_backtester`
  turns the recipe a run wrote back into a backtester: the backtester class,
  its `CrossSectionBacktestConfig`, the price dataset, the model with its
  factors and labels (and the checkpoint reference), and every scalar
  parameter. Re-running the rebuilt backtester reproduces the same target
  weights and the same equity curve, for `run()` in load and train mode and for
  `run_cv()`.
- **D-26, the backtester round trip.** `get_config()` -> JSON -> loader ->
  `get_config()` is the identity, with the declared config classes on the way.
- **D-27, fingerprints on rebuild.** The data fingerprint is a record of what
  the original run read, kept in its `run.json` (#133), never in the recipe.
  `rebuild_backtester` sets it as `expected_fingerprint`, so an unchanged store
  re-runs silently and a changed store re-runs with a warning naming the
  dataset, and still completes.
- **Security (RESEARCH Security Domain).** A `name` that is not a
  `BaseBacktester` subclass is refused before any dataset or model is built.

Fixture classes must live in an importable module (`tests.backtest_fixtures`):
a rebuild resolves every class by its dotted `name`, and a class defined in
`__main__` or inside a test function cannot be imported back.

Everything is synthetic, CPU-only and offline. Configs are constructed
directly, never through the factories in `quantlab/config/__init__.py` (D-32).
"""

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from loguru import logger

import quantlab.core.component as component_rule
from quantlab.backtest.base import BaseBacktester
import quantlab.core.component as component_rule
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.model.config import ModelConfig
from quantlab.utils.jsonable import to_jsonable
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from quantlab.tracking.wandb import WandbTracker
from tests.backtest_fixtures import (
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

N_BARS = 60
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)
TRAIN_END_BAR = 24
WINDOW_START_BAR = 30
WINDOW_END_BAR = 50
REBALANCE_PERIODS = 2
TOP_N = 2
INIT_CASH = 1_000_000.0

#: run_cv geometry, as in tests/test_backtest_run_cv.py: `walk_forward_folds` with
#: train_periods=30 over 80 bars gives 8 folds of 6 test bars, bars 30..77.
CV_N_BARS = 80
CV_BARS = pd.bdate_range("2024-01-01", periods=CV_N_BARS)
CV_TRAIN_PERIODS = 30
CV_FIRST_TEST_BAR = 30
CV_LAST_TEST_BAR = 77

#: `quantlab.runs.record.DataRecorder` starts every mismatch with this.
FINGERPRINT_WARNING = "data fingerprint mismatch"

#: The distinctive substring of `quantlab.runs.record._PARTIAL_NOTE`, the
#: tail a failure-path comparison appends instead of "continuing".
#: Spelled out here rather than imported on purpose: an ImportError at module
#: level would break collection of this whole file, and these locks must be able
#: to go red on code that does not define the constant yet. Any future rewording
#: of that note must keep this substring.
PARTIAL_WARNING = "comparison is PARTIAL"


@pytest.fixture
def warning_messages():
    """Every loguru WARNING emitted during the test, as plain message text."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    yield messages
    logger.remove(handler_id)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _json(cfg: dict) -> dict:
    """What `config.json` holds: `to_jsonable` then a strict JSON trip."""
    return json.loads(json.dumps(to_jsonable(cfg), allow_nan=False))


def _model_dates() -> dict:
    return dict(
        start_date=_day(BARS[0]),
        end_date=_day(BARS[29]),
        train_start=_day(BARS[0]),
        train_end=_day(BARS[TRAIN_END_BAR]),
        test_start=_day(BARS[TRAIN_END_BAR + 1]),
        test_end=_day(BARS[29]),
    )


def _backtester(
    root: Path,
    dataset_config,
    *,
    model_mode: str = "load",
    checkpoint=None,
    **overrides,
) -> USEquityCrossectionSelectStockVectorBt:
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / "backtest", dataset_config, **_model_dates()),
        model_mode=model_mode,
        checkpoint=None if checkpoint is None else str(checkpoint),
        start_date=_day(BARS[WINDOW_START_BAR]),
        end_date=_day(BARS[WINDOW_END_BAR]),
        output_dir=str(root / "runs"),
        rebalance_periods=REBALANCE_PERIODS,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=TOP_N)),
        fees=0.0,
        slippage=0.0,
        init_cash=INIT_CASH,
    )
    kwargs.update(overrides)
    return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(**kwargs))


def _trained(root: Path):
    dataset_config = write_price_store(root / "store", n_bars=N_BARS)
    checkpoint = train_checkpoint(
        make_model(root / "train", dataset_config, **_model_dates())
    )
    return dataset_config, checkpoint


def _rebuilt(run_dir: Path, **overrides) -> USEquityCrossectionSelectStockVectorBt:
    """The backtester a run directory records, rebuilt through `BacktestRun`."""
    return BacktestRun.open(run_dir).rebuild_backtester(**overrides)


def _assert_same_run_artifacts(first_dir: Path, second_dir: Path) -> None:
    """Identical persisted weights and exactly equal persisted equity values."""
    assert first_dir != second_dir
    first, second = BacktestRun.open(first_dir), BacktestRun.open(second_dir)
    xr.testing.assert_identical(first.weights(), second.weights())
    np.testing.assert_array_equal(
        first.equity()["value"].values, second.equity()["value"].values
    )


def _fingerprint_warnings(messages: list[str]) -> list[str]:
    return [m for m in messages if FINGERPRINT_WARNING in m]


def _cv_original(tmp_path: Path) -> USEquityCrossectionSelectStockVectorBt:
    """The run_cv setup: a CV-sized store, a `train_cv` project, and a backtester over it.

    Writes `CV_N_BARS` bars, trains one `train_cv` project (its walk-forward unit), and returns a backtester pointed at that
    project over the `CV_FIRST_TEST_BAR..CV_LAST_TEST_BAR` window. Shared by the
    rebuild lock and the failure-path lock so neither duplicates the setup.
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=CV_N_BARS)
    model_dates = dict(
        start_date=_day(CV_BARS[0]),
        end_date=_day(CV_BARS[CV_N_BARS - 1]),
        train_start=_day(CV_BARS[0]),
        train_end=_day(CV_BARS[CV_TRAIN_PERIODS - 1]),
        test_start=_day(CV_BARS[CV_TRAIN_PERIODS]),
        test_end=_day(CV_BARS[CV_N_BARS - 1]),
    )
    trainer = make_model(tmp_path / "train", dataset_config, **model_dates)
    trainer.collect()
    cv = trainer.train_cv(train_periods=CV_TRAIN_PERIODS)

    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=make_model(tmp_path / "backtest", dataset_config, **model_dates),
            model_mode="load",
            cv_project_dir=str(cv.path),
            start_date=_day(CV_BARS[CV_FIRST_TEST_BAR]),
            end_date=_day(CV_BARS[CV_LAST_TEST_BAR]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=REBALANCE_PERIODS,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=TOP_N)),
            fees=0.0,
            slippage=0.0,
            init_cash=INIT_CASH,
        )
    )


# --------------------------------------------------------------------------
# Task 1: the loader rebuilds the whole backtester (D-25, D-26)
# --------------------------------------------------------------------------


def test_rebuilt_backtester_has_the_same_class_config_class_and_config(tmp_path):
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    original = _backtester(
        tmp_path, dataset_config, checkpoint=tmp_path / "never_read.joblib"
    )
    saved = _json(original.get_config())

    rebuilt = component_rule.rebuild(saved, expected=BaseBacktester)

    assert type(rebuilt) is USEquityCrossectionSelectStockVectorBt
    assert type(rebuilt.config) is CrossSectionBacktestConfig
    assert type(rebuilt.config.model.config) is ModelConfig
    assert rebuilt.expected_fingerprint is None
    assert _json(rebuilt.get_config()) == saved


def test_rebuild_from_a_run_sets_its_fingerprint_as_expected(tmp_path):
    dataset_config, checkpoint = _trained(tmp_path)
    result = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()
    run = BacktestRun.open(result.run_dir)
    assert run.data_fingerprint

    rebuilt = run.rebuild_backtester()

    assert rebuilt.expected_fingerprint == run.data_fingerprint
    assert "data_fingerprint" not in rebuilt.get_config()
    assert rebuilt.config.checkpoint == str(checkpoint)


def test_the_recipe_holds_no_records_and_the_run_records_the_market(tmp_path):
    """#106/#133: the market columns are a record of the class's MARKET, kept in run.json.

    The recipe the run wrote rebuilds a backtester whose config is exactly that
    recipe, with no record mixed in, and the rebuilt backtester takes its columns
    from its class again.
    """
    dataset_config, checkpoint = _trained(tmp_path)
    result = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()
    run = BacktestRun.open(result.run_dir)
    assert (run.market.fill_price_column, run.market.valuation_price_column) == (
        "adjOpen",
        "adjClose",
    )

    rebuilt = run.rebuild_backtester()

    assert rebuilt.MARKET.fill_price_column == "adjOpen"
    config = _json(rebuilt.get_config())
    assert not {"market", "data_fingerprint", "trained_checkpoint"} & set(config)
    assert _json(component_rule.rebuild(config, expected=BaseBacktester).get_config()) == config


def test_backtest_config_path_fields_are_stored_absolute(tmp_path, monkeypatch):
    """Code review WR-03: `checkpoint`, `cv_project_dir` and `output_dir` persist as absolute paths.

    A persisted `config.json` holding relative paths rebuilds against
    whatever directory the rebuild runs in (D-25). The backtester is built
    from `tmp_path` with relative spellings and the config is read back. The
    old setter kept them verbatim and goes red.
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    monkeypatch.chdir(tmp_path)

    config = _json(
        _backtester(
            tmp_path,
            dataset_config,
            checkpoint="ckpt/model.joblib",
            output_dir="runs",
            cv_project_dir="models/project",
        ).get_config()
    )

    for key, relative in (
        ("checkpoint", "ckpt/model.joblib"),
        ("output_dir", "runs"),
        ("cv_project_dir", "models/project"),
    ):
        assert Path(config[key]).is_absolute(), (key, config[key])
        assert Path(config[key]).resolve() == (tmp_path / relative).resolve()


def test_loader_does_not_mutate_its_input(tmp_path):
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    saved = _json(
        _backtester(
            tmp_path, dataset_config, checkpoint=tmp_path / "never_read.joblib"
        ).get_config()
    )
    before = copy.deepcopy(saved)

    component_rule.rebuild(saved, expected=BaseBacktester)

    assert saved == before


@pytest.mark.parametrize("field_name", ["init_cash", "fees", "constructor", "tracker"])
def test_rebuild_refuses_a_config_missing_a_field(tmp_path, monkeypatch, field_name):
    """Code review WR-06: a config missing a field is refused, not filled from today's defaults.

    `init_cash` is the executor-reported gap; `fees` and `tracker` have
    defaults that could change later, and `constructor` holds the rule. The old loader built
    the config with `**config` and silently took the current default, so an
    older config.json rebuilt into a different backtest. The refusal must
    name the field and come before any dataset or model is built. Red on the
    old code: no ValueError, and the stubbed loaders are called.
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    saved = _json(
        _backtester(
            tmp_path, dataset_config, checkpoint=tmp_path / "never_read.joblib"
        ).get_config()
    )
    del saved[field_name]
    built: list[dict] = []
    # Stub the nested rebuilds; the backtester itself goes through the real rule.
    rebuild = component_rule.rebuild
    monkeypatch.setattr(
        component_rule, "rebuild", lambda cfg, run_dir=None: built.append(cfg)
    )

    with pytest.raises(ValueError, match=field_name):
        rebuild(saved, expected=BaseBacktester)

    assert built == []


def test_rebuild_round_trips_every_field_with_non_default_values(tmp_path):
    """Code review WR-06: every config field survives the rebuild, checked with NON-default values.

    A round trip that uses a field's default cannot detect that the field was
    dropped: the loader would fill in the same default. So every defaulted
    field gets a value different from its default, verified at the top of
    the test, except `benchmark_dataset` and `risk_model` (live objects,
    round-tripped in `test_backtest_benchmark.py` and
    `test_backtest_factor_attribution.py`) and
    `name` (rebuilt from the class). Every scalar field must then come back
    equal. This is a lock rather than a red-first test: it passes before the
    fix too, and goes red if `to_dict`/`get_config` ever drops a field or the
    loader substitutes a default.
    """
    from dataclasses import MISSING, fields

    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    original = _backtester(
        tmp_path,
        dataset_config,
        model_mode="load",
        checkpoint=tmp_path / "never_read.joblib",
        cv_project_dir=str(tmp_path / "cv_project"),
        fees=0.0007,
        slippage=0.0003,
        init_cash=250_000.0,
        sizing_basis="valuation",
        constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=1, score_label="fwd_ret_1")),
        tracker=WandbTracker(project="rebuilt", mode="disabled"),
        rebalance_periods=3,
    )
    for field in fields(CrossSectionBacktestConfig):
        if field.default is MISSING or field.name in ("benchmark_dataset", "risk_model", "name"):
            continue
        assert getattr(original.config, field.name) != field.default, field.name

    saved = _json(original.get_config())
    rebuilt = component_rule.rebuild(saved, expected=BaseBacktester)

    for field in fields(CrossSectionBacktestConfig):
        if field.name in ("price_dataset", "model", "benchmark_dataset", "risk_model"):
            continue
        assert getattr(rebuilt.config, field.name) == getattr(
            original.config, field.name
        ), field.name
    assert _json(rebuilt.get_config()) == saved


def test_non_backtester_class_is_refused_before_building_anything(
    tmp_path, monkeypatch
):
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    saved = _json(
        _backtester(
            tmp_path, dataset_config, checkpoint=tmp_path / "never_read.joblib"
        ).get_config()
    )
    tampered = dict(saved, name="quantlab.dataset.stock.StockDataset")

    built: list[dict] = []
    # Stub the nested rebuilds; the backtester itself goes through the real rule.
    rebuild = component_rule.rebuild
    monkeypatch.setattr(
        component_rule, "rebuild", lambda cfg, run_dir=None: built.append(cfg)
    )

    with pytest.raises(TypeError, match=r"quantlab\.dataset\.stock\.StockDataset"):
        rebuild(tampered, expected=BaseBacktester)

    assert built == []


def _imported_modules(path: Path) -> set[str]:
    """Every absolute module name `path` imports, including `from a import b` as `a.b`."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import in {path}: resolve it first"
            names.add(node.module or "")
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_backtester_rebuild_imports_no_config_factories():
    """D-32: neither the loader nor these locks reach `quantlab.config`."""
    # Positive control: the scan sees this file's own real imports.
    assert "quantlab.core.component" in _imported_modules(Path(__file__))

    for path in (Path(__file__), REPO_ROOT / "quantlab/core/component.py"):
        offending = sorted(
            name
            for name in _imported_modules(path)
            if name == "quantlab.config" or name.startswith("quantlab.config.")
        )
        assert offending == [], f"{path} imports {offending}"


# --------------------------------------------------------------------------
# Task 2: a rebuilt config re-runs identically and notices changed data (D-25, D-27)
# --------------------------------------------------------------------------


def test_load_mode_rebuild_reproduces_weights_and_equity(tmp_path, warning_messages):
    dataset_config, checkpoint = _trained(tmp_path)
    first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()

    rebuilt = _rebuilt(first.run_dir)
    second = rebuilt.run()

    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    xr.testing.assert_identical(first.weights, second.weights)
    np.testing.assert_array_equal(
        first.simulation.value.values, second.simulation.value.values
    )
    assert (
        second.metrics["whole"]["Total Return [%]"]
        == first.metrics["whole"]["Total Return [%]"]
    )
    assert rebuilt.expected_fingerprint is not None
    assert _fingerprint_warnings(warning_messages) == []


def test_train_mode_rebuild_retrains_and_reproduces_weights_and_equity(
    tmp_path, warning_messages
):
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    first = _backtester(tmp_path, dataset_config, model_mode="train").run()
    rebuilt = _rebuilt(first.run_dir)
    assert rebuilt.config.model_mode == "train"
    assert rebuilt.config.checkpoint is None
    # Code review WR-04: `BaseModel.train` names its project directory to the
    # microsecond and never reuses an existing one, so the rebuild retrains
    # immediately. The old `time.sleep(1.1)` wall-clock workaround is gone;
    # without the fix this line raises "... already exists" within the second.
    second = rebuilt.run()

    checkpoints = sorted(Path(rebuilt.config.model.config.model_save_dir).rglob("*.joblib"))
    assert len(checkpoints) == 2, "the rebuilt run must train its own model"
    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    assert (
        second.metrics["whole"]["Total Return [%]"]
        == first.metrics["whole"]["Total Return [%]"]
    )
    assert _fingerprint_warnings(warning_messages) == []


def test_train_mode_run_records_its_checkpoint_and_replays_it_in_load_mode(tmp_path):
    """Code review WR-04 / #133: a train-mode run records the unit it trained.

    A train-mode backtest that did not name the model it produced could only be
    retrained, and retraining is not bit-reproducible for torch or GPU heads.
    The run's `trained_run()` opens the one unit trained, `metrics.json` names
    its checkpoint, and a load-mode rebuild of that checkpoint replays the run
    identically. The replay trains nothing and loads that same unit.
    """
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    first = _backtester(tmp_path, dataset_config, model_mode="train").run()
    run = BacktestRun.open(first.run_dir)
    model_save_dir = Path(run.rebuild("model").config.model_save_dir)

    unit = run.trained_run()
    recorded = run.metrics()["trained_checkpoint"]
    assert Path(recorded).is_absolute() and Path(recorded).is_file(), recorded
    assert unit.checkpoint == Path(recorded)
    assert sorted(model_save_dir.rglob("*.joblib")) == [Path(recorded)]

    replay = run.rebuild_backtester(model_mode="load", checkpoint=recorded)
    second = replay.run()

    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    assert BacktestRun.open(second.run_dir).trained_run() == unit
    assert "trained_checkpoint" not in BacktestRun.open(second.run_dir).metrics()
    assert len(sorted(model_save_dir.rglob("*.joblib"))) == 1


def test_run_cv_rebuild_reproduces_the_stitched_curve(tmp_path, warning_messages):
    original = _cv_original(tmp_path)
    first = original.run_cv()

    rebuilt = _rebuilt(first.run_dir)
    second = rebuilt.run_cv()

    assert len(second.folds) == len(first.folds) > 1
    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    xr.testing.assert_identical(first.weights, second.weights)
    np.testing.assert_array_equal(
        first.simulation.value.values, second.simulation.value.values
    )
    assert _fingerprint_warnings(warning_messages) == []


def test_changed_store_rebuild_warns_and_completes(tmp_path, warning_messages):
    dataset_config, checkpoint = _trained(tmp_path)
    first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()

    # Overwrite one adjusted close inside the backtest window directly in the
    # Zarr array (group opened "r+"), the way a Tiingo re-base rewrites history.
    bar, symbol = WINDOW_START_BAR + 5, 0
    group = zarr.open_group(dataset_config.zarr_file_path, mode="r+")
    old = float(group["adjClose"][bar, symbol])
    group["adjClose"][bar, symbol] = old * 1.25
    # Positive control: the change is visible through xarray, where the run reads.
    assert float(
        xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].values[bar, symbol]
    ) == pytest.approx(old * 1.25)

    rebuilt = _rebuilt(first.run_dir)
    assert _fingerprint_warnings(warning_messages) == []
    second = rebuilt.run()

    assert second.run_dir.is_dir()
    mismatches = _fingerprint_warnings(warning_messages)
    assert any("'price_dataset'" in m for m in mismatches), warning_messages


# --------------------------------------------------------------------------
# Task 3: a failed run still reports the changed data (D-03.11-UAT-A)
# --------------------------------------------------------------------------


def _boom(*_args, **_kwargs):
    """Stands in for any real post-fingerprint failure inside the backtest window.

    For example a ticker-era checkpoint predicted against a PERMNO panel, or
    "the feature panel lacks N of the symbols this model was trained on".
    """
    raise ValueError("predict_panel: representative downstream failure")


def test_a_raise_inside_the_window_still_reports_the_changed_data(
    tmp_path, warning_messages, monkeypatch
):
    """A run that dies inside `_backtest_window` still says the data changed (D-03.11-UAT-A).

    Control arm: without any raise, the factor dataset and the price dataset
    are reported changed, every warning ending "continuing", none partial.

    Probe arm: `predict_panel` raises after the factor's dataset was read and
    before any price is. The factor dataset's mismatch is reported, marked
    partial, the run keeps what it had read as its `data_fingerprint`, and
    the original `ValueError` propagates.
    """
    dataset_config, checkpoint = _trained(tmp_path)
    first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()

    # A real data change at the same path `_trained` wrote: one symbol
    # disappears, so every read differs.
    write_price_store(tmp_path / "store", symbols=SYMBOLS[:-1], n_bars=N_BARS)

    # --- control: no raise -----------------------------------------------------
    warning_messages.clear()
    _rebuilt(first.run_dir).run()

    control = _fingerprint_warnings(warning_messages)
    assert any("'model.factors.0.dataset'" in m for m in control), control
    assert any("'price_dataset'" in m for m in control), control
    assert all(m.endswith("; continuing") for m in control), control
    assert all(PARTIAL_WARNING not in m for m in control), control

    # --- probe: a raise inside the window, after the factor data was read ------
    warning_messages.clear()
    rebuilt = _rebuilt(first.run_dir)
    monkeypatch.setattr(rebuilt.config.model, "predict_panel", _boom)

    with pytest.raises(ValueError, match="representative downstream failure"):
        rebuilt.run()

    # No price had been read yet at raise time.
    assert sorted(rebuilt.data_fingerprint) == ["model.factors.0.dataset"]
    probe = _fingerprint_warnings(warning_messages)
    assert len(probe) == 1, probe
    assert "'model.factors.0.dataset'" in probe[0], probe
    assert "x 6 symbols" in probe[0] and "x 5 symbols" in probe[0], probe
    assert PARTIAL_WARNING in probe[0], probe
    # `price_dataset` was not read YET, not "not read": warning about it would
    # be a false alarm.
    assert all("not read by this run" not in m for m in warning_messages), (
        warning_messages
    )


def test_a_failing_partial_diagnostic_never_replaces_the_real_exception(
    tmp_path, warning_messages, monkeypatch
):
    """A broken diagnostic cannot become the exception the caller sees (D-03.11-UAT-A).

    The failure-path guard swallows whatever the diagnostic raises, but it does
    not hide it: the broken diagnostic is reported as its own warning, and the
    original `ValueError` propagates unchanged.
    """
    from quantlab.runs.record import DataRecorder

    dataset_config, checkpoint = _trained(tmp_path)
    first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()
    write_price_store(tmp_path / "store", symbols=SYMBOLS[:-1], n_bars=N_BARS)

    warning_messages.clear()
    rebuilt = _rebuilt(first.run_dir)
    monkeypatch.setattr(rebuilt.config.model, "predict_panel", _boom)

    def broken_diagnostic(*_args, **_kwargs):
        raise RuntimeError("the diagnostic itself is broken")

    monkeypatch.setattr(DataRecorder, "_compare", broken_diagnostic)

    with pytest.raises(ValueError, match="representative downstream failure") as excinfo:
        rebuilt.run()

    assert "the diagnostic itself is broken" not in repr(excinfo.value)
    reported = [
        m
        for m in warning_messages
        if "diagnostic" in m and m not in _fingerprint_warnings(warning_messages)
    ]
    assert reported, warning_messages
    assert any("the diagnostic itself is broken" in m for m in reported), reported


def _shift_close(dataset_config, bar: int, factor: float) -> None:
    """Rewrite one adjusted close in place, the way a re-base rewrites history."""
    group = zarr.open_group(dataset_config.zarr_file_path, mode="r+")
    group["adjClose"][bar, 0] = float(group["adjClose"][bar, 0]) * factor


def test_run_cv_reports_a_partial_comparison_when_a_fold_raises(
    tmp_path, warning_messages, monkeypatch
):
    """`run_cv` carries the same failure-path diagnostic, per fold (D-03.11-UAT-A).

    A close inside fold 0's test segment changes; the rebuilt `run_cv` dies
    in fold 0's prediction. Fold 0 compares with its own record: the factor
    dataset it had read is reported, marked partial and naming the fold, and
    the original exception propagates.
    """
    original = _cv_original(tmp_path)
    first = original.run_cv()
    assert len(first.folds) > 1
    _shift_close(original.config.price_dataset.config, CV_FIRST_TEST_BAR + 2, 1.25)

    rebuilt = _rebuilt(first.run_dir)
    monkeypatch.setattr(rebuilt.config.model, "predict_panel", _boom)
    warning_messages.clear()

    with pytest.raises(ValueError, match="representative downstream failure"):
        rebuilt.run_cv()

    partial = _fingerprint_warnings(warning_messages)
    assert len(partial) == 1, warning_messages
    assert PARTIAL_WARNING in partial[0]
    assert "fold 0:" in partial[0] and "'model.factors.0.dataset'" in partial[0]
    assert all("not read by this run" not in m for m in warning_messages), (
        warning_messages
    )


def test_run_cv_rebuild_names_the_fold_whose_data_changed(tmp_path, warning_messages):
    """Each fold child run holds what its prediction read; a rebuild names the fold that changed.

    Folds test 6 bars each from bar 30 and the factor warms up 5 bars, so a
    close changed at bar 50 lies in the factor reads of fold 3's test segment
    and fold 4's warm-up, and in the stitched pass's prices, but in no other
    fold. A fold reads no prices (ADR 0033).
    """
    original = _cv_original(tmp_path)
    first = original.run_cv()
    run = BacktestRun.open(first.run_dir)
    for fold in run.folds:
        assert set(fold.data_fingerprint) == {"model.factors.0.dataset"}
    assert set(run.data_fingerprint) == {"price_dataset"}  # the stitched pass

    _shift_close(original.config.price_dataset.config, 50, 1.25)
    warning_messages.clear()
    second = _rebuilt(first.run_dir).run_cv()

    assert second.run_dir.is_dir()
    changed = _fingerprint_warnings(warning_messages)
    named = {i for i in range(len(first.folds)) if any(f"fold {i}:" in m for m in changed)}
    assert named == {3, 4}, changed
    assert any(m.startswith("USEquityCrossectionSelectStockVectorBt: data") for m in changed)


# --------------------------------------------------------------- shared components
# #214: a dataset object held at several places of the tree (the price dataset
# and the model's factor and label) is rebuilt as ONE object, so the rebuilt
# run records its reads under the same keys as the original and an unchanged
# run re-runs silently.


def _shared_dataset_backtester(tmp_path: Path, shared):
    dataset_config, checkpoint = _trained(tmp_path)
    model = make_model(
        tmp_path / "backtest", dataset_config, dataset=shared(dataset_config), **_model_dates()
    )
    return _backtester(
        tmp_path,
        dataset_config,
        checkpoint=checkpoint,
        price_dataset=model.config.factors[0].config.dataset,
        model=model,
    )


def _datasets(backtester) -> list:
    """Every dataset in the backtester's tree, once per place it is held."""
    from quantlab.dataset.base import MarketDataset

    return [
        item
        for _, item in component_rule.walk_components(backtester)
        if isinstance(item, MarketDataset)
    ]


def _stock(dataset_config):
    return make_stock_dataset(dataset_config)


def _frame(dataset_config):
    from quantlab.dataset.memory import FrameDataset

    return FrameDataset(make_stock_dataset(dataset_config).panel("1900-01-01", "2100-01-01"))


@pytest.mark.parametrize("shared", [_stock, _frame], ids=["StockDataset", "FrameDataset"])
def test_a_shared_dataset_is_rebuilt_shared_and_reruns_silently(
    tmp_path, warning_messages, shared
):
    first = _shared_dataset_backtester(tmp_path, shared).run()

    rebuilt = _rebuilt(first.run_dir)
    assert len(_datasets(rebuilt)) == 3  # price dataset, factor, label
    assert len({id(dataset) for dataset in _datasets(rebuilt)}) == 1

    second = rebuilt.run()
    _assert_same_run_artifacts(first.run_dir, second.run_dir)
    assert _fingerprint_warnings(warning_messages) == []


def test_datasets_held_apart_are_rebuilt_apart(tmp_path):
    dataset_config, checkpoint = _trained(tmp_path)
    first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()

    assert "shared_as" not in json.dumps(BacktestRun.open(first.run_dir).recipe())
    rebuilt = _rebuilt(first.run_dir)
    assert len({id(dataset) for dataset in _datasets(rebuilt)}) == 3


def test_an_override_drops_the_sharing_it_replaces(tmp_path, warning_messages):
    first = _shared_dataset_backtester(tmp_path, _stock).run()
    dataset_config, checkpoint = _trained(tmp_path / "again")
    model = make_model(tmp_path / "other", dataset_config, **_model_dates())

    rebuilt = _rebuilt(first.run_dir, model=model, checkpoint=str(checkpoint))

    assert rebuilt.config.model is model
    assert len({id(dataset) for dataset in _datasets(rebuilt)}) == 3
