"""`SeedEnsemble` builds, trains, saves, checks and loads a seed ensemble.

What is locked here, and what turns it red:

- Construction refuses fewer than two seeds or a repeated seed; each member
  is the wrapped model's class on its config with `random_seed` replaced.
- `collect()` collects once: every member holds the same collected panel.
- Different seeds give different member predictions for a head that draws
  from the seeded generator, and `predict_window` is `average_predictions`
  of the members' predictions.
- `train()` writes `SeedEnsemble_trial_*/` holding `member_{k}/` (each with
  the usual checkpoint, config.json, ic_series.csv, test_predictions.zarr
  and run.json), the ensemble-level evaluation files (see
  test_ensemble_evaluation_files.py) and, last, `run.json`, which makes the
  directory an "ensemble" trained unit recording every member's directory
  and seed and the windows (#123); it returns the path of that `run.json`.
- A member failing mid-training leaves no ensemble `run.json`, and the
  member directories already written stay.
- `load(run.json)` restores every member; `check_checkpoint` refuses a
  missing file, an unknown `format_version`, a unit of another kind, a member
  class, count or seed that does not match, a missing member checkpoint and
  a member without `run.json`; an old `ensemble.json` layout is refused with
  a message to retrain.
- `get_config()` / `from_config()` round-trip the wrapped model's config and
  the seeds.
- Layout: `quantlab/model/` has empty `__init__.py` files, no other
  quantlab module imports it, and it imports no backtest module.

Everything is synthetic, CPU-only and offline.

The unit's file names appear here only in the directory-listing lock of its layout
and in rewriting or removing a record to check a refusal; results are read through
`TrainedRun`.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.model.ensemble import average_predictions
from quantlab.utils.jsonable import to_jsonable
from quantlab.runs.trained_run import TrainedRun
from tests.test_backtest_contracts import (
    REPO_ROOT,
    _is_or_under,
    _python_files,
    _resolved_imports,
)
from tests.backtest_fixtures import (
    FirstFeatureHead,
    SeededHead,
    make_model,
    write_price_store,
)

N_BARS = 60
SEEDS = [0, 1, 2]
MEMBER_FILES = [
    "config.json",
    "ic_series.csv",
    "run.json",
    "test_predictions.zarr",
]


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    return dataset_config, bars


def _dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )


def _model(tmp_path, dataset_config, bars, *, head=SeededHead, name="models"):
    return make_model(tmp_path / name, dataset_config, head=head, **_dates(bars))


def _trained(tmp_path, seeds=SEEDS):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), list(seeds))
    checkpoint = ensemble.collect().train()
    return ensemble, checkpoint, dataset_config, bars


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seeds", [[], [0], [3, 3], [0, 1, 0]])
def test_fewer_than_two_or_repeated_seeds_raise(tmp_path, seeds):
    dataset_config, bars = _setup(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        SeedEnsemble(_model(tmp_path, dataset_config, bars), seeds)


def test_members_are_the_model_class_with_each_seed(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    model = _model(tmp_path, dataset_config, bars)

    ensemble = SeedEnsemble(model, [7, 3])

    assert [type(m) for m in ensemble.members] == [SeededHead, SeededHead]
    assert [m.config.random_seed for m in ensemble.members] == [7, 3]
    assert ensemble.seeds == (7, 3)
    assert all(m is not model for m in ensemble.members)
    assert ensemble.members[0].get_config() == {**model.get_config(), "random_seed": 7}
    assert ensemble.labels == model.labels
    assert ensemble.train_bounds == model.train_bounds
    assert ensemble.test_bounds == model.test_bounds
    assert ensemble.label_delays == model.label_delays


# --------------------------------------------------------------------------
# Shared panel and predictions
# --------------------------------------------------------------------------


def test_members_share_one_collected_panel(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS)

    ensemble.collect()

    panel = ensemble.members[0].collected_panel
    assert all(m.collected_panel is panel for m in ensemble.members)
    assert panel.data.sizes["timestamp"] > 0


def test_seeds_give_different_members_and_the_prediction_is_their_average(tmp_path):
    ensemble, _, _, bars = _trained(tmp_path)
    start, end = _day(bars[30]), _day(bars[50])

    members = [m.predict_window(start, end) for m in ensemble.members]
    out = ensemble.predict_window(start, end)

    for a, b in zip(members, members[1:]):
        assert not np.allclose(a["fwd_ret_1"].values, b["fwd_ret_1"].values, equal_nan=True)
    xr.testing.assert_allclose(out, average_predictions(members))
    assert out.timestamp.values[0] == bars[30] and out.timestamp.values[-1] == bars[50]


# --------------------------------------------------------------------------
# train() on disk
# --------------------------------------------------------------------------


def test_train_writes_an_ensemble_unit_of_member_units(tmp_path):
    ensemble, checkpoint, _, bars = _trained(tmp_path)
    directory = checkpoint.parent

    assert checkpoint.is_absolute() and checkpoint.name == "run.json"
    assert directory.parent == Path(ensemble.members[0].config.model_save_dir).absolute()
    assert directory.name.startswith("SeedEnsemble_trial_")
    assert sorted(p.name for p in directory.iterdir()) == [
        "ic_series.csv",
        "member_0",
        "member_1",
        "member_2",
        "run.json",
        "test_predictions.zarr",
    ]

    run = TrainedRun.open(checkpoint)
    assert run.kind == "ensemble" and run.checkpoint == checkpoint
    dates = _dates(bars)
    assert run.train_window == (dates["train_start"], dates["train_end"])
    assert run.test_window == (dates["test_start"], dates["test_end"])
    assert run.fitted_train_window == ensemble.fitted_train_bounds
    assert [member.seed for member in run.members] == list(SEEDS)
    for k, (member, seed) in enumerate(zip(run.members, SEEDS)):
        assert member.kind == "model"
        assert member.path == directory / f"member_{k}"
        assert member.checkpoint.name == f"SeededHead_member_{k}.joblib"
        assert sorted(p.name for p in member.path.iterdir()) == sorted(
            [f"SeededHead_member_{k}.joblib", *MEMBER_FILES]
        )
        assert member.config["name"] == "tests.backtest_fixtures.SeededHead"
        assert member.config["random_seed"] == seed
        assert TrainedRun.open(member.path) == dataclasses.replace(member, seed=None)


def test_two_trainings_get_two_directories(tmp_path):
    ensemble, first, _, _ = _trained(tmp_path)
    second = ensemble.train()
    assert first.parent != second.parent
    assert first.is_file() and second.is_file()


def test_a_member_failing_leaves_no_ensemble_record(tmp_path, monkeypatch):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS)
    fit = SeededHead._fit_model

    def failing(self, train_rows, val_rows):
        if self.config.random_seed == 1:
            raise RuntimeError("member 1 broke")
        fit(self, train_rows, val_rows)

    monkeypatch.setattr(SeededHead, "_fit_model", failing)

    with pytest.raises(RuntimeError, match="member 1 broke"):
        ensemble.collect().train()

    root = Path(ensemble.members[0].config.model_save_dir)
    (directory,) = root.glob("SeedEnsemble_trial_*")
    assert (directory / "member_0" / "SeededHead_member_0.joblib").is_file()
    assert not (directory / "run.json").exists()


# --------------------------------------------------------------------------
# load() and check_checkpoint()
# --------------------------------------------------------------------------


def test_load_restores_every_member(tmp_path):
    trained, checkpoint, dataset_config, bars = _trained(tmp_path)
    fresh = SeedEnsemble(_model(tmp_path, dataset_config, bars, name="other"), SEEDS)

    assert fresh.load(checkpoint) is fresh
    assert fresh.load(str(checkpoint)) is fresh

    start, end = _day(bars[30]), _day(bars[50])
    xr.testing.assert_identical(
        fresh.predict_window(start, end), trained.predict_window(start, end)
    )


def test_check_checkpoint_accepts_the_unit_it_wrote(tmp_path):
    ensemble, checkpoint, _, _ = _trained(tmp_path)
    assert ensemble.check_checkpoint(checkpoint) == TrainedRun.open(checkpoint)
    assert ensemble.check_checkpoint(checkpoint.parent).kind == "ensemble"


def _rewrite(record: Path, edit) -> Path:
    saved = json.loads(record.read_text())
    edit(saved)
    record.write_text(json.dumps(saved))
    return record


def test_check_checkpoint_refuses_bad_units(tmp_path):
    ensemble, checkpoint, dataset_config, bars = _trained(tmp_path)
    original = checkpoint.read_text()

    with pytest.raises(FileNotFoundError):
        ensemble.check_checkpoint(checkpoint.parent / "missing.json")

    _rewrite(checkpoint, lambda saved: saved.update(format_version=99))
    with pytest.raises(ValueError, match="format_version 99.*retrain"):
        ensemble.check_checkpoint(checkpoint)

    checkpoint.write_text(original)
    _rewrite(checkpoint, lambda saved: saved["members"][2].update(seed=9))
    with pytest.raises(ValueError, match="seed"):
        ensemble.check_checkpoint(checkpoint)

    checkpoint.write_text(original)
    member_run = checkpoint.parent / "member_0" / "run.json"
    with pytest.raises(ValueError, match="'model' trained run, not an ensemble"):
        ensemble.check_checkpoint(member_run)

    two = SeedEnsemble(_model(tmp_path, dataset_config, bars, name="two"), [0, 1])
    with pytest.raises(ValueError, match="3 members"):
        two.check_checkpoint(checkpoint)

    other_class = SeedEnsemble(
        _model(tmp_path, dataset_config, bars, head=FirstFeatureHead, name="ffh"), SEEDS
    )
    with pytest.raises(ValueError, match="SeededHead"):
        other_class.check_checkpoint(checkpoint)

    (checkpoint.parent / "member_1" / "SeededHead_member_1.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="member_1"):
        ensemble.check_checkpoint(checkpoint)
    with pytest.raises(FileNotFoundError, match="member_1"):
        ensemble.load(checkpoint)

    (checkpoint.parent / "member_2" / "run.json").unlink()
    with pytest.raises(ValueError, match="member_2 has no run.json.*retrain"):
        ensemble.check_checkpoint(checkpoint)


def test_an_old_ensemble_layout_is_refused_with_a_retrain_message(tmp_path):
    ensemble, checkpoint, _, _ = _trained(tmp_path)
    old = checkpoint.parent / "ensemble.json"
    old.write_text(json.dumps({"format_version": 1, "members": []}))
    checkpoint.unlink()

    with pytest.raises(ValueError, match="no run.json.*retrain"):
        ensemble.load(old)


# --------------------------------------------------------------------------
# get_config / from_config
# --------------------------------------------------------------------------


def test_config_round_trips_the_wrapped_model_and_seeds(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    model = _model(tmp_path, dataset_config, bars)
    ensemble = SeedEnsemble(model, [5, 1, 4])

    config = ensemble.get_config()

    assert config == {
        "name": "quantlab.model.predefined.seed_ensemble.SeedEnsemble",
        "seeds": [5, 1, 4],
        "model": model.get_config(),
    }
    saved = json.loads(json.dumps(to_jsonable(config)))
    rebuilt = SeedEnsemble.from_config(saved)
    assert isinstance(rebuilt, SeedEnsemble)
    assert to_jsonable(rebuilt.get_config()) == to_jsonable(config)
    assert [type(m) for m in rebuilt.members] == [SeededHead] * 3
    assert [m.config.random_seed for m in rebuilt.members] == [5, 1, 4]


# --------------------------------------------------------------------------
# Package layout and layering
# --------------------------------------------------------------------------


def test_model_package_layout_and_layering():
    """The model package's `__init__.py` files are empty and it imports no
    backtest module. Which layers may import it is the layer order's
    (`tests/test_layer_layout.py`, ADR 0022)."""
    package = REPO_ROOT / "quantlab/model"
    for init in (
        package / "__init__.py",
        package / "predefined/__init__.py",
        package / "predefined/_support/__init__.py",
    ):
        assert init.stat().st_size == 0, init

    # Positive control: the resolver sees the package's own imports.
    assert "quantlab.model.predefined.xgb" in _resolved_imports(
        package / "predefined/xgb_td.py"
    )

    for path in _python_files(package):
        backtest = sorted(
            name
            for name in _resolved_imports(path)
            if _is_or_under(name, "quantlab.backtest")
            or _is_or_under(name, "quantlab.backtest.base")
        )
        assert backtest == [], (path, backtest)
