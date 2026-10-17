"""The tracker seam (spec #99, tickets #100 and #104): one contract for every tracker.

The same operations run against the null tracker, the recording tracker of
the tests, the W&B tracker (offline mode, a temporary directory) and the
MLflow tracker (a ``file:`` store in a temporary directory, skipped without
mlflow), all without network: open a run, log steps, summarise nested
metrics, update the config, log a table and a file, and finish on exit and
on error. Each tracker comes with a reader that returns what an outsider can
observe of the runs it opened; the W&B reader parses the offline run's
transaction log and the MLflow reader asks an ``MlflowClient``. The null
tracker leaves nothing to read, and its reader asserts exactly that.
"""

import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quantlab.tracking.base import NullRun, NullTracker, Tracker
from quantlab.tracking.wandb import WandbTracker
from quantlab.core.component import get_cls_from_path
from tests.test_backtest_contracts import REPO_ROOT
from tests.tracking_fixtures import RecordingTracker


def _read_recorded(tracker, workdir):
    return [
        {
            "project": run.project,
            "group": run.group,
            "name": run.name,
            "config": _merged(run.config, run.config_updates),
            "steps": run.steps,
            "summary": run.summary,
            "tables": sorted(
                [*run.tables]
                + [f"{name}_chart" for name, (*_, top) in run.tables.items() if top]
            ),
            "files": [path.name for path in run.files],
            "finished": run.finished,
            "failed": run.failed,
        }
        for run in tracker.runs
    ]


def _merged(config, updates):
    merged = dict(config)
    for update in updates:
        merged.update(update)
    return merged


def _read_wandb(tracker, workdir):
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    runs = []
    for run_dir in sorted((workdir / "wandb").glob("offline-run-*")):
        (log,) = run_dir.glob("*.wandb")
        store = datastore.DataStore()
        store.open_for_scan(str(log))
        observed = {"steps": [], "summary": {}, "config": {}, "tables": [], "files": []}
        history_keys = set()
        while (data := store.scan_data()) is not None:
            record = pb.Record()
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            if kind == "run":
                observed["start"] = record.run.start_time.ToNanoseconds()
                observed["project"] = record.run.project
                observed["group"] = record.run.run_group or None
                observed["name"] = record.run.display_name
                for item in record.run.config.update:
                    observed["config"][item.key] = json.loads(item.value_json)
            elif kind == "config":
                # Keyless items are W&B's own chart settings, not run config.
                for item in record.config.update:
                    if item.key:
                        observed["config"][item.key] = json.loads(item.value_json)
            elif kind == "summary":
                for item in record.summary.update:
                    if item.key and not item.key.startswith("_"):
                        observed["summary"][item.key] = json.loads(item.value_json)
            elif kind == "history":
                # A media value (table, chart, HTML) arrives as several items
                # under one nested key; a scalar as one item.
                values, media = {}, set()
                for item in record.history.item:
                    key = item.nested_key[0] if item.nested_key else item.key
                    if len(item.nested_key) > 1:
                        media.add(key)
                    else:
                        values[key] = json.loads(item.value_json)
                metrics = {
                    key: value
                    for key, value in values.items()
                    if not key.startswith("_") and key not in media
                }
                if metrics:
                    observed["steps"].append((values["_step"], metrics))
                history_keys.update(media)
            elif kind == "files":
                # Media files back logged tables and HTML; keep saved files.
                observed["files"].extend(
                    Path(item.path).name
                    for item in record.files.files
                    if not item.path.startswith("media/")
                )
            elif kind == "exit":
                observed["failed"] = record.exit.exit_code != 0
        observed["finished"] = "failed" in observed
        # An HTML file is logged as a media panel named after its stem, and a
        # chart as ``<key>_table``.
        observed["tables"] = sorted(
            key.removesuffix("_chart_table") + "_chart" if key.endswith("_chart_table") else key
            for key in history_keys
            if key != "report"
        )
        observed["files"] = [
            name for name in observed["files"] if name != "requirements.txt"
        ] + (["report.html"] if "report" in history_keys else [])
        observed["config"].pop("_wandb", None)
        # A summary written by ``log`` echoes the last step; keep what
        # ``summarize`` wrote, the keys no step carries.
        stepped = {key for _, metrics in observed["steps"] for key in metrics}
        observed["summary"] = {
            key: value
            for key, value in observed["summary"].items()
            if key not in stepped and key not in history_keys
        }
        runs.append(observed)
    # Run directories are named to the second plus a random id: order by start.
    runs.sort(key=lambda run: run.pop("start"))
    return runs


def _read_null(tracker, workdir):
    assert list(workdir.iterdir()) == [], "the null tracker wrote files"
    return None


def _read_mlflow(tracker, workdir):
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    runs = []
    for experiment in client.search_experiments():
        for run in client.search_runs(
            [experiment.experiment_id], order_by=["attributes.start_time ASC"]
        ):
            run_id = run.info.run_id
            history = {
                key: client.get_metric_history(run_id, key) for key in run.data.metrics
            }
            # A summary value is logged once; a step metric may be logged
            # once too, but none of the contract's step metrics is.
            steps: dict[int, dict] = {}
            for key, points in history.items():
                if len(points) > 1:
                    for point in points:
                        steps.setdefault(point.step, {})[key] = point.value
            artifacts = [a.path for a in client.list_artifacts(run_id)]
            local = Path(client.download_artifacts(run_id, "run_config.json", str(workdir.parent)))
            runs.append(
                {
                    "project": experiment.name,
                    "group": run.data.tags.get("group"),
                    "name": run.info.run_name,
                    "config": json.loads(local.read_text()),
                    "params": run.data.params,
                    "steps": sorted(steps.items()),
                    "summary": {
                        key: points[0].value
                        for key, points in history.items()
                        if len(points) == 1
                    },
                    "tables": sorted(_mlflow_tables(client, run_id, "tables")),
                    "files": [
                        path for path in artifacts if path not in ("tables", "run_config.json")
                    ],
                    "finished": run.info.status in ("FINISHED", "FAILED"),
                    "failed": run.info.status == "FAILED",
                }
            )
    return runs


def _mlflow_tables(client, run_id, folder):
    for artifact in client.list_artifacts(run_id, folder):
        if artifact.is_dir:
            yield from _mlflow_tables(client, run_id, artifact.path)
        else:
            yield artifact.path.removeprefix("tables/").removesuffix(".json")


def _make_mlflow(store):
    pytest.importorskip("mlflow")
    from quantlab.tracking.mlflow import MlflowTracker

    return MlflowTracker(tracking_uri=f"file:{store}")


TRACKERS = {
    "null": (lambda store: NullTracker(), _read_null),
    "recording": (lambda store: RecordingTracker(), _read_recorded),
    "wandb": (lambda store: WandbTracker(mode="offline"), _read_wandb),
    "mlflow": (_make_mlflow, _read_mlflow),
}


@pytest.fixture(params=sorted(TRACKERS))
def adapter(request, tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("WANDB_DIR", str(workdir))
    # Unset, so the MLflow tracker must open its `file:` store by itself; the
    # value it sets is undone after the test.
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    make, read = TRACKERS[request.param]
    tracker = make(tmp_path / "mlruns")
    yield tracker, (lambda: read(tracker, workdir)), tmp_path
    # wandb reads WANDB_DIR once per process; reset it for the next test.
    import wandb

    wandb.teardown()


def test_a_run_records_steps_summary_config_table_and_file(adapter):
    tracker, read, tmp_path = adapter
    report = tmp_path / "report.html"
    report.write_text("<p>report</p>")
    notes = tmp_path / "notes.txt"
    notes.write_text("notes")

    with tracker.start_run(
        project="XGBoostRegressor",
        group="XGBoostRegressor_trial_1",
        name="XGBoostRegressor_total",
        config={"lr": np.float64(0.1), "start": np.datetime64("2020-01-02")},
    ) as run:
        run.log({"train_loss": 1.0, "val_loss": 2.0}, step=0)
        run.log({"train_loss": 0.5, "val_loss": 1.5}, step=1)
        run.summarize(
            {
                "whole": {"sharpe": 1.25, "trades": np.int64(3), "worst": float("nan")},
                "test_ic": np.float32(0.5),
                "note": "text",
                "flag": True,
                "inf": math.inf,
            }
        )
        run.update_config({"n_estimators": 42})
        run.log_table(
            "importance/gain", ["factor", "importance"], [["a", 2.0], ["b", 1.0]], top_bars=1
        )
        run.log_file(report)
        run.log_file(notes)

    observed = read()
    if observed is None:
        return
    (only,) = observed
    assert only["project"] == "XGBoostRegressor"
    assert only["group"] == "XGBoostRegressor_trial_1"
    assert only["name"] == "XGBoostRegressor_total"
    assert only["config"] == {"lr": 0.1, "start": "2020-01-02T00:00:00", "n_estimators": 42}
    assert only["steps"] == [
        (0, {"train_loss": 1.0, "val_loss": 2.0}),
        (1, {"train_loss": 0.5, "val_loss": 1.5}),
    ]
    assert only["summary"] == {"whole/sharpe": 1.25, "whole/trades": 3, "test_ic": 0.5}
    # MLflow draws no bar chart; W&B does, and the recording tracker records the ask.
    chart = [] if type(tracker).__name__ == "MlflowTracker" else ["importance/gain_chart"]
    assert only["tables"] == ["importance/gain", *chart]
    assert sorted(only["files"]) == ["notes.txt", "report.html"]
    assert only["finished"] and not only["failed"]


def test_a_run_is_finished_when_the_body_raises(adapter):
    tracker, read, _ = adapter
    with pytest.raises(ZeroDivisionError):
        with tracker.start_run(project="P", group=None, name="boom", config={}) as run:
            run.log({"loss": 1.0}, step=0)
            1 / 0
    observed = read()
    if observed is None:
        return
    (only,) = observed
    assert only["name"] == "boom"
    assert only["group"] is None
    assert only["finished"] and only["failed"]


def test_runs_opened_in_turn_are_kept_apart(adapter):
    tracker, read, _ = adapter
    for fold in range(2):
        with tracker.start_run(
            project="P", group="P_trial_1", name=f"fold_{fold}", config={"fold": fold}
        ) as run:
            run.summarize({"test_ic": float(fold)})
    observed = read()
    if observed is None:
        return
    assert [(run["name"], run["config"], run["summary"]) for run in observed] == [
        ("fold_0", {"fold": 0}, {"test_ic": 0.0}),
        ("fold_1", {"fold": 1}, {"test_ic": 1.0}),
    ]


def test_the_trackers_own_project_overrides_the_callers_default(adapter):
    tracker, read, _ = adapter
    # ``replace`` keeps a recording tracker's ``runs`` list, so ``read`` sees it.
    tracker = dataclasses.replace(tracker, project="research_thread")
    with tracker.start_run(project="XGBoostRegressor", group="g", name="n", config={}):
        pass
    observed = read()
    if observed is None:
        return
    assert [run["project"] for run in observed] == ["research_thread"]


def test_logging_a_missing_file_raises(adapter):
    tracker, _, tmp_path = adapter
    with pytest.raises(FileNotFoundError):
        with tracker.start_run(project="P", group=None, name="n", config={}) as run:
            run.log_file(tmp_path / "absent.html")


@pytest.mark.parametrize(
    "tracker",
    [NullTracker(), NullTracker(project="p"), WandbTracker(), WandbTracker(
        project="p", entity="team", mode="disabled"
    )],
    ids=repr,
)
def test_a_tracker_round_trips_through_its_config(tracker):
    config = tracker.get_config()
    assert json.loads(json.dumps(config)) == config
    rebuilt = get_cls_from_path(config["name"]).from_config(config)
    assert rebuilt == tracker
    assert type(rebuilt) is type(tracker)


def test_the_wandb_tracker_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        WandbTracker(mode="sometimes")


def test_the_null_run_is_usable_outside_any_tracker():
    run = NullRun()
    run.log({"loss": 1.0}, step=0)
    run.summarize({"a": {"b": 1.0}})
    run.update_config({"x": 1})
    run.log_table("t", ["a"], [[1]])


def test_tracker_is_abstract():
    with pytest.raises(TypeError):
        Tracker()


class _BrokenFinishRun(NullRun):
    def _finish(self, *, failed):
        raise ConnectionError("tracker unreachable")


@dataclasses.dataclass(frozen=True, kw_only=True)
class _BrokenFinishTracker(Tracker):
    def _open(self, *, project, group, name, config):
        return _BrokenFinishRun()


def test_an_error_while_finishing_does_not_replace_the_bodys_exception():
    with pytest.raises(ZeroDivisionError):
        with _BrokenFinishTracker().start_run(project="P", group=None, name="n", config={}):
            1 / 0


def test_an_error_while_finishing_a_clean_run_propagates():
    with pytest.raises(ConnectionError):
        with _BrokenFinishTracker().start_run(project="P", group=None, name="n", config={}):
            pass


# --------------------------------------------------------------------------
# MLflow specifics: params, changed config values, the optional extra
# --------------------------------------------------------------------------


@pytest.fixture
def mlflow_tracker(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    return _make_mlflow(tmp_path / "mlruns")


def test_mlflow_params_are_the_flattened_config_as_strings(mlflow_tracker, tmp_path):
    with mlflow_tracker.start_run(
        project="P",
        group="g",
        name="n",
        config={
            "hyperparameters": {"max_depth": 6, "eta": 0.1},
            "factors": ["a", "b"],
            "seed": None,
        },
    ) as run:
        run.update_config({"resolved_hyperparameters": {"n_estimators": 42}})

    (observed,) = _read_mlflow(mlflow_tracker, tmp_path)
    assert observed["params"] == {
        "hyperparameters/max_depth": "6",
        "hyperparameters/eta": "0.1",
        "factors": '["a", "b"]',
        "seed": "null",
        "resolved_hyperparameters/n_estimators": "42",
    }


def test_mlflow_keeps_a_changed_config_value_in_the_config_artifact(mlflow_tracker, tmp_path):
    with mlflow_tracker.start_run(project="P", group=None, name="n", config={"lr": 0.1}) as run:
        run.update_config({"lr": 0.2})

    (observed,) = _read_mlflow(mlflow_tracker, tmp_path)
    assert observed["params"] == {"lr": "0.1"}
    assert observed["config"] == {"lr": 0.2}


def test_mlflow_cuts_a_param_value_longer_than_mlflow_takes(mlflow_tracker, tmp_path):
    long = "x" * 7000
    with mlflow_tracker.start_run(project="P", group=None, name="n", config={"long": long}):
        pass

    (observed,) = _read_mlflow(mlflow_tracker, tmp_path)
    assert len(observed["params"]["long"]) == 6000
    assert observed["config"] == {"long": long}


def test_mlflow_cleans_keys_it_would_refuse(mlflow_tracker, tmp_path):
    """Backtest summaries and fingerprints carry `[ ] %` and nested `/`."""
    with mlflow_tracker.start_run(
        project="P",
        group=None,
        name="n",
        config={"data_fingerprint": {"factor[0]:PastReturn": {"algorithm": "sha256"}}},
    ) as run:
        run.log({"val-rmse/ret_5": 1.0}, step=0)
        run.log({"val-rmse/ret_5": 0.5}, step=1)
        run.summarize({"whole": {"Total Return [%]": 12.5}, "./odd//key/": 1.0})

    (observed,) = _read_mlflow(mlflow_tracker, tmp_path)
    assert observed["params"] == {"data_fingerprint/factor_0_:PastReturn/algorithm": "sha256"}
    assert observed["steps"] == [(0, {"val-rmse/ret_5": 1.0}), (1, {"val-rmse/ret_5": 0.5})]
    assert observed["summary"] == {"whole/Total Return ___": 12.5, "_/odd/key": 1.0}
    assert observed["finished"] and not observed["failed"]


def test_mlflow_keeps_the_later_of_two_metrics_cleaned_to_one_key(mlflow_tracker, tmp_path):
    with mlflow_tracker.start_run(project="P", group=None, name="n", config={}) as run:
        run.summarize({"x[": 5.0, "x]": 1.0})

    (observed,) = _read_mlflow(mlflow_tracker, tmp_path)
    assert observed["summary"] == {"x_": 1.0}


def test_mlflow_keeps_a_long_key_valid_after_cutting_it(mlflow_tracker, tmp_path):
    name = "a" * 249 + "/b"
    with mlflow_tracker.start_run(project="P", group=None, name="n", config={name: 1}) as run:
        run.summarize({name: 2.0})

    (observed,) = _read_mlflow(mlflow_tracker, tmp_path)
    assert observed["params"] == {"a" * 249: "1"}
    assert observed["summary"] == {"a" * 249: 2.0}


def test_an_explicitly_disabled_file_store_is_left_to_mlflow(mlflow_tracker, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    with pytest.raises(Exception, match="MLFLOW_ALLOW_FILE_STORE"):
        with mlflow_tracker.start_run(project="P", group=None, name="n", config={}):
            pass


def test_the_mlflow_tracker_round_trips_through_its_config():
    pytest.importorskip("mlflow")
    from quantlab.tracking.mlflow import MlflowTracker

    tracker = MlflowTracker(project="research", tracking_uri="file:/tmp/mlruns")
    config = tracker.get_config()
    assert config == {
        "project": "research",
        "tracking_uri": "file:/tmp/mlruns",
        "name": "quantlab.tracking.mlflow.MlflowTracker",
    }
    assert get_cls_from_path(config["name"]).from_config(config) == tracker


def test_without_mlflow_a_config_rebuilds_and_opening_a_run_names_the_extra():
    code = (
        "import sys\n"
        "sys.modules['mlflow'] = None\n"
        "import quantlab.model.base\n"
        "from quantlab.tracking.mlflow import MlflowTracker\n"
        "tracker = MlflowTracker.from_config(MlflowTracker(project='p').get_config())\n"
        "try:\n"
        "    with tracker.start_run(project='P', group=None, name='n', config={}):\n"
        "        pass\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "quantlab-py[mlflow]" in result.stdout
    assert "--extra mlflow" in result.stdout
