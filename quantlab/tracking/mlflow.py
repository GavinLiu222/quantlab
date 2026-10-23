"""The MLflow tracker; the only module that imports mlflow.

mlflow is an optional extra (``uv sync --extra mlflow`` or
``pip install 'quantlab-py[mlflow]'``) and is imported only when an
``MlflowTracker`` opens a run, so the library imports, and a config naming
an ``MlflowTracker`` is rebuilt, without it.

A run maps onto MLflow as follows. The project is an experiment, created
when missing. The group is a ``group`` tag on the run, not a parent run, so
the runs of one trial stay flat and are filtered with
``tags.group = '<trial>'``. The name is the run name. Step metrics are
metrics logged at their step; summary values are metrics logged once, at the
last step logged so far, so they are the latest value of their key. The
config becomes params, flattened to ``outer/inner`` keys, with every value
that is not a string written as JSON and cut to the 6000 characters MLflow
takes; the whole config, values uncut, is also kept as the artifact
``run_config.json``. MLflow refuses to change a param once logged, so
``update_config`` logs new keys as params, and a key logged before with
another value is changed only in ``run_config.json``. A table is the JSON
artifact ``tables/<name>.json`` (``{"columns": [...], "data": [...]}``, the
format of ``mlflow.log_table``); no bar chart is drawn. A file is an artifact
at the root of the run. Leaving the run sets it ``FINISHED`` or ``FAILED``.
Every param and metric key is cleaned of the characters MLflow refuses
(``whole/Total Return [%]`` is logged as ``whole/Total Return ___``, see
``_mlflow_key``).

Every call goes through an ``MlflowClient`` bound to the tracker's
``tracking_uri``, never through mlflow's global active run, so runs opened one
inside another stay apart. Credentials (``MLFLOW_TRACKING_USERNAME``,
``MLFLOW_TRACKING_PASSWORD``, ``MLFLOW_TRACKING_TOKEN``) are read by mlflow
from the environment, never from the config.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from quantlab.tracking.base import Tracker, TrackingRun

__all__ = ["MlflowRun", "MlflowTracker"]

#: The longest param value MLflow takes.
_MAX_PARAM_LENGTH = 6000
#: The most params, or metrics, MLflow takes in one batch call.
_MAX_PARAMS_PER_BATCH = 100
_MAX_METRICS_PER_BATCH = 1000
#: Artifact holding the run's whole config, values uncut.
_CONFIG_ARTIFACT = "run_config.json"
#: The longest param or metric key MLflow takes.
_MAX_KEY_LENGTH = 250
#: Characters MLflow refuses in a param or metric key.
_BAD_KEY_CHARACTER = re.compile(r"[^/\w.\- :]")


def _mlflow_key(name: str) -> str:
    """Return ``name`` as a param or metric key MLflow takes.

    The name is cut to the length MLflow takes, each refused character
    becomes ``_``, and the ``/``-separated parts are kept non-empty and other
    than ``.`` or ``..``, since MLflow also refuses a key that would resolve
    to another path. Two names cleaned to one key in one call share it, the
    later value winning.

    Examples
    --------
    >>> _mlflow_key("whole/Total Return [%]"), _mlflow_key("./odd//key/")
    ('whole/Total Return ___', '_/odd/key')
    """
    cut = _BAD_KEY_CHARACTER.sub("_", name[:_MAX_KEY_LENGTH])
    parts = ["_" if part in (".", "..") else part for part in cut.split("/") if part]
    return "/".join(parts) or "_"


def _import_mlflow():
    """Return the ``mlflow`` module, or raise naming the extra that installs it."""
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MlflowTracker needs mlflow, an optional extra of quantlab: install it "
            "with `uv sync --extra mlflow` or `pip install 'quantlab-py[mlflow]'`."
        ) from exc
    return mlflow


def _flatten_params(config: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts to ``outer/inner`` keys with JSON string values.

    A string stays as it is; every other value is written as JSON. Keys are
    cleaned (see ``_mlflow_key``) and values cut to the length MLflow takes.

    Examples
    --------
    >>> _flatten_params({"hp": {"eta": 0.1}, "names": ["a"], "seed": None, "kind": "xgb"})
    {'hp/eta': '0.1', 'names': '["a"]', 'seed': 'null', 'kind': 'xgb'}
    """
    out: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(_flatten_params(value, f"{name}/"))
        else:
            text = value if isinstance(value, str) else json.dumps(value)
            out[_mlflow_key(name)] = text[:_MAX_PARAM_LENGTH]
    return out


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


class MlflowRun(TrackingRun):
    """A tracking run backed by one MLflow run; opened by ``MlflowTracker``.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> store = Path(tempfile.mkdtemp()) / "mlruns"
    >>> tracker = MlflowTracker(tracking_uri=f"file:{store}")
    >>> with tracker.start_run(project="P", group="P_trial_1", name="n", config={}) as run:
    ...     isinstance(run, MlflowRun)
    True
    """

    def __init__(self, mlflow, client, run_id: str, config: dict):
        """Wrap the run ``run_id`` of ``client``, opened with ``config``."""
        self._mlflow = mlflow
        self._client = client
        self._run_id = run_id
        self._config = dict(config)
        self._params = _flatten_params(self._config)
        self._step = 0
        self._log_params(self._params)
        self._client.log_dict(self._run_id, self._config, _CONFIG_ARTIFACT)

    def _log_params(self, params: dict[str, str]) -> None:
        Param = self._mlflow.entities.Param
        entries = [Param(key, value) for key, value in params.items()]
        for chunk in _chunks(entries, _MAX_PARAMS_PER_BATCH):
            self._client.log_batch(self._run_id, params=chunk)

    def _log_metrics(self, metrics: dict, step: int) -> None:
        Metric = self._mlflow.entities.Metric
        stamp = int(time.time() * 1000)
        cleaned = {_mlflow_key(key): float(value) for key, value in metrics.items()}
        entries = [Metric(key, value, stamp, step) for key, value in cleaned.items()]
        for chunk in _chunks(entries, _MAX_METRICS_PER_BATCH):
            self._client.log_batch(self._run_id, metrics=chunk)

    def _log(self, metrics, step):
        self._log_metrics(metrics, step)
        self._step = max(self._step, step)

    def _summarize(self, metrics):
        self._log_metrics(metrics, self._step)

    def _update_config(self, params):
        self._config.update(params)
        new = {
            key: value
            for key, value in _flatten_params(params).items()
            if key not in self._params
        }
        self._params.update(new)
        self._log_params(new)
        self._client.log_dict(self._run_id, self._config, _CONFIG_ARTIFACT)

    def _log_table(self, name, columns, rows, top_bars):
        self._client.log_dict(
            self._run_id, {"columns": columns, "data": rows}, f"tables/{name}.json"
        )

    def _log_file(self, path):
        self._client.log_artifact(self._run_id, str(path))

    def _finish(self, *, failed):
        self._client.set_terminated(self._run_id, "FAILED" if failed else "FINISHED")


@dataclass(frozen=True, kw_only=True)
class MlflowTracker(Tracker):
    """Send tracking runs to MLflow.

    mlflow is imported when a run opens; see the module docstring for how a
    run maps onto MLflow.

    Attributes
    ----------
    project : str, optional
        The MLflow experiment every run goes to, created when missing;
        ``None`` keeps the caller's default.
    tracking_uri : str, optional
        Where the experiments live: a server (``http://host:5000``), a
        database (``sqlite:///mlflow.db``, which needs the full ``mlflow``
        package) or a local directory (``file:/path/to/mlruns``). MLflow 3
        opens a local directory only with ``MLFLOW_ALLOW_FILE_STORE`` set;
        naming a ``file:`` URI here sets it to ``true`` for the process
        unless it is already set. ``None`` uses mlflow's own default,
        ``MLFLOW_TRACKING_URI`` when set.

    Raises
    ------
    ImportError
        When a run opens without mlflow installed; the message names the
        extra to install.

    Examples
    --------
    A model trains into a local MLflow store when only the tracker of its
    config changes. With ``config`` a ``ModelConfig`` built earlier:

    >>> import dataclasses, tempfile
    >>> from pathlib import Path
    >>> from mlflow import MlflowClient
    >>> from quantlab.model.predefined.xgb import XGBoostRegressor
    >>> store = f"file:{Path(tempfile.mkdtemp()) / 'mlruns'}"
    >>> tracker = MlflowTracker(tracking_uri=store)
    >>> model = XGBoostRegressor(dataclasses.replace(config, tracker=tracker))
    >>> checkpoint = model.collect().train()
    >>> client = MlflowClient(tracking_uri=store)
    >>> (run,) = client.search_runs(
    ...     [client.get_experiment_by_name("XGBoostRegressor").experiment_id]
    ... )
    >>> from quantlab.runs.trained_run import TrainedRun
    >>> group = TrainedRun.open(checkpoint).path.name
    >>> run.info.run_name, run.info.status, run.data.tags["group"] == group
    ('XGBoostRegressor_total', 'FINISHED', True)
    >>> "test_ic" in run.data.metrics
    True
    """

    tracking_uri: str | None = None

    def _open(self, *, project, group, name, config):
        mlflow = _import_mlflow()
        if self.tracking_uri is not None and urlparse(self.tracking_uri).scheme in ("", "file"):
            # Naming a local store is the opt-in MLflow 3 asks for.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name(project)
        experiment_id = (
            client.create_experiment(project)
            if experiment is None
            else experiment.experiment_id
        )
        tags = {} if group is None else {"group": group}
        run = client.create_run(experiment_id, run_name=name, tags=tags)
        return MlflowRun(mlflow, client, run.info.run_id, config)
