"""The Weights & Biases tracker; the only module that imports wandb (ADR 0015).

A run maps onto a W&B run of the same project, group and name. Step metrics
are logged against their step, the summary goes to the run summary, a table
to a ``wandb.Table`` (plus a bar chart of its top rows when asked), an HTML
file to a ``wandb.Html`` panel named after the file's stem, and any other
file is saved with the run. The API key and the rest of the credentials are
read by wandb from its environment variables, never from the config.
"""

from dataclasses import dataclass
from typing import Literal, get_args

try:
    import wandb
except ImportError as exc:
    raise ImportError(
        "WandbTracker needs wandb, an optional extra of quantlab: install it "
        "with `uv sync --extra wandb` or `pip install 'quantlab[wandb]'`."
    ) from exc

from quantlab.tracking.base import Tracker, TrackingRun

__all__ = ["WandbRun", "WandbTracker"]

_Mode = Literal["online", "offline", "disabled"]
_MODES = get_args(_Mode)


class WandbRun(TrackingRun):
    """A tracking run backed by one W&B run; opened by ``WandbTracker``.

    Tables and HTML files are logged at the last step logged so far (step 0
    before any), so they never move W&B's step counter backwards.

    Examples
    --------
    >>> tracker = WandbTracker(mode="disabled")
    >>> with tracker.start_run(project="P", group=None, name="n", config={}) as run:
    ...     isinstance(run, WandbRun)
    True
    """

    def __init__(self, run):
        """Wrap the run ``wandb.init`` returned."""
        self._run = run
        self._step = 0

    def _log(self, metrics, step):
        self._run.log(metrics, step=step)
        self._step = max(self._step, step)

    def _summarize(self, metrics):
        self._run.summary.update(metrics)

    def _update_config(self, params):
        self._run.config.update(params, allow_val_change=True)

    def _log_table(self, name, columns, rows, top_bars):
        payload = {name: wandb.Table(columns=columns, data=rows)}
        if top_bars is not None:
            top = rows[:top_bars]
            payload[f"{name}_chart"] = wandb.plot.bar(
                wandb.Table(columns=columns, data=top),
                columns[0],
                columns[1],
                title=f"{name} (top {len(top)} of {len(rows)})",
            )
        self._run.log(payload, step=self._step)

    def _log_file(self, path):
        if path.suffix.lower() in (".html", ".htm"):
            self._run.log({path.stem: wandb.Html(path.read_text())}, step=self._step)
        else:
            self._run.save(str(path), base_path=str(path.parent), policy="now")

    def _finish(self, *, failed):
        self._run.finish(exit_code=1 if failed else 0)


@dataclass(frozen=True, kw_only=True)
class WandbTracker(Tracker):
    """Send tracking runs to Weights & Biases.

    Attributes
    ----------
    project : str, optional
        The W&B project every run goes to; ``None`` keeps the caller's
        default.
    entity : str, optional
        The W&B team or user owning the project; ``None`` uses the account's
        default entity.
    mode : {"online", "offline", "disabled"}
        ``"online"`` syncs as it runs, ``"offline"`` writes the run under
        ``wandb/`` for a later ``wandb sync`` (``WANDB_DIR`` moves it), and
        ``"disabled"`` records nothing.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three modes.

    Examples
    --------
    >>> tracker = WandbTracker(project="momentum_research", mode="disabled")
    >>> tracker.get_config()
    {'project': 'momentum_research', 'entity': None, 'mode': 'disabled', 'name': 'quantlab.tracking.wandb.WandbTracker'}
    >>> with tracker.start_run(
    ...     project="XGBoostRegressor", group="XGBoostRegressor_trial_1", name="XGBoostRegressor_total", config={}
    ... ) as run:
    ...     run.log({"val_rmse": 0.02}, step=0)
    """

    entity: str | None = None
    mode: _Mode = "online"

    def __post_init__(self):
        """Refuse an unknown ``mode``."""
        if self.mode not in _MODES:
            raise ValueError(f"WandbTracker mode must be one of {_MODES}, got {self.mode!r}")

    def _open(self, *, project, group, name, config):
        return WandbRun(
            wandb.init(
                project=project,
                entity=self.entity,
                group=group,
                name=name,
                config=config,
                mode=self.mode,
                reinit="create_new",
            )
        )
