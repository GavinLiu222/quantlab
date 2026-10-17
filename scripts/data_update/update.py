"""The daily data update: raw downloads -> Zarr stores -> factors -> risk model.

Runs the stages of a YAML file (``config/data_update.yaml``) in order, under the data
root ``--data-dir``. A store a step names is a folder ``<category>/<group>/<stem>/``
holding ``<stem>.zarr``, its README.md and a ``component.json``: the component's
``get_config()``, from which the step rebuilds the dataset, factor or factor risk model
(``quantlab.core.component.rebuild``). Actions (``ACTIONS``):

- ``sharadar``: ``scripts/sharadar/update.py`` (every Sharadar table);
- ``fred``: download the FRED series from ``start``, then ``update()`` the store in the
  named folder;
- ``benchmarks``: ``scripts/sharadar/price_return_benchmark.py --refresh``;
- ``update``: ``update()`` the dataset of a store folder;
- ``mirror``: append a source dataset's new bars to a store copying its variables
  (``quantlab.backtest.live.mirror_new_bars``);
- ``extend``: extend to t every store the component declares it owns, written by date
  range (``Component.store_uses``: a factor's store, which only its owner may extend, or
  a factor risk model's regression and estimate stores, in that order);
- ``run``: any script under the repository, with ``args`` in which ``{data_dir}`` and
  ``{download_dir}`` are filled in.

A step may list ``writes``: the store folders or folder prefixes (``market/wrds/``) it
writes; ``sharadar``, ``benchmarks`` and ``run`` steps say so there, the others write
their ``store``. A ``run`` step may also list ``reads``. ``--check`` orders every step's
reads against the later steps' writes. A step with ``allow_failure: true`` that raises is
recorded as failed and the run goes on; whether the day is ready is still decided by the
``ready`` stores. A stage with ``after_done: true`` runs only once ``done`` is written (its
steps are added to the status file afterwards, the state kept), so nothing waiting for
``done`` waits for it.

t is the last bar of the file's ``calendar`` store once the raw stage is done. Until
every store of the file's ``ready`` list holds t, and t is newer than ``last_done_t``
(the t of the last run that reached ``done``), the raw stage is retried every
``--retry-minutes`` until ``--retry-until`` (New York time). The outcome is written to
``<data-dir>/update_status.json``: ``{"date", "t", "last_done_t", "state", "started",
"finished", "steps"}``, ``state`` being ``running``, ``done``, ``no_new_bar`` (the vendor
published nothing new by the cut-off, as on a holiday) or ``failed``. The paper trading
waits for ``done`` on its day.

Exit status: 0 ``done``, 1 ``failed`` (or a problem found by ``--check``), 3
``no_new_bar`` (normal on a holiday, but not a done day).

Usage::

    QUANTLAB_DATA_DIR=/data/quantlab python scripts/data_update/update.py \\
        --data-dir /data/quantlab --download-dir /data/quantlab/downloads

``SHARADAR_API_KEY`` must be set for the raw stage. ``--stage NAME`` runs only the
named stages (repeatable), without the retry; ``--dry-run`` prints the plan;
``--check`` downloads and computes nothing: it rebuilds every step's component and
checks that a factor owns its store, that no step reads a store a later step writes and
that the first stage writes the calendar and ready stores, and exits 1 listing the
problems.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import xarray as xr
import yaml
from loguru import logger

from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.fred import FredAcquisition
from quantlab.backtest.live import mirror_new_bars
from quantlab.core.store_uses import OWNS, StoreUse, walk_store_uses
from quantlab.core.store_folder import COMPONENT_FILE, load_component
from quantlab.dataset.base import BaseDataset
from quantlab.utils.cli import add_output_dir_args, resolve_output_dirs
from quantlab.utils.jsonable import to_jsonable

REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_YORK = ZoneInfo("America/New_York")
STATUS_FILE = "update_status.json"
#: The actions a step may name; each is a method of ``Update``.
ACTIONS = ("sharadar", "fred", "benchmarks", "update", "mirror", "extend", "run")
#: The actions whose step writes the folder named by its ``store``.
STORE_ACTIONS = ("update", "extend", "mirror", "fred")
#: The exit status of each final state.
EXIT_CODES = {"done": 0, "failed": 1, "no_new_bar": 3}


def now() -> str:
    """The current New York time, ISO format to the second."""
    return datetime.datetime.now(NEW_YORK).isoformat(timespec="seconds")


def store_path(folder: Path) -> Path:
    """``<folder>/<stem>.zarr`` of a store folder ``.../<stem>/``."""
    return folder / f"{folder.name}.zarr"


def parse_step(step: dict) -> tuple[str, dict, bool]:
    """A step's action, keyword arguments and ``allow_failure``.

    Parameters
    ----------
    step : dict
        One YAML step: ``{action: store}`` or ``{action: {key: value, ...}}``.

    Returns
    -------
    tuple of (str, dict, bool)
        The action, its keyword arguments (``allow_failure`` removed) and whether a
        failure is allowed.
    """
    (action, value), = step.items()
    kwargs = dict(value) if isinstance(value, dict) else {"store": value}
    allow_failure = bool(kwargs.pop("allow_failure", False))
    return action, kwargs, allow_failure


def stores_read(item) -> set[Path]:
    """The stores a component depends on: every store the components below it use.

    Every relation of ``walk_store_uses`` counts (``Component.store_uses``): a store
    read on a day, a store written by date range, a store another is written from (a
    mapped Trade bar store from the Trade bar store under its source view). The
    component's own declaration is left out: those are the stores its step writes.
    """
    return {Path(use.path).absolute() for use in walk_store_uses(item) if use.where and use.path}


def overlaps(a: Path, b: Path) -> bool:
    """Whether two store paths or folder prefixes are the same or one holds the other."""
    return a == b or a in b.parents or b in a.parents


def last_bar(path: Path) -> pd.Timestamp | None:
    """The last timestamp of a store, ``None`` when it does not exist or is empty."""
    if not path.exists():
        return None
    store = xr.open_zarr(path)
    try:
        values = store["timestamp"].values
        return pd.Timestamp(values[-1]) if len(values) else None
    finally:
        store.close()


def range_end(use, where: str) -> pd.Timestamp:
    """The end of a ranged store's recorded range; raises when there is none."""
    recorded = use.handle.store_range()
    if recorded is None:
        raise ValueError(f"{where}: {use.path} has no recorded range; build it first")
    return pd.Timestamp(recorded[1])


def ranged_uses(item) -> list:
    """The stores a component declares itself written by date range, in extension order."""
    return [use for use in item.store_uses() if isinstance(use, StoreUse) and use.ranged]


def extend_problems(item) -> list[str]:
    """Why ``extend`` cannot take a component: it declares no ranged store, or does not own one."""
    uses = ranged_uses(item)
    if not uses:
        return [f"extend takes a component writing stores by date range (a factor or a factor "
                f"risk model), got {type(item).__name__}"]
    return [f"the component is not the owner of {use.path}; write the owner's config"
            for use in uses if use.relation != OWNS]


def write_status(path: Path, status: dict) -> None:
    """Write the status file atomically, stamped with ``updated``."""
    status = {**status, "updated": now()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_jsonable(status), indent=2))
    os.replace(tmp, path)


class Update:
    """One run of the stages over a data root."""

    def __init__(self, plan: dict, data_dir: Path, download_dir: Path, dry_run: bool):
        """Hold the plan, the two roots and the steps run so far."""
        self.plan = plan
        self.data_dir = data_dir
        self.download_dir = download_dir
        self.dry_run = dry_run
        self.steps: list[dict] = []
        self.t: pd.Timestamp | None = None

    def folder(self, relative: str) -> Path:
        """A store folder, or folder prefix, under the data root."""
        return self.data_dir / relative

    def area(self, relative: str) -> Path:
        """What a folder name covers: its store, or the whole prefix when it ends in /."""
        root = self.folder(relative).absolute()
        return root if relative.endswith("/") else store_path(root)

    def calendar_bar(self) -> pd.Timestamp | None:
        """The last bar of the plan's ``calendar`` store."""
        return last_bar(store_path(self.folder(self.plan["calendar"])))

    # -- actions ------------------------------------------------------------------

    def sharadar(self, args: list | None = None, writes: list | None = None) -> str:
        """Run ``scripts/sharadar/update.py`` over both roots."""
        self._script("scripts/sharadar/update.py", "--download-dir", str(self.download_dir),
                     "--data-dir", str(self.data_dir), *(args or []))
        return "updated"

    def fred(self, series: list, store: str, start: str) -> str:
        """Download the FRED ``series`` from ``start``, then update the store."""
        if not self.dry_run:
            FredAcquisition(AcquisitionConfig(
                market="us_equity", frequency="1d", vendor="fred",
                raw_data_dir_path=str(self.download_dir / "fred"),
                watermark_path=str(self.download_dir / "_watermarks" / "fred"),
                symbols=tuple(series), start_date=start,
            )).download()
        return self.update(store)

    def benchmarks(self, tickers: list, writes: list | None = None) -> str:
        """Rebuild the price-return benchmarks of ``tickers`` from the SFP store."""
        sfp = store_path(self.folder("market/sharadar/sharadar_sfp_1d"))
        self._script("scripts/sharadar/price_return_benchmark.py", "--tickers", ",".join(tickers),
                     "--sfp-store", str(sfp), "--raw-dir", str(self.download_dir / "sharadar"),
                     "--data-dir", str(self.data_dir), "--refresh")
        return "rebuilt"

    def run(self, script: str, args: list | None = None, writes: list | None = None,
            reads: list | None = None) -> str:
        """Run a repository script, with ``{data_dir}`` and ``{download_dir}`` filled in."""
        def fill(arg) -> str:
            """One argument with the two placeholders replaced; other braces kept."""
            return (str(arg).replace("{data_dir}", str(self.data_dir))
                    .replace("{download_dir}", str(self.download_dir)))

        self._script(script, *(fill(a) for a in (args or [])))
        return "ran"

    def update(self, store: str) -> str:
        """``update()`` the dataset of a store folder."""
        if self.dry_run:
            return "would update"
        dataset = load_component(self.folder(store))
        dataset.update()
        result = f"last bar {last_bar(store_path(self.folder(store)))}"
        # A dataset that checks itself (the Massive Day aggregates against SEP) says what it found.
        summary = getattr(dataset, "update_summary", None)
        return f"{result}; {summary()}" if callable(summary) else result

    def mirror(self, store: str, source: str) -> str:
        """Append the source dataset's new bars to ``store``."""
        if self.dry_run:
            return "would mirror"
        return mirror_new_bars(load_component(self.folder(source)), store_path(self.folder(store)),
                               self.t.date().isoformat())

    def extend(self, store: str) -> str:
        """Extend every store a component writes by date range (a factor's, a risk model's two) to t."""
        if self.dry_run:
            return "would extend"
        # A date, so the recorded range holds the whole day (a Timestamp ends it at midnight).
        day = self.t.date().isoformat()
        item = load_component(self.folder(store))
        if problems := extend_problems(item):
            raise ValueError(f"{store}: {'; '.join(problems)}")
        done = []
        for use in ranged_uses(item):  # in extension order, every one owned
            name = Path(use.path).name
            if range_end(use, store) < self.t:
                use.handle.extend(day)
                done.append(f"{name} extended to {day}")
            else:
                done.append(f"{name} current")
        return "; ".join(done)

    # -- running ------------------------------------------------------------------

    def _script(self, script: str, *args: str) -> None:
        """Run a repository script with this interpreter; only log it on a dry run."""
        command = [sys.executable, str(REPO_ROOT / script), *args]
        logger.info(f"run {' '.join(command)}")
        if not self.dry_run:
            subprocess.run(command, check=True)

    def run_stage(self, stage: dict) -> list[str]:
        """Run a stage's steps in order; the failures of its ``allow_failure`` steps."""
        failures = []
        for step in stage["steps"]:
            action, kwargs, allow_failure = parse_step(step)
            if action not in ACTIONS:
                raise ValueError(f"{stage['name']}: unknown action {action!r}; one of {ACTIONS}")
            began = time.perf_counter()
            try:
                result = getattr(self, action)(**kwargs)
            except Exception as error:
                if not allow_failure:
                    raise
                # Readiness is decided by the `ready` stores, not by this step.
                logger.opt(exception=error).warning(f"{stage['name']}: {action} failed; continuing")
                result = f"failed: {error!r}"
                failures.append(f"{action} {result}")
            record = {"stage": stage["name"], "action": action, **kwargs, "result": result,
                      "seconds": round(time.perf_counter() - began, 1)}
            self.steps.append(record)
            logger.info(json.dumps(to_jsonable(record)))
        return failures

    def writes(self, action: str, kwargs: dict) -> list[str]:
        """The store folders (or folder prefixes, ending in /) a step writes."""
        if "writes" in kwargs:
            return list(kwargs["writes"])
        return [kwargs["store"]] if action in STORE_ACTIONS else []

    def check_component(self, where: str, action: str, key: str, relative: str) -> tuple[list[str], set[Path]]:
        """Rebuild the component a step names under ``key``; its problems and the stores it reads."""
        folder = self.folder(relative)
        if not (folder / COMPONENT_FILE).is_file():
            return [f"{where}: {folder / COMPONENT_FILE} is missing"], set()
        try:
            item = load_component(folder)
        except Exception as error:
            return [f"{where}: {relative} does not rebuild: {error!r}"], set()
        problems = []
        if action == "extend":
            problems += [f"{where}: {problem}" for problem in extend_problems(item)]
        if action in ("update", "fred") and not isinstance(item, BaseDataset):
            problems.append(f"{where}: {action} takes a dataset")
        if action == "mirror" and not isinstance(item, BaseDataset):
            problems.append(f"{where}: the mirror source must be a dataset")
        own = store_path(folder).absolute()
        return problems, {p for p in stores_read(item) | {own} if p != own}

    def check(self) -> list[str]:
        """Rebuild every step's component and check ownership and order; the problems found."""
        problems, steps = [], []
        if "calendar" not in self.plan:
            problems.append("the plan names no calendar store")
        for stage in self.plan["stages"]:
            for step in stage["steps"]:
                action, kwargs, _ = parse_step(step)
                where = f"{stage['name']}: {action} {kwargs.get('store', kwargs.get('script', ''))}".strip()
                if action not in ACTIONS:
                    problems.append(f"{where}: unknown action; one of {ACTIONS}")
                    continue
                reads: set[Path] = set()
                keys = {"update": ("store",), "extend": ("store",), "fred": ("store",), "mirror": ("source",)}
                for key in keys.get(action, ()):
                    if key not in kwargs:
                        problems.append(f"{where}: no {key}")
                        continue
                    found, read = self.check_component(where, action, key, kwargs[key])
                    problems += found
                    reads |= read
                if action == "mirror" and "source" in kwargs:
                    reads.add(store_path(self.folder(kwargs["source"])).absolute())
                if action == "run":
                    if not (REPO_ROOT / kwargs["script"]).is_file():
                        problems.append(f"{where}: no script {kwargs['script']}")
                    reads |= {self.area(r) for r in kwargs.get("reads", [])}
                steps.append((stage["name"], where, reads, self.writes(action, kwargs)))
        for i, (_, where, reads, _) in enumerate(steps):
            for _, later, _, written in steps[i + 1:]:
                for target in written:
                    hit = [p for p in reads if overlaps(p, self.area(target))]
                    if hit:
                        problems.append(f"{where} reads {hit[0]}, which a later step writes ({later})")
        first = self.plan["stages"][0]["name"] if self.plan["stages"] else None
        first_writes = [self.area(w) for name, _, _, written in steps if name == first for w in written]
        for store in [*([self.plan["calendar"]] if "calendar" in self.plan else []), *self.plan.get("ready", [])]:
            wanted = self.area(store)
            if not any(w == wanted or w in wanted.parents for w in first_writes):
                problems.append(f"{store}: the calendar and ready stores must be written by the first stage ({first})")
        return problems

    def ready(self) -> list[str]:
        """The ``ready`` stores that do not hold t."""
        return [s for s in self.plan.get("ready", [])
                if (bar := last_bar(store_path(self.folder(s)))) is None or bar < self.t]


def main() -> None:
    """Parse the arguments and run, or check, the plan; exit with the final state's code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_output_dir_args(parser)
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "data_update.yaml"))
    parser.add_argument("--stage", action="append", help="Run only these stages (no retry).")
    parser.add_argument("--retry-minutes", type=int, default=15)
    parser.add_argument("--retry-until", default="08:30", help="New York time, HH:MM.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check the plan, run nothing.")
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    plan = yaml.safe_load(Path(args.config).read_text())
    if args.check:
        problems = Update(plan, data_dir, download_dir, dry_run=True).check()
        for problem in problems:
            print(f"PROBLEM {problem}")
        print(f"{len(problems)} problem(s) in {args.config}")
        sys.exit(1 if problems else 0)
    stages = [s for s in plan["stages"] if not args.stage or s["name"] in args.stage]
    status_path = data_dir / STATUS_FILE
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    # The t of the last run that reached "done": a bar is new only after it.
    last_done_t = (previous.get("t") if previous.get("state") == "done" and previous.get("t")
                   else previous.get("last_done_t"))
    status = {"date": datetime.datetime.now(NEW_YORK).date().isoformat(), "state": "running",
              "started": now(), "last_done_t": last_done_t}
    run = Update(plan, data_dir, download_dir, args.dry_run)
    writes_status = not args.dry_run and not args.stage

    def report(state: str, **extra) -> None:
        """Write the run's final status (``done``, ``no_new_bar`` or ``failed``) when this run writes one."""
        if not writes_status:
            return
        t = str(run.t.date()) if run.t is not None else None
        write_status(status_path, {**status, "t": t, "state": state,
                                   "last_done_t": t if state == "done" else last_done_t,
                                   **extra, "steps": run.steps})

    if writes_status:
        write_status(status_path, {**previous, **status})
    try:
        raw, rest = (stages[0], stages[1:]) if stages and stages[0]["name"] == "raw" else (None, stages)
        while raw is not None:
            failures = run.run_stage(raw)
            run.t = run.calendar_bar()
            waiting = run.ready() if run.t is not None else []
            new = run.t is not None and (last_done_t is None or run.t > pd.Timestamp(last_done_t))
            if args.stage or args.dry_run or (not waiting and new):
                break
            if waiting:
                why = f"stores short of {run.t.date()}: {waiting}"
            else:
                why = f"no bar after {last_done_t}" if run.t is not None else "the calendar store is empty"
            if failures:
                why += "; " + "; ".join(failures)
            if datetime.datetime.now(NEW_YORK).strftime("%H:%M") >= args.retry_until:
                logger.warning(f"{why}; past {args.retry_until}, giving up")
                state = "failed" if waiting or run.t is None else "no_new_bar"
                report(state, reason=why)
                sys.exit(EXIT_CODES[state])
            logger.info(f"{why}; retrying in {args.retry_minutes} min")
            time.sleep(args.retry_minutes * 60)
        if run.t is None:
            run.t = run.calendar_bar()
        for stage in rest:
            if not stage.get("after_done"):
                run.run_stage(stage)
    except Exception as error:
        logger.exception("data update failed")
        report("failed", reason=repr(error))
        sys.exit(EXIT_CODES["failed"])
    report("done", finished=now())
    logger.info(f"data update done to {run.t.date() if run.t is not None else None}")
    # Stages nothing waiting for `done` reads: they run once it is written, and a
    # failure there is logged and recorded without changing the state.
    later = [stage for stage in rest if stage.get("after_done")]
    for stage in later:
        try:
            run.run_stage(stage)
        except Exception as error:
            logger.opt(exception=error).warning(f"{stage['name']}: failed after done")
            run.steps.append({"stage": stage["name"], "result": f"failed: {error!r}"})
    if later:
        report("done", finished=now())


if __name__ == "__main__":
    main()
