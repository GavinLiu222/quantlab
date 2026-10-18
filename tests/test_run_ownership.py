"""ADR 0018 and ADR 0020: a run directory's files are written and read only through
the run layer, `quantlab/runs/`.

The run modules own every run directory: `directory.py` the `run.json` every run
type shares, `trained_run.py` a training run's files and `backtest_run.py` a
backtest run's. Every other reader goes through `TrainedRun` or `BacktestRun`, so a
renamed file or record key breaks no reader silently.

What is locked here, and what turns it red, in the library and the examples
(docstring sessions included) and in the code of the docs pages and example READMEs
(their `>>>` / `...` lines, and every line of a ```python block without sessions):

- a string literal naming a run file, alone or as the last part of a path: a
  training run's record or evaluation files (`run.json`, `ic_series.csv`,
  `test_predictions.zarr`; `cv_folds.json` and `ensemble.json` are refused too), a
  backtest run's files (`config.json`, `metrics.json`, `weights.zarr`,
  `equity.zarr`, `settlements.json`, `predictions.zarr`, `report.html`;
  `fingerprint.json` is refused too) or a path under its `inputs/`. Prose may name
  them; code reads them through the run readers. The factor report writes its own
  output directory, not a run, and its lines may name its `config.json`
  (`ALLOWED`);
- indexing a run's config by key (`run.config[...]`, `TrainedRun.open(p).config.get(...)`):
  a run's records are read as the readers' typed values, and its recipe is rebuilt
  into objects (`from_config`, `rebuild_backtester`, `rebuild(field)`);
- walking up two directory levels (`.parent.parent`, `.parents[...]`), the way a
  run's layout would be guessed from a path. A line that starts from `__file__` (a
  package root) is exempt;
- taking `.parent` of a checkpoint: a unit's directory is `TrainedRun.open(...).path`.

Static, offline.
"""

import ast
import re

from tests.test_backtest_contracts import REPO_ROOT

OWNERS = tuple(sorted((REPO_ROOT / "quantlab/runs").glob("*.py")))

_RUN_FILES = (
    "run.json", "cv_folds.json", "ensemble.json", "ic_series.csv", "test_predictions.zarr",
    "config.json", "metrics.json", "weights.zarr", "equity.zarr", "holdings.zarr", "settlements.json",
    "predictions.zarr", "report.html", "fingerprint.json",
)
#: Lines outside the run layer that name a run file for a directory of their own:
#: a file mapped to the text such a line carries. The factor report writes
#: ``config.json`` into its own output directory, not a run.
ALLOWED = {
    "quantlab/analysis/factor_report.py": '(out / "config.json")',
}
_NAMES_A_RUN_FILE = "names a run file"
_WALK_UP = "walks up two directory levels"
_RULES = {
    "indexes a run's config by key": re.compile(
        r"(?:\b\w*run\w*|(?:Trained|Backtest)Run\.open\([^)]*\))\.config\s*(?:\[|\.get\()"
    ),
    _NAMES_A_RUN_FILE: re.compile(
        r"""["'](?:(?:[^"'\n]*/)?(?:%s)|inputs(?:/[^"'\n]*)?)["']"""
        % "|".join(map(re.escape, _RUN_FILES))
    ),
    _WALK_UP: re.compile(r"\.parent\.parent\b|\.parents\["),
    "takes the directory of a checkpoint": re.compile(r"checkpoint[\w\"'\]\)]*\.parent\b"),
}
_SESSION_LINE = re.compile(r"^\s*(?:>>>|\.\.\.)\s?(.*)$")


def _session_code(line: str) -> str:
    """Return the code of a ``>>>`` / ``...`` line, or "" for prose and expected output."""
    match = _SESSION_LINE.match(line)
    return match.group(1) if match else ""


def _docstring_lines(tree: ast.Module) -> set:
    """Return the line numbers the docstrings of ``tree`` span."""
    spans = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                spans.update(range(first.lineno, first.end_lineno + 1))
    return spans


def _python_sources():
    """Yield each library and example file with its code lines; a docstring keeps only its session code."""
    for root in ("quantlab", "examples"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if path in OWNERS or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            docstrings = _docstring_lines(ast.parse(text, filename=str(path)))
            yield path, [
                _session_code(line) if number in docstrings else line
                for number, line in enumerate(text.splitlines(), start=1)
            ]


_FENCE = re.compile(r"^\s*```")


def _page_code(text: str) -> list[str]:
    """Return a page's lines with only its code kept, prose and expected output blanked.

    A session line (``>>>`` / ``...``) is code wherever it is; a ```python block
    without sessions is code throughout, its lines being a script.
    """
    lines: list[str] = []
    block: list[str] | None = None
    for line in text.splitlines():
        if _FENCE.match(line):
            if block is None and line.strip().lstrip("`").strip() == "python":
                block = []
            elif block is not None:
                script = not any(_SESSION_LINE.match(item) for item in block)
                lines.extend(item if script else _session_code(item) for item in block)
                block = None
            lines.append("")
            continue
        if block is not None:
            block.append(line)
        else:
            lines.append(_session_code(line))
    if block is not None:
        lines.extend(_session_code(item) for item in block)
    return lines


def _doc_sessions():
    """Yield each docs page and example README with its code; prose and output are blanked."""
    pages = [
        path
        for path in sorted((REPO_ROOT / "docs").rglob("*.md"))
        if not {"adr", "research"} & set(path.relative_to(REPO_ROOT / "docs").parts)
    ] + sorted((REPO_ROOT / "examples").rglob("*.md"))
    for path in pages:
        yield path, _page_code(path.read_text(encoding="utf-8"))


def _offences(sources):
    """Return ``path:line: rule: code`` for every line a rule matches."""
    found = []
    for path, lines in sources:
        for number, line in enumerate(lines, start=1):
            for rule, pattern in _RULES.items():
                if rule == _WALK_UP and "__file__" in line:
                    continue
                allowed = ALLOWED.get(str(path.relative_to(REPO_ROOT)))
                if rule == _NAMES_A_RUN_FILE and allowed and allowed in line:
                    continue
                if pattern.search(line):
                    found.append(f"{path.relative_to(REPO_ROOT)}:{number}: {rule}: {line.strip()}")
    return found


def test_only_the_run_layer_names_or_lays_out_run_files_in_code():
    # Positive control: the run modules are the owners.
    assert {path.name for path in OWNERS} >= {"directory.py", "trained_run.py", "backtest_run.py"}
    assert _offences(_python_sources()) == []


def test_doc_sessions_read_runs_through_the_run_readers():
    assert _offences(_doc_sessions()) == []


def test_the_rules_catch_what_they_lock():
    caught = _offences(
        [
            (REPO_ROOT / "x.py", [
                'trial = Path(folds[0]["checkpoint"]).parent.parent',
                'manifest = json.loads((trial / "cv_folds.json").read_text())',
                'record = unit / "run.json"',
                'path = f"{unit}/ensemble.json"',
                'trial = checkpoint.parents[1]',
                'metrics = json.loads((checkpoint.parent / "metrics.json").read_text())',
                'config = json.loads((run_dir / "config.json").read_text())',
                'weights = XrBackend().read(result.run_dir / "weights.zarr").data',
                'prices = run_dir / "inputs/price_dataset.zarr"',
                'fingerprints = json.load(open(run_dir / "fingerprint.json"))',
                'eta = run.config["resolved_hyperparameters"]["eta"]',
                'tracker = TrainedRun.open(checkpoint).config.get("tracker")',
            ]),
        ]
    )
    assert {offence.split(":")[1] for offence in caught} == {str(n) for n in range(1, 13)}
    clean = _offences(
        [
            (REPO_ROOT / "x.py", [
                'raise ValueError(f"{path} (its run.json) differs")',
                'ROOT = Path(__file__).resolve().parent.parent',
                'run = TrainedRun.open(checkpoint)',
                'metrics = BacktestRun.open(run_dir).metrics()',
                'raise ValueError(f"{path} has no config.json")',
                'model = XGBoostRegressor.from_config(run.config)',
                'top_n = backtester.config.constructor.config.top_n',
            ]),
        ]
    )
    assert clean == []
    # Code in a docs page: a script block counts whole, a session block only its
    # session lines (its expected output is not code).
    page = "\n".join([
        "```python",
        'saved = run.config["tracker"]',
        "```",
        "```python",
        ">>> sorted(p.name for p in unit.iterdir())",
        "['config.json', 'run.json']",
        "```",
    ])
    assert [line for line in _page_code(page) if line] == [
        'saved = run.config["tracker"]',
        "sorted(p.name for p in unit.iterdir())",
    ]
    # The factor report's own output directory may name its config.json.
    assert _offences(
        [(REPO_ROOT / "quantlab/analysis/factor_report.py", ['(out / "config.json").write_text(text)'])]
    ) == []
