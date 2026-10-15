"""Structural contract of the backtest layer (phase 03.7, plan 14).

This file is the backtest layer's equivalent of `tests/test_model_hierarchy.py`:
it keeps the architecture plan 03.7-01 built from eroding silently. The layer is

- `BaseBacktester` (`quantlab/backtest/base.py`) -- the ONLY home of `run()`;
- `VectorBtBacktester` (`quantlab/backtest/engine_vectorbt.py`) -- the engine
  layer, still abstract;
- `USEquityCrossectionSelectStockVectorBt` (`quantlab/backtest/predefined/us_equity.py`)
  -- a named composition of a market spec, a selector and an engine.

What is locked here, the decision each lock enforces, and what turns it red:

- **Exact abstract-method sets (D-01).** Each layer's `__abstractmethods__` is
  compared with a literal set. A hook that silently gains a default, a new
  abstract hook, or an engine that stops implementing one of its three hooks
  changes a set and fails the test. Since plan 03.7-08 the engine implements
  four hooks: `_simulate`, `_simulate_benchmark`, `_engine_stats` and
  `_period_returns_stats` (slice returns statistics, D-34).
- **`run()` and `run_cv()` are the two templates (D-02).** No class above
  `BaseBacktester` in the concrete class's MRO may define `run`, `run_cv` or
  any public function, classmethod, staticmethod or property of its own. An
  override that "only calls super()" is still a second implementation of the
  public entry, so it is red. The public callables defined on `BaseBacktester`
  itself are also pinned: the entry points `run`, `run_cv` and `run_weights`,
  plus `get_config` and `report_figure` (the report chart of a result in
  memory, which `quantlab.api` draws), so any other public method on the base
  is red too.
- **Type check first (D-01).** A concrete backtester given a plain
  `BacktestConfig` raises `TypeError` naming `CrossSectionBacktestConfig`
  before any dataset or model attribute is read. The config carries bare
  `object()`s, so if another check runs first it raises a different error and
  the test fails.
- **Price column names live only in the market spec (D-04).** No string
  constant inside any function body of `quantlab/backtest/base.py` or
  `quantlab/backtest/*.py` equals the spec's fill or valuation column name. A
  method that hardcodes the column is red. The names are read from
  `US_EQUITY_MARKET`, so the lock follows the spec.
- **No config factories (D-32).** No import in the backtest layer,
  `tests/backtest_fixtures.py` or `tests/test_backtest_*.py` resolves to
  `quantlab.config`. Relative spellings are resolved against the file's
  package, because a substring scan cannot see them.
- **One-directional layering.** No quantlab module outside
  `quantlab/backtest/`, `quantlab/backtest/base.py` and `quantlab/api/` (the
  frame facade above every layer, ADR 0011) imports the backtest layer.
- **vectorbt stays inside the engine (D-31).** No quantlab module other than
  `quantlab/backtest/engine_vectorbt.py` imports vectorbt, with no exemption.
  (Plan 03.7-14 shipped a temporary exemption for the legacy helper package;
  plan 03.7-12 retired that package and removed the exemption.)
- **The legacy helper package stays retired (D-31, CLEAN-02).** `quantlab.vecbt`
  cannot be found by the import system and its directory does not exist, so
  neither a restored module nor a stray `__pycache__`-only directory passes.

Everything is static or construction-only: offline and CPU-only, with no store
and no model.
"""

import ast
import functools
import types
from pathlib import Path

import pytest

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.predefined.us_equity import (
    US_EQUITY_MARKET,
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.backtest.base import BaseBacktester
from quantlab.backtest.config import BacktestConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Kinds of class-dict entry that make a name part of a class's callable
#: surface. A plain class attribute (`config_cls = SomeConfig`, `MARKET = ...`)
#: is configuration, not a method, and is deliberately not listed.
_METHOD_KINDS = (
    types.FunctionType,
    classmethod,
    staticmethod,
    property,
    functools.cached_property,
    functools.partialmethod,
)

#: D-02's user-facing entry points, all defined once on `BaseBacktester`.
_ENTRY_POINTS = ("run", "run_cv", "run_weights")

#: Class-dict entries that are callable methods. Unlike `_METHOD_KINDS` this
#: excludes properties, which are attribute surface rather than entry points.
_CALLABLE_KINDS = (
    types.FunctionType,
    classmethod,
    staticmethod,
    functools.partialmethod,
)


# --------------------------------------------------------------------------
# Class surface (D-01, D-02)
# --------------------------------------------------------------------------


def test_abstract_method_sets_are_exact():
    """The engine-by-inheritance hierarchy, one exact set per layer (D-01)."""
    assert BaseBacktester.__abstractmethods__ == frozenset(
        {
            "config_cls",
            "_generate_signals",
            "_simulate",
            "_simulate_benchmark",
            "_engine_stats",
            # Slice return statistics were an engine hook until #109 made
            # them the engine-free `quantlab.runs.backtest_stats.return_stats`.
        }
    )
    assert VectorBtBacktester.__abstractmethods__ == frozenset(
        {"config_cls", "_generate_signals"}
    )
    assert USEquityCrossectionSelectStockVectorBt.__abstractmethods__ == frozenset()


def test_run_lives_only_on_base_backtester():
    """`run()`, `run_cv()` and `run_weights()` are the templates (D-02): nothing
    above the base redefines one or grows a public method of its own, and the
    public callables the base itself defines are exactly the three entry points,
    `from_config`, `report_figure` and `account` (a simulation's account
    record, which `quantlab.runs.run_metrics` reads; ADR 0034) (`get_config` is
    the component rule's, inherited)."""
    for entry in _ENTRY_POINTS:
        assert entry in vars(BaseBacktester), entry

    mro = USEquityCrossectionSelectStockVectorBt.__mro__
    above = mro[: mro.index(BaseBacktester)]
    assert above, "the concrete backtester must inherit from BaseBacktester"

    offenders = {}
    for cls in above:
        names = sorted(
            name
            for name, value in vars(cls).items()
            if name in _ENTRY_POINTS
            or (not name.startswith("_") and isinstance(value, _METHOD_KINDS))
        )
        if names:
            offenders[cls.__qualname__] = names
    assert offenders == {}, offenders

    # 03.7-10: properties (`config`, `config_cls`, `class_name`, `import_path`)
    # are attribute surface, not callables, and are not counted here.
    public_callables = {
        name
        for name, value in vars(BaseBacktester).items()
        if not name.startswith("_") and isinstance(value, _CALLABLE_KINDS)
    }
    assert public_callables == {
        "run",
        "run_cv",
        "run_weights",
        "from_config",
        "report_figure",
        "account",
    }, public_callables


def test_backtest_configs_are_constructed_with_their_own_classes():
    """The config type check is the setter's first statement (D-01).

    Both object fields are bare `object()`s. Reading any dataset or model
    attribute would raise `AttributeError`, and the model type check would raise
    a `TypeError` that names `Predictor`. Only the class check that runs first
    names `CrossSectionBacktestConfig`.
    """
    config = BacktestConfig(
        price_dataset=object(),
        model=object(),
        model_mode="train",
        start_date="2024-01-01",
        end_date="2024-02-01",
        output_dir="unused",
        rebalance_periods=1,
    )
    with pytest.raises(
        TypeError,
        match=(
            r"^USEquityCrossectionSelectStockVectorBt requires a "
            r"CrossSectionBacktestConfig, got BacktestConfig$"
        ),
    ):
        USEquityCrossectionSelectStockVectorBt(config)


# --------------------------------------------------------------------------
# Source rules (D-04, D-32, layering, vectorbt scope)
# --------------------------------------------------------------------------


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _backtest_layer_files() -> list[Path]:
    return [REPO_ROOT / "quantlab/backtest/base.py"] + _python_files(
        REPO_ROOT / "quantlab/backtest"
    )


def _is_or_under(name: str, module: str) -> bool:
    return name == module or name.startswith(module + ".")


def _resolved_imports(path: Path) -> set[str]:
    """Every module `path` imports, as a fully qualified dotted name.

    - `import a.b` contributes `a.b`;
    - `from a.b import c` contributes `a.b` and `a.b.c`, because `c` may itself
      be a submodule (`from quantlab import config`);
    - a relative import is resolved against the file's own dotted module name,
      derived from its path under the repo root, by stripping `level` trailing
      components and appending `module`. For `quantlab/dataset/base.py`,
      `from ..backtest import x` becomes `quantlab.backtest` and
      `quantlab.backtest.x`. An `__init__.py` keeps `__init__` as its last
      component, so `from . import x` there resolves to its own package.

    Nested imports (inside a function) are included: `ast.walk` sees them.
    """
    module_parts = path.resolve().relative_to(REPO_ROOT).with_suffix("").parts
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(module_parts[: max(len(module_parts) - node.level, 0)])
                target = ".".join(part for part in (base, node.module) if part)
            else:
                target = node.module or ""
            if target:
                found.add(target)
            found.update(
                ".".join(part for part in (target, alias.name) if part)
                for alias in node.names
            )
    return found


def test_price_column_literals_never_appear_inside_a_method_body():
    """D-04: methods read column names from `self.MARKET`, never spell them.

    Every function body (including lambdas and nested functions) in the
    backtest layer is walked for a string constant equal to one of the spec's
    column names. The module-level `US_EQUITY_MARKET` constant is outside every
    function body, so it stays allowed.
    """
    columns = {
        US_EQUITY_MARKET.fill_price_column,
        US_EQUITY_MARKET.valuation_price_column,
    }
    assert len(columns) == 2 and all(isinstance(c, str) and c for c in columns)

    files = _backtest_layer_files()
    assert len(files) >= 4, files

    hits = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Constant) and node.value in columns:
                    hits.add(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} "
                        f"{node.value!r}"
                    )
    assert hits == set(), sorted(hits)


def test_backtest_layer_never_imports_config_factories():
    """D-32: the backtest layer and its tests never reach `quantlab.config`."""
    files = (
        _backtest_layer_files()
        + [REPO_ROOT / "tests/backtest_fixtures.py"]
        + sorted((REPO_ROOT / "tests").glob("test_backtest_*.py"))
    )
    assert REPO_ROOT / "tests/test_backtest_contracts.py" in files

    # Positive control on a real relative spelling: `from .config import
    # BacktestConfig` in quantlab/backtest/base.py must resolve to the sibling
    # module, or every relative import below would be checked as garbage.
    assert {"quantlab.backtest.config", "quantlab.backtest.config.BacktestConfig"} <= (
        _resolved_imports(REPO_ROOT / "quantlab/backtest/base.py")
    )

    offenders = {
        str(path.relative_to(REPO_ROOT)): sorted(
            name
            for name in _resolved_imports(path)
            if _is_or_under(name, "quantlab.config")
        )
        for path in files
    }
    offenders = {path: names for path, names in offenders.items() if names}
    assert offenders == {}, offenders


def test_no_lower_layer_imports_the_backtest_layer():
    """Layering: the backtest layer is imported only by itself and by
    `quantlab/api/`, the facade that sits above every layer."""
    allowed_files = {REPO_ROOT / "quantlab/backtest/base.py"}
    backtest_pkg = REPO_ROOT / "quantlab/backtest"
    api_pkg = REPO_ROOT / "quantlab/api"

    # Positive control: the concrete class really does import the layer, so
    # the resolver is not blind to it.
    assert "quantlab.backtest.engine_vectorbt" in _resolved_imports(
        backtest_pkg / "predefined/us_equity.py"
    )

    offenders = {}
    for path in _python_files(REPO_ROOT / "quantlab"):
        if path in allowed_files or backtest_pkg in path.parents or api_pkg in path.parents:
            continue
        names = sorted(
            name
            for name in _resolved_imports(path)
            if _is_or_under(name, "quantlab.backtest")
            or _is_or_under(name, "quantlab.backtest.base")
        )
        if names:
            offenders[str(path.relative_to(REPO_ROOT))] = names
    assert offenders == {}, offenders


def test_vectorbt_is_imported_only_by_the_engine():
    """vectorbt stays behind the engine layer (D-31)."""
    engine = REPO_ROOT / "quantlab/backtest/engine_vectorbt.py"
    # Positive control: the engine imports vectorbt, so a scan that finds no
    # importer anywhere cannot pass by being blind.
    assert "vectorbt" in _resolved_imports(engine)

    offenders = {}
    for path in _python_files(REPO_ROOT / "quantlab"):
        if path == engine:
            continue
        names = sorted(
            name for name in _resolved_imports(path) if _is_or_under(name, "vectorbt")
        )
        if names:
            offenders[str(path.relative_to(REPO_ROOT))] = names
    assert offenders == {}, offenders


def test_vecbt_package_is_retired():
    """The legacy vectorbt signal helper stays deleted (D-31, CLEAN-02).

    `quantlab/vecbt/bt.py:backtest_from_signals` called `from_signals` with no
    size semantics and never reached a working state; `quantlab/backtest/`
    replaced it. The editable install maps only the top `quantlab` package, so
    a restored `__init__.py` makes `find_spec` succeed, and so does a leftover
    directory holding only `__pycache__`: Python treats it as a namespace
    package (measured in 03.7-12). The existence arm additionally catches a
    non-package filesystem entry at that path, which `find_spec` cannot see.
    """
    import importlib.util

    # Positive control: the lookup does see real subpackages, so a `None`
    # below is evidence of absence, not a blind resolver.
    assert importlib.util.find_spec("quantlab.backtest") is not None

    assert importlib.util.find_spec("quantlab.vecbt") is None
    assert not (REPO_ROOT / "quantlab/vecbt").exists()
