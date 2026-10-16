"""The layer order (ADR 0022) and the layout rules inside a layer.

What is locked here, and what turns it red:

- every ``quantlab`` import, a function-level one included, follows the layer order
  ``LAYER_ORDER``: a module imports only modules of its own layer or of a layer before it.
  Each module's layer is ``MODULE_LAYERS`` (longest dotted prefix wins), and every module
  has one. There is no exception list;
- importing any layer's root class or configs loads no torch (the training target of both model
  variants lives in `quantlab/model/training_target.py`, not on `BaseModel`);
- no layer's framework module (a top-level file of `quantlab/factor`, `quantlab/label`,
  `quantlab/model`, `quantlab/risk`, `quantlab/backtest`, `quantlab/portfolio`) imports that layer's
  `predefined` package, so a user's own factor, label, model, backtester or portfolio
  construction rule needs nothing from the shipped ones;
- cvxpy is imported by the mean-variance optimiser only;
- the tracking root module imports no tracking library, wandb is imported by the W&B
  tracker only and mlflow by the MLflow tracker only (ADR 0015);
- every `predefined` package's `__init__.py` is empty, and so is `quantlab/runs/__init__.py`.

Static checks, plus one subprocess import; offline.
"""

import subprocess
import sys

import pytest

from tests.test_backtest_contracts import (
    REPO_ROOT,
    _is_or_under,
    _python_files,
    _resolved_imports,
)

LAYERS = ("factor", "label", "model", "risk", "backtest", "portfolio")


def test_importing_the_root_classes_and_configs_loads_no_torch():
    # Every layer's root class and configs (``<layer>/base.py``, ``<layer>/config.py``).
    paths = [
        *(REPO_ROOT / "quantlab").glob("*/base.py"),
        *(REPO_ROOT / "quantlab").glob("*/config.py"),
    ]
    modules = sorted(
        _module_name(path) for path in paths if path.stem != "__init__"
    )
    # Positive control: the model layer's root class is among them.
    assert "quantlab.model.base" in modules
    code = (
        "import sys\n"
        + "".join(f"import {name}\n" for name in modules)
        + "print('torch' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.parametrize("layer", LAYERS)
def test_framework_modules_never_import_their_predefined_package(layer):
    root = REPO_ROOT / "quantlab" / layer
    predefined = f"quantlab.{layer}.predefined"
    # Positive control: the predefined package exists and its modules are seen.
    assert (root / "predefined/__init__.py").is_file()
    offenders = {
        path.name: sorted(
            name for name in _resolved_imports(path) if _is_or_under(name, predefined)
        )
        for path in sorted(root.glob("*.py"))
    }
    assert {name: found for name, found in offenders.items() if found} == {}


def test_the_approximate_interface_imports_no_predefined_module_and_no_solver():
    # The framework module chooses the implementations by the rules' import paths, so
    # neither a rule module nor an implementation (nor cvxpy) loads before use.
    path = REPO_ROOT / "quantlab/portfolio/approximate.py"
    # Positive control: the module and its imports are seen.
    assert path.is_file() and _resolved_imports(path)
    offenders = sorted(
        name
        for name in _resolved_imports(path)
        if _is_or_under(name, "quantlab.portfolio.predefined") or _is_or_under(name, "cvxpy")
    )
    assert offenders == []


def test_the_approximate_rule_implementations_import_no_solver():
    root = REPO_ROOT / "quantlab/portfolio/predefined/approximate"
    files = sorted(root.rglob("*.py"))
    # Positive control: the implementations and the kernels are seen.
    assert {path.name for path in files} >= {"top_n.py", "mean_variance.py", "covariance.py", "mean_variance_kernels.py"}
    offenders = {
        str(path.relative_to(root)): sorted(
            name for name in _resolved_imports(path) if _is_or_under(name, "cvxpy")
        )
        for path in files
    }
    assert {name: found for name, found in offenders.items() if found} == {}


def test_importing_the_portfolio_root_and_rules_loads_no_numba():
    # numba is the approximate backtest's (its kernels in quantlab.portfolio.predefined.approximate,
    # quantlab.execution.approximate),
    # imported where it is used; a per-bar rule, run by a backtest run and the live trader, never pays it.
    code = (
        "import sys\n"
        "import quantlab.portfolio.base, quantlab.portfolio.decision_inputs\n"
        "import quantlab.portfolio.predefined.top_n, quantlab.portfolio.predefined.mean_variance\n"
        "import quantlab.portfolio.predefined.active_risk, quantlab.portfolio.predefined.ledoit_wolf\n"
        "import quantlab.portfolio.predefined.factor_risk\n"
        "import quantlab.portfolio.predefined.approximate.top_n\n"
        "import quantlab.portfolio.predefined.approximate.mean_variance\n"
        "import quantlab.portfolio.predefined.approximate.covariance\n"
        "import quantlab.portfolio.approximate, quantlab.backtest.approximate\n"
        "print('numba' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.parametrize("layer", LAYERS)
def test_predefined_init_files_are_empty(layer):
    for init in (REPO_ROOT / "quantlab" / layer / "predefined").rglob("__init__.py"):
        assert init.stat().st_size == 0, init


def test_only_the_mean_variance_optimizer_imports_a_solver():
    importers = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in _python_files(REPO_ROOT / "quantlab")
        if any(_is_or_under(name, "cvxpy") for name in _resolved_imports(path))
    )
    assert importers == ["quantlab/portfolio/predefined/mean_variance.py"]


def test_the_tracking_root_module_imports_no_tracking_library():
    code = (
        "import sys\n"
        "import quantlab.tracking.base\n"
        "print(sorted(name for name in ('wandb', 'mlflow') if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


@pytest.mark.parametrize("library", ["wandb", "mlflow"])
def test_only_its_tracker_imports_a_tracking_library(library):
    importers = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in _python_files(REPO_ROOT / "quantlab")
        if any(_is_or_under(name, library) for name in _resolved_imports(path))
    )
    assert importers == [f"quantlab/tracking/{library}.py"]


def test_the_run_layer_init_is_empty():
    assert (REPO_ROOT / "quantlab/runs/__init__.py").stat().st_size == 0


#: The layers, bottom first (ADR 0022). A module may import its own layer or one before it.
LAYER_ORDER = (
    "utils",
    "core",
    "backend",
    "tracking",
    "execution",
    "runs",
    "universe",
    "dataset",
    "config",
    "acquisition",
    "analysis",
    "factor",
    "label",
    "model",
    "risk",
    "portfolio",
    "backtest",
    "api",
)

#: Dotted prefix -> layer; the longest matching prefix wins.
MODULE_LAYERS = {
    "quantlab.enums": "utils",
    "quantlab.utils": "utils",
    "quantlab.core": "core",
    "quantlab.backend": "backend",
    "quantlab.tracking": "tracking",
    "quantlab.execution": "execution",
    "quantlab.runs": "runs",
    "quantlab.universe": "universe",
    "quantlab.dataset": "dataset",
    "quantlab.config": "config",
    "quantlab.acquisition": "acquisition",
    "quantlab.analysis": "analysis",
    "quantlab.factor": "factor",
    "quantlab.label": "label",
    "quantlab.model": "model",
    "quantlab.risk": "risk",
    "quantlab.portfolio": "portfolio",
    "quantlab.backtest": "backtest",
    "quantlab.api": "api",
}


def _layer_of(name: str) -> str | None:
    matches = [prefix for prefix in MODULE_LAYERS if _is_or_under(name, prefix)]
    return MODULE_LAYERS[max(matches, key=len)] if matches else None


def _module_name(path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts).removesuffix(".__init__")


def _as_module(name: str) -> str:
    """``name`` cut back to the module it names (``from a.b import C`` gives ``a.b.C``)."""
    parts = name.split(".")
    while len(parts) > 1:
        base = REPO_ROOT.joinpath(*parts)
        if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
            break
        parts.pop()
    return ".".join(parts)


def _violations(relpath: str, module: str, imports) -> set[tuple[str, str]]:
    """The imports of one file (``relpath``, dotted ``module``) that go against the order."""
    rank = {layer: index for index, layer in enumerate(LAYER_ORDER)}
    importer = _layer_of(module)
    if importer is None:
        return set()
    return {
        (relpath, _as_module(name))
        for name in imports
        if (imported := _layer_of(name)) is not None and rank[imported] > rank[importer]
    }


def _order_violations() -> set[tuple[str, str]]:
    found = set()
    for path in _python_files(REPO_ROOT / "quantlab"):
        relpath = str(path.relative_to(REPO_ROOT))
        found |= _violations(relpath, _module_name(path), _resolved_imports(path))
    return found


def test_every_module_has_a_layer():
    unmapped = sorted(
        _module_name(path)
        for path in _python_files(REPO_ROOT / "quantlab")
        if _layer_of(_module_name(path)) is None
    )
    # Only the top-level package itself (an empty ``__init__``) has no layer.
    assert unmapped == ["quantlab"]


def test_every_mapped_layer_is_in_the_order():
    assert set(MODULE_LAYERS.values()) <= set(LAYER_ORDER)


def test_imports_follow_the_layer_order():
    assert sorted(_order_violations()) == []


def test_an_upward_import_is_caught():
    # Positive control: an import of a later layer is a violation, of the same or an
    # earlier layer is not (``_resolved_imports`` already includes function-level imports).
    assert _violations(
        "quantlab/utils/timer.py",
        "quantlab.utils.timer",
        {"quantlab.backtest.engine_vectorbt", "quantlab.utils.atomic"},
    ) == {("quantlab/utils/timer.py", "quantlab.backtest.engine_vectorbt")}
