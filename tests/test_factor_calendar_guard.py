"""Only the factor layer reads a factor's dataset calendar (ADR 0051, #369).

A factor answers for its own calendar (``Factor.calendar``, ``bar_before``,
``bar_after``: the bars of its output panel), and every consumer that counts
bars of a factor counts on it. ``config.dataset`` keeps one meaning, where the
factor's input comes from; its calendar is read only inside the factor layer,
for the factor's own warm-up (``_warm_start``), ``IntradayStats``'s sessions and
the chunked builder's input-side counts.

Why: in 8c2d1748 a model counted its warm-up on ``factor.config.dataset``, so a
daily factor computed from one-minute bars (``IntradayStats`` resampled to
``"1d"`` on Massive's 1m trade bars) was requested from ``warmup_bars`` minutes
before start instead of ``warmup_bars`` trading days, and a reloaded us3000 LSTM
differed from its training run on its first 19 test bars. The ``Forward`` label
had the same count for its lookahead. Both were migrated (#366, #367); this test
keeps them from coming back.

What is locked here, and what turns it red:

- in every Python module of the repository (``quantlab/``, ``tests/``,
  ``examples/``, ``scripts/``; parsed with ``ast``, never imported) outside
  ``quantlab/factor/``, no call of ``bar_before``, ``bar_after`` or ``calendar``
  is reached through a ``.config.dataset`` chain. Two forms are caught: the direct
  chain ``<expr>.config.dataset.bar_before(...)`` and the split form the old bug
  sites used, a name assigned from ``<expr>.config.dataset`` in the same function
  scope and then ``name.bar_before(...)``;
- ``self.config.dataset`` is exempt: that is a component's own input (a factor
  subclass defined outside the layer counting its own warm-up, a risk model
  counting back on its own prices), not another component's dataset reached
  through its config. ``self.config.factor.config.dataset`` is not exempt.

A violation is reported as ``(module path relative to the repo root, line)``.
Static, offline.
"""

import ast
from pathlib import Path

from tests.test_backtest_contracts import REPO_ROOT, _python_files

#: The calendar queries a factor answers itself (ADR 0051).
CALENDAR_METHODS = frozenset({"bar_before", "bar_after", "calendar"})

#: The roots walked, and the one place the calls are allowed.
ROOTS = ("quantlab", "tests", "examples", "scripts")
FACTOR_LAYER = Path("quantlab/factor")


def _config_dataset_root(node: ast.AST) -> ast.AST | None:
    """The ``<expr>`` of a ``<expr>.config.dataset`` chain, or ``None``."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "dataset"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "config"
    ):
        return node.value.value
    return None


def _is_another_components_dataset(node: ast.AST) -> bool:
    """``node`` is ``<expr>.config.dataset`` with ``<expr>`` not the bare ``self``."""
    root = _config_dataset_root(node)
    return root is not None and not (isinstance(root, ast.Name) and root.id == "self")


def _scopes(tree: ast.Module):
    """Every scope of ``tree`` with its own statements: the module and each function.

    A function's body belongs to it, not to the enclosing scope, so a name
    assigned in one function is not the name called in another (``ast.walk``
    cannot stop at a nested function, hence the walk by hand). A function's
    decorators and defaults are evaluated in the enclosing scope and stay there.
    """
    pending = [tree.body]
    while pending:
        nodes = []
        stack = list(pending.pop())
        while stack:
            node = stack.pop()
            nodes.append(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pending.append(node.body)
                stack.extend(node.decorator_list)
                stack.extend(node.args.defaults)
                stack.extend(default for default in node.args.kw_defaults if default is not None)
            else:
                stack.extend(ast.iter_child_nodes(node))
        yield nodes


def _calendar_violations(source: str, relpath: str) -> set[tuple[str, int]]:
    """The calendar calls of ``source`` reached through another component's ``.config.dataset``.

    Parameters
    ----------
    source : str
        The module's text.
    relpath : str
        Its path relative to the repository root, used in the report.

    Returns
    -------
    set of (str, int)
        ``(relpath, line)`` of every offending call, in both the direct and the
        split form; empty when the module is clean.
    """
    tree = ast.parse(source, filename=relpath)
    found: set[tuple[str, int]] = set()
    for nodes in _scopes(tree):
        aliases = {
            target.id
            for node in nodes
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and _is_another_components_dataset(node.value)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        for node in nodes:
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in CALENDAR_METHODS:
                continue
            owner = node.func.value
            direct = _is_another_components_dataset(owner)
            split = isinstance(owner, ast.Name) and owner.id in aliases
            if direct or split:
                found.add((relpath, node.lineno))
    return found


def _tree_violations() -> set[tuple[str, int]]:
    found: set[tuple[str, int]] = set()
    for root in ROOTS:
        for path in _python_files(REPO_ROOT / root):
            relative = path.relative_to(REPO_ROOT)
            if relative.parts[: len(FACTOR_LAYER.parts)] == FACTOR_LAYER.parts:
                continue
            found |= _calendar_violations(path.read_text(encoding="utf-8"), str(relative))
    return found


def test_only_the_factor_layer_reads_a_factors_dataset_calendar():
    # Positive control: the walk sees the factor layer's own calls (a factor's
    # ``_warm_start`` and forwarding in quantlab/factor/base.py), which are allowed.
    base = REPO_ROOT / "quantlab/factor/base.py"
    assert _calendar_violations(base.read_text(encoding="utf-8"), "quantlab/factor/base.py") == set()
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in CALENDAR_METHODS
        and _config_dataset_root(node.func.value) is not None
        for node in ast.walk(ast.parse(base.read_text(encoding="utf-8")))
    )
    assert sorted(_tree_violations()) == []


def test_a_direct_chain_is_caught():
    source = (
        "def start(factor, begin, bars):\n"
        "    return factor.config.dataset.bar_before(begin, bars)\n"
        "\n"
        "def end(label, last):\n"
        "    return label.config.factor.config.dataset.bar_after(last, 2)\n"
        "\n"
        "def span(factor, begin, last):\n"
        "    return len(factor.config.dataset.calendar(begin, last))\n"
    )
    assert _calendar_violations(source, "quantlab/model/base.py") == {
        ("quantlab/model/base.py", 2),
        ("quantlab/model/base.py", 5),
        ("quantlab/model/base.py", 8),
    }


def test_a_split_chain_is_caught():
    source = (
        "class Label:\n"
        "    def _later_end(self, end):\n"
        "        dataset = self.config.factor.config.dataset\n"
        "        bars = dataset.calendar(end, end)\n"
        "        return dataset.bar_after(end, 2)\n"
        "\n"
        "def start(factor, begin, bars):\n"
        "    dataset: object = factor.config.dataset\n"
        "    panel = dataset.panel(begin, begin)\n"
        "    return dataset.bar_before(begin, bars)\n"
    )
    assert _calendar_violations(source, "quantlab/label/forward.py") == {
        ("quantlab/label/forward.py", 4),
        ("quantlab/label/forward.py", 5),
        ("quantlab/label/forward.py", 10),
    }


def test_a_components_own_dataset_and_the_factors_methods_are_allowed():
    source = (
        "class Risk:\n"
        "    def _previous(self, first):\n"
        "        dataset = self.config.dataset\n"
        "        return dataset.bar_before(first, 1)\n"
        "\n"
        "    def _start(self, begin):\n"
        "        return self.config.dataset.bar_before(begin, self.warmup_bars)\n"
        "\n"
        "def start(factor, begin, bars):\n"
        "    return factor.bar_before(begin, bars)\n"
        "\n"
        "def read(factor, begin, last):\n"
        "    dataset = factor.config.dataset\n"
        "    return dataset.panel(begin, last)\n"
        "\n"
        "def other(dataset, begin):\n"
        "    # ``dataset`` is a parameter here, not the alias of ``read``.\n"
        "    return dataset.bar_before(begin, 1)\n"
    )
    assert _calendar_violations(source, "quantlab/risk/predefined/use4.py") == set()
