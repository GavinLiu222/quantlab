"""``compile_graphs``: a KunQuant library is compiled once per graph, then reused.

A compiled library is kept in the process and written to the cache
directory (``QUANTLAB_KUNQUANT_CACHE``), keyed by the graph, its compiler
options, the C++ compiler and the KunQuant installation. Here the C++ step
(``_compile_sources``) is counted: the same graph built twice compiles once,
in the same process and from a fresh process cache; a changed graph or
option compiles again, and so does a changed parameter of any quantlab
operator (``kunquant_ts``, ``kunquant_cs``); an entry left unfinished or damaged on disk is
compiled again rather than loaded; and the outputs are the same whether
the cache is used or not. Processes compiling the same graph at once keep
the entry the first of them published, so none removes a library another
is loading.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, Input, OpBase, Output
from KunQuant.Stage import Function

import quantlab.factor.kunquant_compile as kunquant_compile
from quantlab.factor import kunquant_cs, kunquant_ts
from quantlab.factor.kunquant import BATCH_OPTIONS, run_batch_graph
from quantlab.factor.kunquant_compile import CACHE_ENV, compile_graphs, graph_key


def _graph(scale: float = 2.0) -> Function:
    b = Builder()
    with b:
        Output(Input("x") * scale, "y")
    return Function(b.ops)


def _entry(name: str = "Scaled", scale: float = 2.0, **options):
    config = KunCompilerConfig(
        input_layout="TS", output_layout="TS", options={**BATCH_OPTIONS, **options}
    )
    return [(name, _graph(scale), config)]


@pytest.fixture
def compiles(monkeypatch, tmp_path):
    """Count the C++ compilations, with an empty cache directory and process cache."""
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(kunquant_compile, "_LIBRARIES", {})
    calls = []
    compile_sources = kunquant_compile._compile_sources

    def counted(*args, **kwargs):
        calls.append(1)
        return compile_sources(*args, **kwargs)

    monkeypatch.setattr(kunquant_compile, "_compile_sources", counted)
    return calls


def _run(lib, name: str = "Scaled") -> np.ndarray:
    from quantlab.factor.kunquant import shared_executor
    import KunQuant.runner.KunRunner as kr

    x = np.arange(24, dtype=np.float32).reshape(3, 8)
    return kr.runGraph(shared_executor(2), lib.getModule(name), {"x": x}, 0, 3)["y"]


def test_the_same_graph_compiles_once_in_a_process(compiles) -> None:
    first = compile_graphs(_entry(), "Scaled")
    second = compile_graphs(_entry(), "Scaled")
    assert len(compiles) == 1
    assert second is first


def test_a_fresh_process_loads_the_library_from_disk(compiles, monkeypatch) -> None:
    compile_graphs(_entry(), "Scaled")
    monkeypatch.setattr(kunquant_compile, "_LIBRARIES", {})
    lib = compile_graphs(_entry(), "Scaled")
    assert len(compiles) == 1
    np.testing.assert_array_equal(_run(lib), np.arange(24, dtype=np.float32).reshape(3, 8) * 2)


def test_a_changed_graph_option_or_compiler_has_its_own_key() -> None:
    compiler = cfake.CppCompilerConfig()
    key = graph_key(_entry(), "Scaled", compiler)
    assert graph_key(_entry(), "Scaled", compiler) == key  # rebuilt, same graph
    assert graph_key(_entry(scale=3.0), "Scaled", compiler) != key
    assert graph_key(_entry(opt_reduce=True), "Scaled", compiler) != key
    assert graph_key(_entry(name="Other"), "Scaled", compiler) != key
    assert graph_key(_entry(), "Other", compiler) != key
    assert graph_key(_entry(), "Scaled", cfake.CppCompilerConfig(opt_level=2)) != key


def test_outputs_are_the_same_with_and_without_the_cache(compiles, monkeypatch) -> None:
    x = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    times, symbols = np.arange(5), np.array([f"S{i}" for i in range(8)])
    cached = run_batch_graph("Scaled", _graph(), {"x": x}, times, symbols, 2)
    monkeypatch.setattr(kunquant_compile, "_LIBRARIES", {})
    cached_again = run_batch_graph("Scaled", _graph(), {"x": x}, times, symbols, 2)
    monkeypatch.setenv(CACHE_ENV, "")  # no disk cache
    monkeypatch.setattr(kunquant_compile, "_LIBRARIES", {})
    uncached = run_batch_graph("Scaled", _graph(), {"x": x}, times, symbols, 2)
    assert len(compiles) == 2
    np.testing.assert_array_equal(cached["y"].values, uncached["y"].values)
    np.testing.assert_array_equal(cached_again["y"].values, uncached["y"].values)


def _entries(cache: Path) -> list[Path]:
    """The entries of the cache directory: everything but the dot-named lock files and staging directories."""
    return [path for path in cache.iterdir() if not path.name.startswith(".")]


def _compile_in_another_process() -> None:
    """Compile ``_entry()`` into the cache from a separate Python process."""
    code = (
        "from tests.test_kunquant_compile_cache import _entry; "
        "from quantlab.factor.kunquant_compile import compile_graphs; "
        "compile_graphs(_entry(), 'Scaled')"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)


def test_another_process_loads_the_library_without_compiling(compiles, tmp_path) -> None:
    _compile_in_another_process()
    lib = compile_graphs(_entry(), "Scaled")
    assert len(compiles) == 0
    np.testing.assert_array_equal(_run(lib), np.arange(24, dtype=np.float32).reshape(3, 8) * 2)


def test_a_damaged_entry_is_compiled_again(compiles, tmp_path) -> None:
    # The entry comes from another process: overwriting a library this process
    # has loaded would break its mapping (SIGBUS on Linux), and glibc's dlopen
    # would return the loaded handle for the same path anyway.
    _compile_in_another_process()
    (entry,) = _entries(tmp_path / "cache")
    (library,) = [path for path in entry.iterdir() if path.name.startswith("Scaled.")]
    library.write_bytes(b"not a library")
    lib = compile_graphs(_entry(), "Scaled")
    assert len(compiles) == 1
    np.testing.assert_array_equal(_run(lib), np.arange(24, dtype=np.float32).reshape(3, 8) * 2)


def test_an_unfinished_entry_is_compiled_again(compiles, monkeypatch, tmp_path) -> None:
    compile_graphs(_entry(), "Scaled")
    (entry,) = _entries(tmp_path / "cache")
    (entry / kunquant_compile._COMPLETE).unlink()
    monkeypatch.setattr(kunquant_compile, "_LIBRARIES", {})
    compile_graphs(_entry(), "Scaled")
    assert len(compiles) == 2
    assert (entry / kunquant_compile._COMPLETE).exists()


def test_a_compile_that_finishes_after_another_process_keeps_its_entry(monkeypatch, tmp_path) -> None:
    # Processes compiling the same graph at once: another one renames its entry into place
    # while this one is still compiling. Removing that intact entry to put this one's in its
    # place would pull the library from under a third process loading it meanwhile.
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(kunquant_compile, "_LIBRARIES", {})
    compile_sources = kunquant_compile._compile_sources
    written = {}

    def finishes_second(*args, **kwargs):
        library = compile_sources(*args, **kwargs)
        _compile_in_another_process()
        (entry,) = _entries(tmp_path / "cache")
        written["inode"] = (entry / library.name).stat().st_ino
        return library

    monkeypatch.setattr(kunquant_compile, "_compile_sources", finishes_second)
    lib = compile_graphs(_entry(), "Scaled")
    (entry,) = _entries(tmp_path / "cache")
    (library,) = [path for path in entry.iterdir() if path.name.startswith("Scaled.")]
    assert library.stat().st_ino == written["inode"]
    np.testing.assert_array_equal(_run(lib), np.arange(24, dtype=np.float32).reshape(3, 8) * 2)


def _damaged_entry(cache: Path) -> Path:
    """Leave a damaged entry (a library without its marker) at ``_entry()``'s key."""
    entry = cache / graph_key(_entry(), "Scaled", cfake.CppCompilerConfig())
    entry.mkdir(parents=True)
    (entry / "Scaled.so").write_bytes(b"not a library")
    return entry


def test_a_damaged_entry_another_process_repaired_meanwhile_is_kept(monkeypatch, tmp_path) -> None:
    # Two processes find the same damaged entry and compile: the other one replaces it with an
    # intact entry while this one is still compiling, and this one must not remove that.
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(kunquant_compile, "_LIBRARIES", {})
    entry = _damaged_entry(tmp_path / "cache")
    compile_sources = kunquant_compile._compile_sources
    written = {}

    def repaired_meanwhile(*args, **kwargs):
        library = compile_sources(*args, **kwargs)
        _compile_in_another_process()
        (repaired,) = [path for path in entry.iterdir() if path.name.startswith("Scaled.")]
        written["inode"] = repaired.stat().st_ino
        return library

    monkeypatch.setattr(kunquant_compile, "_compile_sources", repaired_meanwhile)
    lib = compile_graphs(_entry(), "Scaled")
    (library,) = [path for path in entry.iterdir() if path.name.startswith("Scaled.")]
    assert library.stat().st_ino == written["inode"]
    np.testing.assert_array_equal(_run(lib), np.arange(24, dtype=np.float32).reshape(3, 8) * 2)


def test_a_file_at_the_key_is_replaced_by_the_entry(compiles, tmp_path) -> None:
    entry = tmp_path / "cache" / graph_key(_entry(), "Scaled", cfake.CppCompilerConfig())
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"not an entry")
    lib = compile_graphs(_entry(), "Scaled")
    assert entry.is_dir() and (entry / kunquant_compile._COMPLETE).is_file()
    np.testing.assert_array_equal(_run(lib), np.arange(24, dtype=np.float32).reshape(3, 8) * 2)


def test_an_empty_cache_variable_writes_nothing(compiles, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(CACHE_ENV, "")
    compile_graphs(_entry(), "Scaled")
    assert not (tmp_path / "cache").exists()


# Each quantlab operator built twice, the second time with one parameter
# changed: the parameter is in the op's attributes, so in ``str(Function)``
# and the key. ``x, y, w, u, s, i, e`` are inputs (value, second value,
# weight, universe, size, industry, eligible).
_EW = ("EWSum", "EWMean", "EWVar", "EWCov", "EWBeta", "EWAlpha", "EWResidualStd")


def _ew(name: str, window: int, half_life: float):
    op = getattr(kunquant_ts, name)
    if name in ("EWSum", "EWMean", "EWVar"):
        return lambda x, y, *_: op(x, window, half_life)
    return lambda x, y, *_: op(y, x, window, half_life)


_CHANGED_PARAMETER = {
    "WindowedZScore window": (
        lambda x, *_: kunquant_ts.WindowedZScore(x, 20),
        lambda x, *_: kunquant_ts.WindowedZScore(x, 21),
    ),
    **{f"{name} window": (_ew(name, 60, 20.0), _ew(name, 61, 20.0)) for name in _EW},
    **{f"{name} half_life": (_ew(name, 60, 20.0), _ew(name, 60, 21.0)) for name in _EW},
    "CMRA months": (lambda x, *_: kunquant_ts.CMRA(x, 12, 21), lambda x, *_: kunquant_ts.CMRA(x, 11, 21)),
    "CMRA month_length": (lambda x, *_: kunquant_ts.CMRA(x, 12, 21), lambda x, *_: kunquant_ts.CMRA(x, 12, 20)),
    **{
        f"{name} {bound}": (
            lambda x, *_, op=getattr(kunquant_cs, name): op(x, 0.01, 0.99),
            lambda x, *_, op=getattr(kunquant_cs, name), bound=bound: op(
                x, 0.02 if bound == "lower" else 0.01, 0.98 if bound == "upper" else 0.99
            ),
        )
        for name in ("CrossSectionalWinsorize", "CrossSectionalTrim")
        for bound in ("lower", "upper")
    },
    "RenormalizedCombine weights": (
        lambda x, y, *_: kunquant_cs.RenormalizedCombine([x, y], [0.75, 0.25]),
        lambda x, y, *_: kunquant_cs.RenormalizedCombine([x, y], [0.7, 0.3]),
    ),
    "CrossSectionalTopN n": (
        lambda x, *_: kunquant_cs.CrossSectionalTopN(x, 100),
        lambda x, *_: kunquant_cs.CrossSectionalTopN(x, 50),
    ),
    "CrossSectionalSigmaClip data_error": (
        lambda x, y, w, u, *_: kunquant_cs.CrossSectionalSigmaClip(x, u, 10.0, 3.0),
        lambda x, y, w, u, *_: kunquant_cs.CrossSectionalSigmaClip(x, u, 9.0, 3.0),
    ),
    "CrossSectionalSigmaClip clip": (
        lambda x, y, w, u, *_: kunquant_cs.CrossSectionalSigmaClip(x, u, 10.0, 3.0),
        lambda x, y, w, u, *_: kunquant_cs.CrossSectionalSigmaClip(x, u, 10.0, 2.5),
    ),
    **{
        f"CrossSectionalNeutralize {flag}": (
            lambda x, y, w, u, s, i, e: kunquant_cs.CrossSectionalNeutralize(x, s, i),
            lambda x, y, w, u, s, i, e, flag=flag: kunquant_cs.CrossSectionalNeutralize(x, s, i, **{flag: False}),
        )
        for flag in ("use_industry", "use_size")
    },
    **{
        f"CrossSectionalIndustrySizeFill {flag}": (
            lambda x, y, w, u, s, i, e: kunquant_cs.CrossSectionalIndustrySizeFill(x, s, i, w, u, e),
            lambda x, y, w, u, s, i, e, flag=flag: kunquant_cs.CrossSectionalIndustrySizeFill(
                x, s, i, w, u, e, **{flag: False}
            ),
        )
        for flag in ("use_industry", "use_size")
    },
}

#: quantlab operators whose only arguments are other ops.
_WITHOUT_PARAMETERS = {
    "CrossSectionalZScore",
    "CrossSectionalWeightedMean",
    "CapWeightedStandardize",
    "CrossSectionalWLSResidual",
}


def _operator_key(make) -> str:
    b = Builder()
    with b:
        Output(make(*(Input(name) for name in "xywusie")), "out")
    config = KunCompilerConfig(input_layout="TS", output_layout="TS", options=dict(BATCH_OPTIONS))
    return graph_key([("Op", Function(b.ops), config)], "Op", cfake.CppCompilerConfig())


@pytest.mark.parametrize("case", sorted(_CHANGED_PARAMETER))
def test_a_changed_operator_parameter_has_its_own_key(case: str) -> None:
    before, after = _CHANGED_PARAMETER[case]
    assert _operator_key(before) == _operator_key(before)  # rebuilt, same graph
    assert _operator_key(after) != _operator_key(before)


def test_every_quantlab_operator_has_a_key_case() -> None:
    operators = {
        name
        for module in (kunquant_ts, kunquant_cs)
        for name, value in vars(module).items()
        if isinstance(value, type)
        and issubclass(value, OpBase)
        and value.__module__ == module.__name__
        and not name.startswith("_")
    }
    covered = {case.split()[0] for case in _CHANGED_PARAMETER} | _WITHOUT_PARAMETERS
    assert operators - covered == set(), "add a case to _CHANGED_PARAMETER for each new operator parameter"
