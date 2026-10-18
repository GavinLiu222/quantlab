"""``compile_graphs``: a KunQuant library is compiled once per graph, then reused.

A compiled library is kept in the process and written to the cache
directory (``QUANTLAB_KUNQUANT_CACHE``), keyed by the graph, its compiler
options, the C++ compiler and the KunQuant installation. Here the C++ step
(``_compile_sources``) is counted: the same graph built twice compiles once,
in the same process and from a fresh process cache; a changed graph or
option compiles again; an entry left unfinished or damaged on disk is
compiled again rather than loaded; and the outputs are the same whether
the cache is used or not.
"""

import numpy as np
import pytest
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

import quantlab.factor.kunquant_compile as kunquant_compile
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


def test_a_damaged_entry_is_compiled_again(compiles, monkeypatch, tmp_path) -> None:
    compile_graphs(_entry(), "Scaled")
    (entry,) = (tmp_path / "cache").iterdir()
    (library,) = [path for path in entry.iterdir() if path.name.startswith("Scaled.")]
    library.write_bytes(b"not a library")
    monkeypatch.setattr(kunquant_compile, "_LIBRARIES", {})
    lib = compile_graphs(_entry(), "Scaled")
    assert len(compiles) == 2
    np.testing.assert_array_equal(_run(lib), np.arange(24, dtype=np.float32).reshape(3, 8) * 2)


def test_an_unfinished_entry_is_compiled_again(compiles, monkeypatch, tmp_path) -> None:
    compile_graphs(_entry(), "Scaled")
    (entry,) = (tmp_path / "cache").iterdir()
    (entry / kunquant_compile._COMPLETE).unlink()
    monkeypatch.setattr(kunquant_compile, "_LIBRARIES", {})
    compile_graphs(_entry(), "Scaled")
    assert len(compiles) == 2
    assert (entry / kunquant_compile._COMPLETE).exists()


def test_an_empty_cache_variable_writes_nothing(compiles, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(CACHE_ENV, "")
    compile_graphs(_entry(), "Scaled")
    assert not (tmp_path / "cache").exists()
