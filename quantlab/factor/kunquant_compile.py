"""Compiled KunQuant libraries, kept in the process and on disk: ``compile_graphs``.

KunQuant turns a graph into C++ and compiles it, which takes a second or
more each time. Every compilation in quantlab goes through ``compile_graphs``, which
compiles a graph once and then reuses the library: the same process gets the
library it already loaded, and another process (a ``ChunkedBuilder`` worker,
the next daily run) loads it from the cache directory.

A library is keyed by everything that changes the binary: each graph's text
(``str(Function)``, deterministic across processes, unlike the generated
C++), its ``KunCompilerConfig``, the source of every quantlab module that
defines one of its operators (a custom operator's C++ is written by its
Python class), the ``CppCompilerConfig`` and the compiler's version, the host
CPU (KunQuant compiles with ``-march=native``) and the KunQuant installation.

The cache directory is ``$QUANTLAB_KUNQUANT_CACHE``, else
``$XDG_CACHE_HOME/quantlab/kunquant``, else ``~/.cache/quantlab/kunquant``; an
empty ``QUANTLAB_KUNQUANT_CACHE`` keeps libraries in the process only. Each
entry is a directory named by the key holding the library and a ``complete``
marker with the library's digest; an entry without the marker, or whose
library does not match it, is compiled again. Processes compiling the same
graph at once publish its entry one at a time under a lock file beside it
(``.<key>.lock``), and an intact entry is never removed. Deleting the
directory clears the cache.
"""

import contextlib
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

import KunQuant.runner.KunRunner as kr
from KunQuant.Driver import KunCompilerConfig, compileit as generate_sources
from KunQuant.jit import cfake
from KunQuant.jit.env import get_compiler_env
from KunQuant.Stage import Function

#: Environment variable naming the cache directory; empty keeps no disk cache.
CACHE_ENV = "QUANTLAB_KUNQUANT_CACHE"

#: Bumped when the layout of an entry or what the key covers changes.
CACHE_FORMAT = 1

#: The marker file that makes an entry complete, holding the library's digest.
_COMPLETE = "complete"


#: The libraries loaded in this process, by key.
_LIBRARIES: dict[str, kr.Library] = {}

_LOCK = threading.Lock()

Graphs = list[tuple[str, Function, KunCompilerConfig]]


def cache_dir() -> Path | None:
    """Return the directory compiled libraries are cached in, or ``None`` for none.

    Examples
    --------
    >>> os.environ[CACHE_ENV] = "/tmp/kunquant"
    >>> cache_dir()
    PosixPath('/tmp/kunquant')
    >>> os.environ[CACHE_ENV] = ""
    >>> cache_dir() is None
    True
    """
    value = os.environ.get(CACHE_ENV)
    if value is not None:
        return Path(value).expanduser() if value else None
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "quantlab" / "kunquant"


def compile_graphs(
    graphs: Graphs, libname: str, compiler: "cfake.CppCompilerConfig | None" = None
) -> kr.Library:
    """Return the compiled library of ``graphs``, compiling only when no cache has it.

    Takes the arguments of ``KunQuant.jit.cfake.compileit``, whose library it
    returns: one ``(module name, graph, KunCompilerConfig)`` per module, the
    library's name and the C++ compiler config (KunQuant's default when
    ``None``). The library is looked up in this process, then in
    ``cache_dir()``, and compiled when neither has it.

    Parameters
    ----------
    graphs : list[tuple[str, KunQuant.Stage.Function, KunCompilerConfig]]
        The modules of the library.
    libname : str
        The library's name.
    compiler : cfake.CppCompilerConfig, optional
        The C++ compiler config.

    Returns
    -------
    KunRunner.Library
        The loaded library; ``getModule(name)`` returns each module.

    Examples
    --------
    >>> b = Builder()
    >>> with b:
    ...     Output(Input("x") * 2.0, "y")
    >>> config = KunCompilerConfig(input_layout="TS", output_layout="TS")
    >>> lib = compile_graphs([("Double", Function(b.ops), config)], "Double")
    >>> compile_graphs([("Double", Function(b.ops), config)], "Double") is lib
    True
    """
    compiler = compiler or cfake.CppCompilerConfig()
    key = graph_key(graphs, libname, compiler)
    with _LOCK:
        lib = _LIBRARIES.get(key)
        if lib is None:
            directory = cache_dir()
            if directory is None:
                lib = _compile_in_temp(graphs, libname, compiler)
            else:
                lib = _load_entry(directory / key, libname, compiler)
                if lib is None:
                    lib = _compile_entry(directory, key, graphs, libname, compiler)
            _LIBRARIES[key] = lib
    return lib


def graph_key(graphs: Graphs, libname: str, compiler: "cfake.CppCompilerConfig") -> str:
    """Return the hex digest a library of ``graphs`` is cached under.

    Two calls give the same key exactly when the same C++ would be compiled
    the same way; see the module docstring for what the key covers.

    Examples
    --------
    >>> len(graph_key(graphs, "Double", cfake.CppCompilerConfig()))
    64
    """
    record = {
        "format": CACHE_FORMAT,
        "libname": libname,
        "kunquant": _kunquant_installation(),
        "host": _host(),
        "compiler": _compiler_record(compiler),
        "graphs": [
            {
                "name": name,
                "graph": str(function),
                "strict_window": function.strict_window,
                "config": json.dumps(dataclasses.asdict(config), sort_keys=True, default=repr),
                "operators": _operator_sources(function),
            }
            for name, function, config in graphs
        ],
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _operator_sources(function: Function) -> dict[str, str]:
    """Return a digest of the source of each non-KunQuant module defining an operator class."""
    sources = {}
    for op in function.ops:
        for cls in type(op).__mro__:
            module = sys.modules.get(cls.__module__)
            path = getattr(module, "__file__", None)
            if path is None or cls.__module__.startswith(("KunQuant", "builtins")):
                continue
            sources[cls.__module__] = _file_digest(path)
    return dict(sorted(sources.items()))


@functools.cache
def _file_digest(path: str) -> str:
    """Return the SHA-256 of a file, once per process."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@functools.cache
def _kunquant_installation() -> list[str]:
    """Return KunQuant's version and its runtime library's directory."""
    return [importlib.metadata.version("KunQuant"), cfake.get_runtime_path()]


@functools.cache
def _host() -> list[str]:
    """Return the platform and the CPU the native code is compiled for."""
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = _output(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
    elif Path("/proc/cpuinfo").exists():
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        cpu = "\n".join(
            sorted({line for line in lines if line.startswith(("model name", "flags"))})
        )
    return [sys.platform, platform.machine(), cpu]


def _compiler_record(compiler: "cfake.CppCompilerConfig") -> dict:
    """Return the fields of ``compiler`` that change the binary, and its version."""
    machine = compiler.machine
    return {
        "opt_level": compiler.opt_level,
        "machine": [type(machine).__name__]
        + ([] if not dataclasses.is_dataclass(machine) else sorted(dataclasses.asdict(machine).items())),
        "other_flags": list(compiler.other_flags),
        "fast_linker_threads": compiler.fast_linker_threads,
        "compiler": compiler.compiler,
        "version": _compiler_version(compiler.compiler),
        "dll_ext": compiler.dll_ext,
    }


@functools.cache
def _compiler_version(compiler: str) -> str:
    """Return the ``--version`` output of the C++ compiler, empty when it cannot run."""
    return _output([compiler, "--version"], env=get_compiler_env())


def _output(command: list[str], env=None) -> str:
    """Return a command's standard output, empty when it fails."""
    try:
        return subprocess.run(command, capture_output=True, text=True, env=env, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_entry(entry: Path, libname: str, compiler: "cfake.CppCompilerConfig") -> "kr.Library | None":
    """Load the library of a complete, intact entry; ``None`` when it is missing or damaged."""
    library = entry / f"{libname}.{compiler.dll_ext}"
    marker = entry / _COMPLETE
    if not (marker.is_file() and library.is_file()):
        return None
    try:
        if json.loads(marker.read_text())["sha256"] != _digest(library):
            return None
        return kr.Library.load(str(library.resolve()))
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def _compile_entry(
    directory: Path, key: str, graphs: Graphs, libname: str, compiler: "cfake.CppCompilerConfig"
) -> kr.Library:
    """Compile into a staging directory, move it to the entry and load it.

    The staging directory becomes the entry by one rename, after its
    ``complete`` marker is written, so no process sees a half-written entry.
    Processes compiling the same graph compile side by side, then publish one
    at a time under the key's lock file: an intact entry already in place
    (another process finished first) is loaded and this one dropped; anything
    else at the key (a damaged or unfinished entry, a stray file) is removed
    before the rename. An intact entry is never removed, so a process loading
    it outside the lock always finds it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    entry = directory / key
    staging = Path(tempfile.mkdtemp(dir=directory, prefix=f".{key[:16]}-"))
    try:
        library = _compile_sources(_sources(graphs), libname, compiler, staging)
        for path in staging.iterdir():
            if path != library:
                path.unlink()
        (staging / _COMPLETE).write_text(json.dumps({"sha256": _digest(library)}))
        with _locked(directory / f".{key}.lock"):
            lib = _load_entry(entry, libname, compiler)
            if lib is not None:
                return lib
            _remove(entry)
            staging.rename(entry)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    lib = _load_entry(entry, libname, compiler)
    if lib is None:
        raise RuntimeError(f"compile_graphs: the library compiled into {entry} cannot be loaded.")
    return lib


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on the file ``path`` (created if missing) across processes.

    Without ``fcntl`` (Windows) the lock is a no-op.
    """
    with open(path, "a") as handle:
        if fcntl is None:
            yield
            return
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _remove(path: Path) -> None:
    """Remove whatever is at ``path``: a directory tree, a file or a link; nothing when it is missing."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _compile_in_temp(graphs: Graphs, libname: str, compiler: "cfake.CppCompilerConfig") -> kr.Library:
    """Compile into a temporary directory, load the library and remove the directory."""
    with tempfile.TemporaryDirectory() as directory:
        return kr.Library.load(str(_compile_sources(_sources(graphs), libname, compiler, Path(directory))))


def _sources(graphs: Graphs) -> list[tuple[str, str]]:
    """Generate the C++ of every module, as ``cfake.compileit`` does."""
    return [
        (f"{name}_{index}", source)
        for name, function, config in graphs
        for index, source in enumerate(generate_sources(function, name, **dataclasses.asdict(config)))
    ]


def _compile_sources(
    sources: list[tuple[str, str]], libname: str, compiler: "cfake.CppCompilerConfig", directory: Path
) -> Path:
    """Compile and link ``sources`` into ``directory`` and return the library's path.

    The steps of ``cfake.compileit`` after source generation, with the files
    kept in ``directory``.
    """
    get_compiler_env()  # as cfake.compileit, set up the compiler environment once
    objects = compiler.for_each(
        sources,
        lambda named: cfake.call_cpp_compiler_src(named[1], named[0], compiler, str(directory)),
    )
    library = directory / f"{libname}.{compiler.dll_ext}"
    cfake.call_cpp_compiler(
        compiler.builder.build_link_options(compiler, objects, str(library)), str(library)
    )
    return library


def _digest(path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["CACHE_ENV", "cache_dir", "compile_graphs", "graph_key"]
