# Developer guide

This guide is for contributors: people who add a vendor, a dataset, a factor, a model head
or a portfolio rule, or who change the machinery underneath. It assumes you know the
pipeline from the [user guide](../user-guide/index.md).

## Architecture overview

The architecture overview is a short illustrated tour of the layers, the panels they
exchange and the run records: {download}`architecture.pdf <../architecture/architecture.pdf>`.

### Panels

Every layer exchanges `xarray.Dataset` panels on the dimensions `(timestamp, symbol)`,
stored as Zarr and read by date range. DataFrames are used inside a layer (Polars factors,
Parquet raw tiers) but never between layers.

### Layer order

The layers form one chain, and a layer imports only the layers before it:

```text
utils / enums -> core -> backend -> tracking -> execution -> runs -> universe -> dataset
  -> config -> acquisition -> analysis -> factor -> label -> model -> risk -> portfolio
  -> backtest -> api
```

The rule has no exceptions: an import inside a function counts too, and
`tests/test_layer_layout.py` fails on any import against the order. The same test pins
the two engines that have exactly one importer: vectorbt only in
`quantlab/backtest/engine_vectorbt.py`, and cvxpy only in
`quantlab/portfolio/predefined/mean_variance.py`.

### Package layout

- Each layer keeps its root class in `<layer>/base.py` and its configs in
  `<layer>/config.py`. The implementations quantlab ships live in `<layer>/predefined/`,
  and a framework module never imports them.
- The public top-level entries of a layer are framework modules (`base.py`, `config.py`,
  shared engines such as `acquisition/registry.py` or `dataset/raw.py`) or one entry per
  vendor or dataset, a package when it has several modules. Code that serves the layer but
  is not its public surface goes in `<layer>/_support/`.
- Layer and `_support/` packages have empty `__init__.py` files. Import every module by
  its full dotted path, and add no re-exports. `quantlab/__init__.py` holds only
  `__version__`. Three packages are entry modules whose `__init__.py` is the module itself:
  `quantlab/acquisition/wrds/`, `quantlab/dataset/crsp/` and `quantlab/dataset/nbbo/`.
  [Internals](internals.md#empty-package-initialisers) explains what depends on this.
- Root modules load no torch; torch is imported only by the modules that need it.
- Every component rebuilds from its recorded config through
  {py:func}`quantlab.core.component.rebuild`.
- The backtest reaches a model only through the
  {py:class}`~quantlab.backtest.base.Predictor` protocol.

There is no command-line interface. `examples/` holds self-contained, runnable
walkthroughs that import only quantlab.

## Running the tests

quantlab uses [uv](https://docs.astral.sh/uv/) and Python 3.13 or later. Install every
extra and the development group, then run the suite in parallel:

```bash
uv sync --all-extras
uv run pytest -n auto
```

The suite has about 5,300 tests and takes about five minutes. `uv sync` without
`--all-extras` uninstalls the extras, and the tests that need them are then skipped or
fail. CI runs `uv sync --locked --dev --all-extras` on Ubuntu and macOS.

A few tests guard the structure rather than behaviour: `tests/test_layer_layout.py` (the
layer order), `tests/test_undefined_names.py` (ruff F821 and F401),
`tests/test_extensibility_contract.py` (no market-specific names in a root class) and
`tests/test_source_inspector.py` (the credential-free read surface).

KunQuant compiles factor graphs at run time and needs a C++ compiler. On macOS, xgboost and
torch link different OpenMP runtimes, and a process that uses both can crash or hang;
`tests/conftest.py` sets `OMP_NUM_THREADS=1` on macOS before any import, and a script that
mixes the two must do the same.

## Building the documentation

The site is built with Sphinx from MyST Markdown pages, and the API reference is generated
from the numpydoc docstrings:

```bash
uv sync --all-extras --group docs
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.

```{toctree}
:maxdepth: 1

extending
internals
style-factors
risk-model
```
