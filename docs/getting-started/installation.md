# Installation

This page explains how to install quantlab, check that the installation works, and
configure the environment variables that hold your data-vendor credentials. Read it before
the [quickstart](quickstart.md). If you only want to run the offline examples and the test
suite, the first three sections are enough; the credential section matters once you start
downloading real market data.

## Requirements

quantlab is a Python library and a set of command-line scripts. It needs:

- Python 3.13 or newer (the `requires-python` field of `pyproject.toml`);
- [uv](https://docs.astral.sh/uv/), which creates the virtual environment, installs the
  pinned dependencies from `uv.lock` and runs commands inside that environment;
- a C++ compiler on the `PATH`. Factors written for the KunQuant backend are compiled to
  native code the first time they run, so a working `g++` or `clang++` is required. Most
  Linux distributions and the macOS Command Line Tools provide one.

You do not need to install Python 3.13 yourself: if uv cannot find a suitable interpreter
it downloads one. Linux and macOS are supported. A GPU is optional (see below).

## Install from PyPI

```bash
pip install quantlab-py         # or: uv add quantlab-py
pip install 'quantlab-py[all]'  # with every optional extra except mosek
```

The distribution is named `quantlab-py` on PyPI (`quantlab` there is an unrelated project);
the package is imported as `quantlab`.

Some components need an optional extra; importing one without it raises an `ImportError` that
names the extra:

| Extra | Installs | Needed by |
|---|---|---|
| `optimize` | cvxpy | `MeanVarianceOptimizer` |
| `tabkit` | pytabkit | `RealMLPRegressor`, `XGBTDRegressor` |
| `wandb` | wandb | `WandbTracker` |
| `mlflow` | mlflow-skinny | `MlflowTracker` |
| `gbdt` | lightgbm, catboost | `LGBMRegressor`, `CatBoostRegressor`, `DoubleEnsembleRegressor` |
| `massive` | boto3 | Massive flat-file downloads |
| `mosek` | cvxpy, mosek | `MeanVarianceOptimizer` with `max_names` (needs a MOSEK licence) |
| `all` | every extra above except `mosek` | |

An installed package keeps its data under `~/.quantlab/data` unless `QUANTLAB_DATA_DIR` is set
(see Environment variables below).

## Install from source

Clone the repository and let uv build the environment:

```bash
git clone https://github.com/ZhaorongDai/quantlab.git
cd quantlab
uv sync --all-extras
```

`uv sync --all-extras` creates a `.venv/` directory in the repository, installs every runtime dependency
listed in `pyproject.toml` and every optional extra at the versions recorded in `uv.lock`
(the test suite covers the optional components too), installs the `dev`
dependency group (which holds `pytest`), and installs quantlab itself in editable mode, so
changes you make to the source are picked up without reinstalling.

The dependency set is large. Besides the numerical stack (NumPy, pandas, Polars, xarray,
Zarr) it includes PyTorch, XGBoost, KunQuant and vectorbt, so the first `uv sync` downloads a few
gigabytes and takes several minutes.

Run every command through `uv run`, which executes it inside the project environment without
you having to activate `.venv/`:

```bash
uv run python -c "import quantlab, sys; print(sys.version)"
```

On the machine these docs were written on, this printed:

```text
3.13.12 (main, Feb  4 2026, 09:25:39) [GCC 13.3.0]
```

## Check the installation

The quickest end-to-end check is the quickstart example. It builds a small synthetic price
panel, computes factors, trains a model and runs a backtest, all offline and on the CPU, in
well under a minute:

```bash
uv run python examples/quickstart.py
```

It should finish with the line `Rebuilt from config.json, same equity curve: True`. The
[quickstart](quickstart.md) walks through what each step does.

## Run the test suite

The test suite is written for pytest and needs no network access and no credentials. Every
vendor client is replaced by a fake and every dataset is synthetic.

```bash
uv run pytest
```

The full suite has close to two thousand tests. The KunQuant tests compile C++ graphs, so a
complete run takes several minutes. To run one area, pass a file or a keyword:

```bash
uv run pytest tests/test_backtest_run.py
uv run pytest -k xgb
```

`tests/test_crsp_rebuild_measurements.py` measures a rebuild of a real CRSP store and fails
with an explanatory message unless the `QUANTLAB_DATA_ROOT` environment variable points at a
checkout that holds one. A plain `uv run pytest` never collects it; it runs only when named:

```bash
QUANTLAB_DATA_ROOT=/path/to/checkout uv run pytest tests/test_crsp_rebuild_measurements.py
```

## GPU support

Deep-learning model heads (subclasses of `quantlab.model.torch_model.TorchModel`) run on a
CUDA GPU when PyTorch can see one and fall back to the CPU otherwise. The choice is made by the
`device` property of `quantlab.model.torch_model.TorchModel`, and nothing needs to be configured. You can
check what PyTorch sees with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On Linux, the PyTorch wheel that uv installs from PyPI already bundles the CUDA runtime, so an
NVIDIA driver is all a GPU machine needs. The shipped torch heads, `GATsRegressor` and
`MASTERRegressor`, run on either device. `RealMLPRegressor` and `XGBoostRegressor` also train on
a CUDA GPU when one is available (XGBoost needs the CUDA build of xgboost, which the Linux wheel
is), and so does `XGBTDRegressor`; none of them uses Apple MPS unless `device="mps"` is passed.
The backtester runs on the CPU, and a model trained on a GPU loads and predicts without one.
See "Choose the training device" in the model guide (`docs/model.md`). The documentation
examples need no GPU.

A torch head also keeps its whole feature panel as tensors on one device. The reserved
hyperparameter `panel_device` chooses it: `"auto"` (the default) puts the panel on the GPU when
it takes at most half of the free GPU memory and in CPU memory otherwise, and `"cuda"` or
`"cpu"` force one. `panel_dtype="float16"` stores the features in half precision so that a
market-wide panel fits on the GPU. See "Keep the training panel on the GPU" in the model guide
(`docs/model.md`).

## macOS: PyTorch and XGBoost in one process

On macOS the XGBoost wheel links Homebrew's OpenMP runtime (`libomp`) while PyTorch bundles
its own copy. When both libraries are loaded into one Python process, the two runtimes clash:
depending on which library is imported first, the process either crashes with an OpenMP error
or hangs. Forcing OpenMP to use a single thread avoids the problem. Set the variable before
Python starts:

```bash
export OMP_NUM_THREADS=1
```

or, in a script or notebook, at the very top, before `numpy`, `torch`, `xgboost` or any
quantlab module is imported:

```python
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
```

The variable is read only when PyTorch loads, so setting it later has no effect. The test suite
applies the same guard in `tests/conftest.py`, and `examples/quickstart.py` does it too.
Linux is not affected, and you do not need the setting on macOS if a process uses only one of
the two libraries.

## Experiment tracking

Nothing is tracked unless a model or backtest config names a tracker in its `tracker` field;
the default sends nothing anywhere, which is what the test suite and the examples rely on.

- **Weights & Biases** is the optional extra `wandb` (`uv sync --extra wandb`, or
  `pip install 'quantlab-py[wandb]'`). Log in once with
  `uv run wandb login`, or set `WANDB_API_KEY`, then use
  `quantlab.tracking.wandb.WandbTracker`. Its `mode="offline"` keeps runs under `./wandb` for
  a later `wandb sync`, and `mode="disabled"` records nothing.
- **MLflow** is an optional extra. Install it with

  ```bash
  uv sync --extra mlflow          # or: pip install 'quantlab-py[mlflow]'
  ```

  which installs `mlflow-skinny`, then use `quantlab.tracking.mlflow.MlflowTracker` with a
  `tracking_uri`: a server, or a local `file:` directory to work offline. A `sqlite:` URI
  needs the full `mlflow` package. Without the extra, quantlab imports as usual and opening
  an MLflow run raises an `ImportError` that names the extra.

See [Track experiments](../model.md#track-experiments) for what each run records.

## Credentials and environment variables

quantlab reads every credential from an environment variable. No script accepts a key on the
command line, and no key is ever written to a config file, a `config.json` or a log. Set only
the variables for the vendors you use:

| Variable | Vendor | Used for |
|----------|--------|----------|
| `TIINGO_API_KEY` | Tiingo | US-equity daily bars |
| `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | Alpaca | US-equity daily and minute bars, quotes and trades |
| `WRDS_USERNAME` | WRDS | CRSP daily stock files and TAQ NBBO quotes |
| `WANDB_API_KEY` | Weights & Biases | Optional experiment tracking through `WandbTracker` |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` or `MLFLOW_TRACKING_TOKEN` | MLflow | Optional experiment tracking through `MlflowTracker`, on a server that asks for credentials |

For WRDS the password is not an environment variable. The PostgreSQL client library reads it
from `~/.pgpass` (or the file named by `PGPASSFILE`), which must contain one line of the form

```text
wrds-pgdata.wharton.upenn.edu:9737:wrds:<your WRDS username>:<your password>
```

and be readable only by you (`chmod 600 ~/.pgpass`). The WRDS client refuses to connect when
`PGHOSTADDR`, `PGSERVICE` or `PGSERVICEFILE` is set, because those would redirect the
connection somewhere other than the WRDS server.

A convenient way to keep the variables out of your shell history is a file you source, kept
outside the repository:

```bash
# ~/.config/quantlab/env, then: source ~/.config/quantlab/env
export TIINGO_API_KEY=...
export APCA_API_KEY_ID=...
export APCA_API_SECRET_KEY=...
export WRDS_USERNAME=...
```

You can ask the data-source registry which credentials it can see. The check reports only
whether each variable is set, never its value. With none of them set, it prints:

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import credential_status

for source in DataSourceRegistry.all():
    print(source.vendor, credential_status(source))
```

```text
alpaca {'APCA_API_KEY_ID': False, 'APCA_API_SECRET_KEY': False}
tiingo {'TIINGO_API_KEY': False}
wrds {'WRDS_USERNAME': False}
```

Two more variables control where data lives. `QUANTLAB_DATA_DIR` sets the root directory the
library's config factories derive raw-download and Zarr paths from; without it the `data/`
directory at the repository root is used in a source checkout, and `~/.quantlab/data` in an
installed package. The WRDS download scripts do not read it: they write
raw files under `--download-dir` and Zarr stores under the data root `--data-dir`, both
defaulting to the current directory. `QUANTLAB_DATA_ROOT` is read only by the CRSP measurement test mentioned above.

## Next steps

- [Quickstart](quickstart.md): the whole pipeline on synthetic data in ten minutes.
- [Concepts](../user-guide/concepts.md): stages, panels, configs and run directories.
- [Data sources](../user-guide/data-sources.md): downloading real data with the credentials
  set up above.
