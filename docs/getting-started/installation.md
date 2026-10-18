# Installation

quantlab needs Python 3.13 or newer and a C++ compiler on the `PATH` (`g++` or `clang++`):
KunQuant factors compile to native code the first time they run. Linux and macOS are
supported; a GPU is optional.

## Install from PyPI

```bash
pip install quantlab-py         # or: uv add quantlab-py
pip install 'quantlab-py[all]'  # with every optional extra except mosek
```

The distribution is `quantlab-py` (`quantlab` on PyPI is an unrelated project); the package
is imported as `quantlab`. Importing a component whose extra is missing raises an
`ImportError` that names the extra:

| Extra | Installs | Needed by |
|---|---|---|
| `optimize` | cvxpy | `MeanVarianceOptimizer` |
| `tabkit` | pytabkit | `RealMLPRegressor`, `XGBTDRegressor` |
| `gbdt` | lightgbm, catboost | `LGBMRegressor`, `CatBoostRegressor`, `DoubleEnsembleRegressor` |
| `wandb` | wandb | `WandbTracker` |
| `mlflow` | mlflow-skinny | `MlflowTracker` |
| `massive` | boto3 | Massive flat-file downloads |
| `mosek` | cvxpy, mosek | `MeanVarianceOptimizer` with `max_names` (needs a MOSEK licence) |
| `all` | every extra above except `mosek` | |

## Install from source

```bash
git clone https://github.com/ZhaorongDai/quantlab.git
cd quantlab
uv sync --all-extras
```

[uv](https://docs.astral.sh/uv/) creates `.venv/`, downloads a Python 3.13 interpreter if
none is found, installs the versions pinned in `uv.lock` with every extra and the `dev`
group, and installs quantlab in editable mode. The first sync downloads a few gigabytes
(PyTorch, XGBoost, KunQuant, vectorbt). Without `--all-extras`, `uv sync` uninstalls the
extras. Run every command through `uv run`:

```bash
uv run python -c "import quantlab, sys; print(sys.version)"
```

```text
3.13.12 (main, Feb  4 2026, 09:25:39) [GCC 13.3.0]
```

## Check the installation

```bash
uv run python examples/quickstart.py      # offline, CPU, under a minute
uv run pytest -n auto                     # the full suite, offline, several minutes
uv run pytest tests/test_backtest_run.py  # one file
uv run pytest -k xgb                      # one keyword
```

The quickstart ends with `Rebuilt from its run directory, same equity curve: True`; the
[quickstart](quickstart.md) page walks through it. The test suite needs no network and no
credentials: every vendor client is faked and every dataset is synthetic.
`tests/test_crsp_rebuild_measurements.py` runs only when named, with `QUANTLAB_DATA_ROOT`
pointing at a checkout that holds a real CRSP store.

## GPU

Torch heads ({py:class}`~quantlab.model.torch_model.TorchModel` subclasses such as
`GATsRegressor` and `MASTERRegressor`), `RealMLPRegressor`, `XGBoostRegressor` and
`XGBTDRegressor` train on a CUDA GPU when one is visible and on the CPU otherwise; none uses
Apple MPS unless `device="mps"` is passed. On Linux the PyPI torch wheel bundles the CUDA
runtime, so an NVIDIA driver is enough. A model trained on a GPU loads and predicts without
one, and the backtester runs on the CPU. The [models](../user-guide/models.md) guide covers
the device and the `panel_device` / `panel_dtype` hyperparameters.

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## macOS: PyTorch and XGBoost in one process

XGBoost's macOS wheel links Homebrew's `libomp` and PyTorch bundles its own; a process that
loads both crashes or hangs. Force a single OpenMP thread before PyTorch loads, either with
`export OMP_NUM_THREADS=1` or at the top of the script, before any other import:

```python
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
```

`tests/conftest.py` and `examples/quickstart.py` do the same. Linux is unaffected.

## Credentials and data location

Every credential comes from an environment variable; none is passed on a command line or
written to a config, a run directory or a log. Set only the ones for the vendors you use:

| Variable | Vendor |
|----------|--------|
| `SHARADAR_API_KEY` | Sharadar (Nasdaq Data Link) |
| `MASSIVE_API_KEY`, `MASSIVE_S3_ACCESS_KEY_ID`, `MASSIVE_S3_SECRET_ACCESS_KEY` (defaults to the API key) | Massive |
| `TIINGO_API_KEY` | Tiingo |
| `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | Alpaca |
| `WRDS_USERNAME`, with the password in `~/.pgpass` | WRDS (legacy) |
| `WANDB_API_KEY` | Weights & Biases tracking |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` or `MLFLOW_TRACKING_TOKEN` | MLflow tracking, on a server that asks |

FRED needs no key. Keep the exports in a file outside the repository that you `source`.
{py:func}`~quantlab.acquisition.registry.credential_status` reports whether each variable is
set, never its value. With none set:

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import credential_status

for source in DataSourceRegistry.all():
    print(source.vendor, credential_status(source))
```

```text
alpaca {'APCA_API_KEY_ID': False, 'APCA_API_SECRET_KEY': False}
fred {}
massive {'MASSIVE_API_KEY': False, 'MASSIVE_S3_ACCESS_KEY_ID': False}
sharadar {'SHARADAR_API_KEY': False}
tiingo {'TIINGO_API_KEY': False}
wrds {'WRDS_USERNAME': False}
```

`QUANTLAB_DATA_DIR` sets the data root that raw-download and Zarr paths are derived from.
Without it, a source checkout uses `data/` at the repository root and an installed package
uses `~/.quantlab/data`.

## Experiment tracking

Nothing is tracked unless a model or backtest config names a `tracker`.
`quantlab.tracking.wandb.WandbTracker` (extra `wandb`; log in with `uv run wandb login` or
set `WANDB_API_KEY`) accepts `mode="offline"` to keep runs under `./wandb` and
`mode="disabled"` to record nothing. `quantlab.tracking.mlflow.MlflowTracker` (extra
`mlflow`) takes a `tracking_uri`: a server or a local `file:` directory; a `sqlite:` URI needs
the full `mlflow` package.
