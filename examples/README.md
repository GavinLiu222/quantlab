# Examples

Each script in this directory is a complete, runnable program. They all work offline: they
generate synthetic prices in a temporary directory, need no credentials and no GPU, and finish
in well under a minute on a laptop CPU. Run any of them from the repository root with `uv`:

```bash
uv run python examples/quickstart.py
```

| Script | What it shows | Guide |
|--------|---------------|-------|
| [`quickstart.py`](quickstart.py) | The whole pipeline: a price panel, factors and a label, an XGBoost model, a backtest, and rebuilding the run from its `config.json` | [Quickstart](../docs/getting-started/quickstart.md) |
| [`inspect_data_sources.py`](inspect_data_sources.py) | The data-source registry, credential checks, a resumable and cancellable download with a stand-in vendor, inspecting files on disk, and the volume check | [Data sources](../docs/user-guide/data-sources.md) |
| [`build_panel.py`](build_panel.py) | Converting raw files into a panel, requesting a date range and symbols, the storage backends, chunked conversion and updates, and masking a panel with a point-in-time universe | [Datasets](../docs/user-guide/datasets.md), [Universes](../docs/user-guide/universes.md) |
| [`train_model.py`](train_model.py) | A custom factor, a forward-return label, training and evaluating an XGBoost model, reloading it from its checkpoint, and walk-forward cross-validation | [Factors](../docs/user-guide/factors.md), [Models](../docs/user-guide/models.md) |
| [`backtest.py`](backtest.py) | Long-only and long/short backtests, a delisted holding, rebuilding a run, and backtesting cross-validation folds as one stitched curve | [Backtesting](../docs/user-guide/backtesting.md) |

## Real data

| Script | What it shows | Guide |
|--------|---------------|-------|
| [`yahoo_us_equity.py`](yahoo_us_equity.py) | The full pipeline on free Yahoo Finance data, with no account: daily bars of the Dow 30 and SPY downloaded with `yfinance` and held in memory as `FrameDataset`s, Alpha158 factors and a 5-bar forward-return label, an XGBoost model, a top-10 backtest against SPY, and rebuilding the run. Needs network access; run with `uv run --with yfinance python examples/yahoo_us_equity.py`. The universe is today's Dow 30, so it shows the pipeline, not a survivorship-free result | [Frame API](../docs/user-guide/frame-api.md) |
| [`sharadar_us_equity/`](sharadar_us_equity/) | `barra_style.py`, the Barra USE4-style exposures (12 styles, 20 descriptors, the industry code and the estimation universe) of every Sharadar common stock with a factor report of the styles, and `risk_model.py`, the USE4 factor risk model on those exposures with its bias statistics. Needs a Sharadar subscription and the stores `quantlab.dataset.sharadar.stores.build_stores` writes; both rewrite shared stores, so point them at a data root of their own to try them | [README](sharadar_us_equity/README.md), [Sharadar](../docs/user-guide/sharadar.md) |

The prices are random walks, sometimes with a small planted effect so the model has something
to find. The numbers the scripts print show what the output looks like; they say nothing about
real markets.

The examples track nothing: no config names a tracker, so the default null tracker is used. On
macOS the examples also set `OMP_NUM_THREADS=1`, because PyTorch and XGBoost ship
conflicting OpenMP runtimes.
