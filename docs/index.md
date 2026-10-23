# quantlab documentation

**Version**: {{ release }}

**Useful links**:
[Installation](getting-started/installation.md) |
[Source repository](https://github.com/ZhaorongDai/quantlab) |
[Issue tracker](https://github.com/ZhaorongDai/quantlab/issues) |
[PyPI](https://pypi.org/project/quantlab-py/)

quantlab is a Python library for quantitative equity research: prepare market data,
compute factors, train return models and backtest portfolios. Use DataFrames for a single
task, or connect the stages with configurations and saved runs.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

Install quantlab and run an offline example with synthetic data.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc

The pipeline stage by stage: data, universes, factors and labels, models, portfolios,
backtests and the daily jobs.
:::

:::{grid-item-card} API reference
:link: reference/index
:link-type: doc

Functions and classes grouped by task, with signatures, parameters and examples.
:::

:::{grid-item-card} Developer guide
:link: dev/index
:link-type: doc

The architecture, how to extend each layer, and the machinery underneath: resumable jobs,
atomic writes, fingerprints, the risk model.
:::
::::

## Choose a starting point

| I want to… | Start here |
|---|---|
| Compute factors or backtest my own DataFrame | [Frame API](user-guide/frame-api.md) |
| Run the complete research pipeline | [Quickstart](getting-started/quickstart.md) |
| Understand panels, factors and saved runs | [Concepts](user-guide/concepts.md) |
| Look up a function or a parameter | [API reference](reference/index.rst) |
| Add a dataset, model or portfolio rule | [Extending quantlab](dev/extending.md) |

```{toctree}
:hidden:
:maxdepth: 1

getting-started/index
user-guide/index
reference/index
dev/index
release-notes
```
