# quantlab documentation

**Version**: {{ release }}

**Useful links**:
[Installation](getting-started/installation.md) |
[Source repository](https://github.com/ZhaorongDai/quantlab) |
[Issue tracker](https://github.com/ZhaorongDai/quantlab/issues) |
[PyPI](https://pypi.org/project/quantlab-py/)

quantlab is a configuration-driven backend for quantitative equity research. It downloads
market data, turns it into `(timestamp, symbol)` panels stored as Zarr, computes factors,
trains return models, builds portfolios and backtests them, and every stage rebuilds from the
config it records.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

New to quantlab? Install it and run the whole pipeline on synthetic data in ten minutes.
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

Every public module, class and function, generated from the numpydoc docstrings.
:::

:::{grid-item-card} Developer guide
:link: dev/index
:link-type: doc

The architecture, how to extend each layer, and the machinery underneath: resumable jobs,
atomic writes, fingerprints, the risk model.
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

getting-started/index
user-guide/index
reference/index
dev/index
release-notes
```
