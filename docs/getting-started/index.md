# Getting started

Install quantlab, then run the quickstart: it takes the whole pipeline, from a price panel
to a backtest that rebuilds from its own run directory, through synthetic data on a CPU in
about half a minute. After that, [concepts](../user-guide/concepts.md) explains the vocabulary
the rest of the user guide assumes.

If your data is already in a pandas or polars DataFrame and you want one capability (a
factor set, a factor report, a backtest of your own signal), the
[Frame API](../user-guide/frame-api.md) needs no stores or configs.

```{toctree}
:maxdepth: 1

installation
quickstart
```
