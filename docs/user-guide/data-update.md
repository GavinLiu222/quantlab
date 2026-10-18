# Keeping data current

Every store quantlab writes can be brought up to a later bar without being rebuilt. Keeping a
data root current is three steps, run in this order after the vendors publish a day:

1. refresh each vendor's **raw tier** (the downloaded files) from its watermarks;
2. bring the **dataset stores** up from the raw tier;
3. **extend** the stores computed from them: factors, labels and the factor risk model.

Each step only appends: earlier rows are never rewritten, so a step that fails can be run again.

## Refresh the raw tiers

{py:func}`~quantlab.acquisition.registry.refresh_raw_tier` pulls everything a vendor published
since its last pull (a Sharadar table, a Massive REST endpoint, a FRED series), each part from its
own watermark. The caller names the vendor, not a roster or a window. It writes raw files only and
returns every part it tried; a part that fails does not stop the others.

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import refresh_raw_tier

downloads = "/data/quantlab/downloads"
for vendor, options in [("sharadar", {}), ("fred", {"series": ["DTB3"]})]:
    result = refresh_raw_tier(DataSourceRegistry.get(vendor), downloads, **options)
    if result.failed:
        print(vendor, "failed:", result.failed)
```

Vendors with a symbol-batched download and no raw-tier refresh (Tiingo, Alpaca) are pulled with
`download_symbols()` instead; see [Data sources](data-sources.md).

## Bring the dataset stores up

A dataset store appends the new bars from the raw tier with `update()`. For Sharadar,
{py:func}`~quantlab.dataset.sharadar.stores.update_stores` does it for every store at once, builds
a store that does not exist yet, and records a store that fails without stopping the rest:

```python
from quantlab.dataset.sharadar.stores import update_stores

outcome = update_stores(downloads, "/data/quantlab")
print(outcome.updated, outcome.failed)
```

A store refuses to resume when a symbol it holds has left the raw tier; `rebuild_dropped=True`
rebuilds it from the raw tier and keeps the old store beside it. See [Datasets](datasets.md) for
how `update()` handles new listings.

## Extend the derived stores

A factor, label or risk-model store records the component that wrote it. Saving that component
beside the store with {py:func}`~quantlab.core.store_folder.save_component` writes its config to
`component.json`; {py:func}`~quantlab.core.store_folder.load_component` rebuilds it, with every
nested component, so an update needs no recipe code:

```python
from quantlab.core.store_folder import load_component, save_component

# once, when the store is built
save_component(factor, "/data/quantlab/factors/market/alpha158", readme="# alpha158\n...")

# every day: rebuild the factor from its folder and append the new bars
factor = load_component("/data/quantlab/factors/market/alpha158")
factor.extend("2026-10-16")
```

{py:meth}`Factor.extend <quantlab.factor.base.Factor.extend>` computes the new bars with the
factor's warm-up read from history, so a store extended day by day equals one built at once. A
factor risk model's regression and estimate stores extend the same way
({py:meth}`RiskStore.extend <quantlab.risk.base.RiskStore.extend>`), the regression first. Extend a store only through the component
that owns it, never through a view of some of its outputs, and extend the inputs first: the
whole-market alphas before the labels and neutralised stores that read them, the style exposures
before the risk model.

## Scheduling

Run the three steps as one scheduled job after the vendors publish (Sharadar's daily tables are out
early on the next morning, New York time). Run it again until the newest bar has reached every
store a downstream job reads, and have that job wait for it rather than updating data itself: a
prediction made on a partly updated day reads stores that disagree on the last bar.
