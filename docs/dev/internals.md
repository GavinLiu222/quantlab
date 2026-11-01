# Internals

This page describes the machinery that keeps quantlab's long-running jobs safe to interrupt
and its results safe to trust: how downloads and conversions resume, how a store is rebuilt
without losing the old one, how files are written atomically, how date-ranged stores,
store declarations, dataset leaves and views work, how data fingerprints and code records tie a
run to what it read and ran, and how models and ensembles share one evaluation and one
walk-forward training. It is for contributors who change these parts or add a component
that has to cooperate with them. The behaviour users see is described in the
[user guide](../user-guide/index.md).

## Resumable downloads

A full-market download is thousands of requests and can take hours, so every acquisition
must survive being killed at any point and continue without re-fetching finished work or
skipping unfinished work. Three kinds of small JSON file carry that state. They live under
the config's `watermark_path`, never inside the raw directory, where a Parquet scan would
choke on them:

```text
_watermarks/demo/AAA.json                          one watermark per symbol
_watermarks/demo/_failures.json                    the failure manifest
_watermarks/demo/_pages/5b7fc3dffa5ab4a0.pages.json one page ledger per batch
raw/demo/month=2024-01/part-5b7fc3dffa5ab4a0-00000.pqt
```

A *watermark* records the date range of a symbol's data on disk, for example
`{"last_date": "2024-03-29", "start_date": "2024-01-02"}`. Before each pass, the engine asks
`CoverageLedger` (`quantlab.acquisition._support.coverage`) which requested symbols are not
yet covered for the window and fetches only those. A watermark is written only after the
symbol's whole batch has been fetched and written, so a symbol interrupted mid-download has
no watermark and is fetched again. `refresh()` starts each symbol at its own `last_date`.
The failure manifest lists failed symbols with the reason; it accumulates across runs, and
a later success removes the entry. `CoverageLedger.for_config(config)` applies the same
classification without a vendor class, so it works on a machine without API keys.

### Watermark files

Every raw tier keeps a watermark, in one of two shapes.

A **per-symbol sidecar** (`<watermark_path>/<SYMBOL>.json`) is `SymbolCoverage`: compact
JSON holding `last_date`, `start_date` when recorded and `no_data` when true. A corrupt or
non-object sidecar reads as absent: a refresh must never crash on one, and it costs at most
a wider re-fetch.

A **dated watermark** (`_watermark.json` beside the data) is written and read by
{py:mod}`quantlab.utils.watermark`: `write_watermark(path, through, labels)` writes the
labels and `through`, and `read_watermark(path)` returns `through` as a date, raising on a
corrupt file. The module knows no vendor; each raw tier's writer passes its own labels:

| File | Holds | Written by |
|---|---|---|
| `sharadar/<table>/_watermark.json` | `table`, `through` | {py:func}`~quantlab.acquisition.sharadar.raw_tier.write_table_watermark` |
| `massive/<type>/_watermark.json` | `data_type`, `through` | {py:func}`~quantlab.acquisition.massive.raw_tier.write_day_file_watermark` |
| `massive/rest/<endpoint>/_watermark.json` | `endpoint`, `field`, `through` | {py:func}`~quantlab.acquisition.massive.raw_tier.write_rest_watermark` |

A snapshot endpoint's watermark has `field` `"snapshot"` and `through` the day of the last
snapshot. The readers stay in the dataset layer; the writers, like every other raw-tier
writer, are in the acquisition layer. Two rules are shared: `resume_from(through, start,
overlap_days)` (the day after the watermark, less an overlap, never before `start`), used
by Massive's day files and REST endpoints, and `holds(through, day)`, by which a snapshot
already taken today is skipped and a REST watermark never moves back. Sharadar's table
watermark may move back: a date-window pull records the window's last day.

### The page ledger

A *batch* is one request covering several symbols. Vendors such as Alpaca answer it as a
chain of pages linked by opaque tokens, which can only be replayed, never computed, so a
batch can itself be interrupted halfway. The page ledger
(`quantlab.acquisition._support.page_ledger.PageLedger`) records, per batch, every page
fetched, the token the next request must send, and the shard files each page was written
to. `SymbolBatchAcquisition._fetch_batch` is the only pagination loop, and it resumes from
the ledger; a vendor client never calls it directly.

```python
from quantlab.acquisition._support.page_ledger import PageLedger

roster = ["AAPL", "MSFT"]
key = PageLedger.batch_key("alpaca", "1m", "2024-01-02", "2024-01-05", roster)
path = PageLedger.default_path(str(root / "_watermarks"), key)
ledger = PageLedger(path, roster)
ledger.describe(key, "alpaca", "1m", "2024-01-02", "2024-01-05", roster)
ledger.record_page(0, "tok-1", rows=10_000, seen=["AAPL"],
                   shard_paths=[f"date=2024-01-02/part-{key}-00000.pqt"])

print(key, Path(path).relative_to(root))
print(PageLedger(path, roster).resume_point())            # a new process resumes here
print(PageLedger(path, ["AAPL", "NVDA"]).resume_point())  # another roster starts over
```

```text
91c9dc202fdc2cc1 _watermarks/_pages/91c9dc202fdc2cc1.pages.json
(1, 'tok-1')
(0, None)
```

The file is JSON with the batch identity, a `complete` flag, the symbols seen so far and
one record per page (index, rows, next token, shards). Several properties make resuming
safe, and a change to this code must keep all of them:

- The batch key is a 16-character hash of vendor, frequency, dates and the sorted symbols,
  and shard names are `part-{batch_key}-{page:05d}.pqt` with no timestamp or random part.
  Re-fetching a page overwrites the same file, so a retry can cost requests but never
  duplicates rows.
- A page's shard is written before the ledger records the page, and both writes are
  atomic. A crash in between costs one re-fetch; the opposite order could record a page
  whose rows never reached disk.
- The ledger stores a fingerprint of its roster. Opened with a different roster, it reads
  back empty rather than resuming pages fetched for other symbols.
- Before resuming, `PageLedger.assert_consistent` checks that every recorded shard exists
  and raises if one is missing, since resuming past it would leave a hole no later read
  could detect. The message ends with the cure: delete the ledger to re-fetch the batch
  from page 0.
- A vendor that returns the token it was given is refused with `ValueError` instead of
  looping forever and writing a new shard each time.
- A missing or corrupt ledger reads back as empty, which costs that one batch a re-fetch.

`symbols_seen()` says which symbols have no data only once `is_complete()` is true, because
the vendor sorts by symbol and page 0 of a large batch can hold a single symbol; a symbol a
finished batch returned no rows for gets a `no_data` marker. The batch key includes the
dates, so a `refresh()` writes a new ledger and new shard names in the same partition, and
the ledgers of earlier runs stay in `_pages`. Nothing removes them; deleting the directory
is safe once no run is in progress. Each ledger file has one writer, because each batch
runs in one worker thread; a shared ledger would need a lock around the update and the
write.

## Resumable conversions

{py:meth}`RawDataset.from_raw_data_chunked <quantlab.dataset.raw.RawDataset.from_raw_data_chunked>`
splits the time axis into windows with `TimeChunkPlanner`, determines the symbol axis of
the whole range once before the first window (sorted by `sort_symbol_axis`), densifies each
window onto exactly that axis, appends it to the store, and records it in a `ChunkLedger`
next to the store (`<store>.chunks.json`). A symbol with no rows in a window is an all-NaN
column, which keeps every append aligned. The first window pins the chunk grid
(`XrBackend.APPEND_DIM_CHUNK` rows by `XrBackend.SYMBOL_DIM_CHUNK` symbols), so symbols
added later land in chunks already laid out. A re-run skips the windows the ledger lists.

The ledger records each window's first and last timestamp and row count, with a sha256
fingerprint of the ordered symbol list (the same symbols in another order are a different
axis). The ledger and the store are two records of one history, and an append cannot be
undone, so `ChunkLedger.assert_consistent` trusts neither alone. It refuses to resume when
the symbol axis changed, when a store exists without a ledger, when a ledger lists windows
but the store is gone, or when the store's last timestamp differs from the ledger's last
window, which means the process died between appending and recording.

Cleaning runs on each window separately, so a price jump across a window boundary is not
flagged. Integer variables are stored as float64 so that NaN cells created later do not
become zero.

## Rebuilds

A store is derived data, and when the conversion code or the symbol list changes it is
rebuilt from the raw tier. Both mechanisms follow one rule: the old store is never deleted
until the new one is complete.

When the raw tier gains symbols the store lacks, `on_new_listing` decides what chunked
conversion does. `"widen"` adds the new symbols as NaN columns through
`XrBackend.widen_symbol_axis`. It reindexes in memory when the widened store fits under
`XrBackend.MAX_WIDEN_BYTES` (4 GiB) and otherwise rewrites block by block, writing the
widened store beside the original (`.widening.tmp`) and swapping directories with two
renames, briefly parking the original as `.superseded.tmp`. `"rebuild"` renames the store
and its ledger aside with the same suffix and converts every window afresh; if the rebuild
fails or is cancelled, the partial store is removed and the originals are renamed back. If
even that rename fails, the error is logged rather than raised, so it cannot hide the error
that stopped the rebuild, and the log names both copies. A widen refuses to start while a
non-empty `.superseded.tmp` copy is left from an earlier crash, because that copy may be
the only complete version of the data.

`BaseStoreRebuilder` (`quantlab.dataset._support.rebuild`) is the skeleton for an explicit
rebuild of a whole store, used by the CRSP rebuilder in `quantlab/acquisition/wrds/rebuild.py`.
`rebuild()` runs `assert_inputs_present`, `backup`, `clear`, `_convert` and `_measure` in
that order: a missing raw tier is reported while the old store is still on disk
(converting nothing would write an empty panel that looks like a period without trading), a
backup exists before anything is deleted, sidecars are deleted with the store so no stale
bookkeeping survives, and a measurement is returned only after the conversion succeeded.
`data_root` has no default, because a rebuild run from a git worktree, which has no data
directory, must not report success against a tree it never read.

## Empty package initialisers

`quantlab/acquisition/__init__.py` and `quantlab/acquisition/_support/__init__.py` are
empty, and `quantlab/__init__.py` holds only `__version__`, read through
`importlib.metadata`. A package `__init__` runs on every import beneath it, and the
credential-free `SourceInspector` (`quantlab.acquisition._support.inspector`) must be
importable without loading a vendor client. `tests/test_source_inspector.py` fails if a
client becomes reachable from it, if either acquisition `__init__` is not empty, or if
`quantlab/__init__.py` imports anything outside the standard library.

## Atomic writes

Every JSON sidecar (watermarks, ledgers, failure manifests, backtest metrics) is written by
{py:func}`quantlab.utils.atomic.write_json_atomically`: the payload goes to a temporary file
in the destination's own directory, is flushed and `fsync`ed, and is renamed over the
destination with `os.replace`. A crash leaves the previous complete file or the new one,
never a truncated file a resumed run would fail to parse. The temporary file must be in the
same directory, because a rename is atomic only within one filesystem. There is no lock:
two concurrent writers still race, but the loser sees a complete file. New code that
persists state uses this function rather than `open(...).write`.

Directories follow the same pattern. A backtest run directory is written as a hidden
`.{name}.partial` sibling and renamed into place only after every artifact succeeded; the
staging directory is removed on any exception, `KeyboardInterrupt` included. Widening a
store swaps whole directories by rename.

## Date-ranged stores

A factor's store and a factor risk model's two stores answer only inside the date range
they were built for, recorded beside each in `<store>.range.json`. One module keeps that
lifecycle, {py:class}`~quantlab.backend.ranged.RangedStore`: it reads and atomically writes
the range file, builds (compute, drop the range, create, record), extends (refuse a missing
or reached range, compute from the first moment after the recorded end, widen and append,
record), checks a read's range, and holds the refusal texts.
{py:class}`~quantlab.factor.base.Factor` and {py:class}`~quantlab.risk.base.RiskStore` each
make one and supply only what differs: the compute function, the name in messages, and for
a factor the owner rule and the fill values of widened cells. A resampled factor without
its own store checks its source's range. A chunked build
({py:class}`~quantlab.factor.chunking.ChunkedBuilder`) writes a chunk computed in a worker
process through the factor's public chunk-write entries (`write_rows`, `append_rows`); it
is a build tool, not a factor or a view.

Which stores a component uses is one declaration,
`store_uses()` on {py:class}`~quantlab.core.component.Component` (empty by
default). Each component lists only its own direct uses, a
{py:class}`~quantlab.core.store_uses.StoreUse` with relation `reads`, `owns` or `source`,
and a {py:class}`~quantlab.core.store_uses.BuiltFrom` for a sub-component its store is only
built from. One walk, {py:func}`~quantlab.core.store_uses.walk_store_uses`, visits every
sub-component, de-duplicates and orders the uses deepest first, owners first. Callers
filter by relation: the live day checks the dataset `reads` and extends the `owns`, and the
daily data update reads all three and extends a component's own `owns`. An `owns` use
carries a {py:class}`~quantlab.core.store_uses.RangedStoreHandle` (`store_range()`,
`extend(end)`); `Factor` and `RiskStore` are its two adapters.

## Dataset leaves

A dataset is a *leaf* or a *view*. A leaf answers `panel` from a store of its own, or a
panel held in memory; a store written from other datasets is still a leaf's. A view reads
other datasets, its inputs, and keeps no store. The dataset root class
({py:class}`~quantlab.dataset.base.BaseDataset`) describes a leaf only: its `store_path`
and `store_uses` read the leaf's config, and its `inputs()` returns nothing, so the store
walk reaches every dataset. A dataset is read on the bars of its store and is not
resampled; a factor is (`Factor.resample`), cutting its bars as the dataset's
`resample_labels` says.

The root class's defaults are the read of a Zarr leaf. Its config
({py:class}`~quantlab.dataset.config.BaseDatasetConfig`) holds the store path and the
`name`. `panel(start, end, symbols, variables)` asks `_open_bars(start, end)` for every
symbol and variable between the dates, which opens `_read_store()` (the `store_path`)
through `_open_store` and slices it; `panel` then applies the symbol and variable query
and records the read, folding `_build_settings()` into its digest. The calendar,
`stored_symbols` and `head` read the same store. A leaf built from raw data adds the build
path and the build fields (the dates, `symbols`, `kwargs`) through
{py:class}`~quantlab.dataset.raw.RawDataset` and
{py:class}`~quantlab.dataset.config.RawDatasetConfig`; the root has neither.

A leaf overrides a root member only to change how it reads, never to escape a default
that does not fit it. The overrides fall into a few kinds:

| What the leaf changes | Members it overrides | Examples |
|---|---|---|
| Its config: fields to check, state derived from them | `_normalize_config`, `_on_config_installed` | most leaves built from raw data check the fields their config class adds and drop what they cached from the previous config |
| Where its bars come from | `_open_bars`, `_open_store`, `_read_store`, `_calendar`, `head`, `stored_symbols` | the factor store reads every output under a prefix; the mapped bars refuse a day mapped by a ticker map that has since changed; the in-memory panel opens no store |
| What a read records | `_build_settings` | the Trade bars and the NBBO bars record the settings their bars were cut with |
| A market rule or the KunQuant export | `tradable_bars`, `ticker_lookup`, `resample_labels`, `_to_kunquant` | the Sharadar bars need a real trade to be tradable; the Trade bars cut daily bars by exchange session |
| Its identity, for a panel held in memory | `__init__`, `__eq__`, `copy`, `persist_with_run`, `resolve_run_config` | {py:class}`~quantlab.dataset.memory.FrameDataset` holds its panel, compares it, and writes it with a run |

Setting `config_cls` (the config class a leaf rebuilds from) or `COLUMN_MAP` (its own
column names mapped to the shared ones) declares the leaf and is not an override.
`tests/test_dataset_leaf_overrides.py` holds the frozen table of every leaf's overrides,
each with its reason, and fails on an override the table does not list; writing a leaf is
described in [Extending quantlab](extending.md#a-dataset).

## Dataset views

Every view inherits
{py:class}`~quantlab.dataset.view.DatasetView`, a `MarketDataset` that holds the view
decisions:

- `store_path` is `None`: a view keeps no store;
- `store_uses()` declares no store of its own; its inputs declare the stores it reads, and
  the walk reaches them, so `stores()` names its inputs' stores;
- the config normalisation sets only `name`, as on the root class; a view's config holds
  only its inputs and its own options.

The forwarded members are the read interface of `MarketDataset` except `panel`: the
calendar (`_calendar`, behind `calendar`, `bar_before` and `bar_after`),
`resample_labels`, `ticker_lookup`, `stored_symbols`, `head`, `own_names`,
`shared_name_map`, `tradable_bars`, `delisting_bars`, `_own_delisting_symbols` and
`_to_kunquant`. With one input, `DatasetView` by default answers each from the input
`_source()` returns, and a view overrides the members it changes (`RosterDataset` its
stored symbols, `PermatickerDataset` its symbol axis, calendar and rows). With several
inputs `_source()` returns `None` and each member raises `NotImplementedError` naming the
member and the view: the calendar of a merge is a union and a market rule belongs to the
input holding the symbol, so no single-input default is right there. `panel` is abstract.

Whether a dataset delists symbols by a rule of its own is the explicit hook
`_own_delisting_symbols()`, empty on `MarketDataset`. A leaf with such a rule returns the
symbols it rules (a leaf that overrides `delisting_bars` and not the hook raises
`TypeError` when its class is defined), a single-input view forwards the hook, and
`MergedDataset` combines its inputs' answers, so a leaf's rule survives any stack of views.
Views compose: a roster over
a bad-print mask over a merge answers its calendar, market rules and store uses through
each layer. `tests/test_dataset_view.py` locks the forwarding rule and
`tests/test_dataset_hierarchy.py` the class layout; writing a view is described in
[Extending quantlab](extending.md#a-dataset-view).

## Data fingerprints

A run is reproducible only if the data under it has not changed, and data does change:
stores are appended to, and adjusted prices are restated after splits and dividends. The
run record ({py:mod}`quantlab.runs.record`) records, for one panel and a list of variables,
a SHA-256 digest of each variable's values, then a SHA-256 digest over the timestamps, the
symbol names and each variable's name, dtype and digest, plus the first and last timestamp
and the axis sizes. The per-variable digests and dtypes are kept (`variable_digests`,
`variable_dtypes`), so a mismatch warning can say which variables changed.

- Values are hashed in their stored dtype, never up-cast, so a float32 store is hashed as
  float32.
- Every axis is hashed in stored order, never sorted, so the same values in another symbol
  order have another digest and a store whose symbol axis grew at its end is not copied.
- Every NaN is rewritten to one canonical bit pattern and every `-0.0` to `0.0`, because
  otherwise two reads of identical data could hash differently.
- Variables are hashed in parallel threads (up to eight and the CPU count), each read in
  blocks of whole bars of about 64 MB rounded to whole store chunks, so a lazily read store
  is never in memory whole. Blocks are fed in order, so neither the thread count nor the
  block size changes a digest.

Nothing decides what to fingerprint: data is recorded where it is read. A run opens a
{py:class}`~quantlab.runs.record.DataRecorder`, and three read seams log every request made
while it is open:

- `BaseDataset.panel(start, end, symbols=None, variables=None)` on a leaf dataset, keyed by
  the store the leaf declares it reads;
- `Factor.read(start, end)`, a factor store;
- `RiskStore.read(start, end)`, the regression or estimate store of a factor risk model,
  read by the decision inputs for the covariance estimator and by the backtest's factor
  attribution. Its reads that differ only in range are one request over the first to the
  last bar read, so a row read per bar costs one entry and one hash. Its rows are not a
  `(timestamp, symbol)` panel: the labels of the other axes (`factor`, `factor_i`,
  `factor_j`) enter the digest after the symbols.

Each call ends with {py:func}`~quantlab.runs.record.record_read`. Outside every recorder it
returns at once, so research reads cost nothing. With recorders nested, only the innermost
logs, and {py:func}`~quantlab.runs.record.unrecorded` keeps a block out of all of them (the
backtester uses it around its training step). A `MergedDataset` asks each input for its own
names of the requested variables and records nothing itself.

When the recorder closes, each distinct request is read once more and hashed once, over the
requested variables or every variable. The record maps each key to its requests in the
order first read: the hash record plus the `request` (`start`, `end`, `symbols`,
`variables`). The keys are component paths, so one dataset read by several consumers is one
key:

| Opened by | Keys are paths in | Record stored in |
| --- | --- | --- |
| `BaseModel.collect()`, an ensemble's `collect()` | the model (`factors.0.dataset`) | the trained unit `train()` or `train_cv()` writes next (`TrainedRun.data_fingerprint`) |
| `run()`, `run_weights()` | the backtester (`price_dataset`, `model.factors.0.dataset`) | the backtest run (`BacktestRun.data_fingerprint`) |
| each `run_cv()` fold | the backtester | the fold's child run |
| the `run_cv()` stitched pass | the backtester | the backtest run |

A dataset found at several paths takes the first; one outside the tree is keyed by its
class and store path.

A recorder given an expected record compares on close, per key and request, by digest
alone. A changed, missing or extra key or request logs one warning showing the ranges and
sizes; it never raises, because changed data can still be worth backtesting.
`BacktestRun.rebuild_backtester()` passes the run's record as `expected_fingerprint`, each
fold's as `expected_fold_fingerprints`, and for a train-mode run the trained unit's as
`expected_training_fingerprint`. If the run fails partway, what was read so far is hashed
and compared before the error propagates, every warning noting that an interrupted read may
explain the difference; the diagnostic swallows its own errors, so it never replaces the
original exception.

### Code records

{py:func}`~quantlab.runs.record.code_record` records the code a run used. A backtest run and
a top trained unit hold it as `code` (`BacktestRun.code`, `TrainedRun.code`), built by
{py:func}`~quantlab.runs.record.code_of` over the component tree:

- `git`: the commit of the repository quantlab is imported from and `dirty` (tracked files
  only), or `None` outside a working tree. It is context and never compared.
- `modules`: for every module defining a class of the tree or a base class of one, the
  SHA-256 of its source file, whether it is a framework module (a quantlab module outside
  `predefined/` and `quantlab.dataset`) or a component module (a shipped implementation or a
  user's class), and the component paths that use it. Standard-library and third-party
  modules are skipped.
- `libraries`: the installed versions of numpy, pandas, xarray, polars, xgboost, torch,
  vectorbt, KunQuant and cvxpy, read from package metadata without importing them.

{py:func}`~quantlab.runs.record.compare` warns once per changed, missing or extra module
digest, component modules first and each warning naming the component paths, and once per
changed library version. `rebuild_backtester()` compares the rebuilt tree with the run at
once; a train-mode rebuild also compares the retrained unit with the one the run used.

## Evaluation and walk-forward training

A model and an ensemble are two kinds of trained unit: an ensemble composes models and
inherits from none of them. They still score and cross-validate the same way, because each
procedure is one module used by both, and neither module knows ensembles.

### Evaluation

{py:func}`quantlab.model.evaluation.evaluate` scores a trained unit after training. It takes
the unit's prediction panel, the labels' raw values, the label objects with their
`label_scales`, the train, validation and test segments, the test bounds and a directory,
returns the metrics and writes the IC series and the test predictions into the directory,
so all three come from one set of predictions. It needs no model.

- A model (`BaseModel._evaluate`, called by `train_into` after the variant's `_fit`)
  predicts the bars it is scored on with `predict_scored` and passes its own
  `evaluation_segments()`. The variant's `_fit` returns only `{split}_loss`; `train_into`
  merges it with the evaluation metrics and writes them once.
- An ensemble (`BaseEnsemble._evaluate`) combines its members' predictions with `_combine`
  and passes, per label, the segments of the first member predicting it plus each label's
  member predictions; a label with at least two gets `{split}_member_correlation` through
  `extra_metrics`.

The rules (every label scored; the IC family always; error metrics only for a `"raw"`
label; `qlike` and `variance_ratio` for a raw volatility label) live in that one function,
so a model and an ensemble can be compared key for key.

### Walk-forward training

{py:func}`~quantlab.model.walk_forward_training.train_walk_forward` runs every `train_cv`
call. It needs only the public protocol
{py:class}`~quantlab.model.walk_forward_training.WalkForwardTrainable`:

| Member | `BaseModel` | `BaseEnsemble` |
|---|---|---|
| `class_name`, `get_config()` | its class, its config | its class, its config |
| `model_save_dir` | `config.model_save_dir` | the first member's |
| `walk_forward_bars()` | the collected bars between `start_date` and `end_date` | the first member's |
| `purge_bars` | the largest `lookahead_bars()` of its labels | the largest over its members |
| `check_hyperparameters()` | the variant's check | every member's, in order |
| `train_fold(fold, run_dir, group)` | `train_into` on the fold's config, then the old config back | every member on the fold's dates, the ensemble unit, then every member's config back |
| `tracker`, `tracking_project` | `config.tracker`, its class name | the first member's |
| `provenance()` | the training record and `code_of(self)` | the same for the ensemble |

The module owns the order: check the hyperparameters before any directory exists; lay out
the folds with `walk_forward_folds` over `walk_forward_bars()` and `purge_bars`, refusing bad
settings first and an empty range next; create the trial directory; train fold `i` into
`fold_{i}/`; read the folds' trained runs; average their metrics; open the summary through
the tracker; write the walk-forward record with `provenance()`. A fold records no
provenance; the top unit does. `train_fold` restores the unit's config in a `finally`, so
after `train_cv`, failed or not, a model and every ensemble member hold their configured
dates. A new kind of trained unit gets cross-validation by implementing the protocol
publicly. `tests/test_walk_forward_training.py` drives the module with a stub that trains
nothing.

A model's `collect()` leaves it a {py:class}`~quantlab.model.collected_panel.CollectedPanel`:
the panel sorted on both axes, the record of the reads and what it was read for.
`panel.array(names, dtype)` builds each array once per variable list and type and keeps it,
so every fit reads the same float32 features and labels, and on a float32 panel the
evaluation shares the same array. Walk-forward folds, a second `train()`, seed-ensemble
members and models sharing a panel through `use_panel` all convert it once; `use_panel`
refuses a panel read for other factor or label configs or dates. The arrays are shared
memory (a torch head's panel is a view of them), so a head reads them and never writes into
them: `_transform_feature` returns a new array, and rows are taken by indexing, which
copies. `tests/test_collected_panel.py` locks the sharing.
