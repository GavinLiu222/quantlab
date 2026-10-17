# Internals

This page describes the machinery that keeps quantlab's long-running jobs
safe to interrupt and its results safe to trust: how downloads and
conversions resume where they stopped, how a store is rebuilt without losing
the old one, why the package `__init__.py` files import nothing, how files are
written atomically, how data fingerprints and code records tie a
run to the data it read and the code it ran, and how a model and an ensemble
share one Evaluation and one Walk-forward training. It is written for contributors who change these
parts of the code or add a component that has to cooperate with them. Users
only need the behaviour, which [Data sources](../user-guide/data-sources.md),
[Datasets](../user-guide/datasets.md) and
[Backtesting](../user-guide/backtesting.md) describe.

The examples below were run offline; the file listing comes from the demo
vendor in [Extending quantlab](extending.md#a-data-source).

## Resumable downloads

A full-market download is thousands of requests and can take hours, so every
acquisition must survive being killed at any point and continue without
re-fetching finished work or skipping unfinished work. Three kinds of small
JSON file, all kept under the config's `watermark_path` and never inside the
raw data directory (a Parquet directory scan would choke on them), carry that
state:

```text
_watermarks/demo/AAA.json                          one watermark per symbol
_watermarks/demo/_failures.json                    the failure manifest
_watermarks/demo/_pages/5b7fc3dffa5ab4a0.pages.json one page ledger per batch
raw/demo/month=2024-01/part-5b7fc3dffa5ab4a0-00000.pqt
```

A *watermark* records the date range of a symbol's data already on disk, for
example `{"last_date": "2024-03-29", "start_date": "2024-01-02"}`. Before each
pass, `SymbolBatchAcquisition._run` asks `CoverageLedger` (`quantlab.acquisition._support.coverage`)
which requested symbols are not yet covered for the requested window, and
fetches only those. A watermark is written only after the symbol's whole
batch has been fetched and written, so a symbol interrupted mid-download has
no watermark and is fetched again. `refresh()` starts each symbol at its own `last_date` instead
of the config's start date. The failure manifest lists symbols whose download
failed, with the reason; it accumulates across runs, and a later success
removes the entry.

### Watermark files

Every raw tier keeps a watermark, in one of two shapes.

A **per-symbol sidecar** (`<watermark_path>/<SYMBOL>.json`) is
`SymbolCoverage` in `quantlab.acquisition._support.coverage`: compact JSON
holding `last_date`, `start_date` when recorded and `no_data` when true.
`SymbolBatchAcquisition._write_watermark` writes it and `CoverageLedger`
reads it. A corrupt or non-object sidecar reads as absent (a refresh must
never crash on one; it costs at most a wider re-fetch). Its `last_date` is
the string as recorded, which `refresh()` resumes from per batch of symbols
sharing it.

A **dated watermark** (`_watermark.json` beside the data) is
`quantlab.utils.watermark`: `write_watermark(path, through, labels)` writes
the labels and `through`, `indent=2` with sorted keys, and
`read_watermark(path)` returns `through` as a date, raising on a corrupt file.
The module knows no vendor; each raw tier's writer passes its own labels and
path:

| file | holds | written by |
|---|---|---|
| `sharadar/<table>/_watermark.json` | `table`, `through` | `quantlab.acquisition.sharadar.raw_tier.write_table_watermark` |
| `massive/<type>/_watermark.json` | `data_type`, `through` | `quantlab.acquisition.massive.raw_tier.write_day_file_watermark` |
| `massive/rest/<endpoint>/_watermark.json` | `endpoint`, `field`, `through` | `quantlab.acquisition.massive.raw_tier.write_rest_watermark` |

A snapshot endpoint's watermark has `field` `"snapshot"`, its `through` the
day of the last snapshot. The files are the ones the raw tiers already hold,
so no raw tier is migrated (locked by `tests/test_watermark_record.py`). The
readers stay in the dataset layer (`CoverageLedger.read_coverage`,
`quantlab.dataset.sharadar.tables.read_watermark` and `raw_through`,
`quantlab.dataset.massive.raw.read_watermark` and `read_rest_watermark`); the
writers, like every other raw-tier writer (REST pages, bulk-pull records, the
TICKERS history), are in the acquisition layer.

Two rules are shared: `resume_from(through, start, overlap_days)` (the day
after the watermark, less an overlap, never before `start`), used by Massive's
day files and REST endpoints, and `holds(through, day)`, by which a snapshot
already taken today is skipped and a REST watermark never moves back.
Sharadar's table watermark may move back: a date-window pull records the
window's last day, and an updated pull the day it started, whatever was there.

A *batch* is one request covering several symbols. Vendors such as Alpaca
answer it as a chain of pages linked by opaque tokens, so a batch can itself
be interrupted halfway. The *page ledger* (`quantlab.acquisition._support.page_ledger`)
records, per batch, every page fetched, the token the next request must send,
and the shard files each page was written to. `SymbolBatchAcquisition._fetch_batch` is
the only pagination loop, and it resumes from the ledger:

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

Several properties make resuming safe, and a change to this code must keep
all of them:

- The batch key is a hash of vendor, frequency, dates and the sorted symbol
  list, and shard names are `part-{batch_key}-{page:05d}.pqt` with no
  timestamp or random part. Re-fetching a page overwrites the same file, so a
  retry can cost requests but never duplicates rows.
- A page's shard is written before the ledger records the page. A crash in
  between costs one re-fetch; the opposite order could record a page whose
  data never reached disk.
- The ledger stores a fingerprint of its roster. Opened with a different
  roster, it reads back empty rather than resuming pages fetched for other
  symbols.
- Before resuming, `PageLedger.assert_consistent` checks that every recorded
  shard still exists and raises if one is missing, since resuming past it
  would leave a hole no later read could detect.
- A vendor that returns the token it was given is refused with `ValueError`
  instead of looping forever and writing a new shard on every iteration.
- A missing or corrupt ledger file reads back as an empty ledger, which costs
  that one batch a re-fetch and nothing more.

Each ledger file has one writer, because each batch runs in one worker thread.
Merging ledgers into a shared file would need a lock around the update and the
write.

## Resumable conversions

Converting a large raw tier into a Zarr store is also resumable.
`RawDataset.from_raw_data_chunked` (`quantlab.dataset.raw`, the build path of
every dataset that converts raw files into a store it owns) splits the time axis into windows with
`TimeChunkPlanner` (`quantlab.dataset._support.ledger`), densifies each window onto one
symbol axis fixed for the whole range before the first window (so all windows
line up column by column), appends it to the store, and records it in a
`ChunkLedger` next to the store (`<store>.chunks.json`). A re-run skips the
windows the ledger lists.

The ledger and the store are two records of the same history, and an append
cannot be undone, so `ChunkLedger.assert_consistent` trusts neither alone. It
refuses to resume when the symbol axis changed between runs, when a store
exists but the ledger is empty, when the ledger lists windows but the store is
gone, or when the store's last timestamp differs from the ledger's last
window, which means the process died between appending and recording.

## Rebuilds

A store is derived data: when the conversion code or the symbol list changes,
the store on disk has to be rebuilt from the raw tier. Two mechanisms do
this, and both follow the same rule: the old store is never deleted until the
new one is complete.

When the raw tier gains symbols the store does not have, `on_new_listing`
decides what chunked conversion does. `"widen"` adds the new symbols as NaN
columns through `XrBackend.widen_symbol_axis`, which writes the widened store
beside the original (`.widening.tmp`), then swaps directories with two
renames, briefly parking the original as `.superseded.tmp`. `"rebuild"` renames
the store and its ledger aside with the same `.superseded.tmp` suffix and
converts every window afresh; if the rebuild fails or is cancelled, the
partial store is removed and the originals are renamed back. If even that
rename fails, the error is logged rather than raised, so it cannot hide the
error that stopped the rebuild, and the log names both copies for manual
recovery. A widen refuses to start while a non-empty `.superseded.tmp` copy
is left over from an earlier crash, because that copy may be the only
complete version of the data.

`BaseStoreRebuilder` (`quantlab.dataset._support.rebuild`) is the skeleton for an explicit
rebuild of a whole store, used by the CRSP rebuilder in
`quantlab/acquisition/wrds/rebuild.py`. `rebuild()` runs
`assert_inputs_present`, `backup`, `clear`, `_convert` and `_measure` in that
order: a missing raw tier is reported while the old store is still on disk
(converting nothing would write an empty panel that looks like a period with
no trading), a backup exists before anything is deleted, sidecars are deleted
together with the store so no stale bookkeeping survives, and a measurement is
returned only after the conversion succeeded. `data_root` is required and has
no default, because a rebuild run from a git worktree, which has no `data/`
directory, must not report success against a tree it never read.

## Package `__init__` files that import nothing

`quantlab/acquisition/__init__.py` and `quantlab/acquisition/_support/__init__.py`
are empty, and `quantlab/__init__.py` holds only `__version__`, read through
`importlib.metadata`. A package `__init__` runs on every import beneath it, and
the credential-free `SourceInspector` (`quantlab.acquisition._support.inspector`)
must be importable without loading a vendor client: `tests/test_source_inspector.py`
fails if a client becomes reachable from it, if either acquisition `__init__` is
not empty, or if `quantlab/__init__.py` imports anything outside the standard
library. Do not add imports to those files. Downloads are not estimated or
refused by size.

## Atomic writes

Every JSON sidecar (watermarks, ledgers, failure manifests, backtest
metrics) is written by `quantlab.utils.atomic.write_json_atomically`: the
payload goes to a temporary file in the destination's own directory, is
flushed and `fsync`ed, and is renamed over the destination with `os.replace`.
A crash leaves either the previous complete file or the new one, never a
truncated file that a resumed run would fail to parse. The temporary file has
to be in the same directory, because a rename is atomic only within one
filesystem. There is no lock; two concurrent writers still race, but the
loser sees a complete file. New code that persists state should use this
function rather than `open(...).write`.

Directories follow the same pattern. A backtest run directory is written as a
hidden `.{name}.partial` sibling and renamed into place only after every
artifact succeeded, and the staging directory is removed on any exception,
including `KeyboardInterrupt`. Widening a store swaps whole directories by
rename, as described above.

## Date-ranged stores

A factor's store and a factor risk model's two stores answer only inside the
date range they were built for, recorded beside each in
`<store>.range.json`. One module keeps that lifecycle,
`quantlab.backend.ranged.RangedStore`: it reads and atomically writes the
range file, builds (compute, drop the range, create, record), extends
(refuse a missing or reached range, compute from the first moment after the
recorded end, widen and append when there are bars, record), checks a read's
range, and holds the refusal texts. `Factor` and `RiskStore` each make one
and supply only what differs: the compute function, the name in messages,
and for a factor the owner rule (ADR 0029) and the fill values of widened
cells. A resampled factor without its own store checks its source's range
through one on `config.file_path` (ADR 0002). A chunked build
(`quantlab.factor.chunking.ChunkedBuilder`) writes a chunk computed in a
worker process through the factor's public chunk-write entry
(`write_rows`, `append_rows`), which runs the ranged store's `write` and
`append`; it is a build tool, not a factor or a view (ADR 0043).

Which stores a component uses is one declaration, `Component.store_uses()`
(`()` by default): each component lists only its own direct uses, a
`StoreUse(path, relation, handle)` with relation `reads`, `owns` or
`source`, and a `BuiltFrom(component)` for a sub-component its store is only
built from (a dataset's non-input sub-components). One walk in core,
`quantlab.core.store_uses.walk_store_uses`, visits every sub-component,
de-duplicates and orders the uses deepest first, owners first. Callers filter
by relation: the live day (`quantlab.backtest.live`) checks the dataset
`reads` and extends the `owns`, the daily data update
(`scripts/data_update/update.py`) reads all three and extends a component's
own `owns`. An `owns` use carries a `RangedStoreHandle` (`store_range()`,
`extend(end)`); `Factor` and `RiskStore` are its two adapters; neither
inherits from the other.

## Data fingerprints

A run is reproducible only if the data under it has not changed, and data
does change: stores are appended to, and adjusted prices are restated after
splits and dividends. The run record (`quantlab.runs.record`) records, for
one panel and a list of variables, a SHA-256 digest of each
variable's values, then a SHA-256 digest over the timestamps, the symbol
names and each variable's name, dtype and digest, plus the first and last
timestamp and the axis sizes. The per-variable digests and dtypes are kept in
the record (`variable_digests`, `variable_dtypes`), so a mismatch warning can
say which variables changed. Values are hashed in the dtype they are stored
in, never up-cast, so a float32 store is hashed as float32 and a variable
whose dtype changed has another digest. Every axis is hashed in the order it
is stored, never sorted first, so a store whose symbol axis grew at its end
(ADR 0032) is not copied, and the ordered symbol names are part of the digest:
the same values in another symbol order have another digest. Every NaN is
rewritten to one canonical bit pattern and every `-0.0` to `0.0` in its own
precision, because otherwise two reads of identical data could hash
differently.

Variables are hashed in parallel threads, one variable per thread, up to
eight and the CPU count; a request smaller than one block is hashed in one
thread. Each variable is read and hashed in blocks of whole bars, about 64 MB
each and rounded up to whole store chunks along `timestamp`, so a lazily read
store is never in memory whole, and only one block is copied to make its NaN
canonical. The blocks of a variable are fed to its digest in order, so
neither the thread count nor the block size changes a digest.

Nothing decides what to fingerprint. The data is recorded where it is read
(ADR 0021). A run opens a `DataRecorder`, and the three read seams log every
request made while it is open:

- `BaseDataset.panel(start, end, symbols=None, variables=None)` on a leaf
  dataset, a store or a panel held in memory, keyed by the store the leaf
  declares it reads (`BaseDataset.store_uses()`: its own store, or the source
  store while a resampled dataset resamples on read);
- `Factor.read(start, end)`, a factor store;
- `RiskStore.read(start, end)`, the regression or estimate store of a factor
  risk model. Both its readers go through it: the decision inputs
  (`DecisionInputs`), which read the estimate rows a rule's contexts carry
  (over a panel's rebalance bars, one bar, or a preloaded window) for the
  covariance estimator (`FactorRiskStoreEstimator`), and the backtest's
  factor attribution, which reads both stores over its window.
  A risk store is keyed by the model's path and the store
  (`risk_model.regression`, `risk_model.estimate`), and its reads that differ
  only in their range are one request, over the first to the last bar read,
  so a row read per bar costs one entry and one hash. The bars between reads
  are hashed too. The rows are not a `(timestamp, symbol)` panel: the labels
  of the read variables' other axes (`factor`, `factor_i`, `factor_j`) enter the digest
  after the symbols, and each variable is hashed `timestamp` first. A panel
  has no other axis, so its digest is unchanged.

Each call ends with `record_read(...)`. Outside every recorder that call
returns at once, so research reads cost nothing. With recorders nested, only
the innermost logs. `unrecorded()` keeps a block out of all of them; the
backtester uses it around its training step and around the read of the whole
regression store that only checks the risk model's bar interval. A `MergedDataset` asks each
input for its own names of the requested variables and records nothing
itself. A resampled dataset without its own store records the source store
it read.

When the recorder closes, each distinct request is read once more and hashed
once, over the requested variables, or every variable when none were
requested. `records` maps each key to the list of requests in the order they
were first read: the hash record plus the `request` (`start`,
`end`, `symbols`, `variables`).

The keys are component paths, so one dataset read by several consumers is
one key:

| Opened by | Keys are paths in | Record stored in |
| --- | --- | --- |
| `BaseModel.collect()`, an ensemble's `collect()` | the model (`factors.0.dataset`) | the `run.json` of the unit `train()` or `train_cv()` writes next (`TrainedRun.data_fingerprint`); members and folds hold none |
| `run()`, `run_weights()` | the backtester (`price_dataset`, `model.factors.0.dataset`) | the run's `run.json` (`BacktestRun.data_fingerprint`) |
| each `run_cv()` fold | the backtester | the fold's child run |
| the `run_cv()` stitched pass | the backtester | the run's `run.json` |

A dataset found at several paths takes the first. A dataset outside the tree
is keyed by its class and store path.

A recorder given an expected record compares on close, per key and request,
by `digest` alone. A changed, missing or extra key or request logs one
warning that shows the ranges and sizes as explanation. It never raises,
because changed data can still be worth backtesting.
`BacktestRun.rebuild_backtester()` passes the run's record as
`expected_fingerprint`, each fold's as `expected_fold_fingerprints`, and, for
a train-mode run, the trained unit's as `expected_training_fingerprint`. The
last is compared with `compare` once the retrained unit is written.

If the run fails partway, what was read so far is hashed and compared before
the error propagates. A key or request not read yet is skipped, and every
warning carries a note that an interrupted read may explain the difference.
That diagnostic swallows its own errors, so it can never replace the original
exception.

### Code records

`quantlab.runs.record.code_record` records the code a run used. The
`run.json` of a backtest run and of a top trained unit holds it as `code`
(`BacktestRun.code`, `TrainedRun.code`), built by
`quantlab.core.component.code_of(root)` over the component tree:

- `git` holds the commit of the repository quantlab is imported from and
  `dirty`, which covers tracked files only. It is `None` outside a working
  tree. It is context and is never compared.
- `modules` holds, for every module defining a class of the tree or a base
  class of one:
  - the SHA-256 of its source file;
  - whether it is a framework module, meaning a quantlab module outside
    `predefined/` and `quantlab.dataset`, or a component module, meaning a
    shipped implementation or a user's own class;
  - the component paths that use it.

  Standard-library and installed third-party modules, and classes without a
  source file, are skipped. quantlab's own modules are recorded however it is
  installed.
- `libraries` holds the installed versions of `LIBRARIES` (numpy, pandas,
  xarray, polars, xgboost, torch, vectorbt, KunQuant and cvxpy). They are
  read from package metadata without importing the packages.

`compare` warns once per changed, missing or extra module digest, with
component modules before framework modules and each warning naming the
component paths, and once per changed library version.
`rebuild_backtester()` compares the rebuilt tree with the run at once. A
train-mode rebuild also compares the retrained unit with the unit the run
used (`expected_training_code`).

## Evaluation and walk-forward training

A model and an ensemble are two different kinds of trained unit: an ensemble
composes models and inherits from none of them (ADR 0013, 0017). They still
score and cross-validate the same way, because each of the two procedures is
one module of the model layer, used by both: `quantlab/model/evaluation.py`
and `quantlab/model/walk_forward_training.py`. Neither knows ensembles: an
ensemble adds its member agreement to `evaluate` through `extra_metrics`.

### Evaluation

`quantlab.model.evaluation.evaluate` scores a trained unit after training.
It takes the unit's prediction panel, the labels' raw values, the label
objects with their `label_scales`, the unit's train, validation and test
segments (`Segments`), its test bounds and a directory. It returns the
metrics and writes `ic_series.csv` and `test_predictions.zarr` into the
directory, so all three come from one set of predictions. It needs no model.

- A model (`BaseModel._evaluate`, called by `train_into` after the variant's
  `_fit`) predicts the bars it is scored on with `predict_scored` (from
  `warmup_bars` bars before its first scored bar to its last) and passes its
  own `evaluation_segments()`, in which a segment `config.metric_splits`
  leaves out is empty; the variant's losses cover the same segments
  (`_scored_bars`). The variant's `_fit` returns only
  `{split}_loss`; `train_into` merges it with the evaluation metrics and
  writes the dict once to the tracking run's summary and to `run.json`.
- An ensemble (`BaseEnsemble._evaluate`) combines its members' panel
  predictions with `_combine` and passes, per label, the collected panel and
  `evaluation_segments()` of the first member predicting it (each member
  predicts its own scored bars), plus each
  label's member predictions; a label with at least two gets
  `{split}_member_correlation`.

The rules (every label scored; the IC family always; error metrics only for
a `"raw"` label; `qlike` and `variance_ratio` for a raw volatility label)
live in that one function, so a model and an ensemble can be compared key
for key. A model head supplies its loss only and never scores. No instance
state carries a series between methods: `evaluate` writes the IC series it
computed.

### Walk-forward training

`quantlab.model.walk_forward_training.train_walk_forward(unit,
train_periods, expanding, test_periods)` runs every `train_cv` call. It needs
only the public protocol `WalkForwardTrainable` defined beside it:

| Member | `BaseModel` | `BaseEnsemble` |
|---|---|---|
| `class_name`, `get_config()` | its class, its config | its class, its config |
| `model_save_dir` | `config.model_save_dir` | the first member's |
| `walk_forward_bars()` | the collected bars between `start_date` and `end_date` | the first member's |
| `purge_bars` | the largest `lookahead_bars()` of its labels | the largest over its members |
| `check_hyperparameters()` | the variant's check | every member's, in order |
| `train_fold(fold, run_dir, group)` | `train_into` on `fold_config(config, fold)`, then the old config back | every member on the fold's dates, `_train_into` of the ensemble unit, then every member's config back |
| `tracker`, `tracking_project` | `config.tracker`, its class name | the first member's |
| `provenance()` | `training_record` and `code_of(self)` | the same for the ensemble |

The module owns the order: check the hyperparameters before any directory
exists; lay the folds out with `walk_forward_folds` (unchanged) over
`walk_forward_bars()` with `purge_bars`, refusing bad settings first and an
empty date range next; create the trial directory; train fold i in order into
`fold_{i}/`; read the folds' trained runs; average their metrics
(`cv_mean_metrics`); open `{class}_cv_summary` through `tracker` in the
trial's group; write the walk-forward `run.json` with `provenance()`. A fold
records no provenance; the top unit records it (ADR 0021).

`train_fold` restores the unit's own config in a `finally`, so after
`train_cv`, also a failed one, a model and every ensemble member hold the
dates they were configured with. A new kind of trained unit, such as an
ensemble of different model variants, gets cross-validation by implementing
the protocol publicly; it calls no private method of its members.
`tests/test_walk_forward_training.py` drives the module with a stub that
trains nothing.

A model's `collect()` leaves it a `CollectedPanel`
(`quantlab/model/collected_panel.py`): the panel sorted on both axes, the
record of the reads (`training_record`) and what it was read for.
`panel.array(names, dtype)` builds each array once per variable list and type
and keeps it, so every fit on the panel reads the same float32 features and
labels (the **Training panel**) and the evaluation (`predict_scored`)
reads the features in the panel's own type: on a float32 panel the two share
one array. Walk-forward folds, a second `train()`, seed-ensemble members
(`use_panel`) and models sharing a panel through `use_panel` all convert it
once. `use_panel` refuses a panel read for other factor or label configs or
dates, since the record must describe what the model reads. The arrays are
shared memory (a torch head's panel is a view of them), so a head reads them
and never writes into them: `_transform_feature` returns a new array, and rows
are taken by indexing, which copies. `tests/test_collected_panel.py` locks
the sharing and checks that no fit changes the arrays.
