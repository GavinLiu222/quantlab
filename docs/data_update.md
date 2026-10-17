# Daily data update

One script brings every shared store up to the latest bar, in order: the raw vendor downloads, the
Zarr stores, the factors and the factor risk model. It runs on its own schedule. The paper trading
does not update data; it waits until the update reports that the day is done.

## Contents

- [What it runs](#what-it-runs)
- [Store configs](#store-configs)
- [The status file](#the-status-file)
- [Schedule](#schedule)
- [Add a store](#add-a-store)

## What it runs

`scripts/data_update/update.py` reads `config/data_update.yaml` and runs its stages in order under the data
root `--data-dir`. Each step is one action on one store folder (paths relative to the data root):

| action | what it does |
|--------|--------------|
| `refresh` | keeps a vendor's raw tier current from its watermarks: `{vendor: ..., <options>}` calls `quantlab.acquisition.registry.refresh_raw_tier` with the vendor's options (Sharadar: `trading_days`; Massive: `endpoints`, `end`; FRED: `series`, `start`); raw files only; it raises when a part (a table, an endpoint, a series) failed, and its result names every part |
| `sharadar_stores` | brings every Sharadar store up from the raw tier in the update's own process (`quantlab.dataset.sharadar.stores.update_stores`): `{rebuild_dropped: ..., keep_dropped: [...]}`; it raises when a store failed, after trying every store, and its result counts the stores updated, built and rebuilt |
| `benchmarks` | `scripts/sharadar/price_return_benchmark.py --refresh` for the listed tickers (`market/benchmarks/`) |
| `update` | rebuilds the dataset of a store folder from its `component.json` and calls `update()` |
| `mirror` | appends a source dataset's new bars to a store that copies its variables (`quantlab.backtest.live.mirror_new_bars`) |
| `extend` | rebuilds the folder's component from its `component.json` and extends every store it declares it owns, written by date range, to t (`Component.store_uses()`, the `owns` uses: a factor's store, or a factor risk model's regression and estimate stores, in that order); the step's result names each store, `extended to <t>` or `current` |
| `run` | runs any script under the repository: `{script: ..., args: [...], writes: [...], reads: [...]}`; `{data_dir}` and `{download_dir}` in `args` are filled in (other braces are kept) |

Any other action is an error, in a run and in `--check`. A `benchmarks`, `sharadar_stores` or `run` step
lists in `writes` the store folders, or folder prefixes ending in `/`, it writes (the shipped file:
`market/sharadar/` and `universe/sharadar/` for `sharadar_stores`, which writes those two when it lists none,
`market/benchmarks/` for `benchmarks`); `refresh` writes raw files only and takes no `writes`; `update`, `extend`
and `mirror` write their `store`. A `run` step may list in `reads` the folders or prefixes it reads; the others'
reads come from their components. `--check` refuses a step whose options its action does not take (for `refresh`,
a vendor that is not registered or registers no raw-tier refresh, or an option its refresh does not take, such as
`writes` or a missing FRED `start`), and orders every step's reads against the later steps' writes.

A step with `allow_failure: true` that raises is recorded as failed and the run goes on; whether
the day is ready is still decided by the `ready` stores. A stage with `after_done: true` runs only
after the status file says `done`; a failure there is recorded and the state stays `done`.

t is the last bar of the file's `calendar` store (`market/sharadar/sharadar_sep_1d`) once the
`raw` stage is done. Until every store in the file's `ready` list holds t, and t is newer than
`last_done_t` (the t of the last run that reached `done`, whatever states came after it), the raw
stage is retried every `--retry-minutes` (15) until `--retry-until` (08:30 New York time). A store
is extended only by its owner (`owns_store()`), never by a factor view pinned to some of its
outputs; `extend` refuses a component that declares no ranged store or does not own one.

The shipped file updates what the us3000 paper trading reads, plus the base data: every Sharadar
table (a `refresh` step for the raw tier, then a `sharadar_stores` step with `rebuild_dropped: true`,
which brings every Sharadar store up from it), the FRED
3-month T-bill rate (a `refresh` step, then `update` of its store), the price-return benchmarks, the Barra exposures, the us3000
membership, the us3000 alphas and the USE4 risk stores. The us3000 alphas read Sharadar SEP on
the membership's roster (`RosterDataset`, see the [dataset guide](dataset.md)), so the update keeps
no price copy cut to the roster: SEP is brought up by the raw stage and the membership by the
`universe` stage, both before the `factors` stage. The `universe` stage also brings up the
common-stock membership (`universe/sharadar/sharadar_common_stock_membership`, every security of SEP from
its first to its last priced bar, `CommonStockConstituentDataset`, see the
[constituent guide](constituent.md)), allowed to fail. The raw stage also refreshes
Massive's raw tier (`refresh: {vendor: massive}`, what `scripts/massive/rest_pull.py` runs by hand), which brings every Massive REST endpoint (splits, dividends, news, short interest,
short volume, IPOs, 8-K disclosures and their taxonomy, and the raw-only tables: 10-K sections, 8-K text, risk
factors, Form 3/4, 13-F, reference tickers, ticker events, and a daily Float snapshot) up in the raw tier from each
endpoint's watermark (see "REST pulls" in the [Massive guide](massive.md));
it writes raw files only and is allowed to fail. WRDS data is not in the
daily update; run `scripts/wrds/*.py` when the vendor publishes.

Two failures are handled differently from when the raw stage pulled and updated Sharadar in one
`scripts/sharadar/update.py` step and FRED in one download-and-update step. A failed Sharadar pull (one table the vendor refused or timed out) no longer stops the store
update: the `refresh` step is recorded as failed and the `sharadar_stores` step still brings every store up to
what the raw tier holds; whether the day is ready is decided by the `ready` stores. A FRED series that could
not be downloaded is now a failed step (the `refresh` step raises naming the series, and the step is
allowed to fail), where it was only logged; the `update` of its store still runs on what is on disk.

The last stage, `intraday`, runs once `done` is written (`after_done: true`: its steps are added
to the status file afterwards, the state kept), so nothing waiting for `done` waits for it. It brings
Massive's intraday trades up (see the [Massive guide](massive.md)):
`scripts/massive/convert.py --download` downloads the days after the Trade bar store's last
(trades, minute and day aggregates) and appends their one-minute Trade bars to
`market/massive/massive_trade_bars_1m`, `scripts/massive/day_bars.py` appends the new day
aggregates to Massive's own daily bars `market/massive/massive_day_bars_1d`, then
`scripts/massive/ticker_map.py` appends those days'
rows to the Ticker map `market/massive/massive_ticker_map`, up to the last bar of Sharadar's SEP
store, which the raw stage has brought up; two `update` steps then map the new days onto
permatickers, `market/massive/massive_permaticker_bars_1m` (the Trade bars) and
`market/massive/massive_permaticker_day_bars_1d` (the day aggregates, adjusted by Massive's own
splits and dividends and checked against SEP: the step's result reports how many permaticker-days
disagree, and a disagreement never fails the step; docs/massive.md, "Adjusted by Massive's own
events"), and
`scripts/massive/dividends.py` appends the days SEP, the REST pull and the map all reach to the
dividend panel `market/massive/massive_dividends_1d` (Massive's dividends and splits placed on
permatickers through the map), `scripts/massive/news.py` does the same for the news panel
`market/massive/massive_news_1d` (Massive's news counted per permaticker),
`scripts/massive/short_interest.py` for the short interest panel
`market/massive/massive_short_interest_1d` (a release shown from settlement + 8 trading days),
`scripts/massive/short_volume.py` for the short volume panel `market/massive/massive_short_volume_1d`,
`scripts/massive/disclosures.py` for the 8-K
disclosure panel `market/massive/massive_disclosures_1d` (one flag per primary category of Massive's
disclosure taxonomy), and `scripts/massive/ipos.py` for the IPO panel `market/massive/massive_ipos_1d`
(listed IPOs on their listing day). The scripts are `run` steps with `allow_failure: true`; the
conversion and the map do at most five days each (`--max-days 5`; a store far behind is caught up by a
run of its own), the day bars and the REST panels every new day on disk, and none of the stores is
in `ready`: Massive publishes a day's files at about 00:30 ET the next day, and a late file
never holds the paper trading, which reads none of these stores. Each script writes its folder's
`component.json` and `README.md` when missing.

```bash
cd ~/projects/quantlab2
export SHARADAR_API_KEY=<your-sharadar-key>
QUANTLAB_DATA_DIR=/data/quantlab taskset -c 64-114 .venv/bin/python scripts/data_update/update.py \
    --data-dir /data/quantlab --download-dir /data/quantlab/downloads
```

`--dry-run` prints the steps without running them; `--stage NAME` (repeatable) runs only those
stages, without the retry and without writing the status file. `--check` downloads and computes
nothing: it rebuilds every step's component, checks that `update` names a dataset, that `extend`
names a component declaring ranged stores it owns (a factor owning its store, a factor risk model), and that no step reads a store a later
step writes, that the first stage writes the `calendar` store and every `ready` store, and that a
`mirror` source is a dataset, then prints each problem and exits 1 if there is any. Run it after
editing the file.

## Store configs

A store the update rebuilds holds a `component.json` beside its `README.md`: the component's
`get_config()`, as JSON. `quantlab.core.store_folder.load_component(folder)` turns it back into the
dataset, factor or risk model with every nested component, so the update needs no recipe code.
`save_component(component, folder, readme=...)` writes it (and the README): it first checks that
the config rebuilds into an equal component, so a component held in memory is refused. A recipe
saves the components of the stores it defines; `python
examples/sharadar_us_equity/us3000_h1_mvo.py store-configs` saves the us3000 ones.

## The status file

When it starts (`running`) and when it finishes, the update writes `<data-dir>/update_status.json`
(not with `--dry-run` or `--stage`):

| field | value |
|-------|-------|
| `date` | the New York date of the run |
| `state` | `running`, `done`, `no_new_bar` (nothing new by the cut-off, as on a market holiday) or `failed` |
| `t` | the bar the stores were brought to |
| `last_done_t` | the t of the last run that reached `done` (this run's t when it is `done`); a bar is new only after it |
| `steps` | each step's action, store, result and seconds |
| `reason` | why it stopped, for `no_new_bar` and `failed`; when the raw stage gives up it also names the `allow_failure` steps that failed (e.g. `sharadar failed: ...`) |

`date`, `state`, `t` and `last_done_t` are the contract quantlab-ibkr's `scripts/live_daily.sh`
reads: it waits for `"state": "done"` with today's `date` before it runs the prediction job; a day
that never gets there holds. Change them only together with that script.

The exit status follows the final state: 0 `done`, 1 `failed`, 3 `no_new_bar` (normal on a
holiday, but not a done day). `--check` exits 1 when it finds a problem.

## Schedule

`scripts/data_update/update_daily.sh` is the cron entry. cron calls it every hour and it checks the time in
New York itself, so daylight-saving changes need no edit; at 04 ET on a weekday it runs the update,
with `SHARADAR_API_KEY` from `~/.config/quantlab/sharadar.env` and the Massive keys
(`MASSIVE_API_KEY`, `MASSIVE_S3_ACCESS_KEY_ID`) from the repository's `.env` (cron reads no shell
profile), on NUMA Node 1, under a lock, and logs to `<data-dir>/logs/data_update/<date>.log`.
Massive publishes a day's files at about 00:30 ET the next day (S3 `LastModified`, August to
October 2026: 23:40 to 05:14 ET), before the 04 ET run:

```
0 * * * * $HOME/projects/quantlab2/scripts/data_update/update_daily.sh
```

`update_daily.sh now` runs it at once.

Each run ends with an e-mail in Chinese: the state and t of the status file in the subject, its
steps in the body, and the log's last lines when the update did not exit 0.
`scripts/notify/send_mail.py` sends it over SMTP with the settings in `~/.config/quantlab/mail.env`
(readable by the owner only); without them nothing is sent, and a failed send never fails the run.
The paper trading (quantlab-ibkr `live_daily.sh`) sends its own mails with the same script.

A mail can carry an HTML body with inline images beside the plain text: `--html FILE` adds the
HTML, and each `--image CID=PATH` (repeatable) an image the HTML shows with `<img src="cid:CID">`.
The message is then `multipart/related`, a `multipart/alternative` (the plain body, then the HTML)
followed by the images with their content ids, so a client without HTML still shows the plain
body. Without `--html` the mail is plain text as before. `--image` without `--html`, a malformed or
repeated `CID=PATH`, a missing or unreadable file or one that is not a PNG, JPEG or GIF is
refused before anything is sent.

```bash
python scripts/notify/send_mail.py --subject "收盘简报" --body "纯文本正文" \
    --html briefing.html --image nav=nav.png --image drawdown=drawdown.png
```

```bash
# ~/.config/quantlab/mail.env
QUANTLAB_SMTP_HOST=smtp.163.com          # default; port 465 (SSL) by default, 587 uses STARTTLS
QUANTLAB_SMTP_USER=<sending account>
QUANTLAB_SMTP_PASSWORD=<its SMTP authorisation code>
QUANTLAB_MAIL_TO=<recipients, comma separated>
```

## Add a store

1. Build the store in its folder (`<category>/<group>/<stem>/<stem>.zarr`) and save its component
   there: `save_component(factor, folder, readme="# my_factor\n...")`.
2. Add an `update` or `extend` step to `config/data_update.yaml`, after the steps that update its
   inputs. A new vendor script that takes `--data-dir` goes in as a `run` step.
3. Run `scripts/data_update/update.py --check`; it names any missing `component.json`, a factor
   that does not own its store, or a step placed before the step that writes its inputs.
