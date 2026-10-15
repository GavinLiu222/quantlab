"""Byte lock on a backtest run's ``report.html`` and ``metrics.json`` (#115).

The inputs of the report became public functions in
``quantlab.runs.backtest_report`` (``report_summary``, ``report_windows``,
``report_chart_inputs``, ``report_portfolio_inputs``) so another executor can
write a report in quantlab's format. The backtester builds its own report
through them, and its pages and metrics must not change by a byte. What is
locked here: ``run()`` (long-only with a benchmark, long-short with a
delisting and no benchmark), ``run_weights()`` with and without a benchmark,
and ``run_cv()`` with and without a benchmark each write a ``report.html``
and a ``metrics.json`` whose SHA-256, after replacing the three things that
differ between two identical runs (the temporary directory, the run
directory's timestamped name and plotly's random div ids) and rounding every
decimal number to 10 significant digits, is the one recorded below, captured
on the code before #115.

The rounding is there because the last digits of a float differ between
CPUs: numpy sums in another order on another SIMD width, so a Sharpe ratio
printed on macOS as ``-3.9547411421850533`` prints as ``-3.9547411421850547``
on a Linux runner, from the 13th significant digit on. Ten digits keep every
change a reader of the report could see. A number below 1e-12 in magnitude is
the round-off of a zero (the Holdings tab's cash share of a fully invested
book: ``1.1e-16`` on one CI runner, 0 or ``-2.2e-16`` on another) and is
written 0. The bytes still depend on the
installed plotly, numpy and vectorbt; a version bump that changes them shows
up here first.
The same runs with the holdings left out of ``write_backtest_report`` (a
caller with no holdings to show) must give the reports' bytes from before
the Holdings tab (#225).
Beside the hashes, the recipe ``report_windows`` documents for building
``run_cv()`` fold rows from ``metrics.json`` is checked against the bars each
fold traded.

The hashes lock bytes on disk, so they read ``report.html`` and ``metrics.json``
by name; the fold-row check reads the run through ``BacktestRun`` (#134).

A change that is meant to alter the page or the metrics updates the hashes;
the failure message names the scenario and file, and the normalized text is
written next to the run so it can be diffed.

Everything is synthetic, CPU-only and offline.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from quantlab.utils.date_range import bar_label
from tests.backtest_fixtures import (
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
WINDOW_START = 30
WINDOW_END = 55

CV_BARS = 80
CV_TRAIN_PERIODS = 30
CV_FIRST_TEST_BAR = 30
CV_LAST_TEST_BAR = 77

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_RUN_NAME = re.compile(r"\b(\w+?)_\d{8}_\d{6}_\d{6}\b")
#: A decimal number on its own: not part of a word, a version or a longer number.
_DECIMAL = re.compile(r"(?<![\w.])-?\d+(?:\.\d+(?:[eE][-+]?\d+)?|[eE][-+]?\d+)(?![\w.])")


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _model_dates(bars, last: int) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[last]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[last]),
    )


def _single_run_config(root: Path, *, benchmark: bool, **overrides) -> CrossSectionBacktestConfig:
    """Load-mode config over a checkpoint trained on bars 0-24, window bars 30-55."""
    bars = pd.bdate_range("2024-01-01", periods=N_BARS)
    delist_at = overrides.pop("delist_at", None)
    dataset_config = write_price_store(root / "store", n_bars=N_BARS, delist_at=delist_at)
    dates = _model_dates(bars, 29)
    checkpoint = train_checkpoint(make_model(root / "train", dataset_config, **dates))
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / "backtest", dataset_config, **dates),
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(bars[WINDOW_START]),
        end_date=_day(bars[WINDOW_END]),
        output_dir=str(root / "runs"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        benchmark_dataset=(
            make_stock_dataset(
                write_price_store(root / "benchmark", symbols=["QQQ"], seed=7, n_bars=N_BARS)
            )
            if benchmark
            else None
        ),
    )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


def _run_long_only_benchmark(root: Path) -> Path:
    return USEquityCrossectionSelectStockVectorBt(
        _single_run_config(root, benchmark=True)
    ).run().run_dir


def _run_long_short_delisting(root: Path) -> Path:
    config = _single_run_config(
        root,
        benchmark=False,
        constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=2)),
        rebalance_periods=3,
        fees=0.002,
        delist_at={"CCC": 42},
    )
    return USEquityCrossectionSelectStockVectorBt(config).run().run_dir


def _weights(bars: np.ndarray, symbols: list[str]) -> xr.DataArray:
    """Hold rows except every fifth bar, which rotates a long-short book."""
    rows = np.full((bars.size, len(symbols)), np.nan)
    for i in range(0, bars.size, 5):
        row = np.zeros(len(symbols))
        row[(i // 5) % len(symbols)] = 0.5
        row[(i // 5 + 2) % len(symbols)] = 0.3
        row[(i // 5 + 4) % len(symbols)] = -0.2
        rows[i] = row
    return xr.DataArray(
        rows, dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": symbols}
    )


def _run_weights(root: Path, *, benchmark: bool) -> Path:
    config = _single_run_config(
        root, benchmark=benchmark, model=None, model_mode=None, checkpoint=None
    )
    backtester = USEquityCrossectionSelectStockVectorBt(config)
    prices = xr.open_zarr(config.price_dataset.config.zarr_file_path)
    bars = prices.timestamp.values[WINDOW_START : WINDOW_END + 1]
    return backtester.run_weights(_weights(bars, list(prices.symbol.values))).run_dir


@pytest.fixture(scope="module")
def cv_project(tmp_path_factory):
    """One real ``train_cv`` run: 8 folds of 6 test bars over bars 30..77."""
    root = tmp_path_factory.mktemp("lock_cv_project")
    dataset_config = write_price_store(root, n_bars=CV_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    model = make_model(root / "train", dataset_config, n_forward_periods=2, **_cv_dates(bars))
    model.collect()
    cv = model.train_cv(train_periods=CV_TRAIN_PERIODS)
    return root, dataset_config, bars, cv.path


def _cv_dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[CV_BARS - 1]),
        train_start=_day(bars[0]),
        train_end=_day(bars[CV_TRAIN_PERIODS - 1]),
        test_start=_day(bars[CV_TRAIN_PERIODS]),
        test_end=_day(bars[CV_BARS - 1]),
    )


def _run_cv(root: Path, cv_project, *, benchmark: bool) -> Path:
    project_root, dataset_config, bars, project_dir = cv_project
    config = CrossSectionBacktestConfig(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / "backtest", dataset_config, n_forward_periods=2, **_cv_dates(bars)),
        model_mode="load",
        cv_project_dir=str(project_dir),
        start_date=_day(bars[CV_FIRST_TEST_BAR]),
        end_date=_day(bars[CV_LAST_TEST_BAR]),
        output_dir=str(root / "runs"),
        rebalance_periods=2,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        benchmark_dataset=(
            make_stock_dataset(
                write_price_store(root / "benchmark", symbols=["QQQ"], seed=7, n_bars=CV_BARS)
            )
            if benchmark
            else None
        ),
    )
    return USEquityCrossectionSelectStockVectorBt(config).run_cv().run_dir


def normalized(text: str, roots: list[Path]) -> str:
    """``text`` with temporary directories, run names and plotly div ids replaced
    and decimal numbers rounded to 10 significant digits, below 1e-12 written 0."""
    for root in roots:
        text = text.replace(str(root), "<ROOT>")
    text = _UUID.sub("<UUID>", text)
    text = _RUN_NAME.sub(r"\1_<STAMP>", text)
    return _DECIMAL.sub(lambda match: _rounded(float(match.group())), text)


def _rounded(value: float) -> str:
    """``value`` to 10 significant digits; a round-off of zero is ``0``.

    The Holdings tab's cash share of a fully invested book is ``1 - sum`` of
    the weights: 0 on one CPU, ``1.1e-16`` or ``-2.2e-16`` on another.
    """
    return "0" if abs(value) < 1e-12 else format(value, ".10g")


def run_files(run_dir: Path, roots: list[Path]) -> dict[str, str]:
    """The normalized ``report.html`` and ``metrics.json`` of ``run_dir``."""
    return {
        name: normalized((run_dir / name).read_text(encoding="utf-8"), roots)
        for name in ("report.html", "metrics.json")
    }


def scenarios(tmp_path: Path, cv_project) -> dict[str, tuple[Path, list[Path]]]:
    """Run every scenario under ``tmp_path``: name -> (run dir, roots to hide)."""
    cv_root = cv_project[0]
    out = {}
    for name, run in (
        ("run_long_only_benchmark", _run_long_only_benchmark),
        ("run_long_short_delisting", _run_long_short_delisting),
        ("run_weights_benchmark", lambda root: _run_weights(root, benchmark=True)),
        ("run_weights", lambda root: _run_weights(root, benchmark=False)),
        ("run_cv_benchmark", lambda root: _run_cv(root, cv_project, benchmark=True)),
        ("run_cv", lambda root: _run_cv(root, cv_project, benchmark=False)),
    ):
        root = tmp_path / name
        root.mkdir()
        out[name] = (run(root), [root, cv_root])
    return out


#: SHA-256 of each normalized file, captured on the code before #115. #123
#: moved fold checkpoints from `{Cls}_cv_fold_{i}/` to `fold_{i}/`, which the
#: run_cv metrics.json record; its two hashes were re-captured then, after
#: checking that undoing just that path change gives the old hashes back.
#: All twelve were re-captured when decimals began to be rounded, after
#: checking that the same code at full precision gives the old hashes back
#: and that a Linux runner's output, rounded, gives the new ones.
#: Re-recorded when model runs gained the ``attribution`` block and the
#: Attribution tab; their metrics with that block removed hash as before.
#: The six reports were re-recorded when the Portfolio construction setup
#: row became a scrolling box (d0e4a0cd); removing that box's CSS rule and
#: wrapper gives the old hashes back. The metrics did not change.
#: The six reports were re-recorded for the dashboard page (#215): header,
#: sidebar sections, cards, chart styling, hover explanations and the plain
#: metric definitions. The metrics did not change.
#: The six reports were re-recorded for the Holdings tab (#225); the same
#: runs written without holdings give the #215 hashes back
#: (`WITHOUT_HOLDINGS`). The metrics did not change.
#: Re-recorded when Top-10 holding became the ten largest holdings by size
#: (shorts included) and the tab's explanation gained that and the cash
#: definition. The metrics did not change.
#: Re-recorded when the Holdings tab dropped its total return, annualised
#: return and max drawdown tiles, which repeated the headline cards.
#: Re-recorded when the Holdings table gained its path-in-period column.
#: Re-recorded when a short's target bar became light red beside its dark red holding bar.
#: Re-recorded when the target / holding header gained a key of the two bars,
#: long and short colours of each (which is which on hover only).
#: Re-recorded when each Holdings day embedded its summary (holdings count,
#: top-10 holding, new names since the previous rebalance) for the page to read
#: (#228). The metrics did not change.
#: Re-recorded when the path-in-period sparkline became each row's holding-period
#: return on the valuation prices (#231). The metrics did not change.
#: Re-recorded when the Holdings tab opened with the return by position size
#: (cut-off buttons, cumulative lines, size decile by year; net, gains or losses,
#: losing name-days). The metrics did not change.
#: Re-recorded when the cut-off buttons became a stepless slider over each bar's
#: ranked contributions. The metrics did not change.
#: The two run_cv runs were re-recorded when a fold became the stitched account
#: over its test segment (ADR 0033): their ``stitched`` blocks are unchanged, the
#: report differs in its run_cv note alone, and each fold's metrics are now cut
#: from the stitched curve.
#: Four reports were re-recorded when a round-off of zero (below 1e-12: a fully
#: invested book's cash share, a -0) was normalized to 0; their bytes did not change.
#: Every report and four metrics were re-recorded when the turnover and win-rate
#: rows took the decided rebalances of the run's rebalance record (#272): the
#: annualized turnover is the total over the window's years (5833.2% for 6066.5% on
#: the long-only and weights runs, 5 rebalances in 26 bars), the long-short run's
#: delisting settlement left its turnover (total 1034.8% for 1060.9%, per rebalance
#: 115.0% over 9 rebalances for 106.1% over 10 fill bars, annualized 10029.9% for
#: 8911.8%) and stopped splitting a holding period (rebalance win rate 44.4% for
#: 50%); the run_cv metrics did not change (12 rebalances every 4 bars filled
#: inside 48 bars, so both formulas agree), and the reports' hover texts state the
#: new definitions.
EXPECTED: dict[str, str] = {
    "run_long_only_benchmark/report.html": "07b54704ddf93aff3e5fa07147cfc0ac961cf2545118811584a501898d005f4f",
    "run_long_only_benchmark/metrics.json": "3cd2d405846f44ae7b3260989603f880e0b132e38138f159e44b3a5a2b33e2c4",
    "run_long_short_delisting/report.html": "a1421e7fc44684bcad16c8fc6d7ab8bd607ecacf9966466371bd87b22fefad2a",
    "run_long_short_delisting/metrics.json": "1fae26dd7c6691bd9725161988eabeb3856a9bf8889609bd04447ac48ab33d58",
    "run_weights_benchmark/report.html": "40c755f3d477d5de738a12ed498aaaf68db1a96285556ac752116de4d2ecb784",
    "run_weights_benchmark/metrics.json": "3f305680e5fdb3306df41c15e048c71094872eada67a1368bbd963ba31366dbd",
    "run_weights/report.html": "816972d0dcf921cee3d869d7e30e390a7a0dc0b5879f4f2e537a80e88aeb1f48",
    "run_weights/metrics.json": "b263a751733a6c054d03b557a86d6a9f69200a7d1cdcc8b13f598cc3f79874b4",
    "run_cv_benchmark/report.html": "c273e52dff3a4ff3ac8c99f3c22fab7136b6b0482c5bb2210a21d056393a566a",
    "run_cv_benchmark/metrics.json": "5eb6071ba06b6922deb7bf73aaf73915ce3d2c5400d1e0a5126136172eb5b538",
    "run_cv/report.html": "a2c27db8c8c8c9b20232909e1ae5a983ee994cd1450e990314d4b31930c60ac1",
    "run_cv/metrics.json": "c689af22c94db84dfcfbc21acf48c039ebfc734e79d5a132e9cfecac86a5bbf8",
}


#: The six reports as they were before the Holdings tab (#225), the hashes
#: recorded for #215. A report written without holdings must still be these
#: bytes; the two run_cv reports differ from them in their run_cv note alone
#: (ADR 0033).
WITHOUT_HOLDINGS: dict[str, str] = {
    "run_long_only_benchmark/report.html": "fb5a53725d966c9842b08e4ea5602bbee86d545b53ff531528e8054ad18ed343",
    "run_long_short_delisting/report.html": "cd3369137f5a5d0ba802c2d359d595cd48097f3e1435c0197e80c97e0cc9dad2",
    "run_weights_benchmark/report.html": "4ff31faeb0bf34241826527d22801654c11ee5d0e68e3d1dc6adae5d2ee5af3a",
    "run_weights/report.html": "488d2bad3e4edaf8c040c37b7583f5f542650c08adaff6927aecd75e64bb6976",
    "run_cv_benchmark/report.html": "4286a0dd94205bd8902e0fedb58498ae1212398a8a75e654e3bdca0332495b76",
    "run_cv/report.html": "eec70caa37d5913c073c5391e59df4a6e4149d48e38ade93cee779517276a845",
}


def _assert_hashes(tmp_path: Path, cv_project, expected: dict[str, str]) -> None:
    """Run every scenario and compare the normalized files named in ``expected``."""
    got = {}
    for name, (run_dir, roots) in scenarios(tmp_path, cv_project).items():
        for file, text in run_files(run_dir, roots).items():
            key = f"{name}/{file}"
            if key not in expected:
                continue
            got[key] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            (tmp_path / f"{name}.normalized.{file}").write_text(text, encoding="utf-8")
    changed = sorted(key for key in expected if got.get(key) != expected[key])
    assert not changed, (
        f"changed against the recorded bytes: {changed}; the normalized files are "
        f"under {tmp_path}. Recorded hashes now: {json.dumps(got, indent=1)}"
    )
    assert sorted(got) == sorted(expected)


def test_report_and_metrics_are_byte_identical_to_before_the_public_builders(
    tmp_path, cv_project
):
    _assert_hashes(tmp_path, cv_project, EXPECTED)


def test_without_holdings_the_report_is_byte_identical_to_before_the_holdings_tab(
    tmp_path, cv_project, monkeypatch
):
    """A caller with no holdings to show: ``write_backtest_report`` given none."""
    import quantlab.backtest.base as base

    write = base.write_backtest_report

    def without_holdings(*args, holdings=None, holding_names=None, holding_returns=None,
                         holding_contributions=None, **kwargs):
        write(*args, **kwargs)

    monkeypatch.setattr(base, "write_backtest_report", without_holdings)
    _assert_hashes(tmp_path, cv_project, WITHOUT_HOLDINGS)


def test_fold_rows_built_from_metrics_json_are_the_bars_each_fold_traded(tmp_path, cv_project):
    run_dir = _run_cv(tmp_path, cv_project, benchmark=False)
    run = BacktestRun.open(run_dir)
    folds = {fold.index: fold for fold in run.folds}
    for fold in run.metrics()["folds"]:
        whole = fold["metrics"]["whole"]
        traded = folds[fold["fold"]].equity().timestamp.values
        assert (bar_label(whole["Start"]), bar_label(whole["End"])) == (
            bar_label(traded[0]), bar_label(traded[-1])
        )
