"""Predict a backtest run's strategy on the last closed bar and append it to the live store.

The daily live prediction job (#233, quantlab-ibkr #47). Run it every
morning after the vendor update has appended the last closed bar t to the
run's price store (``scripts/sharadar/update.py`` for the Sharadar S&P 500
strategy). One run, through ``quantlab.backtest.live.predict_live_bar``:

1. rebuilds the run's backtester from its ``config.json`` and takes t, the
   last bar of its price dataset;
2. refuses, appending nothing, when t is already in the live store or when a
   store the strategy reads (each leaf dataset under the price dataset, the
   model and the portfolio rule; the model's labels are not read) lacks t;
3. brings the run's own stores to t: each ``--mirror`` store gains the price
   store's new bars, each factor store is extended (``Factor.extend``) and
   each factor risk model's regression and estimate stores
   (``RiskStore.extend``), then checks that each holds t;
4. loads the run's checkpoint into its model (the membership-masked
   predictor), predicts ``predict_window(t, t)`` under a ``DataRecorder`` and
   appends the row to the live store with its record.

The live store (``quantlab.runs.live_predictions.LivePredictionStore``) is a
Zarr store on ``(timestamp, symbol)``: one float variable per label of the
run's prediction panel (``ret_5`` for the Barra MVO run), NaN for a symbol
without a prediction (not an index member on t); its attrs are the
prediction panel's ``format_version`` and ``labels`` plus
``live_format_version``, ``run_dir``, ``checkpoint`` and ``trained_run``.
Each row's record (``timestamp``, ``written_at``, ``run_dir``,
``checkpoint``, ``data_fingerprint``, ``stores``, ``lagging``) is in
``<store>.rows.json`` under the bar's date. See docs/live.md.

Exit status: 0 when a row was appended; 3 when t is already predicted (the
day is done); 2 when an input lacks t (retry after the vendor update, or
hold the day); 1 for any other refusal (another run's store, a bad
argument). An unexpected error exits 1 with its traceback.

Usage on the training server, after ``scripts/sharadar/update.py``::

    QUANTLAB_DATA_DIR=/data/quantlab taskset -c 64-127 \\
      .venv/bin/python scripts/live/predict_day.py \\
        /data/quantlab/runs/ibkr_barra_closed_loop/real/backtest/<run> \\
        --store /data/quantlab/live/sp500_xgb_mvo/live_predictions.zarr \\
        --may-lag /data/quantlab/market/fred/fred_dtb3_1d/fred_dtb3_1d.zarr
"""

import os
import sys

# macOS only: xgboost and torch ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded. Set before either imports.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

from quantlab.backtest.live import LivePredictionRefused, predict_live_bar

#: Exit status of each refusal reason; any other refusal exits 1.
EXIT_STATUS = {"already_predicted": 3, "missing_data": 2}


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Extend a backtest run's stores to the last bar of its price store and "
            "append the model's prediction of that bar to a live prediction store. "
            "Exit 0 appended, 3 already predicted, 2 an input lacks the bar, 1 otherwise."
        )
    )
    parser.add_argument("run_dir", type=Path, help="The backtest run directory (the recipe).")
    parser.add_argument(
        "--store", type=Path, required=True,
        help="The live prediction store; created by the first run.",
    )
    parser.add_argument(
        "--mirror", type=Path, action="append", default=[],
        help=(
            "A store of the run that copies variables of its price store (the "
            "example's prices.zarr); it gains the price store's new bars. Repeatable."
        ),
    )
    parser.add_argument(
        "--may-lag", type=Path, action="append", default=[],
        help=(
            "A store of the run allowed to end before the bar (a rate published the "
            "next business day, read lagged). Repeatable."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    try:
        done = predict_live_bar(
            args.run_dir, args.store, mirrors=args.mirror, may_lag=args.may_lag
        )
    except LivePredictionRefused as refused:
        print(f"refused ({refused.reason}): {refused}", file=sys.stderr)
        sys.exit(EXIT_STATUS.get(refused.reason, 1))
    variable = next(iter(done.row.data_vars))
    print(json.dumps({
        "timestamp": done.timestamp.isoformat(),
        "store": str(done.store.path),
        "symbols_predicted": int(done.row[variable].notnull().sum()),
        "stores": done.record["stores"],
        "lagging": done.record["lagging"],
    }, indent=2))
