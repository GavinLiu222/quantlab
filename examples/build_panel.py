"""Build, store, filter and mask a (timestamp, symbol) panel, fully offline.

This example walks through the dataset layer of quantlab on a few weeks of
synthetic daily US-equity bars:

1. write a small raw tier in the hive-partitioned parquet layout the Tiingo
   and Alpaca downloaders produce;
2. convert it into a panel with ``StockDataset`` and save it as Zarr;
3. read the panel back, narrowed to a date range and a symbol subset;
4. move the panel through the two storage backends, ``XrBackend`` (Zarr)
   and ``PlBackend`` (Parquet), and use the ``get_xarray_dataset`` shape
   contract;
5. convert the same raw tier window by window with the chunked, resumable
   path, then bring the store up to date after a new month (and a new
   listing) arrives;
6. restrict the panel to a point-in-time index membership with
   ``UniverseMask``;
7. query a point-in-time universe catalog.

Everything runs in a temporary directory: no network, no credentials, CPU
only. Run it from the repository root with::

    uv run python examples/build_panel.py
"""

import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from quantlab.backend.parquet import PlBackend
from quantlab.backend.zarr import XrBackend
from quantlab.dataset.config import ConstituentDatasetConfig, DatasetConfig
from quantlab.universe import UniverseConfig
from quantlab.dataset.raw import IndexConstituentDataset
from quantlab.dataset._support.masking import UniverseMask
from quantlab.dataset.stock import StockDataset
from quantlab.universe import UniverseCatalog

# quantlab logs through loguru at INFO level. Keep only warnings so the
# printed results stay readable.
logger.remove()
logger.add(sys.stderr, level="WARNING", format="{level}: {message}")
# zarr warns on every write that consolidated metadata is not part of the
# Zarr v3 specification; it is harmless here.
warnings.filterwarnings("ignore", message="Consolidated metadata")


def raw_rows(symbol: str, days: pd.DatetimeIndex, close: np.ndarray, volume: float):
    """Return raw rows with the column set the Tiingo downloader writes."""
    rows = []
    for day, price in zip(days, close):
        rows.append(
            {
                "timestamp": datetime(day.year, day.month, day.day),
                "symbol": symbol,
                "open": float(price),
                "high": float(price) * 1.01,
                "low": float(price) * 0.99,
                "close": float(price),
                "volume": float(volume),
                "adjOpen": float(price),
                "adjHigh": float(price) * 1.01,
                "adjLow": float(price) * 0.99,
                "adjClose": float(price),
                "adjVolume": float(volume),
                "divCash": 0.0,
                "splitFactor": 1.0,
            }
        )
    return rows


def write_raw_tier(vendor_root: Path, rows: list[dict], batch: str) -> None:
    """Write rows as one shard per month under ``month=YYYY-MM`` directories.

    Every raw shard carries a literal ``vendor`` column, which the dataset
    checks against the configured vendor before using any row.
    """
    frame = pl.DataFrame(rows).with_columns(pl.lit("tiingo").alias("vendor"))
    frame = frame.with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m").alias("_month")
    )
    for (month,), part in frame.group_by(["_month"]):
        part_dir = vendor_root / f"month={month}"
        part_dir.mkdir(parents=True, exist_ok=True)
        part.drop("_month").write_parquet(part_dir / f"part-{batch}-00000.pqt")


def main() -> None:
    """Run the panel-building walkthrough in a temporary directory."""
    root = Path(tempfile.mkdtemp(prefix="quantlab_build_panel_"))
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # 1. A raw tier: January and February 2024, four symbols.
    #    DDD delists at the end of January; PNY is a penny stock; BBB
    #    prints one bad (zero) close that the cleaning step will flag.
    # ------------------------------------------------------------------
    jan_feb = pd.bdate_range("2024-01-02", "2024-02-29")
    rows = []
    for symbol, start_price, volume in [
        ("AAA", 50.0, 200_000.0),
        ("BBB", 20.0, 150_000.0),
        ("PNY", 1.5, 300_000.0),
    ]:
        path = start_price * np.cumprod(1 + rng.normal(0, 0.01, len(jan_feb)))
        if symbol == "BBB":
            path[10] = 0.0
        rows += raw_rows(symbol, jan_feb, path, volume)
    ddd_days = jan_feb[jan_feb <= "2024-01-31"]
    rows += raw_rows("DDD", ddd_days, np.linspace(30.0, 25.0, len(ddd_days)), 80_000.0)

    vendor_root = root / "downloads" / "us_equity" / "1d" / "demo" / "tiingo"
    write_raw_tier(vendor_root, rows, batch="batch0001")
    print("1. raw shards:", sorted(p.parent.name for p in vendor_root.rglob("*.pqt")))

    # ------------------------------------------------------------------
    # 2. Convert the raw tier into a dense panel and save it as Zarr.
    # ------------------------------------------------------------------
    def stock_config(**overrides) -> DatasetConfig:
        """Return a dataset config for the synthetic store, with fields overridden by keyword."""
        fields = dict(
            raw_data_dir_path=str(vendor_root),
            zarr_file_path=str(root / "data" / "us_equity" / "1d" / "demo.zarr"),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
        )
        fields.update(overrides)
        return DatasetConfig(**fields)

    dataset = StockDataset(stock_config()).from_raw_data()
    dataset.save()
    panel = dataset.get_xarray_dataset()
    print("\n2. full panel:", dict(panel.sizes))
    print("   variables:", list(panel.data_vars))
    print("   symbols:", dataset.symbols)
    print("   bar spacing:", pd.Timedelta(dataset.time_interval))
    flagged = panel["anomaly_flag"].to_pandas()
    print("   anomaly_flag cells:", flagged[flagged.any(axis=1)].stack().loc[lambda s: s].index.tolist())

    # ------------------------------------------------------------------
    # 3. Request a date range and a symbol subset from the store.
    # ------------------------------------------------------------------
    narrow = StockDataset(stock_config()).panel(
        "2024-01-29", "2024-02-02", symbols=["AAA", "DDD"]
    )
    print("\n3. requested panel:", dict(narrow.sizes))
    print(narrow["close"].to_pandas().round(2))

    # ------------------------------------------------------------------
    # 4. Storage backends and the get_xarray_dataset(indexes) contract.
    # ------------------------------------------------------------------
    store = root / "scratch" / "copy.zarr"
    XrBackend().to_internal(panel).write(str(store))
    backend = XrBackend().read(str(store))
    print("\n4. XrBackend dims, (timestamp, symbol):",
          backend.get_xarray_dataset(["timestamp", "symbol"])["close"].dims)
    print("   XrBackend dims, (symbol, timestamp):",
          backend.get_xarray_dataset(["symbol", "timestamp"])["close"].dims)
    time_only = backend.get_xarray_dataset(["timestamp"])
    print("   indexes=['timestamp'] keeps", dict(time_only.sizes), "and variables", list(time_only.data_vars))

    table = root / "scratch" / "close.parquet"
    long_frame = (
        backend.get_lazyframe().select("timestamp", "symbol", "close", "volume")
    )
    PlBackend().to_internal(long_frame).write(str(table))
    parquet = PlBackend().read(str(table))
    parquet.filter_by_date("timestamp", "2024-02-01", "2024-02-02")
    parquet.filter_by_symbol("symbol", ("AAA", "BBB"))
    as_panel = parquet.get_xarray_dataset(["timestamp", "symbol"])
    print("   PlBackend round trip:", dict(as_panel.sizes), list(as_panel.data_vars))

    # ------------------------------------------------------------------
    # 5. Chunked, resumable conversion, then an update after new data.
    # ------------------------------------------------------------------
    chunked_config = stock_config(
        zarr_file_path=str(root / "data" / "us_equity" / "1d" / "chunked.zarr")
    )
    result = StockDataset(chunked_config).from_raw_data_chunked(granularity="month").last_chunk_result
    print("\n5. first chunked run: planned", result.windows_planned,
          "written", result.windows_written, "skipped", result.windows_skipped,
          "rows", result.rows_written)
    result = StockDataset(chunked_config).from_raw_data_chunked(granularity="month").last_chunk_result
    print("   second run (nothing new): written", result.windows_written,
          "skipped", result.windows_skipped, "resumed", result.resumed)

    # March arrives, and with it a new listing, EEE.
    march = pd.bdate_range("2024-03-01", "2024-03-29")
    new_rows = []
    for symbol, start_price, volume in [
        ("AAA", 52.0, 200_000.0),
        ("BBB", 21.0, 150_000.0),
        ("PNY", 1.4, 300_000.0),
        ("EEE", 40.0, 120_000.0),
    ]:
        path = start_price * np.cumprod(1 + rng.normal(0, 0.01, len(march)))
        new_rows += raw_rows(symbol, march, path, volume)
    write_raw_tier(vendor_root, new_rows, batch="batch0002")

    updated = StockDataset(chunked_config).update(granularity="month")
    result = updated.last_chunk_result
    print("   update after March: written", result.windows_written,
          "skipped", result.windows_skipped)
    store_panel = StockDataset(chunked_config).panel("2024-01-01", "2024-03-29")
    print("   store now:", dict(store_panel.sizes), "symbols", store_panel["symbol"].values.tolist())
    print("   EEE closes observed before March:",
          int(store_panel["close"].sel(symbol="EEE", timestamp=slice(None, "2024-02-29")).count()))

    # ------------------------------------------------------------------
    # 6. Point-in-time index membership, applied with UniverseMask.
    # ------------------------------------------------------------------
    class DemoIndex(IndexConstituentDataset):
        """A toy index: AAA always, BBB until Feb 15, DDD in January.

        ZZZ was a member for three weeks but was never downloaded, the
        typical coverage gap that brings survivorship bias back.
        """

        def _pit_coverage_start(self) -> str:
            """Return the first date for which the toy membership data is complete."""
            return "2024-01-01"

        def _build_intervals(self) -> pl.DataFrame:
            """Return the toy membership intervals as ``(symbol, start, end)`` rows."""
            return pl.DataFrame(
                [
                    ("AAA", "2024-01-01", None),
                    ("BBB", "2024-01-01", "2024-02-15"),
                    ("DDD", "2024-01-01", "2024-01-31"),
                    ("ZZZ", "2024-01-01", "2024-01-19"),
                ],
                schema=["symbol", "start_date", "end_date"],
                orient="row",
            )

    index_config = ConstituentDatasetConfig(
        zarr_file_path=str(root / "data" / "reference" / "demo_index.zarr"),
        cache_dir=str(root / "data" / "reference" / "_cache"),
        start_date="2024-01-01",
        end_date="2024-03-29",
        as_of="2024-03-29",
    )
    DemoIndex(index_config).from_raw_data().save()
    membership = DemoIndex(index_config).panel("2024-01-01", "2024-03-29")
    print("\n6. membership panel:", dict(membership.sizes), "(calendar days)")

    mask = UniverseMask(store_panel, membership)
    print("   report:", mask.report())
    masked = mask.apply()
    print("   masked close, first trading day of each month:")
    print(masked["close"].sel(timestamp=["2024-01-02", "2024-02-01", "2024-03-01"]).to_pandas().round(2))

    # ------------------------------------------------------------------
    # 7. A point-in-time universe catalog, queried two ways.
    # ------------------------------------------------------------------
    catalog_config = UniverseConfig(
        output_path=str(root / "data" / "reference" / "universe.parquet"),
        cache_dir=str(root / "data" / "reference" / "_cache"),
    )
    pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "DDD", "EEE"],
            "category": ["us_all"] * 4,
            "start_date": ["2000-01-03", "2010-06-01", "1995-05-01", "2024-03-01"],
            "end_date": [None, None, "2024-01-31", None],
            "end_date_is_inferred": [False] * 4,
        }
    ).write_parquet(catalog_config.output_path)
    catalog = UniverseCatalog.load(catalog_config)
    print("\n7. listed on 2024-03-15:", catalog.get_symbols_as_of("us_all", "2024-03-15"))
    print("   listed at any time in Q1 2024:",
          catalog.get_symbols_in_range("us_all", "2024-01-01", "2024-03-31"))

    print(f"\nAll files were written under {root}")


if __name__ == "__main__":
    main()
