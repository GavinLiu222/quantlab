"""List quantlab's data sources and inspect a small on-disk download, offline.

This example needs no API key, no WRDS account and no network. It shows:

1. every registered data source, what it serves and whether its credentials
   are set in this environment (as booleans; no value is ever printed);
2. a download driven through ``quantlab.acquisition.registry.download_symbols`` with a progress
   callback, a resumed second run, and a run stopped with a ``CancelToken``.
   A real vendor needs credentials and network access, so the download uses
   a small stand-in acquisition class that invents prices locally and writes
   them in the Tiingo raw layout;
3. read-only inspection of the files on disk with ``SourceInspector``;
4. conversion of the raw files into a Zarr panel with ``quantlab.acquisition.registry.convert``.

Everything is written to a temporary directory that is deleted at the end.

Run it with::

    uv run python examples/inspect_data_sources.py
"""

import datetime
import sys
import tempfile
import warnings
from pathlib import Path

import polars as pl
from loguru import logger

from quantlab.acquisition._support.inspector import SourceInspector
from quantlab.acquisition.tiingo import TiingoAcquisition
from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition
from quantlab.utils.progress import CallbackProgressReporter, CancelToken
from quantlab.acquisition.config import stock_acquisition_config
from quantlab.config import set_data_root, stock_kline_config
from quantlab.dataset.stock import StockDataset
from quantlab.acquisition.base import Capability, DataSourceRegistry, SourceDescriptor
from quantlab.acquisition.registry import convert, credential_status, download_symbols

# Keep the output readable: show only warnings from quantlab's logger, and
# silence a Zarr notice about consolidated metadata that is not relevant here.
logger.remove()
logger.add(sys.stderr, level="WARNING", format="{level}: {message}")
warnings.filterwarnings("ignore", message="Consolidated metadata")


def section(title: str) -> None:
    """Print a section heading."""
    print(f"\n== {title} ==")


# ---------------------------------------------------------------------------
# 1. The registry: which sources exist and what each one serves.
# ---------------------------------------------------------------------------
section("Registered data sources")
for source in DataSourceRegistry.all():
    print(f"{source.vendor}: {source.display_name}")
    for cap in source.capabilities:
        converter = cap.dataset_cls.__name__ if cap.dataset_cls else "(raw only)"
        print(
            f"    market={cap.market} frequency={cap.frequency} "
            f"data_type={cap.data_type} -> {converter}"
        )
    # credential_status reports whether each variable is set, never its value.
    print(f"    credentials: {credential_status(source)}")

tiingo = DataSourceRegistry.get("tiingo")
print("Tiingo serves tick data:", tiingo.supports("us_equity", "tick"))


# ---------------------------------------------------------------------------
# 2. A download through download_symbols(), using a local stand-in for a vendor.
# ---------------------------------------------------------------------------
class SyntheticDailyBars(SymbolBatchAcquisition):
    """Stand-in for a vendor client: invents weekday prices, one page per batch.

    It reuses Tiingo's column layout, so the files it writes are exactly what
    ``StockDataset`` expects to convert. A real client would call the
    vendor's API inside ``_fetch_page`` instead.
    """

    VENDOR = "tiingo"
    RAW_COLUMNS = TiingoAcquisition.RAW_COLUMNS

    def _fetch_page(self, symbols, start_date, end_date, page_token=None):
        """Generate one page of random daily bars instead of calling a vendor API."""
        start = datetime.date.fromisoformat(start_date[:10])
        end = datetime.date.fromisoformat(end_date[:10])
        days = pl.date_range(start, end, eager=True)
        days = days.filter(days.dt.weekday() <= 5).cast(pl.Datetime("us"))
        frames = []
        for symbol in symbols:
            # A different, deterministic price level per symbol.
            level = 50.0 + sum(map(ord, symbol)) % 100
            prices = [level + k for k in range(len(days))]
            frame = pl.DataFrame({"timestamp": days}).with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.lit(self.VENDOR).alias("vendor"),
                *(pl.Series(name, prices) for name in (
                    "open", "high", "low", "close",
                    "adjOpen", "adjHigh", "adjLow", "adjClose",
                )),
                pl.lit(1e6).alias("volume"),
                pl.lit(1e6).alias("adjVolume"),
                pl.lit(0.0).alias("divCash"),
                pl.lit(1.0).alias("splitFactor"),
            )
            frames.append(frame.select(self.RAW_COLUMNS))
        # No next-page token: the whole window fits on one page.
        return pl.concat(frames), None


# An unregistered descriptor pointing at the stand-in. download_symbols() only needs a
# descriptor to find the acquisition class; it never looks the vendor up.
demo_source = SourceDescriptor(
    vendor="tiingo",
    display_name="Synthetic daily bars",
    symbol_batch_cls=SyntheticDailyBars,
    config_factory=stock_acquisition_config,
    capabilities=(Capability(market="us_equity", frequency="1d"),),
    required_env=(),
)

with tempfile.TemporaryDirectory() as tmp:
    # Every config factory derives its paths from the data root, so point it
    # at the temporary directory before building any config.
    set_data_root(tmp)
    root = Path(tmp)

    def show(path) -> str:
        """Print paths relative to the temporary root."""
        return str(path).replace(tmp, "<root>")

    # progress=False turns off the default terminal progress bar; one worker
    # keeps the event order deterministic for this demo.
    knobs = {"progress": False, "max_workers": 1}
    acq_config = stock_acquisition_config(
        symbols=("AAPL", "MSFT", "NVDA"),
        start_date="2024-01-01",
        end_date="2024-03-29",
        kwargs=knobs,
    )

    section("First download")
    events = []
    result = download_symbols(demo_source, acq_config, reporter=CallbackProgressReporter(events.append))
    for event in events:
        print(f"    event {event.kind:<16} {event.completed}/{event.total}")
    print("succeeded:", result.succeeded, "failures:", result.failures)

    section("Second run of the same config (resumes, fetches nothing)")
    result = download_symbols(demo_source, acq_config)
    print("coverage:", result.coverage)

    section("A run cancelled after its first batch")
    wider = stock_acquisition_config(
        symbols=("AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM"),
        start_date="2024-01-01",
        end_date="2024-03-29",
        kwargs=knobs,
    )
    token = CancelToken()

    def stop_after_first_batch(event):
        """Set the cancel token as soon as one batch has landed."""
        if event.kind == "batch_completed":
            token.cancel()

    result = download_symbols(
        demo_source,
        wider,
        reporter=CallbackProgressReporter(stop_after_first_batch),
        cancel=token,
    )
    print("cancelled:", result.cancelled, "newly completed:", result.succeeded)

    # ------------------------------------------------------------------
    # 3. Inspect what is on disk. No client is built and no key is needed.
    # ------------------------------------------------------------------
    inspector = SourceInspector()
    section("Coverage of the wider roster")
    print(inspector.coverage(wider))

    section("Raw inventory")
    raw = inspector.inventory(wider)["raw"]
    for key in ("root", "shards", "symbols_with_watermark",
                "coverage_start", "coverage_last_date", "failures"):
        print(f"    {key}: {show(raw[key])}")

    section("Files under the data root")
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            files = [p for p in path.iterdir() if p.is_file()]
            if files:
                print(f"    {show(path)}/  ({len(files)} files)")

    ds_config = stock_kline_config(
        start_date="2024-01-01",
        end_date="2024-03-29",
        symbols=("AAPL", "AMD", "MSFT", "NVDA"),
    )
    section("A narrow look at the raw rows")
    print(inspector.browse_raw(ds_config, ["AAPL"], "2024-01-01", "2024-01-04").collect()
          .select("timestamp", "symbol", "close", "volume"))

    # ------------------------------------------------------------------
    # 4. Convert the raw tier to a Zarr panel. Offline: convert() never
    #    builds a vendor client, so the Tiingo descriptor works without a key.
    # ------------------------------------------------------------------
    section("Convert raw files to a Zarr panel")
    conversion = convert(tiingo, ds_config, granularity="month")
    print(f"    windows written: {conversion.windows_written}/{conversion.windows_planned}, "
          f"rows: {conversion.rows_written}, store: {show(conversion.zarr_path)}")
    zarr_info = inspector.inventory(wider, ds_config)["zarr"]
    print("    dims:", zarr_info["dims"])
    panel = inspector.browse_zarr(ds_config, ["AAPL", "MSFT"], "2024-01-02", "2024-01-04")
    print(panel["close"].to_pandas())
