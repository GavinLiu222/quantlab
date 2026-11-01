"""The live day checks the stores a run reads, as each dataset declares them (#283).

``quantlab.backtest.live.input_stores`` answers "which stores must hold the bar
the live day predicts". It used to guess leaves from the component tree (a
dataset with no dataset under it), which misjudged a ``MappedTradeBarDataset``:
it holds its source view to write its store, not to read it, so the Trade bar
store under the view was checked instead of the mapped store. The stores now
come from the dataset's own declaration (``BaseDataset.stores``).
"""

from __future__ import annotations

import os
from types import SimpleNamespace


from quantlab.backtest.live import input_stores
from quantlab.dataset.config import MappedTradeBarsConfig
from quantlab.dataset.massive.mapped_bars import MappedTradeBarDataset
from quantlab.dataset.roster import RosterDataset
from quantlab.dataset.stock import StockDataset
from tests.backtest_fixtures import write_price_store
from tests.test_massive_permaticker_bars import built  # noqa: F401  (fixture)


def _backtester(price_dataset) -> SimpleNamespace:
    """What ``input_stores`` reads of a backtester: its config's input fields."""
    return SimpleNamespace(config=SimpleNamespace(price_dataset=price_dataset, model=None, constructor=None))


def test_a_mapped_trade_bar_dataset_reads_its_own_store(built, tmp_path):
    path = str(tmp_path / "mapped.zarr")
    mapped = MappedTradeBarDataset(MappedTradeBarsConfig(source=built[2], zarr_file_path=path)).update()

    stores = input_stores(_backtester(mapped))

    assert list(stores) == [os.path.abspath(path)]
    component, leaf = stores[os.path.abspath(path)]
    assert component == "price_dataset" and isinstance(leaf, MappedTradeBarDataset)


def test_a_view_is_keyed_by_its_inputs_stores(tmp_path):
    prices = write_price_store(tmp_path / "prices", n_bars=10)
    roster = write_price_store(tmp_path / "roster", n_bars=10)
    view = RosterDataset(StockDataset(prices), StockDataset(roster))

    stores = input_stores(_backtester(view))

    assert list(stores) == [os.path.abspath(prices.zarr_file_path), os.path.abspath(roster.zarr_file_path)]
    assert stores[os.path.abspath(prices.zarr_file_path)][1] == StockDataset(prices)
