"""``CollectedPanel.assemble``: the features decide the collected panel's axes.

A model reads its features and its labels apart and joins them into one
panel. The join used to be an outer one on both axes, so a label read on a
wider symbol axis than the features (a market-wide label under features read
on a roster) widened every feature to it: on us3000 326 features grew from
10,699 to 17,072 symbols, and with two unconditional sorts copying the panel
the collection peaked at 422 GB and had to be stopped. Now the panel is the
features' bars and symbols: the labels are reindexed onto them (a label's
symbol or bar without features is dropped, a feature's warm-up bar has NaN
labels), the features are sorted only when they are not, so already sorted
features are not copied, and labels sharing no symbol with the features are
refused. ``BaseModel.collect`` assembles its panel this way.

Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.model.collected_panel import CollectedPanel
from tests.test_library_models import FakePanel, StubLabel, StubLibraryHead, _config

BARS = pd.date_range("2024-01-01", periods=6)


def _panel(name, bars, symbols, value=1.0) -> xr.Dataset:
    data = np.full((len(bars), len(symbols)), value, dtype=np.float32)
    return xr.Dataset({name: (("timestamp", "symbol"), data)}, coords={"timestamp": bars, "symbol": symbols})


def test_the_panel_is_on_the_features_bars_and_symbols():
    features = _panel("f", BARS, ["A", "B"])
    # The label is wider on symbols (C), starts after the features' warm-up and ends a bar later.
    labels = _panel("ret", pd.date_range("2024-01-03", periods=5), ["A", "B", "C"], value=2.0)

    data = CollectedPanel.assemble(features, labels).data

    assert data["symbol"].values.tolist() == ["A", "B"]
    assert data["timestamp"].values.tolist() == BARS.values.tolist()
    assert np.isnan(data["ret"].sel(timestamp=BARS[:2]).values).all()
    assert (data["ret"].sel(timestamp=BARS[2:]).values == 2.0).all()


def test_sorted_features_are_not_copied():
    features = _panel("f", BARS, ["A", "B"])

    data = CollectedPanel.assemble(features, _panel("ret", BARS, ["B", "A", "C"])).data

    assert np.shares_memory(data["f"].values, features["f"].values)
    assert data["ret"].dims == ("timestamp", "symbol")


def test_unsorted_features_are_sorted():
    features = _panel("f", BARS[::-1], ["B", "A"])

    data = CollectedPanel.assemble(features, _panel("ret", BARS, ["A", "B"])).data

    assert data.indexes["timestamp"].is_monotonic_increasing
    assert data["symbol"].values.tolist() == ["A", "B"]


def test_labels_sharing_no_symbol_with_the_features_are_refused():
    labels = xr.Dataset({"ret": (("timestamp", "symbol"), np.ones((6, 2)))},
                        coords={"timestamp": BARS, "symbol": [1, 2]})

    with pytest.raises(ValueError, match="none of the 2 label symbol"):
        CollectedPanel.assemble(_panel("f", BARS, ["1", "2"]), labels)


def test_the_source_is_kept():
    panel = CollectedPanel.assemble(_panel("f", BARS, ["A"]), _panel("ret", BARS, ["A"]), source={"k": 1})

    assert panel.source == {"k": 1} and panel.record == {}


class WideLabel(FakePanel):
    """A label panel on more symbols than the features, as a market-wide label under roster features."""

    def __init__(self, names, seed=0):
        super().__init__(names, seed=seed)
        extra = self._ds.rename(symbol="s").assign_coords(s=[f"X{i}" for i in range(self._ds.sizes["symbol"])])
        self._ds = xr.concat([self._ds, extra.rename(s="symbol")], dim="symbol")


def test_a_model_collects_on_the_features_symbols(tmp_path):
    model = StubLibraryHead(_config(tmp_path, labels=[WideLabel(["ret_a"], seed=2)])).collect()

    features = FakePanel(["f_a", "f_b"], seed=1).compute(None, None)
    assert model.collected_panel.data["symbol"].values.tolist() == features["symbol"].values.tolist()
