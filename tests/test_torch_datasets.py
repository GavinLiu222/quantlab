"""The shipped torch datasets, tested directly on hand-built training panels.

`CrossSectionDataset` is the default `TorchModel._dataset`: one item per bar,
every present symbol with its last N bars. `SymbolSequenceDataset` is the
Qlib-style alternative: one item per `(bar, symbol)` cell, the symbol's last
N bars as `[N, F]`. What turns this file red:

- an item's `x`, `mask`, `y`, `y_raw` or `where` has the wrong shape or
  points at the wrong cells;
- evaluation misses a present cell, or training includes a bar or a cell
  without a valid target;
- a window row before the panel's first bar is not NaN;
- batched fetching disagrees with fetching items one by one, or default
  collation does not give `[B, N, F]`;
- a sample for bar t changes when features after t change (lookahead), for
  either dataset.

Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import pytest
import torch

from torch.utils.data import DataLoader

from quantlab.model.torch_data import (
    Batch,
    CrossSectionDataset,
    SymbolSequenceDataset,
    TrainingPanel,
)

T, S, F, L = 6, 4, 2, 1


def _panel(seed=0) -> TrainingPanel:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, S, F)).astype("float32")
    x[:2, 3] = np.nan  # S3 absent on bars 0 and 1
    x[4, :] = np.nan  # bar 4 has no present symbol
    y = rng.standard_normal((T, S, L)).astype("float32")
    panel = TrainingPanel.from_arrays(
        x, timestamps=np.arange(T), symbols=np.array(["A", "B", "C", "D"]), y_raw=y
    )
    panel.mask[:, :2] = panel.present[:, :2]  # only A and B have a valid target
    panel.mask[3] = False  # bar 3 has none
    panel.target[panel.mask] = panel.y_raw[panel.mask]
    return panel


def test_panel_marks_present_cells_and_starts_with_no_target():
    panel = TrainingPanel.from_arrays(
        np.array([[[1.0], [np.nan]]]), timestamps=np.arange(1), symbols=np.arange(2),
        y_raw=np.array([[[0.5], [0.2]]]),
    )
    assert panel.present.tolist() == [[True, False]]
    assert not panel.mask.any() and (panel.target == 0).all()


def test_an_item_is_one_bar_with_every_present_symbol_and_its_where():
    panel = _panel()
    dataset = CrossSectionDataset(panel, bars=[1, 2], window_bars=3, training=False)
    item = dataset[1]  # bar 2: all four symbols present

    assert isinstance(item, Batch)
    assert item.x.shape == (4, 3, F)
    assert item.mask.shape == (4,) and item.y.shape == item.y_raw.shape == (4, L)
    assert item.where[0].tolist() == [2] * 4 and item.where[1].tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(item.x[:, -1], panel.x[2])
    torch.testing.assert_close(item.x[:, 0], panel.x[0], equal_nan=True)
    assert item.mask.tolist() == [True, True, False, False]


def test_window_rows_before_the_panel_are_nan():
    item = CrossSectionDataset(_panel(), bars=[0], window_bars=3, training=False)[0]
    assert torch.isnan(item.x[:, :2]).all() and torch.isfinite(item.x[:, 2]).all()
    assert item.where[1].tolist() == [0, 1, 2]  # S3 is absent on bar 0


def test_evaluation_covers_every_present_cell_and_training_only_valid_targets():
    panel = _panel()
    evaluation = CrossSectionDataset(panel, bars=range(T), window_bars=2, training=False)
    covered = {
        (int(t), int(s)) for item in evaluation for t, s in zip(*item.where)
    }
    assert covered == {tuple(c) for c in torch.nonzero(panel.present).tolist()}
    assert [int(item.where[0][0]) for item in evaluation] == [0, 1, 2, 3, 5]  # bar 4 empty

    training = CrossSectionDataset(panel, bars=range(T), window_bars=2, training=True)
    assert [int(item.where[0][0]) for item in training] == [0, 1, 2, 5]  # bar 3 has no target
    assert all(item.mask.any() for item in training)


@pytest.mark.parametrize("dataset_cls", [CrossSectionDataset, SymbolSequenceDataset])
def test_changing_features_after_a_bar_never_changes_its_sample(dataset_cls):
    before = _panel()
    after = _panel()
    after.x[3:] += 100.0  # everything from bar 3 on
    after.x[3:, 2] = float("nan")  # and C vanishes after bar 2
    for training in (True, False):
        old = dataset_cls(before, bars=[0, 1, 2], window_bars=3, training=training)
        new = dataset_cls(after, bars=[0, 1, 2], window_bars=3, training=training)
        assert len(old) == len(new)
        for a, b in zip(old, new):
            torch.testing.assert_close(a.x, b.x, equal_nan=True)
            torch.testing.assert_close(a.y, b.y, equal_nan=True)
            torch.testing.assert_close(a.y_raw, b.y_raw, equal_nan=True)
            assert torch.equal(a.mask, b.mask)
            assert torch.equal(a.where[0], b.where[0])
            assert torch.equal(a.where[1], b.where[1])


@pytest.mark.parametrize("dataset_cls", [CrossSectionDataset, SymbolSequenceDataset])
def test_window_bars_below_one_is_refused(dataset_cls):
    with pytest.raises(ValueError, match="window_bars"):
        dataset_cls(_panel(), bars=[0], window_bars=0, training=False)


# ---------------------------------------------------------------------------
# SymbolSequenceDataset: one item per (bar, symbol) cell
# ---------------------------------------------------------------------------


def test_a_sequence_item_is_one_cell_with_its_window_and_where():
    panel = _panel()
    dataset = SymbolSequenceDataset(panel, bars=[2], window_bars=3, training=False)
    assert len(dataset) == 4  # bar 2: all four symbols present
    item = dataset[2]  # (bar 2, symbol C)

    assert isinstance(item, Batch)
    assert item.x.shape == (3, F)
    assert item.mask.shape == () and item.y.shape == item.y_raw.shape == (L,)
    assert int(item.where[0]) == 2 and int(item.where[1]) == 2
    torch.testing.assert_close(item.x, panel.x[0:3, 2])
    assert not bool(item.mask) and torch.equal(item.y_raw, panel.y_raw[2, 2])
    first = dataset[0]  # (bar 2, symbol A), a valid target
    assert bool(first.mask) and torch.equal(first.y, panel.target[2, 0])


def test_a_short_history_leaves_nan_rows_at_the_window_start():
    panel = _panel()
    dataset = SymbolSequenceDataset(panel, bars=[0, 1], window_bars=3, training=False)
    by_cell = {(int(i.where[0]), int(i.where[1])): i for i in dataset}
    assert set(by_cell) == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}  # no D

    at_0, at_1 = by_cell[(0, 1)].x, by_cell[(1, 1)].x
    assert torch.isnan(at_0[:2]).all() and torch.equal(at_0[2], panel.x[0, 1])
    assert torch.isnan(at_1[0]).all() and torch.equal(at_1[1:], panel.x[0:2, 1])


def test_sequence_training_items_are_the_valid_target_cells_and_evaluation_the_present():
    panel = _panel()
    bars = [1, 2, 3, 4, 5]

    def cells(dataset):
        return sorted((int(i.where[0]), int(i.where[1])) for i in dataset)

    def expected(flags):
        return sorted((t, s) for t in bars for s in torch.nonzero(flags[t]).flatten().tolist())

    training = SymbolSequenceDataset(panel, bars=bars, window_bars=2, training=True)
    evaluation = SymbolSequenceDataset(panel, bars=bars, window_bars=2, training=False)
    assert cells(training) == expected(panel.mask)
    assert cells(evaluation) == expected(panel.present)
    assert all(bool(item.mask) for item in training)
    assert not any(t in (0, 4) for t, _ in cells(evaluation))  # bar 0 not asked, bar 4 empty


def test_batched_fetching_equals_fetching_items_one_by_one():
    panel = _panel()
    dataset = SymbolSequenceDataset(panel, bars=range(T), window_bars=3, training=False)
    indices = [7, 0, 12, 3, 3, len(dataset) - 1]
    batched = dataset.__getitems__(indices)
    assert len(batched) == len(indices)
    for got, i in zip(batched, indices):
        one = dataset[i]
        for a, b in zip(got[:4], one[:4]):
            torch.testing.assert_close(a, b, equal_nan=True)
        assert torch.equal(got.where[0], one.where[0])
        assert torch.equal(got.where[1], one.where[1])


def test_default_collation_batches_sequence_items_to_b_n_f():
    panel = _panel()
    dataset = SymbolSequenceDataset(panel, bars=range(T), window_bars=3, training=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batches = [Batch(*b) for b in loader]

    first = batches[0]
    assert first.x.shape == (4, 3, F) and first.mask.shape == (4,)
    assert first.y.shape == first.y_raw.shape == (4, L)
    assert first.where[0].shape == first.where[1].shape == (4,)
    assert sum(len(b.mask) for b in batches) == len(dataset)
    for j, i in enumerate(range(4, 8)):
        torch.testing.assert_close(batches[1].x[j], dataset[i].x, equal_nan=True)


def test_a_one_bar_window_gives_row_samples():
    panel = _panel()
    dataset = SymbolSequenceDataset(panel, bars=[5], window_bars=1, training=False)
    batch = Batch(*next(iter(DataLoader(dataset, batch_size=len(dataset)))))
    assert batch.x.shape == (4, 1, F)
    torch.testing.assert_close(batch.x[:, 0], panel.x[5, batch.where[1]])


def test_present_is_computed_bar_block_by_bar_block_as_on_the_whole_array(monkeypatch):
    # torch.isfinite on the whole [T, S, F] array held a float32 copy of it and three
    # [T, S, F] boolean temporaries: about 150 GB beside the us3000 features.
    import quantlab.model.torch_data as torch_data_module

    rng = np.random.default_rng(7)
    x = rng.standard_normal((11, 5, 3)).astype(np.float32)
    x[2, 1, :] = np.nan
    x[3, 0, :] = [np.inf, np.nan, -np.inf]
    x[4, 4, 0] = np.nan  # one finite value left: present
    monkeypatch.setattr(torch_data_module, "PRESENT_BLOCK_BARS", 4)

    panel = TrainingPanel.from_arrays(x, timestamps=np.arange(11), symbols=np.arange(5))

    assert panel.present.numpy().tolist() == np.isfinite(x).any(axis=-1).tolist()
