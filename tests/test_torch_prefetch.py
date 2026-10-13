"""`prefetch`: a background thread readies a loader's next batches on the model's device.

It must hand over exactly the batches the loader gives, in the loader's order,
so a fit is as reproducible as without it; read ahead while the caller works
on the current batch; pass the loader's errors on; and leave no thread behind
when the caller stops early.

What turns this file red:
- a batch dropped, repeated, reordered, or left off the device;
- no read-ahead: the next batch is only fetched once the caller asks for it;
- an error in the loader swallowed, or raised in the wrong thread;
- a thread still running after the caller closed the iterator.

The CUDA path (pinning, the side stream) runs only where CUDA exists.
"""

import threading

import pytest
import torch

from quantlab.model.torch_data import Batch, prefetch


def _batch(i: int) -> Batch:
    return Batch(
        x=torch.full((2, 3, 1), float(i)), y=torch.full((2, 1), float(i)),
        mask=torch.tensor([True, i % 2 == 0]), y_raw=torch.full((2, 1), float(i)),
        where=(torch.full((2,), i), torch.tensor([0, 1])),
    )


def _prefetch_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("quantlab-prefetch")]


def test_hands_over_every_batch_in_the_loaders_order():
    batches = [_batch(i) for i in range(5)]
    out = list(prefetch(batches, "cpu"))
    assert [int(b.where[0][0]) for b in out] == [0, 1, 2, 3, 4]
    for got, want in zip(out, batches):
        for a, b in zip((got.x, got.y, got.mask, got.y_raw, *got.where),
                        (want.x, want.y, want.mask, want.y_raw, *want.where)):
            assert a.device.type == "cpu"
            assert torch.equal(a, b)


def test_reads_the_next_batch_while_the_caller_holds_the_current_one():
    """RED under: fetching a batch only when the caller asks for it."""
    second_fetched = threading.Event()

    def source():
        yield _batch(0)
        second_fetched.set()
        yield _batch(1)

    batches = prefetch(source(), "cpu")
    next(batches)
    assert second_fetched.wait(timeout=5)
    batches.close()


def test_an_error_in_the_loader_is_raised_to_the_caller_after_the_batches_before_it():
    def source():
        yield _batch(0)
        raise RuntimeError("gather failed")

    batches = prefetch(source(), "cpu")
    assert int(next(batches).where[0][0]) == 0
    with pytest.raises(RuntimeError, match="gather failed"):
        next(batches)
    assert not _prefetch_threads()


def test_closing_early_stops_the_thread_and_reads_no_further():
    fetched = []

    def source():
        for i in range(100):
            fetched.append(i)
            yield _batch(i)

    batches = prefetch(source(), "cpu")
    next(batches)
    batches.close()
    assert not _prefetch_threads()
    assert len(fetched) < 100


def test_an_empty_loader_gives_nothing():
    assert list(prefetch([], "cpu")) == []
    assert not _prefetch_threads()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_batches_arrive_on_the_gpu_with_the_loaders_values():
    batches = [_batch(i) for i in range(4)]
    out = list(prefetch(batches, "cuda"))
    for got, want in zip(out, batches):
        for a, b in zip(got.tensors(), want.tensors()):
            assert a.device.type == "cuda"
            assert torch.equal(a.cpu(), b)
