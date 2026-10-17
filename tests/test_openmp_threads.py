"""``openmp_threads``: the xgboost heads run on the OpenMP threads they asked for.

torch, started under ``MKL_NUM_THREADS=1``, leaves the OpenMP runtime it
shares with xgboost on Linux at one thread, and xgboost's ``nthread`` cannot
raise it (measured 2026-10-17: a 2,000,000 x 360 DMatrix took 13.5 s at
``nthread=64`` against 4.4 s). The heads set the count around every xgboost
call; these tests lock the rule and that it takes effect. That the heads
apply it is locked in tests/test_xgb_model.py and tests/test_xgb_td_model.py.
"""

import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from quantlab.model.predefined._support.threads import (
    openmp_thread_count,
    openmp_threads,
)


@pytest.mark.parametrize(
    ("requested", "environ", "expected"),
    [
        (32, {}, 32),
        (None, {}, 64),
        (0, {}, 64),
        (-1, {}, 64),
        (32, {"OMP_NUM_THREADS": "8"}, 8),
        (4, {"OMP_NUM_THREADS": "8"}, 4),
        (None, {"OMP_NUM_THREADS": "1"}, 1),
        (32, {"OMP_NUM_THREADS": ""}, 32),
        (32, {"OMP_NUM_THREADS": "many"}, 32),
        (32, {"OMP_NUM_THREADS": "0"}, 32),
    ],
)
def test_the_count_is_the_request_or_every_cpu_capped_by_omp_num_threads(requested, environ, expected):
    assert openmp_thread_count(requested, environ=environ, cpus=64) == expected


def _openmp_counts() -> list[int]:
    return [pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "openmp"]


def test_the_block_runs_on_the_count_and_the_previous_count_comes_back(monkeypatch):
    import xgboost  # noqa: F401 - loads an OpenMP runtime

    if not _openmp_counts():
        pytest.skip("no OpenMP runtime loaded")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with threadpool_limits(limits=1, user_api="openmp"):
        with openmp_threads(2):
            assert set(_openmp_counts()) == {2}
        assert set(_openmp_counts()) == {1}
