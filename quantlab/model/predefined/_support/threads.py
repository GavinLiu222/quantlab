"""OpenMP threads of the xgboost heads.

xgboost runs its CPU work (building a ``DMatrix``, CPU training, CPU
prediction) on OpenMP, and on Linux it shares the process's ``libgomp``
with torch. torch sets that runtime's thread count when it starts: with
``MKL_NUM_THREADS=1`` in the environment and ``OMP_NUM_THREADS`` unset, it
sets one thread, and xgboost's own ``nthread`` cannot raise it again (it is
capped by the runtime's thread count). Measured 2026-10-17 on the training
server: a ``DMatrix`` of 2,000,000 x 360 float32 rows took 13.5 s at
``nthread=64`` after ``import torch`` under ``MKL_NUM_THREADS=1``, against
4.4 s without torch.

``openmp_threads`` sets the OpenMP thread count for the duration of an
xgboost call (through ``threadpoolctl``) and restores it afterwards, so the
head runs on the threads it was asked for whatever torch did.
``openmp_thread_count`` is the rule: the head's requested thread count, all
the CPUs the process may run on when it requested none, and never more than
an ``OMP_NUM_THREADS`` set in the environment (the macOS guard of
``tests/conftest.py`` sets it to 1, see ``docs/model.md``).
"""

import os
from contextlib import contextmanager

from threadpoolctl import threadpool_limits


def available_cpus() -> int:
    """Return the number of CPUs this process may run on.

    The process's CPU affinity where the platform reports it (Linux), else
    ``os.cpu_count()``; at least 1.

    Examples
    --------
    >>> available_cpus() >= 1
    True
    """
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def openmp_thread_count(requested: int | None, environ=None, cpus: int | None = None) -> int:
    """Return the OpenMP thread count for an xgboost call that asked for ``requested``.

    Parameters
    ----------
    requested : int or None
        The head's thread count (xgboost's ``nthread``, pytabkit's
        ``n_threads``). None, 0 or a negative value (xgboost's "all")
        means every available CPU.
    environ : mapping, optional
        The environment, ``os.environ`` by default. A positive
        ``OMP_NUM_THREADS`` in it caps the count.
    cpus : int, optional
        The available CPUs, ``available_cpus()`` by default.

    Returns
    -------
    int
        At least 1.

    Examples
    --------
    >>> openmp_thread_count(32, environ={}, cpus=64)
    32
    >>> openmp_thread_count(None, environ={}, cpus=64)
    64
    >>> openmp_thread_count(-1, environ={"OMP_NUM_THREADS": "1"}, cpus=64)
    1
    """
    environ = os.environ if environ is None else environ
    count = requested if requested is not None and requested > 0 else (cpus or available_cpus())
    try:
        cap = int(environ.get("OMP_NUM_THREADS", ""))
    except ValueError:
        cap = 0
    if cap > 0:
        count = min(count, cap)
    return max(1, int(count))


@contextmanager
def openmp_threads(requested: int | None):
    """Run the block with ``openmp_thread_count(requested)`` OpenMP threads.

    Every OpenMP runtime loaded in the process is set, and set back to its
    previous count when the block exits.

    Parameters
    ----------
    requested : int or None
        See ``openmp_thread_count``.

    Examples
    --------
    >>> with openmp_threads(8):
    ...     booster = xgb.train(params, xgb.DMatrix(x, label=y))
    """
    with threadpool_limits(limits=openmp_thread_count(requested), user_api="openmp"):
        yield
