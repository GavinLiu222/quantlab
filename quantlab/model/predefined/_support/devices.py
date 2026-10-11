"""Default device of the library heads.

A head resolves its device when training starts and again when a
checkpoint is loaded, not when the module is imported: CUDA when a CUDA
device is available, otherwise the CPU. Apple MPS is never chosen
automatically, because MPS runs are not deterministic and a macOS process
mixing torch and xgboost already needs ``OMP_NUM_THREADS=1``; pass
``device="mps"`` explicitly to use it.

``resolve_device`` is the one rule every head applies: an explicit device
wins, an unset one (``None``) becomes the library's default. The defaults
come from ``torch_default_device``, which asks torch (a pytabkit head has
imported it anyway), and ``xgboost_default_device``, which never imports
torch: ``xgboost_cuda_available`` reads xgboost's build information and,
only for a CUDA build, asks ``cuda_driver_device_count`` how many devices
the CUDA driver sees. That answer honours ``CUDA_VISIBLE_DEVICES``. It
creates no CUDA context, but ``cuInit`` does initialise the driver in the
calling process, so a process that later forks workers which use CUDA
should start them with the ``spawn`` method.
"""

import ctypes
import functools
import sys


def resolve_device(requested: str | None, default) -> str:
    """Return ``requested``, or ``default()`` when it is ``None``.

    Parameters
    ----------
    requested : str or None
        The device the caller asked for, ``None`` when unset.
    default : callable
        Returns the default device; called only when ``requested`` is None.

    Examples
    --------
    >>> resolve_device(None, lambda: "cuda")
    'cuda'
    >>> resolve_device("cpu", lambda: "cuda")
    'cpu'
    """
    return default() if requested is None else requested


def cuda_free_bytes(device: str) -> int:
    """Return the free memory of the CUDA ``device`` in bytes, as torch reads it.

    Parameters
    ----------
    device : str
        A CUDA device, ``"cuda"`` (the current one) or ``"cuda:<index>"``.

    Returns
    -------
    int
        The bytes the CUDA driver reports free on the device.

    Examples
    --------
    >>> cuda_free_bytes("cuda") > 0  # on a machine with CUDA
    True
    """
    import torch

    return int(torch.cuda.mem_get_info(torch.device(device))[0])


def torch_default_device() -> str:
    """Return ``"cuda"`` when torch sees a CUDA device, else ``"cpu"``.

    Never ``"mps"``.

    Examples
    --------
    >>> torch_default_device()  # on a machine without CUDA
    'cpu'
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def cuda_driver_device_count() -> int:
    """Return the number of CUDA devices the driver reports, 0 on any failure.

    Loads the CUDA driver library (``libcuda.so.1`` on Linux, ``nvcuda.dll``
    on Windows) and calls ``cuInit`` and ``cuDeviceGetCount``. A missing
    driver, a failed call or an unsupported platform all give 0.

    Examples
    --------
    >>> cuda_driver_device_count()  # on a Mac
    0
    """
    names = ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    for name in names:
        try:
            driver = ctypes.CDLL(name)
        except OSError:
            continue
        try:
            if driver.cuInit(0) != 0:
                return 0
            count = ctypes.c_int(0)
            if driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
                return 0
            return int(count.value)
        except (AttributeError, OSError):
            return 0
    return 0


@functools.cache
def xgboost_cuda_available() -> bool:
    """Return whether xgboost can train on CUDA here, without importing torch.

    True when the installed xgboost is a CUDA build (``build_info()["USE_CUDA"]``)
    and the CUDA driver reports at least one visible device. The answer is
    cached for the process; ``xgboost_cuda_available.cache_clear()`` resets it.

    Examples
    --------
    >>> xgboost_cuda_available()  # the macOS wheel is a CPU build
    False
    """
    import xgboost

    if not xgboost.build_info().get("USE_CUDA", False):
        return False
    return cuda_driver_device_count() > 0


def xgboost_default_device() -> str:
    """Return ``"cuda"`` when xgboost can train on CUDA, else ``"cpu"``.

    Examples
    --------
    >>> xgboost_default_device()  # on a Mac
    'cpu'
    """
    return "cuda" if xgboost_cuda_available() else "cpu"
