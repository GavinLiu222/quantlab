"""quantlab: a config-driven quant research backend.

The package imports nothing at the top: every module is imported by its full
dotted path (``from quantlab.dataset.stock import StockDataset``). This file
holds only ``__version__`` and may import the standard library only
(``tests/test_source_inspector.py``).
"""

import importlib.metadata

try:
    #: The installed distribution's version, by its PyPI name ``quantlab-py``
    #: (``0+unknown`` when it is not installed).
    __version__ = importlib.metadata.version("quantlab-py")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+unknown"
