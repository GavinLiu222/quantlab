"""``quantlab.__version__`` is the installed distribution's version."""

import importlib.metadata

import quantlab


def test_version_is_the_installed_distribution_s():
    try:
        expected = importlib.metadata.version("quantlab-py")
    except importlib.metadata.PackageNotFoundError:
        expected = "0+unknown"
    assert quantlab.__version__ == expected
