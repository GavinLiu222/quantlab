"""Regression tests for the `data/{market}/{frequency}/{name}.zarr` storage
path convention (02-CONTEXT.md D-01/D-02) that every config factory in
`quantlab/config/__init__.py` must follow: `stock_kline_config()`,
`stock_acquisition_config()` and `universe_config()` (D-09).
"""

import ast
from pathlib import Path

import quantlab.config as config
from quantlab.acquisition.config import stock_acquisition_config
from quantlab.config import get_data_root, set_data_root, stock_kline_config, universe_config

_CONFIG_SOURCE = Path(config.__file__).resolve()
#: Every module holding a root-derived config factory: the package itself and the
#: acquisition factory beside ``AcquisitionConfig``.
_FACTORY_SOURCES = (
    _CONFIG_SOURCE,
    Path(__file__).resolve().parents[1] / "quantlab/acquisition/config.py",
)

#: Derived from THIS FILE, deliberately not from `config.__file__`. The
#: storage-root default is itself a walk up from the configuration
#: module, so a root derived the same way moves with that expression and
#: would keep agreeing with it however wrong it got. `tests/` sits one
#: level below the repository root, the same independent witness
#: `test_data_dir_cli.py` uses.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_stock_kline_config_uses_market_frequency_path_convention() -> None:
    cfg = stock_kline_config()

    assert "data/us_equity/1d/" in cfg.zarr_file_path.replace("\\", "/")
    assert cfg.market == "us_equity"
    assert cfg.frequency == "1d"


def test_stock_acquisition_config_has_no_credential_field() -> None:
    cfg = stock_acquisition_config(symbols=("AAPL",))

    assert cfg.market == "us_equity"
    assert cfg.frequency == "1d"
    assert cfg.vendor == "tiingo"

    # `to_dict()` returns `asdict(self)`, and that dict lands in persisted
    # configs and in the JSON saved beside model checkpoints. A credential
    # field here is a credential committed to disk in an artefact nobody
    # audits -- which is how this repo leaked a real Tiingo key once already.
    # Every vendor's credentials are read from `os.environ` inside the client's
    # `__init__` (D-15, T-03.2-02).
    payload = cfg.to_dict()
    forbidden = {
        "api_key",
        "token",
        "secret",
        "tiingo_api_key",
        "apca_api_key_id",
        "apca_api_secret_key",
    }
    assert not (set(payload) & forbidden), sorted(set(payload) & forbidden)


def test_stock_config_factories_carry_the_alpaca_vendor_through_both_paths() -> None:
    """03.2 D-11: `vendor` is DERIVED into the paths by the factory rather than
    accepted pre-built, so the raw root and the watermark root cannot drift
    apart at a call site.
    """
    acq = stock_acquisition_config(symbols=("AAPL",), vendor="alpaca")
    ds = stock_kline_config(vendor="alpaca")

    assert acq.vendor == "alpaca"
    assert Path(acq.raw_data_dir_path).name == "alpaca"
    assert acq.raw_data_dir_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/nasdaq_data/alpaca"
    )
    assert acq.watermark_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/nasdaq_data/_watermarks/alpaca"
    )
    assert not acq.watermark_path.startswith(acq.raw_data_dir_path)

    # The dataset side records the same vendor AND terminates at it, which is
    # what makes `StockDataset._scan_raw`'s basename assertion expressible.
    assert ds.vendor == "alpaca"
    assert Path(ds.raw_data_dir_path).name == ds.vendor
    assert ds.raw_data_dir_path == acq.raw_data_dir_path

    # Two vendors resolve to SIBLING roots under one shared parent -- that is
    # the tree SC-7's isolation assertions are written against.
    tiingo = stock_acquisition_config(symbols=("AAPL",), vendor="tiingo")
    assert (
        Path(tiingo.raw_data_dir_path).parent
        == Path(acq.raw_data_dir_path).parent
    )
    assert tiingo.raw_data_dir_path != acq.raw_data_dir_path


def test_stock_config_defaults_are_byte_identical_without_the_new_arguments() -> None:
    """260906-0iy Task 3. The full-market roster needs its own raw-data
    subdirectory and Zarr store, but every existing call site passes neither
    argument -- so the defaults must reproduce today's paths exactly, not
    merely "something under data/us_equity/1d/".
    """
    acq = stock_acquisition_config(symbols=("AAPL",))
    ds = stock_kline_config()

    # 03.2 D-11/D-19: both paths now carry the vendor segment, and `"tiingo"`
    # is the documented incumbent -- every pre-03.2 caller of these factories
    # fetched from Tiingo, so the default names what is already on disk.
    assert acq.raw_data_dir_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/nasdaq_data/tiingo"
    )
    # The watermark root is a SIBLING of the raw root, never inside it: a
    # polars directory scan walks every file beneath the root it is given, so a
    # `.json` sidecar in the raw tree breaks `pl.scan_parquet` outright.
    assert acq.watermark_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/nasdaq_data/_watermarks/tiingo"
    )
    assert not acq.watermark_path.startswith(acq.raw_data_dir_path)
    assert ds.raw_data_dir_path == acq.raw_data_dir_path
    assert ds.zarr_file_path.replace("\\", "/").endswith(
        "data/us_equity/1d/stock.zarr"
    )


def test_stock_config_subdir_and_store_name_redirect_under_the_same_root() -> None:
    """D-04: there stays exactly ONE storage root, whichever of the three
    levels answers it (`set_data_root` > `QUANTLAB_DATA_DIR` > repo-root `data/`,
    260907-rjq D-01). These arguments select a subdirectory/filename BENEATH
    the existing `data/{market}/{frequency}/` convention -- they are not a
    second root and they never hardcode a volume.
    """
    acq = stock_acquisition_config(symbols=("AAPL",), subdir="us_all")
    ds = stock_kline_config(subdir="us_all", store_name="us_all.zarr")

    assert acq.raw_data_dir_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/us_all/tiingo"
    )
    assert acq.watermark_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/us_all/_watermarks/tiingo"
    )
    assert ds.raw_data_dir_path == acq.raw_data_dir_path
    assert "data/us_equity/1d/" in ds.zarr_file_path.replace("\\", "/")
    assert ds.zarr_file_path.replace("\\", "/").endswith("us_all.zarr")


# ---------------------------------------------------------------------------
# The storage root: one root, three levels (260907-rjq D-01), asserted through
# real factory paths rather than through the resolver alone.
# ---------------------------------------------------------------------------

def test_cli_override_beats_the_env_var_in_a_real_factory_path(
    monkeypatch, tmp_path
) -> None:
    """D-01 cells 1 and 2, through `stock_kline_config()` rather than through
    `get_data_root()` -- a resolver that answers correctly while no factory
    consults it would pass a resolver-only test.
    """
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    set_data_root(tmp_path / "cli")
    assert stock_kline_config().zarr_file_path.startswith(str(tmp_path / "cli"))

    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(tmp_path / "env"))
    assert stock_kline_config().zarr_file_path.startswith(str(tmp_path / "cli"))


def test_env_var_and_repo_default_answer_when_the_override_is_cleared(
    monkeypatch, tmp_path
) -> None:
    """D-01 cells 3 and 4."""
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(tmp_path / "env"))
    set_data_root(None)
    assert stock_kline_config().zarr_file_path.startswith(str(tmp_path / "env"))

    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    assert stock_kline_config().zarr_file_path.startswith(
        str(_REPO_ROOT / "data")
    )


def test_an_installed_package_defaults_to_the_users_home_directory(
    monkeypatch, tmp_path
) -> None:
    """A wheel installed into site-packages has no repository beside it, so
    the default root is `~/.quantlab/data`, never a `data/` directory inside
    site-packages."""
    site_packages = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    monkeypatch.setattr(
        config, "__file__", str(site_packages / "quantlab" / "config" / "__init__.py")
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    set_data_root(None)

    assert get_data_root() == tmp_path / "home" / ".quantlab" / "data"


def test_the_override_reaches_every_path_field_the_flag_promises(
    monkeypatch, tmp_path
) -> None:
    """DDIR-01: `set_data_root("/X")` must move the four path fields a run actually
    writes -- `raw_data_dir_path`, `watermark_path`, `zarr_file_path` and the
    universe table's `output_path` -- across every factory, not just the one
    factory the flag was first wired to.
    """
    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    set_data_root(tmp_path)
    root = str(tmp_path)

    acq = stock_acquisition_config(symbols=("AAPL",))
    assert acq.raw_data_dir_path.startswith(root)
    assert acq.watermark_path.startswith(root)
    assert stock_kline_config().zarr_file_path.startswith(root)
    assert universe_config().output_path.startswith(root)
    assert universe_config().cache_dir.startswith(root)


#: Names that root a path expression: the resolver itself and the two roots
#: derived from it. Anything else appearing at the head of a path expression in
#: a factory body is a second root.
_ROOTING_CALLS = {"get_data_root", "_market_data_root", "_market_downloads_root"}


def _rooted_path_keywords() -> dict[str, list[str]]:
    """Map each `*_config` factory in `_FACTORY_SOURCES` to the path-shaped
    keyword arguments it passes, flagging any that is not rooted in
    `get_data_root()` or one of its two derived roots.

    Reads the module's own source, so reach is exhaustive by CONSTRUCTION. An
    enumerated list of factories would go stale the day a thirteenth factory is
    added -- the failure mode this repo has already hit once, with the
    `--universe` choices that drifted from `UNIVERSE_CATEGORY_MAP`.
    """
    offenders: dict[str, list[str]] = {}

    for func in (f for source in _FACTORY_SOURCES for f in ast.parse(source.read_text()).body):
        if not isinstance(func, ast.FunctionDef) or not func.name.endswith("_config"):
            continue

        # Local names bound to a rooted expression, e.g.
        # `downloads = _market_downloads_root(...)`, count as rooted too.
        rooted_locals: set[str] = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            calls = {
                c.func.id
                for c in ast.walk(node.value)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
            if calls & _ROOTING_CALLS:
                rooted_locals.update(
                    t.id for t in node.targets if isinstance(t, ast.Name)
                )

        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg is None or not (
                    kw.arg.endswith("_path") or kw.arg.endswith("_dir")
                ):
                    continue
                names = {
                    n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)
                }
                if not (names & (_ROOTING_CALLS | rooted_locals)):
                    offenders.setdefault(func.name, []).append(kw.arg)

    return offenders


def test_no_factory_path_bypasses_the_single_root() -> None:
    """DDIR-02: exactly one storage root, and no hardcoded volume anywhere.

    Checked by CONSTRUCTION over the module's own source rather than by
    enumerating factories, so a factory added tomorrow is covered the day it
    lands.
    """
    offenders = _rooted_path_keywords()

    assert not offenders, (
        "these config factory path arguments are not rooted in "
        f"get_data_root() or a root derived from it: {offenders}. Every "
        "storage path must derive from the one resolved root -- a second root "
        "or a hardcoded volume is what the three-level precedence exists to "
        "make unnecessary."
    )


def test_the_by_construction_root_check_actually_inspects_something() -> None:
    """A by-construction check that matches nothing passes vacuously. Count the
    path keywords the scan CONSIDERED, so a keyword-naming change that makes
    the scan blind fails here instead of going quietly green.
    """
    considered = [
        kw.arg
        for source in _FACTORY_SOURCES
        for func in ast.parse(source.read_text()).body
        if isinstance(func, ast.FunctionDef) and func.name.endswith("_config")
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg and (kw.arg.endswith("_path") or kw.arg.endswith("_dir"))
    ]

    # stock_kline_config (raw_data_dir_path, zarr_file_path),
    # stock_acquisition_config (raw_data_dir_path, watermark_path) and
    # universe_config (output_path, cache_dir).
    assert len(considered) >= 6, (
        f"only {len(considered)} path-shaped keyword arguments found in "
        "the config factories; the by-construction root check "
        "above "
        "is scanning less than the file contains."
    )


def test_env_var_users_and_default_users_both_reproduce_todays_paths(
    monkeypatch, tmp_path
) -> None:
    """DDIR-03: the flag is additive. With no override in play, an env-var user
    still gets an env-rooted tree and a repo-default user still gets the exact
    literals `test_stock_config_defaults_are_byte_identical_without_the_new_
    arguments` pins.
    """
    set_data_root(None)

    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(tmp_path / "env"))
    assert stock_acquisition_config(
        symbols=("AAPL",)
    ).raw_data_dir_path.startswith(str(tmp_path / "env"))
    assert stock_kline_config().zarr_file_path.startswith(str(tmp_path / "env"))

    monkeypatch.delenv("QUANTLAB_DATA_DIR", raising=False)
    acq = stock_acquisition_config(symbols=("AAPL",))
    ds = stock_kline_config()
    assert acq.raw_data_dir_path.replace("\\", "/").endswith(
        "downloads/us_equity/1d/nasdaq_data/tiingo"
    )
    assert ds.zarr_file_path.replace("\\", "/").endswith(
        "data/us_equity/1d/stock.zarr"
    )
    assert acq.raw_data_dir_path.startswith(str(get_data_root()))
    assert get_data_root() == _REPO_ROOT / "data"
