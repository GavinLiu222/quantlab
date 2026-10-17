"""Home for the data-source registry proofs: ROADMAP success criteria SC-1 and
SC-2, requirements D-01..D-07.

SC-1 — every acquirable source is enumerable from ONE registry, with no vendor
class named at the call site; each descriptor carries its capabilities, its
credential env-var NAMES and its acquisition class.
SC-2 — enumeration and credential status never return a credential VALUE and
never require one to be present.

Scaffolded by plan 03.4-01 (Wave 0) and filled in by plan 03.4-02, which built
`quantlab/acquisition/registry.py` and both vendor descriptors.

TWO RULES THIS FILE IS SUBJECT TO, both from incidents this repository has
already had (recorded in `.planning/STATE.md`):

1. EVERY test here must be a real assertion. A pytest file containing zero
   tests exits **5** ("no tests ran"), and a per-file command reads that exit
   as success. A placeholder, a body that is only a no-op statement, or a
   skip/xfail marker is therefore indistinguishable from a passing file.

   On pytest 9.1.1 that same exit 5 is ALSO what a `-k` selector matching
   nothing produces ("no tests collected (N deselected)"). The two are told
   apart by which is expected: for a scaffold file exit 5 is the failure this
   file exists to prevent; for one of the not-yet-implemented selectors listed
   in rule 2 it is the required result. Every test below is a genuine
   assertion against code that exists: the first three are the Wave-0
   infrastructure self-tests (fixture layout, credential-name anchoring), the
   rest pin the registry contract itself.

2. A `-k` selector name must not be attached to a test that does not honestly
   cover that selector's behaviour. In 03.2 a mechanism was deleted and
   `-k fingerprint` stayed green, because the only test covering it was named
   outside its own selector. Every selector `03.4-VALIDATION.md` assigns to
   this file -- `one_descriptor_per_vendor`, `capabilities`,
   `capabilities_match_the_vendor_class`, `direct_class_reference`,
   `env_names_are_exactly_what_gates_construction`,
   `never_returns_a_credential_value`, `registration_tuple_shape`,
   `decorator_registers`, `enumeration_order`,
   `enumeration_is_complete_from_a_cold_import` -- now matches at least one
   test, and each of those tests genuinely covers the behaviour its selector
   names. Renaming one of them without moving its assertions is how the 03.2
   incident repeats.
"""

import json
import os
from pathlib import Path

import pytest

import quantlab.acquisition.alpaca as alpaca
import quantlab.acquisition.tiingo as tiingo
import quantlab.acquisition.wrds.taq as wrds_taq


def test_the_acquisition_config_fixture_serves_both_vendors_with_a_terminated_raw_root(
    acquisition_config,
) -> None:
    """The two path invariants every registry/inspector/run test inherits.

    `acquisition_config` is the fixture plan 02's registry tests and plan 04's
    inspector tests both build on, so its layout is load-bearing for work that
    has not been written yet. Two properties in particular:

    - `raw_data_dir_path` TERMINATES at the vendor segment, which is the
      equality `StockDataset._scan_raw` asserts to make a cross-vendor silent
      merge unreachable by accident.
    - the watermark root is a SIBLING of the raw root, never inside it. A
      polars directory scan of the raw root walks every file beneath it, so a
      `.json` sidecar in that tree would break the scan outright -- and
      `browse_raw` (D-10) is exactly such a scan.

    Both are asserted here rather than assumed, because a fixture whose layout
    quietly changed would take the later pruning and isolation tests with it
    while they still looked green.
    """
    tiingo_cfg = acquisition_config(vendor="tiingo")
    alpaca_cfg = acquisition_config(vendor="alpaca")

    for cfg in (tiingo_cfg, alpaca_cfg):
        raw_root = Path(cfg.raw_data_dir_path)
        watermark_root = Path(cfg.watermark_path).parent

        assert raw_root.name == cfg.vendor
        # Sibling, not descendant. Stated both ways: the positive form pins
        # the exact layout, the negative form pins the property that matters
        # (a layout change that kept the property would still be fine).
        assert watermark_root == raw_root.parent / "_watermarks"
        assert raw_root not in watermark_root.parents
        assert raw_root != watermark_root

    assert Path(tiingo_cfg.raw_data_dir_path) != Path(alpaca_cfg.raw_data_dir_path)


def test_vendor_credential_env_names_are_module_level_constants_not_client_attributes() -> (
    None
):
    """Each vendor's `CREDENTIAL_ENV_VARS` is exactly its module-level names.

    This is the pin plan 02's `SourceDescriptor.required_env` will be checked
    against in BOTH directions (D-04): the descriptor must declare exactly the
    names that gate construction in the acquisition class. That comparison is
    only meaningful if the acquisition class's own declaration is itself
    anchored to the module-level constants, so that half is asserted first,
    here, before anything depends on it.

    The constants are module-level rather than client-class attributes on
    purpose (03.2-02 deviation #2): a security control must not be reachable
    through an indirection whose whole purpose is to be replaced by a test
    double. `_AlpacaMarketDataClient` re-exposes both names, but they are
    DEFINED at module scope, and `CREDENTIAL_ENV_VARS` is built from there.

    Only NAMES are compared. No value is read, and none of these names is
    required to be set for this test to run.
    """
    assert tiingo.KEY_ENV == "TIINGO_API_KEY"
    assert alpaca.KEY_ENV == "APCA_API_KEY_ID"
    assert alpaca.SECRET_ENV == "APCA_API_SECRET_KEY"

    assert tiingo.TiingoAcquisition.CREDENTIAL_ENV_VARS == (tiingo.KEY_ENV,)
    assert alpaca.AlpacaAcquisition.CREDENTIAL_ENV_VARS == (
        alpaca.KEY_ENV,
        alpaca.SECRET_ENV,
    )

    assert wrds_taq.USERNAME_ENV == "WRDS_USERNAME"
    assert wrds_taq.WrdsTaqNbboAcquisition.CREDENTIAL_ENV_VARS == (
        wrds_taq.USERNAME_ENV,
    )


def test_the_no_credentials_fixture_empties_every_declared_name(
    no_credentials,
) -> None:
    """The `no_credentials` fixture actually clears what it claims to clear.

    Every SC-3 proof in plan 04 -- "the read surface answers with NO
    credentials present" -- rests entirely on this fixture. If it silently
    stopped deleting one of the declared names on a machine where that name
    happens to be set, those tests would pass while proving nothing, which is
    the same green-on-nothing failure as the exit-5 trap.

    The fixture's return is asserted to be the NAMES, and only the names: it
    must never carry a credential value out of the environment it just
    emptied.
    """
    assert no_credentials == (
        "TIINGO_API_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "WRDS_USERNAME",
        "SHARADAR_API_KEY",
    )

    for name in no_credentials:
        assert name not in os.environ


# ---------------------------------------------------------------------------
# 03.4-02 Task 1 (TRACER) -- the ONE path, end to end
# ---------------------------------------------------------------------------


def test_tracer_end_to_end_registry_to_raw_shard(
    monkeypatch, acquisition_config, mock_tiingo_client
) -> None:
    """One path through every layer this phase touches, asserted end to end.

    Registry -> descriptor -> vendor module -> `SymbolBatchAcquisition` base -> raw shard
    on disk, in one test. This is deliberately NOT a per-layer unit test: the
    per-layer contracts are pinned separately, and what THIS proves is that the
    seams between them actually meet -- an architectural dead end surfaces here,
    after one commit, rather than after the inspector, the reporter, the cancel
    token and three shells have all been built on top of it.

    Four claims, in the order a caller meets them:

    1. the descriptor is reachable by VENDOR TOKEN, with no vendor class named
       anywhere in this test (SC-1);
    2. `is_configured` answers `False` with the key absent and `True` with it
       set -- and answers at all without a credential present (SC-2);
    3. `download_symbols(descriptor, config)` reaches the vendor through the descriptor's
       own class reference and returns an `AcquisitionResult` naming the
       requested symbols (D-12 / D-14 / D-18);
    4. `_failures.json` still lands on disk beside the watermarks, because the
       in-process result and the crash-durable record are two outputs on
       purpose, not one replacing the other (D-18).

    Issues zero real requests: `mock_tiingo_client` replaces the transport
    wholesale.
    """
    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.registry import download_symbols, is_configured
    from quantlab.acquisition.symbol_batch import AcquisitionResult

    descriptor = DataSourceRegistry.get("tiingo")
    config = acquisition_config(vendor="tiingo")

    # (2) -- both directions, and neither reads a VALUE.
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    assert is_configured(descriptor) is False
    monkeypatch.setenv("TIINGO_API_KEY", "not-a-real-credential")
    assert is_configured(descriptor) is True

    # (3)
    result = download_symbols(descriptor, config)

    assert isinstance(result, AcquisitionResult)
    assert result.vendor == "tiingo"
    assert set(result.requested) == set(config.symbols)
    assert set(result.succeeded) == set(config.symbols)
    assert result.failures == {}
    assert result.cancelled is False
    assert result.quota_aborted is False
    assert result.coverage["requested"] == len(config.symbols)

    # (4) -- the manifest is a SIBLING artefact, not a replacement. The two
    # coincide here because this store has seen exactly one clean run, so
    # there is nothing to carry forward; the general contract is CONTAINMENT,
    # `set(result.failures) <= set(manifest)` -- the manifest accumulates
    # across runs and may hold more than any single result (REVIEW CR-01,
    # pinned in `tests/test_acquisition_progress.py`).
    manifest = Path(config.watermark_path) / "_failures.json"
    assert manifest.exists()
    on_disk = json.loads(manifest.read_text())
    assert on_disk == {}, on_disk
    assert result.failures.items() <= on_disk.items()

    # ... and the raw tier actually received shards, under the vendor-
    # terminated root. Asserted on the ROOT the config names rather than on a
    # path rebuilt here, so a layout change fails at the config invariant test
    # above instead of silently passing a reconstructed path.
    raw_root = Path(config.raw_data_dir_path)
    assert raw_root.name == "tiingo"
    assert list(raw_root.rglob("*.pqt"))


# ---------------------------------------------------------------------------
# 03.4-02 Task 2 -- the D-01 / D-02 / D-03 / D-05 / D-06 registry contract
# ---------------------------------------------------------------------------


def _fake_descriptor(vendor: str, **overrides):
    """A descriptor over a NOVEL vendor token, built for one test.

    Novel by construction, in the spirit of
    `tests/test_extensibility_contract.py`'s `FakeDataset`: registering a token
    the production code has never heard of proves registration works by
    CONSTRUCTION rather than by re-asserting what the two shipped descriptors
    happen to do. Anything a caller would notice can be overridden.

    `symbol_batch_cls` defaults to the real `TiingoAcquisition` because D-03's
    "two descriptors may name the same class" case needs exactly that, and
    nothing here constructs it.
    """
    from quantlab.acquisition.base import Capability, SourceDescriptor

    fields = {
        "vendor": vendor,
        "display_name": f"Fake {vendor}",
        "symbol_batch_cls": tiingo.TiingoAcquisition,
        "config_factory": lambda **kw: None,
        "capabilities": (Capability(market="us_equity", frequency="1d"),),
        "required_env": (),
    }
    fields.update(overrides)
    return SourceDescriptor(**fields)  # type: ignore[arg-type]


def test_one_descriptor_per_vendor_rejects_a_duplicate(isolated_registry) -> None:
    """D-01: a second descriptor for a registered vendor RAISES.

    It must not merge the two capability lists and must not replace the first
    -- a registry that quietly took the last definition would let IMPORT ORDER
    decide what a vendor can do, which is the one thing enumeration order was
    sorted to stop mattering about.

    Registered under a NOVEL vendor token so the collision is created here
    rather than borrowed from the shipped descriptors: the mechanism is proved
    by construction, not by re-asserting a fact that already holds.
    """
    from quantlab.acquisition.base import register_source

    first = _fake_descriptor("fakevendor")
    register_source(first)
    assert isolated_registry.get("fakevendor") is first

    second = _fake_descriptor("fakevendor", display_name="Impostor")
    with pytest.raises(ValueError, match="fakevendor"):
        register_source(second)

    # Neither replaced nor merged.
    assert isolated_registry.get("fakevendor") is first
    assert second not in isolated_registry.SOURCES
    assert [d.vendor for d in isolated_registry.SOURCES].count("fakevendor") == 1


def test_capabilities_are_dataclass_instances_and_reject_the_cross_product() -> None:
    """D-02: capabilities are `Capability` INSTANCES, and the set is not a
    cross-product of two flat tuples.

    Alpaca's four entries and Tiingo's one are asserted exactly. A
    `markets x frequencies` product over Alpaca would yield `(us_equity, 1d)`,
    `(us_equity, 1m)` and `(us_equity, tick)` with no data_type at all -- it
    could express neither the quotes/trades split nor the fact that `tick` is
    the only frequency carrying one. `supports()` is asserted in both
    directions for the same reason: the negative case is the one a product
    would get wrong.
    """
    from quantlab.acquisition.base import Capability, DataSourceRegistry

    alpaca_source = DataSourceRegistry.get("alpaca")
    tiingo_source = DataSourceRegistry.get("tiingo")

    for descriptor in (alpaca_source, tiingo_source):
        assert isinstance(descriptor.capabilities, tuple)
        assert descriptor.capabilities
        for capability in descriptor.capabilities:
            assert isinstance(capability, Capability)

    def triples(descriptor):
        return {
            (c.market, c.frequency, c.data_type) for c in descriptor.capabilities
        }

    assert triples(alpaca_source) == {
        ("us_equity", "1d", "bars"),
        ("us_equity", "1m", "bars"),
        ("us_equity", "tick", "quotes"),
        ("us_equity", "tick", "trades"),
    }
    assert triples(tiingo_source) == {("us_equity", "1d", None)}

    assert tiingo_source.supports("us_equity", "1d") is True
    assert tiingo_source.supports("us_equity", "tick") is False
    assert alpaca_source.supports("us_equity", "tick", "quotes") is True
    assert alpaca_source.supports("us_equity", "tick", "trades") is True
    assert alpaca_source.supports("us_equity", "1m", "quotes") is False
    assert alpaca_source.supports("crypto_spot", "1d") is False


def test_capabilities_preserve_declaration_order_across_repeated_reads() -> None:
    """D-02: `capabilities` is an ordered tuple, not a set.

    Enumerating one descriptor's capabilities twice must yield the same
    sequence, because an operator surface renders them in that order and a set
    would reorder them per process (string hashing is salted per interpreter
    run). Asserted as a SEQUENCE, so a change to a frozenset fails here rather
    than only on a machine whose hash seed happens to differ.
    """
    from quantlab.acquisition.base import DataSourceRegistry

    for descriptor in DataSourceRegistry.all():
        first = list(descriptor.capabilities)
        second = list(descriptor.capabilities)
        assert first == second
        assert isinstance(descriptor.capabilities, tuple)

    assert [
        (c.frequency, c.data_type)
        for c in DataSourceRegistry.get("alpaca").capabilities
    ] == [("1d", "bars"), ("1m", "bars"), ("tick", "quotes"), ("tick", "trades")]


def test_capabilities_match_the_vendor_class_constants() -> None:
    """D-02: the capability set is PINNED against the vendor class's own
    constants, not restated independently of them.

    This is what turns "someone added an endpoint to `AlpacaAcquisition` and
    forgot the descriptor" into a red test instead of a source the console
    under-advertises forever. Three separate pins, because the class carries
    three separate facts:

    - bar frequencies <-> `TIMEFRAME_MAP` keys;
    - tick data types <-> `TICK_DATA_TYPES`;
    - every declared data_type is a key of `ENDPOINT_MAP`, i.e. a capability
      cannot name a shape the class has no endpoint for.

    Tiingo is pinned the other way round: it declares `data_type=None`, so the
    assertion is that it has no endpoint map to disagree with and exactly one
    frequency, matching `_FREQUENCY_MAP`.
    """
    from quantlab.acquisition.base import DataSourceRegistry

    alpaca_source = DataSourceRegistry.get("alpaca")
    alpaca_cls = alpaca.AlpacaAcquisition

    bar_frequencies = {
        c.frequency for c in alpaca_source.capabilities if c.data_type == "bars"
    }
    assert bar_frequencies == set(alpaca_cls.TIMEFRAME_MAP)

    tick_types = {
        c.data_type for c in alpaca_source.capabilities if c.frequency == "tick"
    }
    assert tick_types == set(alpaca_cls.TICK_DATA_TYPES)

    for capability in alpaca_source.capabilities:
        assert capability.data_type in alpaca_cls.ENDPOINT_MAP

    tiingo_source = DataSourceRegistry.get("tiingo")
    assert {c.frequency for c in tiingo_source.capabilities} == set(
        tiingo._FREQUENCY_MAP
    )
    assert all(c.data_type is None for c in tiingo_source.capabilities)
    assert not hasattr(tiingo.TiingoAcquisition, "ENDPOINT_MAP")


def test_wrds_descriptor_serves_nbbo_and_crsp_daily_capabilities() -> None:
    """D-17/D-27 + 03.10 D-12: ONE `wrds` descriptor, TWO capabilities.

    This is the shape plan 01's per-capability resolution existed for, now
    actually populated: a WRDS subscription is an ACCOUNT, not a product, and
    the two products it serves -- NYSE TAQ millisecond NBBO and CRSP Stock v2
    daily -- have different tables, different entitlements, different
    acquisition classes and differently-shaped configs. D-01 keeps that as one
    descriptor, so "which class downloads this" is answered per CAPABILITY.

    The descriptor-level pair is asserted to be STILL the TAQ one. That is not
    an arbitrary tie-break: four ingest shells read
    `SOURCE.symbol_batch_cls.DEFAULT_BATCH_SIZE` (and friends) directly, and
    `add-alongside` (plan 01's recorded decision) is what keeps them working.

    `supports(..., "quotes")` is asserted False because the Alpaca tick rows
    share `(us_equity, tick)` with the NBBO one: a registry that matched on
    the pair alone would hand an Alpaca quotes request a WRDS NBBO class.
    """
    from quantlab.acquisition.wrds import crsp
    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.dataset.crsp import CrspStockDataset
    from quantlab.dataset.nbbo import NbboPanelDataset

    source = DataSourceRegistry.get("wrds")
    assert source is WRDS_SOURCE

    assert {
        (c.market, c.frequency, c.data_type) for c in source.capabilities
    } == {
        ("us_equity", "tick", "nbbo"),
        ("us_equity", "1m", "nbbo_bars"),
        ("us_equity", "1d", "crsp_daily"),
    }

    by_key = {
        (c.market, c.frequency, c.data_type): c for c in source.capabilities
    }
    nbbo = by_key[("us_equity", "tick", "nbbo")]
    crsp_cap = by_key[("us_equity", "1d", "crsp_daily")]

    assert nbbo.dataset_cls is NbboPanelDataset
    assert crsp_cap.dataset_cls is CrspStockDataset

    assert nbbo.symbol_batch_cls is wrds_taq.WrdsTaqNbboAcquisition
    assert nbbo.config_factory == wrds_taq.WrdsTaqNbboAcquisition.build_config
    assert crsp_cap.symbol_batch_cls is crsp.WrdsCrspDailyAcquisition
    assert crsp_cap.config_factory == crsp.WrdsCrspDailyAcquisition.build_config

    # Resolved through the PUBLIC resolvers, per triple, not read off the row:
    # that is the path `registry.download_symbols()` takes.
    assert (
        source.symbol_batch_cls_for("us_equity", "tick", "nbbo")
        is wrds_taq.WrdsTaqNbboAcquisition
    )
    assert (
        source.config_factory_for("us_equity", "tick", "nbbo")
        == wrds_taq.WrdsTaqNbboAcquisition.build_config
    )
    assert (
        source.symbol_batch_cls_for("us_equity", "1d", "crsp_daily")
        is crsp.WrdsCrspDailyAcquisition
    )
    assert (
        source.config_factory_for("us_equity", "1d", "crsp_daily")
        == crsp.WrdsCrspDailyAcquisition.build_config
    )

    # The DESCRIPTOR default is unchanged (the `add-alongside` decision).
    assert source.symbol_batch_cls is wrds_taq.WrdsTaqNbboAcquisition
    assert source.config_factory == wrds_taq.WrdsTaqNbboAcquisition.build_config

    # One account, one credential name, whichever product is requested.
    assert source.required_env == ("WRDS_USERNAME",)

    assert source.supports("us_equity", "1d", "crsp_daily") is True
    assert source.supports("us_equity", "tick", "nbbo") is True
    assert source.supports("us_equity", "tick", "quotes") is False


def test_direct_class_reference_is_the_class_object(isolated_registry) -> None:
    """D-03: `symbol_batch_cls` is the CLASS OBJECT, never a dotted string.

    Asserted with `is`, so a dotted path that happened to resolve to the same
    class would still fail -- the point of D-03 is the direct reference itself,
    which is what makes `descriptor.symbol_batch_cls.DEFAULT_BATCH_SIZE`
    readable at a call site without a resolver.

    The second half is the D-03 corollary: only VENDOR collides. Two
    descriptors may name the SAME class under different vendors (alpaca paper
    vs live is the motivating case), so registering one must not raise.
    """
    from quantlab.acquisition.base import register_source

    assert isolated_registry.get("tiingo").symbol_batch_cls is tiingo.TiingoAcquisition
    assert isolated_registry.get("alpaca").symbol_batch_cls is alpaca.AlpacaAcquisition
    for descriptor in isolated_registry.all():
        # None only for a vendor its own scripts download (Sharadar, Massive).
        if descriptor.symbol_batch_cls is None:
            assert descriptor.vendor in ("sharadar", "massive")
            continue
        assert isinstance(descriptor.symbol_batch_cls, type)
        assert not isinstance(descriptor.symbol_batch_cls, str)

    twin = _fake_descriptor(
        "tiingotwin", symbol_batch_cls=tiingo.TiingoAcquisition
    )
    assert register_source(twin) is twin
    assert isolated_registry.get("tiingotwin").symbol_batch_cls is (
        isolated_registry.get("tiingo").symbol_batch_cls
    )


def test_registration_tuple_shape_is_instances_in_a_rebound_tuple() -> None:
    """D-05: `SOURCES` is a `tuple` of `SourceDescriptor` INSTANCES, rebound
    rather than mutated.

    This test is the ONLY thing keeping `tests/conftest.py:isolated_registry`
    honest, and the fixture's own docstring says so. That fixture isolates by
    `monkeypatch.setattr`-ing `SOURCES`, which saves the old OBJECT and
    restores it on teardown -- correct only while registration REBINDS the
    attribute. Switch `register_source` to `list.append` and the fixture would
    mutate the very object it restores: isolation would silently stop working
    and a fake descriptor would leak into every later test in the session, with
    nothing anywhere going red. Hence the source-level arm below.

    The instance-vs-class arm is the other half of D-05: `MEMBERSHIP_FETCHERS`
    holds classes, this holds instances, and a class accidentally registered
    here would still satisfy a `len()` check.
    """
    import ast

    from quantlab.acquisition.base import DataSourceRegistry, SourceDescriptor

    assert isinstance(DataSourceRegistry.SOURCES, tuple)
    assert DataSourceRegistry.SOURCES
    for element in DataSourceRegistry.SOURCES:
        assert isinstance(element, SourceDescriptor)
        assert not isinstance(element, type)

    source = Path("quantlab/acquisition/base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # No `.append` may reach SOURCES -- neither `SOURCES.append(...)` nor
    # `DataSourceRegistry.SOURCES.append(...)`. Matched on the ATTRIBUTE CHAIN
    # rather than on the rendered text, so a comment or docstring mentioning
    # `.append` (this one does) is not a false positive.
    appends = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and "SOURCES" in ast.unparse(node.func)
    ]
    assert appends == [], appends

    # ... and a REBIND is really there. Accepted in either syntactic form
    # (`SOURCES += (d,)` or `SOURCES = SOURCES + (d,)`), because what the
    # fixture depends on is the rebinding, not the spelling -- pinning the
    # exact node type would fail an equivalent and equally correct refactor.
    #
    # The `.append` scan above does NOT cover the whole risk on its own:
    # switching `SOURCES` to a LIST would make `SOURCES += (d,)` an in-place
    # `list.__iadd__` extend, passing both this arm and the append scan. The
    # `isinstance(..., tuple)` assertion at the top is what catches that, and
    # the three arms are only jointly sufficient.
    rebinds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.AugAssign, ast.Assign))
        and any(
            isinstance(target, ast.Attribute) and target.attr == "SOURCES"
            for target in (
                [node.target] if isinstance(node, ast.AugAssign) else node.targets
            )
        )
    ]
    assert len(rebinds) == 1, [ast.unparse(node) for node in rebinds]


def test_decorator_registers_and_returns_the_descriptor(isolated_registry) -> None:
    """D-06: `register_source(d) is d`, and applying it twice raises.

    The identity return is what makes the decorated module-level name the
    descriptor rather than `None` -- the classic decorator bug, which would
    only surface at the first attribute read somewhere far away.

    Applying the decorator to the SAME OBJECT a second time must take the
    duplicate-vendor path rather than being special-cased as idempotent: a
    module imported twice under two names is a real way that happens, and
    registering it twice would double the row in an operator's list.
    """
    from quantlab.acquisition.base import register_source

    descriptor = _fake_descriptor("decoratedvendor")
    assert register_source(descriptor) is descriptor
    assert descriptor in isolated_registry.SOURCES
    assert isolated_registry.get("decoratedvendor") is descriptor

    with pytest.raises(ValueError, match="already registered"):
        register_source(descriptor)
    assert isolated_registry.SOURCES.count(descriptor) == 1


def DataSourceRegistry_all():
    """The live registry's `all()`, named apart so the sorted-order test can
    assert on BOTH the shipped registry and a synthetic reversed one without
    the fixture-bound name shadowing the module-level one."""
    from quantlab.acquisition.base import DataSourceRegistry

    return DataSourceRegistry.all()


def test_enumeration_order_is_sorted_by_vendor(isolated_registry, monkeypatch) -> None:
    """D-06: `all()` SORTS by vendor rather than returning registration order.

    Import order is a function of which module the caller touched first, so two
    installations of the same code would otherwise render an operator's source
    list differently.

    The shipped registry cannot prove this on its own, and asserting only on it
    would be the 03.2 failure recorded in `.planning/STATE.md` -- "a lock that
    passes on arrival is mutation-verified rather than accepted". The three real
    descriptors happen to register in the order `alpaca`, `tiingo`, `wrds`
    (the order of the vendor imports at the bottom of `registry.py`), which is
    ALREADY sorted, so `all()` returning `tuple(cls.SOURCES)` unsorted would
    pass a live-registry assertion unchanged. So the ordering is exercised
    against a registry whose registration order is deliberately the REVERSE of
    its sorted order; deleting the `sorted(...)` call turns this red.
    """
    assert [d.vendor for d in DataSourceRegistry_all()] == [
        "alpaca", "fred", "massive", "sharadar", "tiingo", "wrds",
    ]

    reversed_registration = (
        _fake_descriptor("zzz-last-alphabetically"),
        _fake_descriptor("aaa-first-alphabetically"),
    )
    monkeypatch.setattr(isolated_registry, "SOURCES", reversed_registration)

    assert [d.vendor for d in isolated_registry.SOURCES] == [
        "zzz-last-alphabetically",
        "aaa-first-alphabetically",
    ]
    assert [d.vendor for d in isolated_registry.all()] == [
        "aaa-first-alphabetically",
        "zzz-last-alphabetically",
    ]


def test_registry_get_is_empty_and_unknown_safe(
    isolated_registry, monkeypatch
) -> None:
    """D-01: `all()` over an EMPTY registry returns `()` rather than raising,
    and `get()` for an unknown vendor names both the request and what exists.

    "Nothing is registered" is a legitimate state a console must be able to
    render -- an exception would make the empty case the caller's problem at
    exactly the moment it has the least information. `get()` is the opposite:
    a miss is a caller error, so it raises, and the message lists the
    registered vendors because a bare `KeyError` says nothing about a registry
    the caller cannot see.
    """
    monkeypatch.setattr(isolated_registry, "SOURCES", ())

    assert isolated_registry.all() == ()

    with pytest.raises(ValueError, match="tiingo") as excinfo:
        isolated_registry.get("tiingo")
    assert "Registered vendors: []" in str(excinfo.value)

    monkeypatch.setattr(isolated_registry, "SOURCES", (_fake_descriptor("only"),))
    with pytest.raises(ValueError) as excinfo:
        isolated_registry.get("nosuchvendor")
    assert "nosuchvendor" in str(excinfo.value)
    assert "only" in str(excinfo.value)


def test_registry_reaches_no_zarr_writer() -> None:
    """D-14's acquisition-only amendment, proved NEGATIVELY.

    `download_symbols()` downloads to the raw parquet tier and stops; the raw-to-Zarr
    conversion stays in the shells, because the three entry points convert in
    three different modes with three differently-sized RAM guards. The cheapest
    durable proof of "reaches no Zarr writer" is that this module imports no
    dataset of `quantlab.dataset` -- a behavioural test could only show that
    one particular call did not convert. The layer's root types and configs
    (`quantlab.dataset.base`, `quantlab.dataset.raw` with its
    `ConversionResult`, `quantlab.dataset.config`), which the registry names
    in its signatures, are not datasets (ADR 0022, ADR 0036).

    An `ast` walk rather than a substring scan: the docstrings in this file
    discuss Zarr conversion by name, and a grep would fail on the prose that
    explains the rule.
    """
    import ast

    tree = ast.parse(
        Path("quantlab/acquisition/registry.py").read_text(encoding="utf-8")
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)

    offending = sorted(
        name
        for name in imported
        if (name == "quantlab.dataset" or name.startswith("quantlab.dataset."))
        and not name.startswith(("quantlab.dataset.base", "quantlab.dataset.raw", "quantlab.dataset.config"))
    )
    assert offending == [], offending
    assert "quantlab.acquisition.base" in imported


# ---------------------------------------------------------------------------
# 03.4-02 Task 3 -- cold-import completeness (D-07/SC-1) and the credential
# contract (D-04/SC-2)
# ---------------------------------------------------------------------------

#: What the two subprocess tests below strip from the child's environment.
#: Written as LITERALS, not sourced from `CREDENTIAL_ENV_VARS` or from the
#: descriptors, for the same reason `tests/conftest.py:_CREDENTIAL_ENV_NAMES`
#: is: a proof that reads its own subject's declaration proves only that the
#: declaration is self-consistent.
_CHILD_CREDENTIALS = (
    "TIINGO_API_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "WRDS_USERNAME",
    "SHARADAR_API_KEY",
)


def _child_env() -> dict:
    """A copy of this process's environment with every vendor credential gone."""
    env = dict(os.environ)
    for name in _CHILD_CREDENTIALS:
        env.pop(name, None)
    return env


def _run_child(body: str):
    """Run `body` in a FRESH interpreter and return the completed process.

    A subprocess is REQUIRED rather than fastidious: this pytest session has
    already imported both vendor modules for other reasons, so an in-process
    assertion about what a "cold import" reaches would be measuring the state
    other tests left behind, not the import graph.
    """
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=_child_env(),
    )


def test_enumeration_is_complete_from_a_cold_import() -> None:
    """D-07 / SC-1: importing ONLY the registry enumerates every source.

    A decorator-populated registry is exactly as complete as the set of modules
    that have been imported, so "one registry an operator surface can
    enumerate" is a claim about the IMPORT GRAPH, not about the decorator. The
    child imports the registry entry point and nothing else.

    Run with every credential stripped from the child's environment, so this
    doubles as an SC-2 proof: enumeration needs no credential to be present.
    A descriptor that reached its credential at import time -- or an
    `symbol_batch_cls` constructed at module scope -- would make the child exit
    non-zero here.
    """
    child = _run_child(
        "import quantlab.acquisition.registry\n"
        "from quantlab.acquisition.base import DataSourceRegistry\n"
        "print(sorted(d.vendor for d in DataSourceRegistry.all()))\n"
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "['alpaca', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']", child.stdout
    assert "Traceback" not in child.stderr


def _no_network(monkeypatch) -> None:
    """Make ANY socket allocation raise.

    Copied from `tests/test_volume_guard.py:_no_network` (tests/ is not a
    package, so helpers travel by copy with an attribution comment).

    Used INSTEAD OF the `mock_tiingo_client` / `mock_alpaca_client` transport
    fixtures in the credential test below, and that substitution is the whole
    point of it: `mock_alpaca_client` replaces `_AlpacaMarketDataClient`
    wholesale, and that class IS Alpaca's missing-credential guard. Mocking the
    transport therefore DISABLES the control the test exists to prove --
    threat T-03.4-02-05, "a test double replacing the transport and disabling
    redaction", one row over. A socket tripwire keeps the real guards running
    while making a real request impossible.
    """
    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "a socket was opened; the credential contract is checked at "
            "construction and must cost zero vendor requests"
        )

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)


@pytest.mark.parametrize("vendor", ["alpaca", "tiingo", "wrds"])
def test_env_names_are_exactly_what_gates_construction(
    vendor, monkeypatch, acquisition_config
) -> None:
    """D-04: the descriptor declares EXACTLY the names that gate construction.

    The env var name now lives in two places -- `descriptor.required_env` and
    the vendor's own credential check -- so it is pinned in BOTH directions,
    because either half alone passes for the wrong reason:

    - **no name is missing:** with every declared name set, construction
      SUCCEEDS, so the descriptor does not UNDER-declare. A descriptor missing
      a name would report a source "configured" whose client then raises.
    - **no name is extra:** deleting ANY ONE declared name makes construction
      RAISE, naming that variable, so the descriptor does not OVER-declare. A
      descriptor with a spurious name would report a source unconfigured that
      would in fact have worked.

    Finished with the cheap structural half -- `required_env ==
    symbol_batch_cls.CREDENTIAL_ENV_VARS` -- which catches a rename applied in
    only one place. That comparison is not a tautology because
    `required_env` is a restated literal on the descriptor, never derived from
    `CREDENTIAL_ENV_VARS`; the other half of that anchoring is asserted in
    `test_vendor_credential_env_names_are_module_level_constants_...` above.

    Issues zero requests: a socket tripwire is installed instead of a transport
    mock (see `_no_network`), so both vendors' REAL guards run.
    """
    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.registry import is_configured

    _no_network(monkeypatch)
    descriptor = DataSourceRegistry.get(vendor)
    config = acquisition_config(vendor=vendor)

    assert descriptor.required_env, vendor

    # Direction 1 -- no name is missing.
    for name in descriptor.required_env:
        monkeypatch.setenv(name, "not-a-real-credential")
    assert is_configured(descriptor) is True
    descriptor.symbol_batch_cls(config)

    # Direction 2 -- no name is extra.
    for name in descriptor.required_env:
        monkeypatch.delenv(name)
        assert is_configured(descriptor) is False
        with pytest.raises(RuntimeError, match=name):
            descriptor.symbol_batch_cls(config)
        monkeypatch.setenv(name, "not-a-real-credential")

    # The structural half.
    assert descriptor.required_env == descriptor.symbol_batch_cls.CREDENTIAL_ENV_VARS


def test_is_configured_never_returns_a_credential_value(
    monkeypatch, acquisition_config, mock_tiingo_client
) -> None:
    """SC-2 / T-03.4-02-01: a planted credential VALUE escapes nowhere.

    This repository has already leaked one real Tiingo key, and an operator
    dashboard that renders an env var is how the next one happens. So a
    recognisable sentinel is planted in every credential name and asserted
    ABSENT from every egress path the registry has: both return values, the
    descriptor's `repr`, the whole enumeration's `repr`, the string form of an
    `AcquisitionResult` produced by a real `download_symbols()`, and every loguru record
    emitted while those calls run.

    There is deliberately no masked or partially-redacted variant to test:
    `is_configured` returns a `bool` and `credential_status` returns a `bool`
    per NAME, full stop. A masked display looks responsible and is the shape a
    leak takes next.

    The types are asserted to be ACTUAL bools rather than merely truthy, since
    a non-empty credential string is itself truthy -- returning the value
    unchanged would satisfy a truthiness assertion perfectly.
    """
    from loguru import logger

    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.registry import credential_status, download_symbols, is_configured

    sentinel = "SENTINEL-c0ffee-DO-NOT-LEAK"
    for name in _CHILD_CREDENTIALS:
        monkeypatch.setenv(name, sentinel)

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    try:
        tiingo_source = DataSourceRegistry.get("tiingo")

        configured = is_configured(tiingo_source)
        status = credential_status(tiingo_source)
        result = download_symbols(tiingo_source, acquisition_config(vendor="tiingo"))

        rendered = [
            repr(configured),
            repr(status),
            repr(tiingo_source),
            repr(DataSourceRegistry.all()),
            str(result),
            repr(result),
        ]
    finally:
        logger.remove(sink_id)

    for text in rendered + records:
        assert sentinel not in text, text

    # Booleans, not values dressed as booleans.
    assert configured is True
    assert type(configured) is bool
    assert set(status) == set(tiingo_source.required_env)
    for value in status.values():
        assert type(value) is bool

    # The empty-string case, matching `if not os.environ.get(...)` in both
    # vendor modules exactly: set-but-empty is NOT configured.
    monkeypatch.setenv("TIINGO_API_KEY", "")
    assert is_configured(tiingo_source) is False
    assert credential_status(tiingo_source) == {"TIINGO_API_KEY": False}


def test_is_configured_on_a_descriptor_with_no_required_env(
    isolated_registry, monkeypatch
) -> None:
    """D-04: `required_env=()` reports configured, and reads NO environment.

    A source needing no credential is a legitimate future case (a local-file
    source, a public endpoint), and it must answer `True` / `{}` rather than
    raising or falling into an "unknown" third state a console would have to
    render.

    "Reads no environment variable at all" is asserted rather than assumed, by
    swapping the module's `os` for one whose `environ.get` raises. A future
    implementation that consulted the environment first and special-cased the
    empty tuple afterwards would still return the right answers and would
    still fail here -- which is the point, because that shape is one refactor
    away from reading a name it was never given.
    """
    import types

    import quantlab.acquisition.registry as registry

    from quantlab.acquisition.base import register_source

    descriptor = _fake_descriptor("nocredvendor", required_env=())
    register_source(descriptor)

    class _ExplodingEnviron(dict):
        def get(self, *args, **kwargs):
            raise AssertionError(
                "the environment was read for a descriptor declaring no "
                "credential names"
            )

    monkeypatch.setattr(
        registry, "os", types.SimpleNamespace(environ=_ExplodingEnviron())
    )

    assert registry.is_configured(descriptor) is True
    assert registry.credential_status(descriptor) == {}
    assert isolated_registry.get("nocredvendor") is descriptor


# ---------------------------------------------------------------------------
# 03.10-01 Task 1 -- per-CAPABILITY acquisition class and config factory
#
# One vendor, several products: the `wrds` account serves TAQ NBBO through one
# acquisition class and (plan 02) CRSP daily through another. D-12 keeps that
# as ONE descriptor, so the class a request resolves to becomes a property of
# the CAPABILITY rather than of the vendor.
#
# The fields are OPTIONAL and the descriptor-level pair stays REQUIRED -- the
# `add-alongside` decision recorded in 03.10-01-PLAN.md. Four ingest shells
# read `SOURCE.symbol_batch_cls` / `SOURCE.config_factory` directly
# (`DEFAULT_BATCH_SIZE`, `TICK_DATA_TYPES`, ...), and making the descriptor
# field optional would ripple through all of them for no behavioural gain.
# ---------------------------------------------------------------------------


def _resolver_descriptor(**overrides):
    """An UNREGISTERED descriptor whose two capabilities differ in exactly the
    way the resolver has to tell apart.

    Built here rather than borrowed from a shipped vendor deliberately: today
    no shipped descriptor has two capabilities on one `(market, frequency)`
    with different classes, so a test written against the live registry would
    pass against a resolver that always returned the descriptor default. The
    ambiguity is created by construction.

    Not registered, so no `isolated_registry` is needed and nothing can leak.
    """
    from quantlab.acquisition.base import Capability, SourceDescriptor

    fields = {
        "vendor": "resolvervendor",
        "display_name": "Fake resolver vendor",
        "symbol_batch_cls": _A0,
        "config_factory": _f0,
        "capabilities": (
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="x",
                symbol_batch_cls=_A1,
                config_factory=_f1,
            ),
            Capability(market="us_equity", frequency="1d", data_type="y"),
        ),
        "required_env": (),
    }
    fields.update(overrides)
    return SourceDescriptor(**fields)  # type: ignore[arg-type]


def _recording_acquisition(name: str):
    """A minimal CONCRETE `SymbolBatchAcquisition` recording every construction.

    Its `_fetch_page` returns an EMPTY frame and no next token, so `download_symbols()`
    completes without a vendor request, a credential or a socket. The class
    carries a `constructed` list so a test can assert WHICH class `download_symbols()`
    reached rather than inferring it from a side effect.
    """
    from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition

    class _Recording(SymbolBatchAcquisition):
        VENDOR = "tiingo"
        RAW_COLUMNS = ("timestamp", "symbol", "vendor")
        label = name
        constructed: list = []

        def __init__(self, config):
            super().__init__(config)
            type(self).constructed.append(config)

        def _fetch_page(self, symbols, start_date, end_date, page_token=None):
            import polars as pl

            frame = pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "symbol": pl.String,
                    "vendor": pl.String,
                }
            )
            return frame.select(self.RAW_COLUMNS), None

    _Recording.__name__ = name
    return _Recording


_A0 = _recording_acquisition("_A0")
_A1 = _recording_acquisition("_A1")


def _f0(**kwargs):
    return "f0"


def _f1(**kwargs):
    return "f1"


def test_capability_resolution_prefers_the_capability_over_the_descriptor_default() -> (
    None
):
    """D-12: a capability that names its own class/factory wins; one that
    leaves them `None` falls back to the descriptor's default.

    Both directions are asserted on the SAME descriptor, because either half
    alone passes for the wrong reason: a resolver that always returned the
    capability field would fail the `"y"` case, and one that always returned
    the descriptor default would fail the `"x"` case.

    The third arm -- a request matching NO capability -- is today's behaviour,
    and it must stay today's behaviour: every existing `download_symbols()` call site
    reaches the descriptor default through this path.
    """
    descriptor = _resolver_descriptor()

    assert descriptor.symbol_batch_cls_for("us_equity", "1d", "x") is _A1
    assert descriptor.config_factory_for("us_equity", "1d", "x") is _f1

    assert descriptor.symbol_batch_cls_for("us_equity", "1d", "y") is _A0
    assert descriptor.config_factory_for("us_equity", "1d", "y") is _f0

    # No capability matches -> the vendor default, unchanged.
    assert descriptor.symbol_batch_cls_for("us_equity", "1m", None) is _A0
    assert descriptor.config_factory_for("us_equity", "1m", None) is _f0


def test_ambiguous_capability_resolution_refuses_rather_than_picking_by_order() -> None:
    """D-12 / T-03.10-41: two matches that DISAGREE raise, naming the request.

    Picking `matches[0]` would let capability declaration order -- an authoring
    detail invisible at the call site -- decide which vendor class receives the
    config, and so which credential is demanded. `convert()` already states
    this rule one axis over; the message is built the same way, from the
    descriptor's own data.

    The second half is the case that must NOT refuse: when both matches resolve
    to the same value there is no ambiguity to report, and raising would make a
    legitimate `data_type=None` request fail for a vendor that happens to
    serve two shapes through one class.
    """
    from quantlab.acquisition.base import Capability

    descriptor = _resolver_descriptor()

    for resolve in (descriptor.symbol_batch_cls_for, descriptor.config_factory_for):
        with pytest.raises(ValueError) as excinfo:
            resolve("us_equity", "1d", None)
        message = str(excinfo.value)
        assert "x" in message
        assert "y" in message
        assert descriptor.display_name in message
        assert "data_type" in message

    agreeing = _resolver_descriptor(
        capabilities=(
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="x",
                symbol_batch_cls=_A0,
                config_factory=_f0,
            ),
            Capability(market="us_equity", frequency="1d", data_type="y"),
        )
    )
    assert agreeing.symbol_batch_cls_for("us_equity", "1d", None) is _A0
    assert agreeing.config_factory_for("us_equity", "1d", None) is _f0


def test_run_constructs_the_capabilitys_acquisition_class(acquisition_config) -> None:
    """D-12: `download_symbols()` resolves through the capability, not
    `descriptor.symbol_batch_cls`.

    The resolver being correct proves nothing on its own -- the failure this
    guards against is a resolver that exists and is never called, which is
    exactly how plan 02's CRSP request would silently construct the TAQ class
    and demand a TAQ entitlement.

    The key comes off the config: `(market, frequency, kwargs["data_type"])`,
    which is `Capability`'s own key. Asserted on the RECORDED construction, so
    a `download_symbols()` that resolved correctly and then constructed the default anyway
    still fails here.
    """
    from quantlab.acquisition.registry import download_symbols

    _A0.constructed.clear()
    _A1.constructed.clear()

    descriptor = _resolver_descriptor()
    config = acquisition_config(vendor="tiingo", kwargs={"data_type": "x"})

    result = download_symbols(descriptor, config)

    assert result.vendor == "tiingo"
    assert len(_A1.constructed) == 1
    assert _A1.constructed[0] is config
    assert _A0.constructed == []


def test_capability_defaults_leave_the_new_fields_none() -> None:
    """D-12: `Capability` is still constructible from market + frequency alone,
    and the two new fields default to `None`.

    `None` means "the descriptor's default", never "no class" -- which is what
    makes the change ADDITIVE: every capability declared before this plan keeps
    resolving exactly as it did. Asserted on a bare construction rather than on
    a shipped capability, so a shipped descriptor that starts filling the
    fields cannot mask a lost default.

    The field ORDER is pinned too: the new fields come after `dataset_cls`, so
    no positional construction of an existing capability changes meaning.
    """
    import dataclasses

    from quantlab.acquisition.base import Capability

    capability = Capability(market="us_equity", frequency="1d")

    assert capability.symbol_batch_cls is None
    assert capability.config_factory is None
    assert capability.data_type is None
    assert capability.dataset_cls is None

    names = [field.name for field in dataclasses.fields(Capability)]
    assert names.index("symbol_batch_cls") > names.index("dataset_cls")
    assert names.index("config_factory") > names.index("symbol_batch_cls")


def test_every_registered_capability_resolves_to_its_own_class_or_the_default() -> None:
    """The INVARIANT, over the LIVE registry: for every descriptor and every
    capability it declares, resolving that capability's own triple returns
    `capability.<field> or descriptor.<field>`.

    This is the arm that keeps the resolver honest as vendors are added: a new
    capability whose class disagrees with what its own triple resolves to is a
    registry an operator surface cannot reason about. It also catches the
    ambiguity case for free -- two capabilities sharing a triple with different
    classes make their own lookup raise, and the raise is not caught here.
    """
    from quantlab.acquisition.base import DataSourceRegistry

    for descriptor in DataSourceRegistry.all():
        for capability in descriptor.capabilities:
            key = (capability.market, capability.frequency, capability.data_type)

            assert descriptor.symbol_batch_cls_for(*key) is (
                capability.symbol_batch_cls or descriptor.symbol_batch_cls
            ), (descriptor.vendor, key)
            assert descriptor.config_factory_for(*key) == (
                capability.config_factory or descriptor.config_factory
            ), (descriptor.vendor, key)


# ---------------------------------------------------------------------------
# LAST in file order, deliberately: the teardown half of every isolated test
# ---------------------------------------------------------------------------


def test_the_isolated_registry_fixture_restored_every_fake_vendor() -> None:
    """`isolated_registry` really RESTORES, which no test using it can assert.

    A fixture that snapshots but never restores looks identical from inside the
    test that used it -- and a leak would surface much later as an
    inexplicable third row in some other plan's enumeration assertion, with
    nothing pointing back here. pytest runs tests in file-definition order, so
    this sits last and names every fake vendor this module registers.

    Add a fake vendor above without adding its token here and this test
    silently stops covering it, so the list is the maintenance obligation that
    comes with `_fake_descriptor`.
    """
    from quantlab.acquisition.base import DataSourceRegistry

    assert [d.vendor for d in DataSourceRegistry.all()] == [
        "alpaca", "fred", "massive", "sharadar", "tiingo", "wrds",
    ]

    registered = {d.vendor for d in DataSourceRegistry.SOURCES}
    for leaked in (
        "fakevendor",
        "tiingotwin",
        "decoratedvendor",
        "nocredvendor",
        "only",
    ):
        assert leaked not in registered, leaked
