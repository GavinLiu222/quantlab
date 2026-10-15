"""The `.first()` adjustment anchor: the four invariants that pin the switch.

Phase 03.12 moves the CRSP adjustment anchor from "the LAST usable row inside
the conversion window" (forward adjustment) to "the PERMNO's FIRST usable row
inside the store" (backward adjustment). The anchor then never moves under an
append-only store, which is the whole point: a store that only grows forward
stops restating its own history.

Four invariants carry that claim, and this module is where they live:

- **I-1** the anchor row's `adjClose` IS its own `close`, and the same row's
  `adjVolume` IS its own `volume`;
- **I-2** a full build and an incremental build agree bit for bit (plan 02);
- **I-3** switching the anchor is a CONSTANT per-PERMNO rescale -- any seam
  (one column carrying two anchors) makes the ratio vary in `t`;
- **I-4** the day-over-day ratios are untouched by the switch -- only the
  scale changes, never the economics.

**Every quantlab import is INSIDE a test or helper body.** That is not style:
these tests are written before the behaviour they assert on exists, and a
module-scope import would turn the RED run into a collection error -- zero
tests discovered, which proves nothing about the behaviour (TDD gate #3770).

**Provenance rule, inherited from `tests/crsp_fixtures.py`.** A row reused from
that module is VERBATIM live data; a row built here with `dsf_row` is invented
and carries a `# SYNTHETIC` comment. Every synthetic row keeps `dsf_row`'s
NS/EQTY/COM defaults, so the common-stock filter keeps it.

**Tolerances are always explicit.** I-1 is `rel=1e-12` and MUST NOT be spelled
`==`: on the real store `close_A * G_A / G_A == close_A` holds exactly for only
514 of 520 securities (RESEARCH 19.4.5). I-3 / I-4 are `rel=1e-9`, against a
measured worst case of `4.900e-16`.
"""

from __future__ import annotations

from datetime import date

import pytest

#: A real PERMNO (Microsoft) used for every SYNTHETIC scenario, so the digits
#: in these tests are not a PERMNO that means something else.
SYNTHETIC_PERMNO = "10107"
SYNTHETIC_SYMBOL = "MSFT"

#: The SAME security, spelled the way the PANEL spells it: its symbol axis is
#: the int64 PERMNO (D-01, phase 03.11), never the ticker.
SYNTHETIC_AXIS = int(SYNTHETIC_PERMNO)


# ---------------------------------------------------------------------------
# Helpers: raw tier -> reference tier -> converted store
#
# REPLICATED, not imported, from `tests/test_crsp_dataset.py:66-210`. Every
# CRSP test module in this suite holds its OWN copy of this five-piece
# scaffold -- `tests/test_crsp_tracer.py:193-196` even defines its `at()` as a
# closure inside the test body -- and nothing imports a helper across test
# files. A cross-file helper import would make one module's fixture refactor
# silently re-point another module's assertions.
# ---------------------------------------------------------------------------


def _synthetic_secinfo():
    """A `stksecurityinfohist` interval for `SYNTHETIC_PERMNO`.

    SYNTHETIC: the live check never sampled 10107's security info. The shape
    is `secinfo_row`'s ordinary-common-share default, and the span is wide
    enough to label every scenario below.
    """
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(
            int(SYNTHETIC_PERMNO),
            "1986-03-13",
            "2025-12-31",
            SYNTHETIC_SYMBOL,
            SYNTHETIC_SYMBOL,
            None,
        )
    ]


def _reference_rows(extra_secinfo=()):
    """The fixture module's default reference rows plus `extra_secinfo`."""
    from tests.crsp_fixtures import (
        CCM_ROWS,
        DELISTS_ROWS,
        DISTRIBUTION_ROWS,
        DSP500_ROWS,
        IDXCST_ROWS,
        SECINFO_ROWS,
    )

    return {
        "crsp_a_stock.stksecurityinfohist": list(SECINFO_ROWS) + list(extra_secinfo),
        "crsp_a_stock.stkdelists": list(DELISTS_ROWS),
        "crsp_a_stock.stkdistributions": list(DISTRIBUTION_ROWS),
        "crsp_a_indexes.dsp500list_v2": list(DSP500_ROWS),
        "comp.idxcst_his": list(IDXCST_ROWS),
        "crsp_a_ccm.ccmxpf_lnkhist": list(CCM_ROWS),
    }


def _pull(
    tmp_path,
    rows,
    permnos,
    *,
    start,
    end,
    product_end="2025-12-31",
    extra_secinfo=(),
):
    """Serve `rows` through the fake session, land the raw + reference tiers."""
    from quantlab.acquisition.wrds.crsp import WrdsCrspDailyAcquisition
    from tests.crsp_fixtures import (
        FakeCrspSession,
        run_crsp_pull,
        write_reference_tables,
    )

    FakeCrspSession.daily_rows = list(rows)
    FakeCrspSession.product_end = date.fromisoformat(product_end)

    cfg, result = run_crsp_pull(tmp_path, permnos, start_date=start, end_date=end)
    assert result.failures == {}, result.failures

    reference_dir = WrdsCrspDailyAcquisition.reference_dir_for(cfg)
    write_reference_tables(
        reference_dir,
        _reference_rows(extra_secinfo),
        product_end=product_end,
    )
    return cfg, str(reference_dir)


def _dataset_config(tmp_path, cfg, reference_dir, *, start, end, store="crsp.zarr"):
    from quantlab.dataset.config import CrspDatasetConfig

    return CrspDatasetConfig(
        zarr_file_path=str(tmp_path / store),
        raw_data_dir_path=cfg.raw_data_dir_path,
        reference_dir=str(reference_dir),
        start_date=start,
        end_date=end,
    )


def _convert(dataset_config, granularity="year"):
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE

    return registry.convert(
        WRDS_SOURCE,
        dataset_config,
        data_type="crsp_daily",
        granularity=granularity,
    )


def _panel(dataset_config):
    import xarray as xr

    return xr.open_zarr(dataset_config.zarr_file_path).load()


def _build_store(
    tmp_path,
    rows,
    permnos,
    *,
    start,
    end,
    granularity="year",
    extra_secinfo=(),
    product_end="2025-12-31",
):
    """Pull, convert, and return the CONVERTED store's config."""
    cfg, reference_dir = _pull(
        tmp_path,
        rows,
        permnos,
        start=start,
        end=end,
        product_end=product_end,
        extra_secinfo=extra_secinfo,
    )
    dataset_config = _dataset_config(
        tmp_path, cfg, reference_dir, start=start, end=end
    )
    _convert(dataset_config, granularity=granularity)
    return dataset_config


def _build(tmp_path, rows, permnos, **kwargs):
    """`_build_store`, opened -- the common three-line preamble."""
    return _panel(_build_store(tmp_path, rows, permnos, **kwargs))


def _at(panel, variable, day, symbol):
    return float(panel[variable].sel(timestamp=day, symbol=symbol).values)


# ---------------------------------------------------------------------------
# The eleven-month synthetic series.
#
# REPLICATED from `tests/test_crsp_dataset.py:972-1035`, for the same reason
# the scaffold above is replicated: CRSP test modules do not import helpers
# from one another. It carries a 4:1 split and a dividend, so an anchor at the
# FIRST row and an anchor at the LAST row sit on different `dlycumfacshr`
# levels -- which is what makes I-3 and I-4 able to tell them apart at all.
# ---------------------------------------------------------------------------

ANCHOR_START = "2019-11-01"
ANCHOR_END = "2020-09-30"
#: The 4:1 split and the dividend inside the anchor window.
ANCHOR_SPLIT_DAY = "2020-08-31"
ANCHOR_DIVIDEND_DAY = "2020-08-07"
ANCHOR_DIVIDEND = 0.82


def _anchor_series_rows():
    """SYNTHETIC: ~11 months of daily rows with one 4:1 split and a dividend.

    Long enough that the panel carries more than 200 trading days, which is
    what lets I-3 and I-4 compare hundreds of cells rather than a handful.
    """
    import pandas as pd

    from tests.crsp_fixtures import dsf_row

    days = pd.bdate_range(ANCHOR_START, ANCHOR_END)
    rows = []
    value = 60.0
    previous_raw = None
    for index, day in enumerate(days):
        iso = day.date().isoformat()
        daily_return = round(0.02 - 0.035 * (index % 2) + 0.0005 * (index % 11), 6)
        value *= 1.0 + daily_return
        pre_split = 4.0 if iso < ANCHOR_SPLIT_DAY else 1.0
        raw = value * pre_split

        without_dividend = daily_return
        if iso == ANCHOR_DIVIDEND_DAY and previous_raw:
            without_dividend = round(daily_return - ANCHOR_DIVIDEND / previous_raw, 6)

        rows.append(
            dsf_row(  # SYNTHETIC: an invented ~11-month series, not a live row.
                SYNTHETIC_PERMNO,
                iso,
                dlyprc=f"{raw:.6f}",
                dlyclose=f"{raw:.6f}",
                dlyopen=f"{raw * 0.99:.6f}",
                dlyhigh=f"{raw * 1.02:.6f}",
                dlylow=f"{raw * 0.98:.6f}",
                dlyret=f"{daily_return:.6f}",
                dlyretx=f"{without_dividend:.6f}",
                dlyorddivamt=(
                    f"{ANCHOR_DIVIDEND:.6f}"
                    if iso == ANCHOR_DIVIDEND_DAY
                    else "0.000000"
                ),
                dlyfacprc="4.000000" if iso == ANCHOR_SPLIT_DAY else "1.000000",
                dlycumfacpr=(
                    "4.000000000000" if iso < ANCHOR_SPLIT_DAY else "1.000000000000"
                ),
                dlycumfacshr=(
                    "4.000000000000" if iso < ANCHOR_SPLIT_DAY else "1.000000000000"
                ),
            )
        )
        previous_raw = raw
    return rows


# ---------------------------------------------------------------------------
# I-1: the anchor is the FIRST usable row, and it publishes its own level
# ---------------------------------------------------------------------------


def test_the_anchor_is_first_usable_row_and_its_adjclose_equals_its_close(
    mock_crsp_session, tmp_path
):
    """I-1: the anchor skips the leading no-price day and IS its own level.

    The window's FIRST row is a Missing-Price day. A naive "first row" anchor
    would anchor the whole series on a null and make every adjusted value NaN,
    so the anchor must step forward to `2020-03-03` -- the first row carrying
    BOTH a strictly positive `close` and a non-null `dlycumfacshr`.

    On that row, and only on that row, the adjusted series is the raw series:
    `adjClose == close` and `adjVolume == volume`. That is the whole content
    of I-1, and it is what makes a backward-adjusted store readable -- the
    earliest priced day of every security is stated in its own historical
    dollars.

    The 4:1 split on `2020-03-04` is what gives this test its teeth. Without
    it the synthetic price series is return-consistent, `close_t / close_{t-1}`
    equals `1 + ret_t` everywhere, and the first-row and last-row anchors
    produce the SAME numbers -- a green test under either anchor, which would
    assert nothing. With the split, the two anchors sit on different
    `dlycumfacshr` levels (4.0 vs 1.0) and different raw closes (100.0 vs
    25.5), so every assertion below discriminates.

    `pytest.approx(..., rel=1e-12)`, never `==`: on the real store the exact
    identity holds for 514 of 520 securities, and a bare `==` would invite the
    next reader to loosen the tolerance to `approx()`'s 1e-6 default -- which
    would also hide a genuine 5e-06 drift (RESEARCH 19.4.5).
    """
    import numpy as np

    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: the window's FIRST row has no price.
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-02",
            dlyprc=None,
            dlyprcflg=None,
            dlyret=None,
            dlyretmissflg="MP",
            dlycumfacpr="4.000000000000",
            dlycumfacshr="4.000000000000",
        ),
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-03",
            dlyprc="100.000000",
            dlyret="0.010000",
            dlycumfacpr="4.000000000000",
            dlycumfacshr="4.000000000000",
        ),
        # SYNTHETIC: the 4:1 split lands here, so the raw close drops to
        # 102.0 / 4 while `dlyret` still states the +2% total return.
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-04",
            dlyprc="25.500000",
            dlyret="0.020000",
            dlyfacprc="4.000000",
            dlycumfacpr="1.000000000000",
            dlycumfacshr="1.000000000000",
        ),
    ]
    panel = _build(
        tmp_path,
        rows,
        [SYNTHETIC_PERMNO],
        start="2020-03-01",
        end="2020-03-31",
        extra_secinfo=_synthetic_secinfo(),
    )

    # The anchor row publishes ITSELF: its adjusted close is its close and its
    # adjusted volume is its volume.
    assert _at(panel, "adjClose", "2020-03-03", SYNTHETIC_AXIS) == pytest.approx(
        _at(panel, "close", "2020-03-03", SYNTHETIC_AXIS), rel=1e-12
    )
    assert _at(panel, "adjClose", "2020-03-03", SYNTHETIC_AXIS) == pytest.approx(
        100.0, rel=1e-12
    )
    assert _at(panel, "adjVolume", "2020-03-03", SYNTHETIC_AXIS) == pytest.approx(
        _at(panel, "volume", "2020-03-03", SYNTHETIC_AXIS), rel=1e-12
    )
    assert _at(panel, "adjVolume", "2020-03-03", SYNTHETIC_AXIS) == pytest.approx(
        1_000_000.0, rel=1e-12
    )

    # The day AFTER the anchor is scaled UP by `G_t / G_A = 1.02`, which is
    # 102.0 -- and is NOT its own raw close (25.5) any more, because the split
    # sits between it and the anchor.
    assert _at(panel, "adjClose", "2020-03-04", SYNTHETIC_AXIS) == pytest.approx(
        102.0, rel=1e-12
    )
    assert _at(panel, "close", "2020-03-04", SYNTHETIC_AXIS) == pytest.approx(
        25.5, rel=1e-12
    )
    # Volume rides `dlycumfacshr / cumfacshr_A` = 1.0 / 4.0.
    assert _at(panel, "adjVolume", "2020-03-04", SYNTHETIC_AXIS) == pytest.approx(
        250_000.0, rel=1e-12
    )

    # The no-price day stays NaN. There was no price that day, and publishing
    # the anchor's level there would invent one -- the mask is decided by
    # `close`, and has nothing to do with which row the anchor is.
    assert np.isnan(_at(panel, "adjClose", "2020-03-02", SYNTHETIC_AXIS))


def test_an_eleven_month_series_converts_without_an_unusable_anchor_refusal(
    mock_crsp_session, tmp_path
):
    """`_assert_anchor_usable` (D-05) does not become a routine refusal.

    The `.first()` switch reverses the direction the guard looks, so the risk
    it was written for -- a PERMNO whose qualifying row falls outside the
    window -- now lands on the EARLY end rather than the late one. An ordinary
    eleven-month series must still convert with no `ValueError` at all.
    """
    panel = _build(
        tmp_path,
        _anchor_series_rows(),
        [SYNTHETIC_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo(),
    )

    assert panel.sizes["timestamp"] > 200, panel.sizes
    assert panel["symbol"].values.tolist() == [SYNTHETIC_AXIS]


# ---------------------------------------------------------------------------
# I-3 / I-4: the switch is a constant rescale, and the returns are untouched
# ---------------------------------------------------------------------------


def _adjclose_anchored_last(panel, symbol):
    """Recompute `adjClose` under the OLD `.last()` anchor, from raw columns.

    A numpy replica of `quantlab/dataset/crsp/__init__.py:571-640` with the
    reduction pointed at the LAST qualifying row instead of the first. It is
    the control series both I-3 and I-4 compare against, and it is built from
    the panel's OWN `close` / `ret` / `cumfacshr` -- never from `adjClose`,
    which is the quantity under test.

    Two things it deliberately does NOT do:

    - it does not re-derive the price by testing the RAW price column for
      positivity. The panel's `close` has already been through the
      no-price-sentinel guard, so reading it is safe; re-deriving from the
      raw price column would route around that guard and put CR-01 back
      (D-05 / R-02).
    - it does not chunk the cumulative product. `np.cumprod` in one call
      keeps the floating-point association order the polars `cum_prod`
      used; a hand-rolled blocked reduction would change it and make the
      `rel=1e-9` comparisons below meaningless.

    The three anchor quantities come from ONE index, exactly as the single
    `group_by().agg()` in production does -- selecting on `close` and then
    reading `cumfacshr` off a different row is CR-02.
    """
    import numpy as np

    series = panel.sortby("timestamp").sel(symbol=symbol)
    close = np.asarray(series["close"].values, dtype=float)
    ret = np.asarray(series["ret"].values, dtype=float)
    cumfacshr = np.asarray(series["cumfacshr"].values, dtype=float)

    growth = np.cumprod(1.0 + np.nan_to_num(ret, nan=0.0))

    usable = ~np.isnan(close) & (close > 0.0) & ~np.isnan(cumfacshr)
    indices = np.flatnonzero(usable)
    assert indices.size > 0, "no usable anchor row in the control series"
    index = int(indices[-1])

    return np.where(
        np.isnan(close),
        np.nan,
        close[index] * growth / growth[index],
    )


def _last_anchor_index(panel, symbol):
    """The row `_adjclose_anchored_last` anchored on -- for its self-check."""
    import numpy as np

    series = panel.sortby("timestamp").sel(symbol=symbol)
    close = np.asarray(series["close"].values, dtype=float)
    cumfacshr = np.asarray(series["cumfacshr"].values, dtype=float)
    usable = ~np.isnan(close) & (close > 0.0) & ~np.isnan(cumfacshr)
    return int(np.flatnonzero(usable)[-1])


def test_switching_the_anchor_is_a_constant_rescale_per_permno(
    mock_crsp_session, tmp_path
):
    """I-3: `adjClose^first_t / adjClose^last_t` is CONSTANT in `t`.

    That is the whole claim of the switch: moving the anchor multiplies a
    PERMNO's entire adjusted history by one number and changes nothing else.
    The ratio is what catches the failure that would matter -- a SEAM. If any
    part of the column were written under one anchor and the rest under
    another (a chunk boundary, an in-place append, a per-window anchor), the
    ratio would take one value on one side of the seam and another value on
    the other, and `max == min` would fail. A per-cell tolerance check would
    not: each side is individually plausible.

    `rel=1e-9` against a measured worst case of `4.900e-16` on the real store.
    """
    import numpy as np

    panel = _build(
        tmp_path,
        _anchor_series_rows(),
        [SYNTHETIC_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo(),
    )

    for symbol in panel["symbol"].values.tolist():
        control = _adjclose_anchored_last(panel, symbol)
        current = np.asarray(
            panel.sortby("timestamp")["adjClose"].sel(symbol=symbol).values,
            dtype=float,
        )

        # The control series is genuinely anchored on the LAST usable row:
        # there, and only there, it equals that row's own close. Without this
        # self-check the test could pass while `_adjclose_anchored_last` had
        # quietly copied the store's own numbers.
        index = _last_anchor_index(panel, symbol)
        close = np.asarray(
            panel.sortby("timestamp")["close"].sel(symbol=symbol).values,
            dtype=float,
        )
        assert control[index] == pytest.approx(close[index], rel=1e-12)

        ratio = current / control
        finite = np.isfinite(ratio)
        assert finite.sum() > 200, finite.sum()
        assert np.nanmax(ratio[finite]) == pytest.approx(
            np.nanmin(ratio[finite]), rel=1e-9
        )


def test_the_day_over_day_ratios_unchanged_by_the_anchor_switch(
    mock_crsp_session, tmp_path
):
    """I-4: the RETURN series is bit-for-bit the same series it always was.

    I-3 says the switch is a rescale; I-4 says the rescale is the ONLY thing
    it is. `adjClose_t / adjClose_{t-1}` is what every factor, label and
    backtest actually consumes, and it must not move: the store changes its
    units, not its economics. The real store measures `max rel 4.900e-16`
    across 129,756 cells, so `rel=1e-9` is six orders of magnitude of slack
    over the noise floor and still far tighter than any real drift.
    """
    import numpy as np

    panel = _build(
        tmp_path,
        _anchor_series_rows(),
        [SYNTHETIC_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo(),
    )

    compared = 0
    for symbol in panel["symbol"].values.tolist():
        control = _adjclose_anchored_last(panel, symbol)
        current = np.asarray(
            panel.sortby("timestamp")["adjClose"].sel(symbol=symbol).values,
            dtype=float,
        )

        current_steps = current[1:] / current[:-1]
        control_steps = control[1:] / control[:-1]

        both = np.isfinite(current_steps) & np.isfinite(control_steps)
        assert both.sum() > 200, both.sum()
        compared += int(both.sum())

        assert current_steps[both] == pytest.approx(
            control_steps[both], rel=1e-9
        )

        relative = np.abs(
            current_steps[both] - control_steps[both]
        ) / np.abs(control_steps[both])
        assert float(relative.max()) <= 1e-9, (symbol, float(relative.max()))

    assert compared > 200, compared


# ---------------------------------------------------------------------------
# I-2: a full build and an incremental build agree BIT FOR BIT
# ---------------------------------------------------------------------------


def test_a_full_and_incremental_build_agree_bit_for_bit(mock_crsp_session, tmp_path):
    """I-2, the acceptance anchor D-10 names: appending forward restates nothing.

    Two stores over the SAME raw tier and the SAME `start_date`. One is built
    in a single conversion across `[ANCHOR_START, ANCHOR_END]`. The other is
    built across `[ANCHOR_START, 2019-12-31]` first and then brought up to
    `ANCHOR_END` by `RawDataset.update()` -- a real second RUN, against a
    store that already exists on disk, through the append path the chunk ledger
    drives.

    **How this differs from its sister test.**
    `tests/test_crsp_dataset.py:test_year_and_month_granularity_produce_identical_stores`
    compares two CHUNKINGS of ONE run, so it can only prove the anchor is not
    recomputed per window inside a single conversion. It says nothing about a
    second process, on a second day, appending to a finished store. I-2 pins
    exactly that: the values written by run 1 are still the values on disk
    after run 2 (no historical rewrite), and the join between the two runs
    carries no fabricated return (no seam).

    `xr.testing.assert_identical`, never a tolerance-based comparison: the
    claim is bit-for-bit, and it covers variable NAMES and COORDINATES too. Any
    tolerance here would let exactly the drift this test exists to catch pass.

    Before 03.12 this scenario RAISED. The anchor was each PERMNO's last
    priced day, so extending `end_date` moved it, and a cross-run gate compared
    a recorded window quadruple and REFUSED the second run outright. Under the
    first-usable-row anchor the anchor does not move when the window only grows
    forward, so the refusal became wrong and the machinery that carried it is
    gone; this test is what replaced it.

    **What it does NOT cover.** Only the forward-growth shape is pinned here.
    Moving `start_date` later, and back-filling the raw tier underneath a fixed
    `start_date`, both DO rewrite history, are both measured, and are checked
    by nothing -- see the KNOWN LIMITATIONS section of
    `CrspStockDataset._derivation()`.
    """
    import json
    from pathlib import Path

    import xarray as xr

    from quantlab.dataset.crsp import CrspStockDataset

    cfg, reference_dir = _pull(
        tmp_path,
        _anchor_series_rows(),
        [SYNTHETIC_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo(),
    )

    # Store A: the whole window in one conversion.
    full_config = _dataset_config(
        tmp_path,
        cfg,
        reference_dir,
        start=ANCHOR_START,
        end=ANCHOR_END,
        store="crsp_full.zarr",
    )
    _convert(full_config, granularity="year")
    panel_full = _panel(full_config)

    # Store B, run 1: stop at the year boundary, which is also a window
    # boundary for `granularity="year"` -- so run 2 appends a whole window
    # rather than reopening a half-written one.
    partial_config = _dataset_config(
        tmp_path,
        cfg,
        reference_dir,
        start=ANCHOR_START,
        end="2019-12-31",
        store="crsp_incremental.zarr",
    )
    _convert(partial_config, granularity="year")

    # Store B, run 2. A NEW config object, not a mutated one: the `config`
    # setter is what clears `_derivation_cache`, so a field poked onto the old
    # config after construction would re-use run 1's derivation and prove
    # nothing about a second run.
    extended_config = _dataset_config(
        tmp_path,
        cfg,
        reference_dir,
        start=ANCHOR_START,
        end=ANCHOR_END,
        store="crsp_incremental.zarr",
    )
    CrspStockDataset(extended_config).update(granularity="year")
    panel_incremental = _panel(extended_config)

    xr.testing.assert_identical(panel_full, panel_incremental)

    # Non-emptiness. Two empty panels are `assert_identical` too, so the
    # comparison above is only worth something once the panel is known to hold
    # the eleven months of trading days the fixture builds.
    assert panel_full.sizes["timestamp"] > 200, panel_full.sizes

    # Incrementality. Store B must have been APPENDED to, not rewritten whole:
    # its ledger records the window run 1 wrote AND the window run 2 added.
    ledger = json.loads(
        Path(str(extended_config.zarr_file_path) + ".chunks.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(ledger["windows"]) >= 2, ledger
