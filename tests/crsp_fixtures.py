"""Offline stand-ins for WRDS CRSP Stock v2 daily data (phase 03.10).

Nothing here touches the network. `FakeCrspSession` subclasses
`tests.wrds_fixtures.FakeWrdsSession` and implements the PROVIDER-NEUTRAL
session surface plan 03.10-01 added to the real
`quantlab.acquisition.wrds.taq.WrdsSession` -- `schema_usable`, `fetch_rows`
and `copy_csv`. `tests/conftest.py:mock_crsp_session` patches it over the same
dotted target `mock_wrds_session` uses, so the autouse `_forbid_wrds_network`
tripwire stays live underneath: a fake that failed to install would fail the
test instead of pushing a Duo prompt (D-13).

**Provenance rule for every value in this module.** A row transcribed from a
live check is VERBATIM and names the JSON key it came from; a value invented
for the test carries a `# SYNTHETIC` comment on the spot. The live sources are:

- `03.10-LIVE-CHECK.json` -- `C3_columns` (server column lists and their
  PostgreSQL types), `C4_sample_crsp_a_stock.dsf_v2` (AAPL 1980),
  `C4_sample_crsp_a_stock.stksecurityinfohist` + `C5_ticker_hist_...` (ticker
  history), `C4_sample_crsp_a_stock.stkdistributions` (AAPL 1987),
  `C4_sample_crsp_a_indexes.dsp500list_v2`;
- `03.10-LIVE-CHECK-2.json` -- `L3_1`/`L3_2`/`L3_3` (Lehman 2008), `L4_1`
  (the AAPL 2020 split and dividend rows), `L6_1` (the BF share classes),
  `L7_4` (the Alphabet CCM links), `L8_2` (the Alphabet index spells);
- `03.10-LIVE-CHECK-NDX-QQQ.json` -- `C1_qqq_names`, `C3_qqq_daily_sample`.

The fake renders the REAL composed SQL through
`tests.wrds_fixtures.render_composed` and then parses it back, so a query this
module answers is a query the production builders actually emit -- a test that
passes here cannot pass against SQL the provider does not issue.
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import polars as pl

from tests.wrds_fixtures import FakeWrdsSession, render_composed

# ---------------------------------------------------------------------------
# Server column lists (VERBATIM, `03.10-LIVE-CHECK.json` key `C3_columns`)
# ---------------------------------------------------------------------------

#: The 50 `crsp_a_stock.dsf_v2` columns, in SERVER order.
DSF_V2_SERVER_COLUMNS: tuple[str, ...] = (
    "permno",
    "hdrcusip",
    "permco",
    "siccd",
    "nasdissuno",
    "yyyymmdd",
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
    "primaryexch",
    "conditionaltype",
    "tradingstatusflg",
    "dlycaldt",
    "dlydelflg",
    "dlyprc",
    "dlyprcflg",
    "dlycap",
    "dlycapflg",
    "dlyprevprc",
    "dlyprevprcflg",
    "dlyprevdt",
    "dlyprevcap",
    "dlyprevcapflg",
    "dlyret",
    "dlyretx",
    "dlyreti",
    "dlyretmissflg",
    "dlyretdurflg",
    "dlyorddivamt",
    "dlynonorddivamt",
    "dlyfacprc",
    "dlydistretflg",
    "dlyvol",
    "dlyclose",
    "dlylow",
    "dlyhigh",
    "dlybid",
    "dlyask",
    "dlyopen",
    "dlynumtrd",
    "dlymmcnt",
    "dlyprcvol",
    "dlycumfacpr",
    "dlycumfacshr",
    "cusip",
    "ticker",
    "exchangetier",
    "shrout",
)

#: The 36 `crsp_a_stock.stksecurityinfohist` columns, in SERVER order.
SECINFO_SERVER_COLUMNS: tuple[str, ...] = (
    "permno",
    "secinfostartdt",
    "secinfoenddt",
    "securitybegdt",
    "securityenddt",
    "securityhdrflg",
    "hdrcusip",
    "hdrcusip9",
    "cusip",
    "cusip9",
    "primaryexch",
    "conditionaltype",
    "exchangetier",
    "tradingstatusflg",
    "securitynm",
    "shareclass",
    "usincflg",
    "issuertype",
    "securitytype",
    "securitysubtype",
    "sharetype",
    "securityactiveflg",
    "delactiontype",
    "delstatustype",
    "delreasontype",
    "delpaymenttype",
    "ticker",
    "tradingsymbol",
    "permco",
    "siccd",
    "naics",
    "icbindustry",
    "uesindustry",
    "nasdcompno",
    "nasdissuno",
    "issuernm",
)


def _row(columns: tuple[str, ...], values) -> dict[str, str | None]:
    """A live-check `rows` entry as the `{column: text-or-NULL}` dict a COPY
    would carry.

    The live-check writer rendered a SQL NULL as the literal string `"None"`
    (it `str()`-ed the driver's `None`), so that token is mapped back to
    `None` here -- otherwise every null would arrive downstream as the
    four-character string `None` and cast to a non-null value.
    """
    return {
        name: (None if value in (None, "None") else str(value))
        for name, value in zip(columns, values)
    }


# ---------------------------------------------------------------------------
# Daily rows
# ---------------------------------------------------------------------------


def dsf_row(permno, day, **overrides) -> dict[str, str | None]:
    """One `dsf_v2` record for `permno` on `day`, as COPY CSV would carry it.

    Every one of the 50 columns is present; a value is a string or `None`
    (NULL). The defaults are the ORDINARY-DAY shape read off the live rows:
    an active, US-incorporated common share on Nasdaq, a closing-trade price,
    no missing return, no distribution, and unit price/share factors.

    `dlyvol` defaults to 1,000,000.  # SYNTHETIC -- the L4_1 live projection
    did not select `dlyvol`, and the tracer needs a volume to multiply by
    `dlycumfacshr`. Every other default below is a value observed verbatim on
    at least one live row.
    """
    day = date.fromisoformat(str(day)[:10])
    record: dict[str, str | None] = {name: None for name in DSF_V2_SERVER_COLUMNS}
    record.update(
        {
            "permno": str(int(permno)),
            "dlycaldt": day.isoformat(),
            "yyyymmdd": f"{day:%Y%m%d}",
            "sharetype": "NS",
            "securitytype": "EQTY",
            "securitysubtype": "COM",
            "usincflg": "Y",
            "issuertype": "CORP",
            "primaryexch": "Q",
            "conditionaltype": "RW",
            "tradingstatusflg": "A",
            "dlydelflg": "N",
            "dlyprcflg": "TR",
            "dlyretmissflg": "NA",
            "dlyretdurflg": "D1",
            "dlydistretflg": "NO",
            "dlyfacprc": "1.000000",
            "dlycumfacpr": "1.000000000000",
            "dlycumfacshr": "1.000000000000",
            "dlyorddivamt": "0.000000",
            "dlynonorddivamt": "0.000000",
            "dlyvol": "1000000",  # SYNTHETIC
        }
    )
    for name, value in overrides.items():
        if name not in record:
            raise KeyError(f"dsf_row: {name!r} is not a dsf_v2 column")
        record[name] = None if value is None else str(value)
    return record


#: AAPL (PERMNO 14593) around the 2020-08-31 4:1 split.
#:
#: VERBATIM `03.10-LIVE-CHECK-2.json` key `L4_1`, which selected
#: `dlycaldt, dlyprc, dlyret, dlyretx, dlyfacprc, dlycumfacpr, dlycumfacshr,
#: dlyorddivamt, dlydistretflg`. The other 41 columns come from `dsf_row`'s
#: ordinary-day defaults, `dlyvol` included (SYNTHETIC, see `dsf_row`).
AAPL_AUG_2020_ROWS: list[dict[str, str | None]] = [
    dsf_row(
        14593,
        "2020-08-06",
        dlyprc="455.610000",
        dlyret="0.034889",
        dlyretx="0.034889",
        dlyfacprc="1.000000",
        dlycumfacpr="4.000000000000",
        dlycumfacshr="4.000000000000",
        dlyorddivamt="0.000000",
        dlydistretflg="NO",
        ticker="AAPL",
    ),
    dsf_row(
        14593,
        "2020-08-07",
        dlyprc="444.450000",
        dlyret="-0.022695",
        dlyretx="-0.024495",
        dlyfacprc="1.000000",
        dlycumfacpr="4.000000000000",
        dlycumfacshr="4.000000000000",
        dlyorddivamt="0.820000",
        dlydistretflg="C1",
        ticker="AAPL",
    ),
    dsf_row(
        14593,
        "2020-08-28",
        dlyprc="499.230000",
        dlyret="-0.001620",
        dlyretx="-0.001620",
        dlyfacprc="1.000000",
        dlycumfacpr="4.000000000000",
        dlycumfacshr="4.000000000000",
        dlyorddivamt="0.000000",
        dlydistretflg="NO",
        ticker="AAPL",
    ),
    dsf_row(
        14593,
        "2020-08-31",
        dlyprc="129.040000",
        dlyret="0.033912",
        dlyretx="0.033912",
        dlyfacprc="4.000000",
        dlycumfacpr="1.000000000000",
        dlycumfacshr="1.000000000000",
        dlyorddivamt="0.000000",
        dlydistretflg="S1",
        ticker="AAPL",
    ),
]

#: Lehman Brothers (PERMNO 80599) through its 2008 delisting.
#:
#: VERBATIM `03.10-LIVE-CHECK-2.json` key `L3_1`, which selected
#: `permno, dlycaldt, dlydelflg, dlyprc, dlyprcflg, dlyret, dlyretmissflg,
#: ticker`. Note the last row: `dlydelflg='Y'`, price flag `DP`, and a NULL
#: ticker -- the D-19 fact that a delisting return is its own daily row whose
#: symbol must be carried forward.
LEHMAN_2008_ROWS: list[dict[str, str | None]] = [
    dsf_row(
        80599, "2008-09-12", dlydelflg="N", dlyprc="3.650000", dlyprcflg="TR",
        dlyret="-0.135071", dlyretmissflg="NA", ticker="LEH",
    ),
    dsf_row(
        80599, "2008-09-15", dlydelflg="N", dlyprc="0.210000", dlyprcflg="TR",
        dlyret="-0.942466", dlyretmissflg="NA", ticker="LEH",
    ),
    dsf_row(
        80599, "2008-09-16", dlydelflg="N", dlyprc="0.300000", dlyprcflg="TR",
        dlyret="0.428571", dlyretmissflg="NA", ticker="LEH",
    ),
    dsf_row(
        80599, "2008-09-17", dlydelflg="N", dlyprc="0.130000", dlyprcflg="TR",
        dlyret="-0.566667", dlyretmissflg="NA", ticker="LEH",
    ),
    dsf_row(
        80599, "2008-09-18", dlydelflg="Y", dlyprc="0.052000", dlyprcflg="DP",
        dlyret="-0.600000", dlyretmissflg="NA", ticker=None,
    ),
]

#: WestRock (PERMNO 21186) through its 2024-07-08 delisting -- the MODERN CIZ
#: delisting shape, which is a DIFFERENT shape from Lehman's above.
#:
#: The two delisting shapes, plainly:
#:
#: - `dlyprcflg='DP'` (delisting PRICE) carries a REAL price. Lehman 2008-09-18
#:   is `dlyprc = 0.052`, an actual value a holding was worth. Its factor and
#:   volume columns are ordinary.
#: - `dlyprcflg='DA'` (delisting AMOUNT) carries NO price. CRSP writes
#:   `dlyprc = 0.000000` there as a NO-PRICE SENTINEL, and leaves `dlyclose`,
#:   `dlyvol`, `dlycumfacpr` and `dlycumfacshr` NULL. Reading that 0.0 as a
#:   close both publishes a fabricated $0.00 trade and -- because 0.0 is not
#:   NULL -- lets the sentinel row become the adjustment anchor.
#:
#: `DA` is **5 of 5** delisting rows in the raw tier this phase actually pulled
#: (flag distribution `TR` 138,888 / `DA` 5 / `DP` 0). The corpus carried only
#: the `DP` shape, which is why 179 CRSP tests passed green over two blockers.
#:
#: VERIFIED against that raw tier: the three rows below are the values
#: 03.10-REVIEW.md CR-01 reproduced read-only from
#: `data/downloads/us_equity/1d/wrds_crsp/wrds/` for PERMNO 21186 -- `dlycaldt`,
#: `dlyprc`, `dlyprcflg`, `dlyret`, `dlycumfacpr`, `dlycumfacshr` and `dlyvol`,
#: `dlyvol` included (so these are LIVE volumes, not `dsf_row`'s SYNTHETIC
#: default). `dlyopen` is the same table's open on those two days. The NULL type
#: columns on the delisting row are live too, and they are not decoration: they
#: are what makes that row exercise the D-10 filter carry.
WESTROCK_2024_ROWS: list[dict[str, str | None]] = [
    dsf_row(
        21186, "2024-07-03", dlydelflg="N", dlyprc="49.750000", dlyprcflg="TR",
        dlyret="0.019676", dlyretmissflg="NA", dlyopen="49.550000",
        dlyvol="4435075", dlycumfacpr="1.000000000000",
        dlycumfacshr="1.000000000000", ticker="WRK",
    ),
    dsf_row(
        21186, "2024-07-05", dlydelflg="N", dlyprc="51.510000", dlyprcflg="TR",
        dlyret="0.035377", dlyretmissflg="NA", dlyopen="50.780000",
        dlyvol="11862010", dlycumfacpr="1.000000000000",
        dlycumfacshr="1.000000000000", ticker="WRK",
    ),
    # The modern delisting row: a no-price sentinel, not a trade.
    dsf_row(
        21186, "2024-07-08", dlydelflg="Y", dlyprc="0.000000", dlyprcflg="DA",
        dlyret="-0.005630", dlyretmissflg="NA", dlyclose=None, dlyvol=None,
        dlyopen=None, dlycumfacpr=None, dlycumfacshr=None, ticker=None,
        sharetype=None, securitytype=None, securitysubtype=None,
        usincflg=None, issuertype=None,
    ),
]

#: AAPL's first three trading days, all 50 columns VERBATIM from
#: `03.10-LIVE-CHECK.json` key `C4_sample_crsp_a_stock.dsf_v2`. The pre-1992
#: era has no OHLC and no volume, which is why every such field is NULL here.
AAPL_1980_ROWS: list[dict[str, str | None]] = [
    _row(DSF_V2_SERVER_COLUMNS, values)
    for values in (
        [
            "14593", "03783310", "7", "3663", "8", "19801212", "NS", "EQTY",
            "COM", "Y", "CORP", "Q", "RW", "A", "1980-12-12", "N",
            "28.812500", "BA", "1588606.00", "BP", "None", "NS", "None",
            "None", "MP", "None", "None", "None", "NS", "MR", "0.000000",
            "0.000000", "1.000000", "NO", "None", "None", "None", "None",
            "28.750000", "28.875000", "None", "None", "None", "None",
            "224.000000000000", "224.000000000000", "03783310", "AAPL",
            "N/A", "55136",
        ],
        [
            "14593", "03783310", "7", "3663", "8", "19801215", "NS", "EQTY",
            "COM", "Y", "CORP", "Q", "RW", "A", "1980-12-15", "N",
            "27.312500", "BA", "1505902.00", "BP", "28.812500", "BA",
            "1980-12-12", "1588606.00", "PB", "-0.052061", "-0.052061",
            "0.000000", "NA", "D3", "0.000000", "0.000000", "1.000000", "NO",
            "None", "None", "None", "None", "27.250000", "27.375000", "None",
            "None", "None", "None", "224.000000000000", "224.000000000000",
            "03783310", "AAPL", "N/A", "55136",
        ],
        [
            "14593", "03783310", "7", "3663", "8", "19801216", "NS", "EQTY",
            "COM", "Y", "CORP", "Q", "RW", "A", "1980-12-16", "N",
            "25.312500", "BA", "1395630.00", "BP", "27.312500", "BA",
            "1980-12-15", "1505902.00", "PB", "-0.073227", "-0.073227",
            "0.000000", "NA", "D1", "0.000000", "0.000000", "1.000000", "NO",
            "None", "None", "None", "None", "25.250000", "25.375000", "None",
            "None", "None", "None", "224.000000000000", "224.000000000000",
            "03783310", "AAPL", "N/A", "55136",
        ],
    )
]

#: QQQ (PERMNO 86755), all 50 columns VERBATIM from
#: `03.10-LIVE-CHECK-NDX-QQQ.json` key `C3_qqq_daily_sample`. An ETF, so
#: `securitytype='FUND'` / `securitysubtype='ETF'` -- the shape D-06's
#: common-stock filter would exclude if it were applied unconditionally.
QQQ_ROWS: list[dict[str, str | None]] = [
    _row(DSF_V2_SERVER_COLUMNS, values)
    for values in (
        [
            "86755", "46090E10", "35013", "6726", "22669", "19990310", "NS",
            "FUND", "ETF", "Y", "ACOR", "A", "RW", "A", "1999-03-10", "N",
            "102.125000", "TR", "6198987.50", "BP", "None", "NS", "None",
            "None", "MP", "None", "None", "None", "NS", "MR", "0.000000",
            "0.000000", "1.000000", "NO", "2616100", "102.125000",
            "100.562500", "102.312500", "None", "None", "None", "None",
            "None", "267169212.5", "2.000000000000", "2.000000000000",
            "63110010", "QQQ", "N/A", "60700",
        ],
        [
            "86755", "46090E10", "35013", "6726", "22669", "20100601", "NS",
            "FUND", "ETF", "Y", "ACOR", "Q", "RW", "A", "2010-06-01", "N",
            "45.180000", "TR", "17744445.00", "BP", "45.600000", "TR",
            "2010-05-28", "18413280.00", "PB", "-0.009211", "-0.009211",
            "0.000000", "NA", "D4", "0.000000", "0.000000", "1.000000", "NO",
            "104509340", "45.180000", "45.130000", "46.250000", "45.170000",
            "45.180000", "45.450000", "187121", "66", "4721731981.2",
            "1.000000000000", "1.000000000000", "73935A10", "QQQQ", "G",
            "392750",
        ],
        [
            "86755", "46090E10", "35013", "6726", "22669", "20251231", "NS",
            "FUND", "ETF", "Y", "ACOR", "Q", "RW", "A", "2025-12-31", "N",
            "614.310000", "TR", "407778978.00", "BP", "619.430000", "TR",
            "2025-12-30", "410651118.50", "PB", "-0.008266", "-0.008266",
            "0.000000", "NA", "D1", "0.000000", "0.000000", "1.000000", "NO",
            "41563119", "614.310000", "614.050000", "619.960000",
            "614.260000", "614.280000", "619.650000", "693926", "58",
            "25532639632.9", "1.000000000000", "1.000000000000", "46090E10",
            "QQQ", "G", "663800",
        ],
    )
]


# ---------------------------------------------------------------------------
# Reference rows
# ---------------------------------------------------------------------------


def secinfo_row(
    permno,
    start,
    end,
    ticker,
    tradingsymbol=None,
    shareclass=None,
    **overrides,
) -> dict[str, str | None]:
    """One `stksecurityinfohist` interval, all 36 columns.

    The five arguments are the ones symbology reads (D-04); everything else
    takes the ordinary-common-share shape the live rows show. `securitybegdt`
    and `securityenddt` default to the interval's own edges, which is what a
    single-interval fixture wants; a multi-interval PERMNO overrides them with
    the security's real span.
    """
    record: dict[str, str | None] = {name: None for name in SECINFO_SERVER_COLUMNS}
    record.update(
        {
            "permno": str(int(permno)),
            "secinfostartdt": str(start),
            "secinfoenddt": str(end),
            "securitybegdt": str(start),
            "securityenddt": str(end),
            "securityhdrflg": "N",
            "primaryexch": "Q",
            "conditionaltype": "RW",
            "exchangetier": "Q",
            "tradingstatusflg": "A",
            "shareclass": None if shareclass is None else str(shareclass),
            "usincflg": "Y",
            "issuertype": "CORP",
            "securitytype": "EQTY",
            "securitysubtype": "COM",
            "sharetype": "NS",
            "securityactiveflg": "Y",
            "delactiontype": "N/A",
            "delstatustype": "UNAV",
            "delreasontype": "NACT",
            "delpaymenttype": "UNAV",
            "ticker": None if ticker is None else str(ticker),
            "tradingsymbol": None if tradingsymbol is None else str(tradingsymbol),
        }
    )
    for name, value in overrides.items():
        if name not in record:
            raise KeyError(
                f"secinfo_row: {name!r} is not a stksecurityinfohist column"
            )
        record[name] = None if value is None else str(value)
    return record


#: `stksecurityinfohist` intervals covering every symbology case the phase
#: needs. The AAPL/FB/BRK/GOOGL rows are VERBATIM
#: `03.10-LIVE-CHECK.json` keys `C4_sample_crsp_a_stock.stksecurityinfohist`
#: and `C5_ticker_hist_crsp_a_stock.stksecurityinfohist`; the Lehman rows are
#: `03.10-LIVE-CHECK-2.json` key `L3_3`; the BF rows are its key `L6_1`.
SECINFO_ROWS: list[dict[str, str | None]] = [
    # -- AAPL 14593, VERBATIM C4 sample (three intervals, 1980-12-12..2004-06-09)
    secinfo_row(14593, "1980-12-12", "1982-03-31", "AAPL", None, None,
                securitybegdt="1980-12-12", securityenddt="2025-12-31",
                exchangetier="N/A", securitynm="APPLE COMPUTER INC; COM NONE; CONS"),
    secinfo_row(14593, "1982-04-01", "1982-10-31", "AAPL", None, None,
                securitybegdt="1980-12-12", securityenddt="2025-12-31",
                exchangetier="NMS", securitynm="APPLE COMPUTER INC; COM NONE; CONS"),
    secinfo_row(14593, "1982-11-01", "2004-06-09", "AAPL", "AAPL", None,
                securitybegdt="1980-12-12", securityenddt="2025-12-31",
                exchangetier="NMS", securitynm="APPLE COMPUTER INC; COM NONE; CONS"),
    # SYNTHETIC: the live C4 sample stopped at three rows, so AAPL's modern
    # history is continued by ONE invented interval. Its ticker/tradingsymbol
    # are the live 2004-06-09 values carried forward, and its end is the
    # product end -- this is what makes the 2020-08 tracer rows labellable.
    secinfo_row(14593, "2004-06-10", "2025-12-31", "AAPL", "AAPL", None,
                securitybegdt="1980-12-12", securityenddt="2025-12-31",
                exchangetier="NMS", securitynm="APPLE INC; COM NONE; CONS"),
    # -- FB -> META 13407, VERBATIM C5 (the rename case)
    secinfo_row(13407, "2012-05-18", "2022-06-08", "FB", "FB", "A",
                securitybegdt="2012-05-18", securityenddt="2025-12-31"),
    secinfo_row(13407, "2022-06-09", "2025-12-31", "META", "META", "A",
                securitybegdt="2012-05-18", securityenddt="2025-12-31"),
    # -- BRK.B 83443, VERBATIM C5 (the class-suffix case; tradingsymbol is
    #    NULL before 2002, which is the collision-driven arm of D-04)
    secinfo_row(83443, "1996-05-09", "2002-01-01", "BRK", None, "B",
                securitybegdt="1996-05-09", securityenddt="2025-12-31"),
    secinfo_row(83443, "2002-01-02", "2025-12-31", "BRK", "BRKB", "B",
                securitybegdt="1996-05-09", securityenddt="2025-12-31"),
    # -- GOOG -> GOOGL 90319, VERBATIM C5
    secinfo_row(90319, "2004-08-19", "2014-04-02", "GOOG", "GOOG", "A",
                securitybegdt="2004-08-19", securityenddt="2025-12-31"),
    secinfo_row(90319, "2014-04-03", "2025-12-31", "GOOGL", "GOOGL", "A",
                securitybegdt="2004-08-19", securityenddt="2025-12-31"),
    # -- Lehman 80599, VERBATIM L3_3 (note the final NULL-ticker interval)
    secinfo_row(80599, "1994-05-31", "2002-01-01", "LEH", None, None),
    secinfo_row(80599, "2002-01-02", "2004-06-09", "LEH", None, None),
    secinfo_row(80599, "2004-06-10", "2006-07-06", "LEH", None, None),
    secinfo_row(80599, "2006-07-07", "2006-07-10", "LEH", None, None),
    secinfo_row(80599, "2006-07-11", "2008-06-26", "LEH", None, None),
    secinfo_row(80599, "2008-06-27", "2008-09-17", "LEH", None, None),
    secinfo_row(80599, "2008-09-18", "2008-09-18", None, None, None),
    # -- WestRock 21186, the MODERN delisting shape's intervals. Mirrors the
    #    Lehman block: a long trading interval, then a NULL-ticker interval on
    #    the delisting day itself, so the symbol on that row survives only
    #    through symbology's carry rule.
    # SYNTHETIC start: the live projection sampled 21186's DAILY rows (CR-01),
    #    never its security info, so the 2015-07-01 start is invented. The
    #    ticker and the 2024-07-05 end are the live values.
    secinfo_row(21186, "2015-07-01", "2024-07-05", "WRK", "WRK", None),
    secinfo_row(21186, "2024-07-08", "2024-07-08", None, None, None),
    # -- Brown-Forman / BF share classes, VERBATIM L6_1 (the live projection
    #    selected permno, secinfostartdt, secinfoenddt, ticker, tradingsymbol,
    #    shareclass only)
    secinfo_row(29946, "1991-05-09", "2002-01-01", "BF", None, "B"),
    secinfo_row(29938, "1991-05-09", "2002-01-01", "BF", None, "A"),
    secinfo_row(88279, "2000-06-07", "2000-10-31", "BF", None, None),
    secinfo_row(88279, "2000-11-01", "2001-05-31", "BF", None, None),
    secinfo_row(88279, "2001-06-01", "2002-01-01", "BF", None, None),
    secinfo_row(88279, "2002-01-02", "2004-06-09", "BF", "BF", None),
    secinfo_row(29938, "2002-01-02", "2004-06-09", "BF", "BFA", "A"),
    secinfo_row(29946, "2002-01-02", "2004-06-09", "BF", "BFB", "B"),
    secinfo_row(29938, "2004-06-10", "2004-06-30", "BF", "BFA", "A"),
    secinfo_row(29946, "2004-06-10", "2004-06-30", "BF", "BFB", "B"),
    secinfo_row(88279, "2004-06-10", "2007-09-05", "BF", "BF", None),
    secinfo_row(29938, "2004-07-01", "2004-07-26", "BF", "BFA", "A"),
    secinfo_row(29946, "2004-07-01", "2004-07-26", "BF", "BFB", "B"),
    secinfo_row(29938, "2004-07-27", "2008-06-26", "BF", "BFA", "A"),
    secinfo_row(29946, "2004-07-27", "2008-06-26", "BF", "BFB", "B"),
    # -- QQQ 86755. SYNTHETIC as `stksecurityinfohist` rows: the periods and
    #    tickers are VERBATIM from NDX-QQQ `C1_qqq_names`, but that live query
    #    read `stocknames_v2`, a DIFFERENT table with `namedt`/`nameenddt`
    #    rather than `secinfostartdt`/`secinfoenddt`. Restating them here is
    #    the invention, not the values.
    secinfo_row(86755, "1999-03-10", "2004-11-30", "QQQ", None, None,
                securitytype="FUND", securitysubtype="ETF", issuertype="ACOR",
                primaryexch="A", securitynm="NASDAQ 100 TRUST SERIES I"),
    secinfo_row(86755, "2004-12-01", "2011-03-22", "QQQQ", None, None,
                securitytype="FUND", securitysubtype="ETF", issuertype="ACOR",
                securitynm="POWERSHARES QQQ TRUST"),
    secinfo_row(86755, "2011-03-23", "2025-12-31", "QQQ", None, None,
                securitytype="FUND", securitysubtype="ETF", issuertype="ACOR",
                securitynm="INVESCO QQQ TRUST"),
    # -- 7000, a PERMNO that NEVER had a ticker. SYNTHETIC in its digits, LIVE
    #    in its shape (RESEARCH R5c): 1,012 PERMNOs in the raw tier carry no
    #    ticker on ANY of their intervals, 1,003 of them read `EQTY/COM/NS` --
    #    the ordinary-common-stock combination, which is why no type predicate
    #    can pick them out (RULING 1). Every such interval ends before
    #    1983-04-13; this one ends on that date, which is what makes the
    #    "zero admissions after 1990-08-20" boundary testable.
    #
    #    Four digits on purpose: 7000 also puts the numeric-vs-lexicographic
    #    axis order (D-19) into the corpus, since '10107' < '7000' as text.
    secinfo_row(7000, "1970-01-02", "1983-04-13", None, None, None,
                securitybegdt="1970-01-02", securityenddt="1983-04-13"),
]

#: `stkdelists`, VERBATIM `03.10-LIVE-CHECK-2.json` key `L3_2` (Lehman).
DELISTS_ROWS: list[dict[str, str | None]] = [
    _row(
        (
            "permno", "delistingdt", "deldtprc", "deldtprcflg", "delactiontype",
            "delstatustype", "delreasontype", "delpaymenttype", "delpermno",
            "delpermco", "delret", "delretmisstype", "delnextdt", "delnextprc",
            "delnextprcflg", "delamtdt", "deldivamt", "deldistype", "deldlydt",
        ),
        [
            "80599", "2008-09-17", "0.130000", "TR", "GDR", "VCL", "BKPY",
            "PRCF", "0", "0", "-0.600000", "NA", "2008-09-18", "0.052000",
            "DP", "2008-09-18", "0.000000", "NO", "2008-09-18",
        ],
    )
]

_DISTRIBUTION_COLUMNS = (
    "permno", "disexdt", "disseqnbr", "disordinaryflg", "distype",
    "disfreqtype", "dispaymenttype", "disdetailtype", "distaxtype",
    "disorigcurtype", "disdivamt", "disfacpr", "disfacshr", "disdeclaredt",
    "disrecorddt", "dispaydt", "dispermno", "dispermco", "disamountsourcetype",
)

#: `stkdistributions`, VERBATIM `03.10-LIVE-CHECK.json` key
#: `C4_sample_crsp_a_stock.stkdistributions` (AAPL 1987: a cash dividend, the
#: 2:1 split, another dividend).
DISTRIBUTION_ROWS: list[dict[str, str | None]] = [
    _row(_DISTRIBUTION_COLUMNS, values)
    for values in (
        [
            "14593", "1987-05-11", "1", "Y", "CD", "Q", "USD", "CDIV", "D",
            "USD", "0.120000", "0.000000", "0.000000", "1987-04-22",
            "1987-05-15", "1987-06-15", "0", "0", "N/A",
        ],
        [
            "14593", "1987-06-16", "1", "N", "FRS", "N/A", "SS", "STKSPL",
            "N", "N/A", "None", "1.000000", "1.000000", "1987-04-22",
            "1987-05-15", "1987-06-15", "0", "0", "N/A",
        ],
        [
            "14593", "1987-08-10", "1", "Y", "CD", "Q", "USD", "CDIV", "D",
            "USD", "0.060000", "0.000000", "0.000000", "1987-07-31",
            "1987-08-14", "1987-09-15", "0", "0", "N/A",
        ],
    )
]

#: `dsp500list_v2`, VERBATIM `03.10-LIVE-CHECK.json` key
#: `C4_sample_crsp_a_indexes.dsp500list_v2` (AAPL's open S&P 500 membership).
DSP500_ROWS: list[dict[str, str | None]] = [
    _row(
        ("permno", "indno", "mbrstartdt", "mbrenddt", "mbrflg", "indfam"),
        ["14593", "1000500", "1982-11-18", "2025-12-31", "NORM", "1100500"],
    )
]

#: `comp.idxcst_his`, VERBATIM `03.10-LIVE-CHECK-2.json` key `L8_2` (the two
#: Alphabet issues' Nasdaq-100 spells; `thru` NULL means still a member).
IDXCST_ROWS: list[dict[str, str | None]] = [
    _row(("gvkey", "iid", "gvkeyx", "from", "thru"), values)
    for values in (
        ["160329", "01", "000208", "2005-12-21", "None"],
        ["160329", "03", "000208", "2014-04-03", "None"],
    )
]

#: `crsp_a_ccm.ccmxpf_lnkhist`, VERBATIM `03.10-LIVE-CHECK-2.json` key `L7_4`.
#: `lpermno`/`lpermco` arrive as floats because the live column type is
#: `double precision` (key `L7_2`).
CCM_ROWS: list[dict[str, str | None]] = [
    _row(
        (
            "gvkey", "linkprim", "liid", "linktype", "lpermno", "lpermco",
            "linkdt", "linkenddt",
        ),
        values,
    )
    for values in (
        ["160329", "C", "00X", "NR", "None", "None", "2002-01-01", "2004-08-18"],
        ["160329", "P", "01", "LC", "90319.0", "45483.0", "2004-08-19", "None"],
        ["160329", "J", "03", "NR", "None", "None", "2014-03-31", "2014-04-02"],
        ["160329", "J", "03", "LC", "14542.0", "45483.0", "2014-04-03", "None"],
        ["160329", "J", "90C", "NR", "None", "None", "2024-12-31", "None"],
    )
]


def _default_reference_rows() -> dict[str, list[dict[str, str | None]]]:
    """Every reference table this module can serve, keyed `"schema.table"`."""
    return {
        "crsp_a_stock.stksecurityinfohist": list(SECINFO_ROWS),
        "crsp_a_stock.stkdelists": list(DELISTS_ROWS),
        "crsp_a_stock.stkdistributions": list(DISTRIBUTION_ROWS),
        "crsp_a_indexes.dsp500list_v2": list(DSP500_ROWS),
        "comp.idxcst_his": list(IDXCST_ROWS),
        "crsp_a_ccm.ccmxpf_lnkhist": list(CCM_ROWS),
    }


# ---------------------------------------------------------------------------
# The SQL parser
# ---------------------------------------------------------------------------
#
# The fake answers by PARSING the statement the production builders actually
# rendered, rather than by inspecting the arguments a method was called with.
# That is what makes a passing test evidence about the SQL: a builder that
# stopped emitting the PERMNO predicate would serve the wrong rows here
# instead of silently serving the right ones.

_SELECT_RE = re.compile(r"SELECT\s+(?P<cols>.+?)\s+FROM\s", re.IGNORECASE | re.DOTALL)
_FROM_RE = re.compile(r'FROM\s+"(?P<schema>[^"]+)"\."(?P<table>[^"]+)"', re.IGNORECASE)
_COLUMN_RE = re.compile(r'"([^"]+)"')
_ANY_RE = re.compile(r'"(?P<col>[a-z_0-9]+)"\s*=\s*ANY\(ARRAY\[(?P<values>[^\]]*)\]\)')
_BETWEEN_RE = re.compile(
    r"\"(?P<col>[a-z_0-9]+)\"\s+BETWEEN\s+'(?P<low>[^']*)'\s+AND\s+'(?P<high>[^']*)'",
    re.IGNORECASE,
)
_EQ_RE = re.compile(r"\"(?P<col>[a-z_0-9]+)\"\s*=\s*'(?P<value>[^']*)'")


def _parse_table(text: str) -> tuple[str, str]:
    match = _FROM_RE.search(text)
    if match is None:
        raise AssertionError(f"FakeCrspSession: no FROM \"schema\".\"table\" in {text!r}")
    return match.group("schema"), match.group("table")


def _parse_selected_columns(text: str) -> tuple[str, ...]:
    match = _SELECT_RE.search(text)
    if match is None:
        raise AssertionError(f"FakeCrspSession: no SELECT list in {text!r}")
    return tuple(_COLUMN_RE.findall(match.group("cols")))


def _parse_predicates(text: str) -> tuple[dict, dict, dict]:
    """`(in_sets, ranges, equalities)` parsed out of the WHERE conjuncts.

    The `ANY(ARRAY[...])` matches are removed from the text before the plain
    equality scan, so an array element never reads as an `= 'value'` conjunct.
    """
    in_sets: dict[str, set[str]] = {}
    for match in _ANY_RE.finditer(text):
        raw = match.group("values").strip()
        values = [item.strip().strip("'") for item in raw.split(",") if item.strip()]
        in_sets[match.group("col")] = {str(int(float(v))) if _is_number(v) else v
                                       for v in values}
    ranges = {
        match.group("col"): (match.group("low"), match.group("high"))
        for match in _BETWEEN_RE.finditer(text)
    }
    stripped = _ANY_RE.sub(" ", text)
    # Everything left of the WHERE is the projection and the table name; the
    # equality scan must not see `"table_name" = 'dsf_v2'`-style catalogue
    # predicates as data filters, which is why the caller passes only the
    # portion it wants filtered.
    equalities = {
        match.group("col"): match.group("value")
        for match in _EQ_RE.finditer(stripped)
    }
    return in_sets, ranges, equalities


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _matches(record: dict, in_sets: dict, ranges: dict, equalities: dict) -> bool:
    for column, wanted in in_sets.items():
        value = record.get(column)
        if value is None:
            return False
        normalised = str(int(float(value))) if _is_number(str(value)) else str(value)
        if normalised not in wanted:
            return False
    for column, (low, high) in ranges.items():
        value = record.get(column)
        if value is None or not (low <= str(value) <= high):
            return False
    for column, wanted in equalities.items():
        if str(record.get(column)) != wanted:
            return False
    return True


def _csv_bytes(records, columns: tuple[str, ...]) -> bytes:
    """`records` as COPY CSV over exactly `columns`, NULL as an empty field."""
    schema = {name: pl.String for name in columns}
    rows = [{name: record.get(name) for name in columns} for record in records]
    frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    return frame.write_csv(null_value="").encode()


# ---------------------------------------------------------------------------
# The fake session
# ---------------------------------------------------------------------------


class FakeCrspSession(FakeWrdsSession):
    """The offline double of the CRSP half of `WrdsSession`.

    Subclasses `FakeWrdsSession` rather than replacing it, so ONE fixture can
    serve both WRDS providers -- which is what lets
    `tests/test_acquisition_batching.py`'s subclass walk exercise the TAQ and
    the CRSP class in the same run.

    Class-level state, all reset by `reset()`:

    - `daily_rows` -- the `dsf_v2` records the daily COPY filters;
    - `reference_rows` -- `"schema.table" -> [record, ...]`;
    - `product_end` -- what `SELECT max("dlycaldt")` answers;
    - `usable_schemas` -- the schemas this account may read (`None` = all);
    - `server_columns` -- `"schema.table" -> columns`, what
      `information_schema.columns` reports;
    - `inject_rows` -- extra records EVERY daily COPY returns, whatever its
      WHERE says, for page-ownership tests;
    - `count_adjust` -- `{BETWEEN lower bound: delta}` added to that page's
      `count(*)`, to fake a COPY that disagrees with its count;
    - `raise_on_copy` -- `{daily COPY index: exception}`;
    - the recorders `crsp_copy_calls`, `crsp_count_calls`, `fetch_calls` and
      `schema_probes`.
    """

    daily_rows: list[dict] = []
    reference_rows: dict[str, list[dict]] = {}
    product_end: date = date(2025, 12, 31)
    usable_schemas: set[str] | None = None
    server_columns: dict[str, tuple[str, ...]] = {}
    inject_rows: list[dict] = []
    count_adjust: dict[str, int] = {}
    raise_on_copy: dict[int, BaseException] = {}
    crsp_copy_calls: list[dict] = []
    crsp_count_calls: list[dict] = []
    fetch_calls: list[str] = []
    schema_probes: list[str] = []

    @classmethod
    def reset(cls) -> None:
        """Reset BOTH halves of the fake.

        `FakeWrdsSession.reset()` is called EXPLICITLY on the base class, never
        as `super().reset()`. The base `reset` assigns through `cls`, so via
        `super()` it would create SUBCLASS-level shadows of `connections`,
        `rows`, `copy_calls` and the rest -- while `FakeWrdsSession.__init__`
        increments the BASE `connections`. The counter a test reads off
        `FakeWrdsSession` and the one this class resets would then be two
        different attributes, and the connection count would look frozen at 0.

        The `instance` shadow is deleted for the same reason, one step
        further on. `FakeWrdsSession.shared()` assigns `cls.instance`, so
        calling it THROUGH this subclass (which every WRDS test now does --
        the conftest patches this class over `wrds.taq.WrdsSession`) leaves a
        subclass-level `instance` that the base `reset()` cannot see. The next
        test would then be handed the PREVIOUS test's session object, no
        `__init__` would run, and `FakeWrdsSession.connections` would read 0
        for a run that did open a session. Deleting the shadow restores a
        single storage location for `instance`, on the base.
        """
        FakeWrdsSession.reset()
        if "instance" in vars(cls):
            delattr(cls, "instance")
        cls.daily_rows = (
            list(AAPL_AUG_2020_ROWS)
            + list(LEHMAN_2008_ROWS)
            + list(AAPL_1980_ROWS)
            + list(QQQ_ROWS)
        )
        cls.reference_rows = _default_reference_rows()
        cls.product_end = date(2025, 12, 31)
        cls.usable_schemas = None
        cls.server_columns = {}
        cls.inject_rows = []
        cls.count_adjust = {}
        cls.raise_on_copy = {}
        cls.crsp_copy_calls = []
        cls.crsp_count_calls = []
        cls.fetch_calls = []
        cls.schema_probes = []

    # -- the provider-neutral surface plan 03.10-01 added --------------------

    @classmethod
    def columns_for(cls, schema: str, table: str) -> tuple[str, ...]:
        """What `information_schema.columns` reports for one table.

        `server_columns` wins when it names the table; otherwise the answer is
        the pinned live column list -- `DSF_V2_SERVER_COLUMNS` for the daily
        table, and each `ReferenceTableSpec`'s own columns for the reference
        tables. The spec import is INSIDE the body so this module stays
        importable before `quantlab/dataset/crsp/reference.py` exists.
        """
        key = f"{schema}.{table}"
        if key in cls.server_columns:
            return tuple(cls.server_columns[key])
        if key == "crsp_a_stock.dsf_v2":
            return DSF_V2_SERVER_COLUMNS
        from quantlab.dataset.crsp.reference import REFERENCE_TABLES

        for spec in REFERENCE_TABLES:
            if (spec.schema, spec.table) == (schema, table):
                return tuple(spec.columns)
        raise AssertionError(f"FakeCrspSession: no column list for {key}")

    def schema_usable(self, schema: str) -> bool:
        FakeCrspSession.schema_probes.append(str(schema))
        return (
            FakeCrspSession.usable_schemas is None
            or str(schema) in FakeCrspSession.usable_schemas
        )

    def fetch_rows(self, query) -> list[tuple]:
        """Answer the catalogue reads: schema privilege, column lists, the
        product end and `count(*)`."""
        text = render_composed(query)
        FakeCrspSession.fetch_calls.append(text)

        if "has_schema_privilege" in text:
            schema = _EQ_RE.search(text)
            name = schema.group("value") if schema else ""
            return [(self.schema_usable(name),)]

        if "information_schema" in text.lower():
            _, _, equalities = _parse_predicates(text)
            schema = equalities.get("table_schema", "")
            table = equalities.get("table_name", "")
            columns = self.columns_for(schema, table)
            return [(name, index + 1) for index, name in enumerate(columns)]

        if "max(" in text.lower():
            return [(FakeCrspSession.product_end,)]

        if "count(*)" in text.lower():
            schema, table = _parse_table(text)
            where = text.split(" WHERE ", 1)[1] if " WHERE " in text else ""
            in_sets, ranges, equalities = _parse_predicates(where)
            records = self._records(schema, table)
            matched = [r for r in records if _matches(r, in_sets, ranges, equalities)]
            if f"{schema}.{table}" == "crsp_a_stock.dsf_v2":
                FakeCrspSession.crsp_count_calls.append({"sql": text})
                matched = matched + list(FakeCrspSession.inject_rows)
                low = ranges.get("dlycaldt", ("", ""))[0]
                return [(len(matched) + FakeCrspSession.count_adjust.get(low, 0),)]
            return [(len(matched),)]

        raise AssertionError(f"FakeCrspSession.fetch_rows: unhandled query {text!r}")

    def copy_csv(self, query) -> bytes:
        text = render_composed(query)
        schema, table = _parse_table(text)
        columns = _parse_selected_columns(text)
        where = text.split(" WHERE ", 1)[1] if " WHERE " in text else ""
        in_sets, ranges, equalities = _parse_predicates(where)

        records = [
            record
            for record in self._records(schema, table)
            if _matches(record, in_sets, ranges, equalities)
        ]
        if f"{schema}.{table}" == "crsp_a_stock.dsf_v2":
            index = len(FakeCrspSession.crsp_copy_calls)
            FakeCrspSession.crsp_copy_calls.append({"sql": text, "columns": columns})
            failure = FakeCrspSession.raise_on_copy.get(index)
            if failure is not None:
                raise failure
            records = records + list(FakeCrspSession.inject_rows)
        return _csv_bytes(records, columns)

    @staticmethod
    def _records(schema: str, table: str) -> list[dict]:
        key = f"{schema}.{table}"
        if key == "crsp_a_stock.dsf_v2":
            return list(FakeCrspSession.daily_rows)
        return list(FakeCrspSession.reference_rows.get(key, []))


# ---------------------------------------------------------------------------
# Helpers the tests drive the fake with
# ---------------------------------------------------------------------------


def write_reference_tables(
    reference_dir,
    rows_by_table: dict[str, list[dict]] | None = None,
    product_end: str = "2025-12-31",
) -> Path:
    """Write the reference parquet tier plus its manifest, offline.

    The same SHAPE plan 04's real writer produces: one `{name}.parquet` per
    `ReferenceTableSpec`, cast through the spec, beside a `manifest.json`.
    `rows_by_table` is keyed `"schema.table"` and defaults to this module's
    live rows; a table with no rows is written EMPTY rather than skipped, so a
    reader meets the schema rather than a `FileNotFoundError`.
    """
    from quantlab.dataset.crsp.reference import MANIFEST_NAME, REFERENCE_TABLES
    from quantlab.utils.atomic import write_json_atomically

    rows_by_table = (
        _default_reference_rows() if rows_by_table is None else rows_by_table
    )
    directory = Path(reference_dir)
    directory.mkdir(parents=True, exist_ok=True)

    tables: dict[str, int] = {}
    for spec in REFERENCE_TABLES:
        records = rows_by_table.get(f"{spec.schema}.{spec.table}", [])
        frame = pl.DataFrame(
            [{name: record.get(name) for name in spec.columns} for record in records],
            schema={name: pl.String for name in spec.columns},
        )
        frame = spec.cast(frame)
        frame.write_parquet(directory / f"{spec.name}.parquet")
        tables[spec.name] = frame.height

    write_json_atomically(
        directory / MANIFEST_NAME,
        {
            "product_end": str(product_end),
            "pulled_at": "2026-09-20T00:00:00Z",  # SYNTHETIC
            "tables": tables,
        },
        indent=2,
        sort_keys=True,
    )
    return directory


def run_crsp_pull(
    tmp_path,
    permnos,
    start_date: str,
    end_date: str,
    kwargs: dict | None = None,
):
    """Point the data root at `tmp_path`, run a CRSP pull, return its config.

    Goes through the registry exactly as a caller would: the config comes from
    the `("us_equity", "1d", "crsp_daily")` capability's own `config_factory`,
    and the acquisition class from the same capability -- so this helper
    cannot accidentally exercise a class the registry would not resolve.
    """
    import quantlab.config as config
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE

    config.set_data_root(Path(tmp_path))
    factory = WRDS_SOURCE.config_factory_for("us_equity", "1d", "crsp_daily")
    cfg = factory(
        symbols=tuple(str(permno) for permno in permnos),
        start_date=start_date,
        end_date=end_date,
        kwargs=kwargs,
    )
    result = registry.download_symbols(WRDS_SOURCE, cfg)
    return cfg, result


def crsp_username_is_set() -> bool:
    """Whether `WRDS_USERNAME` is set -- the precondition `shared()` enforces."""
    return bool(os.environ.get("WRDS_USERNAME"))
