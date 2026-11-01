"""Sharadar corporate actions (ACTIONS) as a daily panel by family, keyed by permaticker.

ACTIONS holds one row per security, action and date: a dividend, a split, a
takeover, a delisting, a ticker change. Its ``action`` column takes one of
66 names, most of them a handful of rows (30-odd ``acquisition*`` names
record the exact consideration of single deals). ``SharadarActionsDataset``
folds them into *families*, one variable or a few per family, each row
placed on its effective date (``date``; for a takeover or a delisting the
security's last trading day): on the first SEP trading day on or after it,
and on no other day (``quantlab.dataset.sharadar.filings.FilingPanelDataset``).
A day without the action is NaN for a value and False for a flag.

The families and the vendor names in each (``FAMILIES``), with what each
value means as Sharadar's ``descriptions`` table (``actiontypes`` rows,
2026-10-16) states it:

- ``dividend`` (``dividend``): cash per share paid on the ex-date, USD.
  The vendor adjusts the value for every split it had applied when the
  file was pulled; the panel multiplies it back by the ratios of the
  security's ACTIONS ``split`` rows dated after the ex-date and before the
  file's pull day (``quantlab.dataset.sharadar.tables.split_basis_change``),
  so it is the cash per share held on the day. Several on one day are
  summed.
- ``split`` (``split``): shares held after the split per share held
  before, on its effective date. Several on one day are multiplied.
- ``spinoff`` (``spinoff``, ``spunofffrom``): spun-off shares issued per
  parent share, on the parent (``spinoff``) and on the spun-off security
  (``spunofffrom``); ``spinoff_value`` (``spinoffdividend``) the USD value
  of those shares per parent share, restored for later splits like a
  dividend; ``spinoff_counterparty`` the other side's permaticker;
  ``spinoff_side`` 1 parent, 2 spun off, 3 both.
- ``acquisition`` (``acquisitionby`` on the acquired company,
  ``acquisitionof`` on the acquirer): the acquired company's final market
  cap, USD millions; ``acquisition_counterparty`` the other side's
  permaticker; ``acquisition_side`` 1 acquired, 2 acquirer, 3 both;
  ``acquisition_consideration`` what the acquired company's holders
  received, from every other ``acquisition*`` row of the day: 1 cash,
  2 stock, 3 mixed (several kinds), 4 other (warrants, special securities).
  A consideration row also follows a merger.
- ``merger`` (``mergerto`` on the non-surviving company, ``mergerfrom`` on
  the survivor): the non-surviving company's final market cap, USD
  millions; ``merger_counterparty``; ``merger_side`` 1 non-surviving,
  2 surviving, 3 both.
- ``delisting`` (``delisted``, ``regulatorydelisting``,
  ``voluntarydelisting``, ``bankruptcyliquidation``): the final market cap,
  USD millions (empty when the security is not the company's primary
  common stock); ``delisting_reason`` 1 delisted, 2 regulatory,
  3 voluntary, 4 bankruptcy or liquidation, the largest when several.
- flags, True on the action's day: ``listed``, ``initiated`` (coverage
  start of an already listed security), ``ticker_change``, ``name_change``,
  ``exchange_change``, ``sic_change`` (each from its ``...from`` / ``...to``
  pair), ``spac_merger``, ``spac_unit_separation``, ``relation``,
  ``reorganization`` (``reorganizationfrom``, ``reorganizationto``,
  ``reorganizationpostsplitcommon``) and ``equity_cancellation``.

``adrratiosplit`` (a split of an ADR's ratio) is read and left out: ADRs are
outside the panel's universe and SEP's prices do not apply it. An action
name in none of these is refused, naming it, so a new vendor action is
never dropped silently: add it to ``FAMILIES``.

Each variable's attributes give its ``description``, ``unit``, the vendor
``actions`` it is built from and, for a coded variable, ``codes`` (code, as
a string, to meaning). A counterparty is the permaticker SEP's TICKERS rows
give the row's ``contraticker`` as of the raw file's pull, like the row's
own ticker (``quantlab.dataset.sharadar.permatickers``); NaN when it maps to
no SEP security (a private acquirer, a fund, ``N/A``), and the smallest one
when a day has several. A ``contraticker`` TICKERS gives to several
permatickers is not guessed either: the counterparty is NaN and the rows are
counted in ``<store>.counterparties.json`` (``{"owner", "unmapped":
[{"contraticker", "first_date", "last_date", "rows", "reason"}]}``, deleted
when there are none), with a warning. ACTIONS is keyed by SEP's tickers and also holds
funds' rows; a row mapping to no SEP security is left out. The panel covers
the whole market's common stock by default (SEP's category filter).

ACTIONS is pulled whole every day; ``update()`` appends the new trading
days and never rewrites a stored one. The vendor can record an action weeks
after its effective date; a stored store keeps the days as built, and only
a rebuild shows such a late row.

Examples
--------
Build the store and read a quarter's dividends::

    config = SharadarActionsConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_actions_1d/sharadar_actions_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarActionsDataset(config).update()
    panel = SharadarActionsDataset(config).panel("2024-01-02", "2024-03-28")
    panel["dividend"].attrs["unit"]  # 'USD/share'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from loguru import logger

from quantlab.dataset.config import SharadarActionsConfig
from quantlab.dataset.sharadar.filings import FilingPanelDataset
from quantlab.dataset.sharadar.permatickers import PermatickerResolver
from quantlab.dataset.sharadar.tables import (
    scan_raw_table,
    split_basis_change,
    with_pull_day,
)
from quantlab.dataset.sharadar.universe import universe
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.timer import Timer

#: How the vendor fills an unused contra column.
_NOT_APPLICABLE = "N/A"

#: The consideration ``acquisition*`` names, by what the holders received.
_CASH = (
    "acquisitioncash", "acquisitioncashcad", "acquisitioncasheur", "acquisitioncashgbp",
    "acquisitioncashhkd", "acquisitioncashsgd", "acquisitioncashusd",
    "acquisitioncohortcashusd", "acquisitiondefaultcasheur",
    "acquisitiondistributioncashusd", "acquisitiondividendusd", "acquisitionelectcash",
    "acquisitionelectcashusd", "acquisitionestimatedcashusd",
    "acquisitionqualificationcashcad", "acquisitionterminalcashusd",
    # Sharadar 2026-10: best-effort variants of the cash consideration.
    "acquisitionbesteffortcashcad", "acquisitionbesteffortcashusd",
)
_STOCK = (
    "acquisitionstock", "acquisitioncommon", "acquisitionpreferred",
    "acquisitionalternativecommon", "acquisitioncohortcommon", "acquisitioncohortpreferred",
    "acquisitiondistributionstock", "acquisitionelectcommon", "acquisitionelectstock",
    "acquisitionestimatedcommon", "acquisitionestimatedceilcommon",
    "acquisitionestimatedpackagecommon", "acquisitionestimatedpackagepreferred",
    "acquisitionestimatedroundupcommon", "acquisitionqualificationdefaultstock",
    "acquisitionqualificationstock",
    # Sharadar 2026-10: best-effort variants of the common and preferred consideration.
    "acquisitionbesteffortcommon", "acquisitionbesteffortpreferred", "acquisitionbesteffortpreferred2",
)
_OTHER = (
    "acquisitionwarrant", "acquisitionalternativewarrant", "acquisitionspecial",
    # Sharadar 2026-10: notes and rights, neither cash nor stock, as the warrants.
    "acquisitionbesteffortnote", "acquisitionbesteffortright",
)

#: Code of each kind of consideration; several kinds on one day are ``MIXED``.
CONSIDERATION_CODES: dict[str, str] = {"1": "cash", "2": "stock", "3": "mixed", "4": "other"}
_MIXED = 3
_CONSIDERATION_KIND: dict[str, int] = {
    **dict.fromkeys(_CASH, 1),
    **dict.fromkeys(_STOCK, 2),
    **dict.fromkeys(_OTHER, 4),
}

#: Code of each delisting reason; the largest of a day's is kept.
DELISTING_REASONS: dict[str, int] = {
    "delisted": 1,
    "regulatorydelisting": 2,
    "voluntarydelisting": 3,
    "bankruptcyliquidation": 4,
}

#: Cash distributions whose value the vendor adjusts for later splits.
_SPLIT_ADJUSTED = ("dividend", "spinoffdividend")

#: Read and left out of the panel: ADRs are outside its universe.
IGNORED: tuple[str, ...] = ("adrratiosplit",)


@dataclass(frozen=True)
class Variable:
    """One panel variable: the vendor actions it reads and how a day's rows combine.

    ``how`` is ``"sum"`` or ``"product"`` of the rows' values, ``"counterparty"``
    (the smallest counterparty permaticker), ``"side"`` (1 when a row of
    ``actions`` is there, plus 2 when one of ``second`` is), ``"code"`` (the
    largest of ``codes[action]``), ``"consideration"`` or ``"flag"``.
    """

    name: str
    actions: tuple[str, ...]
    how: str
    description: str
    unit: str | None = None
    second: tuple[str, ...] = ()
    codes: dict[str, str] = field(default_factory=dict)

    @property
    def read(self) -> tuple[str, ...]:
        """Return every vendor action the variable reads."""
        return self.actions + self.second


_SPINOFF_PARENT = ("spinoff", "spinoffdividend")
_ACQUISITION = ("acquisitionby", "acquisitionof")
_MERGER = ("mergerto", "mergerfrom")
_DELISTING = tuple(DELISTING_REASONS)

#: Every variable of the panel, in panel order.
VARIABLES: tuple[Variable, ...] = (
    Variable("dividend", ("dividend",), "sum",
             "cash dividend per share held on the ex-date (vendor value restored for later splits)",
             "USD/share"),
    Variable("split", ("split",), "product",
             "shares held after the split per share held before", "ratio"),
    Variable("spinoff", ("spinoff", "spunofffrom"), "sum",
             "spun-off shares issued per parent share, on the parent and on the spun-off security",
             "ratio"),
    Variable("spinoff_value", ("spinoffdividend",), "sum",
             "USD value of the spun-off shares issued per parent share (restored for later splits)",
             "USD/share"),
    Variable("spinoff_counterparty", ("spinoff", "spinoffdividend", "spunofffrom"), "counterparty",
             "permaticker of the other side: the spun-off security on the parent, the parent on it",
             "permaticker"),
    Variable("spinoff_side", _SPINOFF_PARENT, "side", "which side of a spinoff the security is",
             second=("spunofffrom",), codes={"1": "parent", "2": "spun off", "3": "both"}),
    Variable("acquisition", _ACQUISITION, "sum",
             "final market cap of the acquired company", "USD millions"),
    Variable("acquisition_counterparty", _ACQUISITION, "counterparty",
             "permaticker of the other side: the acquirer on the acquired company, and back",
             "permaticker"),
    Variable("acquisition_side", ("acquisitionby",), "side",
             "which side of an acquisition the security is",
             second=("acquisitionof",), codes={"1": "acquired", "2": "acquirer", "3": "both"}),
    Variable("acquisition_consideration", (*_CASH, *_STOCK, *_OTHER), "consideration",
             "what the acquired (or non-surviving) company's holders received",
             codes=CONSIDERATION_CODES),
    Variable("merger", _MERGER, "sum",
             "final market cap of the non-surviving company", "USD millions"),
    Variable("merger_counterparty", _MERGER, "counterparty",
             "permaticker of the other side: the survivor on the non-surviving company, and back",
             "permaticker"),
    Variable("merger_side", ("mergerto",), "side", "which side of a merger the security is",
             second=("mergerfrom",), codes={"1": "non-surviving", "2": "surviving", "3": "both"}),
    Variable("delisting", _DELISTING, "sum",
             "final market cap of the delisted security (primary common stock only)",
             "USD millions"),
    Variable("delisting_reason", _DELISTING, "code", "why the security was delisted",
             codes={str(code): name for name, code in DELISTING_REASONS.items()}),
    Variable("listed", ("listed",), "flag", "the security listed for trading"),
    Variable("initiated", ("initiated",), "flag",
             "the vendor's price coverage of an already listed security started"),
    Variable("ticker_change", ("tickerchangefrom", "tickerchangeto"), "flag",
             "the security's ticker changed"),
    Variable("name_change", ("namechangefrom", "namechangeto"), "flag",
             "the company's name changed"),
    Variable("exchange_change", ("exchangefrom", "exchangeto"), "flag",
             "the security moved to another trading venue"),
    Variable("sic_change", ("sicchangefrom", "sicchangeto"), "flag",
             "the company's SIC industry changed"),
    Variable("spac_merger", ("spacmerger",), "flag", "a SPAC business combination took effect"),
    Variable("spac_unit_separation", ("spacunitseparation",), "flag",
             "a SPAC unit separated into common stock (and warrants)"),
    Variable("relation", ("relation",), "flag",
             "the vendor links the security to another of its issuer's securities"),
    Variable("reorganization",
             ("reorganizationfrom", "reorganizationto", "reorganizationpostsplitcommon"), "flag",
             "a public-security reorganization links the security to a predecessor or successor"),
    Variable("equity_cancellation", ("equitycancellation",), "flag",
             "the original common equity was cancelled for no consideration"),
)

#: Each vendor action name the panel knows, and the variables reading it.
FAMILIES: dict[str, tuple[str, ...]] = {}
for _variable in VARIABLES:
    for _action in _variable.read:
        FAMILIES[_action] = (*FAMILIES.get(_action, ()), _variable.name)
for _action in IGNORED:
    FAMILIES[_action] = ()

#: The suffix of the report of the counterparties left NaN (``<store>.counterparties.json``).
COUNTERPARTIES_SUFFIX = ".counterparties.json"

#: The actions whose ``contraticker`` is mapped to a counterparty permaticker.
_WITH_COUNTERPARTY: tuple[str, ...] = tuple(
    sorted({a for v in VARIABLES if v.how == "counterparty" for a in v.read})
)


class SharadarActionsDataset(FilingPanelDataset):
    """Daily panel of corporate actions by family, each on its effective date.

    Parameters
    ----------
    config : SharadarActionsConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``actions``, ``sep`` (the calendar) and ``tickers`` raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarActionsDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)[:3]
        # ['dividend', 'split', 'spinoff']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarActionsConfig

    TABLE = "actions"

    def _empty(self) -> dict[str, object]:
        """Return NaN for each valued variable and False for each flag: no action."""
        return {v.name: False if v.how == "flag" else float("nan") for v in VARIABLES}

    def _attrs(self) -> dict[str, dict]:
        """Return each variable's description, unit, vendor actions and code table."""
        attrs = {}
        for v in VARIABLES:
            entry: dict[str, object] = {"description": v.description, "actions": list(v.read)}
            if v.unit is not None:
                entry["unit"] = v.unit
            if v.codes:
                entry["codes"] = dict(v.codes)
            attrs[v.name] = entry
        return attrs

    def _released(self) -> pl.DataFrame:
        """Return every day with an action, per security, combined by family.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` and one column per variable, one row
            per pair, sorted by both.

        Raises
        ------
        ValueError
            If ACTIONS holds an action name ``FAMILIES`` does not know, or a
            row's own ticker has two permatickers (a counterparty's is left
            NaN and counted in ``counterparties_path``).
        """
        root = self.config.raw_data_dir_path
        calendar = self._trading_calendar()
        resolver = PermatickerResolver(root, self.TABLE)
        several: dict[str, list[int]] = {}

        def annotate(path, frame: pl.LazyFrame) -> pl.LazyFrame:
            """Add the row's and its counterparty's permatickers, both as of the file's pull."""
            frame = resolver.annotate(path, frame)
            contras = (
                frame.filter(pl.col("action").is_in(_WITH_COUNTERPARTY))
                .select(pl.col("contraticker").unique())
                .collect()
                .get_column("contraticker")
                .to_list()
            )
            contras = [t for t in contras if t is not None and t != _NOT_APPLICABLE]
            # Which of several securities was the other side cannot be told: NaN, counted.
            ambiguous = resolver.several_permatickers(path, contras)
            several.update(ambiguous)
            mapped = resolver.resolve(path, [t for t in contras if t not in ambiguous])
            pairs = pl.DataFrame(
                {"contraticker": list(mapped), "counterparty": list(mapped.values())},
                schema={"contraticker": pl.String, "counterparty": pl.Int64},
            )
            return frame.join(pairs.lazy(), on="contraticker", how="left").with_columns(
                pl.when(pl.col("action").is_in(_WITH_COUNTERPARTY)).then(pl.col("counterparty"))
            )

        with Timer(f"{self.class_name}: place actions"):
            frame = scan_raw_table(root, self.TABLE, annotate=with_pull_day(annotate)).collect()
            unknown = sorted(set(frame.get_column("action").unique().drop_nulls()) - set(FAMILIES))
            if unknown:
                raise ValueError(
                    f"{self.class_name}: ACTIONS holds action name(s) {unknown} that no "
                    f"family of quantlab.dataset.sharadar.actions.FAMILIES knows; add each "
                    f"to a family (or to IGNORED) rather than drop it."
                )
            # ACTIONS also holds funds' rows: a row mapping to no SEP security is another one's.
            frame = resolver.left_out(
                frame,
                owner=self.class_name,
                report_path=f"{self.config.zarr_file_path}.unmapped.json",
                quiet=True,
                report_unmapped=False,
            )
            # Restored before the calendar's end cuts the splits it does not reach.
            frame = self._restore_splits(frame).filter(pl.col("date") <= pl.lit(calendar.through))
            frame = frame.filter(pl.col("permaticker").is_in(universe(self.config)))
            self._report_counterparties(frame, several)
            return (
                calendar.place(self._filings(frame), drop_earlier=True)
                .group_by(pl.col("permaticker").alias("symbol"), "available")
                .agg(self._combine())
                .sort("symbol", "available")
            )

    @property
    def counterparties_path(self) -> str:
        """The report of the counterparties left NaN (``<store>.counterparties.json``).

        Examples
        --------
        With ``config`` a ``SharadarActionsConfig`` of the store ``sharadar_actions_1d.zarr``::

            SharadarActionsDataset(config).counterparties_path
            # '.../sharadar_actions_1d.zarr.counterparties.json'
        """
        return f"{self.config.zarr_file_path}{COUNTERPARTIES_SUFFIX}"

    def _report_counterparties(self, frame: pl.DataFrame, several: dict[str, list[int]]) -> None:
        """Write the panel's rows whose counterparty ticker has several permatickers, or delete the report."""
        rows = frame.filter(
            pl.col("action").is_in(_WITH_COUNTERPARTY) & pl.col("contraticker").is_in(list(several))
        )
        path = Path(self.counterparties_path)
        if rows.height == 0:
            path.unlink(missing_ok=True)
            return
        summary = (
            rows.group_by("contraticker")
            .agg(pl.col("date").min().alias("first"), pl.col("date").max().alias("last"), pl.len().alias("rows"))
            .sort("contraticker")
        )
        entries = [
            {
                "contraticker": ticker,
                "first_date": str(first)[:10],
                "last_date": str(last)[:10],
                "rows": int(count),
                "reason": f"TICKERS gives it to several permatickers ({several[ticker]})",
            }
            for ticker, first, last, count in summary.iter_rows()
        ]
        logger.warning(
            f"{self.class_name}: {rows.height} ACTIONS row(s) name a counterparty ticker of several "
            f"permatickers ({', '.join(e['contraticker'] for e in entries[:5])}); their counterparty "
            f"is NaN, see {path}."
        )
        write_json_atomically(path, {"owner": self.class_name, "unmapped": entries}, indent=2)

    @staticmethod
    def _restore_splits(frame: pl.DataFrame) -> pl.DataFrame:
        """Multiply each split-adjusted cash value back by the later splits its file's pull had applied."""
        splits = frame.filter(pl.col("action") == "split").select("permaticker", "date", "value")
        adjusted = split_basis_change(
            frame.with_row_index("_position"), splits, source="_pulled", target="date"
        )
        return (
            adjusted.with_columns(
                pl.when(pl.col("action").is_in(_SPLIT_ADJUSTED))
                .then(pl.col("value") * pl.col("_basis"))
                .otherwise(pl.col("value"))
                .alias("value")
            )
            .sort("_position")
            .drop("_position", "_basis")
        )

    def _filings(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Return the rows of the panel's actions with their value and counterparty."""
        return raw.filter(pl.col("action").is_in([a for a, names in FAMILIES.items() if names])).select(
            "permaticker", "date", "action", "value", "counterparty"
        )

    def _combine(self) -> list[pl.Expr]:
        """Return each variable's combination of a security's actions of one day."""
        return [_combined(v).alias(v.name) for v in VARIABLES]


def _combined(v: Variable) -> pl.Expr:
    """Return the aggregation of one variable over a security's rows of one day."""
    action = pl.col("action")
    of = action.is_in(v.actions)
    if v.how == "flag":
        return of.any()
    if v.how in ("sum", "product"):
        values = pl.col("value").filter(of)
        combined = values.sum() if v.how == "sum" else values.product()
        return pl.when(values.is_not_null().any()).then(combined).cast(pl.Float64)
    if v.how == "counterparty":
        return pl.col("counterparty").filter(of).min().cast(pl.Float64)
    if v.how == "side":
        side = of.any().cast(pl.Int64) + 2 * action.is_in(v.second).any().cast(pl.Int64)
        return pl.when(side > 0).then(side).cast(pl.Float64)
    if v.how == "code":
        return action.filter(of).replace_strict(DELISTING_REASONS, default=None).max().cast(pl.Float64)
    if v.how == "consideration":
        kinds = action.filter(of).replace_strict(_CONSIDERATION_KIND, default=None)
        return (
            pl.when(kinds.n_unique() > 1)
            .then(pl.lit(_MIXED))
            .otherwise(kinds.first())
            .cast(pl.Float64)
        )
    raise ValueError(f"unknown combination {v.how!r} of variable {v.name!r}")
