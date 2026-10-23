"""The SQL statement that resamples TAQ NBBO records into bars on the WRDS server.

over the slow link to WRDS, downloading every NBBO record and
resampling locally takes weeks for a decade of S&P 500 data, so the
resampling runs in SQL and only bars cross the network. ``NbboBarsQuery``
builds that statement for one day table and one batch of symbols. It
reproduces ``quantlab.dataset.nbbo.resample.NbboResampler`` step by step:

1. **Order.** ``row_number() OVER ()`` numbers the rows in the server's scan
   order before anything else touches them, the same assumption the tick
   path makes with ``wrds_row_ord``. Records are ordered by time (with
   ``time_m_nano`` from 2018 on), then by that number, which before 2018 is
   the only way to order records sharing a microsecond.
2. **Normalise.** A null or NaN price nulls its side and that side's size.
3. **Filter.** The rules of ``NbboFilterPolicy`` that are enabled, in its
   priority order (non-positive price, condition, crossed, locked); a
   dropped record is ignored completely and counted under its first
   matching rule. Records after the session close are dropped too.
4. **Bars.** Bar ``k`` covers ``(open + (k-1)d, open + kd]``; a record at or
   before the open falls in bar 0, the seed. The snapshot of bar ``k`` is
   the last kept record at or before its end, so records sharing an instant
   collapse to the last one and a bar with no update carries the quote in
   force. ``n_updates`` counts every kept record inside bar ``k``, tied ones
   included; ``n_ambiguous_ties`` counts those of them whose instant holds
   more than one distinct (bid, bid size, ask, ask size) state, so identical
   duplicates count zero.
5. **Time weights.** After ties collapse, each quote is in force from its
   instant (the seed from the open) to the next quote or the close. Each
   such span is cut at the bar edges; ``tw_spread``, ``tw_bid_size`` and
   ``tw_ask_size`` of bar ``k`` are the sum of value times duration over
   the pieces in bar ``k`` where the value is defined, divided by the sum of
   their durations, and NULL when the value was never defined in the bar.
6. **Grid.** Every ``(ticker, k)`` for ``k = 1..N`` is returned for every
   ticker with a record that day, also when all its records were dropped,
   so the result has exactly ``tickers x N`` rows. ``page_rows``, the row
   count the server computed, is on every row, so the client can tell a
   truncated transfer from a complete one.
7. **Filter counts.** The ``FILTER_STATS_COUNTS`` of each ticker over all its
   records of the day, as ``NbboResampler.resample_with_stats`` counts them,
   are set on the ticker's bar 1 row and NULL on its other rows.

Steps 4 and 5 share one ordering of the kept records by ticker, time and
scan order: a record is the last at its instant when the next record is
later, and the last in its bar when the next record is in a later bar, so
the server sorts the day's records once. Only records sharing their instant
with a neighbour are grouped by instant for ``n_ambiguous_ties``, and only
quotes in force across a bar edge are split into pieces per bar.

Times are compared as nanoseconds since midnight on TAQ's ``time_m`` clock
(New York); the client computes the session bounds from the XNYS calendar
and passes them in, so the server needs no calendar. The bar labels are
computed by the client from ``k``. ``mid``, ``spread``, ``spread_bps`` and
``imbalance`` are derived by the client from the snapshot. An instant holds
several states when, in any of the four columns, its records hold more than
one distinct value or mix null and non-null ones; values are compared as
numbers, never through their text form.

Every value enters the statement as a ``psycopg2.sql.Literal`` and every
name as a ``psycopg2.sql.Identifier``; nothing is pasted in as text. The
statement is always restricted by the ``(sym_root, sym_suffix)`` pairs
condition of ``WrdsSession.where_clause``, so it never scans a whole day
table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

from psycopg2 import sql

from quantlab.acquisition.wrds.taq import WrdsSession
from quantlab.dataset.nbbo.bars import (
    SERVER_BAR_VARIABLES,
    NbboBarsRequest,
    NbboBarsSession,
)
from quantlab.dataset.nbbo.resample import FILTER_STATS_COUNTS

#: The CSV columns the statement returns, in order. The filter counts are
#: per ticker for the whole day, so they are set on each ticker's bar 1 row
#: only and empty on the others.
RESULT_COLUMNS = (
    "sym_root",
    "sym_suffix",
    "bar",
    *SERVER_BAR_VARIABLES,
    *FILTER_STATS_COUNTS,
    "page_rows",
)

#: Each ``NbboFilterPolicy`` drop reason, in priority order, with the
#: ``FILTER_STATS_COUNTS`` column that counts it.
_REASON_COLUMNS = {
    reason: f"dropped_{reason}"
    for reason in ("nonpositive_price", "condition", "crossed", "locked")
}

_STATEMENT = """COPY (
WITH raw AS (
    SELECT sym_root,
           coalesce(sym_suffix, '') AS sym_suffix,
           row_number() OVER () AS ord,
           round(extract(epoch FROM time_m - time '00:00') * 1000000)::bigint
               * 1000 + {nano} AS t_ns,
           qu_cond, best_bid, best_bidsizeshares, best_ask, best_asksizeshares
    FROM {table}
    WHERE {where}
),
typed AS (
    SELECT sym_root, sym_suffix, ord, t_ns, qu_cond,
           nullif(best_bid::float8, 'NaN') AS bid,
           best_bidsizeshares::float8 AS bid_size,
           nullif(best_ask::float8, 'NaN') AS ask,
           best_asksizeshares::float8 AS ask_size
    FROM raw
),
quotes AS (
    SELECT sym_root, sym_suffix, ord, t_ns, qu_cond, bid, ask,
           CASE WHEN bid IS NOT NULL THEN bid_size END AS bid_size,
           CASE WHEN ask IS NOT NULL THEN ask_size END AS ask_size
    FROM typed
),
judged AS (
    SELECT quotes.*, {reason} AS reason
    FROM quotes
),
filter_stats AS (
    SELECT sym_root, sym_suffix,
           count(*) AS records_in,
           {dropped_counts},
           count(*) FILTER (
               WHERE reason IS NULL AND (bid IS NULL) <> (ask IS NULL)
           ) AS one_sided_kept,
           count(*) FILTER (
               WHERE reason IS NULL AND bid IS NULL AND ask IS NULL
           ) AS both_null_kept
    FROM judged
    GROUP BY sym_root, sym_suffix
),
kept AS (
    SELECT sym_root, sym_suffix, ord, t_ns, bid, bid_size, ask, ask_size,
           CASE WHEN t_ns <= {open_ns} THEN 0
                ELSE (t_ns - {open_ns} + {bar_ns} - 1) / {bar_ns} END AS bar
    FROM judged
    WHERE t_ns <= {close_ns} AND reason IS NULL
),
ordered AS (
    SELECT sym_root, sym_suffix, t_ns, bar, bid, bid_size, ask, ask_size,
           lag(t_ns) OVER record AS t_prev,
           lead(t_ns) OVER record AS t_next,
           lead(bar) OVER record AS bar_next
    FROM kept
    WINDOW record AS (PARTITION BY sym_root, sym_suffix ORDER BY t_ns, ord)
),
updates AS (
    SELECT sym_root, sym_suffix, bar, count(*) AS n_updates
    FROM ordered
    WHERE bar >= 1
    GROUP BY sym_root, sym_suffix, bar
),
tied_instants AS (
    SELECT sym_root, sym_suffix, bar,
           count(*) AS n_records,
           count(DISTINCT bid) > 1 OR count(bid) NOT IN (0, count(*))
           OR count(DISTINCT bid_size) > 1 OR count(bid_size) NOT IN (0, count(*))
           OR count(DISTINCT ask) > 1 OR count(ask) NOT IN (0, count(*))
           OR count(DISTINCT ask_size) > 1 OR count(ask_size) NOT IN (0, count(*))
               AS several_states
    FROM ordered
    WHERE bar >= 1 AND (t_prev = t_ns OR t_next = t_ns)
    GROUP BY sym_root, sym_suffix, bar, t_ns
),
ambiguous AS (
    SELECT sym_root, sym_suffix, bar, sum(n_records) AS n_ambiguous_ties
    FROM tied_instants
    WHERE several_states
    GROUP BY sym_root, sym_suffix, bar
),
in_force AS (
    SELECT sym_root, sym_suffix, t_ns, bar, bid, bid_size, ask, ask_size,
           t_next, bar_next
    FROM ordered
    WHERE t_next IS NULL OR t_next > t_ns
),
last_in_bar AS (
    SELECT sym_root, sym_suffix, bar, bid, bid_size, ask, ask_size
    FROM in_force
    WHERE bar_next IS NULL OR bar_next > bar
),
spans AS (
    SELECT sym_root, sym_suffix, spread, bid_size, ask_size, t_from, t_to,
           (t_from - {open_ns}) / {bar_ns} + 1 AS bar_from,
           (t_to - {open_ns} + {bar_ns} - 1) / {bar_ns} AS bar_to
    FROM (
        SELECT sym_root, sym_suffix, bid_size, ask_size,
               ask - bid AS spread,
               greatest(t_ns, {open_ns}) AS t_from,
               coalesce(t_next, {close_ns}) AS t_to
        FROM in_force
        WHERE t_ns > {open_ns} OR t_next IS NULL OR t_next > {open_ns}
    ) AS bounded
    WHERE t_to > t_from
),
pieces AS (
    SELECT sym_root, sym_suffix, bar_from AS bar,
           spread, bid_size, ask_size, (t_to - t_from)::float8 AS dur
    FROM spans
    WHERE bar_from = bar_to
    UNION ALL
    SELECT spans.sym_root, spans.sym_suffix, k AS bar,
           spans.spread, spans.bid_size, spans.ask_size,
           (least(spans.t_to, {open_ns} + k * {bar_ns})
            - greatest(spans.t_from, {open_ns} + (k - 1) * {bar_ns}))::float8 AS dur
    FROM spans
    CROSS JOIN LATERAL generate_series(spans.bar_from, spans.bar_to) AS k
    WHERE spans.bar_from < spans.bar_to
),
time_weighted AS (
    SELECT sym_root, sym_suffix, bar,
           sum(spread * dur) FILTER (WHERE spread IS NOT NULL)
               / sum(dur) FILTER (WHERE spread IS NOT NULL) AS tw_spread,
           sum(bid_size * dur) FILTER (WHERE bid_size IS NOT NULL)
               / sum(dur) FILTER (WHERE bid_size IS NOT NULL) AS tw_bid_size,
           sum(ask_size * dur) FILTER (WHERE ask_size IS NOT NULL)
               / sum(dur) FILTER (WHERE ask_size IS NOT NULL) AS tw_ask_size
    FROM pieces
    GROUP BY sym_root, sym_suffix, bar
),
grid AS (
    SELECT filter_stats.sym_root, filter_stats.sym_suffix, k AS bar
    FROM filter_stats CROSS JOIN generate_series(0, {n_bars}) AS k
),
marked AS (
    SELECT grid.sym_root, grid.sym_suffix, grid.bar,
           l.bid, l.bid_size, l.ask, l.ask_size,
           count(l.bar) OVER (
               PARTITION BY grid.sym_root, grid.sym_suffix ORDER BY grid.bar
           ) AS held
    FROM grid
    LEFT JOIN last_in_bar AS l
        ON l.sym_root = grid.sym_root
       AND l.sym_suffix = grid.sym_suffix
       AND l.bar = grid.bar
),
snapshots AS (
    SELECT sym_root, sym_suffix, bar,
           first_value(bid) OVER quote AS bid,
           first_value(bid_size) OVER quote AS bid_size,
           first_value(ask) OVER quote AS ask,
           first_value(ask_size) OVER quote AS ask_size
    FROM marked
    WINDOW quote AS (PARTITION BY sym_root, sym_suffix, held ORDER BY bar)
)
SELECT s.sym_root, s.sym_suffix, s.bar,
       s.bid, s.bid_size, s.ask, s.ask_size,
       coalesce(u.n_updates, 0) AS n_updates,
       w.tw_spread, w.tw_bid_size, w.tw_ask_size,
       coalesce(a.n_ambiguous_ties, 0) AS n_ambiguous_ties,
       {filter_counts},
       count(*) OVER () AS page_rows
FROM snapshots AS s
LEFT JOIN updates AS u
    ON u.sym_root = s.sym_root
   AND u.sym_suffix = s.sym_suffix
   AND u.bar = s.bar
LEFT JOIN ambiguous AS a
    ON a.sym_root = s.sym_root
   AND a.sym_suffix = s.sym_suffix
   AND a.bar = s.bar
LEFT JOIN time_weighted AS w
    ON w.sym_root = s.sym_root
   AND w.sym_suffix = s.sym_suffix
   AND w.bar = s.bar
LEFT JOIN filter_stats AS f
    ON f.sym_root = s.sym_root
   AND f.sym_suffix = s.sym_suffix
   AND s.bar = 1
WHERE s.bar >= 1
) TO STDOUT WITH (FORMAT csv, HEADER true)"""


def _nanoseconds(clock: time) -> int:
    """Return a clock time as nanoseconds since midnight.

    Examples
    --------
    >>> _nanoseconds(time(9, 30))
    34200000000000
    """
    seconds = clock.hour * 3600 + clock.minute * 60 + clock.second
    return (seconds * 1_000_000 + clock.microsecond) * 1000


@dataclass(frozen=True)
class NbboBarsQuery:
    """The server-side resampling statement for one day table and one symbol batch.

    Parameters
    ----------
    day : datetime.date
        The trading day, which names the table.
    pairs : tuple of tuple of (str, str or None)
        The batch's ``(sym_root, sym_suffix)`` pairs.
    session : NbboBarsSession
        The day's session bounds, from ``NbboBarsRequest.session``.
    request : NbboBarsRequest
        The bar size and the quote filters.
    has_nano : bool
        Whether the day table has ``time_m_nano`` (from 2018-01-02).

    Raises
    ------
    ValueError
        If ``pairs`` is empty, or ``session`` is for another day.

    Examples
    --------
    >>> request = NbboBarsRequest("1m")
    >>> session = request.session(request.calendar(), date(2024, 1, 24))
    >>> query = NbboBarsQuery(
    ...     date(2024, 1, 24), (("AAPL", None),), session, request, has_nano=True
    ... )
    >>> statement = query.statement()  # a psycopg2.sql.Composed COPY statement
    """

    day: date
    pairs: tuple
    session: NbboBarsSession
    request: NbboBarsRequest
    has_nano: bool

    def __post_init__(self) -> None:
        """Freeze the pairs and check them against the session."""
        object.__setattr__(
            self, "pairs", tuple((str(root), suffix or None) for root, suffix in self.pairs)
        )
        if not self.pairs:
            raise ValueError(
                "NbboBarsQuery: no (sym_root, sym_suffix) pairs; refusing to "
                "build a statement over a whole complete_nbbo table."
            )
        if self.session.day != self.day:
            raise ValueError(
                f"NbboBarsQuery: the session is for {self.session.day}, the "
                f"table for {self.day}."
            )

    def _rules(self) -> list[tuple[str, sql.Composable]]:
        """Return ``(reason, condition)`` for each enabled rule, in priority order.

        Only the enabled rules appear. A null side is never a price, so each
        comparison with a null side counts as false.
        """
        policy = self.request.policy
        rules: list[tuple[str, sql.Composable]] = []
        if policy.drop_nonpositive_price:
            rules.append(
                (
                    "nonpositive_price",
                    sql.SQL("coalesce(bid <= 0, false) OR coalesce(ask <= 0, false)"),
                )
            )
        if policy.keep_qu_cond is not None:
            rules.append(
                (
                    "condition",
                    sql.SQL(
                        "NOT coalesce(qu_cond::text = ANY({codes}::text[]), false)"
                    ).format(codes=sql.Literal(list(policy.keep_qu_cond))),
                )
            )
        if policy.drop_crossed:
            rules.append(("crossed", sql.SQL("coalesce(bid > ask, false)")))
        if policy.drop_locked:
            rules.append(("locked", sql.SQL("coalesce(bid = ask, false)")))
        return rules

    def _reason(self) -> sql.Composable:
        """Return the expression naming a record's drop reason, NULL when it is kept.

        A record matching several rules gets the first, as in
        ``NbboFilterPolicy.reason``.
        """
        rules = self._rules()
        if not rules:
            return sql.SQL("NULL::text")
        return sql.SQL("CASE {whens} END").format(
            whens=sql.SQL(" ").join(
                sql.SQL("WHEN {condition} THEN {reason}").format(
                    condition=condition, reason=sql.Literal(reason)
                )
                for reason, condition in rules
            )
        )

    def _dropped_counts(self) -> sql.Composable:
        """Return the ``dropped_*`` count columns; a disabled rule counts 0."""
        enabled = {reason for reason, _ in self._rules()}
        return sql.SQL(", ").join(
            (
                sql.SQL("count(*) FILTER (WHERE reason = {reason}) AS {name}").format(
                    reason=sql.Literal(reason), name=sql.Identifier(name)
                )
                if reason in enabled
                else sql.SQL("0::bigint AS {name}").format(name=sql.Identifier(name))
            )
            for reason, name in _REASON_COLUMNS.items()
        )

    def statement(self) -> sql.Composed:
        """Return the ``COPY (...) TO STDOUT`` statement.

        Its CSV output has a header line and the ``RESULT_COLUMNS``.

        Returns
        -------
        psycopg2.sql.Composed
            The statement.

        Examples
        --------
        With ``query`` built as in the class example::

            query.statement()   # Composed([SQL("COPY (\\nWITH raw AS (...
        """
        bar_ns = self.request.bar_seconds * 1_000_000_000
        nano = (
            sql.SQL("coalesce(time_m_nano, 0)") if self.has_nano else sql.SQL("0")
        )
        return sql.SQL(_STATEMENT).format(
            nano=nano,
            table=WrdsSession.table_identifier(self.day),
            where=WrdsSession.where_clause(self.pairs),
            open_ns=sql.Literal(_nanoseconds(self.session.open_clock)),
            close_ns=sql.Literal(_nanoseconds(self.session.close_clock)),
            bar_ns=sql.Literal(bar_ns),
            n_bars=sql.Literal(self.session.n_bars),
            reason=self._reason(),
            dropped_counts=self._dropped_counts(),
            filter_counts=sql.SQL(", ").join(
                sql.Identifier("f", name) for name in FILTER_STATS_COUNTS
            ),
        )
