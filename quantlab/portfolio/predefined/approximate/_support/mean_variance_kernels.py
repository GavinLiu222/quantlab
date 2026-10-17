"""Compiled kernels of the approximate backtest's mean-variance solve (numba).

``mean_variance.solve`` hands every chunk of decision bars to one of two
compiled loops, so no bar and no iteration returns to Python:

- ``solve_forward``: the bars in time order, each solved once from the
  holdings its book drifts to (``quantlab.execution.approximate.mark_book``:
  the last decided bar's weights at their fill prices, marked at this bar's
  close), warm-started from those holdings; a bar it holds leaves the book
  as it was;
- ``solve_independent``: bars whose decisions do not read the holdings, each
  from its caps, in parallel over the bars (``prange``) except on macOS,
  where they run in turn (see ``independent_driver``).

Each bar minimises ``-mu' w + risk_aversion / 2 * w' (U U' + D) w +
penalty * |w - holdings|_1`` on its candidate columns by accelerated proximal
gradient (FISTA with adaptive restart) with backtracking: the step's
curvature ``L`` starts below the bound on the budget's tangent space
(``{sum(dw) = 0}``; computed by the caller) and is lowered after every
accepted step and doubled when the sufficient-decrease test fails, never past
the bound, so a bar whose binding caps leave little curvature takes long
steps. The covariance products are reused through their linearity (one per
trial step). The prox is the capped simplex (long-only) or ``{sum(w) = 0,
|w|_1 <= gross, |w| <= cap}`` (long-short), with the L1 pull toward the
holdings; both are found by safeguarded Newton steps on their multipliers.
With a name-count cap the prox keeps the names whose zeroing costs it
most (long and short names counted together) and the momentum is off
(``_solve_bar``). A bar stops when its gradient-mapping residual is at most
``tolerance`` times the largest entry of its starting gradient, or after
``max_iterations``.

The parallel driver uses numba's thread pool: ``numba.set_num_threads(n)``
or ``NUMBA_NUM_THREADS`` caps it (on the training server, keep it within the
jobs' core budget). The code is cached in ``__pycache__`` (``cache=True``);
an install that cannot write there compiles again in each process.
"""

import sys

import numpy as np
from numba import njit, prange

from quantlab.execution.approximate import mark_book
from quantlab.portfolio.config import HELD_WEIGHT_TOLERANCE

#: Below this relative excess a long-only shift or a long-short sum is exact.
_SUM_TOLERANCE = 4 * np.finfo(np.float64).eps
#: A long-only book further than this from its budget after its shift's search is brought back to it.
_BUDGET_REPAIR = 1e-9


@njit(cache=True)
def _covariance_times(ut, specific, weights, load, out):
    """``out = (U U' + D) weights``; ``ut`` holds ``U'`` as ``[k, m]``, ``load`` is scratch ``[k]``."""
    k, m = ut.shape
    for j in range(k):
        acc = 0.0
        for i in range(m):
            acc += ut[j, i] * weights[i]
        load[j] = acc
    for i in range(m):
        out[i] = specific[i] * weights[i]
    for j in range(k):
        factor = load[j]
        if factor != 0.0:
            for i in range(m):
                out[i] += ut[j, i] * factor


@njit(cache=True)
def _project_long_only(values, cap, anchor, threshold, shift, out):
    """The prox of ``threshold * |w - anchor|_1`` over ``{sum(w) = 1, 0 <= w <= cap}``; returns its shift.

    Each entry is ``values - shift`` soft-thresholded toward its anchor,
    then clipped; the sum falls with the shift, piecewise linearly, so the
    shift is found by Newton steps from ``shift`` inside a bisection bracket.
    """
    m = values.shape[0]
    low = np.inf
    high = -np.inf
    for i in range(m):
        top = cap[i] if cap[i] > anchor[i] else anchor[i]
        low = min(low, values[i] - top - threshold)
        high = max(high, values[i] + threshold)
    tau = shift if low <= shift <= high else (low + high) / 2
    for _ in range(200):
        total = 0.0
        slope = 0
        for i in range(m):
            shifted = values[i] - tau
            gap = shifted - anchor[i]
            if abs(gap) > threshold:
                pulled = shifted - threshold * np.sign(gap)
                if 0.0 < pulled < cap[i]:
                    slope += 1
            else:
                pulled = anchor[i]
            out[i] = min(max(pulled, 0.0), cap[i])
            total += out[i]
        excess = total - 1.0
        if abs(excess) <= _SUM_TOLERANCE:
            return tau
        if excess > 0:
            low = tau
        else:
            high = tau
        middle = (low + high) / 2
        if low >= high or middle == low or middle == high:
            break
        newton = tau + excess / slope if slope > 0 else middle
        tau = newton if (slope > 0 and low < newton < high) else middle
    _restore_budget(cap, out)
    return tau


@njit(cache=True)
def _restore_budget(cap, out):
    """Bring a long-only book back to a sum of 1 when the shift's bracket collapsed before it got there.

    An excess is taken off every holding in proportion (a smaller weight
    stays within its cap), a shortfall spread over the room below the caps
    in proportion to it; a book within ``_BUDGET_REPAIR`` of its budget (the
    last bits of a collapsed bracket) is left as it is.
    """
    m = out.shape[0]
    total = 0.0
    for i in range(m):
        total += out[i]
    if abs(total - 1.0) <= _BUDGET_REPAIR:
        return
    if total > 1.0:
        for i in range(m):
            out[i] /= total
        return
    room = 0.0
    for i in range(m):
        room += cap[i] - out[i]
    if room <= 0.0:
        return
    share = min((1.0 - total) / room, 1.0)
    for i in range(m):
        out[i] += share * (cap[i] - out[i])


@njit(cache=True)
def _prox_entry(value, gross_multiplier, cap, anchor, threshold):
    """``argmin 1/2 (w - value)^2 + gross_multiplier |w| + threshold |w - anchor|`` on ``[-cap, cap]``; and whether it moves with ``value``."""
    if threshold == 0.0:
        w = np.sign(value) * max(abs(value) - gross_multiplier, 0.0)
        moving = abs(value) > gross_multiplier
    else:
        low = min(0.0, anchor)
        high = max(0.0, anchor)
        pulled = gross_multiplier + threshold
        middle = value - np.sign(anchor) * (gross_multiplier - threshold)
        if value > high + pulled:
            w = value - pulled
            moving = True
        elif value < low - pulled:
            w = value + pulled
            moving = True
        else:
            w = min(max(middle, low), high)
            moving = low < middle < high
    moving = moving and abs(w) < cap
    return min(max(w, -cap), cap), moving


@njit(cache=True)
def _zero_sum_shift(values, cap, anchor, threshold, gross_multiplier, shift, bound, out, moving):
    """At ``gross_multiplier``, the shift making the prox sum to 0, from ``shift``; returns it.

    ``bound`` brackets every shift that can matter.
    """
    m = values.shape[0]
    low = -(bound + gross_multiplier)
    high = bound + gross_multiplier
    tau = min(max(shift, low), high)
    for _ in range(200):
        total = 0.0
        slope = 0
        for i in range(m):
            w, moves = _prox_entry(values[i] - tau, gross_multiplier, cap[i], anchor[i], threshold)
            out[i] = w
            moving[i] = moves
            total += w
            if moves:
                slope += 1
        if abs(total) <= _SUM_TOLERANCE:
            return tau
        if total > 0:
            low = tau
        else:
            high = tau
        middle = (low + high) / 2
        if low >= high or middle == low or middle == high:
            return tau
        newton = tau + total / slope if slope > 0 else middle
        tau = newton if (slope > 0 and low < newton < high) else middle
    return tau


@njit(cache=True)
def _project_long_short(values, cap, anchor, threshold, gross, multipliers, out, moving):
    """The prox over ``{sum(w) = 0, |w|_1 <= gross, |w| <= cap}`` with the L1 pull; ``multipliers`` (shift, gross multiplier) updated.

    The gross falls as its multiplier grows, piecewise linearly with the
    shift re-solved, so the multiplier is found by Newton steps from the last
    call's, inside a bisection bracket.
    """
    m = values.shape[0]
    bound = 1.0
    largest = 0.0
    for i in range(m):
        largest = max(largest, abs(values[i]) + cap[i])
    bound += largest
    if threshold != 0.0:
        farthest = 0.0
        for i in range(m):
            farthest = max(farthest, abs(anchor[i]))
        bound += threshold + farthest
    gross_tolerance = 1e3 * _SUM_TOLERANCE * max(gross, 1.0)
    tau = multipliers[0]
    gross_multiplier = multipliers[1]
    low = 0.0
    high = np.inf
    for _ in range(200):
        tau = _zero_sum_shift(values, cap, anchor, threshold, gross_multiplier, tau, bound, out, moving)
        total = 0.0
        count = 0
        signs = 0.0
        for i in range(m):
            total += abs(out[i])
            if moving[i]:
                count += 1
                signs += np.sign(out[i])
        excess = total - gross
        if abs(excess) <= gross_tolerance or (gross_multiplier == 0.0 and excess <= 0.0):
            break
        if excess > 0:
            low = gross_multiplier
        else:
            high = gross_multiplier
        slope = count - signs * signs / count if count > 0 else 0.0
        newton = max(gross_multiplier + excess / slope, 0.0) if slope > 0 else -1.0
        if np.isfinite(high):
            middle = (low + high) / 2
            if middle == low or middle == high:
                break
            fallback = middle
        else:
            fallback = 2.0 * max(gross_multiplier, bound)
        accepted = slope > 0 and low <= newton <= high and newton != gross_multiplier
        gross_multiplier = newton if accepted else fallback
    multipliers[0] = tau
    multipliers[1] = gross_multiplier
    total = 0.0
    for i in range(m):
        total += abs(out[i])
    if total > gross:
        for i in range(m):
            out[i] *= gross / total


@njit(cache=True)
def _project_sparse(values, cap, anchor, threshold, long_only, gross, slots, multipliers, out, moving, kept, keys):
    """The rule's prox kept on at most ``slots`` names: those whose zeroing costs the prox most.

    The prox over every candidate is found first (``_project_long_only`` or
    ``_project_long_short``); when it holds more than ``slots`` names, each
    candidate's cost of zeroing is the prox's per-entry objective ``1/2 (w -
    u)^2 + g |w| + threshold |w - anchor|`` (``u`` the values at the prox's
    shift, ``g`` its gross multiplier, 0 long-only) at 0 less at its prox
    weight, the ``slots`` costliest are kept (a stable sort, so ties keep
    the column order), long and short names alike, and the prox runs again
    with the others' caps at 0 (``kept`` and ``keys`` are scratch for those
    caps and the costs). ``multipliers[0:2]`` hold the first projection's
    state, ``multipliers[2:4]`` the second's. Keeping the costliest names at
    the first projection's multipliers is not the exact projection onto the
    sparse set, so a step need not lower the objective and the kept names
    may swap back and forth; such a bar stops at ``max_iterations``,
    reported unconverged.
    """
    m = values.shape[0]
    first = multipliers[0:2]
    second = multipliers[2:4]
    if long_only:
        first[0] = _project_long_only(values, cap, anchor, threshold, first[0], out)
        gross_multiplier = 0.0
    else:
        _project_long_short(values, cap, anchor, threshold, gross, first, out, moving)
        gross_multiplier = first[1]
    count = 0
    for i in range(m):
        if out[i] != 0.0:
            count += 1
    if count <= slots:
        return
    tau = first[0]
    for i in range(m):
        if cap[i] > 0.0:
            u = values[i] - tau
            at_zero = 0.5 * u * u + threshold * abs(anchor[i])
            gap = out[i] - u
            at_prox = 0.5 * gap * gap + gross_multiplier * abs(out[i]) + threshold * abs(out[i] - anchor[i])
            keys[i] = at_prox - at_zero
        else:
            keys[i] = np.inf
    order = np.argsort(keys, kind="mergesort")
    kept[:] = 0.0
    for n in range(slots):
        kept[order[n]] = cap[order[n]]
    if long_only:
        second[0] = _project_long_only(values, kept, anchor, threshold, second[0], out)
    else:
        _project_long_short(values, kept, anchor, threshold, gross, second, out, moving)


@njit(cache=True)
def _prox(values, cap, anchor, threshold, long_only, gross, multipliers, out, moving, slots, kept, keys):
    """The rule's prox: long-only on the capped simplex, long-short on the gross ball; ``multipliers`` carries their state.

    With ``slots > 0`` on that set's books of at most ``slots`` names, long
    and short names counted together (``_project_sparse``).
    """
    if slots > 0:
        _project_sparse(values, cap, anchor, threshold, long_only, gross, slots, multipliers, out, moving, kept, keys)
    elif long_only:
        multipliers[0] = _project_long_only(values, cap, anchor, threshold, multipliers[0], out)
    else:
        _project_long_short(values, cap, anchor, threshold, gross, multipliers, out, moving)


@njit(cache=True)
def _linear_floor(mu, cap, penalty):
    """The lowest curvature a bar's step takes: a millionth of its linear terms per unit of cap.

    ``mu`` and the turnover penalty are in return per unit of weight and a
    curvature in return per unit of weight squared, so the floor makes the
    longest step move a weight by about a million times its largest cap.
    """
    scale = penalty
    widest = 0.0
    for i in range(mu.shape[0]):
        scale = max(scale, abs(mu[i]))
        widest = max(widest, cap[i])
    if scale == 0.0 or widest == 0.0:
        return 1.0
    return 1e-6 * scale / widest


@njit(cache=True)
def _objective(mu, weights, covariance_weights, risk_aversion):
    """``-mu' w + risk_aversion / 2 * w' Sigma w`` from ``Sigma w``."""
    value = 0.0
    for i in range(mu.shape[0]):
        value += weights[i] * (0.5 * risk_aversion * covariance_weights[i] - mu[i])
    return value


@njit(cache=True)
def _solve_bar(mu, cap, ut, specific, bound, start, anchor, settings, out):
    """One bar's weights into ``out`` from ``start``; returns the iterations, negative when it stopped unconverged.

    ``bound`` is the curvature bound on the budget's tangent space;
    ``settings`` is ``(risk_aversion, penalty, long_only, gross, tolerance,
    max_iterations, max_names)``, ``max_names`` 0 for no name-count cap.

    With ``max_names`` every prox is the sparse one, the first
    iterate is a projected gradient step from ``start`` (so a start of
    equal entries, such as cash, ranks the names by its gradient rather than
    by column order), and there is no momentum: on the non-convex set
    acceleration oscillates, so the iteration is plain projected gradient
    with the same backtracking, a local optimum.
    """
    risk_aversion, penalty, long_only, gross, tolerance, max_iterations, max_names = settings
    m = mu.shape[0]
    k = ut.shape[0]
    load = np.empty(k)
    ceiling = risk_aversion * bound
    lowest = ceiling * 1e-9
    floor = _linear_floor(mu, cap, penalty)
    if ceiling < floor:
        # Next to no risk: the bar is (nearly) a linear program, with no
        # curvature to scale the step. A step of the floor's curvature moves
        # a weight by about a million caps, so one prox lands on the
        # program's optimum while the prox's numbers stay far from overflow.
        ceiling = floor
        lowest = floor
    curvature = ceiling / 8.0
    # The prox's state; [2:4] the sparse prox's second projection (_project_sparse).
    multipliers = np.zeros(4)
    multipliers[0] = np.nan if long_only else 0.0
    if long_only:
        multipliers[2] = np.nan
    moving = np.zeros(m, dtype=np.bool_)
    kept = np.empty(m)
    keys = np.empty(m)

    current = np.empty(m)
    current_product = np.empty(m)
    if max_names > 0:
        stepped = np.empty(m)
        _covariance_times(ut, specific, start, load, current_product)
        for i in range(m):
            stepped[i] = start[i] - (risk_aversion * current_product[i] - mu[i]) / curvature
        _prox(
            stepped, cap, anchor, penalty / curvature, long_only, gross, multipliers, current, moving,
            max_names, kept, keys,
        )
    else:
        _prox(start, cap, anchor, penalty / curvature, long_only, gross, multipliers, current, moving, 0, kept, keys)
    _covariance_times(ut, specific, current, load, current_product)
    previous = current.copy()
    previous_product = current_product.copy()
    scale = 0.0
    for i in range(m):
        scale = max(scale, abs(risk_aversion * current_product[i] - mu[i]))
    scale = max(scale, 1e-300)

    extrapolated = np.empty(m)
    extrapolated_product = np.empty(m)
    gradient = np.empty(m)
    trial_point = np.empty(m)
    candidate = np.empty(m)
    candidate_product = np.empty(m)
    momentum = 0.0
    t = 1.0
    for iteration in range(1, max_iterations + 1):
        for i in range(m):
            extrapolated[i] = current[i] + momentum * (current[i] - previous[i])
            extrapolated_product[i] = current_product[i] + momentum * (current_product[i] - previous_product[i])
            gradient[i] = risk_aversion * extrapolated_product[i] - mu[i]
        extrapolated_value = _objective(mu, extrapolated, extrapolated_product, risk_aversion)
        curvature = max(curvature * 0.5, lowest)
        while True:
            for i in range(m):
                trial_point[i] = extrapolated[i] - gradient[i] / curvature
            _prox(
                trial_point, cap, anchor, penalty / curvature, long_only, gross, multipliers, candidate, moving,
                max_names, kept, keys,
            )
            _covariance_times(ut, specific, candidate, load, candidate_product)
            model = extrapolated_value
            for i in range(m):
                step = candidate[i] - extrapolated[i]
                model += gradient[i] * step + 0.5 * curvature * step * step
            value = _objective(mu, candidate, candidate_product, risk_aversion)
            if value <= model + 1e-15 * abs(model) or curvature >= ceiling:
                break
            curvature = min(2.0 * curvature, ceiling)
        residual = 0.0
        inner = 0.0
        for i in range(m):
            residual = max(residual, abs(extrapolated[i] - candidate[i]))
            inner += (extrapolated[i] - candidate[i]) * (candidate[i] - current[i])
        residual *= curvature
        if inner > 0 or max_names > 0:  # restart: the step went against the momentum; none under a cap
            t = 1.0
            momentum = 0.0
        else:
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_next
            t = t_next
        previous[:] = current
        previous_product[:] = current_product
        current[:] = candidate
        current_product[:] = candidate_product
        if residual <= tolerance * scale:
            out[:] = current
            return iteration
    out[:] = current
    return -max_iterations


@njit(cache=True)
def _caps(pool, selectable, holdings, strength, top_k, weight_cap, long_only, cap):
    """The bar's caps: ``weight_cap`` on its candidates; whether it is solvable.

    The candidates are the pooled symbols and the held selectable ones; with
    ``top_k > 0`` only the ``top_k`` strongest of them by ``strength``
    (ranked stably, so ties keep the column order) and the held ones.
    """
    m = pool.shape[0]
    for i in range(m):
        held = selectable[i] and abs(holdings[i]) > HELD_WEIGHT_TOLERANCE
        cap[i] = weight_cap if (pool[i] or held) else 0.0
    if top_k > 0:
        index = np.flatnonzero(cap > 0)
        keys = np.empty(index.shape[0])
        for n in range(index.shape[0]):
            keys[n] = -strength[index[n]]
        order = np.argsort(keys, kind="mergesort")
        for n in range(top_k, index.shape[0]):
            i = index[order[n]]
            if not (selectable[i] and abs(holdings[i]) > HELD_WEIGHT_TOLERANCE):
                cap[i] = 0.0
    count = 0
    for i in range(m):
        if cap[i] > 0:
            count += 1
    if count == 0:
        return False
    return (not long_only) or count * weight_cap >= 1.0 - 1e-12


@njit(cache=True)
def solve_forward(problem, book, rule, stopping):
    """Every bar in time order from the holdings its book drifts to; ``(weights, iterations)``.

    ``problem`` is ``(mu, pool, selectable, strength, ut, specific,
    bound)`` on the solved columns (``ut`` ``[bars, k, m]``, ``bound`` the
    tangent-space curvature bound per bar); ``book`` is ``(previous,
    previous_entry, close, entry, columns)`` on the full axis, ``columns``
    the solved columns' positions there; ``rule`` is ``(weight_cap,
    risk_aversion, penalty, long_only, gross, top_k, max_names)``
    (``max_names`` 0 for none) and ``stopping`` ``(tolerance,
    max_iterations)``. A held bar is a NaN row.
    """
    mu, pool, selectable, strength, ut, specific, bound = problem
    previous, previous_entry, close, entry, columns = book
    weight_cap, risk_aversion, penalty, long_only, gross, top_k, max_names = rule
    tolerance, max_iterations = stopping
    settings = (risk_aversion, penalty, long_only, gross, tolerance, max_iterations, max_names)
    bars, m = mu.shape
    symbols = previous.shape[0]
    weights = np.full((bars, m), np.nan)
    iterations = np.zeros(bars, dtype=np.int64)
    book_weights = previous.copy()
    book_entry = previous_entry.copy()
    marked = np.empty(symbols)
    holdings = np.empty(m)
    anchor = np.empty(m)
    cap = np.empty(m)
    row = np.empty(m)
    for b in range(bars):
        value = mark_book(book_weights, book_entry, close[b], marked)
        for j in range(m):
            holdings[j] = marked[columns[j]] / value if value > 0 else marked[columns[j]]
            anchor[j] = max(holdings[j], 0.0) if long_only else holdings[j]
        if not _caps(pool[b], selectable[b], holdings, strength[b], top_k, weight_cap, long_only, cap):
            continue
        iterations[b] = _solve_bar(mu[b], cap, ut[b], specific[b], bound[b], anchor, anchor, settings, row)
        weights[b] = row
        book_weights[:] = 0.0
        for j in range(m):
            book_weights[columns[j]] = row[j]
        book_entry[:] = entry[b]
    return weights, iterations


@njit(cache=True)
def _solve_independent_bar(b, problem, rule, stopping, weights, iterations):
    """Bar ``b`` from its caps, without holdings, into ``weights[b]`` and ``iterations[b]``."""
    mu, pool, ut, specific, bound = problem
    weight_cap, risk_aversion, long_only, gross, max_names = rule
    tolerance, max_iterations = stopping
    m = mu.shape[1]
    cap = np.empty(m)
    no_holdings = np.zeros(m)
    if not _caps(pool[b], pool[b], no_holdings, mu[b], 0, weight_cap, long_only, cap):
        return
    row = np.empty(m)
    settings = (risk_aversion, 0.0, long_only, gross, tolerance, max_iterations, max_names)
    # Under a cap from cash, whose first step ranks the names by mu; otherwise from the caps.
    start = no_holdings if max_names > 0 else cap
    iterations[b] = _solve_bar(mu[b], cap, ut[b], specific[b], bound[b], start, no_holdings, settings, row)
    weights[b] = row


@njit(parallel=True, cache=True)
def _solve_independent_parallel(problem, rule, stopping):
    """``solve_independent`` over numba's threads."""
    bars, m = problem[0].shape
    weights = np.full((bars, m), np.nan)
    iterations = np.zeros(bars, dtype=np.int64)
    for b in prange(bars):
        _solve_independent_bar(b, problem, rule, stopping, weights, iterations)
    return weights, iterations


@njit(cache=True)
def _solve_independent_serial(problem, rule, stopping):
    """``solve_independent`` one bar after another."""
    bars, m = problem[0].shape
    weights = np.full((bars, m), np.nan)
    iterations = np.zeros(bars, dtype=np.int64)
    for b in range(bars):
        _solve_independent_bar(b, problem, rule, stopping, weights, iterations)
    return weights, iterations


def solve_independent(problem, rule, stopping):
    """Every bar on its own from its caps, without holdings; ``(weights, iterations)``.

    ``problem`` is ``(mu, pool, ut, specific, bound)``, ``rule``
    ``(weight_cap, risk_aversion, long_only, gross, max_names)`` and ``stopping``
    ``(tolerance, max_iterations)``. In parallel over the bars, except on
    macOS: there torch and xgboost each load an OpenMP runtime that a third
    thread pool must not meet (see ``tests/conftest.py``).
    """
    driver = _solve_independent_serial if sys.platform == "darwin" else _solve_independent_parallel
    return driver(problem, rule, stopping)
