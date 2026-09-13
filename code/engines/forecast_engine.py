"""
engines/forecast_engine.py

ForecastEngine: pure Python 90-day balance simulation. See
HANDOFF.md's ForecastEngine section and the problem statement's
"90-Day Safety Check" section for the governing rules.

Builds a baseline day-by-day balance timeline (no payment for the
request being evaluated applied yet), then exposes earliest_safe_date
and amount_safe_to_pay as closed-form calculations over that
timeline. PlanEngine layers candidate payments on top of the baseline
this engine produces.

Two separate recurring-pattern mechanisms feed the timeline:

- "Monthly" pattern: rent, utilities, subscriptions, debt payments,
  salary, same description every time, lands on the same day of
  the month. Grouped by (type, description, category, direction),
  projected forward monthly at that day-of-month. This is the
  original mechanism.
- "Frequent" pattern: groceries, dining, transport-style spending,
  a different description every time (different vendor/merchant),
  no consistent day-of-month, but clearly ongoing. Grouped by
  (category, direction) alone, using only occurrences within
  FREQUENT_LOOKBACK_DAYS before the request date (so a pattern that
  changed recently isn't dragged down by stale history), and only
  if that recent window has at least FREQUENT_MIN_OCCURRENCES
  occurrences. Projected forward at the average interval between
  those occurrences, using their average amount. Added after real
  test data showed these categories were being silently treated as
  one-off and never projected, systematically overestimating future
  balance.

Spending changes (from PlanEngine) are specified by a single
representative event_id, since that's what stop:<event_id> /
reduce_to:<event_id>:<amount> in the final output references. This
engine resolves that representative id to whichever full group
(monthly or frequent) it belongs to and applies the change to every
member, so a "stop" on one dining event correctly stops projecting
the whole ongoing dining pattern, not just that one row.
"""

import calendar
from collections import defaultdict
from datetime import date, timedelta

WINDOW_DAYS = 90
FREQUENT_MIN_OCCURRENCES = 3
FREQUENT_LOOKBACK_DAYS = 90

# Status filtering, decided earlier in this project: failed/
# cancelled/unrealized are always excluded. pending is conditional on
# direction, handled separately below.
EXCLUDED_STATUSES = {"failed", "cancelled", "unrealized"}


def _as_date(value):
    """
    Input: value (date or datetime-like, e.g. pandas Timestamp)
    Output: date
    """
    return value.date() if hasattr(value, "date") else value


# Public alias: PlanEngine needs this same conversion.
as_date = _as_date


def _signals_final_occurrence(description) -> bool:
    """
    Input: description (str or None) - an event's description field
    Output: bool, True if the description signals this is the last
        occurrence of its series (e.g. "Final employer payroll"),
        meaning it should not be projected forward. Confirmed against
        the real dataset: this exact signal appears for multiple
        users, always on the latest known occurrence of an otherwise
        ordinary monthly series, a deliberate "this income/expense
        ends here" marker, not a category-specific label.
    """
    return description is not None and "final" in str(description).lower()


def _add_one_month_same_day(d: date, anchor_day: int) -> date:
    """
    Input: d (date), anchor_day (int)
    Output: date, one calendar month after d, landing on anchor_day
        if that day exists in the target month, otherwise the last
        day of that month (so a 31st-of-the-month recurrence still
        lands somewhere sensible in February).
    """
    year = d.year + (d.month // 12)
    month = d.month % 12 + 1
    days_in_target = calendar.monthrange(year, month)[1]
    day = min(anchor_day, days_in_target)
    return date(year, month, day)


def group_key(event: dict) -> tuple:
    """
    Input: event (dict) - a resolved event from the Context Dict
    Output: tuple, (category, direction). Every monthly-pattern
        category in this dataset (rent, utilities, subscriptions,
        debt payments, salary) is already category-specific enough
        per user that type/description aren't needed to disambiguate,
        and dropping them is what lets differently-worded rows for
        the same underlying series (e.g. "Prorated first salary" vs
        "Next confirmed salary") group together correctly.
    """
    return (event["category"], event["direction"])


class ForecastEngine:
    """
    One instance per run is enough, it holds no per-request state.
    """

    def _include_event(self, event: dict) -> bool:
        """
        Input: event (dict) - a resolved event from the Context Dict
        Output: bool, True if this event should factor into the
            90-day forecast. failed/cancelled/unrealized are always
            excluded. pending credits are excluded (ignore pending
            credits, per the problem statement). pending debits are
            included, reserved as a prudent outflow. settled/
            scheduled are included.
        """
        status = event["status"]
        if status in EXCLUDED_STATUSES:
            return False
        if event["direction"] == "non_cash":
            return False
        if status == "pending" and event["direction"] == "credit":
            return False
        return True

    def _signed_amount(self, event: dict) -> float:
        """
        Input: event (dict)
        Output: float, positive for credit (adds to balance),
            negative for debit (subtracts). Standard accounting
            convention assumed, not stated explicitly in HANDOFF.
        """
        amount = abs(event["amount_home"])
        return amount if event["direction"] == "credit" else -amount

    def _group_recurring(self, events: list[dict]) -> list[list[dict]]:
        """
        Input: events (list[dict])
        Output: list of groups, one per monthly-pattern recurring
            series, grouped by (type, description, category,
            direction). Only events with recurrence_pattern ==
            "monthly" are considered, not the broader is_recurring
            flag: some categories (groceries) have a few descriptions
            that coincidentally repeat exactly without landing on a
            consistent day of the month, and those must go through
            the frequent mechanism instead, not this one.
        """
        groups = defaultdict(list)
        for event in events:
            if event.get("recurrence_pattern") != "monthly":
                continue
            if not self._include_event(event):
                continue
            key = group_key(event)
            groups[key].append(event)
        return [g for g in groups.values() if len(g) >= 2]

    def _project_recurring(self, group: list[dict], request_date: date, window_end: date, override=None):
        """
        Input:
            group (list[dict]) - one monthly-pattern recurring series
            request_date, window_end (date)
            override - None, or ("reduce_to", new_amount) to project
                future occurrences at new_amount instead of the
                series' own amount. A ("stop", None) override is
                handled by the caller skipping this group entirely.
        Output: iterator of (date, signed_amount) for synthetic
            future occurrences, projected monthly from the group's
            latest known settlement_date at the same day-of-month,
            for any month within the window not already covered by an
            explicit row in the group. Projects nothing at all if the
            latest occurrence's description signals it's the last one
            (see _signals_final_occurrence), e.g. "Final employer
            payroll" means no further salary should be assumed.
        """
        dated = [(_as_date(e["settlement_date"]), e) for e in group]
        last_date, last_event = max(dated, key=lambda pair: pair[0])

        if _signals_final_occurrence(last_event.get("description")):
            return

        known_dates = {d for d, _ in dated}
        anchor_day = last_date.day

        if override and override[0] == "reduce_to":
            amount = abs(override[1])
            signed_amount = amount if last_event["direction"] == "credit" else -amount
        else:
            signed_amount = self._signed_amount(last_event)

        cursor = last_date
        while True:
            cursor = _add_one_month_same_day(cursor, anchor_day)
            if cursor > window_end:
                break
            if cursor >= request_date and cursor not in known_dates:
                yield cursor, signed_amount

    def _group_frequent(self, events: list[dict], request_date: date) -> list[list[dict]]:
        """
        Input:
            events (list[dict])
            request_date (date)
        Output: list of groups, one per (category, direction)
            combination with at least FREQUENT_MIN_OCCURRENCES
            occurrences within FREQUENT_LOOKBACK_DAYS before
            request_date. Only events with recurrence_pattern ==
            "frequent" are considered (see _group_recurring's
            docstring for why is_recurring alone isn't precise
            enough). Catches high-frequency, variable-description,
            variable-amount spending (groceries, dining, transport-
            style categories) the monthly mechanism can't.
        """
        lookback_start = request_date - timedelta(days=FREQUENT_LOOKBACK_DAYS)
        candidates = defaultdict(list)
        for event in events:
            if event.get("recurrence_pattern") != "frequent":
                continue
            if not self._include_event(event):
                continue
            settlement = _as_date(event["settlement_date"])
            if lookback_start <= settlement < request_date:
                key = (event["category"], event["direction"])
                candidates[key].append(event)

        return [g for g in candidates.values() if len(g) >= FREQUENT_MIN_OCCURRENCES]

    def _project_frequent(self, group: list[dict], request_date: date, window_end: date, override=None):
        """
        Input:
            group (list[dict]) - one frequent-category group, already
                filtered to the lookback window by _group_frequent
            request_date, window_end (date)
            override - None, or ("reduce_to", new_amount)
        Output: iterator of (date, signed_amount), projected forward
            from the group's latest occurrence at the average
            interval between its occurrences, using either the
            group's average amount or the override amount. Projects
            nothing at all if the latest occurrence's description
            signals it's the last one (see _signals_final_occurrence).
        """
        dated = sorted(((_as_date(e["settlement_date"]), e) for e in group), key=lambda pair: pair[0])

        if _signals_final_occurrence(dated[-1][1].get("description")):
            return

        dates_only = [d for d, _ in dated]
        intervals = [(dates_only[i + 1] - dates_only[i]).days for i in range(len(dates_only) - 1)]
        avg_interval = max(1, round(sum(intervals) / len(intervals))) if intervals else 30

        direction = dated[-1][1]["direction"]
        if override and override[0] == "reduce_to":
            signed_amount = abs(override[1]) if direction == "credit" else -abs(override[1])
        else:
            amounts = [self._signed_amount(e) for _, e in dated]
            signed_amount = sum(amounts) / len(amounts)

        cursor = dates_only[-1]
        while True:
            cursor = cursor + timedelta(days=avg_interval)
            if cursor > window_end:
                break
            if cursor >= request_date:
                yield cursor, signed_amount

    def _group_override(self, group: list[dict], resolved: dict):
        """
        Input: group (list[dict]), resolved (dict) - event_id ->
            (action, new_amount), already expanded to every member of
            whichever group was targeted
        Output: the override tuple if any member of this group is
            targeted, else None
        """
        for event in group:
            if event["event_id"] in resolved:
                return resolved[event["event_id"]]
        return None

    def build_timeline(self, context: dict, event_overrides: dict | None = None) -> list[dict]:
        """
        Input:
            context (dict) - the Context Dict Contract from
                ContextAgent
            event_overrides (dict or None) - maps a single
                representative event_id (as referenced by
                stop:<event_id> / reduce_to:<event_id>:<amount>) to
                ("stop", None) or ("reduce_to", new_amount). Resolved
                internally to every event in whichever group (monthly
                or frequent) that representative event_id belongs to.
                None (the default) produces the plain baseline
                timeline.
        Output: list[dict], one entry per day from request_date
            through request_date + WINDOW_DAYS - 1, each
            {"date": date, "balance": float}.
        """
        event_overrides = event_overrides or {}
        request_date = _as_date(context["request"]["date"])
        window_end = request_date + timedelta(days=WINDOW_DAYS - 1)

        monthly_groups = self._group_recurring(context["events"])
        frequent_groups = self._group_frequent(context["events"], request_date)

        resolved = {}
        for rep_event_id, action in event_overrides.items():
            for group in monthly_groups + frequent_groups:
                if any(e["event_id"] == rep_event_id for e in group):
                    for e in group:
                        resolved[e["event_id"]] = action
                    break
            else:
                # Representative isn't part of any detected group (a
                # single non-recurring event was somehow targeted);
                # apply the override to just that one event_id.
                resolved[rep_event_id] = action

        daily_deltas = defaultdict(float)

        for event in context["events"]:
            if not self._include_event(event):
                continue
            settlement_date = _as_date(event["settlement_date"])
            if not (request_date <= settlement_date <= window_end):
                continue

            override = resolved.get(event["event_id"])
            if override and override[0] == "stop":
                continue
            if override and override[0] == "reduce_to":
                amount = abs(override[1])
                signed = amount if event["direction"] == "credit" else -amount
            else:
                signed = self._signed_amount(event)
            daily_deltas[settlement_date] += signed

        for group in monthly_groups:
            override = self._group_override(group, resolved)
            if override and override[0] == "stop":
                continue
            for projected_date, amount in self._project_recurring(group, request_date, window_end, override):
                daily_deltas[projected_date] += amount

        for group in frequent_groups:
            override = self._group_override(group, resolved)
            if override and override[0] == "stop":
                continue
            for projected_date, amount in self._project_frequent(group, request_date, window_end, override):
                daily_deltas[projected_date] += amount

        balance = context["profile"]["balance"]
        timeline = []
        current_date = request_date
        while current_date <= window_end:
            balance += daily_deltas.get(current_date, 0.0)
            timeline.append({"date": current_date, "balance": balance})
            current_date += timedelta(days=1)

        return timeline

    def earliest_safe_date(self, timeline: list[dict], amount: float, min_balance: float):
        """
        Input:
            timeline (list[dict]) - from build_timeline
            amount (float) - the lump sum to test paying
            min_balance (float) - minimum_balance_to_keep
        Output: date or None, the first date in timeline where paying
            amount as a lump sum on that date keeps every day from
            that date through the end of the timeline at or above
            min_balance. None if no such date exists within the
            window.
        """
        n = len(timeline)
        for i in range(n):
            if all(timeline[j]["balance"] - amount >= min_balance for j in range(i, n)):
                return timeline[i]["date"]
        return None

    def amount_safe_to_pay(self, timeline: list[dict], min_balance: float, requested_amount: float) -> float:
        """
        Input:
            timeline (list[dict]) - from build_timeline
            min_balance (float) - minimum_balance_to_keep
            requested_amount (float)
        Output: float, the most payable on timeline's first date
            without breaking min_balance at any point in the window,
            capped at requested_amount and floored at 0.0. A lump sum
            paid on day 0 reduces every subsequent day's balance by
            the same amount, so this is a direct calculation from the
            minimum balance across the whole window, not a search.
        """
        if not timeline:
            return 0.0
        min_in_window = min(day["balance"] for day in timeline)
        headroom = min_in_window - min_balance
        return max(0.0, min(headroom, requested_amount))
