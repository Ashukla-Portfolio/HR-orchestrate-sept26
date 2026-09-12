"""
engines/forecast_engine.py

ForecastEngine: pure Python 90-day balance simulation. See
HANDOFF.md's ForecastEngine section and the problem statement's
"90-Day Safety Check" section for the governing rules.

Builds a baseline day-by-day balance timeline (no payment for the
request being evaluated applied yet), then exposes
earliest_safe_date and amount_safe_to_pay as closed-form
calculations over that timeline. PlanEngine layers candidate payments
on top of the baseline this engine produces, and can pass
group_overrides to build_timeline to test whether stopping or
reducing a recurring series would make an otherwise-unsafe plan safe.
"""

import calendar
from collections import defaultdict
from datetime import date, timedelta

WINDOW_DAYS = 90

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
    Output: tuple, the recurring-series grouping key, matching what
        DataLoader used to detect is_recurring: (type, description,
        category, direction). Exposed at module level so PlanEngine
        can identify which series a spending-change event_id belongs
        to, using exactly the same grouping.
    """
    return (event["type"], event.get("description"), event["category"], event["direction"])


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
        Output: list of groups, one per recurring series, grouped by
            (type, description, category, direction), the same key
            DataLoader used to detect is_recurring. Only events with
            is_recurring=True and an included status are grouped,
            excluded-status events don't anchor future projections.
        """
        groups = defaultdict(list)
        for event in events:
            if not event.get("is_recurring"):
                continue
            if not self._include_event(event):
                continue
            key = group_key(event)
            groups[key].append(event)
        return list(groups.values())

    def _project_recurring(self, group: list[dict], request_date: date, window_end: date, override=None):
        """
        Input:
            group (list[dict]) - one recurring series
            request_date, window_end (date)
            override - None, or ("reduce_to", new_amount) to project
                future occurrences at new_amount instead of the
                series' own amount. A ("stop", None) override is
                handled by the caller skipping this group entirely,
                not passed in here.
        Output: iterator of (date, signed_amount) for synthetic
            future occurrences, projected monthly from the group's
            latest known settlement_date at the same day-of-month,
            for any month within the window not already covered by an
            explicit row in the group.
        """
        dated = [(_as_date(e["settlement_date"]), e) for e in group]
        last_date, last_event = max(dated, key=lambda pair: pair[0])
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

    def build_timeline(self, context: dict, group_overrides: dict | None = None) -> list[dict]:
        """
        Input:
            context (dict) - the Context Dict Contract from
                ContextAgent
            group_overrides (dict or None) - maps a recurring series'
                group_key(event) to ("stop", None) or
                ("reduce_to", new_amount). Only affects recurring
                series; used by PlanEngine to test whether a
                spending change makes a candidate plan safe. None
                (the default) produces the plain baseline timeline.
        Output: list[dict], one entry per day from request_date
            through request_date + WINDOW_DAYS - 1, each
            {"date": date, "balance": float}.
        """
        group_overrides = group_overrides or {}
        request_date = _as_date(context["request"]["date"])
        window_end = request_date + timedelta(days=WINDOW_DAYS - 1)

        daily_deltas = defaultdict(float)

        for event in context["events"]:
            if not self._include_event(event):
                continue
            settlement_date = _as_date(event["settlement_date"])
            if not (request_date <= settlement_date <= window_end):
                continue

            override = group_overrides.get(group_key(event)) if event.get("is_recurring") else None
            if override and override[0] == "stop":
                continue
            if override and override[0] == "reduce_to":
                amount = abs(override[1])
                signed = amount if event["direction"] == "credit" else -amount
            else:
                signed = self._signed_amount(event)
            daily_deltas[settlement_date] += signed

        for group in self._group_recurring(context["events"]):
            key = group_key(group[0])
            override = group_overrides.get(key)
            if override and override[0] == "stop":
                continue
            for projected_date, amount in self._project_recurring(group, request_date, window_end, override):
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
