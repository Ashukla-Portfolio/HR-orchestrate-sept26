"""
engines/plan_engine.py

PlanEngine: generates all eligible, safe candidate plans, ranks them
by HANDOFF's 6-tier ordering, and produces the final
amount_safe_to_pay / affordability_status / recommended_payment_method
/ payment_plan / earliest_date_for_full_payment / spending_changes_needed
fields. decision_explanation is ExplanationAgent's job, not built here.

See HANDOFF.md's Plan Ranking, Partial Payment Rules, Spending
Changes, and Installment Plans sections, and the problem statement's
"Choosing Between Safe Plans" section.
"""

from engines.forecast_engine import ForecastEngine, group_key

MAX_SPENDING_CHANGES = 3
STOP_CAPABLE = {"stoppable", "reducible_or_stoppable"}
REDUCE_CAPABLE = {"reducible", "reducible_or_stoppable"}


def _apply_payments(timeline: list[dict], payments: list[tuple]) -> list[float]:
    """
    Input: timeline (list[dict]) - from ForecastEngine.build_timeline
           payments (list[tuple]) - (date, amount) pairs to deduct
    Output: list[float], the timeline's balances with each payment's
        amount deducted from its date onward (cumulative, since a
        payment reduces every subsequent day's balance too).
    """
    balances = [row["balance"] for row in timeline]
    dates = [row["date"] for row in timeline]
    for pay_date, amount in payments:
        for i, d in enumerate(dates):
            if d >= pay_date:
                balances[i] -= amount
    return balances


class PlanEngine:
    """
    One instance per run, holds a ForecastEngine to build and rebuild
    timelines (plain and with spending-change overrides applied).
    """

    def __init__(self, forecast_engine: ForecastEngine):
        """
        Input: forecast_engine (ForecastEngine)
        Output: None
        """
        self._forecast = forecast_engine

    def build_plan(self, context: dict) -> dict:
        """
        Input: context (dict) - the Context Dict Contract from
            ContextAgent
        Output: dict with keys: amount_safe_to_pay,
            affordability_status, recommended_payment_method,
            payment_plan, earliest_date_for_full_payment,
            spending_changes_needed
        """
        request = context["request"]
        profile = context["profile"]
        request_date = request["date"]
        desired_date = request["desired_date"]
        requested_amount = request["amount"]
        min_balance = profile["min_balance"]
        payment_methods = set(profile["payment_methods"])

        baseline_timeline = self._forecast.build_timeline(context)
        amount_safe = self._forecast.amount_safe_to_pay(baseline_timeline, min_balance, requested_amount)
        earliest_full_date = self._forecast.earliest_safe_date(baseline_timeline, requested_amount, min_balance)

        candidates = []

        if "full_payment" in payment_methods and amount_safe >= requested_amount:
            candidates.append(self._make_candidate(
                method="full_payment",
                payments=[(request_date, requested_amount)],
                desired_date=desired_date,
                changes=[],
            ))

        if "full_payment" in payment_methods and amount_safe < requested_amount:
            found = self._find_minimal_spending_changes(
                context,
                baseline_timeline,
                lambda tl: min(_apply_payments(tl, [(request_date, requested_amount)])) >= min_balance,
            )
            if found is not None:
                changes = found
                candidates.append(self._make_candidate(
                    method="full_payment",
                    payments=[(request_date, requested_amount)],
                    desired_date=desired_date,
                    changes=changes,
                ))

        if (
            request["allows_partial"]
            and "partial_payment" in payment_methods
            and 0 < amount_safe < requested_amount
            and earliest_full_date is not None
            and earliest_full_date <= desired_date
        ):
            remainder = requested_amount - amount_safe
            candidates.append(self._make_candidate(
                method="partial_payment",
                payments=[(request_date, amount_safe), (earliest_full_date, remainder)],
                desired_date=desired_date,
                changes=[],
            ))

        for opt in context["payment_options"]:
            if opt["method"] not in payment_methods:
                continue
            payments = [(item["date"], item["amount"]) for item in opt["schedule"]]
            if not payments:
                continue
            adjusted = _apply_payments(baseline_timeline, payments)
            if min(adjusted) >= min_balance:
                candidates.append(self._make_candidate(
                    method="installments",
                    payments=payments,
                    desired_date=desired_date,
                    changes=[],
                    total_paid_override=opt["total_payable"],
                    payment_option_id=opt["option_id"],
                ))

        if candidates:
            best = min(candidates, key=self._sort_key)
            return self._finalize(best, amount_safe, earliest_full_date)

        if "full_payment" in payment_methods and earliest_full_date is not None:
            return {
                "amount_safe_to_pay": round(amount_safe, 2),
                "affordability_status": "affordable_later",
                "recommended_payment_method": "wait",
                "payment_plan": "none",
                "earliest_date_for_full_payment": earliest_full_date.isoformat(),
                "spending_changes_needed": "none",
            }

        return {
            "amount_safe_to_pay": round(amount_safe, 2),
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": earliest_full_date.isoformat() if earliest_full_date else "",
            "spending_changes_needed": "none",
        }

    def _make_candidate(self, method, payments, desired_date, changes, total_paid_override=None, payment_option_id=None):
        """
        Input: method (str), payments (list[tuple]), desired_date
            (date), changes (list[dict]), total_paid_override
            (float or None), payment_option_id (str or None)
        Output: dict describing one candidate plan, carrying
            everything the 6-tier ranking needs.
        """
        last_date = max(d for d, _ in payments)
        total_paid = total_paid_override if total_paid_override is not None else sum(a for _, a in payments)
        return {
            "method": method,
            "payments": payments,
            "completes_by_deadline": last_date <= desired_date,
            "needs_changes": bool(changes),
            "changes": changes,
            "total_paid": total_paid,
            "start_date": min(d for d, _ in payments),
            "num_payments": len(payments),
            "payment_option_id": payment_option_id,
        }

    def _sort_key(self, candidate: dict):
        """
        Input: candidate (dict) - from _make_candidate
        Output: tuple, HANDOFF's 6-tier ranking as a sort key
            (ascending, lower is better at every tier)
        """
        return (
            0 if candidate["completes_by_deadline"] else 1,
            0 if not candidate["needs_changes"] else 1,
            candidate["total_paid"],
            candidate["start_date"],
            candidate["num_payments"],
            candidate["payment_option_id"] if candidate["payment_option_id"] is not None else "",
        )

    def _finalize(self, best: dict, amount_safe: float, earliest_full_date) -> dict:
        """
        Input: best (dict) - the winning candidate from _sort_key
               amount_safe (float), earliest_full_date (date or None)
               - both always the baseline figures, unaffected by
               which candidate won
        Output: dict, the final plan fields
        """
        method = best["method"]
        if method == "full_payment":
            status = "affordable_with_plan" if best["needs_changes"] else "affordable_now"
        else:
            status = "affordable_with_plan"

        payment_plan = "|".join(
            f"{d.isoformat()}:{round(a, 2)}" for d, a in sorted(best["payments"], key=lambda p: p[0])
        )

        spending_changes = "none"
        if best["changes"]:
            parts = []
            for c in best["changes"][:MAX_SPENDING_CHANGES]:
                if c["action"] == "stop":
                    parts.append(f"stop:{c['event_id']}")
                else:
                    parts.append(f"reduce_to:{c['event_id']}:{round(c['new_amount'], 2)}")
            spending_changes = "|".join(parts)

        return {
            "amount_safe_to_pay": round(amount_safe, 2),
            "affordability_status": status,
            "recommended_payment_method": method,
            "payment_plan": payment_plan,
            "earliest_date_for_full_payment": earliest_full_date.isoformat() if earliest_full_date else "",
            "spending_changes_needed": spending_changes,
        }

    def _eligible_flexible_events(self, context: dict) -> list[dict]:
        """
        Input: context (dict)
        Output: list[dict], one entry per event eligible for at least
            one spending-change action: {"event": event, "can_stop":
            bool, "can_reduce": bool}. Combines the event's own
            flexibility marking with the user's stated per-category
            willingness (categories_stop / categories_reduce), and
            excludes anything in categories_protect regardless of
            flexibility.
        """
        protect = set(context["profile"]["categories_protect"])
        stop_categories = set(context["profile"]["categories_stop"])
        reduce_categories = set(context["profile"]["categories_reduce"])

        result = []
        for event in context["events"]:
            if not event["is_recurring"]:
                continue
            if event["category"] in protect:
                continue
            can_stop = event["flexibility"] in STOP_CAPABLE and event["category"] in stop_categories
            can_reduce = event["flexibility"] in REDUCE_CAPABLE and event["category"] in reduce_categories
            if can_stop or can_reduce:
                result.append({"event": event, "can_stop": can_stop, "can_reduce": can_reduce})
        return result

    def _find_minimal_spending_changes(self, context: dict, baseline_timeline: list[dict], is_safe):
        """
        Input:
            context (dict)
            baseline_timeline (list[dict])
            is_safe (callable) - takes an adjusted timeline, returns
                bool
        Output: list[dict] of changes applied (each {"event_id",
            "action", "new_amount"}) if a safe combination of at most
            MAX_SPENDING_CHANGES changes was found, else None.

            Tries changes largest-cash-freed first, one at a time,
            stopping as soon as is_safe passes, so the result uses
            the fewest changes needed rather than always using the
            maximum allowed.
        """
        eligible = self._eligible_flexible_events(context)

        scored = []
        for entry in eligible:
            event = entry["event"]
            stop_gain = abs(event["amount_home"]) if entry["can_stop"] else -1
            reduce_gain = (abs(event["amount_home"]) - event["min_allowed"]) if entry["can_reduce"] else -1
            if stop_gain >= reduce_gain and entry["can_stop"]:
                scored.append({
                    "event_id": event["event_id"], "action": "stop", "new_amount": None,
                    "gain": stop_gain, "group_key": group_key(event),
                })
            elif entry["can_reduce"] and reduce_gain > 0:
                scored.append({
                    "event_id": event["event_id"], "action": "reduce_to", "new_amount": event["min_allowed"],
                    "gain": reduce_gain, "group_key": group_key(event),
                })

        scored.sort(key=lambda c: c["gain"], reverse=True)

        seen_groups = set()
        deduped = []
        for c in scored:
            if c["group_key"] in seen_groups:
                continue
            seen_groups.add(c["group_key"])
            deduped.append(c)

        applied = []
        overrides = {}
        for candidate in deduped[:MAX_SPENDING_CHANGES]:
            applied.append(candidate)
            overrides[candidate["group_key"]] = (candidate["action"], candidate["new_amount"])
            adjusted_timeline = self._forecast.build_timeline(context, overrides)
            if is_safe(adjusted_timeline):
                return applied

        return None
