"""
engines/validator.py

ValidatorAgent: final schema and constraint enforcement before a row
is written to output.csv. See HANDOFF.md's ValidatorAgent section and
Non-Negotiable Rules, and the real problem statement's "Allowed
values" and "Choosing Between Safe Plans" sections.

Pure Python, no LLM involvement. Raises ValidationError with a
specific reason on the first check that fails, rather than silently
coercing or dropping a bad value. Takes the complete output row plus
the context dict it was derived from, since several checks (amount
bounds, installment schedule matching, flexible-only spending
changes) require cross-referencing the source data, not just the row
in isolation.
"""

from datetime import date

AFFORDABILITY_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
PAYMENT_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
STOP_CAPABLE = {"stoppable", "reducible_or_stoppable"}
REDUCE_CAPABLE = {"reducible", "reducible_or_stoppable"}
MAX_SPENDING_CHANGES = 3
AMOUNT_TOLERANCE = 0.01


class ValidationError(Exception):
    """Raised when an output row fails schema or constraint enforcement, with the specific reason."""


def _as_date(value):
    """
    Input: value (date or datetime-like)
    Output: date
    """
    return value.date() if hasattr(value, "date") else value


def _parse_date_str(text: str) -> date:
    """
    Input: text (str) - "YYYY-MM-DD"
    Output: date

    Raises ValidationError if text isn't in that exact shape.
    """
    parts = text.split("-")
    if len(parts) != 3:
        raise ValidationError(f"{text!r} is not a valid YYYY-MM-DD date")
    try:
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValidationError(f"{text!r} is not a valid YYYY-MM-DD date: {exc}") from exc


def _parse_payment_plan(text: str) -> list[tuple]:
    """
    Input: text (str) - "none" or "<YYYY-MM-DD>:<amount>|..."
    Output: list[tuple[date, float]], empty list if "none"

    Raises ValidationError on any entry that doesn't match the exact
    "<date>:<amount>" shape.
    """
    if text == "none":
        return []
    entries = []
    for chunk in text.split("|"):
        parts = chunk.split(":")
        if len(parts) != 2:
            raise ValidationError(f"payment_plan entry {chunk!r} is not '<date>:<amount>'")
        d = _parse_date_str(parts[0])
        try:
            amount = float(parts[1])
        except ValueError as exc:
            raise ValidationError(f"payment_plan entry {chunk!r} has a non-numeric amount") from exc
        entries.append((d, amount))
    return entries


def _parse_spending_changes(text: str) -> list[dict]:
    """
    Input: text (str) - "none" or "<change>|..." (max MAX_SPENDING_CHANGES)
    Output: list[dict], each {"action": "stop"|"reduce_to",
        "event_id": str, "new_amount": float | None}

    Raises ValidationError on malformed entries or too many entries.
    """
    if text == "none":
        return []
    chunks = text.split("|")
    if len(chunks) > MAX_SPENDING_CHANGES:
        raise ValidationError(
            f"spending_changes_needed has {len(chunks)} entries, max is {MAX_SPENDING_CHANGES}"
        )
    result = []
    for chunk in chunks:
        parts = chunk.split(":")
        if parts[0] == "stop" and len(parts) == 2:
            result.append({"action": "stop", "event_id": parts[1], "new_amount": None})
        elif parts[0] == "reduce_to" and len(parts) == 3:
            try:
                new_amount = float(parts[2])
            except ValueError as exc:
                raise ValidationError(f"spending change {chunk!r} has a non-numeric amount") from exc
            result.append({"action": "reduce_to", "event_id": parts[1], "new_amount": new_amount})
        else:
            raise ValidationError(
                f"spending change {chunk!r} is not 'stop:<event_id>' or 'reduce_to:<event_id>:<amount>'"
            )
    return result


class ValidatorAgent:
    """
    One instance per run, holds no per-request state.
    """

    def validate(self, row: dict, context: dict) -> dict:
        """
        Input:
            row (dict) - the 8 output.csv columns for one request:
                request_id, amount_safe_to_pay, affordability_status,
                recommended_payment_method, payment_plan,
                earliest_date_for_full_payment, spending_changes_needed,
                decision_explanation
            context (dict) - the Context Dict Contract this row was
                derived from
        Output: dict, the same row, unchanged, if every check passes

        Raises ValidationError with a specific reason on the first
        check that fails.
        """
        if row["request_id"] != context["request"]["id"]:
            raise ValidationError(
                f"row request_id={row['request_id']!r} does not match "
                f"context request id={context['request']['id']!r}"
            )

        request = context["request"]
        requested_amount = request["amount"]
        request_date = _as_date(request["date"])
        desired_date = _as_date(request["desired_date"])

        self._check_enums(row)
        self._check_amount_bounds(row, requested_amount)
        self._check_decision_explanation(row)

        payments = _parse_payment_plan(row["payment_plan"])
        changes = _parse_spending_changes(row["spending_changes_needed"])

        self._check_payment_plan_order(payments)
        self._check_spending_changes_against_context(changes, context)
        self._check_method_specific(row, payments, changes, context, requested_amount, request_date, desired_date)
        self._check_earliest_date(row, request_date)

        return row

    def _check_enums(self, row: dict) -> None:
        if row["affordability_status"] not in AFFORDABILITY_STATUSES:
            raise ValidationError(f"affordability_status {row['affordability_status']!r} is not a valid enum value")
        if row["recommended_payment_method"] not in PAYMENT_METHODS:
            raise ValidationError(
                f"recommended_payment_method {row['recommended_payment_method']!r} is not a valid enum value"
            )

    def _check_amount_bounds(self, row: dict, requested_amount: float) -> None:
        amount = row["amount_safe_to_pay"]
        if not (0 <= amount <= requested_amount + AMOUNT_TOLERANCE):
            raise ValidationError(
                f"amount_safe_to_pay={amount} violates "
                f"0 <= amount_safe_to_pay <= requested_amount={requested_amount}"
            )

    def _check_decision_explanation(self, row: dict) -> None:
        if not row["decision_explanation"] or not row["decision_explanation"].strip():
            raise ValidationError("decision_explanation is empty")

    def _check_payment_plan_order(self, payments: list[tuple]) -> None:
        dates = [d for d, _ in payments]
        if dates != sorted(dates):
            raise ValidationError("payment_plan entries are not in chronological order")

    def _check_spending_changes_against_context(self, changes: list[dict], context: dict) -> None:
        events_by_id = {e["event_id"]: e for e in context["events"]}
        seen_ids = set()
        for change in changes:
            event_id = change["event_id"]
            if event_id in seen_ids:
                raise ValidationError(f"event_id {event_id!r} is referenced more than once in spending_changes_needed")
            seen_ids.add(event_id)

            event = events_by_id.get(event_id)
            if event is None:
                raise ValidationError(f"spending change references unknown event_id {event_id!r}")
            if not event["is_recurring"]:
                raise ValidationError(f"spending change references non-recurring event_id {event_id!r}")

            if change["action"] == "stop":
                if event["flexibility"] not in STOP_CAPABLE:
                    raise ValidationError(
                        f"event_id {event_id!r} is not stop-capable (flexibility={event['flexibility']!r})"
                    )
            else:
                if event["flexibility"] not in REDUCE_CAPABLE:
                    raise ValidationError(
                        f"event_id {event_id!r} is not reduce-capable (flexibility={event['flexibility']!r})"
                    )
                new_amount = change["new_amount"]
                if new_amount >= abs(event["amount_home"]):
                    raise ValidationError(f"reduce_to amount for event_id {event_id!r} is not a reduction")
                if new_amount < event["min_allowed"]:
                    raise ValidationError(f"reduce_to amount for event_id {event_id!r} is below its min_allowed floor")

    def _check_method_specific(
        self, row: dict, payments: list[tuple], changes: list[dict], context: dict,
        requested_amount: float, request_date: date, desired_date: date,
    ) -> None:
        method = row["recommended_payment_method"]
        status = row["affordability_status"]

        if method in ("wait", "not_recommended"):
            if payments:
                raise ValidationError(f"payment_plan must be 'none' for recommended_payment_method={method!r}")
            expected_status = "affordable_later" if method == "wait" else "not_affordable"
            if status != expected_status:
                raise ValidationError(
                    f"affordability_status must be {expected_status!r} for method={method!r}, got {status!r}"
                )
            return

        if not payments:
            raise ValidationError(f"payment_plan must not be 'none' for recommended_payment_method={method!r}")

        if method == "full_payment":
            if status not in ("affordable_now", "affordable_with_plan"):
                raise ValidationError(f"affordability_status={status!r} invalid for full_payment")
            if status == "affordable_now" and changes:
                raise ValidationError("affordable_now must not require spending changes")
            if len(payments) != 1:
                raise ValidationError("full_payment must have exactly one payment_plan entry")
            pay_date, amount = payments[0]
            if pay_date != request_date:
                raise ValidationError("full_payment must be paid on request_date")
            if abs(amount - requested_amount) > AMOUNT_TOLERANCE:
                raise ValidationError("full_payment amount must equal requested_amount")

        elif method == "partial_payment":
            if status != "affordable_with_plan":
                raise ValidationError("partial_payment requires affordability_status=affordable_with_plan")
            if len(payments) != 2:
                raise ValidationError("partial_payment must have exactly two payment_plan entries")
            (first_date, first_amount), (second_date, second_amount) = payments
            if first_date != request_date:
                raise ValidationError("partial_payment's first payment must be on request_date")
            if second_date > desired_date:
                raise ValidationError("partial_payment's second payment is after desired_completion_date")
            if abs((first_amount + second_amount) - requested_amount) > AMOUNT_TOLERANCE:
                raise ValidationError("partial_payment's two payments must sum to requested_amount")
            amount_safe = row["amount_safe_to_pay"]
            if not (0 < amount_safe < requested_amount):
                raise ValidationError("partial_payment requires 0 < amount_safe_to_pay < requested_amount")
            if abs(first_amount - amount_safe) > AMOUNT_TOLERANCE:
                raise ValidationError("partial_payment's first payment must equal amount_safe_to_pay")

        elif method == "installments":
            if status != "affordable_with_plan":
                raise ValidationError("installments requires affordability_status=affordable_with_plan")
            if not self._matches_some_option(payments, context["payment_options"]):
                raise ValidationError("installment payment_plan does not exactly match any supplied payment option")

    def _matches_some_option(self, payments: list[tuple], payment_options: list[dict]) -> bool:
        for opt in payment_options:
            schedule = [(item["date"], item["amount"]) for item in opt["schedule"]]
            if len(schedule) != len(payments):
                continue
            if all(
                d1 == d2 and abs(a1 - a2) <= AMOUNT_TOLERANCE
                for (d1, a1), (d2, a2) in zip(schedule, payments)
            ):
                return True
        return False

    def _check_earliest_date(self, row: dict, request_date: date) -> None:
        text = row["earliest_date_for_full_payment"]
        if text == "":
            return
        d = _parse_date_str(text)
        if row["affordability_status"] == "affordable_now" and d != request_date:
            raise ValidationError("earliest_date_for_full_payment must equal request_date when affordable_now")
        if d < request_date:
            raise ValidationError("earliest_date_for_full_payment cannot be before request_date")
