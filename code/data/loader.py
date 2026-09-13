"""
data/loader.py

DataLoader: loads the pipeline's input CSVs once at startup and
exposes filtered accessors keyed on the dimensions in the star
schema (request_id, user_id, event_id, exchange rate lookups).

Loads 7 of the 9 files named in HANDOFF.md's Dataset Schema section:
exchange_rates, financial_events, financial_profiles, images,
messages, request_payment_options, requests. output.csv is written
by this pipeline, not read by it. sample_requests.csv is a test
harness (requests plus ground truth output columns combined) and is
loaded separately by test code, not by DataLoader.
"""

from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from utils.logger import AgentLogger

# Recurrence detection tolerances. Not specified in HANDOFF.md, set
# per discussion: +/- 3 days on day-of-month, 15% relative tolerance
# on amount. Adjust here if testing against real data shows these are
# too loose or too tight.
DAY_OF_MONTH_TOLERANCE = 3
AMOUNT_SIMILARITY_TOLERANCE = 0.15

# A (category, direction) combination with at least this many total
# historical occurrences and more than one distinct description is
# treated as an ongoing "frequent" recurring pattern (groceries,
# dining, transport-style categories: a different vendor every time,
# no consistent day of the month).
FREQUENT_CATEGORY_MIN_OCCURRENCES = 3

# Tie-break for which copy of an exact duplicate to keep: settled over
# scheduled over pending, matching Conflict Resolution rule 3 ("a
# settled event over an estimate or forecast"). Anything else sorts
# last on this scale.
STATUS_PRECEDENCE = {"settled": 0, "scheduled": 1, "pending": 2}


class MissingExchangeRateError(Exception):
    """Raised when no exchange rate exists for a currency pair on or before the requested date."""


def _month_index(ts: pd.Timestamp) -> int:
    """
    Input: ts (pd.Timestamp)
    Output: int, increases by 1 per calendar month. Used to test
        whether two dates fall in consecutive months.
    """
    return ts.year * 12 + ts.month


def _day_position_matches(d1: pd.Timestamp, d2: pd.Timestamp) -> bool:
    """
    Input: d1, d2 (pd.Timestamp)
    Output: bool, True if d1 and d2 fall on the same day of their
        respective months within DAY_OF_MONTH_TOLERANCE. Also treats
        both dates landing within tolerance of their own month's end
        as a match, so e.g. Jan 31 matches Feb 28.
    """
    if abs(d1.day - d2.day) <= DAY_OF_MONTH_TOLERANCE:
        return True
    d1_from_end = d1.days_in_month - d1.day
    d2_from_end = d2.days_in_month - d2.day
    return d1_from_end <= DAY_OF_MONTH_TOLERANCE and d2_from_end <= DAY_OF_MONTH_TOLERANCE


def _amounts_similar(a1: float, a2: float) -> bool:
    """
    Input: a1, a2 (float)
    Output: bool, True if a1 and a2 are within AMOUNT_SIMILARITY_TOLERANCE
        of each other, relative to the larger of the two absolute values.
    """
    larger = max(abs(a1), abs(a2))
    if larger == 0:
        return a1 == a2
    return abs(a1 - a2) / larger <= AMOUNT_SIMILARITY_TOLERANCE


def _has_monthly_pattern(group_events: list[dict]) -> bool:
    """
    Input: group_events (list[dict]) - same category+direction,
        already confirmed to share one description
    Output: bool, True if any pair falls in consecutive calendar
        months, lands on the same day of the month (within
        tolerance), and has similar amounts.
    """
    for i, a in enumerate(group_events):
        for b in group_events[i + 1:]:
            if abs(_month_index(a["event_date"]) - _month_index(b["event_date"])) != 1:
                continue
            if not _day_position_matches(a["event_date"], b["event_date"]):
                continue
            if not _amounts_similar(a["amount"], b["amount"]):
                continue
            return True
    return False


def _detect_duplicate_event_ids(events: list[dict]) -> set[str]:
    """
    Input: events (list[dict]), all financial_events rows for one user
    Output: set[str], event_ids to discard as duplicates

    A deterministic backstop for "ignore duplicate records" (per the
    90-Day Safety Check section). ContextAgent's own LLM-based dedup
    judgment has no Python-level verification behind it, this catches
    the mechanical case directly: events sharing identical category,
    direction, amount, AND event_date are almost certainly the same
    transaction recorded twice, a genuine recurring bill has its own
    distinct date each time, so an exact date match alongside amount
    match is not something a legitimate recurring series would
    produce by coincidence.

    Within a duplicate group, keeps one copy (settled over scheduled
    over pending, per Conflict Resolution rule 3, "a settled event
    over an estimate or forecast"; ties broken by original row order)
    and flags the rest for removal.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        key = (
            event.get("category"),
            event.get("direction"),
            event.get("amount"),
            event.get("event_date"),
        )
        groups[key].append(event)

    duplicate_ids: set[str] = set()
    for group_events in groups.values():
        if len(group_events) < 2:
            continue
        ranked = sorted(group_events, key=lambda e: STATUS_PRECEDENCE.get(e.get("status"), 99))
        for event in ranked[1:]:
            duplicate_ids.add(event["event_id"])
    return duplicate_ids


def _classify_recurrence(events: list[dict]) -> dict[str, str | None]:
    """
    Input: events (list[dict]), all financial_events rows for one user
    Output: dict, event_id -> "monthly" | "frequent" | None

    Classifies at the (category, direction) level, not per event or
    per description, and never by category name.

    Primary check: find the most common ("dominant") description
    within the group, and require it to account for a strict majority
    (more than half) of the group's total occurrences, not merely be
    the single most common one among many roughly-equally-common
    vendor descriptions. Without the majority requirement, a small
    coincidentally-matching subset (e.g. 4 "Commuter pass" rows out of
    26 total, otherwise-varied transport events) can still swing an
    entire heterogeneous category to monthly, exactly the failure
    this whole category-level approach was meant to avoid. With a
    true majority required, if that dominant description's own
    occurrences fall in consecutive months on the same day (within
    tolerance) with similar amounts, the WHOLE group (every event,
    including any differently-worded minority outlier, e.g. a
    "Scheduled utility debit" row alongside many "Municipal
    utilities" ones) is monthly.

    Fallback, only when the primary check finds no pattern: if any
    event in the group is a CREDIT with status "scheduled", trust the
    whole group as an ongoing monthly income series anchored on that
    event. This catches sparse income history (a new job: one
    prorated first paycheck, worded and amount-wise unlike the next
    one, too few rows to statistically confirm a pattern) without
    naming "salary": it's restricted to credit specifically because,
    checked against the real data, every debit-direction "scheduled"
    row observed was a one-off due bill (an outstanding balance, a
    payable), not a recurring signal, while every credit-direction
    "scheduled" row was the next confirmed instance of ongoing income.
    A scheduled debit doesn't carry the same "this will keep
    recurring" implication a scheduled credit does.

    Otherwise: 3+ total occurrences of the (category, direction)
    combination, regardless of description, is a frequent pattern
    (groceries, dining, transport-style spending: a different vendor
    each time, no consistent day of the month).
    """
    by_cat_dir: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        by_cat_dir[(event.get("category"), event.get("direction"))].append(event)

    result: dict[str, str | None] = {e["event_id"]: None for e in events}

    for (_category, direction), group_events in by_cat_dir.items():
        description_counts: dict = defaultdict(int)
        for e in group_events:
            description_counts[e.get("description")] += 1
        dominant_description = max(description_counts, key=description_counts.get)
        dominant_events = [e for e in group_events if e.get("description") == dominant_description]
        is_majority = len(dominant_events) > len(group_events) / 2

        is_monthly = is_majority and len(dominant_events) >= 2 and _has_monthly_pattern(dominant_events)

        if not is_monthly and direction == "credit":
            if any(e.get("status") == "scheduled" for e in group_events):
                is_monthly = True

        if is_monthly:
            for e in group_events:
                result[e["event_id"]] = "monthly"
        elif len(group_events) >= FREQUENT_CATEGORY_MIN_OCCURRENCES:
            for e in group_events:
                result[e["event_id"]] = "frequent"

    return result


class DataLoader:
    """
    Loads all pipeline input CSVs once into memory and exposes typed,
    filtered accessors. One instance is created by the Orchestrator
    at startup and shared across all agents and engines for a run.
    """

    def __init__(self, data_dir: str, logger: AgentLogger):
        """
        Input:
            data_dir (str) - directory containing the 7 input CSVs
            logger (AgentLogger) - shared logger, used to record
                exchange rate fallback substitutions
        Output: None
        """
        self._logger = logger
        base = Path(data_dir)

        self._requests = pd.read_csv(
            base / "requests.csv",
            parse_dates=["request_date", "desired_completion_date"],
        )
        self._profiles = pd.read_csv(base / "financial_profiles.csv")
        self._events = pd.read_csv(
            base / "financial_events.csv",
            parse_dates=["event_date", "settlement_date"],
        )
        self._messages = pd.read_csv(base / "messages.csv", parse_dates=["sent_at"])
        self._images = pd.read_csv(base / "images.csv")
        self._payment_options = pd.read_csv(
            base / "request_payment_options.csv",
            parse_dates=["first_payment_date"],
        )
        self._rates = pd.read_csv(
            base / "exchange_rates.csv", parse_dates=["rate_date"]
        ).sort_values("rate_date")

        self._base_dir = base
        self._sample_requests = None

        self._requests = self._requests.set_index("request_id", drop=False)
        self._profiles = self._profiles.set_index("user_id", drop=False)

    def all_request_ids(self) -> list[str]:
        """
        Input: none
        Output: list[str], every request_id in requests.csv, in
            original file order. Used by the Orchestrator to iterate
            over every request that needs a prediction.
        """
        return list(self._requests["request_id"])

    def get_sample_requests(self) -> list[dict]:
        """
        Input: none
        Output: list[dict], every row in sample_requests.csv, request
            fields plus the ground-truth output fields. Loaded lazily
            on first call, this is test-only data, never needed
            during a real submission run over requests.csv.
        """
        if self._sample_requests is None:
            path = self._base_dir / "sample_requests.csv"
            self._sample_requests = pd.read_csv(
                path, parse_dates=["request_date", "desired_completion_date"]
            )
        return self._sample_requests.to_dict(orient="records")

    def get_request(self, request_id: str) -> dict:
        """
        Input: request_id (str)
        Output: dict, the single matching row from requests.csv

        Raises KeyError if request_id is not found.
        """
        try:
            row = self._requests.loc[request_id]
        except KeyError:
            raise KeyError(f"No request found for request_id={request_id!r}")
        return row.to_dict()

    def get_profile(self, user_id: str) -> dict:
        """
        Input: user_id (str)
        Output: dict, the single matching row from financial_profiles.csv

        Raises KeyError if user_id is not found.
        """
        try:
            row = self._profiles.loc[user_id]
        except KeyError:
            raise KeyError(f"No profile found for user_id={user_id!r}")
        return row.to_dict()

    def get_events_for_user(self, user_id: str) -> list[dict]:
        """
        Input: user_id (str)
        Output: list[dict], all financial_events.csv rows for this
            user, in original order, EXCLUDING exact duplicates (see
            _detect_duplicate_event_ids), each with is_recurring
            (bool) and recurrence_pattern (str or None) attached, from
            _classify_recurrence (category-level classification, see
            its docstring). financial_events.csv has no request_id
            column, so this is deliberately unfiltered by request,
            matching events to a request is ContextAgent's job, using
            related_event_id values from messages/images.
        """
        matches = self._events[self._events["user_id"] == user_id]
        records = matches.to_dict(orient="records")

        duplicate_ids = _detect_duplicate_event_ids(records)
        if duplicate_ids:
            self._logger.log_turn(
                input_summary=f"deduplicate financial_events for user_id={user_id}",
                output_summary=f"removed {len(duplicate_ids)} duplicate event_id(s): {sorted(duplicate_ids)}",
            )
            records = [r for r in records if r["event_id"] not in duplicate_ids]

        patterns = _classify_recurrence(records)
        for record in records:
            record["recurrence_pattern"] = patterns[record["event_id"]]
            record["is_recurring"] = record["recurrence_pattern"] is not None
        return records

    def get_event(self, event_id: str) -> dict | None:
        """
        Input: event_id (str)
        Output: dict, or None if no such event exists (e.g. a
            linked_event_id chain pointing nowhere).
        """
        matches = self._events[self._events["event_id"] == event_id]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def get_messages_for_request(self, request_id: str) -> list[dict]:
        """
        Input: request_id (str)
        Output: list[dict], all messages.csv rows for this request,
            in original order. Empty list if none exist.
        """
        matches = self._messages[self._messages["request_id"] == request_id]
        return matches.to_dict(orient="records")

    def get_images_for_request(self, request_id: str) -> list[dict]:
        """
        Input: request_id (str)
        Output: list[dict], all images.csv rows for this request, in
            original order. Empty list if none exist.
        """
        matches = self._images[self._images["request_id"] == request_id]
        return matches.to_dict(orient="records")

    def get_payment_options_for_request(self, request_id: str) -> list[dict]:
        """
        Input: request_id (str)
        Output: list[dict], all request_payment_options.csv rows for
            this request, in original order. Empty list if none exist.
        """
        matches = self._payment_options[self._payment_options["request_id"] == request_id]
        return matches.to_dict(orient="records")

    def get_exchange_rate(self, as_of: date, from_currency: str, to_currency: str) -> float:
        """
        Input:
            as_of (date) - the date the rate is needed for
            from_currency (str)
            to_currency (str)
        Output: float, the exchange rate

        Looks for an exact (rate_date, from_currency, to_currency)
        match first. If none exists, falls back to the closest prior
        rate_date for that currency pair and logs the substitution
        via AgentLogger, never silently. Raises
        MissingExchangeRateError if no rate exists on or before
        as_of for that pair at all.
        """
        if from_currency == to_currency:
            return 1.0

        pair = self._rates[
            (self._rates["from_currency"] == from_currency)
            & (self._rates["to_currency"] == to_currency)
        ]

        exact = pair[pair["rate_date"] == pd.Timestamp(as_of)]
        if not exact.empty:
            return float(exact.iloc[0]["rate"])

        prior = pair[pair["rate_date"] < pd.Timestamp(as_of)]
        if prior.empty:
            raise MissingExchangeRateError(
                f"No {from_currency} to {to_currency} rate on or before {as_of}"
            )

        closest = prior.sort_values("rate_date").iloc[-1]
        used_date = closest["rate_date"].date()
        self._logger.log_turn(
            input_summary=f"resolve rate {from_currency} to {to_currency} for {as_of}",
            output_summary=f"no exact rate, substituted nearest prior {used_date}",
        )
        return float(closest["rate"])
