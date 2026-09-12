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

# A second, separate recurring signal: a (category, direction)
# combination with at least this many total historical occurrences is
# treated as an ongoing recurring pattern too, even without a matching
# description or a consistent day-of-month. Catches high-frequency,
# variable-vendor spending (groceries, dining, transport) that the
# exact-match detection above can't, confirmed against real data
# where these categories use a different description every time and
# don't land on a consistent day of the month.
FREQUENT_CATEGORY_MIN_OCCURRENCES = 3


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


def _detect_recurring_event_ids(events: list[dict]) -> set[str]:
    """
    Input: events (list[dict]), all financial_events rows for one user
    Output: set[str], event_ids identified as recurring

    Groups events by (event_type, description, category, direction,
    currency). Within each group, if any two events fall in
    consecutive calendar months, land on the same day of the month
    (within tolerance), and have similar amounts, every event_id in
    that group is marked recurring.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        key = (
            event.get("event_type"),
            event.get("description"),
            event.get("category"),
            event.get("direction"),
            event.get("currency"),
        )
        groups[key].append(event)

    recurring_ids: set[str] = set()
    for group_events in groups.values():
        if len(group_events) < 2:
            continue
        found = False
        for i, a in enumerate(group_events):
            for b in group_events[i + 1:]:
                if abs(_month_index(a["event_date"]) - _month_index(b["event_date"])) != 1:
                    continue
                if not _day_position_matches(a["event_date"], b["event_date"]):
                    continue
                if not _amounts_similar(a["amount"], b["amount"]):
                    continue
                found = True
                break
            if found:
                break
        if found:
            recurring_ids.update(e["event_id"] for e in group_events)

    return recurring_ids


def _detect_frequent_category_ids(events: list[dict]) -> set[str]:
    """
    Input: events (list[dict]), all financial_events rows for one user
    Output: set[str], event_ids belonging to a (category, direction)
        combination with at least FREQUENT_CATEGORY_MIN_OCCURRENCES
        total occurrences. This flags the category as recurring at
        all (used for PlanEngine's spending-change eligibility); the
        request-relative decision of how much to project forward
        into a specific 90-day forecast is ForecastEngine's job, not
        this one, since that needs a lookback window anchored to the
        request date, which this user-level function doesn't have.
    """
    groups: dict[tuple, list[str]] = defaultdict(list)
    for event in events:
        key = (event.get("category"), event.get("direction"))
        groups[key].append(event["event_id"])

    result: set[str] = set()
    for ids in groups.values():
        if len(ids) >= FREQUENT_CATEGORY_MIN_OCCURRENCES:
            result.update(ids)
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
            user, in original order, each with is_recurring (bool)
            and recurrence_pattern (str or None) attached.
            recurrence_pattern is "monthly" (exact description + same
            day-of-month, e.g. rent, subscriptions, salary),
            "frequent" (3+ occurrences of the same category+direction
            regardless of description or day, e.g. groceries, dining,
            transport), or None. Exposed explicitly, not just as the
            is_recurring boolean, because ForecastEngine needs to
            know which projection mechanism applies to each event:
            some categories (groceries) have a few descriptions that
            coincidentally repeat exactly, which would otherwise get
            mis-grouped into the monthly mechanism even though they
            don't actually land on a consistent day of the month.
            financial_events.csv has no request_id column, so this is
            deliberately unfiltered by request, matching events to a
            request is ContextAgent's job, using related_event_id
            values from messages/images.
        """
        matches = self._events[self._events["user_id"] == user_id]
        records = matches.to_dict(orient="records")
        monthly_ids = _detect_recurring_event_ids(records)
        frequent_ids = _detect_frequent_category_ids(records) - monthly_ids
        for record in records:
            eid = record["event_id"]
            if eid in monthly_ids:
                record["recurrence_pattern"] = "monthly"
            elif eid in frequent_ids:
                record["recurrence_pattern"] = "frequent"
            else:
                record["recurrence_pattern"] = None
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
