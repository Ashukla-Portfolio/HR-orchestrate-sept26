"""
orchestrator.py

Orchestrator: wires the full pipeline together. Loads all input CSVs
once, runs the complexity flag check in Python before every
ContextAgent call (per HANDOFF's explicit ownership of that check,
not ContextAgent's own decision), calls each engine in sequence,
validates every row before it's written, and produces output.csv and
evaluation/usage_report.md.

Error handling: the submission requires exactly one output row per
request_id, no omissions, so a request that fails is retried (the
whole per-request pipeline, not just one agent call, since a fresh
ContextAgent call can legitimately produce a different, valid result
on retry) up to MAX_ATTEMPTS times, and only falls back to a
conservative not_affordable row, clearly logged, if every attempt
fails. Never halts the whole run over one bad request, never
silently invents a confident-looking number.
"""

import csv
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from agents.context_agent import ContextAgent, determine_effort_level, gather_request_bundle
from agents.explanation_agent import ExplanationAgent
from agents.image_agent import ImageAgent
from data.loader import DataLoader
from engines.forecast_engine import ForecastEngine
from engines.plan_engine import PlanEngine
from engines.validator import ValidatorAgent
from utils.logger import AgentLogger
from utils.usage_tracker import UsageTracker

# Loaded here, not in main.py, so every entrypoint that constructs an
# Orchestrator gets ANTH_API_KEY regardless of which script does the
# importing. Explicit path (this file's own directory, code/.env) since
# dotenv's default cwd-relative lookup won't find it when run per the
# README's instructions (cwd is the repo root, not code/).
load_dotenv(Path(__file__).resolve().parent / ".env")

MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def _as_date(value):
    """
    Input: value (date or datetime-like)
    Output: date
    """
    return value.date() if hasattr(value, "date") else value


class Orchestrator:
    """
    One instance per run.
    """

    def __init__(
        self,
        data_dir: str = "dataset",
        log_path: str = "log.txt",
        output_path: str = "output.csv",
        usage_report_path: str = "evaluation/usage_report.md",
    ):
        """
        Input:
            data_dir (str) - directory containing the 7 input CSVs
            log_path (str) - path for the append-only log.txt
            output_path (str) - path to write final predictions to,
                repo root per the submission format, not dataset/
            usage_report_path (str) - path for the token usage report
        Output: None

        Reads the API key from ANTH_API_KEY, not the Anthropic SDK's
        default ANTHROPIC_API_KEY, since that's what's actually in
        this repo's .env.
        """
        self._logger = AgentLogger(log_path)
        self._usage_tracker = UsageTracker()
        self._data_loader = DataLoader(data_dir, self._logger)

        client = anthropic.Anthropic(api_key=os.environ["ANTH_API_KEY"])

        self._image_agent = ImageAgent(
            client, self._logger, self._usage_tracker, media_dir=f"{data_dir}/media/images"
        )
        self._context_agent = ContextAgent(
            client, self._data_loader, self._logger, self._usage_tracker, image_agent=self._image_agent
        )
        self._plan_engine = PlanEngine(ForecastEngine())
        self._explanation_agent = ExplanationAgent(client, self._logger, self._usage_tracker)
        self._validator = ValidatorAgent()

        self._output_path = output_path
        self._usage_report_path = usage_report_path

    def run(self) -> None:
        """
        Input: none
        Output: None. Writes output_path and usage_report_path.
        """
        self._logger.log_session_start()

        rows = [self._process_one(request_id) for request_id in self._data_loader.all_request_ids()]

        self._write_output(rows)
        self._usage_tracker.write_report(self._usage_report_path)

    def _process_one(self, request_id: str) -> dict:
        """
        Input: request_id (str)
        Output: dict, one validated output row. Retries the whole
            per-request pipeline up to MAX_ATTEMPTS times; falls back
            to a conservative row if every attempt fails.
        """
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                row = self._run_pipeline_once(request_id)
                self._logger.log_turn(
                    input_summary=f"processed request_id={request_id} (attempt {attempt})",
                    output_summary=(
                        f"method={row['recommended_payment_method']}, "
                        f"status={row['affordability_status']}"
                    ),
                )
                return row
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring
                last_error = exc
                self._logger.log_turn(
                    input_summary=f"request_id={request_id} attempt {attempt} failed",
                    output_summary=f"{type(exc).__name__}: {exc}",
                )
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_DELAY_SECONDS * attempt)

        return self._fallback_row(request_id, last_error)

    def _run_pipeline_once(self, request_id: str, request_override: dict | None = None) -> dict:
        """
        Input:
            request_id (str)
            request_override (dict or None) - passed through to
                gather_request_bundle; lets test code run this same
                pipeline against a sample_requests.csv row
        Output: dict, one validated output row

        One full attempt: gather -> ContextAgent -> PlanEngine ->
        ExplanationAgent -> ValidatorAgent. Raises on any failure,
        caller decides whether to retry.
        """
        bundle = gather_request_bundle(request_id, self._data_loader, request_override=request_override)
        effort = determine_effort_level(bundle)
        context = self._context_agent.build_context(bundle, effort)
        plan = self._plan_engine.build_plan(context)

        request_context = {
            "request_text": bundle["request"]["request_text"],
            "request_type": bundle["request"]["request_type"],
            "requested_amount": bundle["request"]["requested_amount"],
            "desired_completion_date": _as_date(bundle["request"]["desired_completion_date"]).isoformat(),
        }
        explanation = self._explanation_agent.explain(request_context, plan, request_id)

        row = {
            "request_id": request_id,
            "amount_safe_to_pay": plan["amount_safe_to_pay"],
            "affordability_status": plan["affordability_status"],
            "recommended_payment_method": plan["recommended_payment_method"],
            "payment_plan": plan["payment_plan"],
            "earliest_date_for_full_payment": plan["earliest_date_for_full_payment"],
            "spending_changes_needed": plan["spending_changes_needed"],
            "decision_explanation": explanation,
        }
        return self._validator.validate(row, context)

    def _fallback_row(self, request_id: str, error: Exception) -> dict:
        """
        Input: request_id (str), error (Exception) - the last failure
        Output: dict, a conservative, clearly-logged row

        Used only when every attempt at processing a request has
        failed. The submission requires exactly one row per
        request_id with no omissions, so a request that can't be
        reliably decided gets a safe fallback rather than being left
        out of output.csv entirely.
        """
        self._logger.log_turn(
            input_summary=f"request_id={request_id} exhausted all {MAX_ATTEMPTS} attempts",
            output_summary=f"falling back to conservative not_affordable row: {type(error).__name__}: {error}",
        )
        return {
            "request_id": request_id,
            "amount_safe_to_pay": 0,
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": "",
            "spending_changes_needed": "none",
            "decision_explanation": (
                "Unable to complete automated evaluation for this request "
                "after repeated attempts; recommend manual review."
            ),
        }

    def _write_output(self, rows: list[dict]) -> None:
        """
        Input: rows (list[dict])
        Output: None, writes output_path with exactly the required
            columns in the required order.
        """
        with open(self._output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row[col] for col in OUTPUT_COLUMNS})
