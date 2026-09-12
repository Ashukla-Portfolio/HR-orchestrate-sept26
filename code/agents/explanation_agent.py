"""
agents/explanation_agent.py

ExplanationAgent: writes the one-sentence decision_explanation, using
Haiku so the result reads as a natural reply to the user's original
request rather than a recitation of computed fields.

HANDOFF's pipeline diagram restricts this agent's input to "the final
plan dict only, no raw financial data." Extended slightly here: the
user's original request_text, request_type, requested_amount, and
desired_completion_date are also included, since without them the
explanation can only describe numbers, not actually respond to what
was asked. No events, messages, images, or profile/balance data are
included, that restriction is kept.
"""

import json

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are the ExplanationAgent in an automated financial decision pipeline.

You are given the user's original request and the final decision
reached for it. Write one short, natural sentence that responds
directly to what the user asked, stating the recommendation and the
key financial fact behind it (the amount, the date, or the spending
change involved).

Do not mention agents, pipelines, models, or any internal process.
Write as if replying directly to the user. Do not invent any facts
beyond what you're given.

Respond with ONLY the sentence, no quotation marks, no preamble, no
markdown.
"""


class ExplanationAgentResponseError(Exception):
    """Raised when Haiku's response is empty or truncated."""


class ExplanationAgent:
    """
    Haiku-backed agent. One instance is created by the Orchestrator
    and reused across requests.
    """

    def __init__(self, client, logger, usage_tracker):
        """
        Input:
            client - an anthropic.Anthropic() instance
            logger (AgentLogger)
            usage_tracker (UsageTracker)
        Output: None
        """
        self._client = client
        self._logger = logger
        self._usage_tracker = usage_tracker

    def explain(self, request_context: dict, plan: dict, request_id: str) -> str:
        """
        Input:
            request_context (dict) - {"request_text", "request_type",
                "requested_amount", "desired_completion_date"} from
                the original request
            plan (dict) - the final plan dict from PlanEngine:
                amount_safe_to_pay, affordability_status,
                recommended_payment_method, payment_plan,
                earliest_date_for_full_payment,
                spending_changes_needed
            request_id (str) - for logging and usage tracking only
        Output: str, one sentence for decision_explanation

        Raises ExplanationAgentResponseError if the response is
        truncated or empty. No template fallback: a failure here
        should surface, not be silently papered over.
        """
        payload = {**request_context, **plan}

        response = self._client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        )

        if response.stop_reason == "max_tokens":
            raise ExplanationAgentResponseError(
                f"ExplanationAgent response for request_id={request_id!r} was truncated at max_tokens"
            )

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise ExplanationAgentResponseError(
                f"ExplanationAgent returned no text for request_id={request_id!r}"
            )

        self._usage_tracker.record(
            model=MODEL,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            thinking_tokens=0,
            request_id=request_id,
            agent="ExplanationAgent",
        )
        self._logger.log_turn(
            input_summary=f"generate decision_explanation for request_id={request_id}",
            output_summary=text,
        )

        return text
