"""
utils/logger.py

AgentLogger: shared, append-only writer for log.txt, implementing the
AGENTS.md logging contract described in HANDOFF.md. A single instance
is created by the Orchestrator at startup and passed to every agent
and engine that needs to log a turn.
"""

import re
import threading
from datetime import datetime, timezone
from pathlib import Path

TOOL_NAME = "buy-or-wait-python"

# Obvious-secret redaction only — see assumption note. Matches common
# API key shapes and explicit key=value / key: value secret fields.
_REDACT_PATTERNS = [
    (re.compile(r"sk-ant-[a-zA-Z0-9\-_]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED_API_KEY]"),
    (
        re.compile(r"(?i)\b(api[_-]?key|token|secret)\b\s*[:=]\s*\S+"),
        r"\1=[REDACTED]",
    ),
]


def _redact(text: str) -> str:
    """
    Scrub obvious API keys/tokens/secrets from a string before logging.

    Input: text (str) — raw one-line summary destined for log.txt
    Output: str — same text with matched secret patterns replaced
    """
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _iso_now() -> str:
    """
    Input: none
    Output: str — current UTC time in ISO 8601 format
    """
    return datetime.now(timezone.utc).isoformat()


class AgentLogger:
    """
    Shared, append-only writer for log.txt.

    Every write goes through a single lock so concurrent callers (if
    the pipeline ever runs requests in parallel) can't interleave
    partial blocks or race on the turn counter. log.txt is opened in
    "a" mode only — it is never truncated or rewritten.
    """

    def __init__(self, log_path: str = "log.txt"):
        """
        Input: log_path (str) — path to the append-only log file
        Output: None
        """
        self._path = Path(log_path)
        self._lock = threading.Lock()
        self._turn = 0

    def log_session_start(self) -> None:
        """
        Writes the SESSION START block. Called once by the Orchestrator
        at the beginning of a run.

        Input: none
        Output: None (appends to log.txt)
        """
        block = (
            "SESSION START\n"
            f"tool={TOOL_NAME}\n"
            f"timestamp={_iso_now()}\n"
            "---\n"
        )
        with self._lock:
            self._write(block)

    def log_turn(self, input_summary: str, output_summary: str) -> int:
        """
        Writes one TURN block and increments the shared turn counter.
        Any agent or engine calls this via the shared AgentLogger
        instance — including for non-LLM events worth recording, e.g.
        an exchange-rate fallback substitution.

        Input:
            input_summary (str) — one-line summary, no secrets/PII
            output_summary (str) — one-line summary, no secrets/PII
        Output: int — the turn number just written
        """
        with self._lock:
            self._turn += 1
            turn_number = self._turn
            block = (
                f"TURN {turn_number}\n"
                f"tool={TOOL_NAME}\n"
                f"input={_redact(input_summary)}\n"
                f"output={_redact(output_summary)}\n"
                f"timestamp={_iso_now()}\n"
                "---\n"
            )
            self._write(block)
        return turn_number

    def _write(self, block: str) -> None:
        """
        Input: block (str) — fully formatted text to append
        Output: None

        Caller must already hold self._lock. Opens log.txt in append
        mode only ("a") — never "w" — per the append-only rule.
        """
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(block)
