"""
utils/usage_tracker.py

UsageTracker: records per-call token usage for every LLM call in the
pipeline and aggregates it into evaluation/usage_report.md at the end
of a run, per the "Token Usage Tracking" section of HANDOFF.md.
"""

from collections import defaultdict
from pathlib import Path
from threading import Lock


class UsageTracker:
    """
    Shared, in-memory collector for per-call token usage records. One
    instance is created by the Orchestrator at startup and passed to
    every LLM-calling agent. Call `record()` once after every LLM
    call; call `write_report()` once, at the end of the full run.
    """

    def __init__(self):
        """
        Input: none
        Output: None
        """
        self._lock = Lock()
        self._records = []

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int,
        request_id: str,
        agent: str,
    ) -> None:
        """
        Records one LLM call's token usage. Every LLM call in the
        pipeline must call this exactly once, immediately after the
        call completes. Stores exactly the fields in HANDOFF's token
        usage schema.

        Input:
            model (str): model identifier used for the call
            input_tokens (int)
            output_tokens (int)
            thinking_tokens (int): 0 if ultrathink was not used
            request_id (str): the request_id this call was made for
            agent (str): which agent made the call, e.g. "ContextAgent"
        Output: None
        """
        record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "request_id": request_id,
            "agent": agent,
        }
        with self._lock:
            self._records.append(record)

    def write_report(self, report_path: str = "evaluation/usage_report.md") -> None:
        """
        Aggregates all recorded calls and writes evaluation/usage_report.md.
        Called once, by the Orchestrator, at the end of a full run.
        Overwrites any existing report. A fresh summary of
        the run, not an append-only log like log.txt.

        Input: report_path (str): where to write the report
        Output: None (writes the file)
        """
        with self._lock:
            records = list(self._records)

        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Usage Report", ""]

        if not records:
            lines.append("No LLM calls were recorded during this run.")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return

        by_agent = defaultdict(lambda: {"input": 0, "output": 0, "thinking": 0, "calls": 0})
        by_model = defaultdict(lambda: {"input": 0, "output": 0, "thinking": 0, "calls": 0})
        grand_input = grand_output = grand_thinking = 0

        for r in records:
            for bucket, key in ((by_agent, r["agent"]), (by_model, r["model"])):
                bucket[key]["input"] += r["input_tokens"]
                bucket[key]["output"] += r["output_tokens"]
                bucket[key]["thinking"] += r["thinking_tokens"]
                bucket[key]["calls"] += 1
            grand_input += r["input_tokens"]
            grand_output += r["output_tokens"]
            grand_thinking += r["thinking_tokens"]

        lines += [
            f"**Total calls:** {len(records)}  ",
            f"**Total input tokens:** {grand_input}  ",
            f"**Total output tokens:** {grand_output}  ",
            f"**Total thinking tokens:** {grand_thinking}  ",
            "",
            "## By Agent",
            "",
            "| Agent | Calls | Input | Output | Thinking |",
            "|---|---|---|---|---|",
        ]
        for agent, t in sorted(by_agent.items()):
            lines.append(f"| {agent} | {t['calls']} | {t['input']} | {t['output']} | {t['thinking']} |")

        lines += [
            "",
            "## By Model",
            "",
            "| Model | Calls | Input | Output | Thinking |",
            "|---|---|---|---|---|",
        ]
        for model, t in sorted(by_model.items()):
            lines.append(f"| {model} | {t['calls']} | {t['input']} | {t['output']} | {t['thinking']} |")

        lines += [
            "",
            "## Per-Call Detail",
            "",
            "| Request ID | Agent | Model | Input | Output | Thinking |",
            "|---|---|---|---|---|---|",
        ]
        for r in records:
            lines.append(
                f"| {r['request_id']} | {r['agent']} | {r['model']} | "
                f"{r['input_tokens']} | {r['output_tokens']} | {r['thinking_tokens']} |"
            )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
