"""
test_samples.py

Test harness, not part of the production pipeline. Runs the real
pipeline against sample_requests.csv's 25 solved examples and
compares the result to the ground-truth output columns already in
that same file.

Uses separate log/output/report paths so it never touches the real
log.txt, output.csv, or evaluation/usage_report.md that a real
submission run produces.

Run with: python3 code/test_samples.py, from the repository root.
"""

from orchestrator import Orchestrator

COMPARE_FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]

AMOUNT_TOLERANCE = 0.01


def _parse_payment_plan(text) -> list | None:
    """
    Input: text (any) - a payment_plan string, "none", or blank
    Output: list[tuple[str, float]] or None. None means it couldn't
        be parsed as a payment plan at all (caller falls back to
        exact string comparison in that case).
    """
    text = str(text).strip()
    if text.lower() == "none" or text == "" or text.lower() == "nan":
        return []
    entries = []
    for chunk in text.split("|"):
        parts = chunk.split(":")
        if len(parts) != 2:
            return None
        try:
            amount = float(parts[1])
        except ValueError:
            return None
        entries.append((parts[0].strip(), amount))
    return entries


def _fields_match(field: str, expected, actual) -> bool:
    """
    Input: field (str), expected (any, from sample_requests.csv),
        actual (any, from the pipeline's output row)
    Output: bool, True if they match. amount_safe_to_pay compares
        numerically within AMOUNT_TOLERANCE. payment_plan is parsed
        into (date, amount) pairs and compared numerically per pair,
        since the ground truth's own decimal formatting isn't even
        internally consistent (some rows show "25256", others show
        "1574.40"), exact string matching would fail on cosmetic
        differences a real evaluator likely doesn't care about. A
        blank/NaN value (pandas reads an empty CSV cell as NaN, our
        pipeline outputs '' for "never becomes safe") is treated as
        equivalent to '' on either side. Everything else compares as
        trimmed strings.
    """
    def _is_blank_value(v) -> bool:
        if v is None:
            return True
        if isinstance(v, float) and v != v:  # NaN != NaN is True
            return True
        return str(v).strip() == ""

    if _is_blank_value(expected) and _is_blank_value(actual):
        return True

    if field == "amount_safe_to_pay":
        try:
            return abs(float(expected) - float(actual)) <= AMOUNT_TOLERANCE
        except (TypeError, ValueError):
            return str(expected) == str(actual)

    if field == "payment_plan":
        parsed_expected = _parse_payment_plan(expected)
        parsed_actual = _parse_payment_plan(actual)
        if parsed_expected is None or parsed_actual is None:
            return str(expected).strip() == str(actual).strip()
        if len(parsed_expected) != len(parsed_actual):
            return False
        return all(
            d1 == d2 and abs(a1 - a2) <= AMOUNT_TOLERANCE
            for (d1, a1), (d2, a2) in zip(parsed_expected, parsed_actual)
        )

    return str(expected).strip() == str(actual).strip()


def main() -> None:
    """
    Input: none
    Output: None, prints a per-sample and summary report to stdout
    """
    orchestrator = Orchestrator(
        log_path="test_log.txt",
        output_path="test_output.csv",
        usage_report_path="evaluation/test_usage_report.md",
    )

    samples = orchestrator._data_loader.get_sample_requests()

    total = len(samples)
    fully_matched = 0
    field_mismatch_counts = {field: 0 for field in COMPARE_FIELDS}

    for sample in samples:
        request_id = sample["request_id"]
        try:
            actual = orchestrator._run_pipeline_once(request_id, request_override=sample)
        except Exception as exc:  # noqa: BLE001 - test harness, want to see every failure
            print(f"[{request_id}] FAILED: {type(exc).__name__}: {exc}")
            continue

        mismatches = []
        for field in COMPARE_FIELDS:
            expected = sample.get(field)
            got = actual.get(field)
            if not _fields_match(field, expected, got):
                mismatches.append((field, expected, got))
                field_mismatch_counts[field] += 1

        if mismatches:
            print(f"[{request_id}] MISMATCH:")
            for field, expected, got in mismatches:
                print(f"    {field}: expected={expected!r} got={got!r}")
        else:
            fully_matched += 1
            print(f"[{request_id}] OK")

    print()
    print(f"{fully_matched}/{total} samples fully matched on all compared fields")
    print("Per-field mismatch counts:")
    for field, count in field_mismatch_counts.items():
        print(f"    {field}: {count}")


if __name__ == "__main__":
    main()
