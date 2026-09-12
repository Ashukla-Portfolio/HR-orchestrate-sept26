"""
utils/json_parsing.py

strip_json_fences: models occasionally wrap a JSON response in a
markdown code fence (```json ... ```) even when explicitly told not
to. Both ContextAgent and ImageAgent parse JSON out of a model
response, so this is factored out once rather than duplicated, and
applied defensively rather than trusting the prompt instruction alone
to hold every time.
"""


def extract_json_object(text: str) -> str:
    """
    Input: text (str) - a model response that should be JSON, but may
        instead be prose that happens to contain a JSON object
        somewhere in it (seen in practice: the model reasoning out
        loud about an ambiguous case instead of complying with the
        "respond with only JSON" instruction)
    Output: str - the substring from the first "{" to the last "}",
        or the original text unchanged if no "{" is found at all.
        This is a last-resort fallback, tried only after a direct
        parse (and fence-stripping) has already failed.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def strip_json_fences(text: str) -> str:
    """
    Input: text (str) - a model response expected to be raw JSON
    Output: str - text with a leading/trailing ``` or ```json fence
        removed, if present. If there's no fence, returns text
        unchanged (aside from surrounding whitespace).
    """
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.split("\n")
    lines = lines[1:]  # drop the opening ``` or ```json line
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
