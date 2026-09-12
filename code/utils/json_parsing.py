"""
utils/json_parsing.py

strip_json_fences: models occasionally wrap a JSON response in a
markdown code fence (```json ... ```) even when explicitly told not
to. Both ContextAgent and ImageAgent parse JSON out of a model
response, so this is factored out once rather than duplicated, and
applied defensively rather than trusting the prompt instruction alone
to hold every time.
"""


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
