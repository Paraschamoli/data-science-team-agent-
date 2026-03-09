"""Output parsers for data science agent responses."""

import re
from typing import Any

from langchain_core.output_parsers import BaseOutputParser  # type: ignore[import]


class PythonOutputParser(BaseOutputParser[str]):
    """Parse Python code from LLM output."""

    def parse(self, text: str) -> str:
        """Extract Python code from text."""
        # Remove markdown code blocks
        code_pattern = r"```python\n(.*?)\n```"
        matches = re.findall(code_pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback: look for function definitions
        func_pattern = r"def\s+\w+\s*\([^)]*\)\s*:.*?(?=\ndef|\Z)"
        matches = re.findall(func_pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Final fallback: return everything between first def and end
        def_start = text.find("def ")
        if def_start != -1:
            return text[def_start:].strip()

        return text.strip()

    def get_format_instructions(self) -> str:
        """Return format instructions for the parser."""
        return "Return Python code in a ```python``` code block."


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract JSON from text."""
    try:
        import json

        # Look for JSON blocks
        json_pattern = r"```json\n(.*?)\n```"
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            return json.loads(matches[0])

        # Look for JSON objects in text
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                result = json.loads(match)
            except (json.JSONDecodeError, TypeError):
                continue
            else:
                return result
        else:
            return {}
    except Exception:
        return {}
