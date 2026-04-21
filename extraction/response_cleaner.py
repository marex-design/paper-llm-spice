from __future__ import annotations

import re
from typing import Any, Dict


class ResponseCleaner:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("ResponseCleaner.clean expects a string")

        cleaned = text
        cleaned = self._normalize_line_endings(cleaned)
        cleaned = self._strip_bom(cleaned)
        cleaned = self._strip_markdown_fences(cleaned)
        cleaned = self._strip_leading_trailing_whitespace(cleaned)
        cleaned = self._collapse_excess_blank_lines(cleaned)
        return cleaned

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _strip_bom(text: str) -> str:
        return text.lstrip("\ufeff")

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        stripped = text.strip()

        fence_pattern = re.compile(
            r"^\s*```(?:\w+)?\s*\n(?P<body>.*)\n\s*```\s*$",
            re.DOTALL,
        )
        match = fence_pattern.match(stripped)
        if match:
            return match.group("body")

        stripped = re.sub(r"^\s*```(?:\w+)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?\s*```\s*$", "", stripped)
        return stripped

    @staticmethod
    def _strip_leading_trailing_whitespace(text: str) -> str:
        return text.strip()

    @staticmethod
    def _collapse_excess_blank_lines(text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text)