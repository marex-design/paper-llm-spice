from __future__ import annotations

import re
from typing import Dict


class MeasParser:
    def parse(self, log_text: str) -> Dict[str, float]:
        results: Dict[str, float] = {}

        pattern = re.compile(
            r"^\s*([\w\-]+)\s*=\s*([+-]?\d+(\.\d+)?(e[+-]?\d+)?)",
            re.IGNORECASE
        )

        for line in log_text.splitlines():
            match = pattern.match(line.strip())
            if match:
                name = match.group(1)
                value = float(match.group(2))
                results[name] = value

        return results