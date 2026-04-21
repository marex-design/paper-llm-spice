from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List


class SpiceLogParser:
    def parse(self, log_path: Path) -> Dict[str, Any]:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        warnings = self._extract_warnings(lines)
        errors = self._extract_errors(lines)

        return {
            "has_error": len(errors) > 0,
            "ran_simulation": self._ran_simulation(text, lines),
            "warnings": warnings,
            "errors": errors,
            "raw_text": text,
        }

    def _ran_simulation(self, text: str, lines: List[str]) -> bool:
        text_lower = text.lower()

        explicit_no_run_patterns = [
            "no simulations run",
            "no analysis requested",
            "no matching analyses",
            "error: no such command",
        ]
        if any(pattern in text_lower for pattern in explicit_no_run_patterns):
            return False

        positive_run_patterns = [
            "doing analysis",
            "initial transient solution",
            "transient analysis",
            "ac analysis",
            "dc transfer characteristic",
            "operating point",
            "measurement:",
            "cpu time since last call",
        ]
        if any(pattern in text_lower for pattern in positive_run_patterns):
            return True

        # Si des lignes .meas semblent avoir produit des résultats, on considère que ça a tourné
        if self._contains_measure_results(lines):
            return True

        # S'il y a un fichier log non vide mais uniquement des warnings,
        # on reste prudent : on ne valide pas automatiquement
        return False

    def _contains_measure_results(self, lines: List[str]) -> bool:
        for line in lines:
            stripped = line.strip()
            if "=" in stripped and not stripped.lower().startswith(("warning", "error", "note")):
                return True
        return False

    def _extract_warnings(self, lines: List[str]) -> List[str]:
        warnings: List[str] = []
        for line in lines:
            if "warning" in line.lower():
                warnings.append(line.strip())
        return warnings

    def _extract_errors(self, lines: List[str]) -> List[str]:
        errors: List[str] = []

        non_blocking_error_like_patterns = [
            "note:",
            "warning:",
            "can't parse 'vd': ignored",
            "can't parse 'vm': ignored",
        ]

        for line in lines:
            lower = line.lower().strip()

            if any(pattern in lower for pattern in non_blocking_error_like_patterns):
                continue

            blocking_patterns = [
                "error:",
                "fatal",
                "no simulations run",
                "no analysis requested",
                "no such vector",
                "unknown parameter",
                "unknown subckt",
                "singular matrix",
                "timestep too small",
                "unsupported",
                "can't find model",
                "missing .end",
            ]

            if any(pattern in lower for pattern in blocking_patterns):
                errors.append(line.strip())

        return errors