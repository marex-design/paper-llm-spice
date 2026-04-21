from __future__ import annotations

from typing import Any, Dict, List


class NetlistExtractionError(Exception):
    pass


class NetlistExtractor:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.auto_append_end = self.config.get("auto_append_end", True)

    def extract(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("NetlistExtractor.extract expects a string")

        lines = self._normalize_lines(text)
        start_idx = self._find_netlist_start(lines)

        if start_idx is None:
            raise NetlistExtractionError("Could not detect the start of a SPICE netlist")

        end_idx = self._find_netlist_end(lines, start_idx)

        if end_idx is not None:
            netlist_lines = lines[start_idx : end_idx + 1]
            return "\n".join(netlist_lines).strip() + "\n"

        # Si .end est absent mais que le texte ressemble à un netlist valable,
        # on prend tout depuis le début détecté et on ajoute .end.
        candidate_lines = lines[start_idx:]

        if not self._looks_like_valid_partial_netlist(candidate_lines):
            raise NetlistExtractionError("Could not detect '.end' in the SPICE netlist")

        if self.auto_append_end:
            candidate_lines = self._trim_trailing_noise(candidate_lines)
            candidate_lines.append(".end")
            return "\n".join(candidate_lines).strip() + "\n"

        raise NetlistExtractionError("Could not detect '.end' in the SPICE netlist")

    @staticmethod
    def _normalize_lines(text: str) -> List[str]:
        return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def _find_netlist_start(self, lines: List[str]) -> int | None:
        for idx, line in enumerate(lines):
            if self._looks_like_spice_start(line):
                return idx
        return None

    def _find_netlist_end(self, lines: List[str], start_idx: int | None) -> int | None:
        if start_idx is None:
            return None

        for idx in range(start_idx, len(lines)):
            if lines[idx].strip().lower() == ".end":
                return idx
        return None

    def _looks_like_spice_start(self, line: str) -> bool:
        stripped = line.strip()

        if not stripped:
            return False

        if stripped.startswith("*"):
            return True

        spice_prefixes = (
            "v", "r", "c", "l", "d", "q", "m", "x", "s",
            ".",  # directives .ac .tran .model etc.
        )

        first_token = stripped.split()[0].lower()

        return first_token.startswith(spice_prefixes)

    def _looks_like_valid_partial_netlist(self, lines: List[str]) -> bool:
        """
        Heuristique simple :
        on considère qu'un netlist tronqué est récupérable s'il contient
        au moins un composant et au moins une directive de simulation ou de mesure.
        """
        has_component = False
        has_simulation_directive = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("*"):
                continue

            lower = line.lower()

            if lower.startswith((".ac", ".tran", ".dc", ".tf", ".op", ".meas", ".model", ".param")):
                has_simulation_directive = True

            if not line.startswith("."):
                first = line.split()[0][0].upper()
                if first in {"V", "R", "C", "L", "D", "Q", "M", "X", "S", "I"}:
                    has_component = True

        return has_component and has_simulation_directive

    @staticmethod
    def _trim_trailing_noise(lines: List[str]) -> List[str]:
        """
        Enlève les lignes vides finales.
        Garde le reste intact, car la sortie peut être simplement tronquée
        sans bruit explicatif.
        """
        trimmed = list(lines)
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
        return trimmed