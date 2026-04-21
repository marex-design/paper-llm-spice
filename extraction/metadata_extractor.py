from __future__ import annotations

import re
from typing import Any, Dict, List, Set


class MetadataExtractor:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def extract(self, netlist: str) -> Dict[str, Any]:
        if not isinstance(netlist, str):
            raise TypeError("MetadataExtractor.extract expects a string")

        lines = self._normalize_lines(netlist)

        return {
            "line_count": len(lines),
            "has_end": self._has_directive(lines, ".end"),
            "has_ac": self._has_prefix(lines, ".ac"),
            "has_tran": self._has_prefix(lines, ".tran"),
            "has_meas": self._has_prefix(lines, ".meas"),
            "component_counts": self._count_components(lines),
            "component_names": self._list_component_names(lines),
            "node_names": sorted(self._extract_node_names(lines)),
        }

    def _normalize_lines(self, text: str) -> List[str]:
        raw_lines = [
            line.strip()
            for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ]

        filtered: List[str] = []
        inside_control = False

        for line in raw_lines:
            lower = line.lower()

            if lower.startswith(".control"):
                inside_control = True
                continue

            if lower.startswith(".endc"):
                inside_control = False
                continue

            if inside_control:
                continue

            filtered.append(line)

        return filtered

    @staticmethod
    def _has_directive(lines: List[str], directive: str) -> bool:
        directive = directive.lower()
        return any(line.lower() == directive for line in lines)

    @staticmethod
    def _has_prefix(lines: List[str], prefix: str) -> bool:
        prefix = prefix.lower()
        return any(line.lower().startswith(prefix) for line in lines)

    def _count_components(self, lines: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}

        for line in lines:
            if not self._is_component_line(line):
                continue

            comp_type = line[0].upper()
            counts[comp_type] = counts.get(comp_type, 0) + 1

        return counts

    def _list_component_names(self, lines: List[str]) -> List[str]:
        names: List[str] = []

        for line in lines:
            if not self._is_component_line(line):
                continue

            token = line.split()[0]
            names.append(token)

        return names

    def _extract_node_names(self, lines: List[str]) -> Set[str]:
        nodes: Set[str] = set()

        for line in lines:
            if not self._is_component_line(line):
                continue

            tokens = line.split()
            if len(tokens) < 3:
                continue

            comp_name = tokens[0]
            comp_type = comp_name[0].upper()
            node_count = self._guess_node_count(comp_type, tokens)

            for token in tokens[1 : 1 + node_count]:
                node = token.strip()
                if self._looks_like_node(node):
                    nodes.add(node)

        return nodes

    def _is_component_line(self, line: str) -> bool:
        if not line:
            return False

        lower = line.lower()

        if line.startswith("*") or line.startswith("."):
            return False

        if lower in {"run", "resume", "reset"}:
            return False

        if lower.startswith(("print ", "plot ", "echo ", "write ", "wrdata ")):
            return False

        first = line.split()[0]

        # vrais composants SPICE courants
        valid_prefixes = {"R", "C", "L", "V", "I", "D", "Q", "M", "X", "S"}

        if first[0].upper() not in valid_prefixes:
            return False

        # évite de prendre une phrase libre comme "RC Low-Pass Filter Design"
        # il faut que le nom du composant ressemble vraiment à un identifiant compact
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", first):
            return False

        return True

    @staticmethod
    def _guess_node_count(comp_type: str, tokens: List[str]) -> int:
        if comp_type in {"R", "C", "L", "V", "I", "D", "S"}:
            return 2
        if comp_type in {"Q"}:
            return 3
        if comp_type in {"M"}:
            return 4
        if comp_type in {"X"}:
            return max(0, len(tokens) - 2)
        return 2

    @staticmethod
    def _looks_like_node(token: str) -> bool:
        if not token:
            return False

        if token == "0":
            return True

        if token.startswith("{") and token.endswith("}"):
            return False

        if re.fullmatch(r"[-+]?\d+(\.\d+)?", token):
            return False

        return True