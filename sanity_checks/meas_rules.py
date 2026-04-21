from __future__ import annotations

import re
from typing import Any, Dict, List, Set


class MeasurementRules:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec

    def evaluate(self, netlist: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[Dict[str, str]] = []

        lines = self._lines(netlist)
        meas_lines = [line for line in lines if line.lower().startswith(".meas")]
        node_names = set(metadata.get("node_names", []))

        required_measurements = self.spec.get("sanity_checks", {}).get("required_measurements", [])
        measurement_specs = self.spec.get("measurements", [])

        if not meas_lines:
            issues.append(self._issue("E-MEAS-00", "No .meas directive found in netlist."))

        declared_measure_names = self._extract_measure_names(meas_lines)

        for required_name in required_measurements:
            if not self._has_compatible_measure_name(required_name, declared_measure_names):
                issues.append(
                    self._issue(
                        "E-MEAS-01",
                        f"Required measurement '{required_name}' is missing."
                    )
                )

        for meas in measurement_specs:
            target_node = meas.get("target_node")
            expression = meas.get("expression")

            if target_node and target_node not in node_names:
                # on reste strict ici si le spec demande explicitement un node
                issues.append(
                    self._issue(
                        "E-MEAS-02",
                        f"Measurement target node '{target_node}' is missing."
                    )
                )

            if expression:
                self._check_expression_references(expression, node_names, issues)

        return {
            "rule_family": "measurement",
            "ok": len(issues) == 0,
            "issues": issues,
        }

    @staticmethod
    def _lines(netlist: str) -> List[str]:
        return [
            line.strip()
            for line in netlist.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ]

    @staticmethod
    def _extract_measure_names(meas_lines: List[str]) -> List[str]:
        names: List[str] = []
        for line in meas_lines:
            tokens = line.split()
            if len(tokens) >= 3:
                names.append(tokens[2])
        return names

    def _has_compatible_measure_name(self, required_name: str, declared_names: List[str]) -> bool:
        required_norm = self._normalize_measure_name(required_name)

        for declared in declared_names:
            declared_norm = self._normalize_measure_name(declared)

            if declared_norm == required_norm:
                return True

            if self._is_alias_match(required_norm, declared_norm):
                return True

        return False

    def _is_alias_match(self, required_norm: str, declared_norm: str) -> bool:
        alias_groups = [
            # UC1
            {"gain_50", "vout_50"},
            {"gain_200", "vout_200"},
            {"gain_2k", "vout_2k"},

            # UC2
            {"zin_real_1m", "zin_real", "real_zin"},
            {"zin_imag_1m", "zin_imag", "imag_zin"},
            {"zin_mag_1m", "zin_mag", "mag_zin"},
            {"gamma_mag_1m", "gamma_mag"},

            # UC3
            {"vpeak_switch", "vpeak_sw", "vpeak_sw_node", "vpeak_snub"},
            {"snubber_energy", "esnub", "esnub_total", "esnub_diss"},
            {"settling_proxy", "late_vpeak", "residual_peak"},
        ]

        for group in alias_groups:
            if required_norm in group and declared_norm in group:
                return True

        return False

    @staticmethod
    def _normalize_measure_name(name: str) -> str:
        normalized = name.strip().lower()

        # standardiser quelques notations de fréquence
        normalized = normalized.replace("mhz", "m")
        normalized = normalized.replace("khz", "k")
        normalized = normalized.replace("hz", "")

        # enlever suffixes fréquents mais pas toujours essentiels
        removable_tokens = [
            "_db",
            "_v",
            "_j",
            "_value",
            "_total",
            "_node",
        ]
        for token in removable_tokens:
            normalized = normalized.replace(token, "")

        # harmoniser quelques préfixes/synonymes
        replacements = [
            ("vout_", "gain_"),
            ("vpeak_", "vpeak_"),
            ("esnub_", "esnub_"),
            ("snubber_energy_", "snubber_energy_"),
            ("switch_", ""),
        ]
        for old, new in replacements:
            normalized = normalized.replace(old, new)

        # normaliser séparateurs
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")

        return normalized

    def _check_expression_references(
        self,
        expression: str,
        node_names: Set[str],
        issues: List[Dict[str, str]],
    ) -> None:
        voltage_nodes = re.findall(r"v\(([^)]+)\)", expression, flags=re.IGNORECASE)

        for node_expr in voltage_nodes:
            # gère v(a,b) ou v(a)
            for node in [part.strip() for part in node_expr.split(",")]:
                if node and node not in node_names:
                    issues.append(
                        self._issue(
                            "E-MEAS-03",
                            f"Expression references missing voltage node '{node}'."
                        )
                    )

    @staticmethod
    def _issue(code: str, message: str) -> Dict[str, str]:
        return {"code": code, "message": message}