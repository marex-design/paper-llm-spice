from __future__ import annotations

from typing import Any, Dict, List, Set


class ElectricalRules:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec

    def evaluate(self, netlist: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[Dict[str, str]] = []

        node_names = set(metadata.get("node_names", []))
        component_counts = metadata.get("component_counts", {})
        component_names = metadata.get("component_names", [])

        required_nodes = self.spec.get("sanity_checks", {}).get("required_nodes", [])
        required_elements = self.spec.get("sanity_checks", {}).get("required_elements", [])

        if "0" not in node_names:
            issues.append(self._issue("E-NODE-01", "Ground node 0 is missing from detected nodes."))

        for required_node in required_nodes:
            if not self._required_node_is_present(required_node, node_names):
                issues.append(
                    self._issue(
                        "E-NODE-02",
                        f"Required node '{required_node}' is missing."
                    )
                )

        allowed_components = set(
            comp.upper()
            for comp in self.spec.get("circuit_constraints", {}).get("allowed_components", [])
        )

        if allowed_components:
            present_types = set(component_counts.keys())
            unexpected = present_types - allowed_components
            if unexpected:
                issues.append(
                    self._issue(
                        "E-COMP-01",
                        f"Unexpected component types found: {sorted(unexpected)}"
                    )
                )

        self._check_required_elements(required_elements, component_counts, component_names, issues)

        return {
            "rule_family": "electrical",
            "ok": len(issues) == 0,
            "issues": issues,
        }

    def _required_node_is_present(self, required_node: str, node_names: Set[str]) -> bool:
        normalized_nodes = {self._normalize_node_name(n) for n in node_names}
        required_norm = self._normalize_node_name(required_node)

        if required_norm in normalized_nodes:
            return True

        aliases = self._get_node_aliases(required_norm)
        return any(alias in normalized_nodes for alias in aliases)

    def _get_node_aliases(self, required_node: str) -> Set[str]:
        alias_map = {
            "in": {
                "in", "vin", "input", "net_in", "input_node", "source_in", "src_in", "n_in", "1", "src"
            },
            "out": {
                "out", "vout", "output", "net_out", "output_node", "load_out", "n_out", "3", "2"
            },
            "mid": {
                "mid", "match_node", "int", "intermediate", "node_mid", "net_mid", "2", "3",
                # Ajouts pour compatibilité LLM (L-match)
                "n1", "out", "src"
            },
            "vin": {
                "vin", "vdc_pos", "vdc", "source", "source_node", "v_dc", "1", "in", "src"
            },
            "sw": {
                "sw", "sw_node", "switch", "switch_node", "switching_node", "v_sw", "node_switching",
                # Ajouts pour compatibilité LLM (Snubber)
                "n1", "n_switch", "drain"
            },
            "n1": {
                "n1", "load_node", "load_mid", "rload_out", "v_rl_mid", "node_load", "2", "3",
                # Ajouts pour UC3 - le LLM utilise souvent n_load
                "n_load", "load", "l1", "l_mid"
            },
        }
        return alias_map.get(required_node, {required_node})

    def _check_required_elements(
        self,
        required_elements: List[str],
        component_counts: Dict[str, int],
        component_names: List[str],
        issues: List[Dict[str, str]],
    ) -> None:
        names_lower = [name.lower() for name in component_names]

        mapping = {
            "source_voltage": lambda: component_counts.get("V", 0) >= 1,
            "dc_source": lambda: component_counts.get("V", 0) >= 1,
            "source_resistance": lambda: self._has_named_component(names_lower, ("rs", "rsource")) or component_counts.get("R", 0) >= 1,
            "load_resistance": lambda: self._has_named_component(names_lower, ("rload",)) or component_counts.get("R", 0) >= 1,
            "load_resistance_only": lambda: self._has_named_component(names_lower, ("rload",)),
            "capacitor": lambda: component_counts.get("C", 0) >= 1,
            "inductor": lambda: component_counts.get("L", 0) >= 1,
            "rl_load": lambda: component_counts.get("R", 0) >= 1 and component_counts.get("L", 0) >= 1,
            "parasitic_capacitance": lambda: self._has_named_component(names_lower, ("cpar",)) or component_counts.get("C", 0) >= 1,
            "snubber_resistance": lambda: self._has_named_component(names_lower, ("rsnub", "rsnubber")) or component_counts.get("R", 0) >= 1,
            "snubber_capacitance": lambda: self._has_named_component(names_lower, ("csnub", "csnubber", "cs")) or component_counts.get("C", 0) >= 1,
            "controlled_switch": lambda: component_counts.get("S", 0) >= 1,
        }

        for element in required_elements:
            checker = mapping.get(element)
            if checker is None:
                continue
            if not checker():
                issues.append(
                    self._issue(
                        "E-COMP-02",
                        f"Required element '{element}' was not detected."
                    )
                )

    @staticmethod
    def _normalize_node_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _has_named_component(names_lower: List[str], prefixes: tuple[str, ...]) -> bool:
        return any(any(name.startswith(prefix) for prefix in prefixes) for name in names_lower)

    @staticmethod
    def _issue(code: str, message: str) -> Dict[str, str]:
        return {"code": code, "message": message}