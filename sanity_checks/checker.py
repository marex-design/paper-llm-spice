from __future__ import annotations

from typing import Any, Dict, List

from sanity_checks.electrical_rules import ElectricalRules
from sanity_checks.meas_rules import MeasurementRules
from sanity_checks.grid_rules import GridRules


class SanityChecker:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec
        self.electrical_rules = ElectricalRules(spec)
        self.measurement_rules = MeasurementRules(spec)
        self.grid_rules = GridRules(spec)

    def run(self, netlist: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        results = [
            self.electrical_rules.evaluate(netlist, metadata),
            self.measurement_rules.evaluate(netlist, metadata),
            self.grid_rules.evaluate(netlist, metadata),
        ]

        all_issues: List[Dict[str, str]] = []
        overall_ok = True

        for result in results:
            if not result.get("ok", False):
                overall_ok = False
            all_issues.extend(result.get("issues", []))

        return {
            "ok": overall_ok,
            "results_by_family": results,
            "issues": all_issues,
            "issue_count": len(all_issues),
        }