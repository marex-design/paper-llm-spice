from __future__ import annotations

from typing import Any, Dict, List

from evaluation.metrics import MetricsStore


class CriteriaEvaluator:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec

    def evaluate_nominal(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        store = MetricsStore(metrics)
        criteria = self.spec.get("nominal_acceptance_criteria", {}).get("all_of", [])

        results: List[Dict[str, Any]] = []
        all_passed = True

        for criterion in criteria:
            metric_name = criterion["metric"]
            operator = criterion["operator"]
            target_value = criterion["value"]

            actual_value = store.get_float(metric_name)

            if actual_value is None:
                passed = False
                reason = f"Metric '{metric_name}' is missing or non-numeric."
            else:
                passed = self._compare(actual_value, operator, target_value)
                reason = None if passed else (
                    f"Criterion failed: {metric_name} {operator} {target_value}, got {actual_value}"
                )

            if not passed:
                all_passed = False

            results.append({
                "metric": metric_name,
                "operator": operator,
                "target": target_value,
                "actual": actual_value,
                "passed": passed,
                "reason": reason,
                "rationale": criterion.get("rationale"),
            })

        return {
            "ok": all_passed,
            "criteria_results": results,
            "checked_count": len(results),
        }

    @staticmethod
    def _compare(actual: float, operator: str, target: float) -> bool:
        if operator == "<=":
            return actual <= target
        if operator == "<":
            return actual < target
        if operator == ">=":
            return actual >= target
        if operator == ">":
            return actual > target
        if operator == "==":
            return actual == target
        raise ValueError(f"Unsupported comparison operator: {operator}")