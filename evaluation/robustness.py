from __future__ import annotations
from typing import Any, Dict, List

from evaluation.criteria import CriteriaEvaluator


class RobustnessEvaluator:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec
        self.criteria_evaluator = CriteriaEvaluator(spec)

    def evaluate(self, sweep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sweep_results:
            return {
                "ok": False,
                "reason": "No sweep results provided.",
                "sweep_results": [],
                "passed_count": 0,
                "failed_count": 0,
            }

        evaluated_runs: List[Dict[str, Any]] = []
        all_passed = True
        passed_count = 0
        failed_count = 0

        for run in sweep_results:
            run_success = run.get("success", False)
            variation = run.get("variation", {})
            meas = run.get("meas", {})

            if not run_success:
                result = {
                    "variation": variation,
                    "success": False,
                    "passed": False,
                    "reason": "Sweep simulation failed.",
                    "criteria": None,
                }
                all_passed = False
                failed_count += 1
            else:
                criteria_result = self.criteria_evaluator.evaluate_nominal(meas)
                passed = criteria_result["ok"]

                result = {
                    "variation": variation,
                    "success": True,
                    "passed": passed,
                    "reason": None if passed else "One or more nominal criteria failed under sweep condition.",
                    "criteria": criteria_result,
                }

                if passed:
                    passed_count += 1
                else:
                    all_passed = False
                    failed_count += 1

            evaluated_runs.append(result)

        return {
            "ok": all_passed,
            "sweep_results": evaluated_runs,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "total_count": len(evaluated_runs),
        }