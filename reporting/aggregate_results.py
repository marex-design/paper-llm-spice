from __future__ import annotations

from typing import Any, Dict, List


class ResultsAggregator:
    def __init__(self, hierarchical_counting: bool = True):
        """
        Args:
            hierarchical_counting: Si True, ROBUST_PASS incrémente aussi PASS.
                                  Si False, comptage exclusif (ancien comportement).
        """
        self.hierarchical_counting = hierarchical_counting

    def aggregate_case(self, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        case_name = experiment_result["case"]
        baseline = experiment_result.get("baseline", [])
        eg = experiment_result.get("eg", experiment_result.get("hitl", []))

        return {
            "case": case_name,
            "baseline": self._aggregate_mode_results(baseline),
            "eg": self._aggregate_mode_results(eg),
        }

    def _aggregate_mode_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts = {
            "FAIL": 0,
            "RUN": 0,
            "PASS": 0,
            "ROBUST_PASS": 0,
            "SPEC_INFEASIBLE": 0,
        }

        best_candidate = None

        for item in results:
            label = item.get("final_decision", "FAIL")
            if label not in counts:
                counts[label] = 0
            counts[label] += 1

            # Hiérarchie : ROBUST_PASS est aussi un PASS
            if self.hierarchical_counting and label == "ROBUST_PASS":
                counts["PASS"] += 1

            if best_candidate is None:
                best_candidate = item
            else:
                best_candidate = self._pick_better(best_candidate, item)

        # Calculer le total des succès (PASS + ROBUST_PASS)
        total_pass = counts["PASS"] + counts["ROBUST_PASS"]

        return {
            "total_candidates": len(results),
            "counts": counts,
            "total_pass": total_pass,  # Nouveau champ
            "best_candidate": best_candidate,
        }

    def _pick_better(self, current: Dict[str, Any], challenger: Dict[str, Any]) -> Dict[str, Any]:
        ranking = {
            "FAIL": 0,
            "RUN": 1,
            "PASS": 2,
            "ROBUST_PASS": 3,
            "SPEC_INFEASIBLE": -1,
        }

        current_label = current.get("final_decision", "FAIL")
        challenger_label = challenger.get("final_decision", "FAIL")

        if ranking.get(challenger_label, 0) > ranking.get(current_label, 0):
            return challenger

        return current