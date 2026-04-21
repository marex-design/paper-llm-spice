from __future__ import annotations

from typing import Any, Dict, List


class TableBuilder:
    def __init__(self, hierarchical_counting: bool = True):
        self.hierarchical_counting = hierarchical_counting

    def build_iteration_table(self, aggregated_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for case_result in aggregated_results:
            case_name = case_result["case"]

            baseline = case_result["baseline"]
            eg = case_result["eg"]

            baseline_counts = baseline["counts"]
            eg_counts = eg["counts"]

            # Note explicative pour le papier
            note = ""
            if self.hierarchical_counting:
                note = "ROBUST_PASS candidates are also counted in PASS"

            rows.append({
                "use_case": case_name,
                "baseline_fail": baseline_counts.get("FAIL", 0),
                "baseline_run": baseline_counts.get("RUN", 0),
                "baseline_pass": baseline_counts.get("PASS", 0),
                "baseline_robust_pass": baseline_counts.get("ROBUST_PASS", 0),
                "baseline_total_pass": baseline.get("total_pass", 0),
                "baseline_spec_infeasible": baseline_counts.get("SPEC_INFEASIBLE", 0),
                "eg_fail": eg_counts.get("FAIL", 0),
                "eg_run": eg_counts.get("RUN", 0),
                "eg_pass": eg_counts.get("PASS", 0),
                "eg_robust_pass": eg_counts.get("ROBUST_PASS", 0),
                "eg_total_pass": eg.get("total_pass", 0),
                "eg_spec_infeasible": eg_counts.get("SPEC_INFEASIBLE", 0),
                "note": note,
            })

        return rows

    def build_best_candidates_table(self, aggregated_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for case_result in aggregated_results:
            case_name = case_result["case"]

            baseline_best = case_result["baseline"]["best_candidate"]
            eg_best = case_result["eg"]["best_candidate"]

            rows.append({
                "use_case": case_name,
                "baseline_best_candidate": baseline_best.get("candidate_id") if baseline_best else None,
                "baseline_best_decision": baseline_best.get("final_decision") if baseline_best else None,
                "eg_best_candidate": eg_best.get("candidate_id") if eg_best else None,
                "eg_best_decision": eg_best.get("final_decision") if eg_best else None,
            })

        return rows

    def build_paper_summary_table(self, aggregated_results: List[Dict[str, Any]]) -> str:
        """
        Génère un tableau au format papier (Markdown) avec la bonne hiérarchie.
        """
        lines = []
        lines.append("| Use Case | Mode | FAIL | RUN | PASS | ROBUST_PASS | Success Rate |")
        lines.append("|:---|:---|:---|:---|:---|:---|:---|")

        for case_result in aggregated_results:
            case_name = case_result["case"]
            
            for mode in ["baseline", "eg"]:
                mode_data = case_result[mode]
                counts = mode_data["counts"]
                total = mode_data["total_candidates"]
                
                fail = counts.get("FAIL", 0)
                run = counts.get("RUN", 0)
                # Ici, PASS inclut les ROBUST_PASS (car ils sont déjà comptés dedans)
                pass_total = counts.get("PASS", 0)
                robust = counts.get("ROBUST_PASS", 0)
                
                success_rate = (pass_total / total * 100) if total > 0 else 0
                
                mode_display = "BASELINE" if mode == "baseline" else "HITL"
                lines.append(
                    f"| {case_name.upper()} | {mode_display} | {fail} | {run} | {pass_total} | {robust} | {success_rate:.1f}% |"
                )

        return "\n".join(lines)