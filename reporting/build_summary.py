from __future__ import annotations

from typing import Dict, List


class SummaryBuilder:
    def build_markdown_summary(self, aggregated_results: List[Dict[str, object]]) -> str:
        lines: List[str] = []
        lines.append("# Experimental Summary")
        lines.append("")

        for case_result in aggregated_results:
            case_name = case_result["case"]
            baseline = case_result["baseline"]["counts"]
            eg = case_result["eg"]["counts"]

            lines.append(f"## {case_name}")
            lines.append("")

            lines.append("### Baseline")
            lines.append(f"- FAIL: {baseline.get('FAIL', 0)}")
            lines.append(f"- RUN: {baseline.get('RUN', 0)}")
            lines.append(f"- PASS: {baseline.get('PASS', 0)}")
            lines.append(f"- ROBUST_PASS: {baseline.get('ROBUST_PASS', 0)}")
            lines.append(f"- SPEC_INFEASIBLE: {baseline.get('SPEC_INFEASIBLE', 0)}")
            lines.append("")

            lines.append("### EG")
            lines.append(f"- FAIL: {eg.get('FAIL', 0)}")
            lines.append(f"- RUN: {eg.get('RUN', 0)}")
            lines.append(f"- PASS: {eg.get('PASS', 0)}")
            lines.append(f"- ROBUST_PASS: {eg.get('ROBUST_PASS', 0)}")
            lines.append(f"- SPEC_INFEASIBLE: {eg.get('SPEC_INFEASIBLE', 0)}")
            lines.append("")

            baseline_success = baseline.get("PASS", 0) + baseline.get("ROBUST_PASS", 0)
            eg_success = eg.get("PASS", 0) + eg.get("ROBUST_PASS", 0)

            if eg_success > baseline_success:
                lines.append("Observation: EG improved nominal or robust success for this use case.")
            elif eg.get("RUN", 0) > baseline.get("RUN", 0):
                lines.append("Observation: EG improved executability but not final acceptance for this use case.")
            else:
                lines.append("Observation: EG did not improve the final acceptance outcome for this use case.")

            lines.append("")

        return "\n".join(lines)