from __future__ import annotations

from typing import Any, Dict, List


class DecisionEngine:
    def __init__(self, spec: Dict[str, Any], experiment_config: Dict[str, Any] | None = None) -> None:
        self.spec = spec
        self.experiment_config = experiment_config or {}

    def classify(
        self,
        sanity_ok: bool,
        simulation_ok: bool,
        required_measurements_ok: bool,
        nominal_ok: bool,
        robustness_ok: bool | None = None,
    ) -> Dict[str, Any]:
        # Le sanity check 
        #  

        if not simulation_ok:
            if not sanity_ok:
                return self._decision(
                    "FAIL",
                    "Sanity checks reported issues and simulation did not complete successfully."
                )
            return self._decision(
                "FAIL",
                "Simulation failed or did not run successfully."
            )

        if not required_measurements_ok:
            return self._decision(
                "FAIL",
                "Simulation ran, but required measurements are missing or invalid."
            )

        if simulation_ok and not nominal_ok:
            if not sanity_ok:
                return self._decision(
                    "RUN",
                    "Simulation executed despite sanity warnings, but nominal criteria were not satisfied."
                )
            return self._decision(
                "RUN",
                "Simulation executed, but nominal criteria were not satisfied."
            )

        if simulation_ok and nominal_ok and robustness_ok is False:
            return self._decision(
                "PASS",
                "Nominal criteria satisfied, but robustness criteria failed."
            )

        if simulation_ok and nominal_ok and robustness_ok is True:
            return self._decision(
                "ROBUST_PASS",
                "Nominal and robustness criteria were satisfied."
            )

        if simulation_ok and nominal_ok and robustness_ok is None:
            return self._decision(
                "PASS",
                "Nominal criteria satisfied. No robustness result available."
            )

        return self._decision("FAIL", "Unhandled decision state.")

    def classify_spec_infeasibility(
        self,
        candidate_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        protocol = self.experiment_config.get("decision_protocol", {})
        infeasibility_cfg = protocol.get("spec_infeasibility", {})
        enabled = infeasibility_cfg.get("enabled", False)

        if not enabled:
            return {
                "triggered": False,
                "decision": None,
                "reason": "Spec infeasibility detection is disabled.",
            }

        min_executable = infeasibility_cfg.get("minimum_executable_candidates", 2)
        trigger_if_no_pass = infeasibility_cfg.get("trigger_if_no_pass_after_all_candidates", True)

        executable_count = 0
        pass_like_count = 0

        for item in candidate_results:
            final_label = item.get("final_decision")
            if final_label in {"RUN", "PASS", "ROBUST_PASS"}:
                executable_count += 1
            if final_label in {"PASS", "ROBUST_PASS"}:
                pass_like_count += 1

        if trigger_if_no_pass and executable_count >= min_executable and pass_like_count == 0:
            return {
                "triggered": True,
                "decision": "SPEC_INFEASIBLE",
                "reason": (
                    "Multiple candidates were executable or measurable, "
                    "but none satisfied the nominal specification."
                ),
            }

        return {
            "triggered": False,
            "decision": None,
            "reason": "Spec infeasibility conditions were not met.",
        }

    @staticmethod
    def _decision(label: str, reason: str) -> Dict[str, Any]:
        return {
            "label": label,
            "reason": reason,
        }