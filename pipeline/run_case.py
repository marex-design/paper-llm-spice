from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import math

from llm.generator import LLMGenerator
from extraction.response_cleaner import ResponseCleaner
from extraction.netlist_extractor import NetlistExtractor
from extraction.metadata_extractor import MetadataExtractor
from sanity_checks.checker import SanityChecker
from spice.runner import SpiceRunner
from spice.parser import SpiceLogParser
from spice.meas_parser import MeasParser
from spice.sweep_runner import SweepRunner
from evaluation.criteria import CriteriaEvaluator
from evaluation.robustness import RobustnessEvaluator
from evaluation.decision import DecisionEngine
from evaluation.metrics import MetricsStore, MetricsEnhancer


class CaseRunner:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        self.generator = LLMGenerator(config)
        self.cleaner = ResponseCleaner()
        self.extractor = NetlistExtractor()
        self.metadata_extractor = MetadataExtractor()

        self.spice_runner = SpiceRunner(config.get("spice", {}))
        self.log_parser = SpiceLogParser()
        self.meas_parser = MeasParser()
        self.sweep_runner = SweepRunner(
            spice_runner=self.spice_runner,
            parser=self.log_parser,
            meas_parser=self.meas_parser,
        )

    def run(
        self,
        prompt: str,
        spec: Dict[str, Any],
        system_prompt: str | None = None,
        n_candidates: int = 3,
        work_dir: Path = Path("runs/tmp"),
        mode: str = "case",
    ) -> List[Dict[str, Any]]:
        work_dir.mkdir(parents=True, exist_ok=True)

        checker = SanityChecker(spec)
        criteria_eval = CriteriaEvaluator(spec)
        robustness_eval = RobustnessEvaluator(spec)
        decision_engine = DecisionEngine(spec, self.config.get("experiment", {}))

        responses = self.generator.generate_n(
            prompt=prompt,
            n=n_candidates,
            system_prompt=system_prompt,
            metadata={"mode": mode, "use_case": spec.get("id")},
        )

        results: List[Dict[str, Any]] = []

        for idx, response in enumerate(responses):
            candidate_id = f"cand_{idx + 1:02d}"
            cand_dir = work_dir / candidate_id
            cand_dir.mkdir(parents=True, exist_ok=True)

            result: Dict[str, Any] = {
                "candidate_id": candidate_id,
                "mode": mode,
                "use_case": spec.get("id"),
                "llm_success": response.success,
                "backend": response.backend,
                "model": response.model,
                "raw_text": response.text,
                "llm_error": response.error,
                "final_decision": "FAIL",
            }

            (cand_dir / "raw_response.txt").write_text(response.text or "", encoding="utf-8")

            if not response.success:
                result["decision"] = {
                    "label": "FAIL",
                    "reason": f"LLM generation failed: {response.error}",
                }
                results.append(result)
                continue

            try:
                cleaned = self.cleaner.clean(response.text)
                (cand_dir / "cleaned_response.txt").write_text(cleaned, encoding="utf-8")

                netlist = self.extractor.extract(cleaned)
                metadata = self.metadata_extractor.extract(netlist)

                netlist_path = cand_dir / "circuit.cir"
                log_path = cand_dir / "output.log"
                netlist_path.write_text(netlist, encoding="utf-8")

                result["metadata"] = metadata

                sanity = checker.run(netlist, metadata)
                result["sanity"] = sanity

                sim_result = self.spice_runner.run(netlist_path, log_path)
                result["simulation"] = sim_result

                parsed = self.log_parser.parse(log_path) if log_path.exists() else {
                    "has_error": True,
                    "ran_simulation": False,
                    "warnings": [],
                    "errors": ["Missing ngspice log file."],
                    "raw_text": "",
                }

                result["parsed_log"] = {
                    "has_error": parsed.get("has_error", True),
                    "ran_simulation": parsed.get("ran_simulation", False),
                    "warnings": parsed.get("warnings", []),
                    "errors": parsed.get("errors", []),
                }

                simulation_ok = (
                    sim_result.get("success", False)
                    and log_path.exists()
                    and parsed.get("ran_simulation", False)
                    and not parsed.get("has_error", False)
                )

                # ========== NOUVEAU : MetricsEnhancer ==========
                meas_raw = self.meas_parser.parse(parsed.get("raw_text", ""))
                result["meas_raw"] = meas_raw

                # Appliquer les enhancements (fallback intelligent)
                store = MetricsStore(meas_raw)
                enhancer = MetricsEnhancer(spec, store)
                store = enhancer.enhance()
                result["metric_enhancements"] = enhancer.get_enhancement_log()

                # Appliquer les dérivées standards (dB, etc.)
                meas = self._apply_derived_metrics(spec, store.as_dict())
                result["meas"] = meas
                # =============================================

                required_measurements_ok = self._required_measurements_ok(spec, meas)
                result["required_measurements_ok"] = required_measurements_ok

                nominal = criteria_eval.evaluate_nominal(meas)
                result["nominal"] = nominal

                robustness_enabled = spec.get("robustness", {}).get("enabled", False)
                robustness_result = None
                robustness_ok = None

                if simulation_ok and required_measurements_ok and nominal.get("ok", False) and robustness_enabled:
                    sweep_variations = self._build_sweep_variations(spec)

                    if sweep_variations:
                        sweep_dir = cand_dir / "sweeps"
                        sweep_dir.mkdir(parents=True, exist_ok=True)

                        sweep_results = self.sweep_runner.run_sweep(
                            base_netlist=netlist_path,
                            work_dir=sweep_dir,
                            variations=sweep_variations,
                        )

                        for run in sweep_results:
                            raw_meas = run.get("meas", {})
                            run["meas_raw"] = dict(raw_meas)
                            run["meas"] = self._apply_derived_metrics(spec, raw_meas)

                        robustness_result = robustness_eval.evaluate(sweep_results)
                        robustness_ok = robustness_result.get("ok", False)
                    else:
                        robustness_result = {
                            "ok": False,
                            "reason": "Robustness enabled but no sweep variations could be built.",
                            "sweep_results": [],
                            "passed_count": 0,
                            "failed_count": 0,
                            "total_count": 0,
                        }
                        robustness_ok = False

                result["robustness"] = robustness_result

                decision = decision_engine.classify(
                    sanity_ok=sanity.get("ok", False),
                    simulation_ok=simulation_ok,
                    required_measurements_ok=required_measurements_ok,
                    nominal_ok=nominal.get("ok", False),
                    robustness_ok=robustness_ok,
                )

                result["decision"] = decision
                result["final_decision"] = decision["label"]

            except Exception as exc:
                result["error"] = str(exc)
                result["decision"] = {
                    "label": "FAIL",
                    "reason": f"Unhandled pipeline exception: {exc}",
                }
                result["final_decision"] = "FAIL"

            results.append(result)

        spec_infeasibility = decision_engine.classify_spec_infeasibility(results)

        if spec_infeasibility.get("triggered"):
            for item in results:
                if item.get("final_decision") == "RUN":
                    item["spec_infeasibility"] = spec_infeasibility

        return results

    def _required_measurements_ok(self, spec: Dict[str, Any], meas: Dict[str, Any]) -> bool:
        required = spec.get("sanity_checks", {}).get("required_measurements", [])
        if not required:
            return True
        normalized_meas = {str(k).strip().lower() for k in meas.keys()}
        return all(str(name).strip().lower() in normalized_meas for name in required)

    def _build_sweep_variations(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        swept_parameters = spec.get("robustness", {}).get("swept_parameters", [])
        if not swept_parameters:
            return []

        parameter_value_sets: List[tuple[str, List[float]]] = []

        for param in swept_parameters:
            name = param["name"]
            mode = param.get("variation_mode", "relative_percent")
            values_percent = param.get("values_percent", [])

            if mode != "relative_percent":
                continue

            factors = [1.0 + (float(v) / 100.0) for v in values_percent]
            parameter_value_sets.append((name, factors))

        if not parameter_value_sets:
            return []

        combinations: List[Dict[str, Any]] = [{}]

        for param_name, factors in parameter_value_sets:
            new_combinations: List[Dict[str, Any]] = []
            for combo in combinations:
                for factor in factors:
                    updated = dict(combo)
                    updated[param_name] = factor
                    new_combinations.append(updated)
            combinations = new_combinations

        return combinations

    def _apply_derived_metrics(self, spec: Dict[str, Any], meas: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(meas)

        normalized_lookup = {
            str(k).strip().lower(): v for k, v in meas.items()
        }

        derived_metrics = spec.get("derived_metrics", [])

        for item in derived_metrics:
            name = item.get("name")
            source = item.get("from")
            transform = item.get("transform")

            if not name or not source or not transform:
                continue

            source_key = str(source).strip().lower()
            if source_key not in normalized_lookup:
                continue

            try:
                x = float(normalized_lookup[source_key])
            except (TypeError, ValueError):
                continue

            try:
                if transform == "20*log10(x)":
                    if x <= 0:
                        continue
                    enriched[name] = 20.0 * math.log10(x)

                elif transform == "10*log10(x)":
                    if x <= 0:
                        continue
                    enriched[name] = 10.0 * math.log10(x)

            except (ValueError, OverflowError):
                continue

        return enriched