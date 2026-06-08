from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.run_experiment import run_experiment
from reporting.aggregate_results import ResultsAggregator
from reporting.build_summary import SummaryBuilder
from reporting.build_tables import TableBuilder
from reporting.export_csv import CSVExporter


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_root_config() -> Dict[str, Any]:
    configs = [
        ("experiment.yaml", load_yaml(ROOT / "configs" / "experiment.yaml")),
        ("llm.yaml", load_yaml(ROOT / "configs" / "llm.yaml")),
        ("spice.yaml", load_yaml(ROOT / "configs" / "spice.yaml")),
        ("logging.yaml", load_yaml(ROOT / "configs" / "logging.yaml")),
    ]

    merged: Dict[str, Any] = {}
    for name, cfg in configs:
        merged.update(cfg)
        print(f"[CONFIG] Loaded {name}")

    return merged


def run_one_case(
    root_config: Dict[str, Any],
    case_name: str,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    try:
        system_prompt = load_text(ROOT / "prompts" / "shared" / "system_prompt.txt")
        output_format = load_text(ROOT / "prompts" / "shared" / "output_format.txt")
        sanity_rules = load_text(ROOT / "prompts" / "shared" / "sanity_rules.txt")

        shared_suffix = f"\n\n{output_format}\n\n{sanity_rules}"
        baseline_prompt = load_text(ROOT / "prompts" / case_name / "baseline.txt") + shared_suffix
        eg_prompt = load_text(ROOT / "prompts" / case_name / "eg.txt") + shared_suffix
        spec = load_json(ROOT / "specs" / f"{case_name}.json")

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"[RUNNING] {case_name.upper()} - {spec.get('title', case_name)}")
            print(f"{'=' * 70}")

        start_time = time.time()
        result = run_experiment(
            config=root_config,
            case_name=case_name,
            baseline_prompt=baseline_prompt,
            eg_prompt=eg_prompt,
            system_prompt=system_prompt,
            spec=spec,
        )
        elapsed = time.time() - start_time

        if verbose:
            print(f"\n[COMPLETED] {case_name.upper()} in {elapsed:.1f}s")
            _print_case_summary(result)

        return result

    except FileNotFoundError as exc:
        print(f"[ERROR] Missing file for {case_name}: {exc}")
        return None
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in spec for {case_name}: {exc}")
        return None
    except Exception as exc:
        print(f"[ERROR] Unexpected error in {case_name}: {exc}")
        import traceback

        traceback.print_exc()
        return None


def _print_case_summary(result: Dict[str, Any]) -> None:
    case_name = result.get("case", "unknown")
    baseline = result.get("baseline", [])
    eg = result.get("eg", [])

    def count_decisions_hierarchical(results: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"FAIL": 0, "RUN": 0, "PASS": 0, "ROBUST_PASS": 0}

        for item in results:
            decision = item.get("final_decision", "FAIL")
            if decision == "FAIL":
                counts["FAIL"] += 1
            elif decision == "RUN":
                counts["RUN"] += 1
            elif decision == "PASS":
                counts["PASS"] += 1
                counts["RUN"] += 1
            elif decision == "ROBUST_PASS":
                counts["ROBUST_PASS"] += 1
                counts["PASS"] += 1
                counts["RUN"] += 1

        return counts

    baseline_counts = count_decisions_hierarchical(baseline)
    eg_counts = count_decisions_hierarchical(eg)

    print(f"\n  {'=' * 28} {case_name.upper()} Results Summary {'=' * 28}")
    print(
        "  "
        f"BASELINE: FAIL={baseline_counts['FAIL']}, RUN={baseline_counts['RUN']}, "
        f"PASS={baseline_counts['PASS']}, ROBUST_PASS={baseline_counts['ROBUST_PASS']}"
    )
    print(
        "  "
        f"EG:       FAIL={eg_counts['FAIL']}, RUN={eg_counts['RUN']}, "
        f"PASS={eg_counts['PASS']}, ROBUST_PASS={eg_counts['ROBUST_PASS']}"
    )
    print("\n  Hierarchy: FAIL subset RUN subset PASS subset ROBUST_PASS")

    for mode, results in [("BASELINE", baseline), ("EG", eg)]:
        for item in results:
            enhancements = item.get("metric_enhancements", [])
            if enhancements:
                print(f"\n  [ENHANCEMENTS] {mode} {item.get('candidate_id', '?')}:")
                for enhancement in enhancements[:3]:
                    print(f"    - {enhancement}")


def load_existing_results(reports_dir: Path, cases: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for case_name in cases:
        path = reports_dir / f"{case_name}_raw_results.json"
        if not path.exists():
            print(f"[WARN] Missing raw results for {case_name}: {path}")
            continue

        try:
            with path.open("r", encoding="utf-8") as handle:
                results.append(json.load(handle))
        except json.JSONDecodeError:
            with path.open("r", encoding="utf-8-sig") as handle:
                results.append(json.load(handle))

    return results


def build_reports(reports_dir: Path, experiment_results: List[Dict[str, Any]]) -> None:
    if not experiment_results:
        raise RuntimeError("No experiment results available to aggregate.")

    aggregator = ResultsAggregator(hierarchical_counting=True)
    table_builder = TableBuilder(hierarchical_counting=True)
    summary_builder = SummaryBuilder()
    csv_exporter = CSVExporter()

    aggregated = []
    for result in experiment_results:
        try:
            aggregated.append(aggregator.aggregate_case(result))
        except Exception as exc:
            print(f"[WARN] Failed to aggregate case {result.get('case', 'unknown')}: {exc}")

    (reports_dir / "aggregated_results.json").write_text(
        json.dumps(aggregated, indent=2, default=str),
        encoding="utf-8",
    )

    iteration_table = table_builder.build_iteration_table(aggregated)
    best_candidates_table = table_builder.build_best_candidates_table(aggregated)
    paper_table_md = table_builder.build_paper_summary_table(aggregated)
    summary_md = summary_builder.build_markdown_summary(aggregated)
    full_paper_md = f"{paper_table_md}\n\n{summary_md}"

    csv_exporter.export(iteration_table, reports_dir / "table_iterations.csv")
    csv_exporter.export(best_candidates_table, reports_dir / "table_best_candidates.csv")
    (reports_dir / "paper_summary.md").write_text(full_paper_md, encoding="utf-8")

    print("\n[REPORTS] Generated:")
    print(f"  - {reports_dir / 'aggregated_results.json'}")
    print(f"  - {reports_dir / 'table_iterations.csv'}")
    print(f"  - {reports_dir / 'table_best_candidates.csv'}")
    print(f"  - {reports_dir / 'paper_summary.md'}")


def build_taxonomy_report(reports_dir: Path, experiment_results: List[Dict[str, Any]]) -> None:
    taxonomy_counts: Dict[str, int] = {}

    for case_result in experiment_results:
        for mode in ["baseline", "eg"]:
            for candidate in case_result.get(mode, []):
                sanity = candidate.get("sanity", {})
                for issue in sanity.get("issues", []):
                    code = issue.get("code", "UNKNOWN")
                    taxonomy_counts[code] = taxonomy_counts.get(code, 0) + 1

                parsed = candidate.get("parsed_log", {})
                for error in parsed.get("errors", []):
                    lowered = error.lower()
                    if "syntax error" in lowered:
                        taxonomy_counts["E-SYNTAX-01"] = taxonomy_counts.get("E-SYNTAX-01", 0) + 1
                    elif "no such vector" in lowered:
                        taxonomy_counts["E-MEAS-03"] = taxonomy_counts.get("E-MEAS-03", 0) + 1
                    elif "404" in error or "NOT_FOUND" in error:
                        taxonomy_counts["E-API-404"] = taxonomy_counts.get("E-API-404", 0) + 1
                    elif "429" in error or "RESOURCE_EXHAUSTED" in error:
                        taxonomy_counts["E-API-429"] = taxonomy_counts.get("E-API-429", 0) + 1
                    elif "402" in error or "Insufficient Balance" in error:
                        taxonomy_counts["E-API-402"] = taxonomy_counts.get("E-API-402", 0) + 1

    taxonomy_path = reports_dir / "taxonomy_counts.json"
    taxonomy_path.write_text(json.dumps(taxonomy_counts, indent=2), encoding="utf-8")
    print(f"\n[TAXONOMY] Saved to {taxonomy_path}")


def main() -> None:
    print("\n" + "=" * 70)
    print("  spec2testbench v1.0 - Full Reproduction")
    print("=" * 70)

    root_config = build_root_config()
    reports_dir = ROOT / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    execution_cfg = root_config.get("experiment", {}).get("execution", {})
    cases = execution_cfg.get("enabled_use_cases", ["uc1", "uc2", "uc3"])
    print(f"\n[USE CASES] {cases}")

    experiment_results: List[Dict[str, Any]] = []

    for case_name in cases:
        result = run_one_case(root_config, case_name, verbose=True)
        if result is not None:
            case_output = reports_dir / f"{case_name}_raw_results.json"
            case_output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(f"[SAVED] {case_output}")
            experiment_results.append(result)
        else:
            print(f"[SKIPPED] {case_name} - No results generated")

    if not experiment_results:
        print("\n[ERROR] No experiment results were generated.")
        print("        Trying to load existing results from reports directory...")
        experiment_results = load_existing_results(reports_dir, cases)

    if experiment_results:
        build_reports(reports_dir, experiment_results)
        build_taxonomy_report(reports_dir, experiment_results)

        print("\n" + "=" * 70)
        print("  FINAL SUMMARY")
        print("=" * 70)
        _print_global_summary(experiment_results)
    else:
        print("\n[FATAL] No results available. Cannot generate reports.")

    print("\n[DONE] Reproduction complete.")
    print(f"[REPORTS] {reports_dir}")


def _print_global_summary(results: List[Dict[str, Any]]) -> None:
    total_baseline = {"FAIL": 0, "RUN": 0, "PASS": 0, "ROBUST_PASS": 0}
    total_eg = {"FAIL": 0, "RUN": 0, "PASS": 0, "ROBUST_PASS": 0}

    for case in results:
        for item in case.get("baseline", []):
            dec = item.get("final_decision", "FAIL")
            if dec == "FAIL":
                total_baseline["FAIL"] += 1
            elif dec == "RUN":
                total_baseline["RUN"] += 1
            elif dec == "PASS":
                total_baseline["PASS"] += 1
                total_baseline["RUN"] += 1
            elif dec == "ROBUST_PASS":
                total_baseline["ROBUST_PASS"] += 1
                total_baseline["PASS"] += 1
                total_baseline["RUN"] += 1

        for item in case.get("eg", []):
            dec = item.get("final_decision", "FAIL")
            if dec == "FAIL":
                total_eg["FAIL"] += 1
            elif dec == "RUN":
                total_eg["RUN"] += 1
            elif dec == "PASS":
                total_eg["PASS"] += 1
                total_eg["RUN"] += 1
            elif dec == "ROBUST_PASS":
                total_eg["ROBUST_PASS"] += 1
                total_eg["PASS"] += 1
                total_eg["RUN"] += 1

    print("\n  Overall Totals (Hierarchical):")
    print(
        f"    BASELINE: FAIL={total_baseline['FAIL']}, RUN={total_baseline['RUN']}, "
        f"PASS={total_baseline['PASS']}, ROBUST_PASS={total_baseline['ROBUST_PASS']}"
    )
    print(
        f"    EG:       FAIL={total_eg['FAIL']}, RUN={total_eg['RUN']}, "
        f"PASS={total_eg['PASS']}, ROBUST_PASS={total_eg['ROBUST_PASS']}"
    )

    baseline_total = 0
    eg_total = 0
    for case in results:
        baseline_total += len(case.get("baseline", []))
        eg_total += len(case.get("eg", []))

    if baseline_total > 0:
        baseline_success = total_baseline["PASS"]
        print(
            f"\n    BASELINE success rate: {baseline_success}/{baseline_total} "
            f"({100 * baseline_success / baseline_total:.1f}%)"
        )

    if eg_total > 0:
        eg_success = total_eg["PASS"]
        print(f"    EG success rate:       {eg_success}/{eg_total} ({100 * eg_success / eg_total:.1f}%)")


if __name__ == "__main__":
    main()
