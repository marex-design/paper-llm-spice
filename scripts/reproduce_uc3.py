from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
    return {
        **load_yaml(ROOT / "configs" / "experiment.yaml"),
        **load_yaml(ROOT / "configs" / "llm.yaml"),
        **load_yaml(ROOT / "configs" / "spice.yaml"),
        **load_yaml(ROOT / "configs" / "logging.yaml"),
    }


def run_one_case(root_config: Dict[str, Any], case_name: str) -> Dict[str, Any]:
    system_prompt = load_text(ROOT / "prompts" / "shared" / "system_prompt.txt")
    output_format = load_text(ROOT / "prompts" / "shared" / "output_format.txt")
    sanity_rules = load_text(ROOT / "prompts" / "shared" / "sanity_rules.txt")
    shared_suffix = f"\n\n{output_format}\n\n{sanity_rules}"
    baseline_prompt = load_text(ROOT / "prompts" / case_name / "baseline.txt") + shared_suffix
    eg_prompt = load_text(ROOT / "prompts" / case_name / "eg.txt") + shared_suffix
    spec = load_json(ROOT / "specs" / f"{case_name}.json")

    return run_experiment(
        config=root_config,
        case_name=case_name,
        baseline_prompt=baseline_prompt,
        eg_prompt=eg_prompt,
        system_prompt=system_prompt,
        spec=spec,
    )


def load_existing_results(reports_dir: Path, cases: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for case_name in cases:
        path = reports_dir / f"{case_name}_raw_results.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                results.append(json.load(handle))
    return results


def main() -> None:
    case_name = "uc3"
    all_cases = ["uc1", "uc2", "uc3"]
    root_config = build_root_config()
    reports_dir = ROOT / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] {case_name}")
    result = run_one_case(root_config, case_name)

    case_output = reports_dir / f"{case_name}_raw_results.json"
    case_output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    experiment_results = load_existing_results(reports_dir, all_cases)
    aggregator = ResultsAggregator()
    table_builder = TableBuilder()
    summary_builder = SummaryBuilder()
    csv_exporter = CSVExporter()

    aggregated = [aggregator.aggregate_case(item) for item in experiment_results]
    (reports_dir / "aggregated_results.json").write_text(json.dumps(aggregated, indent=2), encoding="utf-8")

    iteration_table = table_builder.build_iteration_table(aggregated)
    best_candidates_table = table_builder.build_best_candidates_table(aggregated)
    summary_md = summary_builder.build_markdown_summary(aggregated)

    csv_exporter.export(iteration_table, reports_dir / "table_iterations.csv")
    csv_exporter.export(best_candidates_table, reports_dir / "table_best_candidates.csv")
    (reports_dir / "paper_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"[DONE] {case_name}")
    print(f"[REPORTS] {reports_dir}")


if __name__ == "__main__":
    main()
