from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.run_experiment import run_experiment


GROQ_MODELS = [
    "qwen/qwen3-32b",
    "allam-2-7b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_root_config() -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for filename in ["experiment.yaml", "llm.yaml", "spice.yaml", "logging.yaml"]:
        merged.update(load_yaml(ROOT / "configs" / filename))
    return merged


def build_case_inputs(case_name: str) -> Dict[str, Any]:
    system_prompt = load_text(ROOT / "prompts" / "shared" / "system_prompt.txt")
    output_format = load_text(ROOT / "prompts" / "shared" / "output_format.txt")
    sanity_rules = load_text(ROOT / "prompts" / "shared" / "sanity_rules.txt")
    shared_suffix = f"\n\n{output_format}\n\n{sanity_rules}"
    return {
        "baseline_prompt": load_text(ROOT / "prompts" / case_name / "baseline.txt") + shared_suffix,
        "eg_prompt": load_text(ROOT / "prompts" / case_name / "eg.txt") + shared_suffix,
        "system_prompt": system_prompt,
        "spec": load_json(ROOT / "specs" / f"{case_name}.json"),
    }


def summarize_mode(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"FAIL": 0, "RUN": 0, "PASS": 0, "ROBUST_PASS": 0}
    for item in results:
        label = item.get("final_decision", "FAIL")
        if label in counts:
            counts[label] += 1
    return counts


def run_model_case(
    root_config: Dict[str, Any],
    model_name: str,
    case_name: str,
    case_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    config = deepcopy(root_config)
    config["llm"]["active_backend"] = "groq"
    config["backends"]["groq"]["enabled"] = True
    config["backends"]["groq"]["model"]["name"] = model_name

    model_slug = model_name.replace("/", "__").replace(":", "_").replace("\\", "_")
    run_case_name = f"groq_{model_slug}_{case_name}"

    started_at = time.time()
    result = run_experiment(
        config=config,
        case_name=run_case_name,
        baseline_prompt=case_inputs["baseline_prompt"],
        eg_prompt=case_inputs["eg_prompt"],
        system_prompt=case_inputs["system_prompt"],
        spec=case_inputs["spec"],
    )
    elapsed = time.time() - started_at

    return {
        "model": model_name,
        "case": case_name,
        "elapsed_seconds": round(elapsed, 2),
        "baseline_summary": summarize_mode(result.get("baseline", [])),
        "eg_summary": summarize_mode(result.get("eg", [])),
        "result": result,
    }


def main() -> None:
    root_config = build_root_config()
    reports_dir = ROOT / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cases = root_config.get("experiment", {}).get("execution", {}).get(
        "enabled_use_cases",
        ["uc1", "uc2", "uc3"],
    )
    case_inputs = {case_name: build_case_inputs(case_name) for case_name in cases}

    benchmark_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("  spec2testbench v1.0 - Groq Multi-Model Benchmark")
    print("=" * 70)

    for model_name in GROQ_MODELS:
        print(f"\n[MODEL] {model_name}")
        for case_name in cases:
            print(f"[CASE] {case_name}")
            benchmark_results.append(
                run_model_case(root_config, model_name, case_name, case_inputs[case_name])
            )

    output_path = reports_dir / "groq_model_benchmark.json"
    output_path.write_text(json.dumps(benchmark_results, indent=2, default=str), encoding="utf-8")
    print(f"\n[SAVED] {output_path}")


if __name__ == "__main__":
    main()
