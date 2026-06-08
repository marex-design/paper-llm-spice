from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pipeline.run_baseline import run_baseline
from pipeline.run_eg import run_eg


def _summarize_results(mode: str, results: List[Dict[str, Any]]) -> None:
    """Print a compact summary for one execution mode."""
    if not results:
        print(f"    {mode}: No results")
        return

    decisions = [item.get("final_decision", "FAIL") for item in results]
    counts = {
        "FAIL": decisions.count("FAIL"),
        "RUN": decisions.count("RUN"),
        "PASS": decisions.count("PASS"),
        "ROBUST_PASS": decisions.count("ROBUST_PASS"),
    }
    print(
        f"    {mode} summary: "
        f"FAIL={counts['FAIL']}, RUN={counts['RUN']}, "
        f"PASS={counts['PASS']}, ROBUST_PASS={counts['ROBUST_PASS']}"
    )


def run_experiment(
    config: Dict[str, Any],
    case_name: str,
    baseline_prompt: str,
    eg_prompt: str,
    system_prompt: str,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one complete use case experiment for baseline and EG."""
    base_dir = Path("runs") / case_name
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {case_name}")
    print(f"{'=' * 60}")

    print("\n[1/2] Running BASELINE mode...")
    baseline_results = run_baseline(
        config=config,
        prompt=baseline_prompt,
        spec=spec,
        work_dir=base_dir,
    )
    _summarize_results("BASELINE", baseline_results)

    print("\n[2/2] Running EG mode...")
    eg_results = run_eg(
        config=config,
        prompt=eg_prompt,
        system_prompt=system_prompt,
        spec=spec,
        work_dir=base_dir,
    )
    _summarize_results("EG", eg_results)

    output_path = base_dir / "results.json"
    full_results = {
        "case": case_name,
        "baseline": baseline_results,
        "eg": eg_results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(full_results, handle, indent=2, default=str)

    print(f"\n Results saved to {output_path}")
    return full_results
