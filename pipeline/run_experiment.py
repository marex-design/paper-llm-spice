from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json

from pipeline.run_baseline import run_baseline
from pipeline.run_hitl import run_hitl


# ============================================================================
# Fonction utilitaire 
# ============================================================================
def _summarize_results(mode: str, results: List[Dict[str, Any]]) -> None:
    """Affiche un résumé des résultats d'un mode."""
    if not results:
        print(f"    {mode}: No results")
        return

    decisions = [r.get("final_decision", "FAIL") for r in results]
    counts = {
        "FAIL": decisions.count("FAIL"),
        "RUN": decisions.count("RUN"),
        "PASS": decisions.count("PASS"),
        "ROBUST_PASS": decisions.count("ROBUST_PASS"),
    }
    print(f"    {mode} summary: FAIL={counts['FAIL']}, RUN={counts['RUN']}, "
          f"PASS={counts['PASS']}, ROBUST_PASS={counts['ROBUST_PASS']}")


# ============================================================================
# Fonction principale
# ============================================================================
def run_experiment(
    config: Dict[str, Any],
    case_name: str,
    baseline_prompt: str,
    hitl_prompt: str,
    system_prompt: str,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Exécute une expérience complète pour un use case (baseline + HITL).
    """
    base_dir = Path("runs") / case_name
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running experiment: {case_name}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------------
    # 1/2 : Mode BASELINE
    # ------------------------------------------------------------------------
    print("\n[1/2] Running BASELINE mode...")
    baseline_results = run_baseline(
        config=config,
        prompt=baseline_prompt,
        spec=spec,
        work_dir=base_dir,
    )
    _summarize_results("BASELINE", baseline_results)  #  PAS de self.

    # ------------------------------------------------------------------------
    # 2/2 : Mode HITL (Explicit Guidance)
    # ------------------------------------------------------------------------
    print("\n[2/2] Running HITL mode...")
    hitl_results = run_hitl(
        config=config,
        prompt=hitl_prompt,
        system_prompt=system_prompt,
        spec=spec,
        work_dir=base_dir,
    )
    _summarize_results("HITL", hitl_results)  #  PAS de self.

    # ------------------------------------------------------------------------
    # Sauvegarde des résultats
    # ------------------------------------------------------------------------
    output_path = base_dir / "results.json"
    full_results = {
        "case": case_name,
        "baseline": baseline_results,
        "hitl": hitl_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n Results saved to {output_path}")

    return full_results