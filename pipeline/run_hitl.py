from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from pipeline.run_case import CaseRunner
from pipeline.retry_logic import RetryManager


def run_hitl(
    config: Dict[str, Any],
    prompt: str,
    system_prompt: str,
    spec: Dict[str, Any],
    work_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Exécute le mode HITL (Explicit Guidance) avec retry intelligent.
    """
    runner = CaseRunner(config)
    retry_manager = RetryManager(config.get("experiment", {}))

    n_candidates = (
        config.get("experiment", {})
        .get("execution", {})
        .get("n_candidates_per_prompt", 3)
    )

    all_results: List[Dict[str, Any]] = []
    current_prompt = prompt
    attempt = 0

    while True:
        attempt_dir = work_dir / "hitl" / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [HITL] Attempt {attempt + 1}...")

        results = runner.run(
            prompt=current_prompt,
            spec=spec,
            system_prompt=system_prompt,
            n_candidates=n_candidates,
            work_dir=attempt_dir,
            mode="hitl",
        )

        # Ajouter l'information de tentative aux résultats
        for r in results:
            r["attempt"] = attempt
            r["attempt_dir"] = str(attempt_dir)

        all_results.extend(results)

        # Résumé de la tentative
        decisions = [r.get("final_decision", "FAIL") for r in results]
        print(f"    Results: FAIL={decisions.count('FAIL')}, RUN={decisions.count('RUN')}, "
              f"PASS={decisions.count('PASS')}, ROBUST_PASS={decisions.count('ROBUST_PASS')}")

        should_retry, feedback = retry_manager.should_retry(results, attempt)

        if not should_retry:
            print(f"    Stopping retry loop.")
            break

        print(f"    Retrying with feedback...")

        # Construire un nouveau prompt avec feedback
        current_prompt = f"{prompt}\n\n{feedback}"
        attempt += 1

    return all_results