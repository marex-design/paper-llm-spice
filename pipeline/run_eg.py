from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pipeline.retry_logic import RetryManager
from pipeline.run_case import CaseRunner


def run_eg(
    config: Dict[str, Any],
    prompt: str,
    system_prompt: str,
    spec: Dict[str, Any],
    work_dir: Path,
) -> List[Dict[str, Any]]:
    """Run the Explicit Guidance mode with retry feedback."""
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
        attempt_dir = work_dir / "eg" / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [EG] Attempt {attempt + 1}...")

        results = runner.run(
            prompt=current_prompt,
            spec=spec,
            system_prompt=system_prompt,
            n_candidates=n_candidates,
            work_dir=attempt_dir,
            mode="eg",
        )

        for result in results:
            result["attempt"] = attempt
            result["attempt_dir"] = str(attempt_dir)

        all_results.extend(results)

        decisions = [item.get("final_decision", "FAIL") for item in results]
        print(
            "    Results: "
            f"FAIL={decisions.count('FAIL')}, "
            f"RUN={decisions.count('RUN')}, "
            f"PASS={decisions.count('PASS')}, "
            f"ROBUST_PASS={decisions.count('ROBUST_PASS')}"
        )

        should_retry, feedback = retry_manager.should_retry(results, attempt)
        if not should_retry:
            print("    Stopping retry loop.")
            break

        print("    Retrying with feedback...")
        current_prompt = f"{prompt}\n\n{feedback}"
        attempt += 1

    return all_results
