from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import re


class SweepRunner:
    def __init__(self, spice_runner, parser, meas_parser) -> None:
        self.spice_runner = spice_runner
        self.parser = parser
        self.meas_parser = meas_parser

    def run_sweep(
        self,
        base_netlist: Path,
        work_dir: Path,
        variations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = []
        base_text = base_netlist.read_text(encoding="utf-8")

        for idx, var in enumerate(variations):
            run_dir = work_dir / f"run_{idx + 1:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            netlist_path = run_dir / "circuit.cir"
            log_path = run_dir / "output.log"

            text = base_text
            for key, factor in var.items():
                text = self._apply_param_factor(text, key, factor)

            netlist_path.write_text(text, encoding="utf-8")

            run_result = self.spice_runner.run(netlist_path, log_path)

            if not run_result.get("success", False):
                results.append({
                    "success": False,
                    "error": "simulation_failed",
                    "variation": var,
                })
                continue

            parsed = self.parser.parse(log_path)
            meas = self.meas_parser.parse(parsed["raw_text"])

            results.append({
                "success": not parsed["has_error"] and parsed["ran_simulation"],
                "variation": var,
                "meas": meas,
                "warnings": parsed["warnings"],
            })

        return results

    def _apply_param_factor(self, text: str, param_name: str, factor: float) -> str:
        pattern = re.compile(
            rf"(^\s*\.param\s+{re.escape(param_name)}\s*=\s*)([^\s]+)",
            re.IGNORECASE | re.MULTILINE,
        )

        match = pattern.search(text)
        if not match:
            return text

        original_value = match.group(2)
        updated_value = f"({original_value})*{factor}"

        return pattern.sub(rf"\1{updated_value}", text, count=1)