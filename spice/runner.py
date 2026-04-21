from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Any


class SpiceRunner:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.ngspice_path = config.get("ngspice_path", "ngspice")
        self.timeout = config.get("timeout_seconds", 30)

    def run(self, netlist_path: Path, output_log_path: Path) -> Dict[str, Any]:
        cmd = [
            self.ngspice_path,
            "-b",
            "-o",
            str(output_log_path),
            str(netlist_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "log_path": str(output_log_path),
            }

        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "error": "timeout",
                "details": str(e),
            }