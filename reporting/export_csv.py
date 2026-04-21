from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Any


class CSVExporter:
    def export(self, rows: List[Dict[str, Any]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            output_path.write_text("")
            return

        fieldnames = list(rows[0].keys())

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)