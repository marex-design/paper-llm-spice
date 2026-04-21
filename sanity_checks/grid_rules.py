from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class GridRules:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec

    def evaluate(self, netlist: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[Dict[str, str]] = []

        analysis_type = self.spec.get("analysis_type", "").lower()
        lines = self._lines(netlist)

        if analysis_type == "ac":
            ac_line = self._find_first_prefix(lines, ".ac")
            if not ac_line:
                issues.append(self._issue("E-GRID-00", "AC analysis is required but no .ac directive was found.", severity="warning"))
            else:
                issues.extend(self._check_ac_grid(ac_line))

        if analysis_type in {"transient", "tran"}:
            tran_line = self._find_first_prefix(lines, ".tran")
            if not tran_line:
                issues.append(self._issue("E-GRID-10", "Transient analysis is required but no .tran directive was found.", severity="warning"))
            else:
                issues.extend(self._check_tran_window(tran_line))

        return {
            "rule_family": "grid",
            "ok": True,
            "issues": issues,
        }

    def _check_ac_grid(self, ac_line: str) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []
        parsed = self._parse_ac_line(ac_line)

        if parsed is None:
            issues.append(self._issue("E-GRID-01", "Unable to parse .ac directive.", severity="warning"))
            return issues

        sweep_type = parsed["sweep_type"]
        points = parsed["points"]
        f_start = parsed["f_start"]
        f_stop = parsed["f_stop"]

        for meas in self.spec.get("measurements", []):
            freq = meas.get("frequency_hz")
            if freq is None:
                continue

            try:
                freq = float(freq)
            except (TypeError, ValueError):
                issues.append(
                    self._issue(
                        "E-GRID-04",
                        f"Invalid measurement frequency '{freq}'.",
                        severity="warning",
                    )
                )
                continue

            if not (f_start <= freq <= f_stop):
                issues.append(
                    self._issue(
                        "E-GRID-02",
                        f"Measurement frequency {freq} Hz is outside the AC sweep range.",
                        severity="warning",
                    )
                )
                continue

            if sweep_type == "lin":
                if not self._is_on_linear_grid(freq, f_start, f_stop, points):
                    issues.append(
                        self._issue(
                            "E-GRID-03",
                            f"Measurement frequency {freq} Hz is not exactly included in the linear AC grid.",
                            severity="warning",
                        )
                    )

        return issues

    def _check_tran_window(self, tran_line: str) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []
        parsed = self._parse_tran_line(tran_line)

        if parsed is None:
            issues.append(self._issue("E-GRID-11", "Unable to parse .tran directive.", severity="warning"))
            return issues

        t_stop = parsed["t_stop"]

        window = self.spec.get("simulation", {}).get("observation_window", {})
        t_end_req = window.get("t_end_s")

        if t_end_req is not None:
            try:
                t_end_req = float(t_end_req)
            except (TypeError, ValueError):
                issues.append(
                    self._issue(
                        "E-GRID-13",
                        f"Invalid required observation window end '{t_end_req}'.",
                        severity="warning",
                    )
                )
                return issues

            if t_stop < t_end_req:
                issues.append(
                    self._issue(
                        "E-GRID-12",
                        f".tran stop time {t_stop} s does not cover required observation window end {t_end_req} s.",
                        severity="warning",
                    )
                )

        return issues

    @staticmethod
    def _lines(netlist: str) -> List[str]:
        return [
            line.strip()
            for line in netlist.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ]

    @staticmethod
    def _find_first_prefix(lines: List[str], prefix: str) -> Optional[str]:
        prefix = prefix.lower()
        for line in lines:
            if line.lower().startswith(prefix):
                return line
        return None

    def _parse_ac_line(self, line: str) -> Optional[Dict[str, Any]]:
        parts = line.split()
        if len(parts) < 5:
            return None

        sweep_type = parts[1].lower()

        try:
            points = int(float(parts[2]))
            f_start = self._parse_spice_number(parts[3])
            f_stop = self._parse_spice_number(parts[4])
        except (TypeError, ValueError):
            return None

        return {
            "sweep_type": sweep_type,
            "points": points,
            "f_start": f_start,
            "f_stop": f_stop,
        }

    def _parse_tran_line(self, line: str) -> Optional[Dict[str, Any]]:
        parts = line.split()
        if len(parts) < 3:
            return None

        try:
            t_step = self._parse_spice_number(parts[1])
            t_stop = self._parse_spice_number(parts[2])
        except (TypeError, ValueError):
            return None

        t_start = None
        t_maxstep = None

        if len(parts) >= 4:
            try:
                t_start = self._parse_spice_number(parts[3])
            except (TypeError, ValueError):
                t_start = None

        if len(parts) >= 5:
            try:
                t_maxstep = self._parse_spice_number(parts[4])
            except (TypeError, ValueError):
                t_maxstep = None

        return {
            "t_step": t_step,
            "t_stop": t_stop,
            "t_start": t_start,
            "t_maxstep": t_maxstep,
        }

    @staticmethod
    def _parse_spice_number(value: str) -> float:
        s = value.strip()

        if s.startswith("="):
            s = s[1:].strip()

        suffix_map = {
            "t": 1e12,
            "g": 1e9,
            "meg": 1e6,
            "k": 1e3,
            "m": 1e-3,
            "u": 1e-6,
            "n": 1e-9,
            "p": 1e-12,
            "f": 1e-15,
        }

        unit_suffixes = ["hz", "sec", "s", "v", "a", "f", "h", "ohm"]

        s_lower = s.lower()

        try:
            return float(s_lower)
        except ValueError:
            pass

        for unit in unit_suffixes:
            if s_lower.endswith(unit):
                s_lower = s_lower[: -len(unit)]
                break

        s_lower = s_lower.strip()

        try:
            return float(s_lower)
        except ValueError:
            pass

        match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*([a-z]+)?",
            s_lower,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"Unable to parse SPICE numeric value: {value}")

        number = float(match.group(1))
        suffix = match.group(2)

        if not suffix:
            return number

        if suffix in suffix_map:
            return number * suffix_map[suffix]

        raise ValueError(f"Unsupported SPICE suffix '{suffix}' in value: {value}")

    @staticmethod
    def _is_on_linear_grid(freq: float, f_start: float, f_stop: float, points: int, tol: float = 1e-9) -> bool:
        if points < 2:
            return abs(freq - f_start) <= tol

        step = (f_stop - f_start) / (points - 1)
        if step == 0:
            return abs(freq - f_start) <= tol

        n = round((freq - f_start) / step)
        reconstructed = f_start + n * step

        return abs(reconstructed - freq) <= max(tol, 1e-9 * max(1.0, abs(freq)))

    @staticmethod
    def _issue(code: str, message: str, severity: str = "warning") -> Dict[str, str]:
        return {
            "code": code,
            "message": message,
            "severity": severity,
        }