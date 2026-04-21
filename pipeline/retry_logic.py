from __future__ import annotations

from typing import List, Dict, Any, Optional


class RetryManager:
    """Gestionnaire de retry intelligent avec feedback pour LLM."""

    def __init__(self, config: Dict[str, Any]) -> None:
        retry_cfg = config.get("retry", {})
        self.enabled = retry_cfg.get("enabled", True)
        self.max_retries = retry_cfg.get("max_retries", 2)
        self.retry_on_fail = retry_cfg.get("retry_on_fail", True)
        self.retry_on_run = retry_cfg.get("retry_on_run", True)
        self.retry_on_missing_measurements = retry_cfg.get("retry_on_missing_measurements", True)
        self.feedback_format = retry_cfg.get(
            "feedback_format",
            "[PREVIOUS ATTEMPT FEEDBACK]\n{feedback}\n\nPlease fix these issues and generate a corrected netlist."
        )

    def should_retry(
        self,
        results: List[Dict[str, Any]],
        attempt_number: int,
    ) -> tuple[bool, Optional[str]]:
        """
        Détermine si un retry est nécessaire et fournit un feedback pour le LLM.

        Args:
            results: Liste des résultats de la tentative actuelle.
            attempt_number: Numéro de la tentative (0-indexed).

        Returns:
            Tuple (should_retry, feedback_message)
        """
        if not self.enabled:
            return False, None

        if attempt_number >= self.max_retries:
            return False, None

        if not results:
            return True, self._format_feedback(
                "No results generated. Please produce a valid SPICE netlist with correct syntax."
            )

        # Chercher le meilleur résultat actuel
        best_result = self._get_best_result(results)

        if best_result is None:
            return True, self._format_feedback(
                "All candidates failed. Check SPICE syntax, node naming conventions, and measurement directives."
            )

        decision = best_result.get("final_decision")
        feedback = None

        if decision == "FAIL":
            reason = self._extract_failure_reason(best_result)
            if self.retry_on_fail:
                feedback = self._format_feedback(
                    f"Generation or simulation failed. {reason}"
                )

        elif decision == "RUN":
            if self.retry_on_run:
                missing_specs = self._extract_missing_specs(best_result)
                feedback = self._format_feedback(
                    f"Simulation succeeded but specifications were not met. {missing_specs} Adjust component values (R, L, C) to meet the target specifications."
                )

        elif decision in ("PASS", "ROBUST_PASS"):
            # Succès : pas de retry
            return False, None

        return feedback is not None, feedback

    def _get_best_result(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Retourne le meilleur résultat selon la hiérarchie de décision."""
        hierarchy = {
            "ROBUST_PASS": 4,
            "PASS": 3,
            "RUN": 2,
            "FAIL": 1,
        }

        best = None
        best_score = -1

        for r in results:
            decision = r.get("final_decision", "FAIL")
            score = hierarchy.get(decision, 0)
            if score > best_score:
                best_score = score
                best = r

        return best

    def _extract_failure_reason(self, result: Dict[str, Any]) -> str:
        """Extrait la raison d'échec pour feedback au LLM avec des conseils spécifiques."""
        reasons = []

        # 1. Erreur LLM
        llm_error = result.get("llm_error")
        if llm_error:
            reasons.append(f"LLM error: {llm_error}")

        # 2. Échec des sanity checks
        sanity = result.get("sanity", {})
        if not sanity.get("ok", True):
            issues = sanity.get("issues", [])
            for issue in issues[:3]:
                code = issue.get("code", "UNKNOWN")
                message = issue.get("message", "")
                reasons.append(f"{code}: {message}")

                # Feedback spécifique pour UC2 - mesures manquantes
                if "E-MEAS-01" in code:
                    if "zin_real" in message or "zin_imag" in message:
                        reasons.append("You MUST include: .meas ac zin_real FIND real(v(in)/(-i(V1))) AT=1e6")
                        reasons.append("You MUST include: .meas ac zin_imag FIND imag(v(in)/(-i(V1))) AT=1e6")
                    if "gamma_mag" in message:
                        reasons.append("You MUST include: .meas ac gamma_mag PARAM 'abs((zin_real + 1i*zin_imag - 50)/(zin_real + 1i*zin_imag + 50))'")

                # Feedback spécifique pour UC3 - nœud manquant
                if "E-NODE-02" in code:
                    if "n1" in message:
                        reasons.append("You MUST name the load intermediate node 'n1' (between Rload and Lload)")
                    if "sw" in message:
                        reasons.append("You MUST name the switching node 'sw'")

        # 3. Erreur de simulation
        simulation = result.get("simulation", {})
        if not simulation.get("success", False):
            reasons.append("SPICE simulation failed to execute.")

        # 4. Erreurs dans le log SPICE
        parsed = result.get("parsed_log", {})
        errors = parsed.get("errors", [])
        for error in errors[:2]:
            reasons.append(f"SPICE error: {error[:150]}")

            # Feedback spécifique pour UC2 - AC analysis
            if "no data saved for AC" in error:
                reasons.append("You MUST include EXACTLY: .ac dec 100 500k 1.5MEG")
                reasons.append("Make sure V1 has 'ac 1' parameter")

            # Feedback spécifique pour UC3 - INTEG
            if "no such vector" in error:
                if "cs" in error.lower():
                    reasons.append("Snubber capacitor MUST be named 'Cs' (case-sensitive)")
                    reasons.append("Use EXACTLY: .meas tran snubber_energy INTEG (v(sw)-v(snub_mid))*i(Cs) FROM=0.9m TO=3m")

            # Erreur de syntaxe générale
            if "syntax error" in error.lower():
                reasons.append("Check SPICE syntax - ensure all components have valid node connections")

        # 5. Mesures requises manquantes
        if not result.get("required_measurements_ok", True):
            reasons.append("Required measurements are missing or invalid.")

        if reasons:
            return " ".join(reasons)
        else:
            return "Unknown failure reason. Check netlist syntax and component connectivity."

    def _extract_missing_specs(self, result: Dict[str, Any]) -> str:
        """Extrait les spécifications non satisfaites pour feedback."""
        nominal = result.get("nominal", {})
        criteria = nominal.get("criteria_results", [])

        failed = [c for c in criteria if not c.get("passed", False)]

        if not failed:
            return "Some specifications were not met."

        details = []
        for f in failed[:3]:
            metric = f.get("metric", "unknown")
            actual = f.get("actual", "N/A")
            target = f.get("target", "N/A")
            operator = f.get("operator", "?")

            if actual is not None:
                details.append(f"{metric} (got {actual:.4f}, need {operator} {target})")
            else:
                details.append(f"{metric} (missing, need {operator} {target})")

        # Ajouter des conseils spécifiques
        advice = ""
        if any("zin_real" in str(f.get("metric", "")) for f in failed):
            advice = " Try adjusting L and C values. For Rs=50, Rload=200, f=1MHz: L ≈ 13.78uH, C ≈ 1.378nF."
        elif any("vpeak_switch" in str(f.get("metric", "")) for f in failed):
            advice = " Try increasing Csnub or adjusting Rsnub. Typical values: Rsnub=10-100, Csnub=1n-100n."

        if details:
            return f"Failed criteria: {'; '.join(details)}.{advice}"
        else:
            return f"Specifications not met. Adjust component values.{advice}"

    def _format_feedback(self, feedback: str) -> str:
        """Formate le feedback selon le template configuré."""
        return self.feedback_format.format(feedback=feedback)