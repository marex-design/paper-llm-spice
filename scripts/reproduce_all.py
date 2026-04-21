from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.run_experiment import run_experiment
from reporting.aggregate_results import ResultsAggregator
from reporting.build_tables import TableBuilder
from reporting.build_summary import SummaryBuilder
from reporting.export_csv import CSVExporter


# ============================================================================
# Configuration Loading
# ============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """Charge un fichier YAML."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text(path: Path) -> str:
    """Charge un fichier texte."""
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    """Charge un fichier JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_root_config() -> Dict[str, Any]:
    """Construit la configuration racine en fusionnant tous les fichiers de config."""
    configs = [
        ("experiment.yaml", load_yaml(ROOT / "configs" / "experiment.yaml")),
        ("llm.yaml", load_yaml(ROOT / "configs" / "llm.yaml")),
        ("spice.yaml", load_yaml(ROOT / "configs" / "spice.yaml")),
        ("logging.yaml", load_yaml(ROOT / "configs" / "logging.yaml")),
    ]

    merged: Dict[str, Any] = {}
    for name, cfg in configs:
        merged.update(cfg)
        print(f"[CONFIG] Loaded {name}")

    return merged


# ============================================================================
# Experiment Execution
# ============================================================================

def run_one_case(
    root_config: Dict[str, Any],
    case_name: str,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Exécute un seul use case complet (baseline + HITL).
    
    Returns:
        Dict contenant les résultats ou None si erreur.
    """
    try:
        # Charger les prompts partagés
        system_prompt = load_text(ROOT / "prompts" / "shared" / "system_prompt.txt")
        output_format = load_text(ROOT / "prompts" / "shared" / "output_format.txt")
        sanity_rules = load_text(ROOT / "prompts" / "shared" / "sanity_rules.txt")

        shared_suffix = f"\n\n{output_format}\n\n{sanity_rules}"

        # Charger les prompts spécifiques au use case
        baseline_prompt = load_text(ROOT / "prompts" / case_name / "baseline.txt") + shared_suffix
        hitl_prompt = load_text(ROOT / "prompts" / case_name / "eg.txt") + shared_suffix

        # Charger la spécification
        spec = load_json(ROOT / "specs" / f"{case_name}.json")

        if verbose:
            print(f"\n{'='*70}")
            print(f"[RUNNING] {case_name.upper()} - {spec.get('title', case_name)}")
            print(f"{'='*70}")

        start_time = time.time()

        result = run_experiment(
            config=root_config,
            case_name=case_name,
            baseline_prompt=baseline_prompt,
            hitl_prompt=hitl_prompt,
            system_prompt=system_prompt,
            spec=spec,
        )

        elapsed = time.time() - start_time

        if verbose:
            print(f"\n[COMPLETED] {case_name.upper()} in {elapsed:.1f}s")
            _print_case_summary(result)

        return result

    except FileNotFoundError as e:
        print(f"[ERROR] Missing file for {case_name}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in spec for {case_name}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error in {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _print_case_summary(result: Dict[str, Any]) -> None:
    """Affiche un résumé des résultats d'un use case avec comptage hiérarchique complet."""
    case_name = result.get("case", "unknown")
    baseline = result.get("baseline", [])
    hitl = result.get("hitl", [])

    def count_decisions_hierarchical(results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Comptage hiérarchique complet : FAIL ⊂ RUN ⊂ PASS ⊂ ROBUST_PASS."""
        counts = {"FAIL": 0, "RUN": 0, "PASS": 0, "ROBUST_PASS": 0}
        
        for r in results:
            decision = r.get("final_decision", "FAIL")
            
            if decision == "FAIL":
                counts["FAIL"] += 1
            elif decision == "RUN":
                counts["RUN"] += 1
            elif decision == "PASS":
                counts["PASS"] += 1
                counts["RUN"] += 1      # PASS implique RUN
            elif decision == "ROBUST_PASS":
                counts["ROBUST_PASS"] += 1
                counts["PASS"] += 1      # ROBUST_PASS implique PASS
                counts["RUN"] += 1       # ROBUST_PASS implique RUN
                
        return counts

    baseline_counts = count_decisions_hierarchical(baseline)
    hitl_counts = count_decisions_hierarchical(hitl)

    print(f"\n  ┌{'─'*60}┐")
    print(f"  │ {case_name.upper()} Results Summary (Hierarchical)")
    print(f"  ├{'─'*60}┤")
    print(f"  │ Mode      │ FAIL │ RUN │ PASS │ ROBUST │")
    print(f"  ├{'─'*60}┤")
    print(f"  │ BASELINE  │ {baseline_counts['FAIL']:4} │ {baseline_counts['RUN']:3} │ {baseline_counts['PASS']:4} │ {baseline_counts['ROBUST_PASS']:6} │")
    print(f"  │ HITL      │ {hitl_counts['FAIL']:4} │ {hitl_counts['RUN']:3} │ {hitl_counts['PASS']:4} │ {hitl_counts['ROBUST_PASS']:6} │")
    print(f"  └{'─'*60}┘")
    
    # Note explicative
    print(f"\n  Hierarchy: FAIL ⊂ RUN ⊂ PASS ⊂ ROBUST_PASS")
    print(f"  (Each level includes all lower levels)")

    # Afficher les enhancements appliquées
    for mode, results in [("BASELINE", baseline), ("HITL", hitl)]:
        for r in results:
            enhancements = r.get("metric_enhancements", [])
            if enhancements:
                print(f"\n  [ENHANCEMENTS] {mode} {r.get('candidate_id', '?')}:")
                for enh in enhancements[:3]:
                    print(f"    - {enh}")


# ============================================================================
# Results Loading and Reporting
# ============================================================================

def load_existing_results(reports_dir: Path, cases: List[str]) -> List[Dict[str, Any]]:
    """Charge les résultats existants depuis le dossier reports."""
    results: List[Dict[str, Any]] = []

    for case_name in cases:
        path = reports_dir / f"{case_name}_raw_results.json"
        if not path.exists():
            print(f"[WARN] Missing raw results for {case_name}: {path}")
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                results.append(json.load(f))
        except json.JSONDecodeError:
            # Essayer avec BOM
            with path.open("r", encoding="utf-8-sig") as f:
                results.append(json.load(f))

    return results


def build_reports(reports_dir: Path, experiment_results: List[Dict[str, Any]]) -> None:
    """Génère tous les rapports à partir des résultats agrégés."""
    if not experiment_results:
        raise RuntimeError("No experiment results available to aggregate.")

    # Utiliser le comptage hiérarchique
    aggregator = ResultsAggregator(hierarchical_counting=True)
    table_builder = TableBuilder(hierarchical_counting=True)
    summary_builder = SummaryBuilder()
    csv_exporter = CSVExporter()

    # Agrégation
    aggregated = []
    for r in experiment_results:
        try:
            agg = aggregator.aggregate_case(r)
            aggregated.append(agg)
        except Exception as e:
            print(f"[WARN] Failed to aggregate case {r.get('case', 'unknown')}: {e}")

    # Sauvegarde des résultats agrégés
    (reports_dir / "aggregated_results.json").write_text(
        json.dumps(aggregated, indent=2, default=str),
        encoding="utf-8",
    )

    # Construction des tables
    iteration_table = table_builder.build_iteration_table(aggregated)
    best_candidates_table = table_builder.build_best_candidates_table(aggregated)
    
    # Tableau format papier (Markdown)
    paper_table_md = table_builder.build_paper_summary_table(aggregated)

    # Résumé Markdown pour le papier
    summary_md = summary_builder.build_markdown_summary(aggregated)
    
    # Combiner avec le tableau papier
    full_paper_md = f"{paper_table_md}\n\n{summary_md}"

    # Export CSV
    csv_exporter.export(iteration_table, reports_dir / "table_iterations.csv")
    csv_exporter.export(best_candidates_table, reports_dir / "table_best_candidates.csv")
    (reports_dir / "paper_summary.md").write_text(full_paper_md, encoding="utf-8")

    print(f"\n[REPORTS] Generated:")
    print(f"  - {reports_dir / 'aggregated_results.json'}")
    print(f"  - {reports_dir / 'table_iterations.csv'}")
    print(f"  - {reports_dir / 'table_best_candidates.csv'}")
    print(f"  - {reports_dir / 'paper_summary.md'}")


def build_taxonomy_report(reports_dir: Path, experiment_results: List[Dict[str, Any]]) -> None:
    """Construit un rapport de taxonomie d'erreurs."""
    taxonomy_counts: Dict[str, int] = {}

    for case_result in experiment_results:
        for mode in ["baseline", "hitl"]:
            for candidate in case_result.get(mode, []):
                sanity = candidate.get("sanity", {})
                for issue in sanity.get("issues", []):
                    code = issue.get("code", "UNKNOWN")
                    taxonomy_counts[code] = taxonomy_counts.get(code, 0) + 1

                parsed = candidate.get("parsed_log", {})
                for error in parsed.get("errors", []):
                    if "syntax error" in error.lower():
                        taxonomy_counts["E-SYNTAX-01"] = taxonomy_counts.get("E-SYNTAX-01", 0) + 1
                    elif "no such vector" in error.lower():
                        taxonomy_counts["E-MEAS-03"] = taxonomy_counts.get("E-MEAS-03", 0) + 1
                    elif "404" in error or "NOT_FOUND" in error:
                        taxonomy_counts["E-API-404"] = taxonomy_counts.get("E-API-404", 0) + 1
                    elif "429" in error or "RESOURCE_EXHAUSTED" in error:
                        taxonomy_counts["E-API-429"] = taxonomy_counts.get("E-API-429", 0) + 1
                    elif "402" in error or "Insufficient Balance" in error:
                        taxonomy_counts["E-API-402"] = taxonomy_counts.get("E-API-402", 0) + 1

    # Sauvegarder la taxonomie
    taxonomy_path = reports_dir / "taxonomy_counts.json"
    taxonomy_path.write_text(
        json.dumps(taxonomy_counts, indent=2),
        encoding="utf-8",
    )
    print(f"\n[TAXONOMY] Saved to {taxonomy_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Point d'entrée principal."""
    print("\n" + "="*70)
    print("  LLM-SPICE Reproducible Framework - Full Reproduction")
    print("="*70)

    # Charger la configuration
    root_config = build_root_config()

    # Créer le dossier de rapports
    reports_dir = ROOT / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Liste des use cases à exécuter
    execution_cfg = root_config.get("experiment", {}).get("execution", {})
    cases = execution_cfg.get("enabled_use_cases", ["uc1", "uc2", "uc3"])

    print(f"\n[USE CASES] {cases}")

    experiment_results: List[Dict[str, Any]] = []

    for case_name in cases:
        result = run_one_case(root_config, case_name, verbose=True)

        if result is not None:
            # Sauvegarder les résultats bruts
            case_output = reports_dir / f"{case_name}_raw_results.json"
            case_output.write_text(
                json.dumps(result, indent=2, default=str),
                encoding="utf-8"
            )
            print(f"[SAVED] {case_output}")

            experiment_results.append(result)
        else:
            print(f"[SKIPPED] {case_name} - No results generated")

    # Vérifier qu'on a des résultats
    if not experiment_results:
        print("\n[ERROR] No experiment results were generated.")
        print("        Trying to load existing results from reports directory...")
        experiment_results = load_existing_results(reports_dir, cases)

    if experiment_results:
        # Générer les rapports
        build_reports(reports_dir, experiment_results)
        build_taxonomy_report(reports_dir, experiment_results)

        # Afficher un résumé global
        print("\n" + "="*70)
        print("  FINAL SUMMARY")
        print("="*70)
        _print_global_summary(experiment_results)
    else:
        print("\n[FATAL] No results available. Cannot generate reports.")

    print("\n[DONE] Reproduction complete.")
    print(f"[REPORTS] {reports_dir}")


def _print_global_summary(results: List[Dict[str, Any]]) -> None:
    """Affiche un résumé global avec comptage hiérarchique complet : FAIL ⊂ RUN ⊂ PASS ⊂ ROBUST_PASS."""
    total_baseline = {"FAIL": 0, "RUN": 0, "PASS": 0, "ROBUST_PASS": 0}
    total_hitl = {"FAIL": 0, "RUN": 0, "PASS": 0, "ROBUST_PASS": 0}

    for case in results:
        for r in case.get("baseline", []):
            dec = r.get("final_decision", "FAIL")
            
            if dec == "FAIL":
                total_baseline["FAIL"] += 1
            elif dec == "RUN":
                total_baseline["RUN"] += 1
            elif dec == "PASS":
                total_baseline["PASS"] += 1
                total_baseline["RUN"] += 1
            elif dec == "ROBUST_PASS":
                total_baseline["ROBUST_PASS"] += 1
                total_baseline["PASS"] += 1
                total_baseline["RUN"] += 1
                
        for r in case.get("hitl", []):
            dec = r.get("final_decision", "FAIL")
            
            if dec == "FAIL":
                total_hitl["FAIL"] += 1
            elif dec == "RUN":
                total_hitl["RUN"] += 1
            elif dec == "PASS":
                total_hitl["PASS"] += 1
                total_hitl["RUN"] += 1
            elif dec == "ROBUST_PASS":
                total_hitl["ROBUST_PASS"] += 1
                total_hitl["PASS"] += 1
                total_hitl["RUN"] += 1

    print("\n  Overall Totals (Hierarchical):")
    print(f"    BASELINE: FAIL={total_baseline['FAIL']}, RUN={total_baseline['RUN']}, "
          f"PASS={total_baseline['PASS']}, ROBUST_PASS={total_baseline['ROBUST_PASS']}")
    print(f"    HITL:     FAIL={total_hitl['FAIL']}, RUN={total_hitl['RUN']}, "
          f"PASS={total_hitl['PASS']}, ROBUST_PASS={total_hitl['ROBUST_PASS']}")
    
    print(f"\n    Hierarchy: FAIL ⊂ RUN ⊂ PASS ⊂ ROBUST_PASS")
    print(f"    (Each level includes all lower levels)")

    # Totaux réels (nombre de candidats)
    baseline_total = 0
    hitl_total = 0
    for case in results:
        baseline_total += len(case.get("baseline", []))
        hitl_total += len(case.get("hitl", []))

    if baseline_total > 0:
        baseline_success = total_baseline['PASS']
        print(f"\n    BASELINE success rate: {baseline_success}/{baseline_total} ({100*baseline_success/baseline_total:.1f}%)")

    if hitl_total > 0:
        hitl_success = total_hitl['PASS']
        print(f"    HITL success rate:     {hitl_success}/{hitl_total} ({100*hitl_success/hitl_total:.1f}%)")


if __name__ == "__main__":
    main()