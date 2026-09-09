"""Milestone 5 orchestrator entry point.

Runs the complete pipeline:
1. Load and validate the M4 analytical dataset
2. Split into Model A (InComm-sourced) and Model B (expanded)
3. Train both models (LightGBM + logistic baseline)
4. Evaluate both models
5. Generate SHAP explanations for Model A
6. Write der_MemberImpactScores DDL proposal
7. Generate scored output and member prioritization
8. Write 5 Word deliverables
9. Write Milestone5_Status.md
10. Run pytest suite
11. Print summary
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime

from ..core.config import Settings, load_config, resolve_path
from ..core.evidence import ModelBasis
from ..core.logging import setup_logging
from ..milestone5_member_impact_scoring.dataset import (
    get_dataset_summary,
    load_analytical_dataset,
    validate_dataset,
)
from ..milestone5_member_impact_scoring.derived_layer import (
    generate_ddl_proposal,
    generate_member_prioritization,
    generate_scored_output,
)
from ..milestone5_member_impact_scoring.diagnostics import run_followup_diagnostics
from ..milestone5_member_impact_scoring.evaluate import (
    evaluate_model,
    generate_subgroup_metrics,
)
from ..milestone5_member_impact_scoring.explain import generate_shap_explanations
from ..milestone5_member_impact_scoring.features import split_by_label_source
from ..milestone5_member_impact_scoring.reports import (
    build_followup_report,
    write_all_reports,
    write_followup_markdown,
)
from ..milestone5_member_impact_scoring.train import (
    save_model,
    save_model_metadata,
    train_model,
)


def write_status_report(
    settings: Settings,
    results: dict,
    success: bool,
) -> str:
    """Write Milestone5_Status.md completion summary."""
    reports_dir = resolve_path(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    status_path = reports_dir / "Milestone5_Status.md"

    status = "COMPLETE" if success else "FAILED"
    timestamp = datetime.now(UTC).isoformat()

    def fmt(val: any, decimals: int = 4) -> str:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return f"{val:.{decimals}f}" if decimals > 0 else f"{val:,}"
        return str(val) if val is not None else "N/A"

    content = f"""# Milestone 5 Status Report

**Status:** {status}
**Completed:** {timestamp}

## Summary

- Dataset rows: {fmt(results.get('total_rows'), 0)}
- Unique members: {fmt(results.get('unique_members'), 0)}
- InComm-sourced rows: {fmt(results.get('incomm_sourced_rows'), 0)}
- Research-computed rows: {fmt(results.get('research_computed_rows'), 0)}

## Model A Performance (Primary, InComm-sourced)

- AUC-ROC: {fmt(results.get('model_a_auc'))}
- Brier Score: {fmt(results.get('model_a_brier'))}
- Precision @ 0.5: {fmt(results.get('model_a_precision'))}
- Recall @ 0.5: {fmt(results.get('model_a_recall'))}

## Model B Performance (Expanded, Mixed Labels)

- AUC-ROC: {fmt(results.get('model_b_auc'))}
- Brier Score: {fmt(results.get('model_b_brier'))}

## Deliverables Generated

{chr(10).join(f'- {p}' for p in results.get('reports_written', []))}

## Models Saved

- models/model_a_v1.pkl
- models/model_b_v1.pkl
- models/metadata.json

## Tests

{results.get('test_summary', 'Not run')}
"""

    with open(status_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(status_path)


def run_tests() -> tuple[bool, str]:
    """Run the pytest suite and return (success, summary)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=resolve_path("."),
        )
        success = result.returncode == 0
        summary = f"Exit code: {result.returncode}\n{result.stdout[-1000:] if result.stdout else ''}"
        return success, summary
    except Exception as e:
        return False, f"Failed to run tests: {e}"


def main() -> int:
    """Run the complete Milestone 5 pipeline."""
    settings = load_config()
    logger = setup_logging(
        log_level=settings.logging.level,
        log_dir=settings.paths.logs_dir,
        log_name="milestone5",
    )

    logger.info("=" * 60)
    logger.info("MILESTONE 5: Member Impact Scoring")
    logger.info("=" * 60)

    results: dict = {}

    try:
        logger.info("Step 1: Loading analytical dataset...")
        df = load_analytical_dataset(settings)
        validation = validate_dataset(df)
        if not validation["valid"]:
            logger.error("Dataset validation failed: %s", validation["issues"])
            return 1

        dataset_summary = get_dataset_summary(df)
        results.update(dataset_summary)
        logger.info("Dataset loaded: %d rows, %d unique members", len(df), dataset_summary["unique_members"])

        logger.info("Step 2: Splitting by label source...")
        splits = split_by_label_source(df)
        df_model_a = splits["model_a"]
        df_model_b = splits["model_b"]
        logger.info("Model A: %d rows, Model B: %d rows", len(df_model_a), len(df_model_b))

        logger.info("Step 3: Training Model A (InComm-sourced)...")
        trained_model_a = train_model(df_model_a, ModelBasis.MODEL_A_INCOMM, settings, model_type="lightgbm")

        logger.info("Step 3: Training Model B (expanded)...")
        trained_model_b = train_model(df_model_b, ModelBasis.MODEL_B_EXPANDED, settings, model_type="lightgbm")

        logger.info("Step 4: Evaluating models...")
        results_a = evaluate_model(trained_model_a, df_model_a, threshold=settings.milestone5.operating_threshold)
        results_b = evaluate_model(trained_model_b, df_model_b, threshold=settings.milestone5.operating_threshold)

        results["model_a_auc"] = results_a.auc_roc
        results["model_a_brier"] = results_a.brier_score
        results["model_a_precision"] = results_a.precision_at_threshold
        results["model_a_recall"] = results_a.recall_at_threshold
        results["model_b_auc"] = results_b.auc_roc
        results["model_b_brier"] = results_b.brier_score

        logger.info("Model A AUC: %.4f, Model B AUC: %.4f", results_a.auc_roc, results_b.auc_roc)

        logger.info("Step 4: Generating subgroup metrics...")
        subgroup_metrics_a = generate_subgroup_metrics(
            trained_model_a,
            df_model_a,
            subgroup_columns=["care_gap_code", "PrimaryCondition"],
            threshold=settings.milestone5.operating_threshold,
        )

        logger.info("Step 5: Generating SHAP explanations for Model A...")
        explanations = generate_shap_explanations(trained_model_a, df_model_a, n_samples=100, n_individual=5)
        logger.info("Top feature: %s", explanations.global_importance[0]["feature_name"])

        logger.info("Step 6: Generating DDL proposal...")
        models_dir = resolve_path(settings.paths.models_dir)
        ddl = generate_ddl_proposal(trained_model_a, output_path=models_dir / "der_MemberImpactScores.sql")

        logger.info("Step 6: Generating scored output...")
        scored_df = generate_scored_output(trained_model_a, df_model_a, explanations, settings)
        prioritized_df = generate_member_prioritization(scored_df, top_n_per_gap=100)

        logger.info("Step 7: Saving models...")
        save_model(trained_model_a, models_dir / "model_a_v1.pkl")
        save_model(trained_model_b, models_dir / "model_b_v1.pkl")
        save_model_metadata(trained_model_a, models_dir / "metadata.json")

        logger.info("Step 7: Writing Word deliverables...")
        reports_written = write_all_reports(
            trained_model_a=trained_model_a,
            trained_model_b=trained_model_b,
            results_a=results_a,
            results_b=results_b,
            subgroup_metrics_a=subgroup_metrics_a,
            explanations=explanations,
            prioritized_df=prioritized_df,
            ddl=ddl,
            dataset_summary=dataset_summary,
            settings=settings,
        )
        results["reports_written"] = reports_written
        logger.info("Wrote %d reports", len(reports_written))

        logger.info("Step 7b: Follow-up analyses for InComm review questions...")
        diagnostics = run_followup_diagnostics(df_model_a, settings)
        reports_dir = resolve_path(settings.paths.reports_dir)
        followup_doc = build_followup_report(diagnostics)
        followup_docx = reports_dir / "Milestone 5 Follow-up Analyses.docx"
        followup_doc.save(followup_docx)
        followup_md = write_followup_markdown(
            diagnostics, reports_dir / "Milestone5_Followup_Analyses.md"
        )
        followup_json = reports_dir / "milestone5_followup_diagnostics.json"
        with open(followup_json, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, default=str)
        results["reports_written"] = reports_written + [
            str(followup_docx),
            followup_md,
            str(followup_json),
        ]
        logger.info("Wrote follow-up report: %s", followup_docx)

        logger.info("Step 8: Running tests...")
        tests_passed, test_summary = run_tests()
        results["test_summary"] = test_summary

        logger.info("Step 9: Writing status report...")
        status_path = write_status_report(settings, results, success=True)
        logger.info("Status report: %s", status_path)

        logger.info("=" * 60)
        logger.info("MILESTONE 5 COMPLETE")
        logger.info("=" * 60)
        logger.info("Model A AUC: %.4f (InComm-sourced labels only)", results_a.auc_roc)
        logger.info("Model B AUC: %.4f (expanded dataset)", results_b.auc_roc)
        logger.info("Reports written: %d", len(reports_written))
        logger.info("Tests passed: %s", tests_passed)

        return 0

    except Exception as e:
        logger.exception("Milestone 5 failed: %s", e)
        results["test_summary"] = f"Pipeline failed: {e}"
        write_status_report(settings, results, success=False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
