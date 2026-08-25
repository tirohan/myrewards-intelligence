"""Word deliverables for Milestone 5.

Generates the 5 required deliverables:
1. Member Impact Scoring model v1 summary
2. Model performance summary
3. SHAP explanation summary
4. Member prioritization output
5. Suggested derived-layer integration approach
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from docx import Document
from docx.shared import Pt

from ..core.config import Settings, resolve_path
from ..core.evidence import ModelBasis
from ..core.reporting import HEADER_LABELS, add_bullets, add_table, new_doc
from .evaluate import EvaluationResults, format_metrics_table
from .explain import ShapExplanations, format_global_importance_table, format_individual_explanation
from .train import TrainedModel

logger = logging.getLogger("myrewards_intelligence")

HEADER_LABELS.update(
    {
        "model_variant": "Model",
        "label_source_basis": "Label Source Basis",
        "auc_roc": "AUC-ROC",
        "brier_score": "Brier Score",
        "operating_threshold": "Operating Threshold",
        "subgroup_column": "Subgroup Type",
        "subgroup_value": "Subgroup",
        "rank": "Rank",
        "impact_score": "Impact Score",
        "score_tier": "Score Tier",
        "member_id": "Member ID",
        "care_gap_code": "Care Gap Code",
    }
)


def build_model_summary_report(
    trained_model_a: TrainedModel,
    trained_model_b: TrainedModel,
    dataset_summary: dict[str, Any],
) -> Document:
    """Build the Member Impact Scoring model v1 summary document."""
    doc = new_doc(
        "Member Impact Scoring Model v1",
        "Milestone 5 — InComm MyRewards / KSU mHealth Lab",
    )

    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(
        "This document summarizes the Member Impact Scoring model developed in Milestone 5. "
        "The model predicts the probability of care gap closure for each member/care-gap "
        "combination, enabling prioritized outreach to members most likely to respond. "
        "Two model variants were trained to maintain transparency about label source composition."
    )

    doc.add_heading("Model Variants", level=1)
    doc.add_paragraph(
        "Per the project's non-negotiable principle of distinguishing InComm-sourced data "
        "from research-computed data, two model variants were trained:"
    )

    p = doc.add_paragraph()
    run = p.add_run("Model A (Primary): ")
    run.bold = True
    p.add_run(
        f"Trained exclusively on {dataset_summary.get('incomm_sourced_rows', 0):,} InComm-sourced "
        f"labels ({dataset_summary.get('incomm_sourced_members', 0):,} unique members). "
        "This is the model whose metrics reflect real closure behavior as confirmed by InComm's "
        "own data. All primary performance metrics and SHAP explanations come from this model."
    )

    p = doc.add_paragraph()
    run = p.add_run("Model B (Expanded, Exploratory): ")
    run.bold = True
    p.add_run(
        f"Trained on the full {dataset_summary.get('total_rows', 0):,} rows including "
        f"{dataset_summary.get('research_computed_rows', 0):,} research-computed labels. "
        "Reported separately with label-source composition always disclosed. Useful for "
        "understanding what performance might look like with more data, but should not be "
        "presented as InComm-validated."
    )

    doc.add_heading("Dataset", level=1)
    dataset_rows = [
        {"metric": "Total rows", "value": f"{dataset_summary.get('total_rows', 0):,}"},
        {"metric": "Unique members", "value": f"{dataset_summary.get('unique_members', 0):,}"},
        {"metric": "Care gap codes", "value": str(len(dataset_summary.get('care_gaps', [])))},
        {"metric": "InComm-sourced rows", "value": f"{dataset_summary.get('incomm_sourced_rows', 0):,}"},
        {"metric": "Research-computed rows", "value": f"{dataset_summary.get('research_computed_rows', 0):,}"},
        {"metric": "Overall closure rate", "value": f"{dataset_summary.get('overall_closure_rate', 0):.1%}"},
        {"metric": "InComm closure rate", "value": f"{dataset_summary.get('incomm_closure_rate', 0):.1%}"},
    ]
    add_table(doc, ["metric", "value"], dataset_rows)

    doc.add_heading("Model Architecture", level=1)
    doc.add_paragraph(
        f"Primary model: LightGBM gradient boosted trees with {len(trained_model_a.feature_names)} features. "
        "Logistic regression baseline trained for comparison."
    )
    add_bullets(
        doc,
        [
            "Grouped train/test split on member_id to prevent leakage",
            "80/20 train/test ratio",
            "LightGBM with default hyperparameters (200 trees, learning rate 0.05)",
            "SHAP TreeExplainer for model interpretation",
        ],
    )

    doc.add_heading("Features", level=1)
    doc.add_paragraph("The model uses the following feature groups from the Milestone 4 analytical dataset:")
    add_bullets(
        doc,
        [
            "Care-gap features: days since gap opened, prior closure rate on other measures",
            "Member demographics: age, sex, primary condition, condition count",
            "Member engagement: claim/OTC/benefit transaction counts, total/healthy spend",
            "Reward timing features: rewards issued/claimed, redemption rate, avg days to claim",
        ],
    )

    doc.add_heading("Data Leakage Analysis", level=1)
    p = doc.add_paragraph()
    run = p.add_run("Important: ")
    run.bold = True
    p.add_run(
        "During validation, we identified and removed features that encode the outcome directly:"
    )

    add_bullets(
        doc,
        [
            "qualifying_claim_count — EXCLUDED: A qualifying claim IS the clinical event that "
            "closes a care gap (e.g., A1C test for diabetes gap). Including this feature would "
            "be circular reasoning, not prediction.",
            "days_since_last_qualifying_claim — EXCLUDED: Derived from the outcome event.",
        ],
    )

    doc.add_paragraph(
        "This correction ensures the model predicts future closure likelihood based on "
        "available information at prediction time, rather than simply detecting already-closed gaps."
    )

    doc.add_heading("Validation Recommendations", level=1)
    doc.add_paragraph(
        "Before production deployment, the following validation steps are recommended:"
    )
    add_bullets(
        doc,
        [
            "Temporal validation: Confirm reward features are computed using only pre-outcome data",
            "Out-of-time testing: Train on historical periods, test on future periods",
            "A/B testing: Validate model lift in controlled pilot deployment",
            "Domain review: Verify feature definitions with InComm data engineering team",
        ],
    )

    return doc


def build_performance_report(
    results_a: EvaluationResults,
    results_b: EvaluationResults,
    subgroup_metrics_a: list[dict[str, Any]],
) -> Document:
    """Build the Model Performance Summary document."""
    doc = new_doc(
        "Model Performance Summary",
        "Milestone 5 — InComm MyRewards / KSU mHealth Lab",
    )

    doc.add_heading("Model A Performance (Primary)", level=1)
    p = doc.add_paragraph()
    run = p.add_run(f"Label Source: {results_a.label_source_composition}")
    run.bold = True

    add_table(doc, ["metric", "value"], format_metrics_table(results_a))

    doc.add_heading("Model B Performance (Expanded, Exploratory)", level=1)
    p = doc.add_paragraph()
    run = p.add_run(f"Label Source: {results_b.label_source_composition}")
    run.bold = True

    p = doc.add_paragraph()
    run = p.add_run(
        "Note: Model B's metrics reflect a mixed-label dataset. Improvements over Model A "
        "may reflect additional data volume rather than true signal."
    )
    run.italic = True

    add_table(doc, ["metric", "value"], format_metrics_table(results_b))

    doc.add_heading("Model Comparison", level=1)
    comparison_rows = [
        {
            "metric": "AUC-ROC",
            "model_a": f"{results_a.auc_roc:.4f}",
            "model_b": f"{results_b.auc_roc:.4f}",
            "difference": f"{results_b.auc_roc - results_a.auc_roc:+.4f}",
        },
        {
            "metric": "Brier Score",
            "model_a": f"{results_a.brier_score:.4f}",
            "model_b": f"{results_b.brier_score:.4f}",
            "difference": f"{results_b.brier_score - results_a.brier_score:+.4f}",
        },
        {
            "metric": f"Precision @ {results_a.threshold}",
            "model_a": f"{results_a.precision_at_threshold:.4f}",
            "model_b": f"{results_b.precision_at_threshold:.4f}",
            "difference": f"{results_b.precision_at_threshold - results_a.precision_at_threshold:+.4f}",
        },
        {
            "metric": f"Recall @ {results_a.threshold}",
            "model_a": f"{results_a.recall_at_threshold:.4f}",
            "model_b": f"{results_b.recall_at_threshold:.4f}",
            "difference": f"{results_b.recall_at_threshold - results_a.recall_at_threshold:+.4f}",
        },
    ]
    add_table(doc, ["metric", "model_a", "model_b", "difference"], comparison_rows)

    doc.add_heading("Subgroup Performance (Model A)", level=1)
    doc.add_paragraph(
        "Performance broken out by care gap code and primary condition. "
        "Wide confidence intervals expected for smaller subgroups."
    )

    if subgroup_metrics_a:
        subgroup_rows = [
            {
                "subgroup_column": m["subgroup_column"],
                "subgroup_value": m["subgroup_value"],
                "n_samples": m["n_samples"],
                "base_rate": f"{m['base_rate']:.1%}",
                "auc": f"{m['auc']:.4f}" if not pd.isna(m["auc"]) else "N/A",
                "precision": f"{m['precision']:.4f}",
                "recall": f"{m['recall']:.4f}",
            }
            for m in subgroup_metrics_a
        ]
        add_table(
            doc,
            ["subgroup_column", "subgroup_value", "n_samples", "base_rate", "auc", "precision", "recall"],
            subgroup_rows,
        )

    doc.add_heading("Calibration", level=1)
    doc.add_paragraph(
        f"Model A Brier score: {results_a.brier_score:.4f} (lower is better, 0 = perfect). "
        "A calibrated model's predicted probabilities should match observed frequencies."
    )

    doc.add_heading("Validation Notes", level=1)
    doc.add_paragraph(
        "These metrics were computed after removing features that encode the outcome directly "
        "(qualifying_claim_count, days_since_last_qualifying_claim). The reported AUC values "
        "represent predictive performance using only features available at prediction time."
    )

    p = doc.add_paragraph()
    run = p.add_run("Industry Benchmarks for Care Gap Prediction: ")
    run.bold = True

    add_bullets(
        doc,
        [
            "AUC 0.50-0.60: Poor (barely better than random)",
            "AUC 0.60-0.70: Fair (some predictive signal)",
            "AUC 0.70-0.80: Good (useful for prioritization)",
            "AUC 0.80-0.90: Very Good (strong predictive model)",
            "AUC 0.90-1.00: Excellent (verify no data leakage)",
        ],
    )

    return doc


def build_shap_report(
    explanations: ShapExplanations,
    trained_model: TrainedModel,
) -> Document:
    """Build the SHAP Explanation Summary document."""
    doc = new_doc(
        "SHAP Explanation Summary",
        "Milestone 5 — InComm MyRewards / KSU mHealth Lab",
    )

    doc.add_heading("Purpose", level=1)
    doc.add_paragraph(
        "This document provides SHAP-based explanations for Model A (the primary, InComm-sourced model). "
        "SHAP values decompose each prediction into feature contributions, enabling: "
        "(1) global understanding of what drives closure probability across all members, and "
        "(2) individual explanations for specific member/care-gap predictions."
    )

    doc.add_heading("Global Feature Importance", level=1)
    doc.add_paragraph(
        "Features ranked by mean absolute SHAP value across the test set. "
        "Higher values indicate greater influence on predictions."
    )
    add_table(doc, ["rank", "feature_name", "importance"], format_global_importance_table(explanations))

    doc.add_heading("Individual Member Explanations", level=1)
    doc.add_paragraph(
        "Example explanations showing the top contributing features for specific members. "
        "Positive SHAP values increase predicted closure probability; negative values decrease it."
    )

    for i, explanation in enumerate(explanations.individual_explanations[:3]):
        member_info = explanation.get("member_info", {})
        doc.add_heading(
            f"Example {i + 1}: {member_info.get('member_id', 'Unknown')} / {member_info.get('care_gap_code', 'Unknown')}",
            level=2,
        )
        doc.add_paragraph(
            f"Predicted probability: {explanation['predicted_probability']:.1%} | "
            f"Actual status: {member_info.get('actual_status', 'Unknown')}"
        )
        add_table(
            doc,
            ["feature_name", "value", "shap_value", "impact"],
            format_individual_explanation(explanation),
        )

    doc.add_heading("Interpretation Guidelines", level=1)
    add_bullets(
        doc,
        [
            "SHAP values are additive: the sum of all feature contributions plus the base value equals the model output.",
            "Feature importance reflects both the magnitude of effect and how often that effect occurs.",
            "Individual explanations show local behavior; global importance shows overall model behavior.",
            "These explanations apply to Model A only; Model B explanations would reflect the mixed-label dataset.",
        ],
    )

    return doc


def build_prioritization_report(
    prioritized_df: pd.DataFrame,
    model_basis: ModelBasis,
) -> Document:
    """Build the Member Prioritization Output document."""
    doc = new_doc(
        "Member Prioritization Output",
        "Milestone 5 — InComm MyRewards / KSU mHealth Lab",
    )

    doc.add_heading("Purpose", level=1)
    doc.add_paragraph(
        "This document provides ranked member lists per care gap, sorted by predicted "
        "closure probability (Impact Score). Members at the top of each list are predicted "
        "to have the highest likelihood of closing their care gap if engaged."
    )

    p = doc.add_paragraph()
    run = p.add_run(f"Model Basis: {model_basis.value}")
    run.bold = True

    doc.add_heading("Score Tier Distribution", level=1)
    tier_counts = prioritized_df["ScoreTier"].value_counts().to_dict()
    tier_rows = [
        {"score_tier": tier, "count": count, "percentage": f"{count / len(prioritized_df):.1%}"}
        for tier, count in sorted(tier_counts.items())
    ]
    add_table(doc, ["score_tier", "count", "percentage"], tier_rows)

    doc.add_heading("Top Members by Care Gap", level=1)
    for gap_code in sorted(prioritized_df["CareGapCode"].unique()):
        gap_df = prioritized_df[prioritized_df["CareGapCode"] == gap_code]
        doc.add_heading(f"{gap_code} (n={len(gap_df)})", level=2)

        top_rows = gap_df.head(20).to_dict("records")
        display_rows = [
            {
                "rank": r["Rank"],
                "member_id": r["MemberId"],
                "impact_score": f"{r['ImpactScore']:.4f}",
                "score_tier": r["ScoreTier"],
            }
            for r in top_rows
        ]
        add_table(doc, ["rank", "member_id", "impact_score", "score_tier"], display_rows)

    return doc


def build_derived_layer_report(
    ddl: str,
    trained_model: TrainedModel,
) -> Document:
    """Build the Suggested Derived-Layer Integration Approach document."""
    doc = new_doc(
        "Derived Layer Integration Approach",
        "Milestone 5 — InComm MyRewards / KSU mHealth Lab",
    )

    doc.add_heading("Purpose", level=1)
    doc.add_paragraph(
        "This document proposes how the Member Impact Scoring model output would integrate "
        "into InComm's existing derived data layer. The proposal mirrors the existing "
        "der_MemberExperienceScores table structure for consistency."
    )

    doc.add_heading("Proposed Table: der_MemberImpactScores", level=1)
    doc.add_paragraph(
        "A new derived table storing per-member, per-care-gap impact scores. "
        "Key design decisions:"
    )
    add_bullets(
        doc,
        [
            "One row per (MemberId, CareGapCode) combination",
            "Calibrated probability stored in ImpactScore column (0.0 to 1.0)",
            "Score tier (High/Medium/Low) for operational filtering",
            "LabelSourceBasis column ensures label source transparency",
            "TopContributingFeaturesJson provides per-row SHAP explanation",
            "Foreign key to der_MlPipelineRuns for version tracking",
        ],
    )

    doc.add_heading("Integration with Existing Tables", level=1)
    doc.add_paragraph(
        "der_MemberImpactScores would sit alongside existing derived tables:"
    )
    add_bullets(
        doc,
        [
            "der_MemberFeatureSnapshots: Member-level features (already consumed by this model)",
            "der_MemberCareGapStatuses: Care gap open/closed status (source of InComm-sourced labels)",
            "der_MemberExperienceScores: Experience/engagement scores (structural pattern followed here)",
            "der_RecommendationCaseOutcomes: Outcome tracking (potential future label source)",
        ],
    )

    doc.add_heading("LabelSourceBasis Column", level=1)
    p = doc.add_paragraph()
    run = p.add_run(
        "The LabelSourceBasis column is unique to this table and ensures that no reader "
        "can mistake a Model-B-derived score for one trained on InComm's real data. "
    )
    run.bold = True
    doc.add_paragraph(
        "Values: 'InComm-sourced' (Model A) or 'Expanded (mixed)' (Model B). "
        "Downstream queries can filter to InComm-sourced scores only for production use cases."
    )

    doc.add_heading("DDL Proposal", level=1)
    doc.add_paragraph("The following DDL is a proposal for InComm review:")
    p = doc.add_paragraph()
    run = p.add_run(ddl[:2000])
    run.font.size = Pt(8)

    doc.add_heading("Deployment Recommendation", level=1)
    doc.add_paragraph(
        "This DDL should be reviewed by InComm's database team before any deployment. "
        "The research team has not applied this to any live database — it exists only as "
        "a proposal artifact."
    )

    return doc


def write_all_reports(
    trained_model_a: TrainedModel,
    trained_model_b: TrainedModel,
    results_a: EvaluationResults,
    results_b: EvaluationResults,
    subgroup_metrics_a: list[dict[str, Any]],
    explanations: ShapExplanations,
    prioritized_df: pd.DataFrame,
    ddl: str,
    dataset_summary: dict[str, Any],
    settings: Settings | None = None,
) -> list[str]:
    """Write all five Milestone 5 Word deliverables."""
    if settings is None:
        from ..core.config import load_config
        settings = load_config()

    out_dir = resolve_path(settings.paths.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    specs = [
        (
            "Member Impact Scoring Model v1.docx",
            build_model_summary_report(trained_model_a, trained_model_b, dataset_summary),
        ),
        (
            "Model Performance Summary.docx",
            build_performance_report(results_a, results_b, subgroup_metrics_a),
        ),
        (
            "SHAP Explanation Summary.docx",
            build_shap_report(explanations, trained_model_a),
        ),
        (
            "Member Prioritization Output.docx",
            build_prioritization_report(prioritized_df, trained_model_a.model_basis),
        ),
        (
            "Derived Layer Integration Approach.docx",
            build_derived_layer_report(ddl, trained_model_a),
        ),
    ]

    for filename, doc in specs:
        path = out_dir / filename
        doc.save(path)
        written.append(str(path))
        logger.info("Wrote report: %s", path)

    return written
