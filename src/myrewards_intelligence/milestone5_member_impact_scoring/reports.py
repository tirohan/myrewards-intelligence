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


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def build_followup_report(diagnostics: dict[str, Any]) -> Document:
    """Dedicated report answering InComm technical-review questions."""
    doc = new_doc(
        "Milestone 5 Follow-up Analyses",
        "Responses to InComm technical review questions — Model A (InComm-sourced)",
    )

    seeds = diagnostics.get("seeds", [])
    doc.add_heading("Purpose", level=1)
    doc.add_paragraph(
        "This report answers the follow-up questions from the Milestone 5 review. "
        "Where training was required, Model A was retrained across multiple random seeds "
        f"({', '.join(str(s) for s in seeds)}) using a member-grouped split. "
        "Headline v1 metrics in the original five deliverables used seed 42; this document "
        "shows whether those results are stable."
    )

    balance = diagnostics.get("class_balance", {})
    doc.add_heading("1. Closure rate and class imbalance", level=1)
    add_table(
        doc,
        ["metric", "value"],
        [
            {"metric": "Model A rows", "value": f"{balance.get('n_rows', 0):,}"},
            {"metric": "Unique members", "value": f"{balance.get('n_members', 0):,}"},
            {"metric": "Closed / Open", "value": f"{balance.get('n_closed', 0):,} / {balance.get('n_open', 0):,}"},
            {"metric": "Closure rate", "value": _fmt_pct(balance.get("closure_rate"))},
            {"metric": "Imbalance", "value": balance.get("imbalance", "—")},
            {"metric": "Handling", "value": balance.get("handling", "—")},
        ],
    )

    contract = diagnostics.get("contract", {})
    doc.add_heading("2. Column contract and missingness", level=1)
    doc.add_paragraph(contract.get("note", ""))
    add_table(
        doc,
        ["metric", "value"],
        [
            {"metric": "Analytical dataset columns", "value": str(contract.get("analytical_dataset_columns", "—"))},
            {"metric": "Loader required columns", "value": str(contract.get("required_columns_in_loader", "—"))},
            {"metric": "Model features", "value": str(contract.get("model_feature_count", "—"))},
        ],
    )

    missing = diagnostics.get("missingness", {})
    doc.add_paragraph(missing.get("policy", ""))
    miss_rows = [
        {
            "feature": r["feature"],
            "pct_rows_missing": _fmt_pct(r["pct_rows_missing"]),
            "pct_members_with_any_missing": _fmt_pct(r["pct_members_with_any_missing"]),
            "n_rows_missing": r["n_rows_missing"],
        }
        for r in missing.get("features", [])
    ]
    if miss_rows:
        add_table(
            doc,
            ["feature", "n_rows_missing", "pct_rows_missing", "pct_members_with_any_missing"],
            miss_rows,
        )

    dropna = diagnostics.get("variants", {}).get("dropna_prior_closure", {})
    drop_sum = dropna.get("auc_summary", {})
    base_sum = diagnostics.get("variants", {}).get("baseline", {}).get("auc_summary", {})
    doc.add_paragraph(
        "Sensitivity: retraining after dropping rows missing prior_closure_rate_other_measures "
        f"(dropped {dropna.get('n_rows_dropped', 0):,} rows) yields mean AUC "
        f"{_fmt_num(drop_sum.get('mean'))} ± {_fmt_num(drop_sum.get('std'))} versus "
        f"impute-all mean AUC {_fmt_num(base_sum.get('mean'))} ± {_fmt_num(base_sum.get('std'))}."
    )

    doc.add_heading("3. Reward features at prediction time (cold start)", level=1)
    doc.add_paragraph(
        "Never-rewarded members receive rewards_issued_count = 0, rewards_claimed_count = 0, "
        "and redemption_rate = 0. avg_days_issue_to_claim is undefined until a claim occurs "
        "and is median-imputed. The table below is test-set performance on the last baseline seed."
    )
    cold = diagnostics.get("cold_start", {})
    if cold.get("available"):
        add_table(
            doc,
            ["group", "n_test", "closure_rate", "mean_predicted_probability", "auc"],
            [
                {
                    "group": g.get("group"),
                    "n_test": g.get("n_test"),
                    "closure_rate": _fmt_pct(g.get("closure_rate")),
                    "mean_predicted_probability": _fmt_num(g.get("mean_predicted_probability")),
                    "auc": _fmt_num(g.get("auc")),
                }
                for g in cold.get("groups", [])
            ],
        )

    ablation = diagnostics.get("variants", {}).get("reward_ablation", {})
    abl_sum = ablation.get("auc_summary", {})
    doc.add_paragraph(
        "Ablation (same seeds, reward features removed): mean AUC "
        f"{_fmt_num(abl_sum.get('mean'))} ± {_fmt_num(abl_sum.get('std'))}. "
        f"Drop vs full feature set: {_fmt_num(diagnostics.get('reward_ablation_auc_drop'))}. "
        "A large drop means the 0.94-range result is engagement-driven and still depends on "
        "reward aggregates being computed as of scoring time, not after gap closure."
    )

    doc.add_heading("4. Grouped split vs care-gap stratification", level=1)
    doc.add_paragraph(
        "The production split groups on member_id so a member cannot appear in both train and test. "
        "It is not stratified by care_gap_code by default. The tables show mix for the last seed, "
        "then mean AUC across seeds for grouped vs grouped+stratified."
    )
    for key, title in (("grouped", "Grouped only"), ("grouped_stratified", "Grouped + stratified by care gap")):
        comp = diagnostics.get("split_composition", {}).get(key, {})
        doc.add_heading(title, level=2)
        add_table(
            doc,
            ["care_gap_code", "n_train", "n_test", "pct_in_test"],
            [
                {
                    "care_gap_code": r["care_gap_code"],
                    "n_train": r["n_train"],
                    "n_test": r["n_test"],
                    "pct_in_test": _fmt_pct(r["pct_in_test"]),
                }
                for r in comp.get("by_care_gap", [])
            ],
        )

    strat_sum = diagnostics.get("variants", {}).get("stratified", {}).get("auc_summary", {})
    add_table(
        doc,
        ["variant", "mean_auc", "std_auc", "min_auc", "max_auc"],
        [
            {
                "variant": "Grouped (member_id)",
                "mean_auc": _fmt_num(base_sum.get("mean")),
                "std_auc": _fmt_num(base_sum.get("std")),
                "min_auc": _fmt_num(base_sum.get("min")),
                "max_auc": _fmt_num(base_sum.get("max")),
            },
            {
                "variant": "Grouped + stratified by care gap",
                "mean_auc": _fmt_num(strat_sum.get("mean")),
                "std_auc": _fmt_num(strat_sum.get("std")),
                "min_auc": _fmt_num(strat_sum.get("min")),
                "max_auc": _fmt_num(strat_sum.get("max")),
            },
        ],
    )

    doc.add_heading("5. AUC stability and what we treat as acceptable", level=1)
    band = diagnostics.get("acceptable_auc_band", {})
    doc.add_paragraph(band.get("interpretation", ""))
    variant_rows = []
    for key, label in (
        ("baseline", "Full features, grouped split, median impute"),
        ("stratified", "Full features, grouped + stratified"),
        ("reward_ablation", "No reward features"),
        ("dropna_prior_closure", "Drop missing prior_closure_rate"),
    ):
        summary = diagnostics.get("variants", {}).get(key, {}).get("auc_summary", {})
        variant_rows.append(
            {
                "variant": label,
                "mean_auc": _fmt_num(summary.get("mean")),
                "std_auc": _fmt_num(summary.get("std")),
                "min_auc": _fmt_num(summary.get("min")),
                "max_auc": _fmt_num(summary.get("max")),
            }
        )
    add_table(doc, ["variant", "mean_auc", "std_auc", "min_auc", "max_auc"], variant_rows)

    doc.add_heading("Per-seed AUC (baseline grouped split)", level=2)
    add_table(
        doc,
        ["seed", "auc", "brier", "precision", "recall"],
        diagnostics.get("variants", {}).get("baseline", {}).get("per_seed", []),
    )

    doc.add_heading("6. Operating threshold vs outreach volume", level=1)
    thresh = diagnostics.get("thresholds", {})
    doc.add_paragraph(thresh.get("reporting_threshold_note", ""))
    add_table(
        doc,
        ["threshold", "precision", "recall", "f1", "n_flagged", "pct_flagged"],
        [
            {
                "threshold": r["threshold"],
                "precision": _fmt_num(r["precision"]),
                "recall": _fmt_num(r["recall"]),
                "f1": _fmt_num(r["f1"]),
                "n_flagged": r["n_flagged"],
                "pct_flagged": _fmt_pct(r["pct_flagged"]),
            }
            for r in thresh.get("curve", [])
        ],
    )
    if thresh.get("tiers"):
        doc.add_heading("High / Medium / Low mix on the test set", level=2)
        add_table(
            doc,
            ["tier", "n_test", "pct_test", "closure_rate"],
            [
                {
                    "tier": t["tier"],
                    "n_test": t["n_test"],
                    "pct_test": _fmt_pct(t["pct_test"]),
                    "closure_rate": _fmt_pct(t["closure_rate"]) if t.get("closure_rate") is not None else "—",
                }
                for t in thresh["tiers"]
            ],
        )
    doc.add_paragraph(
        "Who acts on the score is an InComm operating decision. Short care-manager lists "
        "should use a higher threshold (precision). Broader pharmacist or campaign outreach "
        "can use a lower threshold (recall). We will lock cut-points once capacity is specified."
    )

    return doc


def write_followup_markdown(diagnostics: dict[str, Any], path: Any) -> str:
    """Write a Markdown copy of the follow-up report."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seeds = diagnostics.get("seeds", [])
    balance = diagnostics.get("class_balance", {})
    contract = diagnostics.get("contract", {})
    missing = diagnostics.get("missingness", {})
    variants = diagnostics.get("variants", {})
    band = diagnostics.get("acceptable_auc_band", {})
    thresh = diagnostics.get("thresholds", {})

    def line_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for row in rows:
            out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        return "\n".join(out)

    miss_rows = [
        {
            "feature": r["feature"],
            "rows_missing": f"{r['n_rows_missing']} ({r['pct_rows_missing']:.1%})",
            "members_missing": f"{r['n_members_with_any_missing']} ({r['pct_members_with_any_missing']:.1%})",
        }
        for r in missing.get("features", [])
    ]
    variant_rows = []
    labels = {
        "baseline": "Full features, grouped",
        "stratified": "Grouped + stratified",
        "reward_ablation": "No reward features",
        "dropna_prior_closure": "Drop missing prior_closure_rate",
    }
    for key, label in labels.items():
        s = variants.get(key, {}).get("auc_summary", {})
        variant_rows.append(
            {
                "variant": label,
                "mean_auc": _fmt_num(s.get("mean")),
                "std": _fmt_num(s.get("std")),
                "min": _fmt_num(s.get("min")),
                "max": _fmt_num(s.get("max")),
            }
        )

    cold_lines = ""
    for g in diagnostics.get("cold_start", {}).get("groups", []):
        cold_lines += (
            f"- {g.get('group')}: n={g.get('n_test')}, closure={_fmt_pct(g.get('closure_rate'))}, "
            f"mean p={_fmt_num(g.get('mean_predicted_probability'))}, AUC={_fmt_num(g.get('auc'))}\n"
        )

    curve_rows = [
        {
            "threshold": r["threshold"],
            "precision": _fmt_num(r["precision"]),
            "recall": _fmt_num(r["recall"]),
            "f1": _fmt_num(r["f1"]),
            "n_flagged": r["n_flagged"],
            "pct_flagged": _fmt_pct(r["pct_flagged"]),
        }
        for r in thresh.get("curve", [])
    ]

    content = f"""# Milestone 5 Follow-up Analyses

Prepared by: KSU mHealth Research Lab

This report answers InComm technical-review questions. Training variants used seeds {seeds}.

## 1. Closure rate and imbalance

- Model A rows: {balance.get('n_rows', 0):,}
- Unique members: {balance.get('n_members', 0):,}
- Closed / Open: {balance.get('n_closed', 0):,} / {balance.get('n_open', 0):,}
- Closure rate: {_fmt_pct(balance.get('closure_rate'))}
- Handling: {balance.get('handling', '')}

## 2. Columns and missingness

{contract.get('note', '')}

{line_table(['feature', 'rows_missing', 'members_missing'], miss_rows)}

Policy: {missing.get('policy', '')}

## 3. Reward features / cold start

Never-rewarded members get zeros on issued/claimed/redemption; avg_days_issue_to_claim is median-imputed.

{cold_lines}

Reward ablation mean AUC: {_fmt_num(variants.get('reward_ablation', {}).get('auc_summary', {}).get('mean'))} (drop vs full: {_fmt_num(diagnostics.get('reward_ablation_auc_drop'))}).

## 4-5. Split, AUC stability, acceptable band

{band.get('interpretation', '')}

{line_table(['variant', 'mean_auc', 'std', 'min', 'max'], variant_rows)}

### Baseline per seed

{line_table(['seed', 'auc', 'brier', 'precision', 'recall'], variants.get('baseline', {}).get('per_seed', []))}

## 6. Threshold vs volume

{thresh.get('reporting_threshold_note', '')}

{line_table(['threshold', 'precision', 'recall', 'f1', 'n_flagged', 'pct_flagged'], curve_rows)}
"""
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote follow-up markdown: %s", path)
    return str(path)


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
