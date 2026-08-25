"""Tests for Word report generation."""

from pathlib import Path

import pandas as pd
import pytest
from docx.document import Document

from myrewards_intelligence.core.evidence import ModelBasis
from myrewards_intelligence.milestone5_member_impact_scoring.dataset import get_dataset_summary
from myrewards_intelligence.milestone5_member_impact_scoring.derived_layer import (
    generate_ddl_proposal,
    generate_member_prioritization,
    generate_scored_output,
)
from myrewards_intelligence.milestone5_member_impact_scoring.evaluate import (
    evaluate_model,
    generate_subgroup_metrics,
)
from myrewards_intelligence.milestone5_member_impact_scoring.explain import (
    ShapExplanations,
)
from myrewards_intelligence.milestone5_member_impact_scoring.reports import (
    build_derived_layer_report,
    build_model_summary_report,
    build_performance_report,
    build_prioritization_report,
    build_shap_report,
    write_all_reports,
)
from myrewards_intelligence.milestone5_member_impact_scoring.train import train_model


@pytest.fixture
def full_results(sample_analytical_dataset: pd.DataFrame):
    """Generate full results for report testing."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()

    trained_model_a = train_model(incomm_df, ModelBasis.MODEL_A_INCOMM, model_type="logistic")
    trained_model_b = train_model(sample_analytical_dataset, ModelBasis.MODEL_B_EXPANDED, model_type="logistic")

    results_a = evaluate_model(trained_model_a, incomm_df)
    results_b = evaluate_model(trained_model_b, sample_analytical_dataset)

    subgroup_metrics_a = generate_subgroup_metrics(
        trained_model_a, incomm_df, subgroup_columns=["care_gap_code"]
    )

    explanations = ShapExplanations(
        feature_names=trained_model_a.feature_names,
        global_importance=[
            {"feature_name": "age", "importance": 0.15, "rank": 1},
            {"feature_name": "spend", "importance": 0.10, "rank": 2},
        ],
        individual_explanations=[
            {
                "sample_index": 0,
                "member_info": {"member_id": "M1", "care_gap_code": "DIAB_A1C_TEST", "actual_status": "Closed"},
                "top_contributing_features": [
                    {"feature": "age", "value": 45.0, "shap_value": 0.15, "direction": "increases"}
                ],
                "predicted_probability": 0.65,
            }
        ],
    )

    scored_df = generate_scored_output(trained_model_a, incomm_df)
    prioritized_df = generate_member_prioritization(scored_df, top_n_per_gap=10)
    ddl = generate_ddl_proposal(trained_model_a)
    dataset_summary = get_dataset_summary(sample_analytical_dataset)

    return {
        "trained_model_a": trained_model_a,
        "trained_model_b": trained_model_b,
        "results_a": results_a,
        "results_b": results_b,
        "subgroup_metrics_a": subgroup_metrics_a,
        "explanations": explanations,
        "prioritized_df": prioritized_df,
        "ddl": ddl,
        "dataset_summary": dataset_summary,
    }


def test_build_model_summary_report(full_results) -> None:
    """Verify model summary report builds."""
    doc = build_model_summary_report(
        full_results["trained_model_a"],
        full_results["trained_model_b"],
        full_results["dataset_summary"],
    )
    assert isinstance(doc, Document)


def test_build_performance_report(full_results) -> None:
    """Verify performance report builds."""
    doc = build_performance_report(
        full_results["results_a"],
        full_results["results_b"],
        full_results["subgroup_metrics_a"],
    )
    assert isinstance(doc, Document)


def test_build_shap_report(full_results) -> None:
    """Verify SHAP report builds."""
    doc = build_shap_report(
        full_results["explanations"],
        full_results["trained_model_a"],
    )
    assert isinstance(doc, Document)


def test_build_prioritization_report(full_results) -> None:
    """Verify prioritization report builds."""
    doc = build_prioritization_report(
        full_results["prioritized_df"],
        full_results["trained_model_a"].model_basis,
    )
    assert isinstance(doc, Document)


def test_build_derived_layer_report(full_results) -> None:
    """Verify derived layer report builds."""
    doc = build_derived_layer_report(
        full_results["ddl"],
        full_results["trained_model_a"],
    )
    assert isinstance(doc, Document)


def test_write_all_reports(full_results, tmp_path: Path) -> None:
    """Verify all 5 reports are written."""
    from myrewards_intelligence.core.config import Settings

    settings = Settings(paths={"reports_dir": str(tmp_path)})

    written = write_all_reports(
        trained_model_a=full_results["trained_model_a"],
        trained_model_b=full_results["trained_model_b"],
        results_a=full_results["results_a"],
        results_b=full_results["results_b"],
        subgroup_metrics_a=full_results["subgroup_metrics_a"],
        explanations=full_results["explanations"],
        prioritized_df=full_results["prioritized_df"],
        ddl=full_results["ddl"],
        dataset_summary=full_results["dataset_summary"],
        settings=settings,
    )

    assert len(written) == 5
    for path in written:
        assert Path(path).exists()
        assert path.endswith(".docx")
