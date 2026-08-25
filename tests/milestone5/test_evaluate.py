"""Tests for model evaluation."""

import pandas as pd
import pytest

from myrewards_intelligence.core.evidence import ModelBasis
from myrewards_intelligence.milestone5_member_impact_scoring.evaluate import (
    EvaluationResults,
    compare_models,
    evaluate_model,
    format_metrics_table,
    generate_subgroup_metrics,
)
from myrewards_intelligence.milestone5_member_impact_scoring.train import train_model


@pytest.fixture
def trained_model(sample_analytical_dataset: pd.DataFrame):
    """Train a model for evaluation tests."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    return train_model(incomm_df, ModelBasis.MODEL_A_INCOMM, model_type="logistic")


def test_evaluate_model_returns_results(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify evaluation returns EvaluationResults."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    results = evaluate_model(trained_model, incomm_df)

    assert isinstance(results, EvaluationResults)


def test_evaluate_model_auc_in_range(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify AUC is between 0 and 1."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    results = evaluate_model(trained_model, incomm_df)

    assert 0 <= results.auc_roc <= 1


def test_evaluate_model_brier_in_range(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify Brier score is between 0 and 1."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    results = evaluate_model(trained_model, incomm_df)

    assert 0 <= results.brier_score <= 1


def test_evaluate_model_precision_recall_in_range(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify precision and recall are between 0 and 1."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    results = evaluate_model(trained_model, incomm_df)

    assert 0 <= results.precision_at_threshold <= 1
    assert 0 <= results.recall_at_threshold <= 1


def test_evaluate_model_includes_label_composition(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify label source composition is included."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    results = evaluate_model(trained_model, incomm_df)

    assert "InComm-sourced" in results.label_source_composition


def test_generate_subgroup_metrics(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify subgroup metrics function runs without error."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    metrics = generate_subgroup_metrics(
        trained_model, incomm_df, subgroup_columns=["care_gap_code"]
    )

    # Note: may return empty list if no subgroups have >= 10 samples in test set
    assert isinstance(metrics, list)
    if len(metrics) > 0:
        assert all("subgroup_column" in m for m in metrics)
        assert all("auc" in m for m in metrics)


def test_format_metrics_table(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify metrics table formatting."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    results = evaluate_model(trained_model, incomm_df)
    table = format_metrics_table(results)

    assert len(table) > 0
    assert all("metric" in row and "value" in row for row in table)


def test_compare_models_includes_note() -> None:
    """Verify model comparison includes label source note."""
    results_a = EvaluationResults(
        model_basis="InComm-sourced",
        label_source_composition="100% InComm-sourced",
        auc_roc=0.75,
        brier_score=0.2,
        threshold=0.5,
        precision_at_threshold=0.7,
        recall_at_threshold=0.6,
        f1_at_threshold=0.65,
        accuracy_at_threshold=0.7,
    )
    results_b = EvaluationResults(
        model_basis="Expanded (mixed)",
        label_source_composition="50% InComm-sourced, 50% research-computed",
        auc_roc=0.78,
        brier_score=0.18,
        threshold=0.5,
        precision_at_threshold=0.72,
        recall_at_threshold=0.62,
        f1_at_threshold=0.67,
        accuracy_at_threshold=0.72,
    )

    comparison = compare_models(results_a, results_b)

    assert "note" in comparison
    assert "mixed-label" in comparison["note"].lower() or "research-computed" in comparison["note"].lower()
