"""Tests for SHAP explanations."""

import pandas as pd
import pytest

from myrewards_intelligence.core.evidence import ModelBasis
from myrewards_intelligence.milestone5_member_impact_scoring.explain import (
    ShapExplanations,
    format_global_importance_table,
    format_individual_explanation,
    generate_shap_explanations,
)
from myrewards_intelligence.milestone5_member_impact_scoring.train import train_model


@pytest.fixture
def trained_model(sample_analytical_dataset: pd.DataFrame):
    """Train a model for SHAP tests."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    return train_model(incomm_df, ModelBasis.MODEL_A_INCOMM, model_type="lightgbm")


def test_generate_shap_explanations_returns_results(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify SHAP explanations are generated."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    explanations = generate_shap_explanations(trained_model, incomm_df, n_samples=20, n_individual=2)

    assert isinstance(explanations, ShapExplanations)


def test_generate_shap_explanations_has_global_importance(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify global importance is computed."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    explanations = generate_shap_explanations(trained_model, incomm_df, n_samples=20, n_individual=2)

    assert len(explanations.global_importance) > 0
    assert all("feature_name" in item for item in explanations.global_importance)
    assert all("importance" in item for item in explanations.global_importance)


def test_generate_shap_explanations_has_individual_explanations(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify individual explanations are generated."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    explanations = generate_shap_explanations(trained_model, incomm_df, n_samples=20, n_individual=2)

    assert len(explanations.individual_explanations) > 0


def test_global_importance_is_ranked(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify global importance is ranked by magnitude."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    explanations = generate_shap_explanations(trained_model, incomm_df, n_samples=20, n_individual=2)

    importances = [item["importance"] for item in explanations.global_importance]
    assert importances == sorted(importances, reverse=True)


def test_format_global_importance_table(
    trained_model, sample_analytical_dataset: pd.DataFrame
) -> None:
    """Verify global importance table formatting."""
    incomm_df = sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()
    explanations = generate_shap_explanations(trained_model, incomm_df, n_samples=20, n_individual=2)
    table = format_global_importance_table(explanations, top_n=5)

    assert len(table) <= 5
    assert all("rank" in row for row in table)


def test_format_individual_explanation() -> None:
    """Verify individual explanation formatting."""
    explanation = {
        "sample_index": 0,
        "top_contributing_features": [
            {"feature": "age", "value": 45.0, "shap_value": 0.15, "direction": "increases"},
            {"feature": "spend", "value": 500.0, "shap_value": -0.1, "direction": "decreases"},
        ],
        "predicted_probability": 0.65,
    }

    formatted = format_individual_explanation(explanation)

    assert len(formatted) == 2
    assert all("feature_name" in row for row in formatted)
    assert all("impact" in row for row in formatted)
