"""SHAP-based model explanations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap

from .features import prepare_features
from .train import TrainedModel

logger = logging.getLogger("myrewards_intelligence")


@dataclass
class ShapExplanations:
    """Container for SHAP explanation results."""

    feature_names: list[str]
    global_importance: list[dict[str, Any]]
    individual_explanations: list[dict[str, Any]] = field(default_factory=list)
    shap_values: np.ndarray | None = None


def generate_shap_explanations(
    trained_model: TrainedModel,
    df: pd.DataFrame,
    n_samples: int = 100,
    n_individual: int = 5,
) -> ShapExplanations:
    """Generate SHAP explanations for a trained model.

    Args:
        trained_model: The trained model to explain
        df: The full dataset
        n_samples: Number of samples to use for SHAP calculation
        n_individual: Number of individual explanations to generate

    Returns:
        ShapExplanations containing global and individual explanations
    """
    X, _ = prepare_features(df, fit_encoders=False, encoders=trained_model.encoders)
    feature_names = list(X.columns)

    test_idx = trained_model.test_indices
    X_test = X.iloc[test_idx]

    sample_size = min(n_samples, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    logger.info("Computing SHAP values for %d samples...", sample_size)

    if trained_model.model_type == "lightgbm":
        explainer = shap.TreeExplainer(trained_model.model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        # For logistic regression pipeline, use KernelExplainer for more robust SHAP computation
        background = shap.sample(X_sample, min(50, len(X_sample)))

        def predict_proba(X: np.ndarray) -> np.ndarray:
            return trained_model.model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_order = np.argsort(mean_abs_shap)[::-1]

    global_importance = []
    for idx in importance_order:
        global_importance.append(
            {
                "feature_name": feature_names[idx],
                "importance": float(mean_abs_shap[idx]),
                "rank": len(global_importance) + 1,
            }
        )

    individual_explanations = []
    df_test = df.iloc[test_idx]
    sample_indices = X_sample.index[:n_individual]

    for i, idx in enumerate(sample_indices):
        row_shap = shap_values[i]
        row_features = X_sample.loc[idx]

        top_features = []
        sorted_indices = np.argsort(np.abs(row_shap))[::-1][:5]
        for feat_idx in sorted_indices:
            top_features.append(
                {
                    "feature": feature_names[feat_idx],
                    "value": float(row_features.iloc[feat_idx]),
                    "shap_value": float(row_shap[feat_idx]),
                    "direction": "increases" if row_shap[feat_idx] > 0 else "decreases",
                }
            )

        original_idx = df_test.index.get_loc(idx) if idx in df_test.index else i
        member_info = {}
        if isinstance(original_idx, int) and original_idx < len(df_test):
            row = df_test.iloc[original_idx]
            member_info = {
                "member_id": row.get("member_id", "Unknown"),
                "care_gap_code": row.get("care_gap_code", "Unknown"),
                "actual_status": row.get("status", "Unknown"),
            }

        individual_explanations.append(
            {
                "sample_index": i,
                "member_info": member_info,
                "top_contributing_features": top_features,
                "predicted_probability": float(
                    trained_model.model.predict_proba(X_sample.loc[[idx]])[0, 1]
                ),
            }
        )

    logger.info(
        "Generated SHAP explanations: %d global features, %d individual examples",
        len(global_importance),
        len(individual_explanations),
    )

    return ShapExplanations(
        feature_names=feature_names,
        global_importance=global_importance,
        individual_explanations=individual_explanations,
        shap_values=shap_values,
    )


def format_global_importance_table(
    explanations: ShapExplanations,
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """Format global importance as a table for reporting."""
    return [
        {
            "rank": item["rank"],
            "feature_name": item["feature_name"],
            "importance": f"{item['importance']:.4f}",
        }
        for item in explanations.global_importance[:top_n]
    ]


def format_individual_explanation(
    explanation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Format an individual explanation as a table."""
    return [
        {
            "feature_name": feat["feature"],
            "value": f"{feat['value']:.4f}" if isinstance(feat["value"], float) else str(feat["value"]),
            "shap_value": f"{feat['shap_value']:+.4f}",
            "impact": f"{feat['direction']} closure probability",
        }
        for feat in explanation["top_contributing_features"]
    ]


def get_top_contributing_features_json(
    explanations: ShapExplanations,
    sample_idx: int = 0,
    top_n: int = 5,
) -> str:
    """Get top contributing features as JSON for der_MemberImpactScores."""
    import json

    if sample_idx >= len(explanations.individual_explanations):
        return "{}"

    explanation = explanations.individual_explanations[sample_idx]
    features = explanation["top_contributing_features"][:top_n]

    return json.dumps(
        [
            {
                "feature": f["feature"],
                "impact": f["shap_value"],
                "direction": f["direction"],
            }
            for f in features
        ]
    )
