"""Model evaluation: AUC, precision/recall, calibration, subgroup performance."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..core.evidence import format_label_source_composition
from .features import prepare_features
from .target import create_target_variable
from .train import TrainedModel

logger = logging.getLogger("myrewards_intelligence")


@dataclass
class EvaluationResults:
    """Container for model evaluation results."""

    model_basis: str
    label_source_composition: str
    auc_roc: float
    brier_score: float
    threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    accuracy_at_threshold: float
    roc_curve: dict[str, list[float]] = field(default_factory=dict)
    pr_curve: dict[str, list[float]] = field(default_factory=dict)
    calibration_curve: dict[str, list[float]] = field(default_factory=dict)
    subgroup_metrics: list[dict[str, Any]] = field(default_factory=list)


def evaluate_model(
    trained_model: TrainedModel,
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> EvaluationResults:
    """Evaluate a trained model on the test set.

    Args:
        trained_model: The trained model to evaluate
        df: The full dataset (will use test indices)
        threshold: Operating threshold for precision/recall

    Returns:
        EvaluationResults containing all metrics
    """
    target = create_target_variable(df)
    X, _ = prepare_features(
        df,
        fit_encoders=False,
        encoders=trained_model.encoders,
        exclude_features=trained_model.exclude_features,
    )

    test_idx = trained_model.test_indices
    X_test = X.iloc[test_idx]
    y_test = target.iloc[test_idx]

    # Pipeline/model handles NaN values internally
    y_prob = trained_model.model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    composition = trained_model.label_source_composition
    composition_str = format_label_source_composition(
        composition.get("incomm_sourced", 0),
        composition.get("research_computed", 0),
        composition.get("total", len(df)),
    )

    logger.info(
        "%s model evaluation: AUC=%.4f, Brier=%.4f, Precision@%.2f=%.4f, Recall@%.2f=%.4f",
        trained_model.model_basis.value,
        auc_roc,
        brier,
        threshold,
        precision,
        threshold,
        recall,
    )

    return EvaluationResults(
        model_basis=trained_model.model_basis.value,
        label_source_composition=composition_str,
        auc_roc=auc_roc,
        brier_score=brier,
        threshold=threshold,
        precision_at_threshold=precision,
        recall_at_threshold=recall,
        f1_at_threshold=f1,
        accuracy_at_threshold=accuracy,
        roc_curve={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        pr_curve={"precision": prec_curve.tolist(), "recall": rec_curve.tolist()},
        calibration_curve={"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
    )


def generate_subgroup_metrics(
    trained_model: TrainedModel,
    df: pd.DataFrame,
    subgroup_columns: list[str],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Generate performance metrics broken out by subgroup.

    Args:
        trained_model: The trained model to evaluate
        df: The full dataset (will use test indices)
        subgroup_columns: Columns to break out metrics by (e.g., care_gap_code, PrimaryCondition)
        threshold: Operating threshold for precision/recall

    Returns:
        List of dicts with subgroup metrics
    """
    target = create_target_variable(df)
    X, _ = prepare_features(
        df,
        fit_encoders=False,
        encoders=trained_model.encoders,
        exclude_features=trained_model.exclude_features,
    )

    test_idx = trained_model.test_indices
    X_test = X.iloc[test_idx]
    y_test = target.iloc[test_idx]
    df_test = df.iloc[test_idx]

    # Pipeline/model handles NaN values internally
    y_prob = trained_model.model.predict_proba(X_test)[:, 1]

    results = []

    for col in subgroup_columns:
        if col not in df_test.columns:
            logger.warning("Subgroup column %s not found in dataset", col)
            continue

        for group_value in sorted(df_test[col].unique()):
            mask = df_test[col] == group_value
            if mask.sum() < 10:
                continue

            y_test_group = y_test[mask]
            y_prob_group = y_prob[mask.values]
            y_pred_group = (y_prob_group >= threshold).astype(int)

            n_samples = int(mask.sum())
            n_positive = int(y_test_group.sum())
            base_rate = n_positive / n_samples if n_samples > 0 else 0.0

            try:
                auc_roc = roc_auc_score(y_test_group, y_prob_group)
            except ValueError:
                auc_roc = float("nan")

            precision = precision_score(y_test_group, y_pred_group, zero_division=0)
            recall = recall_score(y_test_group, y_pred_group, zero_division=0)

            results.append(
                {
                    "subgroup_column": col,
                    "subgroup_value": str(group_value),
                    "n_samples": n_samples,
                    "n_positive": n_positive,
                    "base_rate": base_rate,
                    "auc": auc_roc,
                    "precision": precision,
                    "recall": recall,
                }
            )

    return results


def compare_models(
    results_a: EvaluationResults,
    results_b: EvaluationResults,
) -> dict[str, Any]:
    """Compare evaluation results between Model A and Model B."""
    return {
        "model_a": {
            "basis": results_a.model_basis,
            "label_composition": results_a.label_source_composition,
            "auc": results_a.auc_roc,
            "brier": results_a.brier_score,
            "precision": results_a.precision_at_threshold,
            "recall": results_a.recall_at_threshold,
        },
        "model_b": {
            "basis": results_b.model_basis,
            "label_composition": results_b.label_source_composition,
            "auc": results_b.auc_roc,
            "brier": results_b.brier_score,
            "precision": results_b.precision_at_threshold,
            "recall": results_b.recall_at_threshold,
        },
        "auc_difference": results_b.auc_roc - results_a.auc_roc,
        "note": (
            "Model B's metrics reflect a mixed-label dataset. "
            "Improvements may reflect additional data volume rather than "
            "true signal, since most additional labels are research-computed."
        ),
    }


def format_metrics_table(results: EvaluationResults) -> list[dict[str, Any]]:
    """Format evaluation results as a table for reporting."""
    return [
        {"metric": "AUC-ROC", "value": f"{results.auc_roc:.4f}"},
        {"metric": "Brier Score", "value": f"{results.brier_score:.4f}"},
        {"metric": f"Precision @ {results.threshold}", "value": f"{results.precision_at_threshold:.4f}"},
        {"metric": f"Recall @ {results.threshold}", "value": f"{results.recall_at_threshold:.4f}"},
        {"metric": f"F1 Score @ {results.threshold}", "value": f"{results.f1_at_threshold:.4f}"},
        {"metric": f"Accuracy @ {results.threshold}", "value": f"{results.accuracy_at_threshold:.4f}"},
        {"metric": "Label Source Composition", "value": results.label_source_composition},
    ]
