"""Target variable definition for Member Impact Scoring.

Per MILESTONE5_PLAN.md: binary is_closed = (status == "Closed") per
(member_id, care_gap_code, as_of_date).
"""

from __future__ import annotations

import pandas as pd


def create_target_variable(df: pd.DataFrame) -> pd.Series:
    """Create binary target variable from status column.

    Returns:
        Series with 1 for Closed, 0 for Open.
    """
    if "status" not in df.columns:
        raise ValueError("Dataset must contain 'status' column")

    target = (df["status"] == "Closed").astype(int)
    return target


def get_target_statistics(df: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Compute target variable statistics."""
    n_total = len(target)
    n_positive = int(target.sum())
    n_negative = n_total - n_positive

    return {
        "total_samples": n_total,
        "positive_samples": n_positive,
        "negative_samples": n_negative,
        "positive_rate": n_positive / n_total if n_total > 0 else 0.0,
        "negative_rate": n_negative / n_total if n_total > 0 else 0.0,
    }


def get_target_by_label_source(df: pd.DataFrame, target: pd.Series) -> dict[str, dict[str, float]]:
    """Compute target statistics broken out by label source."""
    if "label_source" not in df.columns:
        return {}

    results = {}
    for source in df["label_source"].unique():
        mask = df["label_source"] == source
        source_target = target[mask]
        n_total = len(source_target)
        n_positive = int(source_target.sum())

        results[source] = {
            "total_samples": n_total,
            "positive_samples": n_positive,
            "positive_rate": n_positive / n_total if n_total > 0 else 0.0,
        }

    return results


def get_target_by_care_gap(df: pd.DataFrame, target: pd.Series) -> dict[str, dict[str, float]]:
    """Compute target statistics broken out by care gap code."""
    if "care_gap_code" not in df.columns:
        return {}

    results = {}
    for code in sorted(df["care_gap_code"].unique()):
        mask = df["care_gap_code"] == code
        code_target = target[mask]
        n_total = len(code_target)
        n_positive = int(code_target.sum())

        results[code] = {
            "total_samples": n_total,
            "positive_samples": n_positive,
            "positive_rate": n_positive / n_total if n_total > 0 else 0.0,
        }

    return results
