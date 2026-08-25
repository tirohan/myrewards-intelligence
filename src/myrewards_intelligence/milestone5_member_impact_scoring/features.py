"""Feature preparation and Model A / Model B row filtering.

Per MILESTONE5_PLAN.md:
- Model A (primary): InComm-sourced labels only (11,669 rows / 6,668 members)
- Model B (expanded): combined dataset (53,839 rows)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..core.evidence import LabelSource, ModelBasis

logger = logging.getLogger("myrewards_intelligence")

IDENTIFIER_COLUMNS = ["member_id", "care_gap_code", "as_of_date", "SnapshotDateUtc"]

CATEGORICAL_COLUMNS = ["condition_code", "PrimaryCondition", "Sex"]

DROP_COLUMNS = [
    "status",
    "label_source",
]

# Features available at prediction time (EXCLUDING leakage features)
# REMOVED: qualifying_claim_count - This IS the outcome (qualifying claim = gap closed)
# REMOVED: days_since_last_qualifying_claim - Derived from outcome
FEATURE_COLUMNS = [
    "days_since_gap_opened",
    "prior_closure_rate_other_measures",
    # "qualifying_claim_count",           # LEAKAGE: encodes outcome directly (corr=0.97)
    # "days_since_last_qualifying_claim", # LEAKAGE: derived from outcome
    "AgeYears",
    "ConditionCount",
    "ClaimLineCount",
    "OtcTransactionLineCount",
    "BenefitTransactionCount",
    "TotalSpend",
    "HealthySpend",
    "HealthySpendRatio",
    "DaysSinceClinicalService",
    "DaysSinceOtcTransaction",
    "DaysSinceBenefitTransaction",
    "rewards_issued_count",
    "rewards_claimed_count",
    "redemption_rate",
    "avg_days_issue_to_claim",
]


def split_by_label_source(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split dataset into Model A (InComm-sourced) and Model B (all) subsets.

    Returns:
        Dictionary with keys 'model_a' and 'model_b' containing the respective DataFrames.
    """
    if "label_source" not in df.columns:
        raise ValueError("Dataset must contain 'label_source' column")

    incomm_mask = df["label_source"] == LabelSource.INCOMM_SOURCED.value
    model_a_df = df[incomm_mask].copy()
    model_b_df = df.copy()

    logger.info(
        "Split dataset: Model A = %d rows (InComm-sourced), Model B = %d rows (all)",
        len(model_a_df),
        len(model_b_df),
    )

    return {
        "model_a": model_a_df,
        "model_b": model_b_df,
    }


def get_label_source_composition(df: pd.DataFrame) -> dict[str, Any]:
    """Get the label source composition for a dataset."""
    if "label_source" not in df.columns:
        return {"total": len(df)}

    counts = df["label_source"].value_counts().to_dict()
    total = len(df)

    return {
        "total": total,
        "incomm_sourced": counts.get(LabelSource.INCOMM_SOURCED.value, 0),
        "research_computed": sum(
            v for k, v in counts.items() if k != LabelSource.INCOMM_SOURCED.value
        ),
        "composition": counts,
    }


def prepare_features(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Prepare features for model training.

    Args:
        df: Input DataFrame with all columns
        fit_encoders: Whether to fit new encoders (True for training, False for inference)
        encoders: Pre-fitted encoders to use (required if fit_encoders=False)

    Returns:
        Tuple of (feature DataFrame, fitted encoders dict)

    Note:
        Only columns in FEATURE_COLUMNS + CATEGORICAL_COLUMNS are included.
        This explicitly excludes leakage features like qualifying_claim_count.
    """
    if encoders is None:
        encoders = {}

    # Select only allowed features (explicit inclusion, not exclusion)
    allowed_columns = set(FEATURE_COLUMNS + CATEGORICAL_COLUMNS)
    available_columns = [col for col in allowed_columns if col in df.columns]
    feature_df = df[available_columns].copy()

    # Encode categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col not in feature_df.columns:
            continue

        if fit_encoders:
            le = LabelEncoder()
            feature_df[col] = feature_df[col].fillna("Unknown")
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))
            encoders[col] = le
        else:
            if col not in encoders:
                raise ValueError(f"No encoder provided for column {col}")
            le = encoders[col]
            feature_df[col] = feature_df[col].fillna("Unknown")
            feature_df[col] = le.transform(feature_df[col].astype(str))

    # Drop any remaining object columns
    for col in feature_df.columns:
        if feature_df[col].dtype == "object":
            logger.warning("Dropping non-numeric column: %s", col)
            feature_df = feature_df.drop(columns=[col])

    return feature_df, encoders


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Get the list of feature names after preparation."""
    feature_df, _ = prepare_features(df.head(1), fit_encoders=True)
    return list(feature_df.columns)


def get_groups(df: pd.DataFrame) -> np.ndarray:
    """Get member_id groups for grouped train/test split."""
    if "member_id" not in df.columns:
        raise ValueError("Dataset must contain 'member_id' column for grouped split")
    return df["member_id"].values


def describe_model_basis(model_name: str) -> tuple[ModelBasis, str]:
    """Get the model basis and description for a model variant."""
    if model_name.lower() in ("a", "model_a", "model a"):
        return ModelBasis.MODEL_A_INCOMM, ModelBasis.MODEL_A_INCOMM.description
    elif model_name.lower() in ("b", "model_b", "model b"):
        return ModelBasis.MODEL_B_EXPANDED, ModelBasis.MODEL_B_EXPANDED.description
    else:
        raise ValueError(f"Unknown model name: {model_name}")
