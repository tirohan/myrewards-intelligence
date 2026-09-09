"""Pre-treatment covariates for uplift. Independent of Milestone 5 modules."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("myrewards_intelligence")

LEAKAGE_FEATURES = [
    "qualifying_claim_count",
    "days_since_last_qualifying_claim",
]

REWARD_FEATURES = [
    "rewards_issued_count",
    "rewards_claimed_count",
    "redemption_rate",
    "avg_days_issue_to_claim",
    "t_issued",
    "t_redeemed",
    "issued_amount",
    "y",
]

IDENTIFIER_COLUMNS = ["member_id", "care_gap_code", "as_of_date", "SnapshotDateUtc"]

CATEGORICAL_COLUMNS = ["condition_code", "PrimaryCondition", "Sex"]

COVARIATE_COLUMNS = [
    "days_since_gap_opened",
    "prior_closure_rate_other_measures",
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
]


def prepare_covariates(
    df: pd.DataFrame,
    *,
    exclude_reward_features: bool = True,
    fit_encoders: bool = True,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Return numeric X with leakage and (by default) reward fields removed."""
    if encoders is None:
        encoders = {}

    forbidden = set(LEAKAGE_FEATURES + IDENTIFIER_COLUMNS)
    if exclude_reward_features:
        forbidden.update(REWARD_FEATURES)

    allowed = [c for c in COVARIATE_COLUMNS + CATEGORICAL_COLUMNS if c in df.columns and c not in forbidden]
    feature_df = df[allowed].copy()

    for col in CATEGORICAL_COLUMNS:
        if col not in feature_df.columns:
            continue
        series = feature_df[col].fillna("Unknown").astype(str)
        if fit_encoders:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(series)
            encoders[col] = le
        else:
            if col not in encoders:
                raise ValueError(f"No encoder provided for column {col}")
            known = set(encoders[col].classes_)
            fallback = "Unknown" if "Unknown" in known else str(encoders[col].classes_[0])
            series = series.where(series.isin(known), fallback)
            feature_df[col] = encoders[col].transform(series)

    drop_obj = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    if drop_obj:
        logger.warning("Dropping non-numeric covariates: %s", drop_obj)
        feature_df = feature_df.drop(columns=drop_obj)

    return feature_df, encoders
