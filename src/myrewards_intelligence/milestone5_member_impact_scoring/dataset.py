"""Load and validate the Milestone 4 analytical dataset."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..core.config import Settings, get_dataset_path
from ..core.evidence import LabelSource

logger = logging.getLogger("myrewards_intelligence")

REQUIRED_COLUMNS = [
    "member_id",
    "care_gap_code",
    "condition_code",
    "status",
    "label_source",
    "days_since_gap_opened",
    "prior_closure_rate_other_measures",
    "qualifying_claim_count",
    "days_since_last_qualifying_claim",
    "SnapshotDateUtc",
    "PrimaryCondition",
    "Sex",
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
    "as_of_date",
]


def load_analytical_dataset(settings: Settings | None = None) -> pd.DataFrame:
    """Load the analytical dataset from the configured path."""
    if settings is None:
        from ..core.config import load_config

        settings = load_config()

    path = get_dataset_path(settings)
    logger.info("Loading analytical dataset from: %s", path)

    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    return df


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the analytical dataset and return validation results."""
    results: dict[str, Any] = {
        "valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "issues": [],
    }

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        results["valid"] = False
        results["issues"].append(f"Missing required columns: {sorted(missing_cols)}")

    if "label_source" in df.columns:
        label_sources = df["label_source"].unique().tolist()
        results["label_sources"] = label_sources
        unknown_sources = set(label_sources) - set(LabelSource.values())
        if unknown_sources:
            results["issues"].append(f"Unknown label sources: {unknown_sources}")

    if "status" in df.columns:
        statuses = df["status"].unique().tolist()
        results["statuses"] = statuses
        if not set(statuses).issubset({"Open", "Closed"}):
            results["issues"].append(f"Unexpected status values: {statuses}")

    if "member_id" in df.columns and "care_gap_code" in df.columns:
        dupes = df.duplicated(subset=["member_id", "care_gap_code"]).sum()
        results["duplicate_pairs"] = int(dupes)
        if dupes > 0:
            results["issues"].append(f"{dupes} duplicate (member_id, care_gap_code) pairs")

    if "member_id" in df.columns:
        results["unique_members"] = df["member_id"].nunique()

    if "care_gap_code" in df.columns:
        results["care_gap_codes"] = sorted(df["care_gap_code"].unique().tolist())

    results["label_source_counts"] = (
        df["label_source"].value_counts().to_dict() if "label_source" in df.columns else {}
    )
    results["status_counts"] = (
        df["status"].value_counts().to_dict() if "status" in df.columns else {}
    )

    if results["issues"]:
        results["valid"] = False

    return results


def get_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a summary of the dataset for reporting."""
    summary: dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "unique_members": df["member_id"].nunique() if "member_id" in df.columns else 0,
        "care_gaps": sorted(df["care_gap_code"].unique().tolist()) if "care_gap_code" in df.columns else [],
    }

    if "label_source" in df.columns:
        incomm_mask = df["label_source"] == LabelSource.INCOMM_SOURCED.value
        summary["incomm_sourced_rows"] = int(incomm_mask.sum())
        summary["research_computed_rows"] = int((~incomm_mask).sum())
        summary["incomm_sourced_members"] = (
            df.loc[incomm_mask, "member_id"].nunique() if "member_id" in df.columns else 0
        )

    if "status" in df.columns:
        summary["closed_count"] = int((df["status"] == "Closed").sum())
        summary["open_count"] = int((df["status"] == "Open").sum())
        summary["overall_closure_rate"] = (
            summary["closed_count"] / len(df) if len(df) > 0 else 0.0
        )

    if "label_source" in df.columns and "status" in df.columns:
        incomm_mask = df["label_source"] == LabelSource.INCOMM_SOURCED.value
        incomm_df = df[incomm_mask]
        summary["incomm_closure_rate"] = (
            (incomm_df["status"] == "Closed").mean() if len(incomm_df) > 0 else 0.0
        )
        research_df = df[~incomm_mask]
        summary["research_closure_rate"] = (
            (research_df["status"] == "Closed").mean() if len(research_df) > 0 else 0.0
        )

    return summary
