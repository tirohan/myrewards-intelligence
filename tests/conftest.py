"""Shared test fixtures for milestone5 tests.

Uses synthetic DataFrames shaped like real outputs — no live DB calls required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_analytical_dataset() -> pd.DataFrame:
    """Synthetic dataset shaped like analytical_dataset.csv."""
    np.random.seed(42)
    n_samples = 100

    member_ids = [f"MOCK{i:05d}" for i in range(n_samples)]
    care_gaps = ["DIAB_A1C_TEST", "BH_POST_DISCHARGE_7D", "FUH_POST_DISCHARGE_30D", "BCS_MAMMOGRAPHY"]
    conditions = ["DIABETES", "BEHAVIORAL_HEALTH", "BEHAVIORAL_HEALTH", "Not condition-gated"]

    data = []
    for i in range(n_samples):
        gap_idx = i % len(care_gaps)
        is_incomm = i < 60

        data.append({
            "member_id": member_ids[i % 50],
            "care_gap_code": care_gaps[gap_idx],
            "condition_code": conditions[gap_idx],
            "status": np.random.choice(["Closed", "Open"], p=[0.4, 0.6] if is_incomm else [0.2, 0.8]),
            "label_source": "InComm-sourced" if is_incomm else "Research-computed (domain analysis)",
            "days_since_gap_opened": np.random.randint(30, 500),
            "prior_closure_rate_other_measures": np.random.uniform(0, 1) if i % 3 != 0 else np.nan,
            "qualifying_claim_count": np.random.randint(0, 5),
            "days_since_last_qualifying_claim": np.random.uniform(10, 200) if np.random.random() > 0.3 else np.nan,
            "SnapshotDateUtc": "2026-06-23 00:00:00",
            "PrimaryCondition": conditions[gap_idx],
            "Sex": np.random.choice(["M", "F"]),
            "AgeYears": np.random.randint(25, 75),
            "ConditionCount": np.random.randint(1, 5),
            "ClaimLineCount": np.random.randint(1, 20),
            "OtcTransactionLineCount": np.random.randint(0, 30),
            "BenefitTransactionCount": np.random.randint(0, 10),
            "TotalSpend": np.random.uniform(100, 2000),
            "HealthySpend": np.random.uniform(50, 500),
            "HealthySpendRatio": np.random.uniform(0.1, 0.5),
            "DaysSinceClinicalService": np.random.randint(10, 200),
            "DaysSinceOtcTransaction": np.random.randint(1, 60),
            "DaysSinceBenefitTransaction": np.random.randint(10, 150),
            "rewards_issued_count": np.random.randint(0, 5),
            "rewards_claimed_count": np.random.randint(0, 3),
            "redemption_rate": np.random.uniform(0, 1),
            "avg_days_issue_to_claim": np.random.uniform(1, 30) if np.random.random() > 0.3 else np.nan,
            "as_of_date": "2026-08-13",
        })

    return pd.DataFrame(data)


@pytest.fixture
def incomm_only_dataset(sample_analytical_dataset: pd.DataFrame) -> pd.DataFrame:
    """Dataset with only InComm-sourced rows."""
    return sample_analytical_dataset[
        sample_analytical_dataset["label_source"] == "InComm-sourced"
    ].copy()


@pytest.fixture
def small_dataset() -> pd.DataFrame:
    """Minimal dataset for quick tests."""
    return pd.DataFrame([
        {
            "member_id": "M1",
            "care_gap_code": "DIAB_A1C_TEST",
            "condition_code": "DIABETES",
            "status": "Closed",
            "label_source": "InComm-sourced",
            "days_since_gap_opened": 100,
            "prior_closure_rate_other_measures": 0.5,
            "qualifying_claim_count": 2,
            "days_since_last_qualifying_claim": 30.0,
            "SnapshotDateUtc": "2026-06-23",
            "PrimaryCondition": "DIABETES",
            "Sex": "F",
            "AgeYears": 45,
            "ConditionCount": 2,
            "ClaimLineCount": 5,
            "OtcTransactionLineCount": 10,
            "BenefitTransactionCount": 3,
            "TotalSpend": 500.0,
            "HealthySpend": 150.0,
            "HealthySpendRatio": 0.3,
            "DaysSinceClinicalService": 50,
            "DaysSinceOtcTransaction": 10,
            "DaysSinceBenefitTransaction": 30,
            "rewards_issued_count": 2,
            "rewards_claimed_count": 1,
            "redemption_rate": 0.5,
            "avg_days_issue_to_claim": 15.0,
            "as_of_date": "2026-08-13",
        },
        {
            "member_id": "M1",
            "care_gap_code": "BCS_MAMMOGRAPHY",
            "condition_code": "Not condition-gated",
            "status": "Open",
            "label_source": "InComm-sourced",
            "days_since_gap_opened": 200,
            "prior_closure_rate_other_measures": 1.0,
            "qualifying_claim_count": 0,
            "days_since_last_qualifying_claim": None,
            "SnapshotDateUtc": "2026-06-23",
            "PrimaryCondition": "DIABETES",
            "Sex": "F",
            "AgeYears": 45,
            "ConditionCount": 2,
            "ClaimLineCount": 5,
            "OtcTransactionLineCount": 10,
            "BenefitTransactionCount": 3,
            "TotalSpend": 500.0,
            "HealthySpend": 150.0,
            "HealthySpendRatio": 0.3,
            "DaysSinceClinicalService": 50,
            "DaysSinceOtcTransaction": 10,
            "DaysSinceBenefitTransaction": 30,
            "rewards_issued_count": 2,
            "rewards_claimed_count": 1,
            "redemption_rate": 0.5,
            "avg_days_issue_to_claim": 15.0,
            "as_of_date": "2026-08-13",
        },
        {
            "member_id": "M2",
            "care_gap_code": "DIAB_A1C_TEST",
            "condition_code": "DIABETES",
            "status": "Open",
            "label_source": "Research-computed (domain analysis)",
            "days_since_gap_opened": 150,
            "prior_closure_rate_other_measures": None,
            "qualifying_claim_count": 1,
            "days_since_last_qualifying_claim": 60.0,
            "SnapshotDateUtc": "2026-06-23",
            "PrimaryCondition": "DIABETES",
            "Sex": "M",
            "AgeYears": 55,
            "ConditionCount": 1,
            "ClaimLineCount": 3,
            "OtcTransactionLineCount": 5,
            "BenefitTransactionCount": 2,
            "TotalSpend": 300.0,
            "HealthySpend": 100.0,
            "HealthySpendRatio": 0.33,
            "DaysSinceClinicalService": 80,
            "DaysSinceOtcTransaction": 20,
            "DaysSinceBenefitTransaction": 50,
            "rewards_issued_count": 1,
            "rewards_claimed_count": 0,
            "redemption_rate": 0.0,
            "avg_days_issue_to_claim": None,
            "as_of_date": "2026-08-13",
        },
    ])
