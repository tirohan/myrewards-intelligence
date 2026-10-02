"""Contracts on the generated CSV interfaces (skipped when the artifact has not been generated)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from myrewards_intelligence.milestone6_reward_uplift.data import EVENT_COLUMNS
from myrewards_intelligence.milestone6_reward_uplift.domain_resolutions import EXCLUDE_CARE_GAPS
from myrewards_intelligence.milestone6_reward_uplift.evidence import TreatmentSource

MODELS = Path(__file__).resolve().parents[1] / "models"
MOCK = TreatmentSource.RESEARCH_GENERATED_LEDGER.value


def _load(name: str) -> pd.DataFrame:
    path = MODELS / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return pd.read_csv(path)


def test_events_ledger_shape_and_label():
    df = _load("incentive_rewards_events.csv")
    assert list(df.columns) == EVENT_COLUMNS
    assert set(df["source"]) == {MOCK}  # never tagged InComm-issued
    assert not df["care_gap_code"].isin(EXCLUDE_CARE_GAPS).any()
    assert not df.duplicated(["member_id", "care_gap_code"]).any()


@pytest.mark.parametrize("name", ["scored_uplift_track_r.csv", "m7_roi_scenarios.csv"])
def test_scored_outputs_carry_mock_basis(name):
    df = _load(name)
    assert set(df["TreatmentSourceBasis"]) == {MOCK}


def test_scores_are_finite_and_unique_per_gap():
    df = _load("scored_uplift_track_r.csv")
    assert df["UpliftScore"].notna().all()
    assert not df.duplicated(["MemberId", "CareGapCode"]).any()
