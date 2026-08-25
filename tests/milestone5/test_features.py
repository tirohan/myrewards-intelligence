"""Tests for feature preparation and Model A/B splitting."""

import pandas as pd
import pytest

from myrewards_intelligence.milestone5_member_impact_scoring.features import (
    get_groups,
    get_label_source_composition,
    prepare_features,
    split_by_label_source,
)


def test_split_by_label_source_model_a_only_incomm(
    sample_analytical_dataset: pd.DataFrame,
) -> None:
    """Verify Model A contains only InComm-sourced rows."""
    splits = split_by_label_source(sample_analytical_dataset)
    model_a = splits["model_a"]

    assert all(model_a["label_source"] == "InComm-sourced")


def test_split_by_label_source_model_b_contains_all(
    sample_analytical_dataset: pd.DataFrame,
) -> None:
    """Verify Model B contains all rows."""
    splits = split_by_label_source(sample_analytical_dataset)
    model_b = splits["model_b"]

    assert len(model_b) == len(sample_analytical_dataset)


def test_split_by_label_source_requires_column() -> None:
    """Verify error when label_source column missing."""
    df = pd.DataFrame({"member_id": ["M1"], "status": ["Closed"]})
    with pytest.raises(ValueError, match="label_source"):
        split_by_label_source(df)


def test_get_label_source_composition(sample_analytical_dataset: pd.DataFrame) -> None:
    """Verify label source composition is computed correctly."""
    comp = get_label_source_composition(sample_analytical_dataset)

    assert comp["total"] == len(sample_analytical_dataset)
    assert comp["incomm_sourced"] > 0
    assert comp["research_computed"] > 0
    assert comp["incomm_sourced"] + comp["research_computed"] == comp["total"]


def test_prepare_features_drops_identifiers(small_dataset: pd.DataFrame) -> None:
    """Verify identifier columns are dropped."""
    X, _ = prepare_features(small_dataset)

    assert "member_id" not in X.columns
    assert "care_gap_code" not in X.columns
    assert "as_of_date" not in X.columns


def test_prepare_features_drops_target_and_label(small_dataset: pd.DataFrame) -> None:
    """Verify status and label_source columns are dropped."""
    X, _ = prepare_features(small_dataset)

    assert "status" not in X.columns
    assert "label_source" not in X.columns


def test_prepare_features_encodes_categoricals(small_dataset: pd.DataFrame) -> None:
    """Verify categorical columns are encoded."""
    X, encoders = prepare_features(small_dataset)

    assert "condition_code" in X.columns or "condition_code" in encoders
    assert X["Sex"].dtype in ("int64", "int32", "float64")


def test_prepare_features_returns_encoders(small_dataset: pd.DataFrame) -> None:
    """Verify encoders are returned for reuse."""
    _, encoders = prepare_features(small_dataset)

    assert len(encoders) > 0
    assert "Sex" in encoders or "condition_code" in encoders


def test_prepare_features_reuses_encoders(small_dataset: pd.DataFrame) -> None:
    """Verify encoders can be reused for inference."""
    _, encoders = prepare_features(small_dataset, fit_encoders=True)
    X_reused, _ = prepare_features(small_dataset, fit_encoders=False, encoders=encoders)

    assert X_reused is not None


def test_get_groups_returns_member_ids(small_dataset: pd.DataFrame) -> None:
    """Verify groups are member_id values."""
    groups = get_groups(small_dataset)

    assert len(groups) == len(small_dataset)
    assert set(groups) == {"M1", "M2"}


def test_get_groups_requires_member_id() -> None:
    """Verify error when member_id column missing."""
    df = pd.DataFrame({"status": ["Closed"]})
    with pytest.raises(ValueError, match="member_id"):
        get_groups(df)
