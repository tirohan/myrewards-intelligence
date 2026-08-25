"""Tests for target variable definition."""

import pandas as pd
import pytest

from myrewards_intelligence.milestone5_member_impact_scoring.target import (
    create_target_variable,
    get_target_by_care_gap,
    get_target_by_label_source,
    get_target_statistics,
)


def test_create_target_variable_closed_is_1(small_dataset: pd.DataFrame) -> None:
    """Verify Closed status maps to 1."""
    target = create_target_variable(small_dataset)
    closed_mask = small_dataset["status"] == "Closed"
    assert all(target[closed_mask] == 1)


def test_create_target_variable_open_is_0(small_dataset: pd.DataFrame) -> None:
    """Verify Open status maps to 0."""
    target = create_target_variable(small_dataset)
    open_mask = small_dataset["status"] == "Open"
    assert all(target[open_mask] == 0)


def test_create_target_variable_requires_status_column() -> None:
    """Verify error when status column missing."""
    df = pd.DataFrame({"member_id": ["M1"], "other_col": [1]})
    with pytest.raises(ValueError, match="status"):
        create_target_variable(df)


def test_get_target_statistics(small_dataset: pd.DataFrame) -> None:
    """Verify target statistics are computed correctly."""
    target = create_target_variable(small_dataset)
    stats = get_target_statistics(small_dataset, target)

    assert stats["total_samples"] == 3
    assert stats["positive_samples"] == 1
    assert stats["negative_samples"] == 2
    assert stats["positive_rate"] == pytest.approx(1 / 3)


def test_get_target_by_label_source(small_dataset: pd.DataFrame) -> None:
    """Verify target statistics by label source."""
    target = create_target_variable(small_dataset)
    by_source = get_target_by_label_source(small_dataset, target)

    assert "InComm-sourced" in by_source
    assert by_source["InComm-sourced"]["total_samples"] == 2
    assert by_source["InComm-sourced"]["positive_samples"] == 1


def test_get_target_by_care_gap(small_dataset: pd.DataFrame) -> None:
    """Verify target statistics by care gap code."""
    target = create_target_variable(small_dataset)
    by_gap = get_target_by_care_gap(small_dataset, target)

    assert "DIAB_A1C_TEST" in by_gap
    assert by_gap["DIAB_A1C_TEST"]["total_samples"] == 2
