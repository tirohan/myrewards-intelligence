"""Tests for Milestone 5 follow-up diagnostics."""

from __future__ import annotations

import pandas as pd

from myrewards_intelligence.core.evidence import ModelBasis
from myrewards_intelligence.milestone5_member_impact_scoring.diagnostics import (
    audit_missingness,
    run_followup_diagnostics,
    summarize_class_balance,
    summarize_dataset_contract,
    summarize_split_composition,
)
from myrewards_intelligence.milestone5_member_impact_scoring.reports import (
    build_followup_report,
    write_followup_markdown,
)
from myrewards_intelligence.milestone5_member_impact_scoring.train import train_model


def test_class_balance_and_missingness(incomm_only_dataset: pd.DataFrame) -> None:
    balance = summarize_class_balance(incomm_only_dataset)
    assert balance["n_rows"] == len(incomm_only_dataset)
    assert 0.0 <= balance["closure_rate"] <= 1.0

    missing = audit_missingness(incomm_only_dataset)
    names = {row["feature"] for row in missing["features"]}
    assert "days_since_gap_opened" in names
    assert "prior_closure_rate_other_measures" in names


def test_dataset_contract_not_44_columns(sample_analytical_dataset: pd.DataFrame) -> None:
    contract = summarize_dataset_contract(sample_analytical_dataset)
    assert contract["analytical_dataset_columns"] == sample_analytical_dataset.shape[1]
    assert contract["analytical_dataset_columns"] != 44


def test_split_composition(incomm_only_dataset: pd.DataFrame) -> None:
    trained = train_model(incomm_only_dataset, ModelBasis.MODEL_A_INCOMM, model_type="logistic")
    comp = summarize_split_composition(
        incomm_only_dataset,
        trained.train_indices,
        trained.test_indices,
        split_label="grouped",
    )
    assert comp["n_train"] + comp["n_test"] == len(incomm_only_dataset)
    assert comp["by_care_gap"]


def test_followup_diagnostics_multi_seed(incomm_only_dataset: pd.DataFrame, tmp_path) -> None:
    diagnostics = run_followup_diagnostics(incomm_only_dataset, seeds=[42, 123])
    assert diagnostics["seeds"] == [42, 123]
    baseline = diagnostics["variants"]["baseline"]
    assert len(baseline["per_seed"]) == 2
    assert baseline["auc_summary"]["mean"] is not None
    assert "reward_ablation" in diagnostics["variants"]
    assert diagnostics["thresholds"]["curve"]

    doc = build_followup_report(diagnostics)
    assert doc.paragraphs
    md_path = write_followup_markdown(diagnostics, tmp_path / "followup.md")
    text = open(md_path, encoding="utf-8").read()
    assert "Closure rate" in text
