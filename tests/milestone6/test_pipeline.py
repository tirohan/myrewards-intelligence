"""Milestone 6 behavior tests.

Production change that would fail these tests: dropping TreatmentSourceBasis,
importing Milestone 5, fitting CATE when the control arm has no outcome
variation, or reporting elasticity when issued amount is constant within a gap.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from myrewards_intelligence.core.evidence import NOT_AVAILABLE, LabelSource
from myrewards_intelligence.milestone6_reward_uplift.covariates import (
    LEAKAGE_FEATURES,
    REWARD_FEATURES,
    prepare_covariates,
)
from myrewards_intelligence.milestone6_reward_uplift.derived_layer import (
    generate_ddl_proposal,
    generate_scored_output,
)
from myrewards_intelligence.milestone6_reward_uplift.elasticity import assess_elasticity
from myrewards_intelligence.milestone6_reward_uplift.evaluate_uplift import (
    evaluate_uplift_multi_seed,
    placebo_auuc,
    qini_auuc,
)
from myrewards_intelligence.milestone6_reward_uplift.evidence import (
    AssignmentMechanism,
    TreatmentGrain,
    TreatmentSource,
)
from myrewards_intelligence.milestone6_reward_uplift.overlap import assess_overlap
from myrewards_intelligence.milestone6_reward_uplift.targeting import build_targeting_table
from myrewards_intelligence.milestone6_reward_uplift.treatment import attach_treatment
from myrewards_intelligence.milestone6_reward_uplift.uplift import aipw_ate, fit_uplift_models

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "src" / "myrewards_intelligence"


def _identified_frame(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """Synthetic rows with overlap and a known positive treatment effect."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    age = rng.integers(30, 70, n)
    logit_t = -0.2 + 0.6 * x
    t = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit_t))).astype(int)
    logit_y = -0.8 + 0.4 * x + 1.1 * t
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit_y))).astype(int)
    members = [f"MOCK{i:05d}" for i in range(n)]
    gaps = np.array(["DIAB_A1C_TEST", "HTN_PCP_FOLLOWUP", "COPD_SPIROMETRY"])[np.arange(n) % 3]
    return pd.DataFrame(
        {
            "member_id": members,
            "care_gap_code": gaps,
            "condition_code": "DIABETES",
            "status": np.where(y == 1, "Closed", "Open"),
            "label_source": LabelSource.INCOMM_SOURCED.value,
            "days_since_gap_opened": rng.integers(30, 400, n),
            "prior_closure_rate_other_measures": rng.uniform(0, 1, n),
            "qualifying_claim_count": y * 2,
            "days_since_last_qualifying_claim": np.where(y == 1, 20.0, np.nan),
            "SnapshotDateUtc": "2026-06-23",
            "PrimaryCondition": "DIABETES",
            "Sex": np.where(rng.uniform(0, 1, n) > 0.5, "F", "M"),
            "AgeYears": age,
            "ConditionCount": 1,
            "ClaimLineCount": rng.integers(1, 12, n),
            "OtcTransactionLineCount": rng.integers(0, 8, n),
            "BenefitTransactionCount": rng.integers(0, 4, n),
            "TotalSpend": rng.uniform(100, 800, n),
            "HealthySpend": rng.uniform(20, 200, n),
            "HealthySpendRatio": rng.uniform(0.1, 0.5, n),
            "DaysSinceClinicalService": rng.integers(5, 180, n),
            "DaysSinceOtcTransaction": rng.integers(1, 40, n),
            "DaysSinceBenefitTransaction": rng.integers(5, 90, n),
            "rewards_issued_count": t,
            "rewards_claimed_count": (t * (rng.uniform(0, 1, n) < 0.6)).astype(int),
            "redemption_rate": t.astype(float) * 0.6,
            "avg_days_issue_to_claim": np.where(t == 1, 14.0, np.nan),
            "as_of_date": "2026-08-13",
            "x_latent": x,
        }
    )


def _no_overlap_frame() -> pd.DataFrame:
    """Matches the live extract: untreated InComm rows never close."""
    df = _identified_frame(n=80, seed=7)
    untreated = df["rewards_issued_count"] == 0
    df.loc[untreated, "status"] = "Open"
    df.loc[untreated, "qualifying_claim_count"] = 0
    return df


def test_package_does_not_import_milestone5() -> None:
    root = PACKAGE_DIR / "milestone6_reward_uplift"
    cli = PACKAGE_DIR / "cli" / "run_milestone6.py"
    files = list(root.glob("*.py")) + ([cli] if cli.exists() else [])
    assert files, "milestone6 package should exist"
    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any("milestone5" in name for name in names):
                offenders.append(str(path))
    assert not offenders, f"M6 must not import M5: {offenders}"


def test_treatment_is_tagged_research_simulated(incomm_only_dataset: pd.DataFrame) -> None:
    treated = attach_treatment(incomm_only_dataset)
    assert (treated["treatment_source_basis"] == TreatmentSource.RESEARCH_SIMULATED.value).all()
    assert (treated["treatment_grain"] == TreatmentGrain.MEMBER_BROADCAST.value).all()
    assert treated["t_issued"].isin([0, 1]).all()
    assert "qualifying_claim_count" not in prepare_covariates(treated)[0].columns


def test_covariates_drop_leakage_and_reward_features(incomm_only_dataset: pd.DataFrame) -> None:
    X, _ = prepare_covariates(incomm_only_dataset, exclude_reward_features=True)
    for col in LEAKAGE_FEATURES + REWARD_FEATURES:
        assert col not in X.columns


def test_overlap_rejects_zero_control_closures() -> None:
    df = attach_treatment(_no_overlap_frame())
    X, _ = prepare_covariates(df, exclude_reward_features=True)
    result = assess_overlap(X, df["t_issued"].to_numpy(), df["y"].to_numpy())
    assert result.cate_identified is False
    assert result.control_closure_rate == 0.0


def test_t_learner_recovers_positive_lift_when_overlap_exists() -> None:
    df = attach_treatment(_identified_frame())
    X, _ = prepare_covariates(df, exclude_reward_features=True)
    overlap = assess_overlap(X, df["t_issued"].to_numpy(), df["y"].to_numpy())
    assert overlap.cate_identified is True
    fitted = fit_uplift_models(df, X, random_state=42)
    ate = aipw_ate(df["y"].to_numpy(), df["t_issued"].to_numpy(), fitted.mu0, fitted.mu1, fitted.propensity)
    assert ate["ate"] > 0
    auuc = qini_auuc(df["y"].to_numpy(), df["t_issued"].to_numpy(), fitted.tau)
    placebo = placebo_auuc(df["y"].to_numpy(), df["t_issued"].to_numpy(), fitted.tau, random_state=42)
    assert auuc > placebo


def test_multi_seed_evaluation_records_identification_gate() -> None:
    identified = evaluate_uplift_multi_seed(_identified_frame(), seeds=[42, 123])
    blocked = evaluate_uplift_multi_seed(_no_overlap_frame(), seeds=[42])
    assert identified["cate_identified"] is True
    assert blocked["cate_identified"] is False
    assert blocked["headline_claim"] == "NOT_IDENTIFIED"


def test_elasticity_not_available_when_amount_constant_within_gap() -> None:
    df = attach_treatment(_identified_frame())
    result = assess_elasticity(df)
    assert result["status"] == NOT_AVAILABLE
    assert result["within_gap_amount_varies"] is False


def test_targeting_uses_control_outcome_not_reward_features() -> None:
    df = attach_treatment(_identified_frame())
    X, _ = prepare_covariates(df, exclude_reward_features=True)
    fitted = fit_uplift_models(df, X, random_state=42)
    ranked = build_targeting_table(df, fitted, cate_identified=True)
    assert set(ranked["recommended_band"]).issubset({"Treat", "Hold", "Suppress"})
    assert "Research-simulated" in ranked["treatment_source_basis"].unique()
    assert ranked["control_outcome_score"].between(0, 1).all()


def test_ddl_and_scored_output_carry_treatment_source() -> None:
    df = attach_treatment(_identified_frame(n=60))
    X, _ = prepare_covariates(df, exclude_reward_features=True)
    fitted = fit_uplift_models(df, X, random_state=42)
    ddl = generate_ddl_proposal()
    assert "TreatmentSourceBasis" in ddl
    assert "der_MemberRewardUpliftScores" in ddl
    scored = generate_scored_output(df, fitted, cate_identified=True)
    assert (scored["TreatmentSourceBasis"] == TreatmentSource.RESEARCH_SIMULATED.value).all()
    assert (scored["LabelSourceBasis"] == LabelSource.INCOMM_SOURCED.value).all()


def test_assignment_mechanism_is_documented() -> None:
    assert AssignmentMechanism.ML_REWARD_ELIGIBLE.value.startswith("HEALTH_ACTION_REWARD")
    assert "Bernoulli" in AssignmentMechanism.CAPACITY_MOCK.value


def test_track_r_mock_treatment_independent_of_closed() -> None:
    from myrewards_intelligence.core.config import Milestone6Config
    from myrewards_intelligence.milestone6_reward_uplift.incentive_rewards_mock import (
        generate_incentive_rewards_mock,
        observation_outcome,
    )

    raw = _identified_frame(n=300, seed=42)
    mock = generate_incentive_rewards_mock(raw, Milestone6Config(), random_state=42)
    treated = attach_treatment(
        raw,
        treatment_source=TreatmentSource.RESEARCH_GENERATED_LEDGER,
        events=mock.events,
        eligibility=mock.eligibility,
        window_days=90,
    )
    assert treated["treatment_source_basis"].eq(TreatmentSource.RESEARCH_GENERATED_LEDGER.value).all()
    assert treated["assignment_mechanism"].eq(AssignmentMechanism.CAPACITY_MOCK.value).all()
    assert treated["treatment_grain"].eq(TreatmentGrain.GAP_LEVEL.value).all()
    assert not (treated["t_issued"] == treated["y"]).all()
    y_window = observation_outcome(mock.eligibility, treated, window_days=90)
    assert (treated["y"].to_numpy() == y_window.to_numpy()).all()
    y_closed = (treated["status"] == "Closed").astype(int)
    assert not (treated["y"].to_numpy() == y_closed.to_numpy()).all()
    with pytest.raises(ValueError, match="window_days"):
        attach_treatment(
            raw,
            treatment_source=TreatmentSource.RESEARCH_GENERATED_LEDGER,
            events=mock.events,
        )
    t = treated["t_issued"].to_numpy()
    y = treated["y"].to_numpy()
    assert t.sum() > 0 and (t == 0).sum() > 0
    assert y[t == 1].min() != y[t == 1].max()
    assert y[t == 0].min() != y[t == 0].max()
    evaluated = evaluate_uplift_multi_seed(treated, seeds=[42])
    assert TreatmentSource.INCOMM_ISSUED.value not in treated["treatment_source_basis"].unique()
    if evaluated["cate_identified"]:
        assert evaluated["headline_claim"] == "CATE"


def test_track_r_mock_never_tagged_incomm_issued(tmp_path: Path) -> None:
    from myrewards_intelligence.core.config import Milestone6Config
    from myrewards_intelligence.milestone6_reward_uplift.data import load_issuance_ledger
    from myrewards_intelligence.milestone6_reward_uplift.incentive_rewards_mock import (
        generate_incentive_rewards_mock,
        write_mock_ledger,
    )

    raw = _identified_frame(n=80, seed=3)
    mock = generate_incentive_rewards_mock(raw, Milestone6Config(), random_state=3)
    dest = write_mock_ledger(mock, tmp_path / "incentive_rewards_mock.csv")
    loaded = load_issuance_ledger(dest)
    assert loaded["source"].eq(TreatmentSource.RESEARCH_GENERATED_LEDGER.value).all()
    assert TreatmentSource.INCOMM_ISSUED.value not in loaded["source"].unique()


def test_track_r_empty_ledger_raises() -> None:
    from myrewards_intelligence.milestone6_reward_uplift.data import (
        LedgerNotAvailableError,
        load_issuance_ledger,
    )

    with pytest.raises(LedgerNotAvailableError):
        load_issuance_ledger("")


def test_track_r_ledger_schema(tmp_path: Path) -> None:
    from myrewards_intelligence.milestone6_reward_uplift.data import (
        EVENT_COLUMNS,
        load_issuance_ledger,
    )

    path = tmp_path / "ledger.csv"
    pd.DataFrame({col: [] for col in EVENT_COLUMNS}).to_csv(path, index=False)
    events = load_issuance_ledger(path)
    assert list(events.columns) == EVENT_COLUMNS


def test_s_learner_and_attributions_when_identified() -> None:
    df = attach_treatment(_identified_frame())
    X, _ = prepare_covariates(df, exclude_reward_features=True)
    fitted = fit_uplift_models(df, X, random_state=42)
    assert fitted.tau_s_learner.shape == fitted.tau.shape
    assert not np.allclose(fitted.tau_s_learner, 0)
    parsed = json.loads(fitted.row_attributions[0])
    assert parsed and "feature" in parsed[0]
    scored = generate_scored_output(df, fitted, cate_identified=True)
    assert scored["TopContributingFeaturesJson"].iloc[0] != "[]"


def test_ablation_export_and_blocked_mu0(tmp_path: Path) -> None:
    from myrewards_intelligence.milestone6_reward_uplift.ablation import (
        export_ablation_scores,
        train_ablation_scorer,
    )
    from myrewards_intelligence.milestone6_reward_uplift.uplift import scores_when_not_identified

    df = attach_treatment(_no_overlap_frame())
    X, _ = prepare_covariates(df, exclude_reward_features=True)
    scorer = train_ablation_scorer(X, df["y"].to_numpy())
    parsed = json.loads(scorer.row_attributions[0])
    assert parsed and parsed[0]["feature"]
    dest = export_ablation_scores(df, scorer, tmp_path / "scored_model_a_ablation.csv")
    saved = pd.read_csv(dest)
    assert saved["excludes_reward_features"].all()
    fitted = scores_when_not_identified(
        n=len(df),
        feature_names=list(X.columns),
        propensity=np.full(len(df), 0.5),
        treated_rate=0.4,
        ablation_mu0=scorer.scores,
        row_attributions=scorer.row_attributions,
    )
    ranked = build_targeting_table(df, fitted, cate_identified=False, ablation_scores=scorer.scores)
    assert ranked["control_outcome_score"].max() > 0
    assert (fitted.mu0 == scorer.scores).all()


def test_care_gap_descriptives_flag_untreated_empty() -> None:
    from myrewards_intelligence.milestone6_reward_uplift.evaluate_uplift import (
        care_gap_descriptives,
    )

    df = attach_treatment(_identified_frame())
    diab = df["care_gap_code"] == "DIAB_A1C_TEST"
    df.loc[diab, "t_issued"] = 1
    rows = care_gap_descriptives(df)
    diab_row = next(row for row in rows if row["care_gap_code"] == "DIAB_A1C_TEST")
    assert diab_row["untreated_empty"] is True
    blocked = evaluate_uplift_multi_seed(_no_overlap_frame(), seeds=[42])
    assert blocked["by_care_gap"]
    assert "untreated_empty" in blocked["by_care_gap"][0]


def test_volume_curve_and_shareables(tmp_path: Path) -> None:
    from myrewards_intelligence.milestone6_reward_uplift.reports import copy_shareables
    from myrewards_intelligence.milestone6_reward_uplift.targeting import volume_curve

    curve = volume_curve(np.array([0.2, 0.08, 0.01, -0.02]), [0.0, 0.05, 0.10])
    assert list(curve["n_flagged"]) == [3, 2, 1]
    src = tmp_path / "Milestone 6 Reward Uplift Model v1.docx"
    src.write_bytes(b"PK")
    copied = copy_shareables([src], tmp_path / "milestone6_shareables")
    assert Path(copied[0]).exists()


def test_milestone6_config_is_loaded() -> None:
    from myrewards_intelligence.core.config import load_config

    settings = load_config()
    assert settings.milestone6.diagnostic_seeds == [42, 123, 456, 789, 2026]
    assert settings.milestone6.split_mode == "grouped_stratified"
    assert settings.milestone6.observation_window_days == 365
    assert settings.milestone6.shareable_dir == "milestone6_shareables"
    assert settings.milestone6.treatment_grain == "gap_level"
    assert settings.milestone6.track == "R"
    assert settings.milestone6.generate_incentive_rewards_mock is True
    assert settings.milestone6.issuance_ledger_path == "models/incentive_rewards_mock.csv"


def test_grouped_split_mode_is_recorded() -> None:
    result = evaluate_uplift_multi_seed(_identified_frame(), seeds=[42], split_mode="grouped_stratified")
    assert result["split_mode"] == "grouped_stratified"
    assert result["cate_identified"] is True


def test_gap_level_assignment_equals_closed_status() -> None:
    from myrewards_intelligence.milestone6_reward_uplift.data import reconstruct_assignment_events
    from myrewards_intelligence.milestone6_reward_uplift.evaluate_uplift import grain_contrast

    raw = _identified_frame()
    events = reconstruct_assignment_events(raw)
    treated = attach_treatment(raw, events=events)
    assert treated["treatment_grain"].eq(TreatmentGrain.GAP_LEVEL.value).all()
    assert (treated["t_issued"] == treated["y"]).all()
    blocked = evaluate_uplift_multi_seed(treated, seeds=[42])
    assert blocked["cate_identified"] is False
    assert any("Treatment equals outcome" in r for r in blocked["identification_reasons"])
    diab = next(row for row in blocked["by_care_gap"] if row["care_gap_code"] == "DIAB_A1C_TEST")
    assert diab["untreated_empty"] is False
    contrast = grain_contrast(raw)
    assert contrast["incentive_rewards_feed"] == "NOT_IMPLEMENTED"
    assert contrast["structural_reverse_causation"] is True


def test_member_broadcast_can_empty_an_open_gap_arm() -> None:
    """A member-level reward on one gap must not be read as diabetes untreated-empty."""
    from myrewards_intelligence.milestone6_reward_uplift.data import (
        events_from_analytical_aggregates,
        reconstruct_assignment_events,
    )

    raw = _identified_frame(n=90, seed=1)
    raw.loc[raw["care_gap_code"] == "DIAB_A1C_TEST", "status"] = "Open"
    raw.loc[raw["care_gap_code"] == "DIAB_A1C_TEST", "rewards_issued_count"] = 1
    broadcast = attach_treatment(raw, events=events_from_analytical_aggregates(raw))
    gap = attach_treatment(raw, events=reconstruct_assignment_events(raw))
    b_diab = broadcast.loc[broadcast["care_gap_code"] == "DIAB_A1C_TEST"]
    g_diab = gap.loc[gap["care_gap_code"] == "DIAB_A1C_TEST"]
    assert int(b_diab["t_issued"].sum()) == len(b_diab)
    assert int(g_diab["t_issued"].sum()) == 0


def test_targeting_never_treats_without_distinguishable_lift() -> None:
    df = attach_treatment(_identified_frame())
    X, _ = prepare_covariates(df, exclude_reward_features=True)
    fitted = fit_uplift_models(df, X, random_state=42)
    ranked = build_targeting_table(df, fitted, cate_identified=True, lift_distinguishable=False)
    assert "Treat" not in set(ranked["recommended_band"])
