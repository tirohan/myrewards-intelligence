"""Follow-up analyses answering InComm technical review questions.

Runs Model A (InComm-sourced) diagnostics:
1. Missingness / imputation audit (plus drop-vs-impute sensitivity)
2. Never-rewarded / cold-start subgroup and reward-feature ablation
3. Train/test care-gap mix and grouped+stratified split
4. Threshold vs outreach-volume curve

Training variants are repeated across multiple random seeds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..core.config import Settings
from ..core.evidence import ModelBasis
from .dataset import REQUIRED_COLUMNS
from .evaluate import evaluate_model
from .features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, REWARD_FEATURES
from .target import create_target_variable
from .train import TrainedModel, train_model

logger = logging.getLogger("myrewards_intelligence")

DEFAULT_SEEDS = [42, 123, 456, 789, 2026]
DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
MISSINGNESS_COLUMNS = [
    "days_since_gap_opened",
    "prior_closure_rate_other_measures",
    *REWARD_FEATURES,
]
ACCEPTABLE_AUC_BAND = (0.70, 0.80)


def _mean_std(values: list[float]) -> dict[str, float | None]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
    }


def _safe_auc(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def summarize_dataset_contract(df: pd.DataFrame) -> dict[str, Any]:
    """Clarify column counts vs the 44-column question."""
    model_features = [c for c in FEATURE_COLUMNS + CATEGORICAL_COLUMNS if c in df.columns]
    return {
        "analytical_dataset_columns": int(df.shape[1]),
        "required_columns_in_loader": len(REQUIRED_COLUMNS),
        "model_feature_count": len(model_features),
        "model_features": model_features,
        "note": (
            "The analytical dataset used for Milestone 5 has "
            f"{df.shape[1]} columns (loader requires {len(REQUIRED_COLUMNS)}). "
            f"The fitted model uses {len(model_features)} features after dropping "
            "identifiers, the label, and leaking qualifying-claim fields. "
            "There is no 44-column contract in this codebase."
        ),
    }


def summarize_class_balance(df: pd.DataFrame) -> dict[str, Any]:
    """Model A closure rate and imbalance handling."""
    n = len(df)
    n_closed = int((df["status"] == "Closed").sum()) if "status" in df.columns else 0
    n_open = int((df["status"] == "Open").sum()) if "status" in df.columns else 0
    rate = n_closed / n if n else 0.0
    return {
        "n_rows": n,
        "n_members": int(df["member_id"].nunique()) if "member_id" in df.columns else 0,
        "n_closed": n_closed,
        "n_open": n_open,
        "closure_rate": rate,
        "imbalance": "moderate (not rare-event)" if 0.3 <= rate <= 0.7 else "imbalanced",
        "handling": "class_weight='balanced' in the logistic pipeline; no downsampling or synthetic labels",
    }


def audit_missingness(df: pd.DataFrame) -> dict[str, Any]:
    """Missingness by feature and member; imputation policy."""
    n_rows = len(df)
    n_members = int(df["member_id"].nunique()) if "member_id" in df.columns else 0
    features: list[dict[str, Any]] = []

    for col in MISSINGNESS_COLUMNS:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        member_missing = 0
        if "member_id" in df.columns and n_missing > 0:
            member_missing = int(df.groupby("member_id")[col].apply(lambda s: s.isna().any()).sum())
        features.append(
            {
                "feature": col,
                "n_rows_missing": n_missing,
                "pct_rows_missing": n_missing / n_rows if n_rows else 0.0,
                "n_members_with_any_missing": member_missing,
                "pct_members_with_any_missing": member_missing / n_members if n_members else 0.0,
            }
        )

    return {
        "n_rows": n_rows,
        "n_members": n_members,
        "policy": (
            "Rows are not dropped. No sentinel values. Numeric missings are "
            "median-imputed inside the training pipeline (SimpleImputer)."
        ),
        "features": features,
    }


def summarize_split_composition(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_label: str,
) -> dict[str, Any]:
    """Train vs test mix by care_gap_code under a grouped split."""
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    rows: list[dict[str, Any]] = []
    if "care_gap_code" not in df.columns:
        return {"split_label": split_label, "by_care_gap": [], "n_train": len(train), "n_test": len(test)}

    for gap in sorted(df["care_gap_code"].astype(str).unique()):
        n_train = int((train["care_gap_code"].astype(str) == gap).sum())
        n_test = int((test["care_gap_code"].astype(str) == gap).sum())
        total = n_train + n_test
        rows.append(
            {
                "care_gap_code": gap,
                "n_train": n_train,
                "n_test": n_test,
                "pct_in_test": n_test / total if total else 0.0,
            }
        )

    return {
        "split_label": split_label,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_train_members": int(train["member_id"].nunique()) if "member_id" in train.columns else 0,
        "n_test_members": int(test["member_id"].nunique()) if "member_id" in test.columns else 0,
        "by_care_gap": rows,
    }


def _predict_test(trained: TrainedModel, df: pd.DataFrame) -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    from .features import prepare_features

    target = create_target_variable(df)
    X, _ = prepare_features(
        df,
        fit_encoders=False,
        encoders=trained.encoders,
        exclude_features=trained.exclude_features,
    )
    test_idx = trained.test_indices
    y_test = target.iloc[test_idx]
    X_test = X.iloc[test_idx]
    y_prob = trained.model.predict_proba(X_test)[:, 1]
    df_test = df.iloc[test_idx]
    return y_test, y_prob, df_test


def cold_start_subgroup(trained: TrainedModel, df: pd.DataFrame) -> dict[str, Any]:
    """Test-set performance for never-rewarded vs previously rewarded members."""
    y_test, y_prob, df_test = _predict_test(trained, df)
    if "rewards_issued_count" not in df_test.columns:
        return {"available": False}

    never = df_test["rewards_issued_count"].fillna(0) == 0
    groups = {
        "never_rewarded": never,
        "has_reward_history": ~never,
    }
    out: dict[str, Any] = {"available": True, "groups": []}
    for name, mask in groups.items():
        n = int(mask.sum())
        if n == 0:
            out["groups"].append({"group": name, "n_test": 0})
            continue
        y = y_test[mask]
        p = y_prob[mask.values]
        out["groups"].append(
            {
                "group": name,
                "n_test": n,
                "closure_rate": float(y.mean()),
                "mean_predicted_probability": float(np.mean(p)),
                "auc": _safe_auc(y, p),
            }
        )
    return out


def threshold_volume_curve(
    trained: TrainedModel,
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    score_tiers: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Precision, recall, F1, and flagged volume at several operating points."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    y_test, y_prob, _ = _predict_test(trained, df)
    n = len(y_test)
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        n_flagged = int(pred.sum())
        rows.append(
            {
                "threshold": t,
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "n_flagged": n_flagged,
                "pct_flagged": n_flagged / n if n else 0.0,
            }
        )

    tiers = []
    if score_tiers:
        high = score_tiers.get("high", 0.7)
        medium = score_tiers.get("medium", 0.4)
        for name, mask in (
            ("High", y_prob >= high),
            ("Medium", (y_prob >= medium) & (y_prob < high)),
            ("Low", y_prob < medium),
        ):
            n_tier = int(mask.sum())
            tiers.append(
                {
                    "tier": name,
                    "n_test": n_tier,
                    "pct_test": n_tier / n if n else 0.0,
                    "closure_rate": float(y_test.values[mask].mean()) if n_tier else None,
                }
            )

    return {
        "n_test": n,
        "reporting_threshold_note": (
            "0.50 is a comparable reporting point, not an operational cut. "
            "Choose a threshold from outreach capacity (care managers vs campaigns)."
        ),
        "curve": rows,
        "tiers": tiers,
    }


def _fit_and_score(
    df: pd.DataFrame,
    seed: int,
    settings: Settings,
    exclude_features: list[str] | None = None,
    split_mode: str = "grouped",
) -> dict[str, Any]:
    trained = train_model(
        df,
        ModelBasis.MODEL_A_INCOMM,
        settings=settings,
        model_type="lightgbm",
        random_state=seed,
        exclude_features=exclude_features,
        split_mode=split_mode,
    )
    ev = evaluate_model(trained, df, threshold=settings.milestone5.operating_threshold)
    return {
        "seed": seed,
        "auc": ev.auc_roc,
        "brier": ev.brier_score,
        "precision": ev.precision_at_threshold,
        "recall": ev.recall_at_threshold,
        "f1": ev.f1_at_threshold,
        "n_train": int(len(trained.train_indices)),
        "n_test": int(len(trained.test_indices)),
        "trained": trained,
        "evaluation": ev,
    }


def run_seeded_variant(
    df: pd.DataFrame,
    settings: Settings,
    seeds: list[int],
    *,
    label: str,
    exclude_features: list[str] | None = None,
    split_mode: str = "grouped",
    dropna_column: str | None = None,
) -> dict[str, Any]:
    """Train/evaluate a Model A variant across seeds."""
    work = df
    dropped = 0
    if dropna_column and dropna_column in df.columns:
        work = df.dropna(subset=[dropna_column]).reset_index(drop=True)
        dropped = int(len(df) - len(work))

    per_seed: list[dict[str, Any]] = []
    last_trained: TrainedModel | None = None
    for seed in seeds:
        logger.info("Diagnostic variant '%s' seed=%s", label, seed)
        result = _fit_and_score(
            work,
            seed,
            settings,
            exclude_features=exclude_features,
            split_mode=split_mode,
        )
        last_trained = result.pop("trained")
        result.pop("evaluation", None)
        per_seed.append(result)

    aucs = [r["auc"] for r in per_seed]
    return {
        "label": label,
        "split_mode": split_mode,
        "exclude_features": list(exclude_features or []),
        "dropna_column": dropna_column,
        "n_rows_dropped": dropped,
        "n_rows_used": int(len(work)),
        "per_seed": per_seed,
        "auc_summary": _mean_std(aucs),
        "last_trained": last_trained,
        "working_df": work,
    }


def run_followup_diagnostics(
    df_model_a: pd.DataFrame,
    settings: Settings | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run all four follow-up analyses on Model A."""
    if settings is None:
        from ..core.config import load_config

        settings = load_config()

    seeds = list(seeds or getattr(settings.milestone5, "diagnostic_seeds", DEFAULT_SEEDS) or DEFAULT_SEEDS)
    logger.info("Running follow-up diagnostics on %d Model A rows, seeds=%s", len(df_model_a), seeds)

    contract = summarize_dataset_contract(df_model_a)
    balance = summarize_class_balance(df_model_a)
    missingness = audit_missingness(df_model_a)

    baseline = run_seeded_variant(
        df_model_a,
        settings,
        seeds,
        label="grouped_impute_all_features",
        split_mode="grouped",
    )
    stratified = run_seeded_variant(
        df_model_a,
        settings,
        seeds,
        label="grouped_stratified_by_care_gap",
        split_mode="grouped_stratified",
    )
    ablation = run_seeded_variant(
        df_model_a,
        settings,
        seeds,
        label="grouped_without_reward_features",
        exclude_features=list(REWARD_FEATURES),
        split_mode="grouped",
    )
    dropna = run_seeded_variant(
        df_model_a,
        settings,
        seeds,
        label="grouped_drop_missing_prior_closure_rate",
        split_mode="grouped",
        dropna_column="prior_closure_rate_other_measures",
    )

    baseline_trained = baseline["last_trained"]
    assert baseline_trained is not None
    split_grouped = summarize_split_composition(
        df_model_a,
        baseline_trained.train_indices,
        baseline_trained.test_indices,
        split_label="grouped (seed=last)",
    )
    strat_trained = stratified["last_trained"]
    assert strat_trained is not None
    split_stratified = summarize_split_composition(
        df_model_a,
        strat_trained.train_indices,
        strat_trained.test_indices,
        split_label="grouped_stratified (seed=last)",
    )

    cold_start = cold_start_subgroup(baseline_trained, df_model_a)
    tiers = {
        "high": settings.milestone5.score_tiers.high,
        "medium": settings.milestone5.score_tiers.medium,
    }
    thresholds = threshold_volume_curve(baseline_trained, df_model_a, score_tiers=tiers)

    auc_a_mean = baseline["auc_summary"]["mean"]
    auc_ablation_mean = ablation["auc_summary"]["mean"]
    delta = None
    if auc_a_mean is not None and auc_ablation_mean is not None:
        delta = auc_a_mean - auc_ablation_mean

    return {
        "seeds": seeds,
        "contract": contract,
        "class_balance": balance,
        "missingness": missingness,
        "variants": {
            "baseline": {k: v for k, v in baseline.items() if k not in ("last_trained", "working_df")},
            "stratified": {k: v for k, v in stratified.items() if k not in ("last_trained", "working_df")},
            "reward_ablation": {k: v for k, v in ablation.items() if k not in ("last_trained", "working_df")},
            "dropna_prior_closure": {k: v for k, v in dropna.items() if k not in ("last_trained", "working_df")},
        },
        "split_composition": {
            "grouped": split_grouped,
            "grouped_stratified": split_stratified,
        },
        "cold_start": cold_start,
        "thresholds": thresholds,
        "acceptable_auc_band": {
            "low": ACCEPTABLE_AUC_BAND[0],
            "high": ACCEPTABLE_AUC_BAND[1],
            "interpretation": (
                "For care-gap ranking we treat ~0.70–0.80 as useful for prioritization. "
                "Values above 0.90 warrant a time-safety check on reward features."
            ),
        },
        "reward_ablation_auc_drop": delta,
    }
