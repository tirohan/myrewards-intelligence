"""Qini / AUUC, placebo shuffle, and multi-seed grouped evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .covariates import prepare_covariates
from .data import reconstruct_assignment_events
from .overlap import assess_overlap
from .treatment import attach_treatment
from .uplift import aipw_ate, apply_uplift_models, fit_uplift_models

DEFAULT_SEEDS = [42, 123, 456, 789, 2026]


def qini_auuc(y: np.ndarray, t: np.ndarray, tau: np.ndarray) -> float:
    """Area under the Qini curve (treated minus control cumulative outcome rates)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    tau = np.asarray(tau, dtype=float)
    n_t = t.sum()
    n_c = (1.0 - t).sum()
    if n_t == 0 or n_c == 0 or len(y) < 2:
        return float("nan")
    order = np.argsort(-tau, kind="mergesort")
    y = y[order]
    t = t[order]
    cum_t = np.cumsum(y * t) / n_t
    cum_c = np.cumsum(y * (1.0 - t)) / n_c
    curve = cum_t - cum_c
    area = getattr(np, "trapezoid", np.trapz)
    return float(area(curve, dx=1.0 / (len(curve) - 1)))


def placebo_auuc(
    y: np.ndarray,
    t: np.ndarray,
    tau: np.ndarray,
    *,
    random_state: int = 42,
    n_shuffles: int = 20,
) -> float:
    """Mean AUUC after shuffling T (should collapse if τ is not noise-fitting Y only)."""
    rng = np.random.default_rng(random_state)
    scores = []
    t = np.asarray(t)
    for _ in range(n_shuffles):
        shuffled = rng.permutation(t)
        scores.append(qini_auuc(y, shuffled, tau))
    return float(np.nanmean(scores))


def care_gap_descriptives(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Treated vs control closure by gap. Diabetes untreated-empty is expected on this extract."""
    rows: list[dict[str, Any]] = []
    for gap, part in df.groupby("care_gap_code"):
        t = part["t_issued"].to_numpy().astype(int)
        y = part["y"].to_numpy().astype(int)
        n_t = int(t.sum())
        n_c = int((t == 0).sum())
        treated_rate = float(y[t == 1].mean()) if n_t else None
        control_rate = float(y[t == 0].mean()) if n_c else None
        contrast = None
        if treated_rate is not None and control_rate is not None:
            contrast = treated_rate - control_rate
        rows.append(
            {
                "care_gap_code": str(gap),
                "n": int(len(part)),
                "n_treated": n_t,
                "n_control": n_c,
                "treated_closure": treated_rate,
                "control_closure": control_rate,
                "descriptive_contrast": contrast,
                "untreated_empty": n_c == 0,
                "control_no_variation": bool(n_c > 0 and y[t == 0].min() == y[t == 0].max())
                if n_c
                else True,
            }
        )
    return rows


ASSIGNMENT_AUTOPSY_NOTES = [
    "IncentiveRewards 3rd-party feed is not implemented (InComm 2026-07-20 Q4). "
    "Track R uses a research-generated IncentiveRewards-shaped ledger on real "
    "members/gaps/amounts. It is not InComm-issued; Milestone 7 may use it only as labeled research input.",
    "Live QA: IsRewardEligible iff Status='Closed' (0 violations). HAR cases / sim "
    "rows are generated for those closed gaps (CaseSubtype = care_gap_code). "
    "Gap-level Track S T equals Y on catalog gaps.",
    "Member-broadcast T (M4 lifetime sim count) copies any reward onto every gap "
    "for that member. Diabetes untreated-empty is that artifact, not a missing arm.",
    "ISS-008: clinical claims ServiceFrom <= 2026-06-23; HAR CreatedAtUtc >= 2026-06-25. "
    "Case time is after every claim on this extract — reverse causation is structural "
    "for HAR. The Track R mock reconstructs outreach lag from gap tenure.",
    "Track R mock assignment is Bernoulli among gaps still open at t0. Historical "
    "closure was not caused by mock T; identified CATE here is a pipeline/"
    "negative-control result, not partner-facing lift.",
]


def grain_contrast(df: pd.DataFrame) -> dict[str, Any]:
    """Member-broadcast vs gap-level assignment — the diabetes empty-arm explanation."""
    labels = df.drop(columns=["t_issued", "y"], errors="ignore")
    broadcast = attach_treatment(labels)
    gap = attach_treatment(labels, events=reconstruct_assignment_events(labels))
    return {
        "member_broadcast": care_gap_descriptives(broadcast),
        "gap_level_assignment": care_gap_descriptives(gap),
        "notes": list(ASSIGNMENT_AUTOPSY_NOTES),
        "incentive_rewards_feed": "NOT_IMPLEMENTED",
        "structural_reverse_causation": True,
    }


def grouped_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    split_mode: str = "grouped_stratified",
) -> tuple[np.ndarray, np.ndarray]:
    """Grouped member split; stratified by care_gap_code when requested."""
    groups = df["member_id"].to_numpy()
    if split_mode == "time_aware" and "potential_issued_at" in df.columns:
        # train on members whose earliest outreach is oldest, test on the newest (exposes drift)
        first = pd.to_datetime(df.groupby("member_id")["potential_issued_at"].transform("min"))
        is_test = (first > first.quantile(1 - test_size)).to_numpy()
        if is_test.any() and (~is_test).any():
            return np.flatnonzero(~is_test), np.flatnonzero(is_test)
    if split_mode == "grouped_stratified" and "care_gap_code" in df.columns:
        n_splits = max(2, int(round(1 / test_size)))
        y = df["care_gap_code"].astype(str).to_numpy()
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            return next(splitter.split(df, y=y, groups=groups))
        except ValueError:
            pass
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(df, groups=groups))


def evaluate_uplift_multi_seed(
    df: pd.DataFrame,
    seeds: list[int] | None = None,
    test_size: float = 0.2,
    split_mode: str = "grouped_stratified",
) -> dict[str, Any]:
    """Grouped multi-seed evaluation with an identification gate on the full set."""
    seeds = list(seeds or DEFAULT_SEEDS)
    has_treatment = "t_issued" in df.columns and "y" in df.columns
    treated = df.copy() if has_treatment else attach_treatment(df)
    X, _ = prepare_covariates(treated, exclude_reward_features=True)
    y = treated["y"].to_numpy()
    t = treated["t_issued"].to_numpy()
    overlap = assess_overlap(X, t, y)
    by_gap = care_gap_descriptives(treated)
    autopsy = grain_contrast(treated)

    result: dict[str, Any] = {
        "seeds": seeds,
        "split_mode": split_mode,
        "n_rows": int(len(treated)),
        "n_members": int(treated["member_id"].nunique()),
        "n_treated": overlap.n_treated,
        "n_control": overlap.n_control,
        "treated_closure_rate": overlap.treated_closure_rate,
        "control_closure_rate": overlap.control_closure_rate,
        "ess": overlap.ess,
        "ess_ratio": overlap.ess_ratio,
        "cate_identified": overlap.cate_identified,
        "identification_reasons": overlap.reasons,
        "treatment_source_basis": treated["treatment_source_basis"].iloc[0],
        "treatment_grain": treated["treatment_grain"].iloc[0] if "treatment_grain" in treated.columns else "",
        "headline_claim": "CATE" if overlap.cate_identified else "NOT_IDENTIFIED",
        "by_care_gap": by_gap,
        "grain_contrast": autopsy,
        "per_seed": [],
    }
    if not overlap.cate_identified:
        return result

    for seed in seeds:
        train_idx, test_idx = grouped_split(
            treated, test_size=test_size, random_state=seed, split_mode=split_mode
        )
        train = treated.iloc[train_idx].reset_index(drop=True)
        test = treated.iloc[test_idx].reset_index(drop=True)
        X_train, encoders = prepare_covariates(train, exclude_reward_features=True)
        X_test, _ = prepare_covariates(
            test, exclude_reward_features=True, fit_encoders=False, encoders=encoders
        )
        fitted = fit_uplift_models(train, X_train, random_state=seed)
        scored_test = apply_uplift_models(fitted, X_test)
        ate = aipw_ate(
            test["y"].to_numpy(),
            test["t_issued"].to_numpy(),
            scored_test.mu0,
            scored_test.mu1,
            scored_test.propensity,
        )
        auuc = qini_auuc(test["y"].to_numpy(), test["t_issued"].to_numpy(), scored_test.tau)
        placebo = placebo_auuc(
            test["y"].to_numpy(), test["t_issued"].to_numpy(), scored_test.tau, random_state=seed
        )
        result["per_seed"].append(
            {
                "seed": seed,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "ate": ate["ate"],
                "ate_se": ate["se"],
                "auuc": auuc,
                "placebo_auuc": placebo,
                "train_feature_count": len(fitted.feature_names),
            }
        )
    if result["per_seed"]:
        result["ate_mean"] = float(np.mean([row["ate"] for row in result["per_seed"]]))
        # seeds share data, so the mean per-seed SE is the honest (conservative) uncertainty
        result["ate_se"] = float(np.mean([row["ate_se"] for row in result["per_seed"]]))
        result["auuc_mean"] = float(np.mean([row["auuc"] for row in result["per_seed"]]))
        result["placebo_auuc_mean"] = float(
            np.mean([row["placebo_auuc"] for row in result["per_seed"]])
        )
    return result
