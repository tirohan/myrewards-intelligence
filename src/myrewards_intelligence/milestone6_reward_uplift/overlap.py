"""Propensity overlap and identification gate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_ARM_N = 50
MIN_ESS_RATIO = 0.25


@dataclass
class OverlapResult:
    """Diagnostics for whether CATE is identified on this extract."""

    cate_identified: bool
    n_treated: int
    n_control: int
    treated_closure_rate: float
    control_closure_rate: float
    ess: float
    ess_ratio: float
    propensity: np.ndarray
    common_support_mask: np.ndarray
    reasons: list[str] = field(default_factory=list)


def _propensity_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def assess_overlap(
    X: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    *,
    min_arm_n: int = MIN_ARM_N,
    min_ess_ratio: float = MIN_ESS_RATIO,
) -> OverlapResult:
    """Fit e(x) and decide whether CATE fitting is allowed."""
    t = np.asarray(t).astype(int)
    y = np.asarray(y).astype(int)
    n_treated = int(t.sum())
    n_control = int((t == 0).sum())
    treated_rate = float(y[t == 1].mean()) if n_treated else float("nan")
    control_rate = float(y[t == 0].mean()) if n_control else float("nan")
    reasons: list[str] = []

    propensity = np.full(len(t), 0.5)
    if n_treated > 0 and n_control > 0 and len(np.unique(t)) == 2:
        model = _propensity_pipeline()
        model.fit(X, t)
        propensity = np.clip(model.predict_proba(X)[:, 1], 0.01, 0.99)
    else:
        reasons.append("Need both treated and control rows to fit propensity.")

    w = np.where(t == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))
    ess = float((w.sum() ** 2) / np.square(w).sum()) if w.size else 0.0
    ess_ratio = ess / len(t) if len(t) else 0.0
    common = (propensity >= 0.1) & (propensity <= 0.9)

    if n_treated < min_arm_n:
        reasons.append(f"Treated n={n_treated} < {min_arm_n}.")
    if n_control < min_arm_n:
        reasons.append(f"Control n={n_control} < {min_arm_n}.")
    if n_control and (control_rate == 0.0 or control_rate == 1.0):
        reasons.append(
            f"Control arm has no outcome variation (closure rate={control_rate:.3f})."
        )
    if n_treated and (treated_rate == 0.0 or treated_rate == 1.0):
        reasons.append(
            f"Treated arm has no outcome variation (closure rate={treated_rate:.3f})."
        )
    if ess_ratio < min_ess_ratio:
        reasons.append(f"ESS ratio {ess_ratio:.3f} < {min_ess_ratio}.")
    if len(t) and float(np.mean(t == y)) >= 0.999:
        reasons.append(
            "Treatment equals outcome (InComm assignment: Closed ⇔ IsRewardEligible ⇔ "
            "HEALTH_ACTION_REWARD case). Incremental effect is not identified."
        )

    hard = [
        r
        for r in reasons
        if "outcome variation" in r
        or "Treated n=" in r
        or "Control n=" in r
        or "both treated" in r
        or "Treatment equals outcome" in r
    ]
    identified = not hard
    return OverlapResult(
        cate_identified=identified,
        n_treated=n_treated,
        n_control=n_control,
        treated_closure_rate=treated_rate,
        control_closure_rate=control_rate,
        ess=ess,
        ess_ratio=ess_ratio,
        propensity=propensity,
        common_support_mask=common,
        reasons=reasons,
    )
