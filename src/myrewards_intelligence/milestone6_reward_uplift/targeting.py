"""Uplift targeting quadrants and volume curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .uplift import FittedUplift

TAU_TREAT = 0.05
HIGH_RISK_TOP_SHARE = 0.10  # relative cut: an absolute P(close) bar is meaningless on a ~2% outcome
EXPLORATORY = "Exploratory (no distinguishable population lift)"
DEFAULT_CUTPOINTS = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]


def _quadrant(
    tau: float,
    mu0: float,
    *,
    cate_identified: bool,
    tau_treat: float,
    prop_high: float,
    lift_distinguishable: bool = True,
) -> str:
    high_p = mu0 >= prop_high
    high_t = tau >= tau_treat
    if cate_identified and not lift_distinguishable and high_t:
        return EXPLORATORY  # a ranking without a real average effect is not a spend list
    if not cate_identified:
        return "Sure thing (do not pay)" if high_p else "Lost cause"
    if high_t and not high_p:
        return "Persuadable"
    if high_t and high_p:
        return "Sure thing"
    if not high_t and not high_p:
        return "Lost cause"
    return "Do not disturb"


def _band(quadrant: str) -> str:
    if quadrant == "Persuadable":
        return "Treat"
    if quadrant in {"Sure thing", "Sure thing (do not pay)"}:
        return "Hold"
    return "Suppress"


def volume_curve(
    scores: np.ndarray,
    cutpoints: list[float] | None = None,
) -> pd.DataFrame:
    """Share of rows at or above each score threshold."""
    scores = np.asarray(scores, dtype=float)
    cuts = list(cutpoints or DEFAULT_CUTPOINTS)
    n = max(len(scores), 1)
    rows = []
    for thr in cuts:
        flagged = scores >= thr
        rows.append(
            {
                "threshold": float(thr),
                "n_flagged": int(flagged.sum()),
                "flagged_share": float(flagged.mean()) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def quadrant_counts(ranked: pd.DataFrame) -> pd.DataFrame:
    counts = (
        ranked["quadrant"].value_counts().rename_axis("quadrant").reset_index(name="n")
        if "quadrant" in ranked.columns
        else pd.DataFrame(columns=["quadrant", "n"])
    )
    return counts


def build_targeting_table(
    df: pd.DataFrame,
    fitted: FittedUplift,
    *,
    cate_identified: bool,
    ablation_scores: np.ndarray | None = None,
    tau_treat: float = TAU_TREAT,
    high_risk_top_share: float = HIGH_RISK_TOP_SHARE,
    lift_distinguishable: bool = True,
) -> pd.DataFrame:
    """Rank (member, gap) by τ using ablation P(close), not leaky reward Model A."""
    out = df.copy()
    out["uplift_score"] = fitted.tau
    mu0 = np.asarray(ablation_scores if ablation_scores is not None else fitted.mu0, dtype=float)
    out["control_outcome_score"] = np.clip(mu0, 0, 1)
    out["treated_outcome_score"] = np.clip(fitted.mu1, 0, 1)
    out["propensity_to_treat"] = fitted.propensity
    # strictly above the median guard keeps a constant score from flagging everyone
    prop_high = max(float(np.quantile(out["control_outcome_score"], 1 - high_risk_top_share)), 1e-12)
    out["quadrant"] = [
        _quadrant(
            tau,
            p,
            cate_identified=cate_identified,
            tau_treat=tau_treat,
            prop_high=prop_high,
            lift_distinguishable=lift_distinguishable,
        )
        for tau, p in zip(out["uplift_score"], out["control_outcome_score"], strict=True)
    ]
    if not cate_identified:
        out["recommended_band"] = [_band(q) for q in out["quadrant"]]
        out["targeting_note"] = (
            "CATE not identified — bands use ablation P(close), not incremental lift. "
            "Do not use as a spend list."
        )
    else:
        out["recommended_band"] = [_band(q) for q in out["quadrant"]]
        out["targeting_note"] = "Track S / method only unless TreatmentSourceBasis is InComm-issued."
    out = out.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    keep = [
        "rank",
        "member_id",
        "care_gap_code",
        "uplift_score",
        "control_outcome_score",
        "treated_outcome_score",
        "propensity_to_treat",
        "quadrant",
        "recommended_band",
        "treatment_source_basis",
        "targeting_note",
    ]
    return out[[c for c in keep if c in out.columns]]
