"""Milestone 7 ROI framework on Track R research-generated lift.

Consumes scored_uplift_track_r.csv only as research-generated input. Never
imports milestone5. QBP/Star/RAF dollars are unavailable unless the caller
supplies a value, which is then tagged as a research assumption, not InComm's.
Unavailable numbers are NaN with a sibling status column, never strings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.evidence import NOT_AVAILABLE
from ..milestone6_reward_uplift.domain_resolutions import (
    CATALOG_DEFAULT_AMOUNTS,
    EXCLUDE_CARE_GAPS,
)
from ..milestone6_reward_uplift.evidence import TreatmentSource, m7_use
from .stars import CASES, StarValueModel

BASIS = TreatmentSource.RESEARCH_GENERATED_LEDGER.value
assert m7_use(TreatmentSource.RESEARCH_GENERATED_LEDGER) == "research-only"
ASSUMPTION_TAG = "Research assumption (not InComm financials)"
Z_80_POWER = 2.8  # MDE = (1.96 + 0.84) * SE


def expected_amounts(pools_path: str | None) -> dict[str, float]:
    """Mean offered amount per care gap: empirical InComm pool if present, else catalog default."""
    out = dict(CATALOG_DEFAULT_AMOUNTS)
    if pools_path and Path(pools_path).exists():
        pools = pd.read_csv(pools_path)
        for gap, g in pools.groupby("care_gap_code"):
            out[str(gap)] = float((g["amount"] * g["weight"]).sum() / g["weight"].sum())
    return out


def load_scores(path: str, pools_path: str | None = None) -> pd.DataFrame:
    """Load Track R scores; refuse anything not tagged as the research mock."""
    df = pd.read_csv(path)
    bad = set(df["TreatmentSourceBasis"]) - {BASIS}
    if bad:
        raise ValueError(f"M7 only consumes {BASIS!r}; found {sorted(bad)}")
    df = df[~df["CareGapCode"].isin(EXCLUDE_CARE_GAPS)].copy()
    df["amount"] = df["CareGapCode"].map(expected_amounts(pools_path))
    return df.dropna(subset=["amount"])


def load_lift_uncertainty(results_path: str) -> tuple[float, float]:
    """(ate_mean, ate_se) from the M6 results file; NaN if M6 didn't report them."""
    with open(results_path, encoding="utf-8") as f:
        r = json.load(f)
    return float(r.get("ate_mean", np.nan)), float(r.get("ate_se", np.nan))


def lift_status(ate_mean: float, ate_se: float) -> str:
    if not (np.isfinite(ate_mean) and np.isfinite(ate_se)):
        return NOT_AVAILABLE
    if abs(ate_mean) <= 1.96 * ate_se:
        return "LIFT_NOT_DISTINGUISHABLE_FROM_ZERO"
    return "LIFT_DISTINGUISHABLE" if ate_mean > 0 else "NEGATIVE_LIFT"


def validate_claims(df: pd.DataFrame, ate_mean: float, ate_se: float, claims: dict | None = None) -> dict:
    """Lift diagnostics plus the measured claims validation (validation.py) when available."""
    claims = claims or {}
    measured = claims.get("status") == "MEASURED"
    return {
        "rows": len(df),
        "cate_identified_share": float(df["CateIdentified"].mean()),
        "all_rows_mock_tagged": bool((df["TreatmentSourceBasis"] == BASIS).all()),
        "ate_mean": ate_mean,
        "ate_se": ate_se,
        "minimum_detectable_effect": Z_80_POWER * ate_se if np.isfinite(ate_se) else np.nan,
        "lift_status": lift_status(ate_mean, ate_se),
        "claims_validation": claims or {"status": NOT_AVAILABLE},
        "cpt_hcpcs_matching_status": "MEASURED" if measured else NOT_AVAILABLE,
        "attribution_to_reward_status": NOT_AVAILABLE,  # no issuance record is linked to a claim
    }


def scenarios(
    df: pd.DataFrame,
    fractions=(0.1, 0.25, 0.5, 1.0),
    value_per_closure: float | None = None,
    redeem_p: float = 1.0,
    ate_mean: float = np.nan,
    ate_se: float = np.nan,
    star_model: StarValueModel | None = None,
) -> pd.DataFrame:
    """Treat top-fraction by UpliftScore.

    Cost model = pay on completion (InComm rewards are earned when the health action is done, so
    HEALTH_ACTION_REWARD cases == Closed gaps): expected cost = sum(amount x P(close | offered)) x redeem_p.
    This is why uplift matters: baseline "sure thing" closers are paid but add no lift.
    reward_cost_issued_usd (every offer paid) is the upper bound. Pricing is refused unless population lift
    is distinguishable from zero and the scenario's incremental closures are positive.
    """
    lift = lift_status(ate_mean, ate_se)
    ranked = df.sort_values("UpliftScore", ascending=False).reset_index(drop=True)
    rows = []
    for f in fractions:
        s = ranked.head(max(1, round(len(ranked) * f)))
        incr = float(s["UpliftScore"].sum())
        issued = float(s["amount"].sum())
        if "TreatedOutcomeScore" in s.columns:
            p_close = s["TreatedOutcomeScore"].clip(0, 1)
            cost = float((s["amount"] * p_close).sum()) * redeem_p
            cost_model = "pay_on_completion"
        else:
            cost, cost_model = issued * redeem_p, "pay_on_issuance_upper_bound"
        if lift != "LIFT_DISTINGUISHABLE":
            status = lift
        elif incr <= 0:
            status = "NO_POSITIVE_LIFT"
        else:
            status = "PRICED"
        priced = status == "PRICED"
        stars = {}
        for case in CASES:
            mean_v = np.nan if star_model is None else star_model.mean_value(s["CareGapCode"], case)
            stars[f"stars_value_per_closure_usd_{case}"] = mean_v
            # average lift per treated opportunity needed for ROI = 0; independent of measured lift
            stars[f"required_lift_pp_for_breakeven_{case}"] = 100 * cost / (mean_v * len(s)) if mean_v else np.nan
            stars[f"stars_roi_{case}"] = (incr * mean_v - cost) / cost if priced and mean_v == mean_v else np.nan
        rows.append(
            {
                "fraction_treated": f,
                "n_treated": len(s),
                # in-sample sum of uplift scores of the top-ranked fraction: optimistic under selection on noise
                "incremental_closures": incr,
                "incremental_closures_population_ate": len(s) * ate_mean if np.isfinite(ate_mean) else np.nan,
                # population-level uncertainty: n_treated x ATE standard error (conservative, ignores ranking gain)
                "incremental_closures_ci_low": incr - 1.96 * len(s) * ate_se,
                "incremental_closures_ci_high": incr + 1.96 * len(s) * ate_se,
                "reward_cost_issued_usd": issued,
                "reward_cost_expected_usd": cost,
                "cost_model": cost_model,
                "pricing_status": status,
                "breakeven_value_per_closure_usd": cost / incr if priced else np.nan,
                "roi": (incr * value_per_closure - cost) / cost
                if priced and value_per_closure is not None
                else np.nan,
                "value_basis": f"{ASSUMPTION_TAG}: ${value_per_closure}/closure"
                if priced and value_per_closure is not None
                else NOT_AVAILABLE,
                **stars,  # Stars->QBP range; basis in assumptions.py (research/public/InComm tagged)
                "stars_value_basis": "CMS 2026 public Star Ratings data (boundary density, cut points, weights) + KFF 2026 QBP$ + public prevalence",
                "financial_slots_status": "Stars/QBP: empirical range (CMS public data); RAF: separate arm",
                "TreatmentSourceBasis": BASIS,
            }
        )
    return pd.DataFrame(rows)
