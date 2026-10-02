"""Reward attribution without an issuance-to-claim link.

Real InComm data cannot attribute a closure to a reward: every closing claim precedes its reward case (measured
4,726 of 4,726), so there is no offer-then-claim sequence to observe. Attribution is therefore causal, not
temporal:

  probability of necessity PN = P(closure would not have happened without the offer | offered and closed)

Under exogenous assignment (randomized or mock-random) and monotone response (an offer cannot stop a closure),
PN = (P1 - P0) / P1, the attributable fraction among treated closers. Without monotonicity the Tian-Pearl
bounds are  max(0, (P1-P0)/P1) <= PN <= min(1, (1-P0)/P1). The complement is the windfall share: payouts to
members who would have closed anyway. Pay-on-completion makes this the central economic quantity.

On the research mock these bounds are wide (the mock has no effect), which is the correct statement of what the
data can support. A temporal test (does an offer pull closure earlier?) runs on the research ledger, where
offers precede claims by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ..core.evidence import NOT_AVAILABLE


def pn_bounds(p1: float, p0: float) -> dict[str, float]:
    """Tian-Pearl bounds on the probability of necessity under exogeneity (and the monotone point value)."""
    if not (np.isfinite(p1) and np.isfinite(p0)) or p1 <= 0:
        return {"pn_lower": np.nan, "pn_upper": np.nan, "pn_monotone": np.nan}
    return {
        "pn_lower": max(0.0, (p1 - p0) / p1),
        "pn_upper": min(1.0, (1.0 - p0) / p1),
        "pn_monotone": float(np.clip((p1 - p0) / p1, 0.0, 1.0)),
    }


def pn_from_rr(rr: float) -> float:
    """Attributable fraction among treated closers for a relative risk RR >= 1: 1 - 1/RR."""
    return max(0.0, 1.0 - 1.0 / rr)


def timing_test(eligibility_path: str, window_days: int) -> dict:
    """Does an offer pull closure earlier? Days from the (potential) offer date to the closing claim.

    Uses the research ledger's eligibility sidecar: offers precede claims by construction. Compares offered vs
    not-offered closers inside the window (Mann-Whitney).
    """
    p = Path(eligibility_path)
    if not p.exists():
        return {"status": NOT_AVAILABLE, "reason": f"{p.name} not generated"}
    e = pd.read_csv(p, parse_dates=["potential_issued_at", "closed_at"])
    e = e[e["eligible"].astype(bool) & e["closed_at"].notna()]
    days = (e["closed_at"] - e["potential_issued_at"]).dt.days
    inside = (days > 0) & (days <= window_days)
    e, days = e[inside], days[inside]
    t = days[e["assigned"].astype(bool)]
    c = days[~e["assigned"].astype(bool)]
    if len(t) < 30 or len(c) < 30:
        return {"status": NOT_AVAILABLE, "reason": "too few closers per arm"}
    p_val = float(mannwhitneyu(t, c, alternative="less").pvalue)  # H1: offered close sooner
    return {
        "status": "MEASURED_ON_RESEARCH_LEDGER",
        "n_offered_closers": int(len(t)),
        "n_not_offered_closers": int(len(c)),
        "median_days_offered": float(t.median()),
        "median_days_not_offered": float(c.median()),
        "p_offered_close_sooner": p_val,
        "offer_accelerates_closure": bool(p_val < 0.05),
    }


def load_real_timing(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"status": NOT_AVAILABLE}
    t = json.loads(p.read_text(encoding="utf-8"))
    return {
        "status": "MEASURED",
        **t,
        "share_claims_before_case": t["claim_before_case"] / t["closed_gaps"],
        "closures_after_an_offer": t["closed_gaps"] - t["claim_before_case"],
    }


def attribution_summary(
    treated_rate: float, control_rate: float, ate_mean: float, ate_se: float, real_timing: dict, ledger_timing: dict
) -> dict:
    b = pn_bounds(treated_rate, control_rate)
    lo = pn_bounds(treated_rate, treated_rate - (ate_mean + 1.96 * ate_se))["pn_monotone"]
    hi = pn_bounds(treated_rate, treated_rate - (ate_mean - 1.96 * ate_se))["pn_monotone"]
    return {
        "method": "probability of necessity (Tian-Pearl bounds; monotone point value)",
        "basis": "Research-generated IncentiveRewards mock rates (not InComm lift)",
        **b,
        "pn_monotone_ci": [lo, hi],
        "windfall_share_of_payouts_max": 1.0 - b["pn_lower"] if np.isfinite(b["pn_lower"]) else np.nan,
        "windfall_share_of_payouts_min": 1.0 - b["pn_upper"] if np.isfinite(b["pn_upper"]) else np.nan,
        "real_data_timing": real_timing,
        "ledger_timing_test": ledger_timing,
        "real_data_temporal_attribution": (
            "IMPOSSIBLE: every closing claim precedes its reward case; offer timestamps are required "
            "(see docs/Reward_Pilot_Experiment_Design.md)"
            if real_timing.get("closures_after_an_offer", 1) == 0
            else "partially possible"
        ),
    }
