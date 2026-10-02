"""Avoided medical cost arm. Reported SIDE BY SIDE with Stars/QBP, never silently summed.

Only gaps with a published closure -> utilization link carry a value (assumptions.AVOIDED_COST_GAPS). Every other gap
is NOT_MODELLED and contributes 0 here: absence of evidence, not evidence of no benefit. Preventive screening often
raises near-term spending, so no value is guessed for the screening gaps.

BH 7-day follow-up -> 30-day readmission, per incremental closure:
  p_with  = odds(p0) x (1 - odds_reduction) turned back into a probability   (p0 = no-follow-up readmission rate)
  avoided = (p0 - p_with) x cost per readmission
The low case has zero effect (the evidence is observational), so the arm can be zero but not negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .assumptions import (
    AVOIDED_COST_GAPS,
    BH_COST_PER_READMISSION,
    BH_ODDS_REDUCTION,
    BH_READMIT_RATE,
)

CASES = ("low", "base", "high")


def bh_avoided_value(case: str) -> float:
    p0 = getattr(BH_READMIT_RATE, case)
    odds = p0 / (1 - p0) * (1 - getattr(BH_ODDS_REDUCTION, case))
    return (p0 - odds / (1 + odds)) * getattr(BH_COST_PER_READMISSION, case)


def avoided_value(gap: str, case: str) -> float:
    """USD per incremental closure; NaN where no published link was verified."""
    return bh_avoided_value(case) if gap in AVOIDED_COST_GAPS else np.nan


def combine(lit_gap: pd.DataFrame) -> pd.DataFrame:
    """Add the avoided-cost value and the Stars-only vs Stars+avoided-cost comparison to the per-gap literature grid."""
    d = lit_gap.copy()
    d["avoided_cost_usd"] = [avoided_value(g, c) for g, c in zip(d["care_gap_code"], d["value_case"], strict=True)]
    d["avoided_cost_status"] = np.where(
        d["avoided_cost_usd"].isna(), "NOT_MODELLED: no verified closure-to-utilization link", "ESTIMATED"
    )
    d["value_with_avoided_usd"] = d["closure_value_usd"] + d["avoided_cost_usd"].fillna(0.0)
    d["net_per_offer_with_avoided_usd"] = d["lift_pp"] / 100 * d["value_with_avoided_usd"] - d["cost_per_offer_usd"]
    ar = d["cost_per_offer_usd"] / d["p_close_if_offered"]  # amount x P(redeem)
    v = d["value_with_avoided_usd"]
    d["break_even_rr_with_avoided"] = np.where(v > ar, v / (v - ar), np.nan)
    return d
