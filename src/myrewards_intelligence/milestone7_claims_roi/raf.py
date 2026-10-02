"""RAF (risk adjustment) arm. Reported SEPARATELY from Stars/QBP and never summed with it.

Mechanism: a reward-driven closure produces a claim; that claim moves the Risk Adjustment Factor only if
  (1) the encounter is risk-adjustment eligible (a face-to-face visit, not a laboratory or diagnostic test),
  (2) its diagnosis maps to a payment HCC, and
  (3) that condition was not already coded this year (HCCs are recaptured annually).
Value = P(payment-HCC dx on the closing claim) x P(not already captured) x V28 factor x 12 x base payment PMPM.

Compliance: this measures the value of an encounter OPPORTUNITY. A reward must never be conditioned on a
diagnosis, and RAF value is only real where the condition is genuine and documented. It is excluded from the
headline ROI for that reason.

Encounter classification below is OUR research mapping from each gap's CPT set in ref_CareGapDefinitions
(observed 2026-10-01); acceptable-provider-type rules still apply and should be confirmed with InComm/plan
compliance. Only V28 factors verified in assumptions.py are used; anything else is NOT_AVAILABLE with an upper
bound from the largest verified factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .assumptions import (
    RAF_BASE_PAYMENT_PMPM,
    RAF_MAX_VERIFIED_FACTOR,
    RAF_P_NOT_ALREADY_CAPTURED,
    V28_VERIFIED_FACTORS,
)

# gap -> (risk-adjustment-eligible encounter?, rationale)
GAP_ENCOUNTER: dict[str, tuple[bool, str]] = {
    "DIAB_A1C_TEST": (False, "laboratory test (CPT 83036/83037)"),
    "DIAB_BLOOD_SUGAR": (False, "laboratory test (CPT 82947/82950/82951/83036)"),
    "DIAB_NEPHRO_SCREEN": (False, "laboratory test (CPT 82043/82570/82565)"),
    "DIAB_EYE_EXAM": (False, "retinal imaging (CPT 92250/92227/92228)"),
    "COPD_SPIROMETRY": (False, "diagnostic test (CPT 94010/94014/94060)"),
    "HTN_PCP_FOLLOWUP": (True, "office E/M visit (CPT 99213-99215)"),
    "BH_POST_DISCHARGE_7D": (True, "psychotherapy visit (CPT 90832/90834/90837)"),
    "CKD_NEPHROLOGY_VISIT": (True, "nephrology E/M visit (CPT 99213-99215)"),
}

# ICD-10 category (first 3 characters) -> (condition, maps to a payment HCC?, verified V28 key or None)
DX_CATEGORY: dict[str, tuple[str, bool, str | None]] = {
    "E10": ("diabetes", True, "diabetes"),
    "E11": ("diabetes", True, "diabetes"),
    "E13": ("diabetes", True, "diabetes"),
    "I50": ("heart failure", True, None),
    "J44": ("COPD", True, None),
    "N18": ("chronic kidney disease (stage-dependent)", True, None),
    "F33": ("major depression, recurrent", True, None),
    "F41": ("anxiety (not a payment HCC)", False, None),
}


def _value(share: float, factor: float, case: str) -> float:
    """USD per incremental closure: P(payment-HCC dx) x P(not yet captured) x factor x 12 x base payment PMPM."""
    return (
        share
        * getattr(RAF_P_NOT_ALREADY_CAPTURED, case)
        * factor
        * 12
        * getattr(RAF_BASE_PAYMENT_PMPM, case)
    )


def raf_table(dx_path: str) -> pd.DataFrame:
    """One row per care gap with the RAF value range per incremental closure (or why it is zero / unavailable)."""
    dx = pd.read_csv(dx_path)
    rows = []
    for gap, g in dx.groupby("care_gap_code"):
        eligible, why = GAP_ENCOUNTER.get(str(gap), (False, "unclassified"))
        total = float(g["closing_claims"].sum())
        mapped = g[g["icd10_category"].isin([k for k, v in DX_CATEGORY.items() if v[1]])]
        share = float(mapped["closing_claims"].sum() / total) if total else 0.0
        top = mapped.sort_values("closing_claims", ascending=False)
        cond, key = (DX_CATEGORY[top["icd10_category"].iloc[0]][0], DX_CATEGORY[top["icd10_category"].iloc[0]][2]) if len(top) else ("none", None)
        coef = V28_VERIFIED_FACTORS.get(key) if key else None
        row = {
            "care_gap_code": gap,
            "ra_eligible_encounter": eligible,
            "encounter_basis": why,
            "payment_hcc_dx_share": share,
            "dominant_condition": cond,
            "v28_factor_verified": coef if coef is not None else np.nan,
        }

        for case in ("low", "base", "high"):
            row[f"raf_value_usd_{case}"] = (
                _value(share, coef, case) if (eligible and coef is not None and share > 0) else np.nan
            )
        row["raf_value_upper_bound_usd"] = (
            _value(share, RAF_MAX_VERIFIED_FACTOR, "high") if (eligible and coef is None and share > 0) else np.nan
        )
        if not eligible:
            row["status"] = "ZERO: not a risk-adjustment-eligible encounter"
            row.update(raf_value_usd_low=0.0, raf_value_usd_base=0.0, raf_value_usd_high=0.0)
        elif share == 0:
            row["status"] = "ZERO: diagnosis is not a payment HCC"
            row.update(raf_value_usd_low=0.0, raf_value_usd_base=0.0, raf_value_usd_high=0.0)
        elif coef is None:
            row["status"] = "NOT_AVAILABLE: V28 factor not verified (upper bound shown)"
        else:
            row["status"] = "ESTIMATED (research range)"
        rows.append(row)
    return pd.DataFrame(rows)
