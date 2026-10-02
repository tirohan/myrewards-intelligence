"""Claims validation of care-gap closures (CPT match, member, timing, claim sharing).

Input is models/claims_validation.csv from export_claims_validation.py (aggregates from InComm's export). Claim ids
in that export are MOCK-CLM-*, so a clean result describes the synthetic generator, not InComm production data.
Reward attribution stays NOT_AVAILABLE: no issuance record is linked to a claim.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.evidence import NOT_AVAILABLE

CHECKS = ("claim_found", "same_member", "cpt_in_definition", "date_matches", "in_window")


def validate_closures(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"status": NOT_AVAILABLE, "reason": f"{p.name} not exported (run export_claims_validation.py)"}
    df = pd.read_csv(p)
    total = int(df["closed_gaps"].sum())
    rates = {c: float(df[c].sum() / total) for c in CHECKS}
    shared = int(df["gaps_sharing_a_claim"].sum())
    return {
        "status": "MEASURED",
        "basis": "InComm export (synthetic; claim ids MOCK-CLM-*), not production data quality",
        "closed_gaps": total,
        "rates": rates,
        "all_checks_pass": all(r == 1.0 for r in rates.values()),
        "gaps_sharing_a_claim": shared,
        "shared_claim_note": (
            "Gaps sharing a claim map to the same HEDIS measure (e.g. DIAB_A1C_TEST and DIAB_BLOOD_SUGAR both "
            "-> CDC-A1C), so measure value is de-duplicated across them."
        ),
        "timing_note": "Claims precede outreach cases in the export (ISS-008): reward attribution is not testable.",
        "attribution_status": NOT_AVAILABLE,
        "by_gap": df.to_dict("records"),
    }
