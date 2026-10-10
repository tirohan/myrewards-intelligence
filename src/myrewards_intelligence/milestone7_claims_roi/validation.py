"""Referential-integrity check of care-gap closures (claim, member, CPT, timing, claim sharing), and a test of the check.

Input is models/claims_validation.csv from export_claims_validation.py (aggregates from InComm's export). Claim ids
in that export are MOCK-CLM-*. Closure status is derived from the same claim, CPT and date the checks read back, so
member, CPT, date and window agree by construction: 100% on them shows the export is internally consistent, not that
closures are clinically valid. That is why this is NOT reported as an independent validation.

What does show the CHECK works: fault_injection_test() corrupts a known share of clean closures (missing claim, wrong
member, wrong CPT, shifted date, out of window, claim reused across members) and measures how many the checker flags.
Reward attribution stays NOT_AVAILABLE: no issuance record is linked to a claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
        "basis": (
            "Referential-integrity check on InComm's export (claim ids MOCK-CLM-*). Closure is derived from the same "
            "claim, so member, CPT, date and window agree by construction; 100% shows internal consistency, "
            "not clinical validity or production data quality"
        ),
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


FAULTS = ("missing_claim", "wrong_member", "wrong_cpt", "shifted_date", "out_of_window", "claim_reused_across_members")


def check_closures(closures: pd.DataFrame, lines: pd.DataFrame, qualifying: dict[str, set[str]]) -> pd.DataFrame:
    """Record-level version of the export's checks, plus one the closure derivation does not guarantee (a claim id
    that appears under more than one member). closures: gap, claim_id, member_id, cpt, closing_date, window_start,
    window_end. lines: claim_id, member_id, cpt, service_from (a claim id may repeat). One boolean flag_* per fault."""
    c = closures.reset_index(drop=True).assign(_i=lambda d: range(len(d)))
    j = c.merge(lines.rename(columns={"member_id": "lm", "cpt": "lc", "service_from": "ls"}), on="claim_id", how="left")
    j["has"] = j["lm"].notna()
    j["same"] = j["member_id"] == j["lm"]
    g = j.groupby("_i")
    has, same, members = g["has"].any(), g["same"].any(), g["lm"].nunique()
    r = j.sort_values(["_i", "same"], ascending=[True, False]).drop_duplicates("_i").set_index("_i")  # matching-member line first
    ok = r["has"]
    in_def = pd.Series([cpt in qualifying.get(gap, set()) for gap, cpt in zip(r["gap"], r["cpt"], strict=True)], index=r.index)
    return pd.DataFrame({
        "flag_missing_claim": ~has,
        "flag_wrong_member": has & ~same,
        "flag_wrong_cpt": ok & ((r["cpt"] != r["lc"]) | ~in_def),
        "flag_shifted_date": ok & (r["closing_date"] != r["ls"]),
        "flag_out_of_window": ok & ((r["ls"] < r["window_start"]) | (r["ls"] > r["window_end"])),
        "flag_claim_reused_across_members": members > 1,
    }).reset_index(drop=True)


def fault_injection_test(n: int = 3000, rate: float = 0.05, seed: int = 7) -> pd.DataFrame:
    """Build clean closures, corrupt `rate` of rows per fault type (disjoint rows), run the checker. Returns recall per
    fault type and the false-positive rate on untouched rows. A checker that cannot fail would show recall 0."""
    rng = np.random.default_rng(seed)
    qualifying = {"GAP_A": {"83036", "83037"}, "GAP_B": {"99213", "99214"}}
    gaps = rng.choice(list(qualifying), n)
    cpt = np.array([sorted(qualifying[g])[rng.integers(0, 2)] for g in gaps])
    start = pd.Timestamp("2025-01-01")
    service = start + pd.to_timedelta(rng.integers(30, 300, n), unit="D")
    closures = pd.DataFrame({
        "gap": gaps, "claim_id": [f"C{i}" for i in range(n)], "member_id": [f"M{i}" for i in range(n)], "cpt": cpt,
        "closing_date": service, "window_start": start, "window_end": start + pd.Timedelta(days=365),
    })
    lines = pd.DataFrame({"claim_id": closures["claim_id"], "member_id": closures["member_id"], "cpt": cpt, "service_from": service})
    idx = rng.permutation(n)
    k = int(n * rate)
    hit = {f: idx[i * k:(i + 1) * k] for i, f in enumerate(FAULTS)}
    lines.loc[hit["missing_claim"], "claim_id"] = "GONE"
    lines.loc[hit["wrong_member"], "member_id"] = "OTHER"
    lines.loc[hit["wrong_cpt"], "cpt"] = "00000"
    lines.loc[hit["shifted_date"], "service_from"] += pd.Timedelta(days=3)
    late = start + pd.Timedelta(days=500)
    lines.loc[hit["out_of_window"], "service_from"] = late
    closures.loc[hit["out_of_window"], "closing_date"] = late  # date agrees; only the window is wrong
    reuse = hit["claim_reused_across_members"]
    stranger = pd.DataFrame({"claim_id": closures.loc[reuse, "claim_id"], "member_id": "STRANGER",
                             "cpt": closures.loc[reuse, "cpt"], "service_from": closures.loc[reuse, "closing_date"]})
    flags = check_closures(closures, pd.concat([lines, stranger], ignore_index=True), qualifying)
    rows = [{"fault": f, "injected": len(hit[f]), "detected": int(flags[f"flag_{f}"].iloc[hit[f]].sum()),
             "recall": float(flags[f"flag_{f}"].iloc[hit[f]].mean())} for f in FAULTS]
    clean = np.setdiff1d(np.arange(n), np.concatenate(list(hit.values())))
    fp = flags.iloc[clean].any(axis=1)
    rows.append({"fault": "none (clean rows flagged = false positives)", "injected": len(clean),
                 "detected": int(fp.sum()), "recall": float(fp.mean())})
    return pd.DataFrame(rows)
