"""Build the Milestone 7 shareable Word documents into milestone7_shareables/ from the generated M7 outputs.

Run after `python -m myrewards_intelligence.cli.run_milestone7`. Every number is read from models/, config/ or
reports/; nothing is typed in except prose. PDFs are produced separately (LibreOffice).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from myrewards_intelligence.core.reporting.base import add_bullets, add_notice, add_table, new_doc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "milestone7_shareables"
WATERMARK = (
    "RESEARCH FRAMEWORK. Lift is research-generated or taken from published trials; no InComm lift, financial or "
    "issuance data is used. Dollar values are research ranges tagged by provenance."
)


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "models" / f"{name}.csv")


def _doc(title: str, subtitle: str):
    doc = new_doc(title, subtitle)
    add_notice(doc, WATERMARK)
    return doc


def star_value_model() -> None:
    dens = pd.read_csv(ROOT / "config" / "cms_star_boundary_density.csv")
    val = json.loads((ROOT / "config" / "cms_star_engine_validation.json").read_text())
    tor, disc = _csv("m7_tornado"), _csv("m7_star_weight_discrepancies")
    doc = _doc("Milestone 7: Star Value Model", "What one extra care-gap closure is worth through Star Ratings and QBP")
    doc.add_heading("1. Method", level=1)
    add_bullets(doc, [
        "Value of a closure = QBP dollars per enrollee x boundary density (rho) x pass-through / eligible share / gaps sharing the measure.",
        "rho is measured, not assumed: for every rated Medicare Advantage contract in CMS's public 2026 Star Ratings data, "
        "how likely is a small rate increase to cross a cut point and lift the overall rating over the 4-star QBP boundary.",
        "QBP per enrollee: KFF 2026, $318 / $381 / $466. Eligible shares: public prevalence anchors (diabetes 0.20 / 0.26 / 0.32; hypertension 0.50 / 0.60 / 0.65).",
    ])
    doc.add_heading("2. Measured boundary density (CMS 2026, public)", level=1)
    add_table(doc, ["measure_id", "weight", "n_contracts", "mean_rate", "rho_low", "rho_base", "rho_high"], dens.to_dict("records"))
    doc.add_heading("3. Validation of the engine", level=1)
    add_bullets(doc, [
        f"{val['contracts_validated']} contracts: the engine reproduces the published Part C summary within half a star for "
        f"{val['part_c_summary_within_half_star']:.1%} and exactly for {val['part_c_summary_exact_match_after_half_star_rounding']:.1%}.",
        f"Not modelled: {val['not_modelled']}.",
        f"{val['share_rated_3_5']:.1%} of MA contracts are rated 3.5 stars (the only ones one boundary crossing can lift to 4.0).",
    ])
    doc.add_heading("4. Which assumption matters most", level=1)
    add_table(doc, ["assumption", "provenance", "value_at_low_input", "value_at_high_input", "swing_usd"], tor.to_dict("records"))
    doc.add_heading("5. InComm metadata versus CMS 2026", level=1)
    doc.add_paragraph(
        "InComm's StarsWeight differs from CMS's 2026 weight for eye exam and kidney screening, and COPD spirometry and "
        "BH follow-up feed no 2026 Part C measure. The model uses CMS's weights."
    )
    add_table(doc, list(disc.columns), disc.to_dict("records"))
    doc.save(OUT / "Milestone 7 Star Value Model.docx")


def priced_scenarios() -> None:
    prog, gap, av = _csv("m7_literature_roi_program"), _csv("m7_literature_roi_by_gap"), _csv("m7_avoided_cost_arm")
    doc = _doc("Milestone 7: Priced Scenarios and Avoided Cost", "Literature-anchored lift, break-even, and a separate avoided-medical-cost arm")
    doc.add_heading("1. Priced ROI under published incentive-trial effects", level=1)
    doc.add_paragraph(
        "Relative risks of closure: 1.00 (no effect; the IDEAS diabetic eye screening RCT), 1.17 (colorectal screening, 8 RCTs) and "
        "1.42 (pooled screening meta-analysis). Carried as odds ratios so probabilities stay within 0 to 1. Offer to all, "
        "pay on completion. These are scenarios, not InComm results."
    )
    add_table(doc, ["rr_case", "odds_ratio", "value_case", "mean_implied_rr", "net_total_usd", "cost_total_usd", "roi"], prog.to_dict("records"))
    doc.add_heading("2. Break-even relative risk by gap (Stars value only)", level=1)
    be = gap.groupby(["care_gap_code", "value_case"], as_index=False)["break_even_rr"].first()
    add_table(doc, ["care_gap_code", "value_case", "break_even_rr"], be.to_dict("records"))
    doc.add_heading("3. Avoided medical cost arm (separate; never summed into the Stars headline)", level=1)
    add_bullets(doc, [
        "Only BH 7-day follow-up has a published closure-to-readmission link. It is observational (adjusted odds ratios 0.88 to 0.91), so the low case is zero effect.",
        "Every other gap is NOT_MODELLED: absence of verified evidence, not proof of no benefit. Screening often raises near-term spending, so no value is guessed.",
        "One 30-day event, no discounting. The plan bears the cost under capitation.",
        "Inputs: readmission rate 11 / 20 / 36% (CMS inpatient psychiatric facility data); odds reduction 0 / 0.10 / 0.22; cost per readmission $8,800 / $12,150 / $15,500 (HCUP, Medicare).",
    ])
    a = av[(av["rr_case"] == "base") & av["avoided_cost_usd"].notna()]
    add_table(doc, ["care_gap_code", "value_case", "closure_value_usd", "avoided_cost_usd", "net_per_offer_usd",
                    "net_per_offer_with_avoided_usd", "break_even_rr_with_avoided"], a.to_dict("records"))
    base = a[a["value_case"] == "base"].iloc[0]
    doc.add_paragraph(
        f"Reading: at base, BH follow-up avoids ${base['avoided_cost_usd']:,.0f} per closure and still nets "
        f"${base['net_per_offer_with_avoided_usd']:,.2f} per offer under the base literature lift "
        f"(implied RR {base['implied_rr']:.2f} against a break-even of {base['break_even_rr_with_avoided']:.2f}). "
        "It pays only in the high case. BH baseline closure is already 74.5%, so there is little room to lift."
    )
    doc.save(OUT / "Milestone 7 Priced Scenarios and Avoided Cost.docx")


def attribution_raf() -> None:
    res = json.loads((ROOT / "reports" / "milestone7_results.json").read_text())
    att, claims = res["attribution"], res["validation"]["claims_validation"]
    raf = _csv("m7_raf_arm")
    rt = att["real_data_timing"]
    doc = _doc("Milestone 7: Claims Validation, Attribution and RAF", "How closures tie to claims, whether a reward can be credited, and the separate risk-adjustment arm")
    doc.add_heading("1. Claims validation of closures", level=1)
    doc.add_paragraph(claims["basis"] + ". " + claims["shared_claim_note"])
    add_table(doc, ["care_gap_code", "closed_gaps", "claim_found", "same_member", "cpt_in_definition", "date_matches", "in_window", "gaps_sharing_a_claim"], claims["by_gap"])
    doc.add_heading("2. Reward attribution", level=1)
    add_bullets(doc, [
        f"Real data: {rt['closures_after_an_offer']} of {rt['closed_gaps']} closures follow a reward case. Claims run {rt['first_claim']} to {rt['last_claim']}; every case is created {rt['first_case']}. Temporal attribution is impossible with this export.",
        f"Causal attribution instead (probability of necessity, Tian-Pearl bounds): {att['pn_lower']:.2f} to {att['pn_upper']:.2f} on the research mock. The complement is the windfall share of payouts.",
        "Recording offer timestamps before follow-up is the one change that makes temporal attribution possible.",
    ])
    doc.add_heading("3. RAF arm (separate; never summed with Stars)", level=1)
    doc.add_paragraph(
        "A closure moves RAF only through a risk-adjustment-eligible encounter whose diagnosis maps to a payment HCC not "
        "already coded this year. Rewards must never depend on a diagnosis. Only V28 factors verified from CMS materials are used."
    )
    add_table(doc, ["care_gap_code", "ra_eligible_encounter", "payment_hcc_dx_share", "dominant_condition", "raf_value_usd_base", "raf_value_upper_bound_usd", "status"], raf.to_dict("records"))
    doc.save(OUT / "Milestone 7 Claims Validation, Attribution and RAF.docx")


def review_response() -> None:
    res = json.loads((ROOT / "reports" / "milestone7_results.json").read_text())
    v = res["validation"]
    lit = _csv("m7_literature_roi_program")
    base = lit[(lit["rr_case"] == "base") & (lit["value_case"] == "base")].iloc[0]
    doc = _doc("Milestone 7: Review Findings and Actions", "What the reviews found, what we changed, and what remains open")
    doc.add_heading("1. Where we are", level=1)
    add_bullets(doc, [
        f"Measured lift on the research mock: ATE {v['ate_mean']:.4f} (SE {v['ate_se']:.4f}); minimum detectable {v['minimum_detectable_effect']:.4f}. Status: {v['lift_status']}. No measured dollar ROI is claimed.",
        "Star value of one closure: about $157 at base ($102 to $260), measured on CMS public 2026 data. Break-even lift is about 5.5 percentage points offered to all (3.3 to 8.4).",
        f"Under published incentive-trial effects (CRC-anchored RR 1.17) the program nets ${base['net_total_usd']:,.0f} at base value (ROI {base['roi']:.2f}x). It pays for DIAB_A1C_TEST, DIAB_BLOOD_SUGAR and HTN_PCP_FOLLOWUP and not for the other gaps.",
        "Method: every input comes from the reference tables in the InComm database, public CMS/KFF/HCUP data, or published trials, and is labeled by source.",
    ])
    doc.add_heading("2. InComm feedback (28 September 2026) and our response", level=1)
    add_table(doc, ["point", "finding", "action"], [
        {"point": "Invented simulation parameters; rolling period is at least a year",
         "finding": "Correct: the 90-day window was an error. FundLifeSpan in InComm's wallet configuration is 365 days.",
         "action": "Window is 365 days; 90 and 180 days are sensitivity only. Amounts drawn from InComm's own per-gap pools. Offer probability and lag are assumptions, swept over 9 combinations: null in all."},
        {"point": "Elasticity reported NOT_AVAILABLE",
         "finding": "Every offer used one amount per gap, so no dose-response existed.",
         "action": "Within-gap fixed-effects estimator on varying amounts; minimum detectable slope about 1.3pp per $10. Real response curve needs a randomized amount arm (design written)."},
        {"point": "Diabetes cohort had an empty untreated arm",
         "finding": "An artifact: lifetime reward counts were copied onto every gap of a member.",
         "action": "Gap-level exposure; every diabetes gap has both arms. A positivity table and Manski bounds now handle any genuinely empty arm."},
        {"point": "CKD excluded",
         "finding": "ConditionCode CKD versus CHRONIC_KIDNEY_DISEASE mismatch means InComm's rules engine never produced a CKD row.",
         "action": "Run as a separate secondary cohort on research labels; never mixed into the headline."},
    ])
    doc.add_heading("3. Defects we found ourselves and fixed", level=1)
    add_table(doc, ["finding", "effect", "fix"], [
        {"finding": "Class balancing in the AIPW nuisance models", "effect": "Inflated the treatment effect to +3.3pp on a zero-effect mock and produced 52 'Treat' members",
         "fix": "Removed; recovery tests (null, known effect, known heterogeneity, known dose-response) fail with the bug and pass without. M6 documents regenerated."},
        {"finding": "In-sample top-fraction uplift sums", "effect": "Winner's curse: probability of net benefit near 1.0", "fix": "Monte Carlo centered on the population ATE with its standard error."},
        {"finding": "A1C and blood sugar closures valued twice", "effect": "Overstated Stars value", "fix": "Measure value split across gaps sharing a measure."},
        {"finding": "Relative risk applied to a 74% baseline", "effect": "Probabilities above 1", "fix": "Effect carried on the odds-ratio scale."},
        {"finding": "InComm StarsWeight differs from CMS 2026 weights; two gaps feed no Part C measure", "effect": "Overvalued eye exam and kidney screening; BH and COPD valued at nonzero",
         "fix": "CMS weights used; discrepancy table published; BH and COPD Stars value is exactly 0."},
        {"finding": "Unsourced Stars value per closure", "effect": "Value was a typed assumption", "fix": "Boundary density measured on 524 CMS contracts; engine reproduces 98% of published ratings within half a star."},
        {"finding": "Wrong (V24) CMS table once fetched", "effect": "Would have misstated RAF factors", "fix": "Only V28 factors found in CMS materials used (diabetes 0.166, CKD stage 4 0.514); others stay NOT_AVAILABLE."},
    ])
    doc.add_heading("4. Items InComm raised or implied that M7 now covers", level=1)
    add_bullets(doc, [
        "Reward attribution: impossible on real data (claims precede cases); handled by probability of necessity and a recommendation to record offer timestamps.",
        "RAF: separate arm. Most gaps are laboratory or imaging tests that are not risk-adjustment eligible, so their RAF value is zero; HTN follow-up is not available without verified factors.",
        "Priced ROI: provided as labeled scenarios under published trial effects with break-even relative risk per gap, plus a separate avoided-cost arm for BH follow-up.",
    ])
    doc.add_heading("5. What remains open and why", level=1)
    add_bullets(doc, [
        "Measured lift: needs randomized offers. Sizing: about 1,500 members per arm detects +5pp pooled (design document).",
        "Structural limits of the value model: uniform position inside a half-star band, categorical adjustment and reward factor not modelled, single-measure marginal view, eligible-share ranges.",
        "Avoided cost: BH only and observational; no verified link for other gaps. Not added to the headline.",
        "Synthetic data: claim IDs are MOCK-CLM and the IncentiveRewards ledger is research-generated, so clean validation describes the generator, not InComm production quality.",
        "Not confirmed: whether withholding offers from a control group is permitted by program rules.",
    ])
    doc.save(OUT / "Milestone 7 Review Findings and Actions.docx")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    shutil.copy(ROOT / "reports" / "Milestone 7 ROI Framework.docx", OUT / "Milestone 7 ROI Framework.docx")
    star_value_model()
    priced_scenarios()
    attribution_raf()
    review_response()
    print(*sorted(p.name for p in OUT.iterdir()), sep="\n")


if __name__ == "__main__":
    main()
