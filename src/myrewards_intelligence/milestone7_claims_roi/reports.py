"""Word report for Milestone 7. Every page carries the research-input watermark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.reporting import add_bullets, add_table, new_doc
from .assumptions import ALL
from .roi import BASIS

WATERMARK = (
    f"RESEARCH FRAMEWORK. Lift input: {BASIS}. Not InComm-issued lift, not InComm financials. "
    "Dollar values are research ranges tagged by provenance."
)


def write_roi_report(
    validation: dict,
    scenarios: pd.DataFrame,
    path: Path,
    tornado: pd.DataFrame | None = None,
    monte_carlo: pd.DataFrame | None = None,
    attribution: dict | None = None,
    raf: pd.DataFrame | None = None,
    lit_gap: pd.DataFrame | None = None,
    lit_prog: pd.DataFrame | None = None,
    star_rows: list[dict] | None = None,
    discrepancies: pd.DataFrame | None = None,
    avoided: pd.DataFrame | None = None,
) -> Path:
    doc = new_doc("Milestone 7: Claims-Validated ROI Framework", "Stars/QBP valuation range on research-generated lift")
    doc.add_paragraph(WATERMARK).runs[0].bold = True

    doc.add_heading("1. What the lift evidence says", level=1)
    add_bullets(
        doc,
        [
            f"Lift status: {validation['lift_status']}.",
            f"ATE {validation['ate_mean']:.4f} (SE {validation['ate_se']:.4f}); "
            f"minimum detectable effect {validation['minimum_detectable_effect']:.4f}.",
            "ROI is reported only when lift is distinguishable from zero. A break-even is a floor, not an ROI.",
        ],
    )

    doc.add_heading("2. Required lift for break-even", level=1)
    doc.add_paragraph(
        "Independent of measured lift: the average lift per treated member at which the program pays for "
        "itself, under low/base/high value of one incremental closure."
    )
    central = scenarios[scenarios["redeem_p"] == scenarios["redeem_p"].iloc[0]]
    add_table(
        doc,
        ["fraction_treated", "n_treated", "reward_cost_expected_usd", "required_lift_pp_for_breakeven_low",
         "required_lift_pp_for_breakeven_base", "required_lift_pp_for_breakeven_high"],
        central.to_dict("records"),
    )

    doc.add_heading("3. Redemption sensitivity", level=1)
    add_table(
        doc,
        ["redeem_p", "fraction_treated", "reward_cost_expected_usd", "required_lift_pp_for_breakeven_base"],
        scenarios.to_dict("records"),
    )

    if tornado is not None:
        doc.add_heading("4. Which assumption matters most (tornado)", level=1)
        doc.add_paragraph(
            "USD per incremental closure (mean over the treated gap mix) when one input moves from its "
            "value-lowering to its value-raising end and the others stay at base. Ranked by swing."
        )
        add_table(
            doc,
            ["assumption", "provenance", "value_at_low_input", "value_at_high_input", "swing_usd"],
            tornado.to_dict("records"),
        )
    if monte_carlo is not None:
        doc.add_heading("5. Uncertainty over lift and assumptions (Monte Carlo)", level=1)
        doc.add_paragraph(
            "Population lift is drawn around the average treatment effect with its standard error (not the in-sample "
            "ranking gain, which is optimistic under selection) and each assumption uniformly "
            "between its bounds. These probabilities describe uncertainty in research inputs, not a forecast "
            "of InComm results."
        )
        add_table(
            doc,
            ["fraction_treated", "p_net_positive", "net_p05_usd", "net_p50_usd", "net_p95_usd"],
            monte_carlo.to_dict("records"),
        )
    claims = validation.get("claims_validation", {})
    if claims.get("status") == "MEASURED":
        doc.add_heading("6. Claims validation of closures", level=1)
        doc.add_paragraph(claims["basis"] + ". " + claims["shared_claim_note"])
        add_table(
            doc,
            ["care_gap_code", "closed_gaps", "claim_found", "same_member", "cpt_in_definition", "date_matches",
             "in_window", "gaps_sharing_a_claim"],
            claims["by_gap"],
        )
    if attribution is not None:
        rt = attribution["real_data_timing"]
        doc.add_heading("7. Reward attribution", level=1)
        add_bullets(
            doc,
            [
                f"Real data: {rt.get('closures_after_an_offer')} of {rt.get('closed_gaps')} closures follow a reward case "
                f"(claims {rt.get('first_claim')} to {rt.get('last_claim')}; cases created {rt.get('first_case')}). "
                "Temporal attribution is impossible; offer timestamps are required (pilot design).",
                "Causal attribution instead: probability of necessity (the share of treated closers whose closure the "
                f"offer caused). On the research mock the bounds are {attribution['pn_lower']:.2f} to "
                f"{attribution['pn_upper']:.2f} (monotone point {attribution['pn_monotone']:.2f}). The complement is "
                "the windfall share of payouts.",
                "Timing test on the research ledger: " + str(attribution["ledger_timing_test"].get("status")) + (
                    f"; median days to claim offered {attribution['ledger_timing_test']['median_days_offered']:.0f} vs "
                    f"{attribution['ledger_timing_test']['median_days_not_offered']:.0f} not offered "
                    f"(offer accelerates closure: {attribution['ledger_timing_test']['offer_accelerates_closure']})."
                    if attribution["ledger_timing_test"].get("status") == "MEASURED_ON_RESEARCH_LEDGER" else "."
                ),
            ],
        )
    if raf is not None:
        doc.add_heading("8. RAF arm (separate from Stars; never summed)", level=1)
        doc.add_paragraph(
            "A closure moves RAF only through a risk-adjustment-eligible encounter whose diagnosis maps to a payment "
            "HCC that was not already coded this year. Value of an encounter opportunity, not a reward condition: "
            "rewards must never depend on a diagnosis."
        )
        add_table(
            doc,
            ["care_gap_code", "ra_eligible_encounter", "payment_hcc_dx_share", "dominant_condition", "raf_value_usd_base",
             "raf_value_upper_bound_usd", "status"],
            raf.to_dict("records"),
        )
    if lit_prog is not None:
        doc.add_heading("9. Priced ROI under literature-anchored lift (a scenario, not a result)", level=1)
        doc.add_paragraph(
            "Published incentive trials give relative risks of closure of 1.42 (screening meta-analysis, 95% CI "
            "1.05 to 1.92), 1.17 (colorectal, 8 RCTs) and no effect for diabetic eye screening (IDEAS RCT). These "
            "are an external prior, not InComm lift and not the mock lift. Carried as odds ratios so probabilities stay "
            "within 0 to 1 at every gap's baseline. Offer-to-all, pay-on-completion. The favourable-value rows stack "
            "every favourable bound and are an upper envelope, not an expectation."
        )
        add_table(
            doc,
            ["rr_case", "odds_ratio", "mean_implied_rr", "value_case", "net_total_usd", "cost_total_usd", "roi",
             "attributable_share_of_closers"],
            lit_prog.to_dict("records"),
        )
        if lit_gap is not None:
            be = lit_gap.groupby(["care_gap_code", "value_case"], as_index=False)["break_even_rr"].first()
            doc.add_paragraph("Break-even relative risk by gap and value case (the program pays if true RR exceeds it):")
            add_table(doc, ["care_gap_code", "value_case", "break_even_rr"], be.to_dict("records"))
    if avoided is not None:
        doc.add_heading("9b. Avoided medical cost arm (separate; Stars-only vs Stars plus avoided cost)", level=1)
        doc.add_paragraph(
            "Only BH 7-day follow-up has a published closure-to-readmission link (observational; the low case is zero "
            "effect). Other gaps are NOT_MODELLED, not zero benefit: screening often raises near-term spending, so no "
            "value is guessed. One 30-day event, no discounting; the plan bears the cost under capitation. Under the "
            "base literature lift."
        )
        a = avoided[(avoided["rr_case"] == "base") & avoided["avoided_cost_usd"].notna()]
        add_table(
            doc,
            ["care_gap_code", "value_case", "closure_value_usd", "avoided_cost_usd", "net_per_offer_usd",
             "net_per_offer_with_avoided_usd", "break_even_rr", "break_even_rr_with_avoided"],
            a.to_dict("records"),
        )
    doc.add_heading("10. Assumptions and their provenance", level=1)
    add_table(
        doc,
        ["name", "low", "base", "high", "unit", "provenance", "source"],
        [a.__dict__ | {"provenance": a.provenance.value} for a in ALL] + (star_rows or []),
    )
    if discrepancies is not None:
        doc.add_paragraph(
            "InComm's StarsWeight metadata vs the CMS 2026 weight of the Star measure each gap actually feeds. The "
            "model uses CMS's weights; InComm's differ for some measures and several gaps feed no 2026 Part C measure."
        )
        add_table(
            doc,
            ["CareGapCode", "HedisMeasureId", "StarsWeight", "cms_measure_id", "cms_2026_weight", "note"],
            discrepancies.to_dict("records"),
        )
    doc.add_heading("11. What remains unknown and why", level=1)
    add_bullets(
        doc,
        [
            "Measured lift and a measured dollar ROI: need randomized offers (pilot). The literature-anchored table is "
            "a scenario, not a result.",
            "Temporal reward attribution on real data: every closing claim precedes its reward case in the export; "
            "offer timestamps must be recorded before follow-up.",
            "RAF for heart failure, COPD, depression and non-stage-4 CKD: V28 factors not verified here; an upper "
            "bound from the largest verified factor is shown.",
            "InComm QBP/Star/cost figures: none provided; no InComm financial appears in this document.",
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path
