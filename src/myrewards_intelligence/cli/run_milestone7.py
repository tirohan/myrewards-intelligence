"""Milestone 7: claims-validated ROI framework on Track R research-generated lift.

Measured lift on the mock is zero by construction, so no measured dollar ROI is reported. Instead:
  * a pricing gate (no ROI on lift indistinguishable from zero)
  * causal attribution (probability of necessity) because real data has no offer-then-claim sequence
  * a separate RAF arm (never summed with Stars)
  * a priced ROI under literature-anchored lift scenarios (an external prior, labeled as such)
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from ..core.config import load_config, resolve_path, write_json
from ..core.manifest import write_run_manifest
from ..milestone7_claims_roi.attribution import attribution_summary, load_real_timing, timing_test
from ..milestone7_claims_roi.avoided_cost import combine
from ..milestone7_claims_roi.literature import literature_roi
from ..milestone7_claims_roi.projections import member_roi, monte_carlo, tornado
from ..milestone7_claims_roi.raf import raf_table
from ..milestone7_claims_roi.reports import write_roi_report
from ..milestone7_claims_roi.roi import (
    ASSUMPTION_TAG,
    load_lift_uncertainty,
    load_scores,
    scenarios,
    validate_claims,
)
from ..milestone7_claims_roi.stars import StarValueModel, weight_discrepancies
from ..milestone7_claims_roi.validation import validate_closures


def main() -> int:
    settings = load_config()
    cfg = settings.milestone7
    m6 = settings.milestone6
    p = lambda rel: str(resolve_path(rel))  # noqa: E731
    df = load_scores(p(cfg.scores_path), p(cfg.reward_amount_pools_path))
    ate, se = load_lift_uncertainty(p(cfg.m6_results_path))
    with open(p(cfg.m6_results_path), encoding="utf-8") as f:
        m6_results = json.load(f)
    model = StarValueModel.from_files(p(cfg.star_density_path), p(cfg.gap_measure_path))
    discrepancies = weight_discrepancies(p(cfg.star_weights_path), p(cfg.cms_weights_path), model.mapping.reset_index())
    claims = validate_closures(p(cfg.claims_validation_path))
    central = m6.mock_redeem_probability

    # --- scenarios under measured (mock) lift, gated
    frames = []
    for redeem in [central, *cfg.redeem_sensitivity]:
        f = scenarios(df, cfg.treat_fractions, cfg.value_per_closure_usd, redeem, ate, se, model)
        f.insert(0, "redeem_p", redeem)
        frames.append(f)
    all_scen = pd.concat(frames, ignore_index=True)
    table = frames[0]
    v = validate_claims(df, ate, se, claims)
    priced = v["lift_status"] == "LIFT_DISTINGUISHABLE"
    counts = df["CareGapCode"].value_counts().to_dict()
    torn = tornado(model, counts)
    mc_df = pd.DataFrame(
        [
            {"fraction_treated": r["fraction_treated"]}
            | monte_carlo(ate, r["n_treated"], r["reward_cost_expected_usd"], se, model, counts)
            for r in table.to_dict("records")
        ]
    )

    # --- attribution (causal; real data has no offer-then-claim sequence)
    attribution = attribution_summary(
        m6_results["treated_closure_rate"], m6_results["control_closure_rate"], ate, se,
        load_real_timing(p(cfg.attribution_timing_path)),
        timing_test(p(cfg.mock_eligibility_path), m6.observation_window_days),
    )

    # --- RAF arm (separate; never summed with Stars)
    raf = raf_table(p(cfg.claims_dx_path))

    # --- priced ROI under literature-anchored lift (external prior)
    baselines = {r["care_gap_code"]: r["control_closure"] for r in m6_results["by_care_gap"]}
    lit_gap, lit_prog = literature_roi(df, baselines, model, central)
    avoided = combine(lit_gap)  # separate arm: Stars-only vs Stars + avoided medical cost, never summed by default

    for name, frame in (
        ("m7_avoided_cost_arm", avoided),
        ("m7_roi_scenarios", all_scen), ("m7_tornado", torn), ("m7_monte_carlo", mc_df), ("m7_raf_arm", raf),
        ("m7_literature_roi_by_gap", lit_gap), ("m7_literature_roi_program", lit_prog),
        ("m7_star_weight_discrepancies", discrepancies),
    ):
        frame.to_csv(resolve_path(f"models/{name}.csv"), index=False)
    member_roi(df, model, central, priced).to_csv(resolve_path("models/m7_member_roi.csv"), index=False)

    write_json(
        {
            "validation": v,
            "scenarios": all_scen.to_dict("records"),
            "attribution": attribution,
            "raf_arm": raf.to_dict("records"),
            "literature_program_roi": lit_prog.to_dict("records"),
            "value_assumption": (
                f"{ASSUMPTION_TAG}: {cfg.value_per_closure_usd}"
                if cfg.value_per_closure_usd is not None
                else "NOT_AVAILABLE"
            ),
            "m7_input_basis": "Research-generated IncentiveRewards mock; not InComm-issued",
        },
        "reports/milestone7_results.json",
    )

    def prog(rr, val):
        r = lit_prog[(lit_prog["rr_case"] == rr) & (lit_prog["value_case"] == val)].iloc[0]
        return f"${r['net_total_usd']:,.0f} (ROI {r['roi']:.1f}x)"

    be = lit_gap[(lit_gap["rr_case"] == "base")].groupby("value_case")["break_even_rr"].agg(["min", "max"])
    real_t = attribution["real_data_timing"]
    crc = lit_gap[(lit_gap["rr_case"] == "base") & (lit_gap["value_case"] == "base")]
    pays = ", ".join(crc.loc[crc["net_per_offer_usd"] > 0.5, "care_gap_code"]) or "none"
    not_pays = ", ".join(crc.loc[crc["net_per_offer_usd"] <= 0.5, "care_gap_code"]) or "none"
    no_star = ", ".join(crc.loc[crc["stars_value_is_zero"], "care_gap_code"]) or "none"
    first = table.iloc[-1]
    ac = avoided[(avoided["rr_case"] == "base") & avoided["avoided_cost_usd"].notna()]
    ac_line = "; ".join(
        f"{r.care_gap_code} {r.value_case}: avoided ${r.avoided_cost_usd:,.0f}/closure, break-even RR "
        f"{r.break_even_rr:.2f} (Stars only) -> {r.break_even_rr_with_avoided:.2f} (with avoided cost)"
        for r in ac.itertuples()
    )
    raf_line = ", ".join(f"{r.care_gap_code}: {r.status.split(':')[0]}" for r in raf.itertuples())
    resolve_path("reports/Milestone7_Status.md").write_text(
        "# Milestone 7 Status\n\n"
        "**Input basis:** Research-generated IncentiveRewards mock (not InComm-issued)\n"
        f"**Measured lift (mock):** {v['lift_status']}; ATE {ate:.4f} (SE {se:.4f}); minimum detectable {v['minimum_detectable_effect']:.4f}\n"
        f"**Claims validation:** {claims.get('status')}; all checks pass: {claims.get('all_checks_pass')}; "
        f"{claims.get('gaps_sharing_a_claim')} gap rows share a claim (basis: synthetic MOCK-CLM ids)\n"
        f"**Attribution:** real data {real_t.get('closures_after_an_offer')} of {real_t.get('closed_gaps')} closures follow an "
        f"offer (impossible temporally). Causal: probability of necessity {attribution['pn_lower']:.2f} to "
        f"{attribution['pn_upper']:.2f} on the mock (monotone {attribution['pn_monotone']:.2f}).\n"
        f"**RAF arm (separate, never summed):** {raf_line}\n"
        f"**Avoided medical cost arm (separate; BH follow-up only, other gaps NOT_MODELLED):** {ac_line}\n"
        f"**Stars->QBP value per closure (CMS public data; mean over treated mix, low / base / high):** ${table['stars_value_per_closure_usd_low'].iloc[-1]:,.0f} / "
        f"${table['stars_value_per_closure_usd_base'].iloc[-1]:,.0f} / ${table['stars_value_per_closure_usd_high'].iloc[-1]:,.0f}\n"
        f"**Lift needed per offered member for Stars->QBP value to cover reward cost (offer to all; low / base / high value):** "
        f"{first['required_lift_pp_for_breakeven_low']:.1f} / {first['required_lift_pp_for_breakeven_base']:.1f} / "
        f"{first['required_lift_pp_for_breakeven_high']:.1f} percentage points\n"
        f"**Gaps that pay under the CRC-anchored lift at base value:** {pays}. **Do not:** {not_pays}. "
        f"**No Star measure (Stars value is exactly 0):** {no_star}. Stars/QBP only in this line; avoided medical cost is a separate arm (BH only).\n"
        f"**Priced ROI (literature-anchored lift, NOT InComm lift; offer to all; conservative-value corner / base value):** "
        f"no effect: {prog('low', 'low')} / {prog('low', 'base')}; "
        f"CRC-anchored lift: {prog('base', 'low')} / {prog('base', 'base')}; "
        f"pooled-screening lift: {prog('high', 'low')} / {prog('high', 'base')}. "
        "The favourable-value corner stacks every favourable bound and is an upper envelope only (see CSV).\n"
        f"**Break-even relative risk by value case (range across gaps; program pays if the true RR exceeds it):** "
        + "; ".join(f"{k}: {r['min']:.2f}-{r['max']:.2f}" for k, r in be.iterrows())
        + "\n\nMeasured ROI stays unpriced until a pilot gives distinguishable lift. The literature-anchored table is a "
        "labeled scenario, not a result.\n",
        encoding="utf-8",
    )
    write_roi_report(
        v, all_scen, resolve_path("reports/Milestone 7 ROI Framework.docx"), torn, mc_df,
        attribution=attribution, raf=raf, lit_gap=lit_gap, lit_prog=lit_prog, avoided=avoided,
        star_rows=model.provenance_rows(), discrepancies=discrepancies,
    )
    write_run_manifest(
        "milestone7",
        cfg.model_dump(),
        [resolve_path(cfg.scores_path), resolve_path(cfg.m6_results_path), resolve_path(cfg.claims_dx_path)],
        [resolve_path("models/m7_roi_scenarios.csv"), resolve_path("reports/milestone7_results.json")],
        {k: v[k] for k in ("lift_status", "ate_mean", "ate_se", "minimum_detectable_effect")},
    )
    print(table.drop(columns=["TreatmentSourceBasis", "value_basis", "stars_value_basis", "financial_slots_status"]).to_string(index=False))
    print(lit_prog.drop(columns=["basis"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
