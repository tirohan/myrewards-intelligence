"""Typed financial assumptions with provenance. Nothing here is an InComm financial.

Four classes, never silently mixed:
  INCOMM      read from InComm's own reference tables (e.g. StarsWeight per measure)
  PUBLIC      published by CMS/KFF, cited with year
  LITERATURE  an effect size from published trials (an external prior, NOT InComm lift and NOT the mock lift)
  RESEARCH    our own modelling assumption, always reported as a low/base/high range
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Provenance(StrEnum):
    INCOMM = "InComm-sourced"
    PUBLIC = "Public source (CMS/KFF)"
    LITERATURE = "Published evidence (external; not InComm lift, not mock lift)"
    RESEARCH = "Research assumption (not InComm financials)"


@dataclass(frozen=True)
class Assumption:
    name: str
    low: float
    base: float
    high: float
    unit: str
    provenance: Provenance
    source: str


# Average QBP increase per enrollee per year in 2026 (KFF): $381 individual plans, $318 SNPs, $466 employer.
QBP_PER_ENROLLEE_YEAR = Assumption(
    "qbp_per_enrollee_year", 318, 381, 466, "USD/enrollee/year", Provenance.PUBLIC,
    "KFF, MA Quality Bonus Program 2026 (https://www.kff.org/medicare/medicare-will-spend-more-than-13-billion-on-the-medicare-advantage-quality-bonus-program-in-2026/)",
)

# Share of a contract's enrollees in a measure's denominator. Public Medicare prevalence anchors (CMS Medicare
# Beneficiaries at a Glance, 2022 claims: diabetes 26%, high blood pressure 65%; MCBS 2023/2024 self-report: diabetes
# 32%). HEDIS denominators are narrower (age limits, claims-based identification), so the ranges reach below the
# prevalence. A LARGER share means each closure moves the measure rate less, so it LOWERS value.
ELIGIBLE_SHARE = {
    "diabetes": Assumption(
        "eligible_share_diabetes", 0.20, 0.26, 0.32, "share of enrollees", Provenance.PUBLIC,
        "CMS Medicare Beneficiaries at a Glance (26% claims-based); CMS MCBS 2023/2024 (32% self-reported)",
    ),
    "hypertension": Assumption(
        "eligible_share_hypertension", 0.50, 0.60, 0.65, "share of enrollees", Provenance.PUBLIC,
        "CMS Medicare Beneficiaries at a Glance (65% claims-based); lower bound for the narrower HEDIS denominator",
    ),
}

# The boundary probability, the cut points and the measure weights are NOT assumptions: they are measured on CMS's
# public 2026 Star Ratings data (scripts/build_star_boundary_density.py) and shown by StarValueModel.provenance_rows().


# Relative risk of closure with a financial incentive vs without. External prior for a priced scenario, because
# measured lift on the mock is zero by construction and InComm has no randomized issuance data.
#   high 1.42  pooled RR, 11 studies, cervical/breast/colorectal screening incentives, 95% CI 1.05-1.92
#              (systematic review + meta-analysis; ~1-year follow-up RR 1.57; <=1 month RR 2.69)
#   base 1.17  colorectal screening incentives, 8 RCTs: completion 30% -> 35% (95% CI 31-39%); low-quality evidence
#   low  1.00  no effect. The IDEAS RCT (diabetic eye screening, 1,051 persistent non-attenders, GBP 10 incentive)
#              found attendance 7.8% control vs 5.5% fixed vs 3.3% lottery: no benefit, possibly harm.
# Figures were retrieved via search summaries of the abstracts on 2026-10-01; the primary pages were behind a
# reCAPTCHA and not read directly. External validity to a $15-$30 pay-on-completion reward is unproven.
LITERATURE_RR = Assumption(
    "relative_risk_of_closure_with_incentive", 1.00, 1.17, 1.42, "RR (incentive vs none)", Provenance.LITERATURE,
    "Cancer-screening incentive meta-analyses (RR 1.42, 95% CI 1.05-1.92; CRC 30%->35%); IDEAS RCT diabetic eye "
    "screening (no benefit). See comment above.",
)

# Trial RRs come from low-baseline populations (screening uptake around 30%). Applying an RR to a gap whose baseline
# closure is 74% would exceed 100%, so the effect is carried on the ODDS-RATIO scale, derived at the trial's
# reference baseline and then applied to each gap's own baseline. 0.30 is the CRC meta-analysis's control rate; using
# it for the pooled RR as well is our assumption.
LITERATURE_REFERENCE_BASELINE = 0.30


def odds_ratio_from_rr(rr: float, baseline: float = LITERATURE_REFERENCE_BASELINE) -> float:
    p1 = min(baseline * rr, 0.999)
    return (p1 / (1 - p1)) / (baseline / (1 - baseline))


LITERATURE_OR = Assumption(
    "odds_ratio_of_closure_with_incentive",
    odds_ratio_from_rr(LITERATURE_RR.low),
    odds_ratio_from_rr(LITERATURE_RR.base),
    odds_ratio_from_rr(LITERATURE_RR.high),
    "odds ratio (incentive vs none)",
    Provenance.LITERATURE,
    "Derived from LITERATURE_RR at LITERATURE_REFERENCE_BASELINE so probabilities stay within [0, 1]",
)

# RAF (risk adjustment). V28 community, non-dual, aged relative factors, from CMS's CMS-HCC risk adjustment
# materials (found via search 2026-10-01): HCC37 diabetes with chronic complications 0.166; HCC327 CKD stage 4 0.514.
# Other conditions are NOT verified here and stay unavailable. (The relative-factor table fetched directly from
# cms.gov was the older V24 model, so it was not used.)
V28_VERIFIED_FACTORS = {"diabetes": 0.166, "ckd_stage_4": 0.514}
RAF_MAX_VERIFIED_FACTOR = max(V28_VERIFIED_FACTORS.values())

RAF_BASE_PAYMENT_PMPM = Assumption(
    "ma_base_payment_pmpm", 900, 1100, 1300, "USD per member per month at risk score 1.0", Provenance.RESEARCH,
    "County benchmarks are public and vary; not looked up here, so swept",
)
RAF_P_NOT_ALREADY_CAPTURED = Assumption(
    "p_condition_not_already_captured_this_year", 0.10, 0.30, 0.60, "probability", Provenance.RESEARCH,
    "HCCs must be recaptured every calendar year; share of members whose condition is not yet coded this year",
)

# Avoided medical cost arm (separate from Stars and RAF; never summed by default). Only BH 7-day follow-up has a
# published closure -> utilization link that was read here. Sources retrieved 2026-10-01 via search summaries and
# PMC (the Psychiatric Services pages returned 403/reCAPTCHA, so abstracts came through search snippets).
#   readmission rate  CMS IPF READM-30-IPF across 1,343 facilities: mean 20%, range 11%-36% (Psychiatric Services 2020,
#                     doi 10.1176/appi.ps.201900360). Used as the no-follow-up baseline: our assumption.
#   effect            low 0.00 = no causal effect (the evidence is observational; healthy-user bias is likely).
#                     base 0.10 = adjusted OR 0.88 (schizophrenia) / 0.91 (bipolar) for follow-up within 30 days vs
#                     readmission days 31-120 (Marcus 2017, doi 10.1176/appi.ps.201600498), MarketScan, not Medicare.
#                     high 0.22 = RRR 0.78 low-to-moderate-bias studies, all medical conditions, 76-study meta-analysis
#                     (PMC12587199; excludes psychiatric admissions, so an upper envelope borrowed from medicine).
#   cost              HCUP: Medicare mental/behavioral readmission $8,800 (SB189, 2012); Medicare all-cause readmission
#                     $15,500 (SB278, 2018). Base is the midpoint: our assumption. READM-30-IPF counts any unplanned cause.
# Horizon: one 30-day event, so no discounting. Perspective: the MA plan bears the cost under capitation.
AVOIDED_COST_GAPS = ("BH_POST_DISCHARGE_7D",)
BH_READMIT_RATE = Assumption(
    "bh_30d_readmission_rate_without_followup", 0.11, 0.20, 0.36, "probability", Provenance.PUBLIC,
    "CMS IPF READM-30-IPF mean 20%, facility range 11%-36% (doi 10.1176/appi.ps.201900360)",
)
BH_ODDS_REDUCTION = Assumption(
    "bh_readmission_odds_reduction_from_followup", 0.00, 0.10, 0.22, "relative odds reduction", Provenance.LITERATURE,
    "Marcus 2017 AOR 0.88/0.91 (observational); meta-analysis RRR 0.78 (medical, upper envelope); low = no causal effect",
)
BH_COST_PER_READMISSION = Assumption(
    "bh_cost_per_readmission", 8_800, 12_150, 15_500, "USD per readmission", Provenance.PUBLIC,
    "HCUP SB189 (Medicare mental/behavioral, 2012) to SB278 (Medicare all-cause, 2018); base = midpoint",
)

ALL = (
    QBP_PER_ENROLLEE_YEAR, *ELIGIBLE_SHARE.values(), LITERATURE_RR, RAF_BASE_PAYMENT_PMPM, RAF_P_NOT_ALREADY_CAPTURED,
    BH_READMIT_RATE, BH_ODDS_REDUCTION, BH_COST_PER_READMISSION,
)
