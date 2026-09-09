"""Frozen domain-analysis resolutions for Milestone 6 blockers.

Daniel directed the research team (2026-07-29 onward) not to wait on InComm
for empty or unnamed tables. Track R now runs on a research-generated
IncentiveRewards-shaped ledger. These are not InComm-confirmed production facts.
"""

from __future__ import annotations

from ..core.evidence import NOT_AVAILABLE

# Catalog defaults from ref_ConditionRewardMappings as recorded in
# outputs/csv/care_gap_evidence_dictionary.csv (InComm platform mappings).
# Sim rows use DefaultRewardAmount only — min/max are catalog bounds, not issued variation.
CATALOG_DEFAULT_AMOUNTS: dict[str, float] = {
    "DIAB_A1C_TEST": 20.0,
    "DIAB_EYE_EXAM": 25.0,
    "DIAB_NEPHRO_SCREEN": 20.0,
    "DIAB_BLOOD_SUGAR": 15.0,
    "HTN_PCP_FOLLOWUP": 15.0,
    "COPD_SPIROMETRY": 20.0,
    "BH_POST_DISCHARGE_7D": 30.0,
    "CKD_NEPHROLOGY_VISIT": 18.0,
}

EXCLUDE_CARE_GAPS = ("CKD_NEPHROLOGY_VISIT",)

OBSERVATION_WINDOW_DAYS = 90

DOMAIN_RESOLUTIONS: dict[str, dict[str, str]] = {
    "Q5_issuance_ledger": {
        "blocker": "Production issuance/redemption table not named; IncentiveRewards not implemented.",
        "resolution": (
            "InComm 2026-07-20 Q4: sim_MemberRewardSimulations is a dashboard "
            "what-if tool; real issuance is a 3rd-party IncentiveRewards feed "
            "that is not implemented. Track S reconstructs gap-level assignment "
            "from the live-validated rule IsRewardEligible iff Closed. "
            "Track R generates an IncentiveRewards-shaped ledger on real members, "
            "care gaps, and catalog amounts ($15–$30) with Bernoulli capacity "
            "(default p=0.45) among gaps still open at a lagged outreach date. "
            "TreatmentSourceBasis=Research-generated IncentiveRewards mock. "
            "Never labeled InComm-issued."
        ),
        "claim": (
            "Track R interface is complete on a research-generated ledger. "
            "Not production issuance. Milestone 7 must not consume."
        ),
    },
    "Q11_observation_window": {
        "blocker": "InComm answered 1-year analysis lookback, not post-issuance window.",
        "resolution": (
            f"Default observation window = {OBSERVATION_WINDOW_DAYS} days. Track S "
            "has no IssuedAtUtc so Y is the current Closed flag. Track R mock "
            "reconstructs potential_issued_at (gap open + 7–45 day outreach lag) "
            "and closing-claim dates from days_since_last_qualifying_claim; Y is "
            "closure in (t0, t0+90]."
        ),
        "claim": (
            "Track R applies the 90-day window on reconstructed timestamps, "
            "not live IncentiveRewards clocks."
        ),
    },
    "Q14_amount_variability": {
        "blocker": "Whether issued amount varies within a care gap.",
        "resolution": (
            "Catalog defaults differ across gaps ($15–$30) but both Track S sim "
            "and the Track R mock use DefaultRewardAmount only. Within-gap issued "
            f"amount does not vary. Elasticity status = {NOT_AVAILABLE}."
        ),
        "claim": "Do not estimate dP(close)/d$.",
    },
    "as_of_vs_closing_claim": {
        "blocker": "Issued-at relative to closing claim is unconfirmed.",
        "resolution": (
            "ISS-008 (live extract): every can_ClinicalClaimLines ServiceFrom is "
            "on or before 2026-06-23; earliest der_RecommendationCases.CreatedAtUtc "
            "is 2026-06-25. HAR / sim TriggeredAtUtc is after every claim. "
            "Track R mock reconstructs outreach lag from gap tenure so issued_at "
            "can precede a later closing claim; rows already closed at t0 are "
            "ineligible (real-world: do not issue after close)."
        ),
        "claim": "HAR time order stays reversed. Mock timestamps are research reconstructions.",
    },
    "assignment_mechanism": {
        "blocker": "No campaign A/B or holdout named.",
        "resolution": (
            "Track S assignment = InComm ML HEALTH_ACTION_REWARD eligibility "
            "(Closed ⇔ eligible ⇔ HAR case), so T equals Y. Track R mock "
            "assignment is capacity-constrained Bernoulli among gaps still open "
            "at t0, independent of later Closed. That is a research holdout "
            "shape, not an InComm campaign A/B."
        ),
        "claim": "Track S T is a function of Y. Track R mock T is independent of later Closed.",
    },
    "incentive_rewards_holdout": {
        "blocker": "Observational CATE cannot be identified while T is determined by Y.",
        "resolution": (
            "Track R mock: (1) dated gap-level issued_at, (2) drop issue-after-close, "
            "(3) 90-day Y after potential issue, (4) untreated arm among eligible Open "
            "and later-closed gaps. Production IncentiveRewards is still absent. "
            "Mock CATE is a negative-control / pipeline result (historical Y was not "
            "caused by mock T). Milestone 7 must not consume."
        ),
        "claim": "Production causal lift still requires a named IncentiveRewards feed.",
    },
    "who_acts": {
        "blocker": "Care-manager vs campaign capacity unknown.",
        "resolution": "Targeting is a ranked list plus Treat/Hold/Suppress volume curve.",
        "claim": "No single operating threshold.",
    },
}
