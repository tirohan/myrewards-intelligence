"""Frozen domain-analysis resolutions for Milestone 6 blockers.

Daniel directed the research team (2026-07-29 onward) not to wait on InComm
for empty or unnamed tables. Track R now runs on a research-generated
IncentiveRewards-shaped ledger. These are not InComm-confirmed production facts.
"""

from __future__ import annotations

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

# CKD_NEPHROLOGY_VISIT is excluded from the HEADLINE (InComm-sourced labels) on purpose, and the reason is
# specific: InComm's rules engine never produced a single CKD row in der_MemberCareGapStatuses, because
# ref_CareGapDefinitions.ConditionCode = 'CKD' does not match can_ConditionFlags.ConditionCode =
# 'CHRONIC_KIDNEY_DISEASE' (a naming mismatch, not missing data; 1,666 members do have the condition).
# It is NOT dropped: it runs as a separate secondary cohort on research-computed labels (milestone6.secondary),
# reported beside the headline and never mixed into it.
EXCLUDE_CARE_GAPS = ("CKD_NEPHROLOGY_VISIT",)
SECONDARY_RESEARCH_CARE_GAPS = ("CKD_NEPHROLOGY_VISIT",)

# InComm: reward funds live 365 days (stg_WalletConfigurations FundLifeSpan = 365, grace 30 days, the most
# common "Individual Reward Expiry" setting) and Daniel (2026-09-28) states every rolling period is at least a
# year. Shorter windows are sensitivity analyses only (cli/run_milestone6_sensitivity.py).
OBSERVATION_WINDOW_DAYS = 365

# Every simulation parameter, where it comes from, and how its influence is bounded.
PARAMETER_PROVENANCE: list[dict[str, str]] = [
    {"parameter": "Observation window", "value": "365 days",
     "source": "InComm-sourced",
     "evidence": "stg_WalletConfigurations FundLifeSpan=365 (138+7 incentive wallets), GracePeriod=30; "
                 "InComm feedback 2026-09-28. Follow-up is administratively censored at the data cut."},
    {"parameter": "Catalog reward amounts ($15-$30 defaults)", "value": "$15-$30 per gap",
     "source": "InComm-sourced",
     "evidence": "ref_ConditionRewardMappings Min/Default/Max (e.g. DIAB_A1C $15-$20, DIAB_EYE $20-$25)."},
    {"parameter": "Issued amount distribution within a gap", "value": "empirical pool, $1-$100",
     "source": "InComm-sourced (mapping is ours)",
     "evidence": "stg_WalletIncentivizedActivities: 6,093 real program x activity rows; activities mapped to "
                 "care gaps by keyword (export_reward_amount_pools.py). COPD falls back to catalog bounds."},
    {"parameter": "Reward payout timing", "value": "paid on completion",
     "source": "InComm-sourced",
     "evidence": "HEALTH_ACTION_REWARD cases == Closed gaps (4,726); IsRewardEligible iff Closed. "
                 "The offer, not the payout, is the causal exposure."},
    {"parameter": "Outreach coverage of open gaps", "value": "100% in live data",
     "source": "InComm-sourced",
     "evidence": "CARE_GAP_OUTREACH cases (6,943) == Open gaps; all created in one batch 2026-06-25."},
    {"parameter": "Offer probability (Bernoulli)", "value": "0.45 (swept 0.25-0.75)",
     "source": "Research assumption",
     "evidence": "No InComm table records offers. It only affects power, not validity (assignment is independent "
                 "of outcome). Swept in cli/run_milestone6_sensitivity.py."},
    {"parameter": "Offer lag after gap opens", "value": "7-45 days (swept 0-120)",
     "source": "Research assumption",
     "evidence": "InComm cases share one creation timestamp, so no lag is observable. Swept."},
    {"parameter": "Redeem / activation probability", "value": "0.988",
     "source": "InComm-derived proxy",
     "evidence": "der_OtcActivationStatuses: 98.8% of members activate; lower values swept in M7."},
]

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
            "Not production issuance. Milestone 7 may consume this only as research-generated input, never as InComm-issued lift."
        ),
    },
    "Q11_observation_window": {
        "blocker": "Post-issuance observation window was unconfirmed (earlier default 90 days was a research guess).",
        "resolution": (
            f"Observation window = {OBSERVATION_WINDOW_DAYS} days. InComm reward funds live 365 days "
            "(FundLifeSpan=365, grace 30) and InComm states every rolling period is at least a year. "
            "Track S has no IssuedAtUtc so Y is the current Closed flag. Track R mock "
            "reconstructs potential_issued_at (gap open + outreach lag) "
            "and closing-claim dates from days_since_last_qualifying_claim; Y is "
            f"closure in (t0, t0+{OBSERVATION_WINDOW_DAYS}], administratively censored at the data cut "
            "(follow-up shorter than the window is reported, not hidden). 90/180-day windows are sensitivity only."
        ),
        "claim": (
            f"Track R applies the {OBSERVATION_WINDOW_DAYS}-day window on reconstructed timestamps, "
            "not live IncentiveRewards clocks."
        ),
    },
    "Q14_amount_variability": {
        "blocker": "Whether issued amount varies within a care gap.",
        "resolution": (
            "InComm's catalog (ref_ConditionRewardMappings) has Min/Default/Max per gap, and InComm's wallet "
            "configuration (stg_WalletIncentivizedActivities, 6,093 program x activity rows) shows reward amounts "
            "for the same activity vary across programs ($1-$100, median about $20). Track S sim rows use the "
            "default only (no variation, elasticity NOT_AVAILABLE there). The Track R mock draws each offer's "
            "amount from the empirical per-gap pool, independent of outcome, so the dose-response estimator runs "
            "and its power is known. Real members are not linked to programs (can_Members.ProgramCode is null), "
            "so observational elasticity from InComm data is impossible; a randomized amount arm is the route."
        ),
        "claim": (
            "Elasticity is estimated on the research mock only (method + power), never as InComm's response curve."
        ),
    },
    "diabetes_positivity": {
        "blocker": "Diabetes cohort has an empty untreated arm in the member-broadcast contrast.",
        "resolution": (
            "That empty arm is an artifact of broadcasting a member's lifetime reward count onto every gap "
            "(M4 grouped sim rows by MemberId). At gap level, every diabetes gap has both arms. A real "
            "positivity violation (every unit in a stratum treated) is handled by: (1) never extrapolating "
            "tau into the stratum, (2) reporting it NOT_IDENTIFIED with partial-identification bounds, and "
            "(3) specifying the holdout size that would restore identification (positivity table, "
            "reports/milestone6_positivity.json)."
        ),
        "claim": "No tau is reported for a stratum that fails positivity.",
    },
    "ckd_exclusion": {
        "blocker": "CKD_NEPHROLOGY_VISIT is excluded from the headline.",
        "resolution": (
            "Deliberate, and the reason is a platform defect, not a data gap: InComm's rule never fired because "
            "ref_CareGapDefinitions.ConditionCode='CKD' does not match can_ConditionFlags 'CHRONIC_KIDNEY_DISEASE', "
            "so der_MemberCareGapStatuses has zero CKD rows. The headline uses InComm-sourced labels only. CKD "
            "runs as a separate secondary cohort on research-computed labels (1,666 rows), reported beside the "
            "headline and never mixed in."
        ),
        "claim": "CKD results are research-labeled and secondary.",
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
            "(3) 365-day Y after potential issue, (4) untreated arm among eligible Open "
            "and later-closed gaps. Production IncentiveRewards is still absent. "
            "Mock CATE is a negative-control / pipeline result (historical Y was not "
            "caused by mock T). Milestone 7 may consume this only as research-generated input, never as InComm-issued lift."
        ),
        "claim": "Production causal lift still requires a named IncentiveRewards feed.",
    },
    "who_acts": {
        "blocker": "Care-manager vs campaign capacity unknown.",
        "resolution": "Targeting is a ranked list plus Treat/Hold/Suppress volume curve.",
        "claim": "No single operating threshold.",
    },
}
