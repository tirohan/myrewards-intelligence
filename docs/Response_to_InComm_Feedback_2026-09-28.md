# Milestone 6: Response to InComm Feedback of 28 September 2026

DRAFT for review before sending. Prepared by the KSU mHealth research team.

Thank you, Daniel. Your four points were fair and one of them (the 90-day window) was a real error. Each is
answered below with the evidence and what we changed. Since InComm cannot share further data, every missing input is
generated from your own tables, public CMS data, or published evidence, and each is labeled by source. Everything is
a research construction and is labeled that way in every file; none of it is presented as InComm-issued lift.

## 1. Simulation parameters

You were right that several parameters had no InComm basis. We re-audited every one against your database.

| Parameter | Status | Evidence |
|---|---|---|
| Observation window | **Was wrong; now 365 days** | `stg_WalletConfigurations`: reward `FundLifeSpan` = 365 days with a 30-day grace period (the most common setting). Matches your statement that rolling periods are at least a year. 90 and 180 days are now sensitivity checks only. |
| Catalog amounts ($15 to $30) | **Sourced from InComm** | `ref_ConditionRewardMappings` Min / Default / Max per care gap (for example DIAB_A1C $15 to $20, DIAB_EYE $20 to $25, BH follow-up $25 to $30). We had only used the default. |
| Issued amount within a gap | **Now sourced from InComm** | `stg_WalletIncentivizedActivities` (6,093 program x activity rows): amounts for the same activity range from $1 to $100 across programs, median about $20. The mock now draws each offer's amount from that empirical per-gap distribution. |
| Offer probability (45%) and lag (7 to 45 days) | **Assumptions, not sourced** | No InComm table records offers, and every case shares one creation timestamp (2026-06-25), so no lag is observable. We swept offer probability 25 / 45 / 75% and lag 0 / 7 to 45 / 45 to 120 days (9 combinations): the result is null in all of them (effect within plus or minus 1.7 points, never distinguishable from zero). These parameters affect statistical power only, not validity. |

Every parameter and its source is now listed in the package (`PARAMETER_PROVENANCE`) and in the domain
resolutions report.

## 2. Elasticity ("does reward value affect closure?")

Before: no answer, because every offer used one amount per gap. Now there is a preliminary, honestly bounded
answer. Offered amounts vary within each gap (sourced as above) and the estimator regresses closure on amount
with care-gap fixed effects. On the research mock the estimate is +0.24 points per +$10 (standard error 0.46),
and the smallest slope detectable with the current 5,217 offers is about 1.3 points per $10. The mock has no true
effect by construction, so this demonstrates that the method works (it recovers known effects and a known null in
our tests) and how large a response would need to be to be seen. It is **not** InComm's response curve. Your
members are not linked to wallet programs (`can_Members.ProgramCode` is empty), so the real curve cannot be
estimated from existing data. A small randomized pilot with an amount arm inside your catalog Min/Max would answer
it; the design and sample sizes are in `docs/Reward_Pilot_Experiment_Design.md`.

One finding that changes how we model cost: your rewards are paid on completion (`HEALTH_ACTION_REWARD` cases
equal Closed gaps, 4,726 each). A payout is therefore a consequence of closure, which is why the real feed could
not show causal lift. The causal question is about the **offer**, and cost is incurred only when a member
completes the action. Our ROI model now charges cost that way.

## 3. Diabetes cohort with an empty untreated arm

This is an artifact, not a missing arm. The earlier analysis copied a member's lifetime reward count onto every
one of their gaps, so any member with a reward looked treated on all gaps, and diabetes ended up with 1,667 of
1,667 treated. At gap level, every diabetes gap has both arms (for example DIAB_A1C_TEST: 752 offered, 915 not).
For the case where a stratum really has no untreated arm, the rule is now explicit and automatic: we never
extrapolate an effect into it; we report it as not identified together with bounds on the effect (without
assumptions, and under the assumption that a reward cannot lower closure); and we report the size of the holdout
that would restore identification. This is in the new positivity table (the `positivity` field of the results
file).

## 4. CKD_NEPHROLOGY_VISIT

Yes, the exclusion was deliberate, but "no ledger data" was the wrong description. The real reason is that
InComm's rules engine never produced a CKD row: `ref_CareGapDefinitions.ConditionCode` is `CKD` while
`can_ConditionFlags` uses `CHRONIC_KIDNEY_DISEASE`, so `der_MemberCareGapStatuses` has zero CKD rows even though
1,666 members have the condition. We no longer drop it silently. It runs as a separate secondary cohort on
research-computed labels (1,650 analyzable gaps; closure 31.6% offered vs 35.2% not; effect not distinguishable
from zero), reported beside the headline and never mixed into it. The headline stays InComm-sourced labels only.

## Financial inputs (generated from public data, not requested from InComm)
The ROI inputs InComm does not hold are now measured on CMS's public 2026 Star Ratings data (770 contracts: measure
rates, cut points, weights, published ratings). Our engine reproduces 98% of published Part C summary ratings within
half a star. This corrected two things in our earlier work: InComm's `StarsWeight` metadata (3 for eye exam and for
KED) differs from CMS's 2026 weights (1 each), and COPD spirometry and BH follow-up feed no 2026 Part C Star measure,
so their Stars value is zero. With the corrected value (about $157 per closure at base, $102 to $260 across public
ranges), Stars/QBP value covers the reward cost only if the offer lifts closure by about 5.5 percentage points
(3.3 to 8.4). Measured lift on the generated data is not distinguishable from zero by design, so no dollar ROI is
claimed; priced results are labeled scenarios under published incentive-trial effect sizes.

## Assumptions we are making on your behalf (please correct us if any is wrong; none blocks us)
1. A reward is paid only after the closing claim is validated (consistent with your data: reward cases equal closed gaps).
2. Production program amounts (median near $10 for HbA1c-related activities) and catalog defaults ($20) differ; we
   sample from the production distribution and report the catalog default alongside.
3. The CKD `ConditionCode` mismatch (`CKD` vs `CHRONIC_KIDNEY_DISEASE`) is a platform defect we worked around, not
   something we need you to fix.
