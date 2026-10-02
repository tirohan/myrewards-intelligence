# Reward Pilot: Experiment Design (DRAFT for internal review, not sent)

**Purpose.** InComm's rewards are paid on completion (HEALTH_ACTION_REWARD cases equal Closed gaps), so a real
ledger of payouts is a consequence of closure and cannot show whether a reward *changes* behavior. Only
randomizing who is **offered** the reward (and at what amount) can. InComm cannot share data, so no milestone
depends on this: all results stand on generated and public data. This is the spec for the one experiment that would
replace the labeled scenarios with a measured effect, if a partner program ever runs it.

## Design
- **Unit:** (member, care gap) opportunity open at t0. Exclude gaps already closed at t0.
- **Arms:** (A) no offer (holdout), (B) offer at the care gap's catalog default, (C) optional: offer at a different
  amount inside InComm's catalog Min/Max for that gap (e.g. DIAB_A1C $15 vs $20). Arm C is the only route to
  dP(close)/d$; without it elasticity stays unidentified on real data (members are not linked to programs, so
  cross-program amount differences cannot be joined to individuals).
- **Assignment:** independent, stratified by care gap, persisted for ALL arms with seed and `AssignedAtUtc`
  (current feeds lose the control arm). This also guarantees positivity: every stratum has an untreated arm.
- **Outcome / window:** closing claim with ServiceFrom in (t0, t0+365 days], linked through the claim-to-gap
  bridge. Reward funds live 365 days (FundLifeSpan=365, grace 30), and InComm states rolling periods are at
  least a year. Primary metric: 365-day closure rate difference. Shorter windows are secondary.
- **Pre-registered subgroups:** care gap, age band, spend tier only.
- **Compliance check before launch:** whether rewards-and-incentives program rules allow withholding an offer
  from a randomized control group (not verified here; the running partner's compliance team must confirm).

## Sample size (two-sided alpha 0.05, power 0.8, per arm), 365-day window
Baselines are the research mock's control-arm 365-day closure by gap. The mock has no true effect, so these are
baseline levels for planning, not InComm results.

| Gap | Baseline | +3pp | +5pp | +10pp |
|---|---|---|---|---|
| Pooled | 38.4% | 4,183 | 1,517 | 385 |
| DIAB_A1C_TEST | 19.5% | 2,888 | 1,075 | 289 |
| DIAB_BLOOD_SUGAR | 30.7% | 3,806 | 1,392 | 360 |
| DIAB_EYE_EXAM | 30.0% | 3,764 | 1,377 | 356 |
| DIAB_NEPHRO_SCREEN | 28.6% | 3,668 | 1,344 | 349 |
| HTN_PCP_FOLLOWUP | 26.0% | 3,476 | 1,279 | 335 |
| COPD_SPIROMETRY | 59.3% | 4,155 | 1,481 | 359 |
| BH_POST_DISCHARGE_7D | 74.5% | 3,179 | 1,111 | 255 |

Pooled, detecting +1pp needs 37,317 per arm and +2pp needs 9,371 per arm. A 365-day window has a high baseline,
so the absolute effect that can be detected is larger than at 90 days even though the relative effect is
smaller. The pilot should be sized for the smallest lift that would make the program pay for itself. With the Star value
measured on CMS public data (about $157 per closure at base), that break-even lift is about 5.5 percentage points
offered-to-all (3.3 to 8.4 across public ranges), so a pooled pilot of roughly 1,500 members per arm (+5pp) is
enough to settle the question, not the tens of thousands a +1pp target would need.

## Elasticity (arm C) sensitivity
On the research mock the within-gap dose-response estimator has a minimum detectable slope of about 1.3pp of
closure per +$10 at the current n (5,217 offers). A real pilot needs an amount arm with at least that many
offers per amount level to detect a slope that size.

## What the pilot would unlock
Real tau for Milestone 6 (without a positivity gap), a Milestone 7 ROI with distinguishable lift, and (with
arm C) a real elasticity. Method validity is already supported by the semi-synthetic tests
(`tests/milestone6/test_recovery.py`): known effects, a known heterogeneous effect, and a known dose-response
are recovered at the current sample size when they are large.
