# Milestone 7: Claims-Validated ROI Framework

Translates uplift into a financial screening range. Runs on the **research-generated** Track R lift
(`TreatmentSourceBasis = Research-generated IncentiveRewards mock`); it never uses Track S and never imports
Milestone 5. InComm cannot share data, so no InComm financial appears anywhere: every dollar input comes from
public CMS/KFF data, published trials, InComm's own reference tables, or a labeled research range (`assumptions.py`),
reported as low/base/high.

Run: `python -m myrewards_intelligence.cli.run_milestone7` (or `myrewards-m7`, after M6).

## What it does
| Module | Purpose |
|---|---|
| `assumptions.py` | Typed assumptions with provenance (QBP $381/enrollee/yr from KFF 2026; contract-level inputs as research ranges) |
| `stars.py` | `StarValueModel`: Stars -> QBP value of one closure = QBP$ x rho x pass-through / eligible share. rho is **measured on CMS's public 2026 Star Ratings** (`scripts/build_star_boundary_density.py`: cut points, weights, published ratings of ~520 MA contracts; enrollment cancels). Gaps with no 2026 Part C Star measure (COPD spirometry, BH follow-up, CKD visit) are exactly 0. Value is split across gaps that roll up to one measure (DIAB_A1C_TEST + DIAB_BLOOD_SUGAR -> C12). InComm's StarsWeight differs from CMS's for eye exam and KED (3 vs 1); CMS's is used and the discrepancy is reported |
| `roi.py` | Scenarios (top fraction by uplift): pay-on-completion cost, redemption sensitivity, required lift for break-even, pricing gate |
| `projections.py` | Per-member ROI, tornado sensitivity, Monte Carlo over lift and assumptions |
| `attribution.py` | Causal reward attribution (probability of necessity, Tian-Pearl bounds) + offer-timing test on the research ledger |
| `avoided_cost.py` | Avoided medical cost arm (BH 7-day follow-up -> 30-day readmission only; other gaps NOT_MODELLED). Side by side with Stars, never summed by default; low case is zero effect |
| `raf.py` | RAF arm: risk-adjustment-eligible encounter x payment-HCC diagnosis x not-yet-captured x verified V28 factor. Separate from Stars, never summed |
| `literature.py` | Priced ROI under literature-anchored lift (odds ratios from published incentive trials) and break-even relative risk per gap |
| `validation.py` | Claims validation of closures (claim found, member, CPT in definition, date, window, shared claims) |
| `reports.py` | Watermarked Word report |

## Rules
- **Pricing gate:** no ROI is reported unless the population lift is distinguishable from zero. Per-member net
  value is NaN until then; cost, closure value and break-even uplift are always given.
- **Cost model:** InComm rewards are paid on completion (HEALTH_ACTION_REWARD cases == Closed gaps), so expected
  cost is `amount x P(close | offered) x P(redeem)`. Paying every offer is kept only as an upper bound.
- **Monte Carlo** is centred on the population ATE, not the in-sample ranking gain of a top fraction (which is
  optimistic under selection on noise).
- **Attribution is causal, not temporal.** Real data: all 4,726 closing claims precede their reward case, so no
  closure follows an offer. Attribution therefore uses the probability of necessity (share of treated closers whose
  closure the offer caused); its complement is the windfall share of pay-on-completion payouts. Offer timestamps are
  required for temporal attribution (pilot design).
- **RAF is a separate arm.** A closure moves RAF only via a risk-adjustment-eligible face-to-face encounter whose
  diagnosis maps to a payment HCC not already coded this year. For lab/diagnostic gaps the value is zero; for visit
  gaps it is zero (anxiety is not a payment HCC) or NOT_AVAILABLE with an upper bound (heart-failure V28 factor not
  verified). Only V28 factors verified in `assumptions.py` are used. Rewards must never depend on a diagnosis.
- **Priced ROI is a labeled scenario.** Measured lift on the mock is zero by construction and InComm has no
  randomized issuance, so no measured dollar ROI exists. `literature.py` prices the program under effect sizes from
  published incentive trials (RR 1.00 / 1.17 / 1.42, carried as odds ratios so probabilities stay in [0, 1]) and
  reports the break-even relative risk. These are external priors (provenance LITERATURE), not InComm lift. IDEAS
  (diabetic eye screening) found no benefit, so "no effect" is a live scenario. The favourable-value rows stack every
  favourable bound and are an upper envelope.

## Inputs
`models/scored_uplift_track_r.csv`, `reports/milestone6_results.json`, `config/cms_star_boundary_density.csv`,
`config/cms_2026_measure_weights.csv`, `config/care_gap_cms_measure.csv`, `config/hedis_star_weights.csv` (InComm
metadata, discrepancy table only), `config/reward_amount_pools.csv`, `models/incentive_rewards_mock_eligibility.csv`, and from
`export_claims_validation.py`: `models/claims_validation.csv`, `closing_claim_diagnoses.csv`,
`attribution_timing.json`.

## Outputs
`models/m7_roi_scenarios.csv`, `m7_member_roi.csv`, `m7_tornado.csv`, `m7_monte_carlo.csv`, `m7_raf_arm.csv`,
`m7_literature_roi_by_gap.csv`, `m7_literature_roi_program.csv`,
`reports/milestone7_results.json`, `reports/Milestone7_Status.md`, `reports/Milestone 7 ROI Framework.docx`,
and a manifest under `runs/`.

## Non-negotiable principle
Never invent a business figure and present it as InComm's. Where InComm has not provided a value, the structure
stays and the value is a labeled research range or NOT_AVAILABLE.
