# Milestone 6: Reward Uplift / Elasticity Analysis

**Plan:** [`MILESTONE6_PLAN.md`](../../../../../MILESTONE6_PLAN.md) at the
workspace root (same location as `MILESTONE5_PLAN.md`). Read that before
implementing. This README is the package stub; the plan is the contract.

## What this milestone answers

Milestone 5 estimates **P(close)**. Milestone 6 estimates the **incremental
effect of issuing a reward** (CATE). Targeting high Model A scores pays
members who would have closed anyway. Milestone 7 ROI needs lift, not
propensity.

## Two tracks (disclosed on every artifact)

| Track | Treatment source | What we may claim |
|---|---|---|
| **Track S** | HAR / `sim_MemberRewardSimulations` assignment reconstruction | Method / pipeline only. Never partner-facing lift. Never M7 input. |
| **Track R (mock)** | Research-generated IncentiveRewards-shaped ledger (real members, gaps, catalog $15–$30) | Pipeline CATE on independent assignment. **Not** InComm-issued. **Must not** feed Milestone 7. |
| **Track R (production)** | Named IncentiveRewards / InComm issuance ledger | Only this `TreatmentSourceBasis=InComm-issued` may feed Milestone 7. Not available. |

`TreatmentSourceBasis` is `Research-simulated`, `Research-generated IncentiveRewards mock`, or `InComm-issued`. Daniel does not have IncentiveRewards; Track R currently generates the mock.

## M5 constraints that are now design inputs

- Never use qualifying-claim features (`qualifying_claim_count` *is* the
  closing event).
- Do **not** use leaky Model A (reward features, AUC ~0.94) as μ0 or the
  targeting propensity axis. Use the **ablation Model A (~0.75 AUC)** or a
  time-safe refit.
- Treatment is a **dated gap-level issued event**, not lifetime reward
  counts. Headline **Y** is InComm-sourced closure only.
- Grouped member split; seeds `[42, 123, 456, 789, 2026]`.
- This package **must not import** `milestone5_member_impact_scoring`. Read
  score files/tables. `core/` stays milestone-agnostic.
- Naive treated-vs-untreated is invalid: never-rewarded M5 holdout members
  had 0% closure (no overlap).

## Status

**Track R implemented on a research-generated IncentiveRewards mock** (config
`track: R`, `generate_incentive_rewards_mock: true`). Production IncentiveRewards
is still absent. Scores are tagged `Research-generated IncentiveRewards mock`;
`m7_may_consume=false`. Track S remains available by setting `track: S`.

```
python -m myrewards_intelligence.cli.run_milestone6
```

## Modules

```
milestone6_reward_uplift/
├── evidence.py          # TreatmentSource, AssignmentMechanism
├── domain_resolutions.py
├── incentive_rewards_mock.py # Research-generated IncentiveRewards ledger
├── data.py              # Track S assignment + Track R ledger adapter
├── treatment.py         # T at (member, gap); ledger join for Track R
├── covariates.py
├── overlap.py           # propensity, ESS, common support
├── ablation.py          # no-reward P(close); never imports Milestone 5
├── uplift.py            # S/T/X learners + AIPW ATE
├── evaluate_uplift.py   # Qini/AUUC, placebo, grouped+stratified seeds
├── elasticity.py        # dose-response or NOT_AVAILABLE
├── targeting.py         # quadrants, ranked list, volume curve
├── plots.py             # propensity, gap contrast, τ volume, quadrants
├── derived_layer.py     # DDL + scored dataframe
└── reports.py           # six Word deliverables + shareable copy
```

CLI: `python -m myrewards_intelligence.cli.run_milestone6`

## Non-negotiable

Do not mix Model B labels into headline AUUC. Do not invent QBP dollars.
Do not treat `sim_` rows as production treatment.
