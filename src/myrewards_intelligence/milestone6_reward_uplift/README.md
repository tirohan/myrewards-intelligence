# Milestone 6: Reward Uplift / Elasticity Analysis

## Scope (from Revised Technical Milestone Plan)

This milestone focuses on causal modeling of reward effectiveness:

1. **Uplift Modeling**: Estimate the incremental effect of rewards on care gap closure
   - Treatment group: Members who received rewards
   - Control group: Members who did not receive rewards (or received different reward types)
   - Outcome: Care gap closure (using Model A's InComm-sourced labels)

2. **Elasticity Analysis**: Quantify how closure probability changes with reward value
   - Price elasticity of demand framework
   - Heterogeneous treatment effects by member characteristics

3. **Optimal Targeting**: Identify which members benefit most from rewards
   - Combine Model A's closure probability with uplift estimates
   - Expected value = P(closure | reward) × uplift × reward efficiency

## Dependencies

- Milestone 5 output: Model A's scored members with calibrated closure probabilities
- Real reward issuance data (when available from InComm)
- Currently blocked on: `sim_MemberRewardSimulations` is synthetic; real reward data needed

## Implementation Status

**Not yet implemented.** This package is scaffolded and ready for September development.

## Data Requirements

| Data | Source | Status |
|------|--------|--------|
| Member-level features | der_MemberFeatureSnapshots | Available |
| Care gap closure labels | der_MemberCareGapStatuses | Available (InComm-sourced) |
| Model A closure probabilities | Milestone 5 output | Available after M5 |
| Real reward issuance | (not yet available) | **Blocking** |
| Real reward redemption | (not yet available) | **Blocking** |

## Proposed Modules

```
milestone6_reward_uplift/
├── __init__.py
├── README.md (this file)
├── data.py          # Load M5 output + reward data
├── uplift.py        # Uplift modeling (T-learner, S-learner, etc.)
├── elasticity.py    # Price elasticity estimation
├── targeting.py     # Optimal targeting recommendations
└── reports.py       # Word deliverables
```

## Non-Negotiable Principle

Any analysis using synthetic reward data must be clearly labeled as such.
Real uplift estimates require real reward issuance data from InComm.
