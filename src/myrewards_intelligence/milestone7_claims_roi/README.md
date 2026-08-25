# Milestone 7: Claims-Validated ROI Intelligence

## Scope (from Revised Technical Milestone Plan)

This milestone translates predictive scores and uplift estimates into financial impact:

1. **ROI Framework**: Connect care gap closure to financial outcomes
   - Quality Bonus Payment (QBP) impact per closed gap
   - Star Rating implications
   - Risk Adjustment Factor (RAF) considerations

2. **Claims Validation**: Verify closures through actual claims data
   - CPT/HCPCS code matching per care gap
   - Timing validation (within measurement period)
   - Attribution to reward-influenced behavior

3. **Financial Projections**: Estimate program-level ROI
   - Per-member ROI given Model A closure probability
   - Program-level expected value
   - Sensitivity analysis on key assumptions

## Dependencies

- Milestone 5: Model A closure probabilities
- Milestone 6: Uplift estimates (incremental value of rewards)
- QBP/Star Rating/RAF assumptions from InComm (not yet available)

## Implementation Status

**Not yet implemented.** This package is scaffolded and ready for development.

## Data Requirements

| Data | Source | Status |
|------|--------|--------|
| Model A closure probabilities | Milestone 5 | Available after M5 |
| Uplift estimates | Milestone 6 | Available after M6 |
| QBP assumptions | InComm | **Not yet provided** |
| Star Rating weights | InComm | **Not yet provided** |
| RAF coefficients | InComm | **Not yet provided** |
| Reward program costs | InComm | **Not yet provided** |

## Proposed Modules

```
milestone7_claims_roi/
├── __init__.py
├── README.md (this file)
├── assumptions.py   # Financial assumption management (flagged NOT_AVAILABLE)
├── validation.py    # Claims-based closure validation
├── roi.py           # ROI calculation framework
├── projections.py   # Program-level financial projections
└── reports.py       # Word deliverables
```

## Non-Negotiable Principle

**Never invent a specific business figure (a dollar amount, a percentage, a coefficient)
and present it as InComm's real assumption.**

If QBP, Star Rating, or RAF values are not available:
- Build the structure that would hold them
- Flag them explicitly as `NOT_AVAILABLE`
- Do not make up placeholder numbers
