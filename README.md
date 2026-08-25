# MyRewards Intelligence

Predictive scoring, claims validation, and ROI intelligence for InComm MyRewards Recommendation System.

This repository extends InComm's MyRewards platform with:

- **Milestone 5**: Member Impact Scoring — supervised predictive model estimating care-gap closure probability
- **Milestone 6**: Reward Uplift / Elasticity Analysis — causal impact of rewards on member behavior
- **Milestone 7**: Claims-Validated ROI Intelligence — financial impact estimation

## Project Structure

```
myrewards-intelligence/
├── src/myrewards_intelligence/
│   ├── core/                  # Shared infrastructure (database, config, evidence tracking)
│   ├── milestone5_*/          # Member Impact Scoring model
│   ├── milestone6_*/          # Reward Uplift (stub, ready for implementation)
│   ├── milestone7_*/          # Claims ROI (stub, ready for implementation)
│   └── cli/                   # Entry points
├── tests/                     # pytest test suite
├── config/                    # Configuration files
├── data/                      # Input data (gitignored)
├── models/                    # Trained model artifacts (gitignored)
├── reports/                   # Generated .docx deliverables (gitignored)
└── docs/                      # Documentation
```

## Installation

```bash
# Clone and install in development mode
cd myrewards-intelligence
python -m pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Data Setup

Copy the analytical dataset from the research pipeline:

```bash
cp /path/to/research-repo/outputs/csv/analytical_dataset.csv data/
```

Or read it directly from the reference repo by configuring `config/config.yaml`.

## Usage

Run the complete Milestone 5 pipeline:

```bash
python -m myrewards_intelligence.cli.run_milestone5
```

Or use the installed script:

```bash
myrewards-m5
```

This will:
1. Load and validate the Milestone 4 analytical dataset
2. Train Model A (InComm-sourced labels) and Model B (expanded dataset)
3. Evaluate both models (AUC, precision/recall, calibration, subgroup performance)
4. Generate SHAP explanations for Model A
5. Write the `der_MemberImpactScores` DDL proposal
6. Generate 5 Word deliverables
7. Write `reports/Milestone5_Status.md`
8. Run the test suite

## Development

```bash
# Run linting
ruff check .

# Run formatting check
black --check .

# Run type checking
mypy src/

# Run tests
pytest -v
```

## Key Design Principles

1. **Label-source transparency**: Every model output distinguishes InComm-sourced data from research-computed data. Model B's metrics are never presented without disclosing the label-source composition.

2. **Milestone isolation**: Each milestone package depends only on `core/` and reads predecessor outputs as file/table artifacts — never as Python imports. This keeps the package graph clean.

3. **No invented business figures**: If a required real value doesn't exist, the structure is built but the value is explicitly flagged as not yet available.

## Documentation

- `ARCHITECTURE.md` — Layering rules and package dependencies
- `CONTRIBUTING.md` — Style guide and development workflow
- `docs/milestone_plan.pdf` — Full technical milestone plan

## License

MIT License — see LICENSE file.
