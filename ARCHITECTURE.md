# Architecture

## Package Layering

```
┌─────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                               │
│   cli/run_milestone5.py, cli/run_milestone6.py, etc.                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────────┐
│                         Milestone Packages                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   milestone5_   │  │   milestone6_   │  │   milestone7_   │     │
│  │ member_impact_  │  │ reward_uplift   │  │  claims_roi     │     │
│  │    scoring      │  │                 │  │                 │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
└───────────┼─────────────────────┼────────────────────┼──────────────┘
            │                     │                    │
            │    File/Table       │    File/Table      │
            │    Artifacts        │    Artifacts       │
            │    (not imports)    │    (not imports)   │
            │                     │                    │
┌───────────┴─────────────────────┴────────────────────┴──────────────┐
│                              Core Layer                              │
│   core/database.py, core/config.py, core/logging.py,                │
│   core/evidence.py, core/reporting/base.py                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Layering Rules

**Rule 1:** `core/` has no knowledge of any milestone.

The core layer provides infrastructure (database connections, configuration loading, logging, evidence-source tracking, report generation helpers) that is milestone-agnostic. It never imports from or references any `milestone*` package.

**Rule 2:** Milestone packages depend on `core/` and never on each other directly.

If Milestone 6 needs Milestone 5's scored output, it reads it as a file artifact (CSV, pickle) or database table — not as a Python import from `milestone5_member_impact_scoring`. This keeps the package dependency graph acyclic and allows each milestone to ship independently.

**Rule 3:** CLI orchestrators can import from multiple packages.

The CLI layer (`cli/run_milestone5.py`, etc.) is the only place where imports from both `core/` and a specific milestone package appear together. The CLI coordinates execution but does not contain business logic.

## Data Flow

```
Reference Repo                  This Repo
(research pipeline)            (InComm-facing)
       │
       │  outputs/csv/
       │  analytical_dataset.csv
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  data/analytical_dataset.csv                                        │
│  (gitignored — copied in or read from reference path)               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  milestone5_member_impact_scoring/                                  │
│  ├── dataset.py      → loads, validates, reports schema             │
│  ├── target.py       → defines is_closed binary target              │
│  ├── features.py     → feature prep, Model A/B row filtering        │
│  ├── train.py        → trains LightGBM + logistic baseline          │
│  ├── evaluate.py     → AUC, precision/recall, calibration, subgroup │
│  ├── explain.py      → SHAP global + individual explanations        │
│  ├── derived_layer.py→ der_MemberImpactScores DDL proposal          │
│  └── reports.py      → 5 Word deliverables                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  models/                                                            │
│  ├── model_a_v1.pkl          (primary, InComm-sourced labels)       │
│  ├── model_b_v1.pkl          (expanded, mixed labels)               │
│  └── metadata.json           (version, training date, metrics)      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  reports/                                                           │
│  ├── Member Impact Scoring Model v1.docx                            │
│  ├── Model Performance Summary.docx                                 │
│  ├── SHAP Explanation Summary.docx                                  │
│  ├── Member Prioritization Output.docx                              │
│  ├── Derived Layer Integration Approach.docx                        │
│  └── Milestone5_Status.md                                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Evidence-Source Tracking

All data in this project carries a label-source designation:

| Source Type | Meaning | Example |
|-------------|---------|---------|
| InComm-sourced | Real data confirmed by InComm | `der_MemberCareGapStatuses` labels |
| Research-computed | Generated by our domain analysis | Milestone 3 care-gap closures |
| Research-generated | Synthetic simulation data | Reward timing features |

This distinction flows through the entire pipeline:

1. **Dataset loading** (`dataset.py`): Validates `label_source` column exists
2. **Feature prep** (`features.py`): Filters rows into Model A (InComm-sourced only) vs Model B (all)
3. **Training** (`train.py`): Trains separate models, never blends silently
4. **Evaluation** (`evaluate.py`): Reports label-source composition alongside every metric
5. **Derived layer** (`derived_layer.py`): `LabelSourceBasis` column in DDL
6. **Reports** (`reports.py`): Every table shows which model and label source

This is the non-negotiable project principle: never present research-computed numbers as InComm-confirmed.
