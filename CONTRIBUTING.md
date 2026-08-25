# Contributing

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Style Guide

### Formatting

- **Black** with line length 100
- **Ruff** for import sorting (isort rules)

Run before committing:
```bash
black .
ruff check --fix .
```

### Linting

- **Ruff** with pyflakes, pycodestyle, bugbear, and flake8-simplify rule sets
- **mypy** in strict-ish mode on `src/`

Run checks:
```bash
ruff check .
mypy src/
```

### Type Hints

Type hints are required on every public function signature:

```python
# Good
def compute_closure_rate(labels: pd.DataFrame, member_id: str) -> float:
    ...

# Bad - missing type hints
def compute_closure_rate(labels, member_id):
    ...
```

### Docstrings

One-line summary for every public function. Longer docstrings only where a non-obvious invariant needs explaining:

```python
# Good - concise, informative
def filter_incomm_sourced(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with label_source == 'InComm-sourced'."""
    return df[df["label_source"] == "InComm-sourced"]

# Bad - restates the obvious
def filter_incomm_sourced(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataframe.
    
    This function takes a dataframe and filters it to only include
    rows where the label_source column equals 'InComm-sourced'.
    It returns a new dataframe with only those rows.
    """
    return df[df["label_source"] == "InComm-sourced"]
```

### Comments

Comments explain non-obvious intent, trade-offs, or constraints — not what the code does:

```python
# Good - explains why
# Grouped split prevents member ID leakage across train/test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2)

# Bad - restates the code
# Create a GroupShuffleSplit with 1 split and 20% test size
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2)
```

## Testing

- pytest with offline-friendly fixtures (synthetic DataFrames, no live DB calls)
- Every module that trains or evaluates a model has a corresponding pytest file
- Mirror the fixture style in the reference repo's tests

Run tests:
```bash
pytest -v
pytest -v --cov=src/myrewards_intelligence  # with coverage
```

### Fixture Pattern

```python
@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Synthetic dataset shaped like analytical_dataset.csv."""
    return pd.DataFrame([
        {"member_id": "M1", "care_gap_code": "DIAB_A1C_TEST", "status": "Closed",
         "label_source": "InComm-sourced", ...},
        ...
    ])
```

## Pre-Commit Checks

Pre-commit runs automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

Checks include:
- Black formatting
- Ruff linting
- isort import ordering
- mypy type checking
- Trailing whitespace / end-of-file fixes

## Commit History

Commit after each logical unit of work:
1. Repository scaffold
2. Target variable + dataset loading
3. Training
4. Evaluation + SHAP
5. Reports

This makes the history document the build process.

## Pull Request Checklist

Before submitting:

- [ ] `pytest -q` passes
- [ ] `ruff check . && black --check . && mypy src/` all clean
- [ ] Any new public functions have type hints and docstrings
- [ ] Model A vs Model B distinction is maintained in any new metrics/reports
