"""Typed configuration loading using pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def _get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


PROJECT_ROOT = _get_project_root()


class PathsConfig(BaseModel):
    """Path configuration."""

    data_dir: str = "data"
    analytical_dataset: str = "data/analytical_dataset.csv"
    reference_repo: str = "/Volumes/research/mhealth/InComm/Milestones/InComm_Milestone2"
    reference_dataset: str = "outputs/csv/analytical_dataset.csv"
    models_dir: str = "models"
    reports_dir: str = "reports"
    logs_dir: str = "logs"


class LGBMParams(BaseModel):
    """LightGBM hyperparameters."""

    objective: str = "binary"
    metric: str = "auc"
    boosting_type: str = "gbdt"
    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    verbose: int = -1
    n_estimators: int = 200


class ScoreTiers(BaseModel):
    """Score tier thresholds."""

    high: float = 0.7
    medium: float = 0.4


class Milestone5Config(BaseModel):
    """Milestone 5 specific configuration."""

    test_size: float = 0.2
    random_state: int = 42
    diagnostic_seeds: list[int] = Field(default_factory=lambda: [42, 123, 456, 789, 2026])
    lgbm_params: LGBMParams = Field(default_factory=LGBMParams)
    operating_threshold: float = 0.5
    score_tiers: ScoreTiers = Field(default_factory=ScoreTiers)


class UpliftTiers(BaseModel):
    """Uplift targeting cut-points."""

    tau_treat: float = 0.05
    high_risk_top_share: float = 0.10
    high: float = 0.10
    medium: float = 0.03


class Milestone6Config(BaseModel):
    """Milestone 6 reward-uplift configuration."""

    track: str = "S"
    test_size: float = 0.2
    random_state: int = 42
    diagnostic_seeds: list[int] = Field(default_factory=lambda: [42, 123, 456, 789, 2026])
    observation_window_days: int = 365
    split_mode: str = "grouped_stratified"
    shareable_dir: str = "milestone6_shareables"
    ablation_scores_path: str = "models/scored_model_a_ablation.csv"
    reward_amount_pools_path: str = "config/reward_amount_pools.csv"
    issuance_ledger_path: str = "models/incentive_rewards_mock.csv"
    treatment_grain: str = "gap_level"
    generate_incentive_rewards_mock: bool = True
    mock_treat_probability: float = 0.45
    mock_issue_lag_days_min: int = 7
    mock_issue_lag_days_max: int = 45
    mock_redeem_probability: float = 0.988
    min_arm_n: int = 50
    min_ess_ratio: float = 0.25
    volume_cutpoints: list[float] = Field(
        default_factory=lambda: [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]
    )
    uplift_tiers: UpliftTiers = Field(default_factory=UpliftTiers)


class Milestone7Config(BaseModel):
    """Milestone 7 ROI framework configuration."""

    scores_path: str = "models/scored_uplift_track_r.csv"
    m6_results_path: str = "reports/milestone6_results.json"
    star_weights_path: str = "config/hedis_star_weights.csv"  # InComm metadata, kept for the discrepancy table
    star_density_path: str = "config/cms_star_boundary_density.csv"
    cms_weights_path: str = "config/cms_2026_measure_weights.csv"
    gap_measure_path: str = "config/care_gap_cms_measure.csv"
    reward_amount_pools_path: str = "config/reward_amount_pools.csv"
    claims_validation_path: str = "models/claims_validation.csv"
    claims_dx_path: str = "models/closing_claim_diagnoses.csv"
    attribution_timing_path: str = "models/attribution_timing.json"
    mock_eligibility_path: str = "models/incentive_rewards_mock_eligibility.csv"
    treat_fractions: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])
    # None => ROI is NOT_AVAILABLE; a value is a disclosed research assumption.
    value_per_closure_usd: float | None = None
    redeem_sensitivity: list[float] = Field(default_factory=lambda: [0.55, 0.75])


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"


class Settings(BaseSettings):
    """Application settings loaded from config file and environment."""

    project_name: str = "MyRewards Intelligence"
    project_description: str = "Predictive scoring, claims validation, and ROI intelligence"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    mode_requested: str = "auto"
    layer_prefixes: dict[str, str] = Field(
        default_factory=lambda: {
            "stg_": "Staging",
            "can_": "Canonical",
            "der_": "Derived",
            "ref_": "Reference",
            "cam_": "Campaign",
            "sim_": "Simulation",
        }
    )
    milestone5: Milestone5Config = Field(default_factory=Milestone5Config)
    milestone6: Milestone6Config = Field(default_factory=Milestone6Config)
    milestone7: Milestone7Config = Field(default_factory=Milestone7Config)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: str | Path = "config/config.yaml") -> Settings:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return Settings()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    project = raw.get("project", {})
    paths = raw.get("paths", {})
    mode = raw.get("mode", {})
    m5 = raw.get("milestone5", {})
    m6 = raw.get("milestone6", {})
    m7 = raw.get("milestone7", {})
    logging_cfg = raw.get("logging", {})

    return Settings(
        project_name=project.get("name", "MyRewards Intelligence"),
        project_description=project.get("description", ""),
        paths=PathsConfig(**paths) if paths else PathsConfig(),
        mode_requested=mode.get("requested", "auto"),
        layer_prefixes=raw.get("layer_prefixes", {}),
        milestone5=Milestone5Config(**m5) if m5 else Milestone5Config(),
        milestone6=Milestone6Config(**m6) if m6 else Milestone6Config(),
        milestone7=Milestone7Config(**m7) if m7 else Milestone7Config(),
        logging=LoggingConfig(**logging_cfg) if logging_cfg else LoggingConfig(),
    )


def resolve_path(relative_or_absolute: str | Path) -> Path:
    """Resolve a path relative to project root if not absolute."""
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_dataset_path(settings: Settings) -> Path:
    """Get the path to the analytical dataset, checking local then reference."""
    local_path = resolve_path(settings.paths.analytical_dataset)
    if local_path.exists():
        return local_path

    ref_path = Path(settings.paths.reference_repo) / settings.paths.reference_dataset
    if ref_path.exists():
        return ref_path

    raise FileNotFoundError(
        f"Analytical dataset not found at {local_path} or {ref_path}. "
        "Copy the dataset to data/ or configure the reference_repo path."
    )


def write_json(obj: Any, path: str | Path) -> None:
    """Write object to JSON file, creating parent directories as needed."""
    import json

    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
