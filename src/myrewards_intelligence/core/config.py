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
    logging_cfg = raw.get("logging", {})

    return Settings(
        project_name=project.get("name", "MyRewards Intelligence"),
        project_description=project.get("description", ""),
        paths=PathsConfig(**paths) if paths else PathsConfig(),
        mode_requested=mode.get("requested", "auto"),
        layer_prefixes=raw.get("layer_prefixes", {}),
        milestone5=Milestone5Config(**m5) if m5 else Milestone5Config(),
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
