"""Model training for Member Impact Scoring.

Trains LightGBM (primary) and logistic regression (baseline) models using
grouped train/test split on member_id to prevent leakage.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from ..core.config import Settings, resolve_path
from ..core.evidence import ModelBasis
from .features import get_groups, get_label_source_composition, prepare_features
from .target import create_target_variable

logger = logging.getLogger("myrewards_intelligence")


@dataclass
class TrainedModel:
    """Container for a trained model and its metadata."""

    model: Any
    model_type: str
    model_basis: ModelBasis
    feature_names: list[str]
    encoders: dict[str, LabelEncoder]
    train_indices: np.ndarray
    test_indices: np.ndarray
    label_source_composition: dict[str, Any]
    training_date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    version: str = "v1"
    hyperparameters: dict[str, Any] = field(default_factory=dict)


def train_test_split_grouped(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform grouped train/test split on member_id.

    Returns indices rather than data to allow reuse across multiple targets.
    """
    groups = get_groups(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx


def _check_lightgbm_available() -> bool:
    """Check if LightGBM is available and working."""
    try:
        import lightgbm as lgb  # noqa: F401
        return True
    except (ImportError, OSError) as e:
        logger.warning("LightGBM not available: %s", e)
        return False


LIGHTGBM_AVAILABLE = _check_lightgbm_available()


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> Any:
    """Train a LightGBM classifier."""
    import lightgbm as lgb

    default_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 200,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    return model


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train a logistic regression baseline model with feature scaling."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    return pipeline


def train_model(
    df: pd.DataFrame,
    model_basis: ModelBasis,
    settings: Settings | None = None,
    model_type: str = "lightgbm",
) -> TrainedModel:
    """Train a model on the provided dataset.

    Args:
        df: Dataset to train on (already filtered for Model A or Model B)
        model_basis: Whether this is Model A (InComm-sourced) or Model B (expanded)
        settings: Configuration settings
        model_type: Type of model to train ('lightgbm' or 'logistic')

    Returns:
        TrainedModel containing the model and metadata
    """
    if settings is None:
        from ..core.config import load_config

        settings = load_config()

    target = create_target_variable(df)
    X, encoders = prepare_features(df, fit_encoders=True)
    feature_names = list(X.columns)

    train_idx, test_idx = train_test_split_grouped(
        df,
        test_size=settings.milestone5.test_size,
        random_state=settings.milestone5.random_state,
    )

    X_train = X.iloc[train_idx]
    y_train = target.iloc[train_idx]

    logger.info(
        "Training %s model (%s): %d train samples, %d test samples",
        model_type,
        model_basis.value,
        len(train_idx),
        len(test_idx),
    )

    actual_model_type = model_type
    if model_type == "lightgbm":
        if LIGHTGBM_AVAILABLE:
            lgbm_params = settings.milestone5.lgbm_params.model_dump()
            model = train_lightgbm(X_train, y_train, params=lgbm_params)
            hyperparameters = lgbm_params
        else:
            logger.warning("LightGBM not available, falling back to logistic regression")
            actual_model_type = "logistic"
            model = train_logistic_regression(X_train, y_train)
            hyperparameters = {"solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced", "fallback": True}
    elif model_type == "logistic":
        model = train_logistic_regression(X_train, y_train)
        hyperparameters = {"solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    label_composition = get_label_source_composition(df)

    return TrainedModel(
        model=model,
        model_type=actual_model_type,
        model_basis=model_basis,
        feature_names=feature_names,
        encoders=encoders,
        train_indices=train_idx,
        test_indices=test_idx,
        label_source_composition=label_composition,
        hyperparameters=hyperparameters,
    )


def save_model(trained_model: TrainedModel, path: str | Path) -> None:
    """Save a trained model to disk."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(trained_model, f)

    logger.info("Saved model to: %s", path)


def load_model(path: str | Path) -> TrainedModel:
    """Load a trained model from disk."""
    path = resolve_path(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def save_model_metadata(trained_model: TrainedModel, path: str | Path) -> None:
    """Save model metadata to JSON."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_type": trained_model.model_type,
        "model_basis": trained_model.model_basis.value,
        "version": trained_model.version,
        "training_date": trained_model.training_date,
        "feature_count": len(trained_model.feature_names),
        "feature_names": trained_model.feature_names,
        "train_samples": len(trained_model.train_indices),
        "test_samples": len(trained_model.test_indices),
        "label_source_composition": trained_model.label_source_composition,
        "hyperparameters": trained_model.hyperparameters,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Saved model metadata to: %s", path)
