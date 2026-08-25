"""Milestone 5: Member Impact Scoring — supervised predictive model for care-gap closure."""

from .dataset import load_analytical_dataset, validate_dataset
from .derived_layer import generate_ddl_proposal, generate_scored_output
from .evaluate import evaluate_model, generate_subgroup_metrics
from .explain import generate_shap_explanations
from .features import prepare_features, split_by_label_source
from .target import create_target_variable
from .train import train_model

__all__ = [
    "create_target_variable",
    "evaluate_model",
    "generate_ddl_proposal",
    "generate_scored_output",
    "generate_shap_explanations",
    "generate_subgroup_metrics",
    "load_analytical_dataset",
    "prepare_features",
    "split_by_label_source",
    "train_model",
    "validate_dataset",
]
